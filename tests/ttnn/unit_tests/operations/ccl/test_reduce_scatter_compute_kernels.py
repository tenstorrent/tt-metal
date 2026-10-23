# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Single-device test of the reduce_scatter_minimal_async compute kernels (line_reduction, dim_zero_line_reduction,
ring_reduction, dim_zero_ring_reduction). A generic_op program runs the real compute kernel with the op's compile-time
arguments; a stand-in reader and writer replace the fabric workers. The reader streams one tile_granularity chunk per
reduction event, the same chunks the compute kernel consumes, so an event list mirroring each kernel's loop nest
decides what is pushed and what the output must hold.
"""

import os
import statistics

import pytest
import torch
import ttnn

KDIR = "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels"
HERE = os.path.dirname(os.path.abspath(__file__))
READER = os.path.join(HERE, "rs_compute_kernels", "reader.cpp")
WRITER = os.path.join(HERE, "rs_compute_kernels", "writer.cpp")
TILE_BYTES = 2048
CB_IN, CB_INTERM, CB_INTERM2, CB_OUT = 0, 1, 2, 16


def line_events(S, B, C, G, start, total):
    ev = []
    for _ in range(B * S * C):
        tr = start
        while tr < total:
            n = min(total - tr, G)
            ev.append((n, False))
            tr += n
    return ev


def dim_zero_line_events(S, B, G, start, total):
    return line_events(S, B, 1, G, start, total)


def ring_events(ring_size, B, C, G, fuse_op, direction, start, total, unit_start, unit_end):
    ev = []
    num_traversals = B if (fuse_op and B > 1) else 1
    for t in range(num_traversals):
        if num_traversals == 1:
            ts, te = unit_start, unit_end
        else:
            ts, te = max(unit_start, t * C), min(unit_end, (t + 1) * C)
        half = ring_size // 2
        for i in range(half + 1):
            if i == 0:
                even, odd, re, ro, ro_out = direction, not direction, False, False, False
            elif i == half:
                even, odd = direction, not direction
                re, ro, ro_out = even, odd, True
            elif i == 1:
                even, odd, re, ro, ro_out = True, True, direction, not direction, False
            else:
                even, odd, re, ro, ro_out = True, True, True, True, False
            for _ in range(ts, te):
                tr = start
                while tr < total:
                    is_even = ((tr // G) % 2) == 0
                    n = min(((tr // G) + 1) * G, total) - tr
                    if (is_even and not even) or (not is_even and not odd) or n == 0:
                        tr += n
                        continue
                    if (is_even and re) or (not is_even and ro):
                        ev.append((n, ro_out))
                    tr += n
    return ev


def dim_zero_ring_events(ring_size, B, G, direction, start, total):
    ev = []
    for _ in range(ring_size - 1):
        for _ in range(B):
            tr = start
            if not direction:
                tr += min((total - tr) // 2, G)
            while tr < total:
                rem = total - tr
                n = min(rem // 2, G) if direction else min(rem, G)
                ev.append((n, False))
                tr += n
                rem = total - tr
                if rem > 0:
                    tr += min(rem // 2, G) if not direction else min(rem, G)
    return ev


def make_kernel_spec(name, G, p):
    if name == "line":
        ev = line_events(p["S"], p["B"], p["C"], G, p["start"], p["total"])
        return ev, dict(
            path=f"{KDIR}/line_reduction.cpp",
            cta=[CB_IN, CB_INTERM, CB_OUT, G, p["B"], p["C"]],
            named=[],
            rt=[p["S"], p["start"], p["total"]],
        )
    if name == "dim_zero_line":
        ev = dim_zero_line_events(p["S"], p["B"], G, p["start"], p["total"])
        return ev, dict(
            path=f"{KDIR}/dim_zero_line_reduction.cpp",
            cta=[CB_IN, CB_INTERM, CB_OUT, G, p["B"]],
            named=[],
            rt=[p["S"], p["start"], p["total"]],
        )
    if name == "ring":
        ev = ring_events(
            p["ring"], p["B"], p["C"], G, p["fuse"], p["dir"], p["start"], p["total"], p["u0"], p["u1"]
        )
        named = [
            ("cb_input_id", CB_IN),
            ("cb_interm_id", CB_INTERM),
            ("cb_interm2_id", CB_INTERM2),
            ("cb_compute_output_id", CB_OUT),
            ("tile_granularity", G),
            ("ring_size", p["ring"]),
            ("input_tensor_B", p["B"]),
            ("slice_C", p["C"]),
            ("fuse_op", p["fuse"]),
        ]
        return ev, dict(
            path=f"{KDIR}/ring_reduction.cpp",
            cta=[],
            named=named,
            rt=[p["start"], p["total"], int(p["dir"]), p["u0"], p["u1"]],
        )
    if name == "dim_zero_ring":
        ev = dim_zero_ring_events(p["ring"], p["B"], G, p["dir"], p["start"], p["total"])
        named = [
            ("cb_input_id", CB_IN),
            ("cb_interm_id", CB_INTERM),
            ("cb_compute_output_id", CB_OUT),
            ("tile_granularity", G),
            ("ring_size", p["ring"]),
            ("slice_B", p["B"]),
        ]
        return ev, dict(
            path=f"{KDIR}/dim_zero_ring_reduction.cpp",
            cta=[],
            named=named,
            rt=[p["start"], p["total"], int(p["dir"])],
        )
    raise ValueError(name)


def flag_words(ev):
    words = [0] * ((len(ev) + 31) // 32 or 1)
    for e, (_, f) in enumerate(ev):
        if f:
            words[e // 32] |= 1 << (e % 32)
    return words


def run_kernel(device, name, p, fp32_acc, cores_xy=(1, 1), do_io=True, seed=0):
    G = 4 if fp32_acc else 8
    ev, spec = make_kernel_spec(name, G, p)
    ne = len(ev)
    n2 = sum(1 for _, f in ev if f)
    ncores = cores_xy[0] * cores_xy[1]
    dt = ttnn.bfloat16
    chunks_in = ne * G
    chunks2 = max(n2, 1) * G

    torch.manual_seed(seed)
    ta = torch.randn(1, 1, 32, 32 * chunks_in * ncores).to(torch.bfloat16)
    tb = torch.randn(1, 1, 32, 32 * chunks_in * ncores).to(torch.bfloat16)
    tc = torch.randn(1, 1, 32, 32 * chunks2 * ncores).to(torch.bfloat16)

    def to_dev(t):
        return ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    da, db, dc = to_dev(ta), to_dev(tb), to_dev(tc)
    dout = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 32, 32 * chunks_in * ncores]), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )

    grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cores_xy[0] - 1, cores_xy[1] - 1))]
    )
    cb_pages = {CB_IN: 3 * G, CB_INTERM: 3 * G, CB_INTERM2: 2 * G, CB_OUT: 3 * G}
    cbs = [
        ttnn.CBDescriptor(
            total_size=TILE_BYTES * n,
            core_ranges=grid,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=TILE_BYTES)],
        )
        for i, n in cb_pages.items()
    ]

    cta_r = [G, int(do_io)]
    for t in (da, db, dc):
        cta_r += list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    cta_w = [CB_OUT, G, int(do_io)] + list(ttnn.TensorAccessorArgs(dout).get_compile_time_args())
    rt_r, rt_w, rt_c = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    fw = flag_words(ev)
    for ci in range(ncores):
        x, y = ci % cores_xy[0], ci // cores_xy[0]
        rt_r[x][y] = [
            da.buffer_address(),
            db.buffer_address(),
            dc.buffer_address(),
            ci * chunks_in,
            ci * chunks_in,
            ci * chunks2,
            ne,
        ] + fw
        rt_w[x][y] = [dout.buffer_address(), ci * chunks_in, ne]
        rt_c[x][y] = list(spec["rt"])

    def src(path, cta, cfg, named=None, rt=None):
        return ttnn.KernelDescriptor(
            kernel_source=path,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=grid,
            compile_time_args=cta,
            named_compile_time_args=named or [],
            runtime_args=rt,
            config=cfg,
        )

    reader = src(READER, cta_r, ttnn.ReaderConfigDescriptor(), rt=rt_r)
    writer = src(WRITER, cta_w, ttnn.WriterConfigDescriptor(), rt=rt_w)
    compute = src(
        spec["path"],
        spec["cta"],
        ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=fp32_acc),
        named=spec["named"],
        rt=rt_c,
    )
    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)

    def call():
        return ttnn.generic_op([da, db, dc, dout], program)

    return call, (ta, tb, tc, ev, G, ncores, chunks_in, chunks2), dout


def check(call, ctx):
    ta, tb, tc, ev, G, ncores, chunks_in, chunks2 = ctx
    out = ttnn.to_torch(call()).to(torch.float32)

    def tiles(t, per):
        return t.to(torch.float32).view(32, ncores * per, 32)

    A, B_, C_, O = tiles(ta, chunks_in), tiles(tb, chunks_in), tiles(tc, chunks2), out.view(32, ncores * chunks_in, 32)
    for ci in range(ncores):
        j2 = 0
        for e, (n, f) in enumerate(ev):
            base = ci * chunks_in + e * G
            want = A[:, base : base + n] + B_[:, base : base + n]
            if f:
                b2 = ci * chunks2 + j2 * G
                want = want + C_[:, b2 : b2 + n]
                j2 += 1
            got = O[:, base : base + n]
            assert torch.allclose(got, want, rtol=0.02, atol=0.03), (
                f"core {ci} event {e} (n={n}, interm2={f}): max abs error {(got - want).abs().max().item()}"
            )


CASES = {
    "line": [
        dict(S=1, B=1, C=1, start=0, total=64),
        dict(S=3, B=2, C=2, start=0, total=20),
        dict(S=2, B=1, C=3, start=5, total=37),
    ],
    "dim_zero_line": [
        dict(S=1, B=1, start=0, total=64),
        dict(S=3, B=2, start=0, total=20),
        dict(S=2, B=3, start=5, total=37),
    ],
    "ring": [
        dict(ring=2, B=1, C=1, fuse=0, dir=True, start=0, total=48, u0=0, u1=1),
        dict(ring=2, B=1, C=1, fuse=0, dir=False, start=0, total=48, u0=0, u1=1),
        dict(ring=4, B=2, C=2, fuse=0, dir=True, start=8, total=40, u0=1, u1=4),
        dict(ring=4, B=2, C=2, fuse=0, dir=False, start=8, total=40, u0=0, u1=3),
        dict(ring=4, B=2, C=2, fuse=1, dir=True, start=0, total=30, u0=0, u1=4),
        dict(ring=8, B=3, C=1, fuse=1, dir=False, start=3, total=27, u0=0, u1=3),
    ],
    "dim_zero_ring": [
        dict(ring=2, B=1, start=0, total=48, dir=True),
        dict(ring=2, B=1, start=0, total=48, dir=False),
        dict(ring=4, B=2, start=0, total=37, dir=True),
        dict(ring=4, B=3, start=6, total=41, dir=False),
    ],
}
PARAMS = [(k, i, p) for k, ps in CASES.items() for i, p in enumerate(ps)]
IDS = [f"{k}-{i}" for k, i, _ in PARAMS]


@pytest.mark.parametrize("fp32_acc", [False, True], ids=["bf16acc", "fp32acc"])
@pytest.mark.parametrize("name,idx,p", PARAMS, ids=IDS)
def test_rs_compute_kernel(device, name, idx, p, fp32_acc):
    call, ctx, _ = run_kernel(device, name, p, fp32_acc, seed=idx)
    check(call, ctx)


LOUDBOX_PARTIAL = (8192, 5120)


@pytest.mark.parametrize("fp32_acc", [False, True], ids=["bf16acc", "fp32acc"])
def test_rs_line_loudbox_shape(device, fp32_acc):
    """Per-device LoudBox partial 8192 x 5120 reduced over a 2-device TP ring: one step over the 8192 x 2560 half."""
    out_tiles = (LOUDBOX_PARTIAL[0] // 32) * (LOUDBOX_PARTIAL[1] // 2 // 32)
    cores = (8, 8)
    per_core = out_tiles // (cores[0] * cores[1])
    call, ctx, _ = run_kernel(
        device, "line", dict(S=1, B=1, C=1, start=0, total=per_core), fp32_acc, cores_xy=cores, seed=7
    )
    check(call, ctx)


def _perf(device, name, p, fp32_acc, cores, do_io, label):
    from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program

    call, _, _ = run_kernel(device, name, p, fp32_acc, cores_xy=cores, do_io=do_io)
    for _ in range(2):
        call()
    ttnn.synchronize_device(device)
    n = int(os.environ.get("RSFOLD_N", "10"))
    ns = []
    for _ in range(n):
        _, rec = profile_realtime_program(device, call, collect_all=True)
        ns.append(rec[-1]["duration_ns"])
    for i, v in enumerate(ns):
        print(f"RSFOLD-SAMPLE {label} i={i} duration_ns={v:.0f}")
    print(f"RSFOLD-MEDIAN {label} duration_ns={statistics.median(ns):.0f}")


@pytest.mark.parametrize("do_io", [True, False], ids=["dram", "computeonly"])
@pytest.mark.parametrize("fp32_acc", [False, True], ids=["bf16acc", "fp32acc"])
def test_rs_line_loudbox_perf(device, fp32_acc, do_io):
    out_tiles = (LOUDBOX_PARTIAL[0] // 32) * (LOUDBOX_PARTIAL[1] // 2 // 32)
    cores = (8, 8)
    per_core = out_tiles // (cores[0] * cores[1])
    label = f"line-{'fp32acc' if fp32_acc else 'bf16acc'}-{'dram' if do_io else 'computeonly'}"
    _perf(device, "line", dict(S=1, B=1, C=1, start=0, total=per_core), fp32_acc, cores, do_io, label)
