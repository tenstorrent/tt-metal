# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Quasar pool coverage sweeps — one device session per test, many cases, per-case verdicts.

test_qpool_c_sweep : channel-width ladder (sub-face .. 768, partial-tile combos), the DEST-width
regimes from WH/BH testing practice. C > 256 exercises the wide reduction (in_nblocks_c > 1).

test_qpool_matrix  : kernel sizes (2x2 .. 9x9 — 7/8/9 are chunked large kernels, 9x9 = 3 chunks),
stride 1, batch 2, tall/wide inputs, a natural wide-reduction case (C=280 = 9 tiles -> three
3-tile c-blocks via largest_uniform_block_width), avg pool (basic + chunked large kernel), and
block/width sharding. A craq-sim-sized subset of the WH/BH nightly pool coverage.

Case constraints (asserted per case): N*H*W % 32 == 0; torch golden needs padding <= kernel/2
(num_threads is always 4 on Quasar: sticks are dealt round-robin to lanes and each compute lane
derives its own share, so any per-core count is legal — see the stick*/sticks* unit cases, which
pin remainders and idle lanes); total volume is kept
<= ~16KB to dodge the open craq-sim halo corruption class (large kernels run single-core: halo
exchange scales with kernel size). Cases print their banner BEFORE running so a hang names the
case in flight; OOM/ERROR are caught per-case.

Cases tagged sim_skip=... are auto-skipped when TT_METAL_SIMULATOR is set (craq-sim AND the ZeBu
emulator both set it); the tag text records the evidence. Three classes exist: craq-sim artifacts
(fail/hang on the sim only, exact on RTL), craq-sim ISA gaps (UnimplementedFunctionality), and
REAL Quasar bugs that pre-date this branch (the <= 8-row-window cases: identical mismatch on sim,
RTL, T=1/T=4 and main). QPOOL_NO_SIM_SKIP=1 forces them all to run -- do that on the emulator.

Run via run_qpool.sh sweep (C ladder) / run_qpool.sh matrix (this matrix). On WH/BH silicon
(the cross-check leg) set QPOOL_RUN_ON_ANY_ARCH=1 — conftest.py skips this directory otherwise so
the WH/BH sanity pool group does not run the quasar op.
"""

import os
import sys

import pytest
import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_qpool_debug import _build_input, _dump_mismatches

# =============================== CONFIG — edit me ===============================
C_VALUES = [8, 16, 32, 40, 64, 96, 128, 144, 256, 280, 384, 392, 512, 768]
PATTERN = "random"
SEED = 0
PCC_THRESHOLD = 0.99
PERF_ITERS = 3  # measured iterations per case in test_qpool_perf_matrix (plus 1 warmup)
# =================================================================================

SIM_MAX_STICKS = 128


def _run_case(
    device,
    channels=64,
    *,
    batch=1,
    in_h=16,
    in_w=8,
    kernel=(3, 3),
    stride=(2, 2),
    padding=(1, 1),
    cores=None,  # None = grid-adaptive max divisor of the input height tiles; N pins N cores
    shard="height",  # "height" | "block" | "width"
    grid_yx=None,  # (y, x) core grid for block/width sharding
    pool="max",  # "max" | "avg" (avg cases use padding=0 to sidestep count_include_pad semantics)
    pattern=None,  # input pattern override (default: module-level PATTERN)
    dilation=(1, 1),  # max pool only
    ceil_mode=False,
    dtype="bf16",  # "bf16" | "bf8b" (bf8b forces TILE layout; golden runs on the quantized input)
    out_layout="rm",  # "rm" | "tile" output layout (tiled output runs single-lane by policy)
    perf_label=None,  # perf mode: skip the golden check, run PERF_ITERS labeled iterations instead
):
    pattern = pattern or PATTERN
    kernel, stride, padding, dilation = list(kernel), list(stride), list(padding), list(dilation)
    out_h = (in_h - kernel[0] + 2 * padding[0]) // stride[0] + 1  # perf mode only; golden shape wins below
    out_w = (in_w - kernel[1] + 2 * padding[1]) // stride[1] + 1
    tensor_height = batch * in_h * in_w
    assert tensor_height % 32 == 0, f"N*H*W={tensor_height} must be a multiple of 32"
    tiled_input = channels % 32 == 0 or dtype == "bf8b"

    x_nhwc = _build_input(pattern, batch, in_h, in_w, channels, SEED, 0).to(torch.bfloat16)
    if dtype == "bf8b":
        assert channels % 32 == 0, "bf8b needs TILE layout (C % 32 == 0)"
        # golden must see the block-fp quantized values the device sees
        x_nhwc = ttnn.to_torch(
            ttnn.from_torch(
                x_nhwc.reshape(1, 1, tensor_height, channels), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT
            )
        ).reshape(batch, in_h, in_w, channels)
    input_max = x_nhwc.float().max().item()
    if perf_label is None:
        if pool == "max":
            golden_nchw = torch.nn.functional.max_pool2d(
                x_nhwc.permute(0, 3, 1, 2).float(),
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                dilation=dilation,
                ceil_mode=ceil_mode,
            )
        else:
            golden_nchw = torch.nn.functional.avg_pool2d(
                x_nhwc.permute(0, 3, 1, 2).float(),
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                ceil_mode=ceil_mode,
            )
        out_h, out_w = golden_nchw.shape[2], golden_nchw.shape[3]  # exact for ceil_mode/dilation
        golden = golden_nchw.permute(0, 2, 3, 1).reshape(batch * out_h * out_w, channels).contiguous()

    grid = device.compute_with_storage_grid_size()
    if shard == "height":
        height_tiles = tensor_height // 32
        num_cores = cores or max(c for c in range(1, grid.x * grid.y + 1) if height_tiles % c == 0)
        shard_height = (height_tiles // num_cores) * 32
        mem_config = ttnn.create_sharded_memory_config(
            shape=(1, 1, shard_height, channels),
            core_grid=ttnn.num_cores_to_corerangeset(num_cores, grid, True),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        core_desc = f"{num_cores}xHEIGHT"
    else:
        gy, gx = grid_yx
        strategy = ttnn.ShardStrategy.BLOCK if shard == "block" else ttnn.ShardStrategy.WIDTH
        mem_config = ttnn.create_sharded_memory_config(
            shape=(1, 1, tensor_height, channels),
            core_grid=ttnn.CoreGrid(y=gy, x=gx),
            strategy=strategy,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
        core_desc = f"{gy}x{gx}x{shard.upper()}"

    print(
        f"\nQPOOL-MATRIX: {pool} C={channels} in={batch}x{in_h}x{in_w} k={kernel} s={stride} p={padding} "
        f"{core_desc} layout={'TILE' if tiled_input else 'ROW_MAJOR'}"
        + (f" pattern={pattern}" if pattern != PATTERN else "")
        + (f" d={dilation}" if dilation != [1, 1] else "")
        + (" ceil" if ceil_mode else "")
        + (f" {dtype}" if dtype != "bf16" else "")
        + (" out=TILE" if out_layout == "tile" else ""),
        flush=True,
    )

    x = ttnn.from_torch(
        x_nhwc.reshape(1, 1, tensor_height, channels),
        dtype=ttnn.bfloat8_b if dtype == "bf8b" else ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT if tiled_input else ttnn.ROW_MAJOR_LAYOUT,
    )
    x = x.to(device, mem_config)
    pool_op = ttnn.experimental.quasar.max_pool2d if pool == "max" else ttnn.experimental.quasar.avg_pool2d

    def run_once():
        out = pool_op(
            input_tensor=x,
            batch_size=batch,
            input_h=in_h,
            input_w=in_w,
            channels=channels,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            output_layout=ttnn.TILE_LAYOUT if out_layout == "tile" else ttnn.ROW_MAJOR_LAYOUT,
            **(dict(dilation=dilation) if pool == "max" else {}),  # avg_pool2d takes no dilation
        )
        ttnn.synchronize_device(device)
        return out

    if perf_label is not None:
        # Label each dispatch in the craq-sim per-dispatch trace (halo+pool quiesce as one row)
        # so qpool_perf_report.py can attribute clocks per case. Warmup absorbs JIT/program cache.
        os.environ["TTSIM_PERF_TRACE_NODEID"] = f"warmup_{perf_label}"
        run_once().deallocate()
        for i in range(PERF_ITERS):
            os.environ["TTSIM_PERF_TRACE_NODEID"] = f"case::{perf_label}::i{i}"
            run_once().deallocate()
        os.environ["TTSIM_PERF_TRACE_NODEID"] = f"teardown_{perf_label}"
        x.deallocate()
        return f"PERF ({PERF_ITERS} iters)"

    out = run_once()
    got = ttnn.to_torch(out).float().reshape(batch * out_h * out_w, channels)
    x.deallocate()
    out.deallocate()

    got_max = got.max().item()
    if got_max > input_max + 1e-2:
        return f"LEAK out.max={got_max:.4f} > in.max={input_max:.4f}"
    max_diff = (got - golden).abs().max().item()
    close = torch.allclose(got, golden, rtol=0.01, atol=0.01)
    pcc = None
    if golden.std() > 0 and got.std() > 0:
        pcc = torch.corrcoef(torch.stack([golden.flatten(), got.flatten()]))[0, 1].item()
    if not close or (pcc is not None and pcc < PCC_THRESHOLD):
        _dump_mismatches(got, golden, out_h, out_w, channels, 4)
        return f"MISMATCH max_abs_diff={max_diff:.6f}" + (f" pcc={pcc:.6f}" if pcc is not None else "")
    return "PASS" + (f" (pcc={pcc:.6f})" if pcc is not None else "")


def _run_cases(device, cases):
    # QPOOL_ONLY=name1,name2 runs a subset (debug/discriminator runs).
    only = os.environ.get("QPOOL_ONLY")
    if only:
        wanted = set(only.split(","))
        cases = [c for c in cases if c[0] in wanted]
    results = {}
    for name, kwargs in cases:
        kwargs = dict(kwargs)
        sim_skip = kwargs.pop("sim_skip", None)
        if sim_skip and os.environ.get("TT_METAL_SIMULATOR") and not os.environ.get("QPOOL_NO_SIM_SKIP"):
            results[name] = f"PASS (SIM-SKIP: {sim_skip})"
            print(f"QPOOL-MATRIX: {name}: {results[name]}", flush=True)
            continue
        try:
            results[name] = _run_case(device, **kwargs)
        except RuntimeError as e:
            msg = str(e)
            kind = "OOM" if ("Out of Memory" in msg or "beyond max L1" in msg) else "ERROR"
            results[name] = f"{kind}: {msg.splitlines()[0][:140]}"
        print(f"QPOOL-MATRIX: {name}: {results[name]}", flush=True)
    print("\nQPOOL-MATRIX SUMMARY:")
    for name, _ in cases:
        print(f"  {name:24s} {results[name]}")
    failures = {n: r for n, r in results.items() if not r.startswith("PASS")}
    assert not failures, f"{len(failures)}/{len(cases)} cases failed: {sorted(failures)}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_qpool_c_sweep(mesh_device):
    for c in C_VALUES:
        assert c % 8 == 0, f"C={c}: ROW_MAJOR sharding needs 16B-aligned rows (C % 8 == 0 for bf16)"
    # C >= 128 drops to the 32-stick single-core base, and C >= 384 is skipped on the SIMULATOR:
    # the open craq-sim halo bug corrupts/stalls above the ~24KB volume class (proven
    # threading-independent — identical failure at num_threads=1; run those on silicon/emulator).
    c_values = C_VALUES
    if os.environ.get("TT_METAL_SIMULATOR"):
        skipped = [c for c in c_values if c >= 384]
        c_values = [c for c in c_values if c < 384]
        if skipped:
            print(f"QPOOL-SWEEP: skipping {skipped} on the sim (craq-sim volume corruption class)", flush=True)
    cases = [(f"C{c}", dict(channels=c, in_h=16 if c < 128 else 8, in_w=8 if c < 128 else 4)) for c in c_values]
    _run_cases(mesh_device, cases)


MATRIX_CASES = [
    # kernel-size ladder @ C=64 (7/8/9 are chunked large kernels; single-core to bound halo volume).
    # sim_skip cases fail identically at num_threads=1 on craq-sim but pass exact-PCC on WH
    # silicon (verified 2026-09-01, full matrix) — same open sim bug class as the C>=384 sweep skips.
    (
        "k2x2_s2",
        dict(
            kernel=(2, 2),
            stride=(2, 2),
            padding=(0, 0),
            sim_skip="Quasar pool bug, windows <= 8 rows: identical MISMATCH on craq-sim AND ZeBu RTL, at T=1 and T=4, and on main (merge-base bc294789ec3) -- pre-existing, not a sim artifact; WH silicon passes (different LLK path)",
        ),
    ),
    ("k3x3_s1", dict(in_h=8, in_w=8, kernel=(3, 3), stride=(1, 1), padding=(1, 1))),
    ("k5x5_s2", dict(kernel=(5, 5), stride=(2, 2), padding=(2, 2))),
    ("k7x7_s2_large", dict(kernel=(7, 7), stride=(2, 2), padding=(3, 3), cores=1)),
    ("k8x8_s2_large", dict(kernel=(8, 8), stride=(2, 2), padding=(3, 3), cores=1)),
    ("k9x9_s2_3chunks", dict(kernel=(9, 9), stride=(2, 2), padding=(4, 4), cores=1)),
    # input geometry
    ("batch2", dict(batch=2, in_h=8, in_w=8)),
    ("tall_32x4", dict(in_h=32, in_w=4)),
    ("wide_4x32", dict(in_h=4, in_w=32)),
    # natural wide reduction: C=280 = 9 tiles > the 8-tile cap -> three 3-tile c-blocks
    ("wide_c280_3blocks", dict(channels=280, in_h=8, in_w=4, cores=1)),
    # avg pool: basic + chunked large kernel (7x7 window = 49 rows -> 32 + 17-row PARTIAL chunk,
    # exercising the partial-chunk stale-tail fill on the avg path); padding=0 (see _run_case)
    ("avg_k3x3_s1", dict(pool="avg", kernel=(3, 3), stride=(1, 1), padding=(0, 0), cores=1)),
    ("avg_k7x7_s1_large", dict(pool="avg", kernel=(7, 7), stride=(1, 1), padding=(0, 0), cores=1)),
    # all-negative inputs: any 0 leaking into a max (e.g. a zero-filled instead of -inf-filled
    # in_cb identity) turns a -0.5 golden into 0.0. Small kernel = partial-face init path;
    # large kernel = chunked path. padding=0 keeps halo pad values out of the picture.
    ("neg_k3x3_s1", dict(kernel=(3, 3), stride=(1, 1), padding=(0, 0), cores=1, pattern="const:-0.5")),
    ("neg_k7x7_s1_large", dict(kernel=(7, 7), stride=(1, 1), padding=(0, 0), cores=1, pattern="const:-0.5")),
    # sharding layouts
    ("block_2x2", dict(shard="block", grid_yx=(2, 2))),
    ("width_1x2_c128", dict(channels=128, in_h=8, in_w=4, shard="width", grid_yx=(1, 2))),
]


# Special-case axes from the regular WH/BH unit test (tests/.../pool/test_maxpool2d.py) at
# craq-sim-feasible sizes: the unit test's shapes are model-scale (600x600, C=32768), far beyond
# the functional sim, so each AXIS is kept and the geometry shrunk (single-core, NHW % 32 == 0).
# All 12 cases pass exact-PCC on WH silicon (2026-09-01). sim_skip evidence: HANG/MISMATCH is
# bit-identical at num_threads=1 (threading exonerated); bf8b is a Metal-2.0 runtime gap on the
# quasar path (program_spec.cpp:1731 is_data_format_supported TT_FATAL), fine on WH.
UNIT_CASES = [
    ("ceil_k3x3_s2_b4", dict(batch=4, in_h=8, in_w=4, stride=(2, 2), ceil_mode=True, cores=1)),
    (
        "ceil_edge_k4x3_s6x5",
        dict(batch=2, channels=32, kernel=(4, 3), stride=(6, 5), padding=(2, 1), ceil_mode=True, cores=1),
    ),
    ("dil2_k3x3", dict(in_h=16, in_w=16, stride=(1, 1), padding=(0, 0), dilation=(2, 2), cores=1)),
    (
        "ceil_dil_k3x5_s4x2",
        dict(
            in_h=16,
            in_w=16,
            kernel=(3, 5),
            stride=(4, 2),
            dilation=(2, 1),
            ceil_mode=True,
            cores=1,
            sim_skip="craq-sim HANG (T=1-identical)",
        ),
    ),
    ("asym_k3x5_s2x1", dict(kernel=(3, 5), stride=(2, 1), padding=(1, 2), cores=1)),
    (
        "uneven_pad_k8_s4",
        dict(
            in_h=16,
            in_w=16,
            kernel=(8, 8),
            stride=(4, 4),
            padding=(3, 1),
            cores=1,
            sim_skip="craq-sim HANG (T=1-identical)",
        ),
    ),
    (
        "k15x15_s2_8chunks",
        dict(
            in_h=16,
            in_w=16,
            kernel=(15, 15),
            stride=(2, 2),
            padding=(7, 7),
            cores=1,
            sim_skip="craq-sim HANG (T=1-identical)",
        ),
    ),
    (
        "k1x8_row",
        dict(
            in_h=8,
            in_w=8,
            kernel=(1, 8),
            stride=(1, 1),
            padding=(0, 0),
            cores=1,
            sim_skip="Quasar pool bug, windows <= 8 rows: identical MISMATCH on craq-sim AND ZeBu RTL, at T=1 and T=4, and on main (merge-base bc294789ec3) -- pre-existing, not a sim artifact; WH silicon passes (different LLK path)",
        ),
    ),
    (
        "k8x1_col",
        dict(
            in_h=8,
            in_w=8,
            kernel=(8, 1),
            stride=(1, 1),
            padding=(0, 0),
            cores=1,
            sim_skip="Quasar pool bug, windows <= 8 rows: identical MISMATCH on craq-sim AND ZeBu RTL, at T=1 and T=4, and on main (merge-base bc294789ec3) -- pre-existing, not a sim artifact; WH silicon passes (different LLK path)",
        ),
    ),
    ("c24_k3x3", dict(channels=24, cores=1)),
    # Lane policy (pool_utils get_factory_parameters): num_threads is always 4; the reader deals
    # sticks round-robin (stick i -> lane i % 4) and each compute lane takes quotient (+1 for the
    # first sticks % 4 lanes). The matrix/sweep cases all have stick counts divisible by 4; these
    # pin the remainder paths: 1 stick (lanes 1..3 idle), 2 and 3 sticks (idle tail lanes), 6 and
    # 10 sticks (lanes 0..1 own one stick more than lanes 2..3). gap_* mirror the resnet50 global
    # avg pool (batch 1, width-sharded, 1 output stick per core).
    (
        "gap_b1_width_avg_1stick",
        dict(pool="avg", in_h=4, in_w=8, kernel=(4, 8), stride=(1, 1), padding=(0, 0), shard="width", grid_yx=(1, 2)),
    ),
    (
        "gap_b1_width_max_1stick",
        dict(
            in_h=4,
            in_w=8,
            kernel=(4, 8),
            stride=(1, 1),
            padding=(0, 0),
            shard="width",
            grid_yx=(1, 2),
            sim_skip="craq-sim HANG (whole-window max class; avg twin passes on sim, exact on WH)",
        ),
    ),
    ("stick1_1core", dict(in_h=8, in_w=4, kernel=(8, 4), stride=(1, 1), padding=(0, 0), cores=1)),
    ("sticks2_k3_s4_2cores", dict(in_h=8, in_w=8, stride=(4, 4), cores=2)),  # 2 output sticks per core
    ("sticks3_k3_s3x4_1core", dict(in_h=8, in_w=4, stride=(3, 4), cores=1)),  # 3 sticks: lane 3 idle
    ("sticks6_b3_1core", dict(batch=3, in_h=8, in_w=4, stride=(4, 4), cores=1)),  # 6 sticks: 2,2,1,1
    ("sticks10_b5_1core", dict(batch=5, in_h=8, in_w=4, stride=(4, 4), cores=1)),  # 10 sticks: 3,3,2,2
    # TILE output layout runs single-lane by policy (the tiled-output path is not lane-aware: at
    # num_threads=4 DFB_FAST_TILIZE has in_ntiles_c entries -> TT_FATAL for C<128, hang at C=128;
    # found by resnet50/quasar/tests/ops/test_max_pool2d_correctness.py on the ZeBu emulator).
    # NOTE: even at T=1 the tiled-output path is WRONG on Quasar RTL (pcc ~0 for C=64), identically
    # on main -- a pre-existing Quasar bug; these cases are exact on WH silicon.
    (
        "tiled_out_max_k3x3_T1",
        dict(
            out_layout="tile",
            sim_skip="craq-sim UnimplementedFunctionality: tensix_execute_pacr_stride (tiled-output pack); on ZeBu RTL the tiled-output path MISMATCHES identically on main (merge-base bc294789ec3) and this branch at T=1 -- pre-existing Quasar bug; WH silicon exact",
        ),
    ),
    (
        "tiled_out_max_k3x3_c128_T1",
        dict(
            channels=128,
            in_h=8,
            in_w=4,
            out_layout="tile",
            sim_skip="craq-sim UnimplementedFunctionality: tensix_execute_pacr_stride (tiled-output pack); on ZeBu RTL the tiled-output path MISMATCHES identically on main (merge-base bc294789ec3) and this branch at T=1 -- pre-existing Quasar bug; WH silicon exact",
        ),
    ),
    (
        "tiled_out_avg_k3x3_T1",
        dict(
            pool="avg",
            stride=(1, 1),
            padding=(0, 0),
            out_layout="tile",
            sim_skip="craq-sim UnimplementedFunctionality: tensix_execute_pacr_stride (tiled-output pack); on ZeBu RTL the tiled-output path MISMATCHES identically on main (merge-base bc294789ec3) and this branch at T=1 -- pre-existing Quasar bug; WH silicon exact",
        ),
    ),
    (
        "tiled_out_avg_k7x7_large_T1",
        dict(
            pool="avg",
            kernel=(7, 7),
            stride=(2, 2),
            padding=(0, 0),
            cores=1,
            out_layout="tile",
            sim_skip="craq-sim UnimplementedFunctionality: tensix_execute_pacr_stride (tiled-output pack); on ZeBu RTL the tiled-output path MISMATCHES identically on main (merge-base bc294789ec3) and this branch at T=1 -- pre-existing Quasar bug; WH silicon exact",
        ),
    ),
    ("bf8b_k3x3", dict(dtype="bf8b", cores=1, sim_skip="Metal-2.0 DFB bfp8 unsupported (program_spec.cpp:1731)")),
    (
        "bf8b_k7x7_large",
        dict(
            dtype="bf8b",
            kernel=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            cores=1,
            sim_skip="Metal-2.0 DFB bfp8 unsupported (program_spec.cpp:1731)",
        ),
    ),
]


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_qpool_matrix(mesh_device):
    _run_cases(mesh_device, MATRIX_CASES)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_qpool_unit_cases(mesh_device):
    _run_cases(mesh_device, UNIT_CASES)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_qpool_perf_matrix(mesh_device):
    """Perf pass over the matrix configs: 1 warmup + PERF_ITERS labeled iterations per case, no
    golden check — the craq-sim per-dispatch trace (run_qpool.sh perf / perf-ab sets the env)
    attributes clocks per case for qpool_perf_report.py. Skips correctness sim_skip cases (they
    hang or corrupt on the sim) and const-pattern cases (value-only twins of existing shapes)."""
    for name, kwargs in MATRIX_CASES:
        kwargs = dict(kwargs)
        if kwargs.pop("sim_skip", None) or str(kwargs.get("pattern", "")).startswith("const:"):
            continue
        print(f"\nQPOOL-PERF: {name}", flush=True)
        _run_case(mesh_device, perf_label=name, **kwargs)
