# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Round 3 eltwise binary (#58723 third review): single-core twins of the compute loops that multiply above LoFi on tiles the
PR head keeps on the per-face program (8x32 and 16x32 column broadcasts, post_sdpa's srcB-reuse multiply). Each twin runs the caller's compute loop (kernels/tw_*.cpp, copied from the caller and
cited there) on core (0, 0) through ttnn.generic_op, with the caller's tile shapes, data formats, math fidelity and DEST
section sizes. The inputs sit in L1 tensors under globally allocated CBs and the reader only pushes (the writer only
pops), so the device kernel time is the compute loop's. k1 runs the loop once (the op's shape); kN runs it N times on the
same inputs. Main's program against the whole-tile program: tests/eb_r3_ci/twins/optin_wt.txt with ab_set.sh (time) and
bits_ab.sh (bits)."""
import struct

import pytest
import torch
import ttnn

KDIR = "tests/eb_r3_ci/twins/kernels"
CORE = ttnn.CoreCoord(0, 0)
CORES = ttnn.CoreRangeSet([ttnn.CoreRange(CORE, CORE)])


@pytest.fixture(scope="module")
def device():
    dev = ttnn.CreateDevice(device_id=0)
    yield dev
    ttnn.close_device(dev)


def _bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def _gen(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _l1(dev, t, dtype, tile):
    """A single-core L1 tensor holding t (2D) in tiles of shape `tile`; page i is tile i in row-major tile order."""
    spec = ttnn.ShardSpec(CORES, [t.shape[0], t.shape[1]], ttnn.ShardOrientation.ROW_MAJOR)
    mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec)
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, tile=ttnn.Tile(list(tile)), device=dev, memory_config=mc)


def _local_cb(idx, dtype, tile, pages):
    t = ttnn.Tile(list(tile))
    page = t.get_tile_size(dtype)
    fmt = ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dtype, page_size=page, tile=ttnn.TileDescriptor(t))
    return ttnn.CBDescriptor(total_size=pages * page, core_ranges=CORES, format_descriptors=[fmt])


def _rt(args):
    r = ttnn.RuntimeArgs()
    r[0][0] = list(args)
    return r


def _run(dev, compute, ct, cfg, ins, outs, local_cbs, iters, defines=(), common_rt=(), compute_rt=(0,)):
    """ins / outs: (cb index, L1 tensor, pages per push / pop, pushes / pops per iteration). Returns the outputs."""
    cbs = [ttnn.cb_descriptor_from_sharded_tensor(i, t) for i, t, _, _ in ins + outs] + list(local_cbs)
    rd = [iters, len(ins), max(p for *_, p in ins)] + [x for i, _, n, p in ins for x in (i, n, p)]
    wr = [iters, len(outs), max(p for *_, p in outs)] + [x for i, _, n, p in outs for x in (i, n, p)]
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/tw_reader.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=CORES,
            compile_time_args=[],
            runtime_args=_rt(rd),
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/tw_writer.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=CORES,
            compile_time_args=[],
            runtime_args=_rt(wr),
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/{compute}",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=CORES,
            compile_time_args=list(ct),
            defines=list(defines),
            runtime_args=_rt(compute_rt),
            common_runtime_args=list(common_rt),
            config=cfg,
        ),
    ]
    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    ttnn.generic_op([t for _, t, _, _ in ins + outs], program)
    return [ttnn.to_torch(t).float() for _, t, _, _ in outs]


def _cfg(fidelity, fp32=False, approx=False):
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=fidelity, fp32_dest_acc_en=fp32, dst_full_sync_en=False, math_approx_mode=approx
    )


def _pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


ITERS = {"k1": 1}


# ---------------------------------------------------------------------------------------------------------------------
# 2. reduce_to_root: the whole compute kernel; 8x32 stats tiles, bf16 CBs, fp32 DEST, HiFi4 (reduce_to_root_program.cpp:636);
#    the CI trace test's shapes (Sq_chunk_t 1, vDHt 4); loop_size 1 (a non-root receiver) and 2 (the root).
@pytest.mark.parametrize("iters", [1, 32], ids=["k1", "k32"])
@pytest.mark.parametrize("loop_size", [1, 2], ids=["receiver", "root"])
def test_twin_r2r(device, loop_size, iters):
    g = _gen(2)
    T = (8, 32)
    bf = ttnn.bfloat16
    l1, l2 = torch.randn(8, 128, generator=g) * 0.5, torch.randn(8, 128, generator=g) * 0.5 + 1
    s1, s2 = torch.rand(8, 32, generator=g) * 0.5 + 1.0, torch.rand(8, 32, generator=g) * 0.5 + 1.1
    m1, m2 = torch.randn(8, 32, generator=g) * 0.5, torch.randn(8, 32, generator=g) * 0.5 + 1
    t = {c: _l1(device, x, bf, T) for c, x in ((5, m1), (2, m2), (4, s1), (1, s2), (3, l1), (0, l2))}
    o = {c: _l1(device, torch.zeros(8, w), bf, T) for c, w in ((11, 128), (13, 32), (12, 32))}
    twice = 2 if loop_size == 2 else 1
    ins = [(5, t[5], 1, twice), (2, t[2], 1, 1), (4, t[4], 1, twice), (1, t[1], 1, 1), (3, t[3], 4, twice), (0, t[0], 4, 1)]
    outs = [(11, o[11], 4, 1), (13, o[13], 1, 1), (12, o[12], 1, 1)]
    locs = [_local_cb(c, bf, T, n) for c, n in ((14, 1), (15, 1), (16, 1), (19, 1), (20, 1), (21, 1), (22, 4), (23, 4), (8, 4), (9, 1), (10, 1))]
    ct = [11, 0, 3, 1, 5, 2, 13, 14, 4, 15, 12, 16, 19, 20, 21, 22, 23, _bits(1.0), 1, 4, loop_size, 8, 9, 10, iters]
    outs_t = _run(device, "tw_r2r.cpp", ct, _cfg(ttnn.MathFidelity.HiFi4, fp32=True, approx=True), ins, outs, locs, iters)
    assert all(torch.isfinite(x).all() for x in outs_t)


# ---------------------------------------------------------------------------------------------------------------------
# 7. deepseek_v3_b1 post_sdpa's SDPA reduce: 8x32 tiles, L [8, 512] per worker (16 tiles), compute block 8 in 2 blocks
#    (micro_ops/sdpa_reduce_to_all/config.py, 15232 B payload), HiFi4 and bf16 DEST (post_sdpa/op.py:1533-1536); round 1
#    (unnormalized, two blocks of 8) and round 2 (normalized and untilized, one dense block of 16).
@pytest.mark.parametrize("iters", [1, 32], ids=["k1", "k32"])
@pytest.mark.parametrize("rnd", [1, 2], ids=["r1", "r2"])
def test_twin_psdpa(device, rnd, iters):
    g = _gen(7)
    bf = ttnn.bfloat16
    T = (8, 32)
    ll, nl = torch.randn(8, 512, generator=g), torch.randn(8, 512, generator=g)
    lms, nms = torch.randn(8, 32, generator=g), torch.randn(8, 32, generator=g)
    lms[:, 1], nms[:, 1] = lms[:, 1].abs() + 1, nms[:, 1].abs() + 1
    t_ll, t_lms, t_nl, t_nms = (_l1(device, v, bf, T) for v in (ll, lms, nl, nms))
    t_rl, t_rms = _l1(device, torch.zeros(8, 512), bf, T), _l1(device, torch.zeros(8, 32), bf, T)
    outs = [(4, t_rl, 8, 2), (5, t_rms, 1, 1)] if rnd == 1 else [(4, t_rl, 16, 1)]
    res = _run(
        device,
        "tw_psdpa.cpp",
        [0, 1, 2, 3, 4, 5, _bits(1.0), 8, 2, iters, rnd],
        _cfg(ttnn.MathFidelity.HiFi4),
        [(0, t_ll, 16, 1), (1, t_lms, 1, 1), (2, t_nl, 16, 1), (3, t_nms, 1, 1)],
        outs,
        [],
        iters,
    )
    assert all(torch.isfinite(x).all() for x in res)


# ---------------------------------------------------------------------------------------------------------------------
# 8. SDPA decode's output rescale on half tiles (sdpa_decode_program_factory.cpp:479-517: 16x32 im and stats tiles for causal
#    decode with up to 16 q heads, bf16), HiFi2 by default (sdpa_decode.cpp:74) and HiFi4, 16-bit and fp32 DEST; Sq_chunk_t 1,
#    vDHt 4 (head dim 128) and 8 (256); mode 0 the k-chunk rescale (immediate pop), mode 1 the DHT_GRANULARITY form of the
#    tree reduction and the root (DHT_GRANULARITY = min(vDHt, DEST tiles), sdpa_decode_program_factory.cpp:438-443).
@pytest.mark.parametrize("iters", [1, 32], ids=["k1", "k32"])
@pytest.mark.parametrize("mode", [0, 1], ids=["immediate", "granular"])
@pytest.mark.parametrize("vdht", [4, 8], ids=["vDHt4", "vDHt8"])
@pytest.mark.parametrize("fp32", [False, True], ids=["bf16dest", "fp32dest"])
@pytest.mark.parametrize("fid", ["HiFi2", "HiFi4"])
def test_twin_sdpad(device, fid, fp32, vdht, mode, iters):
    g = _gen(8)
    bf = ttnn.bfloat16
    T = (16, 32)
    acc = torch.randn(16, 32 * vdht, generator=g)
    stats = torch.rand(16, 32, generator=g) + 0.25
    t_acc, t_stats = _l1(device, acc, bf, T), _l1(device, stats, bf, T)
    t_out = _l1(device, torch.zeros(16, 32 * vdht), bf, T)
    gran = min(vdht, 4 if fp32 else 8)
    (out,) = _run(
        device,
        "tw_sdpad.cpp",
        [0, 1, 2, 1, vdht, mode, iters],
        _cfg(getattr(ttnn.MathFidelity, fid), fp32=fp32),
        [(0, t_acc, vdht, 1), (1, t_stats, 1, 1)],
        [(2, t_out, vdht, 1)],
        [],
        iters,
        defines=[("DHT_GRANULARITY", str(gran))],
    )
    a = ttnn.to_torch(t_acc).float()
    s = ttnn.to_torch(t_stats).float()[:, 0:1]
    assert _pcc(out, a * s) > 0.999
