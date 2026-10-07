# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Round 3 eltwise binary (#58723, #58724 review): single-core twins of the compute loops of production callers that the PR
keeps on main's per-face program. Each twin runs the caller's compute loop (kernels/tw_*.cpp, copied from the caller and
cited there) on core (0, 0) through ttnn.generic_op, with the caller's tile shapes, data formats, math fidelity and DEST
section sizes. The inputs sit in L1 tensors under globally allocated CBs and the reader only pushes (the writer only
pops), so the device kernel time is the compute loop's. k1 runs the loop once (the op's shape); kN runs it N times on the
same inputs. Main against the opt-in: tests/eb_r3_ci/twins/optin_twins.txt with ab_set.sh (time) and bits_ab.sh (bits)."""
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
# 1. zero_padded_kv_cache: mul_tiles of each cache tile by the bf16 row mask, one tile per DEST section, HiFi4
#    (ComputeConfigDescriptor{}); cache bfloat8_b (the TILE path); Wt = 2 (the CI test's head_dim 64), 18 (kvpe 576).
@pytest.mark.parametrize("iters", [1, 64], ids=["k1", "k64"])
@pytest.mark.parametrize("wt", [2, 18], ids=["Wt2", "Wt18"])
def test_twin_zpkv(device, wt, iters):
    g = _gen(1)
    src = torch.randn(32, 32 * wt, generator=g)
    mask = torch.randn(32, 32, generator=g)
    t_src = _l1(device, src, ttnn.bfloat8_b, (32, 32))
    t_mask = _l1(device, mask, ttnn.bfloat16, (32, 32))
    t_out = _l1(device, torch.zeros(32, 32 * wt), ttnn.bfloat8_b, (32, 32))
    (out,) = _run(
        device,
        "tw_zpkv.cpp",
        [0, 1, 2, iters],
        _cfg(ttnn.MathFidelity.HiFi4),
        ins=[(0, t_src, wt, 1), (1, t_mask, 1, 1)],
        outs=[(2, t_out, wt, 1)],
        local_cbs=[],
        iters=iters,
        common_rt=[0, 0, 0, 0, 0, 0, 0, wt],
    )
    s = ttnn.to_torch(t_src).float()
    m = ttnn.to_torch(t_mask).float()
    assert _pcc(out, s * m.repeat(1, wt)) > 0.99


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
# 3. all_gather_minimal_matmul_async's fused addcmul (tt_dit Wan, bf16 gate, with bias): HiFi2 and fp32 DEST
#    (models/tt_dit/layers/linear.py:114-120), fp32 intermediate CB; block 8x4 (Wan to_out / ff2, models/tt_dit/utils/
#    matmul.py) and 8x8 (the Wan addcmul gate test); broadcast gate (mul_tiles_bcast<ROW>) and full gate (mul_tiles).
@pytest.mark.parametrize("iters", [1, 8], ids=["k1", "k8"])
@pytest.mark.parametrize("gate", ["broadcast", "full"])
@pytest.mark.parametrize("mn", [(8, 4), (8, 8)], ids=["m8n4", "m8n8"])
def test_twin_agmm(device, mn, gate, iters):
    M, N = mn
    g = _gen(3)
    bf = ttnn.bfloat16
    T = (32, 32)
    x = torch.randn(M * 32, N * 32, generator=g)
    bias = torch.randn(32, N * 32, generator=g)
    a = torch.randn(M * 32, N * 32, generator=g)
    bc = gate == "broadcast"
    b = torch.randn(32 if bc else M * 32, N * 32, generator=g)
    t_x = _l1(device, x, ttnn.float32, T)
    t_bias, t_a, t_b = _l1(device, bias, bf, T), _l1(device, a, bf, T), _l1(device, b, bf, T)
    t_out = _l1(device, torch.zeros(M * 32, N * 32), bf, T)
    ins = [(0, t_x, M * N, 1), (4, t_bias, N, 1), (6, t_b, N, 1 if bc else M), (5, t_a, N, M)]
    (out,) = _run(
        device,
        "tw_agmm.cpp",
        [M, N, iters],
        _cfg(ttnn.MathFidelity.HiFi2, fp32=True),
        ins,
        [(2, t_out, N, M)],
        [_local_cb(3, ttnn.float32, T, M * N)],
        iters,
        defines=[("FUSE_BIAS", "1"), ("FUSE_TERNARY", "1")],
        common_rt=[_bits(1.0), 1 if bc else 0],
    )
    xb = ttnn.to_torch(t_x).float() + ttnn.to_torch(t_bias).float()[0:1, :]
    bb = ttnn.to_torch(t_b).float()
    bb = bb.reshape(1, 32, N * 32)[:, 0:1, :].expand(M, 32, N * 32).reshape(M * 32, N * 32) if bc else bb
    assert _pcc(out, ttnn.to_torch(t_a).float() + xb * bb) > 0.99


# ---------------------------------------------------------------------------------------------------------------------
# 4. strided reduce-scatter's reduction with tt_dit's fused addcmul: bf16, HiFi4 and bf16 DEST (ComputeConfig{} in
#    strided_reduce_scatter_async_program.cpp:736), tile_granularity 8 (bf16, 4 KB packets); ring 4 (the Galaxy's axis 0)
#    and 8 (axis 1): ring - 2 plain accumulation steps and the final step with the addcmul; broadcast and full gate; scalar 0.5.
@pytest.mark.parametrize("iters", [1, 16], ids=["k1", "k16"])
@pytest.mark.parametrize("gate", ["broadcast", "full"])
@pytest.mark.parametrize("ring", [4, 8], ids=["ring4", "ring8"])
def test_twin_srs(device, ring, gate, iters):
    tg = 8
    g = _gen(4)
    bf = ttnn.bfloat16
    T = (32, 32)
    steps = ring - 1
    inp = torch.randn(32, 32 * tg * steps, generator=g)
    inter = torch.randn(32, 32 * tg * steps, generator=g)
    a, b = torch.randn(32, 32 * tg, generator=g), torch.randn(32, 32 * tg, generator=g)
    t_in, t_im, t_a, t_b = (_l1(device, v, bf, T) for v in (inp, inter, a, b))
    t_out = _l1(device, torch.zeros(32, 32 * tg * steps), bf, T)
    defines = [("FUSE_RS_ADDCMUL", "1")] + ([("ADDCMUL_B_BROADCAST", "1")] if gate == "broadcast" else [])
    (out,) = _run(
        device,
        "tw_srs.cpp",
        [0, 1, 2, tg, ring, 4, 5, 6, iters],
        _cfg(ttnn.MathFidelity.HiFi4),
        [(0, t_in, tg, steps), (1, t_im, tg, steps), (6, t_b, tg, 1), (5, t_a, tg, 1)],
        [(2, t_out, tg, steps)],
        [_local_cb(4, bf, T, 6 * tg)],
        iters,
        defines=defines,
        compute_rt=[_bits(0.5)],
    )
    acc = ttnn.to_torch(t_in).float() + ttnn.to_torch(t_im).float()
    bb = ttnn.to_torch(t_b).float()
    if gate == "broadcast":
        bb = bb.reshape(32, tg, 32)[0:1].expand(32, tg, 32).reshape(32, 32 * tg)
    last = slice(32 * tg * (steps - 1), None)
    ref = acc.clone()
    ref[:, last] = ttnn.to_torch(t_a).float() + 0.5 * acc[:, last] * bb
    assert _pcc(out, ref) > 0.99


# ---------------------------------------------------------------------------------------------------------------------
# 5. deepseek_v3_b1 reduce_to_one (#58724): copy_tile then add_reuse_dest_tiles<DEST_TO_SRCA> into one DEST section, one
#    32x32 bf16 compute tile per worker (shard [1, 1024]), HiFi4 and bf16 DEST (micro_ops/reduce_to_one_b1/op.py:566-569);
#    ROOT3, ROOT2, ROOT1 (1, 2, 3 accumulations).
@pytest.mark.parametrize("iters", [1, 64], ids=["k1", "k64"])
@pytest.mark.parametrize("rounds", [1, 2, 3], ids=["root3", "root2", "root1"])
def test_twin_r21(device, rounds, iters):
    g = _gen(5)
    bf = ttnn.bfloat16
    T = (32, 32)
    local = torch.randn(32, 32, generator=g)
    recv = torch.randn(32, 32 * rounds, generator=g)
    t_l, t_r = _l1(device, local, bf, T), _l1(device, recv, bf, T)
    t_out = _l1(device, torch.zeros(32, 32), bf, T)
    (out,) = _run(
        device,
        "tw_r21.cpp",
        [1, 0, 1, 5, rounds, iters],
        _cfg(ttnn.MathFidelity.HiFi4),
        [(0, t_l, 1, 1), (1, t_r, 1, rounds)],
        [(5, t_out, 1, 1)],
        [],
        iters,
    )
    ref = ttnn.to_torch(t_l).float() + ttnn.to_torch(t_r).float().reshape(32, rounds, 32).sum(1)
    assert _pcc(out, ref) > 0.999


# ---------------------------------------------------------------------------------------------------------------------
# 6. deepseek_v3_b1 GatedReduce with the per-K scalar (#58724, deepseek_binary_dest_reuse_tiles): moe / decoder_block's
#    SRAM path, 8x32 tiles (fused_ops/moe/op.py:5528), LoFi and bf16 DEST (moe/op.py:7093-7098), tiles_per_k 8, 8 active
#    experts; and the 16x16 face-view tile of the shared path (shared_expert / gated_local_reduce_down_proj op.py).
@pytest.mark.parametrize("iters", [1, 16], ids=["k1", "k16"])
@pytest.mark.parametrize("tile", [(8, 32), (16, 16)], ids=["t8x32", "t16x16"])
def test_twin_gr(device, tile, iters):
    tpk, kn = 8, 8
    g = _gen(6)
    bf = ttnn.bfloat16
    h, w = tile
    g1 = torch.randn(h, w * tpk * kn, generator=g) * 0.25
    g2 = torch.randn(h, w * tpk * kn, generator=g) * 0.25
    sc = torch.rand(h, w * kn, generator=g) + 0.5
    t_g1, t_g2, t_sc = _l1(device, g1, bf, tile), _l1(device, g2, bf, tile), _l1(device, sc, bf, tile)
    t_out = _l1(device, torch.zeros(h, w * kn), bf, tile)
    (out,) = _run(
        device,
        "tw_gr.cpp",
        [0, 1, 2, 3, 4, tpk, kn, iters],
        _cfg(ttnn.MathFidelity.LoFi),
        [(0, t_g1, tpk, kn), (1, t_g2, tpk, kn), (4, t_sc, 1, kn)],
        [(3, t_out, 1, kn)],
        [_local_cb(2, bf, tile, 2)],
        iters,
    )
    s1 = ttnn.to_torch(t_g1).float().reshape(h, kn, tpk, w).sum(2)
    s2 = ttnn.to_torch(t_g2).float().reshape(h, kn, tpk, w).sum(2)
    scal = ttnn.to_torch(t_sc).float().reshape(h, kn, w)[0, :, 0].reshape(1, kn, 1)
    ref = (torch.nn.functional.silu(s1) * scal * s2).reshape(h, kn * w)
    assert _pcc(out, ref) > 0.98


# ---------------------------------------------------------------------------------------------------------------------
# 7. deepseek_v3_b1 post_sdpa's SDPA reduce (round 1): 8x32 tiles, L [8, 512] per worker (16 tiles), compute block 8 in
#    2 blocks (micro_ops/sdpa_reduce_to_all/config.py, 15232 B payload), HiFi4 and bf16 DEST (post_sdpa/op.py:1533-1536).
@pytest.mark.parametrize("iters", [1, 32], ids=["k1", "k32"])
def test_twin_psdpa(device, iters):
    g = _gen(7)
    bf = ttnn.bfloat16
    T = (8, 32)
    ll, nl = torch.randn(8, 512, generator=g), torch.randn(8, 512, generator=g)
    lms, nms = torch.randn(8, 32, generator=g), torch.randn(8, 32, generator=g)
    lms[:, 1], nms[:, 1] = lms[:, 1].abs() + 1, nms[:, 1].abs() + 1
    t_ll, t_lms, t_nl, t_nms = (_l1(device, v, bf, T) for v in (ll, lms, nl, nms))
    t_rl, t_rms = _l1(device, torch.zeros(8, 512), bf, T), _l1(device, torch.zeros(8, 32), bf, T)
    outs = _run(
        device,
        "tw_psdpa.cpp",
        [0, 1, 2, 3, 4, 5, _bits(1.0), 8, 2, iters],
        _cfg(ttnn.MathFidelity.HiFi4),
        [(0, t_ll, 16, 1), (1, t_lms, 1, 1), (2, t_nl, 16, 1), (3, t_nms, 1, 1)],
        [(4, t_rl, 8, 2), (5, t_rms, 1, 1)],
        [],
        iters,
    )
    assert all(torch.isfinite(x).all() for x in outs)


# ---------------------------------------------------------------------------------------------------------------------
# 8. deepseek_v3_b1 EltwiseMul with the expert scale (#58724 third review, deepseek_binary_dest_reuse_tiles after the scalar
#    multiply): the fused moe's and decoder block's 1x32 tiles, one per expert for 8 experts (moe/op.py:913-914, mul_num_tiles 1
#    and mul_num_experts 8 in the mock-cluster build of test_moe_fused_with_reduce), and moe_routed_expert's one 16x16 tile
#    (moe_routed_expert/op.py:104-125); LoFi and bf16 DEST (moe/op.py:7093-7098, moe_routed_expert/op.py:2030-2032).
@pytest.mark.parametrize("iters", [1, 64], ids=["k1", "k64"])
@pytest.mark.parametrize("shape", [((1, 32), 1, 8), ((16, 16), 1, 1)], ids=["t1x32e8", "t16x16e1"])
def test_twin_em(device, shape, iters):
    tile, nt, ne = shape
    h, w = tile
    total = nt * ne
    g = _gen(8)
    bf = ttnn.bfloat16
    in0 = torch.randn(h, w * total, generator=g)
    in1 = torch.randn(h, w * total, generator=g)
    sc = torch.rand(h, w * ne, generator=g) + 0.5
    t_in0, t_in1, t_sc = _l1(device, in0, bf, tile), _l1(device, in1, bf, tile), _l1(device, sc, bf, tile)
    t_out = _l1(device, torch.zeros(h, w * total), bf, tile)
    (out,) = _run(
        device,
        "tw_em.cpp",
        [0, 1, 3, 4, nt, ne, 0, iters],
        _cfg(ttnn.MathFidelity.LoFi),
        [(0, t_in0, total, 1), (1, t_in1, total, 1), (4, t_sc, 1, ne)],
        [(3, t_out, total, 1)],
        [],
        iters,
    )
    s = ttnn.to_torch(t_sc).float()
    scal = torch.stack([s[0, e * w] for e in range(ne)]).repeat_interleave(nt * w).reshape(1, -1)
    ref = ttnn.to_torch(t_in0).float() * scal * ttnn.to_torch(t_in1).float()
    assert _pcc(out, ref) > 0.98
