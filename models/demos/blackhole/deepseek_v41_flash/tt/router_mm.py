# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused JIT router (``ttnn.generic_op``): split-K fp32 gate matmul + sqrt(softplus) + bias + exact top-k, ONE program.

Replaces the 4 ops of ``DSV41Gate._forward_fused`` (``ttnn.matmul`` fp32 HiFi4 on 6 cores 22 us, ``ttnn.add(activations)`` 6 us,
``ttnn.add(bias)`` 2.7 us, ``router_select`` 9.8 us).  Core (n, s) of a 12 x S grid (the 12 cores of a K slice are a 6 x 2 rectangle)
owns column tile n (12 tiles = 384 experts) and K slice s of S:
  * it accumulates its 160/S tiles in fp32 (same LLK matmul / fidelity / fp32 dest as the ttnn op; the bf16 x tile page is read
    half-size, rows 0..15) and the S-1 other slices send their partial tile to core (n, 0),
  * (n, 0) sums the partials (fixed order), applies the same SFPU softplus / sqrt and the bias add as the ttnn chain
    (INP_FLOAT32 polynomials: bit-identical to the ttnn ops), keeping the fp32 ``rank`` / ``score`` tiles in L1,
  * row t is owned by core (t % 12, (t / 12) % S): it pulls the row's 384 ranks / scores from the 12 reducers' L1, selects the
    exact fp32 top-k (lowest index wins ties), and writes w_i = bf16(score_i / (sum score + eps) * scale) and the uint16 ids.
With S == 1 everything is bit-identical to the ttnn chain + router_select; with S > 1 the K summation order differs (fp32 noise).
"""

import os

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.mhc_collapse import _acc, _cb, _hash, _kernel
from models.demos.blackhole.deepseek_v41_flash.tt.router_select import _bits

KT = 160  # 5120 / 32
NT = 12  # 384 / 32
GX, GY = (
    12,
    1,
)  # the NT cores of a K slice: logical x 0..GX-1, rows GY*s .. (core n = (n % GX, GY*s + n // GX)); 6 x 2 also works
SEM_PART, SEM_ROW, SEM_FA, SEM_FB = (
    0,
    1,
    2,
    3,
)  # partial arrivals, row arrivals, intra-core scan hand-shake (writer <-> reader RISC)


def relayout_weight(w, S):
    """w [5120, 384] (torch) -> torch [1, 1, KS*NT*S*32, 32]: tile page j*NT*S + n*S + s = w[(s*KS + j)*32 : +32, n*32 : +32]
    (step j of all cores reads consecutive pages = all DRAM banks evenly)."""
    KS = KT // S
    t = w.reshape(S, KS, 32, NT, 32).permute(1, 3, 0, 2, 4)  # [j, n, s, 32, 32]
    return t.reshape(1, 1, KS * NT * S * 32, 32).contiguous()


def bias_tile(bias_torch):
    """[384] fp32 -> torch [1, 1, 32, 384]: the bias in every row (so the kernel adds tile by tile)."""
    return bias_torch.reshape(1, 1, 1, NT * 32).expand(1, 1, 32, NT * 32).contiguous()


def scores_shard_config(T, k):
    """the height-sharded L1 config the MoE dispatch wants for the routing weights (tt_moe_decode_config)"""
    ny = min(8, T)
    nx = (T + ny - 1) // ny
    spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(nx - 1, ny - 1))]),
        [1, k],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec)


def router_fused(
    x, wk, bias, S, k, eps, scale, block=None, dump=False, out_l1=False, half=True, opage=None, gx=None, base=(0, 0)
):
    """x bf16 TILE [1,1,T,5120] (T <= 32), wk = relayout_weight(w, S), bias = bias_tile(b) -> (weights bf16 RM [T,1,1,k],
    indices uint16 RM [T,1,1,k]) [, rank, score fp32 tiles when ``dump``].  ``out_l1``: outputs in the layouts the MoE dispatch
    wants (ids L1 interleaved, weights height-sharded L1).  ``half``: SFPU over rows 0..15 only (T <= 16)."""
    _, _, T, K = (int(v) for v in x.shape)
    gx = gx or GX
    gy = NT // gx
    assert T <= 32 and K == KT * 32 and KT % S == 0 and gy * S <= 10
    KS = KT // S
    B = block or next(b for b in (10, 8, 5, 4, 2, 1) if KS % b == 0)
    mesh = x.device()
    bx, by = base
    px = [mesh.worker_core_from_logical_core(ttnn.CoreCoord(bx + c, 0)).x for c in range(gx)]
    py = [mesh.worker_core_from_logical_core(ttnn.CoreCoord(0, by + r)).y for r in range(gy * S)]
    mk = lambda dt, mc: ttnn.allocate_tensor_on_device(ttnn.Shape([T, 1, 1, k]), dt, ttnn.ROW_MAJOR_LAYOUT, mesh, mc)
    if out_l1:
        w_out, i_out = mk(ttnn.bfloat16, scores_shard_config(T, k)), mk(ttnn.uint16, ttnn.L1_MEMORY_CONFIG)
    else:
        w_out, i_out = mk(ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG), mk(ttnn.uint16, ttnn.DRAM_MEMORY_CONFIG)
    opage = opage or k * 2
    if dump:
        mkt = lambda: ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, T, NT * 32]), ttnn.float32, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
        )
        rank, score = mkt(), mkt()
    else:
        rank, score = bias, bias  # accessor placeholders (never written)
    core_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(bx, by), ttnn.CoreCoord(bx + gx - 1, by + gy * S - 1))])
    rt_r, rt_w, rt_c = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for s in range(S):
        for n in range(NT):
            cx, cy = bx + n % gx, by + gy * s + n // gx
            rt_r[cx][cy] = [n, s]
            rt_w[cx][cy] = [n, s]
            rt_c[cx][cy] = [1 if s == 0 else 0]
    f32, bf = ttnn.float32, ttnn.bfloat16
    CX, CW, CP, CR, CB, CK, CS, CG, CO = range(9)
    cbs = [
        _cb(core_set, CX, KS, 2048, bf),
        _cb(core_set, CW, KS, 2048, bf),
        _cb(core_set, CP, 1, 4096, f32),
        _cb(core_set, CR, max(1, S - 1), 4096, f32),
        _cb(core_set, CB, 1, 4096, f32),
        _cb(core_set, CK, 1, 4096, f32),
        _cb(core_set, CS, 1, 4096, f32),
        _cb(core_set, CG, 1, 4096, f32),
        _cb(core_set, CO, 1, 128, f32),
    ]
    XB = 1024 if T <= 16 else 2048
    reader = _kernel(
        "router_mm_reader.cpp",
        core_set,
        [CX, CW, KS, B, XB, NT * S, T, CG, SEM_FA, SEM_FB, k] + _acc(x) + _acc(wk),
        rt_r,
        None,
        ttnn.ReaderConfigDescriptor(),
        common=[x.buffer_address(), wk.buffer_address()],
    )
    reader.defines = [(f"SKIP{c.upper()}", "1") for c in os.environ.get("RMM_SKIP", "")]
    writer = _kernel(
        "router_fused_writer.cpp",
        core_set,
        [
            CP,
            CR,
            CB,
            CK,
            CS,
            CG,
            CO,
            S,
            SEM_PART,
            SEM_ROW,
            T,
            k,
            _bits(eps),
            _bits(scale),
            int(dump),
            opage,
            gx,
            SEM_FA,
            SEM_FB,
        ]
        + _acc(bias)
        + _acc(w_out)
        + _acc(i_out)
        + _acc(rank)
        + _acc(score),
        rt_w,
        None,
        ttnn.WriterConfigDescriptor(),
        common=[
            bias.buffer_address(),
            w_out.buffer_address(),
            i_out.buffer_address(),
            rank.buffer_address(),
            score.buffer_address(),
        ]
        + px
        + py,
    )
    cc = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
    )
    modes = [ttnn.UnpackToDestMode.Default] * 64
    for c in (CP, CR, CB):
        modes[c] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cc.unpack_to_dest_mode = modes
    compute = _kernel(
        "router_mm_compute.cpp",
        core_set,
        [CX, CW, CP, CR, CB, CK, CS, KS, B, S, _bits(1.0), _bits(1.0), _bits(20.0), int(T <= 16 and half)],
        rt_c,
        None,
        cc,
    )
    compute.defines = [("INP_FLOAT32", "1")]  # fp32 softplus polynomial (as ttnn defines it for fp32 inputs)
    sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=core_set, initial_value=0) for i in range(4)]
    prog = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=sems, cbs=cbs)
    prog.custom_program_hash = _hash(
        0x5B4,
        gx,
        base,
        int(half),
        T,
        S,
        B,
        XB,
        k,
        _bits(eps),
        _bits(scale),
        int(dump),
        out_l1,
        opage,
        os.environ.get("RMM_SKIP", ""),
        tuple(px),
        tuple(py),
        tuple(_acc(x)),
        tuple(_acc(wk)),
        tuple(_acc(bias)),
        tuple(_acc(w_out)),
        tuple(_acc(i_out)),
        tuple(_acc(rank)),
        tuple(_acc(score)),
    )
    ttnn.generic_op([x, wk, bias, rank, score, w_out, i_out] if dump else [x, wk, bias, w_out, i_out], prog)
    return (w_out, i_out, rank, score) if dump else (w_out, i_out)
