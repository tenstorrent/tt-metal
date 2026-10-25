# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""JIT router front end (``ttnn.generic_op``): split-K fp32 gate matmul + sqrt(softplus) + bias in ONE program.

Replaces ``ttnn.matmul`` (fp32, HiFi4, 6 cores x 2 column tiles, 22 us) + ``ttnn.add(activations=[softplus, sqrt])`` (6 us) +
``ttnn.add(bias)`` (2.7 us) of ``DSV41Gate._forward_fused``.  Core (n, s) of a 12 x S grid owns column tile n (12 tiles = 384
experts) and K slice s of S: it accumulates its 160/S tiles in fp32 (same LLK matmul / fidelity / fp32 dest as the ttnn op), the
S-1 slices send their partial tile to the core (n, 0), which sums the partials (fixed order), applies the same SFPU
softplus / sqrt and the bias add as the ttnn chain, and writes the fp32 ``rank`` and ``score`` tiles ``router_select`` reads.
With S == 1 the logits are bit-identical to the ttnn matmul; with S > 1 the K summation order differs (fp32 rounding noise).
"""

import os

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.mhc_collapse import _acc, _cb, _hash, _kernel
from models.demos.blackhole.deepseek_v41_flash.tt.router_select import _bits

KT = 160  # 5120 / 32
NT = 12  # 384 / 32


def relayout_weight(w, S):
    """w [5120, 384] (torch) -> torch [1, 1, KS*NT*S*32, 32]: tile page j*NT*S + n*S + s = w[(s*KS + j)*32 : +32, n*32 : +32]
    (step j of all cores reads consecutive pages = all DRAM banks evenly)."""
    KS = KT // S
    t = w.reshape(S, KS, 32, NT, 32).permute(1, 3, 0, 2, 4)  # [j, n, s, 32, 32]: page j*NT*S + n*S + s
    return t.reshape(1, 1, KS * NT * S * 32, 32).contiguous()


def bias_tile(bias_torch):
    """[384] fp32 -> torch [1, 1, 32, 384]: the bias in every row (so the kernel adds tile by tile)."""
    return bias_torch.reshape(1, 1, 1, NT * 32).expand(1, 1, 32, NT * 32).contiguous()


def router_mm(x, wk, bias, S, block=None, mode=0, fid=None, approx=False):
    """x bf16 TILE [1,1,T,5120] (T <= 32), wk = relayout_weight(w, S) bf16 TILE, bias = bias_tile(b) fp32 TILE [1,1,32,384].
    Returns (rank, score): fp32 TILE [1,1,T,384] (rows >= 16 are garbage when T <= 16)."""
    _, _, T, K = (int(v) for v in x.shape)
    assert T <= 32 and K == KT * 32 and KT % S == 0
    KS = KT // S
    B = block or next(b for b in (10, 8, 5, 4, 2, 1) if KS % b == 0)
    mesh = x.device()
    mk = lambda: ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, T, NT * 32]), ttnn.float32, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
    )
    rank, score = mk(), mk()
    cores = [(n, s) for s in range(S) for n in range(NT)]
    core_set = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(NT - 1, S - 1))]
    )  # rectangle n in [0, NT), s in [0, S)
    rt_r, rt_w, rt_c = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for n, s in cores:
        red = mesh.worker_core_from_logical_core(ttnn.CoreCoord(n, 0))
        rt_r[n][s] = [s * KS, n * S + s]
        rt_w[n][s] = [n, s, red.x, red.y]
        rt_c[n][s] = [1 if s == 0 else 0]
    f32, bf = ttnn.float32, ttnn.bfloat16
    CX, CW, CP, CR, CB, CK, CS = range(7)
    cbs = [
        _cb(core_set, CX, KS, 2048, bf),
        _cb(core_set, CW, KS, 2048, bf),
        _cb(core_set, CP, 1, 4096, f32),
        _cb(core_set, CR, max(1, S - 1), 4096, f32),
        _cb(core_set, CB, 1, 4096, f32),
        _cb(core_set, CK, 1, 4096, f32),
        _cb(core_set, CS, 1, 4096, f32),
    ]
    XB = 1024 if T <= 16 else 2048
    reader = _kernel(
        "router_mm_reader.cpp",
        core_set,
        [CX, CW, KS, B, XB, NT * S] + _acc(x) + _acc(wk),
        rt_r,
        None,
        ttnn.ReaderConfigDescriptor(),
        common=[x.buffer_address(), wk.buffer_address()],
    )
    reader.defines = [(f"SKIP{c.upper()}", "1") for c in os.environ.get("RMM_SKIP", "")]
    writer = _kernel(
        "router_mm_writer.cpp",
        core_set,
        [CP, CR, CB, CK, CS, S, 0] + _acc(bias) + _acc(rank) + _acc(score),
        rt_w,
        None,
        ttnn.WriterConfigDescriptor(),
        common=[bias.buffer_address(), rank.buffer_address(), score.buffer_address()],
    )
    cc = ttnn.ComputeConfigDescriptor(
        math_fidelity=fid or ttnn.MathFidelity.HiFi4, math_approx_mode=approx, fp32_dest_acc_en=True
    )
    modes = [ttnn.UnpackToDestMode.Default] * 64
    for c in (CP, CR, CB):
        modes[c] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cc.unpack_to_dest_mode = modes
    compute = _kernel(
        "router_mm_compute.cpp",
        core_set,
        [CX, CW, CP, CR, CB, CK, CS, KS, B, S, _bits(1.0), _bits(1.0 / 1.0), _bits(20.0), mode],
        rt_c,
        None,
        cc,
    )
    compute.defines = [("INP_FLOAT32", "1")]  # fp32 softplus polynomial (as ttnn defines it for fp32 inputs)
    sem = ttnn.SemaphoreDescriptor(id=0, core_ranges=core_set, initial_value=0)
    prog = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[sem], cbs=cbs)
    prog.custom_program_hash = _hash(
        0x5B3,
        mode,
        os.environ.get("RMM_SKIP", ""),
        str(fid),
        approx,
        T,
        S,
        B,
        XB,
        tuple(_acc(x)),
        tuple(_acc(wk)),
        tuple(_acc(bias)),
        tuple(_acc(rank)),
        tuple(_acc(score)),
    )
    ttnn.generic_op([x, wk, bias, rank, score], prog)
    return rank, score
