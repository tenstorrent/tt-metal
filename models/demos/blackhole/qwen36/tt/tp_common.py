# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TP helpers for Qwen3.5/3.6 on Blackhole (9B single-device + 27B TP=4 / TP=8).

Used only when num_devices > 1. DRAM-sharded matmul cfgs, prefill progcfgs,
mesh shard/replicate, FP8 dequant, HF weight reorder for per-device sharding.
"""
import functools
import math
import os

import torch

import ttnn
from models.common.utility_functions import is_blackhole

# Hardware constants
TILE_SIZE = 32
DRAM_CORES = 8
DRAM_GRID = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(DRAM_CORES - 1, 0))})


# Compute kernel configs
COMPUTE_HIFI2 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=True,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)


# --- Prefill matmul compute-kernel-config levers (step2, 2026-09-22; read once at import) --------
# QWEN36_PREFILL_MM_PACKER_L1_ACC (default "1"): packer_l1_acc used by every PREFILL (T>1) matmul
#   ckc built via prefill_matmul_ckc(). "0" = legacy (False -- what mlp.py's Qwen36MLP,
#   gdn/gated_deltanet.py's Qwen36GatedDeltaNet, and attention/gated_attention.py's
#   Qwen36GatedAttention each built inline, before this helper existed).
# QWEN36_PREFILL_MM_FP32_ACC (default "0" (off) since 2026-09-22 per user decision -- traced 4k
#   TTFT 0.153 -> 0.146 s, logits PCC vs legacy 0.9997, argmax unchanged): fp32_dest_acc_en for the
#   same ckc. "1" restores fp32 accumulation. Every `_pick_prefill_progcfg` caller reads this back
#   off the ckc itself (`getattr(ckc, "fp32_dest_acc_en", True)`), so program configs stay legal
#   either way with no separate plumbing. legacy = QWEN36_PREFILL_MM_PACKER_L1_ACC=0
#   QWEN36_PREFILL_MM_FP32_ACC=1 QWEN36_PREFILL_MINIMAL_CFG=0.
# QWEN36_PREFILL_MINIMAL_CFG (default "1"): use an explicit MinimalMatmulConfig (see
#   prefill_minimal_matmul_config below) for the fused-SwiGLU (T<4096, mlp.py forward()) and
#   GDN-qkv (ttnn_gated_deltanet.py) minimal_matmul calls. "0" = legacy (config=None at both).
#   The T>=4096 fused-SwiGLU minimal_matmul keeps an explicit MinimalMatmulConfig(8,8,8) either
#   way; this flag only changes whether it also sets a non-default subblock.
# math_fidelity stays LoFi, and math_approx_mode stays True (the WormholeComputeKernelConfig
# default every one of these ckcs relied on implicitly) for every value of these three flags.
# Decode ckcs (packer_l1_acc=True, built separately as `compute_kernel_config_decode`) are never
# touched by this table.
PREFILL_MM_PACKER_L1_ACC = os.environ.get("QWEN36_PREFILL_MM_PACKER_L1_ACC", "1") != "0"
PREFILL_MM_FP32_ACC = os.environ.get("QWEN36_PREFILL_MM_FP32_ACC", "0") != "0"
PREFILL_MINIMAL_CFG = os.environ.get("QWEN36_PREFILL_MINIMAL_CFG", "1") != "0"


# --- I-1 integration flags (2026-09-25; single device; plan_0925 task I-1) ------------------------
# Each item has its own env flag QWEN36_I1_<ITEM>; "0" restores the pre-I-1 code path exactly.
#   D3       decode decoder RMSNorm (2 per layer + final norm): width-sharded 8-core rms_norm
#            (I2S + sharded rms_norm + S2I) instead of the 1-core interleaved norm (+ L1 copy).
#   D15      decode full-attention q|k|v projections: 2 matmuls (qkv_fused + gate_deint) instead of 3.
#   D4A      decode full-attention RoPE: rotary_embedding_hf on [1,1,H,64] with row-replicated
#            [1,32,64] cos/sin, and no head transposes (D4a + D5).
#   D6       decode full-attention head concat: one reshape instead of transpose + concatenate_heads.
#   P5       prefill GDN [g|a|b] projection: a separate tile-padded [g|a|0|b|0] weight, so a/b are
#            tile-aligned slices (no untilize). The decode mega weight layout is unchanged.
#   ROPEWARM compile the RoPE table slice + cos/sin copy of prefill_traced_chunked before the chunked
#            prefill trace capture (no post-capture compile on request 0, issue #48536 class).
# Default ON only for items whose gate passed (greedy tokens identical to the pre-I-1 path at ISL
# 4096 / demo 2642 / ISL 1000, OSL 64), or whose numerics change was accepted against an HF fp32
# reference. D3 and D4A each change a value by ~1 ulp (D3: 8-core reduction order in the norm; D4A:
# rotary_embedding_hf vs the rotate_half chain) and that flips a near-tie greedy token (D3 alone:
# ISL 1000 token 24; D4A alone: ISL 4096 token 6); the continuations stay coherent.
# D3 default ON (2026-09-25, user-approved after the HF fp32 check HF2: teacher-forced mean KL vs HF
# fp32 not worse than D3=0, top-1 identical). D4A stays default OFF (D3+D4A missed the +5% KL line on
# one prompt); set QWEN36_I1_D4A=1 to enable it.
I1_FLAG_DEFAULTS = {"D3": "1", "D15": "1", "D4A": "0", "D6": "1", "P5": "1", "ROPEWARM": "1"}


def i1_enabled(item):
    """True if the I-1 item (a key of I1_FLAG_DEFAULTS) is enabled: env QWEN36_I1_<item> != "0"."""
    return os.environ.get("QWEN36_I1_" + item, I1_FLAG_DEFAULTS[item]) != "0"


# --- I-2 integration flags (2026-09-25; single device; plan_0925 task I-2) ------------------------
# Prefill matmul configs measured best by the standalone T2 sweep (T=2048 chunk, 13x10 grid, device
# kernel medians). Each item has its own env flag QWEN36_I2_<ITEM>; "0" restores the pre-I-2 path
# exactly. Every item applies only with fp32 dest acc OFF (QWEN36_PREFILL_MM_FP32_ACC=0, the default:
# a 1x5 subblock is illegal with fp32 dest) and on the swept 13x10 worker grid; else the pre-I-2 path.
#   S1     fused SwiGLU gate|up minimal_matmul (T < 4096), instead of 4/8/16 2x4:
#          "1" = M_block min(7, ceil(Mt/10)), K_block 8, N_block 10, subblock 1x5 (T2 S1-M085: 435.1 ->
#          374.3 us at T=2048). NOT bit-identical: minimal_matmul snakes the K-block order on every
#          other N block (in0 reuse), so a tile's accumulation order depends on its N-block parity and
#          N_block 10 vs 16 changes it for ~39% of the output (same error vs fp32: op-level PCC 0.999939
#          both, plan_0925/I2 op_diff).
#          "2" = M_block min(7, ceil(Mt/10)), K_block 8, N_block 16, subblock 1x8 (T2 S1-X022: 387.0 us):
#          same N-block partition as the pre-I-2 config -> bit-identical.
#   S2     MLP down-proj (T <= 2048): ttnn.linear 2D mcast 13x10, in0_block_w 16, subblock 1x5,
#          per_core_M ceil(Mt/10), per_core_N 5 (T2 S2-D008: 191.5 -> 103.0 us) instead of
#          minimal_matmul(config=None). Output placement unchanged. "8" = in0_block_w 8 variant.
#   S3     GDN qkv in-proj (T <= 2048): ttnn.linear 2D mcast 13x10, in0_block_w 16, subblock 1x5,
#          per_core_N 15 (T2 S3-D008: 226.2 -> 175.2 us) instead of minimal_matmul 4/8/16 2x4.
#          Input/output placement unchanged (DRAM at T=2048). "8" = in0_block_w 8 variant.
#   PICKER _pick_prefill_progcfg: output-subblock area cap 8 instead of 4 (the fp32-dest-era limit).
#          N=2048/2080 shapes then take 13x10 + 1x5 (130 cores) instead of 12x10 + 1x3 (110 cores);
#          N=2112 (P5 gab) 1x3 -> 1x6 and N=3072 (FA qkv) 1x4 -> 1x8 on the same grid. Same in0_block_w
#          -> bit-identical.
# S2/S3 change the K blocking / accumulation sequence (2D-mcast kernel vs minimal_matmul), so they are
# not bit-identical to the pre-I-2 path.
# Defaults (2026-09-25, plan_0925/I2 gates): S1 = "2" and PICKER = "1" are ON -- both bit-identical to
# the pre-I-2 path (prefill logits at 512/1024 and on the traced 4096/demo/1000 paths, greedy tokens,
# teacher-forced logits), trace guard clean, 0 post-capture compiles, tail 3/3, traced_4k PASS.
# S1 = "1", S2 and S3 (both "1" and "8") stay OFF: each changes numerics and misses the logits gate
# (PCC 0.99965-0.99975 < 0.9998 vs logs/step1/logits_fla_c1b.pt at T=512/1024) and the HF fp32 KL
# "+5% on every prompt" line (plan_0925/I2 report). Set QWEN36_I2_<item> to enable them.
I2_FLAG_DEFAULTS = {"S1": "2", "S2": "0", "S3": "0", "PICKER": "1"}
_I2_SWEPT_GRID = (13, 10)


def i2_value(item):
    """Raw value of the I-2 item flag (a key of I2_FLAG_DEFAULTS): env QWEN36_I2_<item>."""
    return os.environ.get("QWEN36_I2_" + item, I2_FLAG_DEFAULTS[item])


def i2_enabled(item):
    """True if the I-2 item is enabled: env QWEN36_I2_<item> != "0"."""
    return i2_value(item) != "0"


def _i2_applies(item, grid):
    """Common I-2 guard: flag on, fp32 dest acc off, swept 13x10 grid."""
    if not i2_enabled(item) or PREFILL_MM_FP32_ACC:
        return False
    return (int(grid.x), int(grid.y)) == _I2_SWEPT_GRID


def i2_swiglu_minimal_config(M, grid):
    """I-2 S1: MinimalMatmulConfig for the fused-SwiGLU minimal_matmul (T < 4096), or None (flag off
    / fp32 dest on / grid not 13x10 / QWEN36_PREFILL_MINIMAL_CFG=0) to keep the pre-I-2 config.
    QWEN36_I2_S1=1: N_block 10, subblock 1x5 (T2 S1-M085); =2: N_block 16, subblock 1x8 (T2 S1-X022,
    bit-identical to the pre-I-2 config). M_block = min(7, per-core M tiles), K_block 8 for both."""
    if not PREFILL_MINIMAL_CFG or not _i2_applies("S1", grid):
        return None
    n_block, sub_w = (16, 8) if i2_value("S1") == "2" else (10, 5)
    m_block = max(1, min(7, math.ceil(math.ceil(M / TILE_SIZE) / int(grid.y))))
    return ttnn.MinimalMatmulConfig(
        M_block_size=m_block,
        K_block_size=8,
        N_block_size=n_block,
        subblock_h=1,
        subblock_w=sub_w,
        compute_with_storage_grid_size=grid,
    )


def i2_prefill_2d_progcfg(item, M, K, N, grid, max_m=2048):
    """I-2 S2 / S3: 2D-mcast MatmulMultiCoreReuseMultiCastProgramConfig on the full 13x10 grid
    (T2 D008 family), or None to keep the pre-I-2 minimal_matmul call.

    in0_block_w 16 (flag "1") or the flag's integer value (e.g. "8"); per_core_M = ceil(Mt/10),
    per_core_N = ceil(Nt/13), out subblock 1 x (largest divisor of per_core_N <= 8). Applies only for
    32 <= M <= max_m (the swept chunk size; larger M needs CBs the sweep never validated) and when
    in0_block_w divides Kt."""
    if not _i2_applies(item, grid) or not (TILE_SIZE <= M <= max_m):
        return None
    val = i2_value(item)
    bw = 16 if val == "1" else int(val)
    Mt, Kt, Nt = math.ceil(M / TILE_SIZE), math.ceil(K / TILE_SIZE), math.ceil(N / TILE_SIZE)
    if bw <= 0 or Kt % bw:
        return None
    gx, gy = int(grid.x), int(grid.y)
    per_core_M = math.ceil(Mt / gy)
    per_core_N = math.ceil(Nt / gx)
    sw = _get_out_subblock_w(per_core_N, 1, max_area=8)
    return _mk_2d_progcfg(gx, gy, bw, per_core_M, per_core_N, sw)


# --- M1 prefill-matmul flags (2026-09-26; single device, T == 2048 chunk path only) ----------------
# Source: qwen35_2b_handoff/analysis_50ms/H (H_results.md, h_bench.py variant V2_bw8). Each item has its
# own env flag QWEN36_M1_<ITEM>; "0" (default) keeps the current code path exactly. All items use the
# same call: ttnn.matmul(x, w, program_config=P, compute_kernel_config=<the call's prefill ckc: LoFi,
# approx, fp32 dest off, packer L1 acc on>, memory_config=<output>, dtype=bf16) with
#   P = MatmulMultiCoreReuseMultiCastProgramConfig(grid (13,10), in0_block_w 8, out_subblock_h 1,
#       out_subblock_w W, per_core_M 7, per_core_N N, transpose_mcast False, fused_activation None,
#       fuse_batch True).
#   S2  MLP down projection (K 6144, N 2048; mlp.py): N 5, W 5. in0 (SwiGLU output) and output keep
#       their current placement (L1 / L1 with QWEN36_MLP_L1_OUT=1). H: 194.0 -> 112.0 us.
#   S3  GDN q|k|v in-proj (K 2048, N 6144; ttnn_gated_deltanet.py): N 15, W 5. in0 = attention_norm
#       output, written to L1 interleaved (layer.py); output L1 interleaved (read by the tiled KDA conv,
#       freed right after it). H: 228.4 -> 119.5 us.
#   S4  GDN z|a|0|b|0 in-proj (K 2048, N 2112 = I1_P5 padded; ttnn_gated_deltanet.py): N 6, W 6. in0 =
#       the same L1 norm output; output L1 interleaved; the gate/a/b slices keep their DRAM placement.
#       H: 79.7 -> 50.8 us.
# With S3 or S4 on, the GDN attention_norm output is L1 (layer.py) and is freed right after its last
# consumer (the z|a|0|b|0 matmul) instead of after the whole GDN layer. Every item applies only with
# fp32 dest acc off (subblock area > 4), on the 13x10 grid, at M == 2048 with the exact (K, N) above;
# otherwise the current call runs. S2/S3 change the K accumulation order vs minimal_matmul (not
# bit-identical); S4 uses the picker's program family (in0_block_w 8, 1x6) with fuse_batch True.
# When both are set, QWEN36_M1_S2 / _S3 take precedence over QWEN36_I2_S2 / _S3.
M1_FLAG_DEFAULTS = {"S2": "0", "S3": "0", "S4": "0"}
_M1_T = 2048
_M1_SHAPES = {"S2": (6144, 2048, 5, 5), "S3": (2048, 6144, 15, 5), "S4": (2048, 2112, 6, 6)}  # K, N, pcN, W
_M1_LOGGED = set()


def m1_value(item):
    """Raw value of the M1 item flag (a key of M1_FLAG_DEFAULTS): env QWEN36_M1_<item>."""
    return os.environ.get("QWEN36_M1_" + item, M1_FLAG_DEFAULTS[item])


def m1_enabled(item):
    """True if the M1 item is enabled: env QWEN36_M1_<item> != "0"."""
    return m1_value(item) != "0"


def _m1_applies(item, M, grid):
    """Common M1 guard: flag on, fp32 dest acc off, 13x10 grid, M == 2048."""
    if not m1_enabled(item) or PREFILL_MM_FP32_ACC or grid is None:
        return False
    return (int(grid.x), int(grid.y)) == _I2_SWEPT_GRID and M == _M1_T


def m1_prefill_2d_progcfg(item, M, K, N, grid):
    """M1 item S2 / S3 / S4: the H V2_bw8 2D-mcast program config (see the table above), or None to
    keep the current call (flag off, fp32 dest on, grid not 13x10, M != 2048, or another (K, N)).

    MM BW16 (plan_0928 P3_MMSWEEP / P5_INT1B): with QWEN36_MM_BW16=1, S2 and S4 get in0_block_w 16
    instead of 8 (measured faster, PCC ~0.99995, not bit-exact). S3 (GDN q|k|v in-proj) is excluded:
    bw16 there overflows a kernel-config limit (TT_THROW), confirmed by P3_MMSWEEP."""
    if not _m1_applies(item, M, grid):
        return None
    k, n, per_core_N, sub_w = _M1_SHAPES[item]
    if (int(K), int(N)) != (k, n):
        return None
    bw = 16 if (item != "S3" and mm_enabled("BW16")) else 8
    if item not in _M1_LOGGED:
        _M1_LOGGED.add(item)
        print(
            f"[M1] QWEN36_M1_{item}=1 active: M={M} K={K} N={N} 2D mcast 13x10 bw{bw} pcM7 pcN{per_core_N} "
            f"sb1x{sub_w} fuse_batch=True",
            flush=True,
        )
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(13, 10),
        in0_block_w=bw,
        out_subblock_h=1,
        out_subblock_w=sub_w,
        per_core_M=7,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=True,
    )


def m1_gdn_norm_l1(T, grid):
    """True if the GDN attention_norm output goes to L1 interleaved for M1 S3 / S4 (layer.py): either
    item passes the common guard at this T (the (K, N) check happens at the matmul)."""
    return _m1_applies("S3", T, grid) or _m1_applies("S4", T, grid)


# --- M2 traced-prefill flags (2026-09-26; single device, traced chunked prefill only) ----------------
# Source: qwen35_2b_handoff/analysis_50ms (D_prefill_op_map.md, F_last_layer_skip.md). Each item has its
# own env flag QWEN36_M2_<ITEM>; "0" (default) keeps the current code path exactly. model.py reads them
# once per prepare (Qwen36Model._prepare_prefill_trace_chunked_setup), so the warm-up forward, the
# captured trace and the request-time code always agree.
#   NOWHERE      The traced chunk forward skips the vision-splice ttnn.where (and prefill_traced_chunked
#                skips the per-chunk zero-mask upload) when the model is text-only at prepare/capture
#                (Qwen36Model.vision_model is None). A multimodal request against such a trace raises.
#                Bit-exact (where(0, vis, x) == x).
#   REPACK_LATE  prefill_traced_chunked (exact-multiple prompts) no longer runs the eager GDN conv-history
#                repack before the LM head; it marks it pending and returns after the first-token host
#                read. The next model entry that uses GDN state (Generator.decode_forward -> switch_mode,
#                prepare_decode_inputs_host, decode, GDN reset / save / restore) runs it first, so it
#                completes on device before the first decode step. Bit-exact.
#   LASTROW      F design Stage 1: in the traced chunk, the last layer (full attention) keeps the K/V
#                cache fill and the full SDPA, then computes the rest (gate, head concat, gate multiply,
#                o_proj, residual, ffn_norm, MLP, residual) for row chunk_size - 1 only (1-row matmuls
#                use the decode program configs or None, never the M = chunk_size prefill config). The
#                trace output becomes [1, 1, dim]. Only the exact-multiple return path reads it (always
#                row chunk_size - 1). Numerics change (1-row matmul configs + decode MLP).
M2_FLAG_DEFAULTS = {"NOWHERE": "0", "REPACK_LATE": "0", "LASTROW": "0"}


def m2_value(item):
    """Raw value of the M2 item flag (a key of M2_FLAG_DEFAULTS): env QWEN36_M2_<item>."""
    return os.environ.get("QWEN36_M2_" + item, M2_FLAG_DEFAULTS[item])


def m2_enabled(item):
    """True if the M2 item is enabled: env QWEN36_M2_<item> != "0"."""
    return m2_value(item) != "0"


# --- M3 flags (2026-09-26; single device) ----------------------------------------------------------
# Source: qwen35_2b_handoff/analysis_50ms (N1_results.md variant N-f; M2/NOTES.txt REPACK_LATE). Each
# item has its own env flag QWEN36_M3_<ITEM>; "0" (default) keeps the current code path exactly.
#   ZB            With QWEN36_M1_S2 and/or QWEN36_M1_S3 on, the M1 2D-mcast call becomes
#                 ttnn.linear(x, w, bias=<zero bias>, program_config=<the same M1 config>, ...) instead
#                 of ttnn.matmul (N1 N-f). The zero bias is bf16 TILE [1, N] DRAM interleaved (N 2048 for
#                 S2, 6144 for S3). Qwen36Model allocates each one ONCE at model load (m3_zero_bias),
#                 before any trace capture, and shares it across layers; the forward never allocates it.
#                 With FUSE_BIAS the 2D kernel skips the bf16 DEST reload on the last K block, so the
#                 result equals minimal_matmul bit for bit (N1: S2 / S3 on real activations).
#   REPACK_TRACE  The GDN conv-history repack (QWEN36_GDN_DECODE_FUSED=2; ~8 ops per GDN layer) that
#                 prefill_traced_chunked runs eagerly after the last chunk replay of an exact-multiple
#                 prompt is captured once into its own trace (capture_prefill_trace_chunked with
#                 prepared=True, right after the chunk-trace capture, after an eager warm-up) and replayed
#                 at the same point instead. It reads only the persistent GDN conv-state buffers and
#                 writes the persistent conv_hist buffers (no host inputs). QWEN36_M2_REPACK_LATE=1
#                 takes precedence (the repack is then deferred and runs eagerly).
M3_FLAG_DEFAULTS = {"ZB": "0", "REPACK_TRACE": "0"}


def m3_value(item):
    """Raw value of the M3 item flag (a key of M3_FLAG_DEFAULTS): env QWEN36_M3_<item>."""
    return os.environ.get("QWEN36_M3_" + item, M3_FLAG_DEFAULTS[item])


def m3_enabled(item):
    """True if the M3 item is enabled: env QWEN36_M3_<item> != "0"."""
    return m3_value(item) != "0"


def m3_zero_bias(item, device):
    """M3 ZB: allocate the zero bias for M1 item S2 / S3: bf16 TILE [1, N] DRAM interleaved, N = the M1
    table's N (N1 N-f). Call once at model load (outside any trace capture)."""
    n = _M1_SHAPES[item][1]
    return ttnn.from_torch(
        torch.zeros(1, n, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


# --- C2 flags (2026-09-26; single device, unmasked T == 2048 GDN chunk prefill only) -----------------
# Source: qwen35_2b_handoff/analysis_50ms/G1 (extended op ttnn.experimental.kda.sigmoid_gated_rms_norm with
# gate_activation / gate_col_offset_tiles). Each item has its own env flag QWEN36_C2_<ITEM>; "0" (default)
# keeps the current code path exactly.
#   SGRN  After ChunkGdnFused, the 5 post-scan glue ops (z-gate slice of the z|a|0|b|0 in-proj output,
#         typecast o fp32 -> bf16, per-head rms_norm, nlp_concat_heads, multiply with SILU gate) become ONE
#         sigmoid_gated_rms_norm(o [B*Nv,T,Dv] fp32, gab [1,T,2112] bf16 (z = columns 0..2047), o_norm weight
#         [Dv] bf16, Nv, epsilon=norm_eps, gate_activation="silu", gate_col_offset_tiles=0, output bf16 in the
#         memory the gate multiply wrote (mc_scan)); compute config C2_SGRN_CKC (HiFi4, approx off, fp32 dest
#         on, packer L1 acc off). Applies only where the M1 S4 in-proj applies (QWEN36_M1_S4=1): gab
#         stays alive through the conv and ChunkGdnFused and is freed right after the new op; the a/b slices
#         are unchanged. Decode and masked (valid_len) chunks are unchanged. Numerics change (o is no longer
#         rounded to bf16 before the norm).
#         gab placement (C4, c2_sgrn_gab_dram): "variant a" (QWEN36_LAYER_RESID_L1 != "1"): gab L1, as M1 S4
#         writes it. "variant b" (QWEN36_LAYER_RESID_L1=1, the runner default since C3): the same M1 S4 program
#         writes gab DRAM interleaved; the a/b slices and sigmoid_gated_rms_norm read it from DRAM (with the
#         residual stream also in L1, an L1 gab alive through ChunkGdnFused clashes with its static CBs).
#         Placement only: the same numerics in both variants.
C2_FLAG_DEFAULTS = {"SGRN": "0"}
# sigmoid_gated_rms_norm rejects packer_l1_acc=True (the compute kernel does not accumulate through L1).
C2_SGRN_CKC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)


def c2_value(item):
    """Raw value of the C2 item flag (a key of C2_FLAG_DEFAULTS): env QWEN36_C2_<item>."""
    return os.environ.get("QWEN36_C2_" + item, C2_FLAG_DEFAULTS[item])


def c2_enabled(item):
    """True if the C2 item is enabled: env QWEN36_C2_<item> != "0"."""
    return c2_value(item) != "0"


def c2_sgrn_gab_dram():
    """C2 SGRN gab placement (C4): True = "variant b" (gab DRAM interleaved) when the residual stream is L1
    (env QWEN36_LAYER_RESID_L1 == "1", as layer.py; SGRN runs only at T == 2048 <= its T limit); False =
    "variant a" (gab L1). R3 SGRN_GAB_L1=1 (R3 table below) forces variant a."""
    return os.environ.get("QWEN36_LAYER_RESID_L1", "0") == "1" and not r3_enabled("SGRN_GAB_L1")


def sgrn_kernel_variant():
    """P6_INT1C item 2 (fused gated RMSNorm, sgrn_fast_vs_r3.patch): the sigmoid_gated_rms_norm op's
    kernel_variant (0 = legacy 7-pass kernel; 1-3 = fused kernel variants, bit-exact with each other;
    4 = fused kernel with an exp_21f sigmoid, within 1 bf16 ulp of 0 for >99.9% of values, not
    bit-exact). Env QWEN36_SGRN_VARIANT: when set, returns int(value); unset -> None (the op's own
    default, kernel_variant=4). Runner default since P6_INT1C: 4."""
    v = os.environ.get("QWEN36_SGRN_VARIANT")
    return int(v) if v is not None else None


# --- R3 L1 placement flags (2026-09-26; analysis_50ms/R3_fla_cb_spec.md sec 4 items 5-7) --------------
# They need the Ct == 1 ChunkGdnFused CB shrink (R3 op patch: producer CB region ends at 480,256 B instead of
# 1,217,536 B); without it these L1 tensors clash with the fused FLA op's static CBs. Each flag applies only to
# unmasked T == R3_T GDN prefill chunks on the fused FLA path; masked / other buckets keep DRAM. Placement
# only (bit-exact). Env QWEN36_R3_<item>, values 0|1, default 0 = the current path.
#   FLA_IN_L1   : the tiled KDA conv writes q/k/v to L1 and the beta/g chain stays L1 (mc_small), i.e. the
#                 QWEN36_GDN_FLA_INPUTS_DRAM behavior is switched off for these chunks.
#   SGRN_GAB_L1 : with QWEN36_C2_SGRN=1 and QWEN36_LAYER_RESID_L1=1, gab stays L1 (C2 variant a) instead of the
#                 C4 variant b (DRAM): c2_sgrn_gab_dram() returns False.
#   O_L1        : the FLA op writes o and the final state to L1 (memory_config=L1_MEMORY_CONFIG).
R3_FLAG_DEFAULTS = {"FLA_IN_L1": "0", "SGRN_GAB_L1": "0", "O_L1": "0"}
R3_T = 2048


def r3_value(item):
    """Raw value of the R3 item flag (a key of R3_FLAG_DEFAULTS): env QWEN36_R3_<item>."""
    return os.environ.get("QWEN36_R3_" + item, R3_FLAG_DEFAULTS[item])


def r3_enabled(item):
    """True if the R3 item is enabled: env QWEN36_R3_<item> != "0"."""
    return r3_value(item) != "0"


# --- M4 flags (2026-09-26; single device, traced chunked prefill with QWEN36_M2_LASTROW on) ----------
# Source: qwen35_2b_handoff/analysis_50ms/R_small_items_spec.md (R4 = LASTROW stage 2, the last layer).
# Each item has its own env flag QWEN36_M4_<ITEM>; "0" (default) keeps the current code path exactly.
# model.py fixes both once per prepare (Qwen36Model._prepare_prefill_trace_chunked_setup), and only
# when M2 LASTROW is active, so the warm-up forward, the captured trace and the request path agree.
#   R4A  The 3 one-row reads of the last layer ([T-1:T] slices of TILE tensors: the gate input, the SDPA
#        output and the residual row) each become a tile-aligned [T-32:T] slice (TILE path, no untilize of
#        the whole tensor) and then row 31 of that 32-row block (the small untilize + slice + tilize path).
#        The SDPA output block goes through concatenate_heads ([1, H, 32, D] -> [1, 32, H*D]) before its
#        row cut. Copies only: bit-exact.
#   R4B  The last layer's chunked prefill SDPA (all T query rows) becomes paged_scaled_dot_product_attention_
#        decode for query row T - 1 only (the decode kernel and program config). Q: rows [T-32:T] of the
#        heads, q-norm and RoPE (rows [T-32:T] of the chunk cos/sin) on that block, then row 31. K/V: the
#        full chunk as today (k-norm, RoPE, paged fills before the SDPA). The absolute position of row
#        T - 1 comes from a persistent int32 [1] device buffer (Qwen36Model._chunk_last_pos_tensor) that
#        prefill_traced_chunked writes (cs + chunk_size - 1) before every chunk replay; the host asserts
#        the value it wrote for each chunk. Numerics change (decode SDPA kernel). Independent of R4A (the
#        SDPA-output part of R4A does not apply when R4B is on).
M4_FLAG_DEFAULTS = {"R4A": "0", "R4B": "0"}


def m4_value(item):
    """Raw value of the M4 item flag (a key of M4_FLAG_DEFAULTS): env QWEN36_M4_<item>."""
    return os.environ.get("QWEN36_M4_" + item, M4_FLAG_DEFAULTS[item])


def m4_enabled(item):
    """True if the M4 item is enabled: env QWEN36_M4_<item> != "0"."""
    return m4_value(item) != "0"


# --- M5 flags (2026-09-26; single device, traced chunked prefill) -----------------------------------------
# Each item has its own env flag QWEN36_M5_<ITEM>; "0" (default) keeps the current code path exactly. model.py
# fixes both once per prepare (Qwen36Model._prepare_prefill_trace_chunked_setup), so the warm-up, the captured
# traces and the request path agree.
#   TAIL_TRACE  Prepared order only (prepare_prefill_trace_chunked -> prime decode -> capture(prepared=True)) and
#               a prompt that is an exact multiple of the chunk size: the eager tail of prefill_traced_chunked
#               (last-row slice when M2 LASTROW is off, to_layout / to_memory_config, final norm, the LM head of
#               the active mode (QWEN36_LMHEAD_SPLIT / QWEN36_I3_LMHEAD) and, with set_greedy_token_output(True),
#               the argmax) is captured into its own small trace, right after the chunk trace and the M3 repack
#               trace (after an eager warm-up; 0 new program-cache entries asserted), and replayed right after the
#               repack replay. The tail's last op copies its result (uint32 token, or the logits) into a
#               persistent DRAM buffer allocated in prepare, before the decode trace is primed; the request reads
#               that buffer. The same code (Qwen36Model._exact_multiple_tail_device) runs in the prepare warm-up,
#               the capture warm-up, the capture and (flag off) the eager tail: bit-exact.
#   ADDNORM     Unmasked T == M5_ADDNORM_T chunk of the traced chunk forward (_forward_prefill_chunk): each
#               residual add + the RMSNorm that reads its sum become one ttnn.rms_norm(a,
#               residual_input_tensor=b, residual_output_tensor=h) (R6 stage-1 op: writes h = a + b and
#               n = rmsnorm(h) * gamma). h is allocated by the caller with the memory config the residual add
#               writes today (L1 with QWEN36_LAYER_RESID_L1=1); n keeps the norm's placement (GDN attention_norm
#               L1 on the M1 path, else the input's). Intra-layer pair: attention residual add -> ffn_norm.
#               Cross-layer pair: MLP residual add of layer i -> attention_norm of layer i + 1 (layer i returns
#               (h, mlp_out) and layer i + 1 makes x with the fused op). One-row calls (M2 LASTROW: the last
#               layer's attention residual add, ffn_norm and MLP residual add) stay unfused. Bit-exact by the R6
#               op tests (fused h == ttnn.add, fused n == ttnn.rms_norm(ttnn.add)).
M5_FLAG_DEFAULTS = {"TAIL_TRACE": "0", "ADDNORM": "0"}
M5_ADDNORM_T = 2048


def m5_value(item):
    """Raw value of the M5 item flag (a key of M5_FLAG_DEFAULTS): env QWEN36_M5_<item>."""
    return os.environ.get("QWEN36_M5_" + item, M5_FLAG_DEFAULTS[item])


def m5_enabled(item):
    """True if the M5 item is enabled: env QWEN36_M5_<item> != "0"."""
    return m5_value(item) != "0"


# --- F flags (2026-09-26; single device; accuracy items, analysis_50ms/FINAL) -----------------------------------
# Each item has its own env flag QWEN36_F_<ITEM>; "0" (default) keeps the current code path exactly.
#   MLP_GU_BF8  One device only (tt/mlp.py load_mlp_weights): the MLP gate/up weights are bfloat8_b instead of
#               bfloat4_b -- the decode w1 / w3 and the prefill packed [gate|up] w_gate_up of the fused-SwiGLU
#               minimal_matmul (down stays bfloat8_b). Numerics change (D4: the bfp4 gate/up weights are a
#               secondary contributor to the long-context KL). The matmul configs stay the same (the fused-SwiGLU
#               minimal_matmul 7/8/16 in1 CB grows from 147,456 to 278,528 B per core); the weight-cache files are
#               new ones (ttnn.as_tensor puts the dtype in the file name). TP (tp > 1) keeps bfloat4_b.
F_FLAG_DEFAULTS = {"MLP_GU_BF8": "0"}


def f_value(item):
    """Raw value of the F item flag (a key of F_FLAG_DEFAULTS): env QWEN36_F_<item>."""
    return os.environ.get("QWEN36_F_" + item, F_FLAG_DEFAULTS[item])


def f_enabled(item):
    """True if the F item is enabled: env QWEN36_F_<item> != "0"."""
    return f_value(item) != "0"


def mlp_gate_up_dtype():
    """Single-device MLP gate/up weight dtype: bfloat8_b with QWEN36_F_MLP_GU_BF8=1, else bfloat4_b (default)."""
    return ttnn.bfloat8_b if f_enabled("MLP_GU_BF8") else ttnn.bfloat4_b


# --- N flags (2026-09-28; norm weight placement) -----------------------------------------------
# Each item has its own env flag QWEN36_N_<ITEM>; "0" (default) keeps the current code path exactly.
#   GAMMA_L1  layer.py's _make_norm (attention_norm / ffn_norm, the two norms _m5_add_norm reads):
#             the RMSNorm gamma (ROW_MAJOR [1,1,dim/32,32] bf16) in L1 interleaved instead of DRAM.
#             Placement only (bit-exact); standalone P2_NORM measurement: -1.3 us/call of 92 calls.
#             The tensor is created at layer load (RMSNorm.__init__), before trace capture. The final
#             norm and the FA q/k norms are built elsewhere and are not affected by this flag.
N_FLAG_DEFAULTS = {"GAMMA_L1": "0"}


def n_value(item):
    """Raw value of the N item flag (a key of N_FLAG_DEFAULTS): env QWEN36_N_<item>."""
    return os.environ.get("QWEN36_N_" + item, N_FLAG_DEFAULTS[item])


def n_enabled(item):
    """True if the N item is enabled: env QWEN36_N_<item> != "0"."""
    return n_value(item) != "0"


# --- R5 flags (2026-09-28; single device prefill; plan_0928 P3_GLU / P4_GLU_INT) ------------------------
# Each item has its own env flag QWEN36_R5_<ITEM>; "0" (default) keeps the current code path exactly.
#   GLU  The fused-SwiGLU gate/up matmul of a T == 2048 prefill chunk (tt/mlp.py forward) runs as the 2D-mcast
#        ttnn.matmul with the fused SwiGLU epilogue -- MatmulMultiCoreReuseMultiCastProgramConfig(fuse_swiglu=True),
#        13x10, in0_block_w 4, per_core_M 7, per_core_N 30, subblock 1x6 -- instead of minimal_matmul(fuse_swiglu=True).
#        Same tile-pair interleaved [gate|up] weight (no new weight cache) and the same output placement. Applies
#        only to that swept shape: x [1, 2048, 2048] bf16 interleaved, weight [2048, 12288] bfloat8_b
#        (QWEN36_F_MLP_GU_BF8=1), 13x10 grid, fp32 dest accumulation off; every other call keeps minimal_matmul.
#        Needs the C++ fuse_swiglu program-config field (plan_0928/P3_GLU/glu.patch). Numerics change (K block 4
#        vs 8: not bit-exact vs minimal_matmul; P3_GLU unit test PCC 0.99977 vs fp32, same as minimal_matmul).
R5_FLAG_DEFAULTS = {"GLU": "0"}
R5_GLU_T = 2048  # the swept chunk size (M = 64 tiles over 10 core rows at per_core_M 7)


def act_bf8_resid():
    """QWEN36_ACT_BF8_RESID=1 (default 0): prefill (T > 1) o-proj / down-proj outputs (G3, F3, M2) in bfloat8_b."""
    return os.environ.get("QWEN36_ACT_BF8_RESID", "0") == "1"


def act_bf8_norm():
    """QWEN36_ACT_BF8_NORM=1 (default 0): prefill (T == M5_ADDNORM_T) norm outputs n (fused add+norm, layer-0 norm) in
    bfloat8_b; the residual h stays bf16 and every matmul reading n sets its output dtype explicitly."""
    return os.environ.get("QWEN36_ACT_BF8_NORM", "0") == "1"


def r5_value(item):
    """Raw value of the R5 item flag (a key of R5_FLAG_DEFAULTS): env QWEN36_R5_<item>."""
    return os.environ.get("QWEN36_R5_" + item, R5_FLAG_DEFAULTS[item])


def r5_enabled(item):
    """True if the R5 item is enabled: env QWEN36_R5_<item> != "0"."""
    return r5_value(item) != "0"


def r5_glu_progcfg(x, w_gate_up, grid, compute_kernel_config):
    """R5 GLU: the fused-SwiGLU 2D-mcast program config for this gate/up call, or None (flag off / shape, dtype,
    grid or compute config outside the swept case) to keep the minimal_matmul fused-SwiGLU path."""
    glu = r5_value("GLU")
    if glu not in ("1", "2") or grid is None or (int(grid.x), int(grid.y)) != (13, 10):
        return None
    xs, ws = list(x.shape), list(w_gate_up.shape)
    if len(xs) < 2 or xs[-2:] != [R5_GLU_T, 2048] or any(d != 1 for d in xs[:-2]):
        return None
    if ws[-2:] != [2048, 12288] or any(d != 1 for d in ws[:-2]) or w_gate_up.dtype != ttnn.bfloat8_b:
        return None
    if x.dtype != (ttnn.bfloat8_b if act_bf8_norm() else ttnn.bfloat16) and x.dtype != ttnn.bfloat16:
        return None
    if x.memory_config().is_sharded():
        return None
    if getattr(compute_kernel_config, "fp32_dest_acc_en", True):
        return None
    if glu == "2":
        # GLU=2 (plan_0928 P9_GLU2/P10_INT1H): in0_block_w 16, out block 7x6, SwiGLU applied in the last K block
        # on DEST with the SFPU on the PACK thread (needs the C++ glu_last_block / glu_sfpu_on_pack fields).
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(13, 10),
            in0_block_w=16,
            out_subblock_h=1,
            out_subblock_w=6,
            out_block_h=7,
            out_block_w=6,
            per_core_M=7,
            per_core_N=30,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
            fuse_swiglu=True,
            glu_last_block=True,
            glu_sfpu_on_pack=True,
        )
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(13, 10),
        in0_block_w=4,
        out_subblock_h=1,
        out_subblock_w=6,
        per_core_M=7,
        per_core_N=30,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=True,
        fuse_swiglu=True,
    )


# --- MM flags (2026-09-28; single device prefill; plan_0928 P3_MMSWEEP / P5_INT1B) ---------------------
# Each item has its own env flag QWEN36_MM_<ITEM>; "0" (default) keeps the current code path exactly.
#   BW16  For exactly four T == 2048 prefill 2D-mcast matmul shape families (P3_MMSWEEP), in0_block_w
#         becomes 16 instead of 8 (per_core_M/N and the output subblock unchanged): M1 S2 (MLP down,
#         K6144 N2048) and M1 S4 (GDN z|a|0|b|0 in-proj, K2048 N2112) via m1_prefill_2d_progcfg; the
#         o-proj family (FA o_proj, GDN o_proj; K2048 N2048) and the FA q|k|v fused proj (K2048 N3072)
#         via _pick_prefill_progcfg's _MM_BW16_OVERRIDES table. Measured (P3_MMSWEEP): MLP down
#         113.0->108.7 us, GDN z|a|b 49.5->47.5, o-proj 45.2->43.1, FA qkv 64.2->61.9 (PCC ~0.99995 in
#         each case: fewer bf16 L1-acc roundings in the K loop, not bit-exact). Deliberately excludes
#         M1 S3 (GDN q|k|v in-proj, K2048 N6144): bw16 there overflows a kernel-config limit (TT_THROW
#         in program.cpp), confirmed by P3_MMSWEEP.
MM_FLAG_DEFAULTS = {"BW16": "0"}


def mm_value(item):
    """Raw value of the MM item flag (a key of MM_FLAG_DEFAULTS): env QWEN36_MM_<item>."""
    return os.environ.get("QWEN36_MM_" + item, MM_FLAG_DEFAULTS[item])


def mm_enabled(item):
    """True if the MM item is enabled: env QWEN36_MM_<item> != "0"."""
    return mm_value(item) != "0"


# --- I-3 integration flags (2026-09-25; single device, single-user decode; plan_0925 task I-3) ------
# Decode matmul configs and LM-head variants measured best by the T4 decode sweep (plan_0925/T2
# analyze_decode.py) and re-checked by plan_0925/I3/op_probe.py (bit-exactness vs the current call on
# random data + device-kernel time). Each item has its own env flag QWEN36_I3_<ITEM>; "0" restores
# the pre-I-3 path exactly. Every item applies only on one device with the swept 13x10 worker grid,
# to an input with one tile row (M <= 32, i.e. the B = 1 decode token / the prefill last-token row).
#   FA_PROGCFG   full-attention decode projections: explicit 1D mcast_in0 program configs (table
#                _I3_FA_DECODE_PROGCFGS) instead of program_config=None (ttnn auto: 1D, in0_block_w 2,
#                per_core_N 1). Same compute config (LoFi, fp32 dest, packer L1 acc). Op probe: q|k|v
#                (D15) 46.9 -> 19.1 us, gate 37.3 -> 12.8, o_proj 37.8 -> 12.8, q_proj q|gate 56.7 -> 24.0,
#                k/v 26.5 -> 6.6; outputs bit-identical to the auto config.
#   MLP_PROGCFG  MLP decode gate (silu) / up projections: 1D 13x4 in0_block_w 4 per_core_N 4 subblock
#                1x4 instead of _pick_decode_progcfg's 13x3 bw 8 pcN 5 1x1 (op probe 31.1/27.3 ->
#                28.0/24.2 us; gate+up+mul chain 61.7 -> 56.0 us; bit-identical).
#   MLP_FUSED_GU MLP decode gate|up as ONE DRAM-sharded matmul on a [gate | up] weight stored DRAM
#                width-sharded (T2 D6f-R028: in0 L1 width-sharded 8x8, wpb 2, bw 4, pcN 6), then
#                S2I + 2 slices + multiply(silu(gate), up). Op probe chain 61.7 -> 49.8 us. NOT
#                bit-identical: silu runs on the bf16 gate output instead of the fp32 dest (840 of
#                6144 values differ by <= 1 bf16 ulp in the op probe). Overrides MLP_PROGCFG for gate/up.
#                +25 MB DRAM per layer (built on device from w1/w3 at load).
#   LMHEAD       LM head (decode token and the prefill last-token row), in place of the
#                QWEN36_LMHEAD_SPLIT=8 minimal_matmul path (QWEN36_LMHEAD_SPLIT is ignored when set):
#                "A" = 4 column splits, each a DRAM-sharded ttnn.linear (weights DRAM width-sharded,
#                      244 tiles/bank; in0 L1 width-sharded 8x8; wpb 2, bw 1, pcN 31; LoFi, fp32 dest
#                      off) -> S2I; op probe 2468 -> 1282 us (logits), 2542 -> 1258 us (greedy token).
#                      NOT bit-identical (LoFi; 92% of logits differ, max 0.125).
#                "A2"= A with the current path's compute config (HiFi2, fp32 dest; minimal_matmul's
#                      default) and 8 splits (122 tiles/bank, pcN 16: at 4 splits its fp32 CBs clashed with
#                      live L1 buffers on the eager prefill_paged path). Op probe (4 splits) 1654 us logits /
#                      1629 us token. NOT bit-identical (7% of logits differ by <= 1 bf16 ulp in the op probe).
#                "B" = unsplit ttnn.linear 1D mcast 13x10 bw 2 pcN 65 subblock 1x5 (LoFi, fp32 dest off);
#                      op probe 1447 us (logits); token = untilize to 1 row + argmax. NOT bit-identical.
#                      (bw 8 = T4 D8-O062, 1503 us, needs 1.41 MB of CBs: clashed with live L1 buffers on
#                      the eager prefill_paged path; fp32 dest does not fit either.)
#                "C" = 4 column splits, minimal_matmul on a 13x2 grid, M/K/N block 1/16/16, subblock 1x4,
#                      op-default compute config (HiFi2, fp32 dest; the T4 LoFi variant was not faster);
#                      op probe 2044 us (logits), 2088 us (token), bit-identical there; in the model NOT
#                      strictly bit-identical: K block 16 vs 8 changes the fp32 summation grouping, and 1
#                      logit of 16 x 3 teacher-forced decode steps differed by 1 bf16 ulp (greedy tokens and
#                      HF-KL identical).
#                "A3"= (M4; analysis_50ms/R9 winner A3_s10_bw2_i64_gd) A2's numerics at A's speed: 10 column
#                      splits (776 tiles each), each DRAM width-sharded over the 8 banks (98 tiles/bank, the
#                      last bank 90 valid), built ONCE at model load from the loaded weight (device slice +
#                      reshard); the unsplit weight is then freed (no second full copy). in0 L1 width-sharded
#                      8x8 (1 tile per core): one I2S for an interleaved input, one reshard for the D3
#                      8-core decode norm output (the traced decode skips that norm's S2I). DRAM-sharded
#                      linear in0_block_w 2, per_core_M 1, per_core_N 13, 2 workers per bank; HiFi2 / approx
#                      off / fp32 dest / packer L1 acc (= A2 = the minimal_matmul default). Token: each split
#                      output is untilized straight from the width-sharded output to L1 interleaved (one
#                      row) right after its matmul, then RM concat + argmax. Logits: S2I (DRAM) per split
#                      + TILE concat. R9 microbench (token): 1.26 ms vs 2.55 ms. NOT bit-identical (A2
#                      class: ~8% of logits differ by <= 1 bf16 ulp in R9; argmax 16/16 there).
#                +540 MB DRAM for the A/C column chunks (built on device on the first call, eager).
#                Greedy token output (set_greedy_token_output) works with every value.
# Defaults (2026-09-25, plan_0925/I3 gates): FA_PROGCFG and MLP_PROGCFG are ON -- both bit-identical to
# the pre-I-3 path (greedy tokens ISL 4096 / demo / ISL 1000 x 64, teacher-forced prefill + 16 decode-step
# logits, traced/eager prefill logits), trace guard clean, 0 post-capture compiles, tail 3/3, traced_4k PASS.
# MLP_FUSED_GU and LMHEAD stay OFF: every value changes numerics. A, B and MLP_FUSED_GU also miss the HF fp32
# KL "+5% on every prompt" line; A2 and C pass it (C: 1 ulp on 1 logit) -- enabling them is the user's call.
I3_FLAG_DEFAULTS = {"FA_PROGCFG": "1", "MLP_PROGCFG": "1", "MLP_FUSED_GU": "0", "LMHEAD": "0"}
I3_FLAG_VALUES = {
    "FA_PROGCFG": ("0", "1"),
    "MLP_PROGCFG": ("0", "1"),
    "MLP_FUSED_GU": ("0", "1"),
    "LMHEAD": ("0", "A", "A2", "A3", "B", "C"),
}
# I-3 LMHEAD "A3" (M4): (column splits, workers per DRAM bank, in0_block_w, per_core_N, in0 L1 grid).
I3_A3_LM_CFG = {"split": 10, "wpb": 2, "bw": 2, "pcn": 13, "in0_grid": (8, 8)}


def i3_a3_shard_w_tiles(vocab_tiles):
    """A3 chunk shard width (tiles per DRAM bank): wpb * ceil(chunk_tiles / (8 * wpb)) (776 -> 98)."""
    c = I3_A3_LM_CFG
    nt = vocab_tiles // c["split"]
    return c["wpb"] * math.ceil(nt / (DRAM_CORES * c["wpb"]))


@functools.lru_cache(maxsize=None)
def i3_a3_lm_progcfg():
    """A3 DRAM-sharded program config (R9 A3_s10_bw2: bw 2, per_core_M 1, per_core_N 13, 2 workers/bank)."""
    c = I3_A3_LM_CFG
    return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=c["bw"],
        per_core_M=1,
        per_core_N=c["pcn"],
        fused_activation=None,
        num_workers_per_dram_bank=c["wpb"],
    )


_I3_GRID = (13, 10)
# (K, N) -> (grid_x, grid_y, in0_block_w, per_core_N, out_subblock_w); per_core_M 1, mcast_in0, fuse_batch.
_I3_FA_DECODE_PROGCFGS = {
    (2048, 3072): (13, 8, 8, 1, 1),  # D15 fused q|k|v (qkv_fused)
    (2048, 2048): (13, 5, 8, 1, 1),  # D15 gate_deint (in0 L1) and o_proj (in0 DRAM)
    (2048, 4096): (13, 10, 8, 1, 1),  # q_proj q|gate interleaved (D15 off)
    (2048, 512): (8, 2, 32, 1, 1),  # k_proj / v_proj (D15 off)
}
_I3_MLP_DECODE_PROGCFGS = {(2048, 6144): (13, 4, 4, 4, 4)}


def i3_value(item):
    """Raw value of the I-3 item flag (a key of I3_FLAG_DEFAULTS): env QWEN36_I3_<item>. Raises on a
    value outside I3_FLAG_VALUES (a typo would otherwise silently run the default path)."""
    v = os.environ.get("QWEN36_I3_" + item, I3_FLAG_DEFAULTS[item])
    if v not in I3_FLAG_VALUES[item]:
        raise ValueError(f"QWEN36_I3_{item}={v!r}: expected one of {I3_FLAG_VALUES[item]}")
    return v


def i3_enabled(item):
    """True if the I-3 item is enabled: env QWEN36_I3_<item> != "0"."""
    return i3_value(item) != "0"


def i3_grid_ok(device):
    """I-3 items apply only on the swept 13x10 worker grid."""
    g = device.compute_with_storage_grid_size()
    return (int(g.x), int(g.y)) == _I3_GRID


def i3_one_tile_row(x):
    """True if x ([..., K]) has at most one tile row (M <= 32): the per_core_M = 1 configs below."""
    return math.prod(int(d) for d in list(x.shape)[:-1]) <= TILE_SIZE


@functools.lru_cache(maxsize=None)
def _i3_1d_progcfg(gx, gy, bw, pcn, sw):
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=bw,
        out_subblock_h=1,
        out_subblock_w=sw,
        per_core_M=1,
        per_core_N=pcn,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


def _i3_decode_fn(table):
    def fn(k, n):
        cfg = table.get((int(k), int(n)))
        return _i3_1d_progcfg(*cfg) if cfg is not None else None

    return fn


def i3_fa_decode_progcfg_fn(device):
    """I-3 FA_PROGCFG: `(k, n) -> 1D progcfg | None` for the full-attention decode projections, or
    None (flag off / grid not 13x10) to keep program_config=None. Shapes outside the table -> None."""
    if not i3_enabled("FA_PROGCFG") or not i3_grid_ok(device):
        return None
    return _i3_decode_fn(_I3_FA_DECODE_PROGCFGS)


def i3_mlp_decode_progcfg_fn(device):
    """I-3 MLP_PROGCFG: `(k, n) -> 1D progcfg | None` for the MLP decode gate/up projections, or None
    (flag off / grid not 13x10) to keep _pick_decode_progcfg."""
    if not i3_enabled("MLP_PROGCFG") or not i3_grid_ok(device):
        return None
    return _i3_decode_fn(_I3_MLP_DECODE_PROGCFGS)


def i3_dram_width_memcfg(k, shard_w_tiles, num_banks=DRAM_CORES):
    """DRAM WIDTH_SHARDED memory config for a [k, n] weight with `shard_w_tiles` tiles per bank."""
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_banks - 1, 0))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(crs, [k, shard_w_tiles * TILE_SIZE], ttnn.ShardOrientation.ROW_MAJOR),
    )


def i3_l1_width_memcfg(k, gx, gy):
    """L1 WIDTH_SHARDED memory config for a one-tile-row [.., k] activation on the gx x gy grid."""
    kt = k // TILE_SIZE
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(crs, [TILE_SIZE, math.ceil(kt / (gx * gy)) * TILE_SIZE], ttnn.ShardOrientation.ROW_MAJOR),
    )


def prefill_matmul_ckc():
    """Compute-kernel-config for PREFILL (T>1) matmuls; centralizes the ckc flags above.

    QWEN36_PREFILL_MM_FP32_ACC default "0" (off) since 2026-09-22 per user decision (traced 4k
    TTFT 0.153 -> 0.146 s, logits PCC vs legacy 0.9997, argmax unchanged); "1" restores fp32
    accumulation. legacy = QWEN36_PREFILL_MM_PACKER_L1_ACC=0 QWEN36_PREFILL_MM_FP32_ACC=1
    QWEN36_PREFILL_MINIMAL_CFG=0, which reproduces, bit-for-bit, every prefill ckc mlp.py /
    gdn/gated_deltanet.py / attention/gated_attention.py built inline before this helper existed:
        ttnn.WormholeComputeKernelConfig(math_fidelity=LoFi, fp32_dest_acc_en=True, packer_l1_acc=False)
    """
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=PREFILL_MM_FP32_ACC,
        packer_l1_acc=PREFILL_MM_PACKER_L1_ACC,
    )


def prefill_minimal_matmul_config(M, K, N, grid):
    """MinimalMatmulConfig for a PREFILL minimal_matmul call, or None to keep the op's own default.

    M, K, N: the call's matmul shape, kept for the call-site record / future shape-based tuning --
    the swept config below is a single fixed block/subblock choice validated (step1 sweeps) across
    the current GDN-qkv and fused-SwiGLU (T<4096) call shapes, not derived from M/K/N.
    grid: caller's cached ttnn.CoreCoord from device.compute_with_storage_grid_size() (13x10 on
    P150 -- same grid mlp.py's `self._mm_grid` already captures for `_prefill_matmul`'s
    minimal_matmul calls).

    QWEN36_PREFILL_MM_FP32_ACC default "0" (off) since 2026-09-22 per user decision (traced 4k
    TTFT 0.153 -> 0.146 s, logits PCC vs legacy 0.9997, argmax unchanged); "1" restores fp32
    accumulation. legacy = QWEN36_PREFILL_MM_PACKER_L1_ACC=0 QWEN36_PREFILL_MM_FP32_ACC=1
    QWEN36_PREFILL_MINIMAL_CFG=0.

    QWEN36_PREFILL_MINIMAL_CFG=0 (legacy): always None.
    QWEN36_PREFILL_MINIMAL_CFG=1 (default) with fp32_dest_acc_en on (QWEN36_PREFILL_MM_FP32_ACC=1,
    legacy value): also None -- a 2x4 subblock is illegal with fp32 dest, and 2x2 with the default
    (1,1,1) blocks measured no gain, so the op's own default config wins.
    QWEN36_PREFILL_MINIMAL_CFG=1 with fp32_dest_acc_en off (QWEN36_PREFILL_MM_FP32_ACC=0, now the
    default): swept M_block_size=4, K_block_size=8, N_block_size=16, subblock_h=2, subblock_w=4.
    """
    del M, K, N  # shape kept for interface symmetry with _pick_prefill_progcfg; unused today
    if not PREFILL_MINIMAL_CFG:
        return None
    if PREFILL_MM_FP32_ACC:
        return None
    return ttnn.MinimalMatmulConfig(
        M_block_size=4,
        K_block_size=8,
        N_block_size=16,
        subblock_h=2,
        subblock_w=4,
        compute_with_storage_grid_size=grid,
    )


# Grid helpers
def prefill_grid_default():
    """BH P150: (8,10); WH: (8,8). y capped at 10 on BH (grid_x=10 breaks matmul)."""
    return (8, 10) if is_blackhole() else (8, 8)


# Max grid COLUMNS a tuned prefill config may use. A Blackhole galaxy reports a 12-wide worker
# grid, but harvested P150s expose only 11, so tuning to 12 would not port. 11 x 10 = 110 cores.
PREFILL_MAX_COLS_PORTABLE = 11

# Why TP=8 wants different values (measured at S=2048, 27B, 1x8 Ring):
#   * widest_cols -- `_best_prefill_cols` ranks candidate widths by (out_subblock_w, cols), i.e.
#     subblock first. At TP=8 the halved N makes wide grids yield a small per_core_N and hence a
#     narrow subblock, so that ranking retreats to fewer columns and leaves cores idle. Measured
#     device time is monotonically decreasing in column count instead: attn_wo went 1944us @ 60
#     cores -> 700us @ 110, and mlp_gate 2943us @ 60 -> 1935us @ 110. So take the width.
#   * in0_block_w_divisor -- `min(cap, k_tiles // grid_x)` is a function of the per-device K, which
#     halves. attn_wo/gdn_out go k_tiles 48 -> 24 and `24 // 11 = 2`, but in0_block_w only has to
#     DIVIDE k_tiles, so a larger block is legal and much faster (attn_wo @ 11 cols, from the sweep:
#     bw2 786us, bw4 719us, bw6 700us, bw8 705us).
#
# in0_block_w_cap is L1-BOUND, NOT just a legality bound. in0_block_w sizes the in0 circular
# buffer, and `_wo_proj` / the MLP prefill arm write their OUTPUT to L1 (attention/tp.py:246,
# mlp.py:284) -- so the CBs and a resident L1 output tensor compete for the same 1536 KB. Measured
# on the real model: cap=8 overflows and test_model_tp_long_prefill dies with
#   "Statically allocated circular buffers in program N clash with L1 buffers on core range
#    [0-0 - 10-8]. L1 buffer allocated at 1314560 and static circular buffer region ends at 1372032"
# from attention/tp.py:241. A standalone per-op sweep CANNOT see this: in isolation the only L1
# tenant is the op under test, so it reports a win that the full model has no room for. Any future
# raise of this cap must be validated by test_model_tp_long_prefill, not by the sweep alone.
_PREFILL_TUNING = {
    4: dict(widest_cols=False, in0_block_w_divisor=False, in0_block_w_cap=4),
    8: dict(widest_cols=True, in0_block_w_divisor=True, in0_block_w_cap=4),
}


def prefill_tuning(num_devices):
    """Prefill matmul tuning for this TP; unknown TP falls back to the frozen TP=4 values."""
    return _PREFILL_TUNING.get(num_devices, _PREFILL_TUNING[4])


def _roundup(a, b):
    return b * math.ceil(a / b)


def _find_largest_divisor(n, max_div=8):
    for d in range(max_div, 0, -1):
        if n % d == 0:
            return d
    return 1


def _find_grid(n_tiles, target=32):
    max_r, max_c = 8, 8
    possible = [k for k in range(1, max_r * max_c + 1) if n_tiles % k == 0]
    possible.sort(key=lambda x: abs(x - target))
    for cores in possible:
        for rows in range(1, max_r + 1):
            if cores % rows == 0:
                cols = cores // rows
                if cols <= max_c:
                    return rows, cols
    raise ValueError(f"Cannot find grid for {n_tiles} tiles")


# DRAM-sharded config builders
def create_dram_sharded_mem_config(k, n):
    """WIDTH_SHARDED DRAM memory config for a weight matrix [k, n]."""
    padded_n = _roundup(n, TILE_SIZE * DRAM_CORES)
    shard_spec = ttnn.ShardSpec(
        DRAM_GRID,
        (k, padded_n // DRAM_CORES),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        shard_spec,
    )


def create_dram_sharded_matmul_program_config(m, k, n, num_cores=None):
    """DRAM-sharded matmul program config (decode, small M)."""
    m_tiles = math.ceil(m / TILE_SIZE)
    k_tiles = math.ceil(k / TILE_SIZE)
    n_padded = _roundup(n, TILE_SIZE * DRAM_CORES)
    n_tiles = n_padded // TILE_SIZE

    if num_cores is None:
        rows, cols = _find_grid(k_tiles)
        num_cores = rows * cols

    k_tiles_per_core = k_tiles // num_cores
    if k_tiles_per_core == 0:
        k_tiles_per_core = k_tiles
        num_cores = 1
    in0_block_w = _find_largest_divisor(k_tiles_per_core)
    per_core_N = n_tiles // num_cores if n_tiles >= num_cores else 1

    return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=in0_block_w,
        per_core_M=m_tiles,
        per_core_N=per_core_N,
        fused_activation=None,
    )


def create_matmul_1d_decode_progcfg(m, k, n, num_cores, fused_activation=None, fp32_acc=True, grid_w=8):
    """Explicit-grid 1D (mcast_in0) decode matmul progcfg on ~`num_cores` cores — small grids beat
    the ~80-core DRAM-sharded grid on the bandwidth-bound skinny decode matmuls. Weight must be interleaved.

    Grid is shaped WIDE-first (cols up to `grid_w`, the device worker-grid width — 11 on BH P150, 8 on
    WH): for a fixed core budget a wide-short grid shortens the in0 multicast column and beats a
    tall-narrow one (~2% on this matmul; see test_mlp_matmul_sweep wide1d_* vs forced1d_*). Default
    grid_w=8 preserves the legacy shaping for callers that don't pass the device width."""
    cols = min(grid_w, num_cores)
    rows = math.ceil(num_cores / cols)
    m_tiles = math.ceil(m / TILE_SIZE)
    k_tiles = math.ceil(k / TILE_SIZE)
    n_tiles = math.ceil(n / TILE_SIZE)
    # mcast_in0: every core streams the full K, so in0_block_w must divide the full k_tiles.
    per_core_k = _find_largest_divisor(k_tiles)
    per_core_n = math.ceil(n_tiles / (cols * rows))
    cap = 4 if fp32_acc else 8  # fp32_dest_acc caps subblock area at 4
    sub_w = max(i for i in range(1, cap + 1) if per_core_n % i == 0)
    sub_h = max(i for i in range(1, cap + 1) if m_tiles % i == 0 and i * sub_w <= cap)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(cols, rows),
        in0_block_w=per_core_k,
        out_subblock_h=sub_h,
        out_subblock_w=sub_w,
        per_core_M=m_tiles,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=fused_activation,
        mcast_in0=True,
    )


def matmul_1d_decode(x, weight, decode_1d_progcfg, compute_cfg, out_memory_config=ttnn.L1_MEMORY_CONFIG):
    """Small-grid 1D (mcast_in0) decode matmul on an interleaved weight; interleaves the K-sharded
    activation first since mcast_in0 needs the full K per core. See test_mlp_matmul_sweep."""
    x_il = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
    out = ttnn.linear(
        x_il,
        weight,
        compute_kernel_config=compute_cfg,
        program_config=decode_1d_progcfg,
        memory_config=out_memory_config,
    )
    if x_il is not x:
        ttnn.deallocate(x_il)
    return out


# Single-device DECODE (T==1, activations padded to M=32) 1D matmul progcfg selection, by shape
# class. Swept on P150 with the model's decode compute config (LoFi, fp32_dest_acc_en=True,
# packer_l1_acc=True), x in L1 interleaved, weights DRAM interleaved; every config built via
# create_matmul_1d_decode_progcfg(m=32, k, n, num_cores=c, grid_w=device_grid.x):
#
#   matmul                    K      N      dtype   auto ttnn      1D (num_cores)   speedup
#   w2 (MLP down-proj)        6144   2048   bfp8    154 us         c=110   74 us    2.1x
#   w1/w3 (MLP up-proj)       2048   6144   bfp4     95 us         c=32    63 us    1.5x
#   GDN out-proj              2048   2048   bfp8     81 us         c=110   50 us    1.6x
#   GDN mega in-proj          2048   8224   bfp8    103 us         c=64    85 us    1.2x
#
# Shape class -> num_cores: K>N (down-proj) and K==N (square) both want the ~full-grid 110-core 1D
# config. Of the two N>K (up-proj-like) shapes, the moderately-wide one (w1/w3, N=3K) wants the
# small 32-core grid; the very-wide one (GDN mega in-proj, N~=4.02K) wants 64 instead -- num_cores is
# NOT a monotonic function of N/K here, so this is a lookup keyed on the 4 measured shapes, not a
# smooth formula. N >= 4*K routes to the wide-N bucket (64); K < N < 4*K routes to the up-proj
# bucket (32). Do not extrapolate this table to unmeasured shapes without a sweep.
@functools.lru_cache(maxsize=None)
def _pick_decode_progcfg(k, n, gx, gy):
    if k >= n:
        num_cores = 110  # down-proj (k>n) or square (k==n)
    elif n >= 4 * k:
        num_cores = 64  # very wide N (e.g. GDN mega in-proj)
    else:
        num_cores = 32  # moderately wide N (e.g. MLP up-proj)
    num_cores = min(num_cores, gx * gy)
    return create_matmul_1d_decode_progcfg(m=32, k=k, n=n, num_cores=num_cores, grid_w=gx)


def make_decode_progcfg_fn(device):
    """Per-device callable `(k, n) -> progcfg | None` for single-token (T==1, M padded to 32)
    decode matmuls (see the measured table above `_pick_decode_progcfg`).

    Captures the worker grid once (grid.x, NOT a hardcoded default -- e.g. 13 on this P150, vs the
    create_matmul_1d_decode_progcfg default grid_w=8 for callers that don't pass the device width).
    QWEN36_DECODE_PROGCFG=0 disables it (every call returns None -> ttnn auto-config)."""
    if os.environ.get("QWEN36_DECODE_PROGCFG", "1") == "0":
        return lambda k, n: None
    grid = device.compute_with_storage_grid_size()
    gx, gy = grid.x, grid.y

    def fn(k, n):
        return _pick_decode_progcfg(int(k), int(n), gx, gy)

    return fn


def create_activation_shard_config(k):
    """WIDTH_SHARDED L1 activation config for a [*, k] activation."""
    k_tiles = k // TILE_SIZE
    rows, cols = _find_grid(k_tiles)
    num_cores = rows * cols
    width_per_core = k // num_cores
    return ttnn.create_sharded_memory_config(
        shape=(TILE_SIZE, width_per_core),
        core_grid=ttnn.CoreGrid(x=cols, y=rows),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


# 2D prefill matmul config
def _get_out_subblock_w(per_core_n, out_subblock_h, max_area=4):
    """Widest out_subblock_w dividing per_core_n with out_subblock_h * w <= max_area. max_area 4 = the
    fp32-dest limit (default for every caller); 8 is legal only with fp32 dest acc off (I-2)."""
    for w in range(min(per_core_n, max_area // out_subblock_h), 0, -1):
        if per_core_n % w == 0:
            return w
    return 1


def _full_grid_crs(grid):
    """Full-grid allowed_worker_cores for CCL-fused matmuls, which bypass ttnn::prim::matmul()'s normalize_program_config()."""
    gx, gy = grid
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})


def create_prefill_matmul_program_config(m, k, n, grid_size=None, fused_activation=None, tuning=None):
    """2D prefill matmul progcfg (DRAM-interleaved).

    fused_activation in packer; sharded kernel rejects ttnn.linear(activation=...) with progcfg.
    tuning: a `_PREFILL_TUNING` entry (see `prefill_tuning`); None = the frozen TP=4 behavior."""
    if grid_size is None:
        grid_size = prefill_grid_default()
    tuning = tuning or _PREFILL_TUNING[4]
    per_core_M = max(1, math.ceil(m / TILE_SIZE / grid_size[1]))
    per_core_N = max(1, math.ceil(n / TILE_SIZE / grid_size[0]))

    out_subblock_h = 1
    out_subblock_w = _get_out_subblock_w(per_core_N, out_subblock_h)

    k_tiles = math.ceil(k / TILE_SIZE)
    cap = tuning["in0_block_w_cap"]
    if tuning["in0_block_w_divisor"]:
        # in0_block_w only has to divide k_tiles (no K tail in the 2D mcast kernel), so take the
        # largest legal block rather than scaling with grid width -- see _PREFILL_TUNING.
        in0_block_w = _find_largest_divisor(k_tiles, cap)
    else:
        in0_block_w = min(cap, max(1, k_tiles // grid_size[0]))

    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=fused_activation,
        fuse_batch=False,
    )


_PREFILL_TILE_BYTES = {ttnn.bfloat16: 2048, ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576, ttnn.float32: 4096}


def _mk_2d_progcfg(cols, rows, in0_block_w, per_core_M, per_core_N, out_subblock_w):
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(cols, rows),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        out_block_h=per_core_M,
        out_block_w=per_core_N,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )


# Per-shape overrides for prefill matmuls where logs/step1/mm_sweep_summary.md found a config
# faster than the general _pick_prefill_progcfg rule below. Keyed by (m, k, n); only applied on the
# swept 13x10 grid (BH P150). QWEN36_PREFILL_PROGCFG_OVERRIDES=0 disables this table.
_PREFILL_PROGCFG_OVERRIDES = {
    # GDN mega in-proj (bf16 x bfloat8_b): sweep measured 400us vs 544us for the rule's config.
    (2048, 2048, 8224): dict(in0_block_w=8, out_subblock_h=1, out_subblock_w=4, per_core_M=7, per_core_N=20),
}

# MM BW16 (2026-09-28; plan_0928 P3_MMSWEEP / P5_INT1B), env QWEN36_MM_BW16 (tp_common.mm_enabled):
# in0_block_w 16 instead of the rule's 8 for the o-proj family (FA o_proj, GDN o_proj) and the FA
# q|k|v fused proj, same per_core_M / per_core_N / out_subblock_w as the bw8 config the rule already
# picks for these two shapes. Keyed by (m, k, n), applied only on the swept 13x10 grid, same as
# _PREFILL_PROGCFG_OVERRIDES. Not bit-exact (PCC ~0.99995: fewer bf16 L1-acc roundings in the K loop).
_MM_BW16_OVERRIDES = {
    (2048, 2048, 2048): dict(in0_block_w=16, out_subblock_h=1, out_subblock_w=5, per_core_M=7, per_core_N=5),
    (2048, 2048, 3072): dict(in0_block_w=16, out_subblock_h=1, out_subblock_w=8, per_core_M=7, per_core_N=8),
}


@functools.lru_cache(maxsize=None)
def _pick_prefill_progcfg(m, k, n, in0_tile_b, in1_tile_b, fp32_acc, gx, gy, budget, sub_area=4):
    """Single-device, DRAM-output prefill matmul config, or None to keep ttnn auto-config.

    Swept on P150 (Qwen3.5-2B shapes): the largest in0_block_w whose static CBs fit the per-core L1
    budget wins, then the widest column count whose per-core N still gives an output subblock >= 2
    (a 1-wide subblock costs ~15%). Auto-config picks in0_block_w=1 on the 13x10 grid (Kt % 13 != 0)
    and measured 2.5-5x slower on every shape. Tiny N is left to auto-config.

    sub_area: output-subblock area cap (4 = fp32-dest limit, the pre-I-2 rule; 8 with I-2 PICKER when
    fp32 dest acc is off -- see make_prefill_progcfg_fn)."""
    if gx == 13 and gy == 10 and os.environ.get("QWEN36_PREFILL_PROGCFG_OVERRIDES", "1") != "0":
        override = _PREFILL_PROGCFG_OVERRIDES.get((m, k, n))
        if override is not None:
            return _mk_2d_progcfg(
                gx,
                gy,
                override["in0_block_w"],
                override["per_core_M"],
                override["per_core_N"],
                override["out_subblock_w"],
            )
        if mm_enabled("BW16"):
            bw16_ov = _MM_BW16_OVERRIDES.get((m, k, n))
            if bw16_ov is not None:
                return _mk_2d_progcfg(
                    gx,
                    gy,
                    bw16_ov["in0_block_w"],
                    bw16_ov["per_core_M"],
                    bw16_ov["per_core_N"],
                    bw16_ov["out_subblock_w"],
                )
    Mt, Kt, Nt = math.ceil(m / TILE_SIZE), math.ceil(k / TILE_SIZE), math.ceil(n / TILE_SIZE)
    if Nt < 8:
        return None
    per_core_M = max(1, math.ceil(Mt / gy))
    out_bytes = 2048 + (4096 if fp32_acc else 2048)  # bf16 out CB + interm CB per tile
    for bw in (8, 4, 2, 1):
        if Kt % bw:
            continue
        fallback = None
        for cols in range(min(gx, Nt), 0, -1):
            per_core_N = math.ceil(Nt / cols)
            est = (
                per_core_M * bw * 2 * in0_tile_b
                + per_core_N * bw * 2 * in1_tile_b
                + per_core_M * per_core_N * out_bytes
            )
            if est > budget:
                continue
            sw = _get_out_subblock_w(per_core_N, 1, max_area=sub_area)
            if sw >= 2 or per_core_N == 1:
                return _mk_2d_progcfg(cols, gy, bw, per_core_M, per_core_N, sw)
            if fallback is None:
                fallback = (cols, per_core_N, sw)
        if fallback is not None:
            return _mk_2d_progcfg(fallback[0], gy, bw, per_core_M, fallback[1], fallback[2])
    return None


def make_prefill_progcfg_fn(device):
    """Per-device callable `(m, k, n, in0_dtype, in1_dtype, fp32_acc=True) -> progcfg | None`.

    Captures the worker grid and the allocator's per-bank L1 budget once. QWEN36_PREFILL_PROGCFG=0
    disables it (every call returns None -> ttnn auto-config). I-2 PICKER (QWEN36_I2_PICKER=1, read
    per call): subblock area cap 8 instead of 4 for calls with fp32_acc False."""
    if os.environ.get("QWEN36_PREFILL_PROGCFG", "1") == "0":
        return lambda *args, **kwargs: None
    grid = device.compute_with_storage_grid_size()
    budget = ttnn.get_memory_view(device, ttnn.BufferType.L1).total_bytes_per_bank

    def fn(m, k, n, in0_dtype, in1_dtype, fp32_acc=True):
        sub_area = 8 if (not fp32_acc and i2_enabled("PICKER")) else 4
        return _pick_prefill_progcfg(
            int(m),
            int(k),
            int(n),
            _PREFILL_TILE_BYTES[in0_dtype],
            _PREFILL_TILE_BYTES[in1_dtype],
            bool(fp32_acc),
            grid.x,
            grid.y,
            budget,
            sub_area,
        )

    return fn


def _widest_prefill_cols(n, max_cols, subblock_slack=1):
    """Widest grid whose output subblock stays within `subblock_slack` of the best achievable.

    The TP=8 counterpart to `_best_prefill_cols`. More columns is usually a win at TP=8 (the halved
    per-device N leaves cores idle), but NOT when the extra width collapses the subblock: measured
    at S=2048, mlp_gate (N=2176 -> 68 tiles) goes cols 9 -> 11, per_core_N 8 -> 7, and 7 is prime so
    out_subblock_w drops 4 -> 1 -- a 2058us -> 2118us REGRESSION, i.e. the subblock-first ranking
    was right for that shape. Guarding on the subblock keeps the wide grid exactly where it pays:

        matmul     default        this rule       measured
        attn_wo    c10_bw2_sw4    c11_bw4_sw3     803.5 -> 718.7us
        gdn_out    c10_bw2_sw4    c11_bw4_sw3     802.3 -> 719.9us
        mlp_down   c10_bw4_sw4    c11_bw4_sw3    1787.4 -> 1724.9us
        mlp_gate   c9_bw4_sw4     c9_bw4_sw4     2058.1us (unchanged -- already optimal)
    """
    n_tiles = math.ceil(n / TILE_SIZE)
    sw = {cols: _get_out_subblock_w(math.ceil(n_tiles / cols), 1) for cols in range(1, max_cols + 1)}
    floor = max(sw.values()) - subblock_slack
    return max((cols for cols, w in sw.items() if w >= floor), default=1)


def _best_prefill_cols(n, max_cols):
    """Grid width (<=max_cols) maximizing the output subblock, tie-broken to more cores — avoids the
    1x1-subblock stall (e.g. gate/up N=4352 -> 7-wide -> 1x4) the default full width can force."""
    n_tiles = math.ceil(n / TILE_SIZE)
    best_cols, best_key = 1, None
    for cols in range(1, max_cols + 1):
        sw = _get_out_subblock_w(math.ceil(n_tiles / cols), 1)
        key = (sw, cols)  # prefer wider subblock, then more columns (more compute cores)
        if best_key is None or key > best_key:
            best_key, best_cols = key, cols
    return best_cols


def create_prefill_mlp_matmul_program_config(m, k, n, fused_activation=None, max_cols=None, tuning=None):
    """FPU-tuned 2D prefill progcfg for MLP matmuls: picks the grid width that maximizes the output
    subblock (drives prefill FPU) instead of the default full width.

    max_cols caps the grid width. Default = prefill_grid_default()[0] (8). Pass the device worker-grid
    width (11 on BH P150) to let the subblock heuristic go wide -> the measured prefill winners
    (gate 9-wide, down/wo 10-wide, gdn_qkvz 11-wide; test_mlp_matmul_sweep_prefill). Fused AG/RS paths
    pin 8-wide separately and are unaffected.

    tuning: a `_PREFILL_TUNING` entry. With `widest_cols` (TP=8) the subblock-first width heuristic
    is replaced by "take the width, clamped to PREFILL_MAX_COLS_PORTABLE" -- measured device time at
    TP=8 falls monotonically with column count, so trading cores for a wider subblock loses."""
    grid = prefill_grid_default()
    tuning = tuning or _PREFILL_TUNING[4]
    limit = max_cols or grid[0]
    if tuning["widest_cols"]:
        # Cap the width at PREFILL_MAX_COLS_PORTABLE (harvested parts expose 11, not 12) and never
        # exceed the output tile count -- columns beyond it get per_core_N=1 with nothing to compute,
        # paying mcast cost for no work.
        cols = _widest_prefill_cols(n, max(1, min(limit, PREFILL_MAX_COLS_PORTABLE, math.ceil(n / TILE_SIZE))))
    else:
        cols = _best_prefill_cols(n, limit)
    return create_prefill_matmul_program_config(
        m, k, n, grid_size=(cols, grid[1]), fused_activation=fused_activation, tuning=tuning
    )


# Mesh tensor helpers
def shard_w(torch_tensor, mesh, dim, memory_config, cache_path, dtype=ttnn.bfloat8_b):
    """Torch weight [out,in] -> sharded mesh tensor. Transpose to [in,out]; dim=-1 column, dim=0 row.

    The bf16 cast + transpose runs as the as_tensor preprocess, so it only executes on a tensor-cache
    miss. On a hit the checkpoint tensor is never materialised (it may be a memory-mapped safetensor
    on a network mount, and reading 27B parameters through it is what pushed the CI weight load past
    its 1200 s pytest timeout)."""
    return ttnn.as_tensor(
        torch_tensor,
        preprocess=lambda t: t.to(torch.bfloat16).T.contiguous(),
        dtype=dtype,
        device=mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=dim),
        layout=ttnn.TILE_LAYOUT,
        memory_config=memory_config,
        cache_file_name=cache_path,
    )


def agmm_k_block_size(k_local, default=8):
    """Largest power-of-2 K_block_size <= `default` that divides K_tiles/device (AGMM Ring has no tail).

    TP=4: 1280->40 tiles->8; TP=8: 640->20 tiles->4. Odd divisors (e.g. 5|20) are unsafe on Ring.
    """
    k_tiles = k_local // TILE_SIZE
    b = 1 << (min(default, max(1, k_tiles)).bit_length() - 1)
    while b > 1 and k_tiles % b:
        b //= 2
    return b


def all_gather_matmul_prefill(
    x,
    weight,
    tt_ccl,
    compute_cfg,
    topology,
    grid=(7, 9),
    cluster_axis=1,
    fused_activation=None,
    out_memory_config=ttnn.DRAM_MEMORY_CONFIG,
):
    """Fused all-gather(dim=3) + column-parallel matmul for prefill (all_gather_minimal_matmul_async).

    x: K-sharded activation [.,S,K/tp]; weight: [K,N] col-sharded (K full). Gathers x to full K and
    matmuls in one op, replacing a separate all_gather + linear. fused_activation applied per tile
    before pack (non-parametrized op, e.g. ttnn.UnaryOpType.SILU). out_memory_config places the result
    (default DRAM; L1 keeps it resident for downstream slices)."""
    S, K_local = x.shape[-2], x.shape[-1]
    x4 = ttnn.reshape(x, (1, 1, S, K_local))
    # AG-bound: 2 ethernet links parallelize the gather (P150x4 max; traced_8k TTFT win). grid.x must
    # = num_links*workers, and the 7-wide default (prime) forces 1 link -> widen to 8 (2 links, 4 workers).
    num_links = 2
    grid = (8, grid[1])
    workers = grid[0] // num_links
    cfg = ttnn.MinimalMatmulConfig(
        M_block_size=4,
        K_block_size=agmm_k_block_size(K_local),
        N_block_size=8,
        subblock_h=1,
        subblock_w=4,
        compute_with_storage_grid_size=ttnn.CoreCoord(grid[0], grid[1]),
    )
    out = ttnn.experimental.all_gather_minimal_matmul_async(
        input_tensor=x4,
        weight_tensor=weight,
        config=cfg,
        fused_activation=fused_activation,
        compute_kernel_config=compute_cfg,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
        num_links=num_links,
        topology=topology,
        cluster_axis=cluster_axis,
        memory_config=out_memory_config,
        dtype=ttnn.bfloat16,
        force_transpose=True,
        num_workers_per_link=workers,
        num_buffers_per_channel=8,
    )[0]

    return out


def mlp_gateup_agmm_enabled(num_devices):
    """Fuse the ff_norm all-gather into the MLP gate/up matmul (prefill). TP-only (needs the gather)."""
    return num_devices > 1


def all_gather_swiglu_prefill(
    x, weight, tt_ccl, compute_cfg, topology, grid=(7, 9), cluster_axis=1, out_memory_config=ttnn.DRAM_MEMORY_CONFIG
):
    """Fused all-gather + col-parallel gate/up matmul + SwiGLU for prefill (packing gate+up lets ff_norm's AG fuse in).

    x: K-sharded [.,S,K/tp]; weight: tile-pair-interleaved [gate|up] [K, 2N/tp]. Emits silu(gate)*up of width N/tp."""
    S, K_local = x.shape[-2], x.shape[-1]
    x4 = ttnn.reshape(x, (1, 1, S, K_local))
    num_links = 2
    grid = (8, grid[1])
    workers = grid[0] // num_links
    cfg = ttnn.MinimalMatmulConfig(
        M_block_size=8,
        K_block_size=agmm_k_block_size(K_local),
        N_block_size=16,
        subblock_h=1,
        subblock_w=4,
        compute_with_storage_grid_size=ttnn.CoreCoord(grid[0], grid[1]),
    )
    return ttnn.experimental.all_gather_minimal_matmul_async(
        input_tensor=x4,
        weight_tensor=weight,
        config=cfg,
        compute_kernel_config=compute_cfg,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
        num_links=num_links,
        topology=topology,
        cluster_axis=cluster_axis,
        memory_config=out_memory_config,
        dtype=ttnn.bfloat16,
        force_transpose=True,
        num_workers_per_link=workers,
        num_buffers_per_channel=8,
        fuse_swiglu=True,
    )[0]


def build_mmrs_decode_state(mesh_device, M, K_local, N, nd, dtype=ttnn.bfloat16):
    """Build (progcfg, intermediate_buffer, output_buffer) for a decode matmul_reduce_scatter out-proj.

    M = LOGICAL decode batch (max_batch_size) — the op returns the persistent buffer with its logical
    shape, so an oversized (tile-padded) M leaks into the residual stream. TILE layout pads M<32.
    dtype MUST match the out-proj input activation (bf16 for MLP/attn; FLOAT32 for GDN, which keeps
    fp32 for stability) — the op's default output dtype is the input's, and writing it into a
    mismatched buffer corrupts the output. Matmul on reduced grid (8,6); RS workers at offset (0,6).
    interm [1,1,M,N], out [1,1,M,N/nd]."""
    cg = (8, 6)
    per_core_N = max(1, math.ceil(N / TILE_SIZE / cg[0]))
    pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=cg,
        in0_block_w=min(4, max(1, K_local // TILE_SIZE // cg[0])),
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=max(1, math.ceil(M / TILE_SIZE / cg[1])),
        per_core_N=per_core_N,
        out_block_w=max(1, per_core_N // 2),
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
        allowed_worker_cores=_full_grid_crs(cg),
    )
    mk = lambda w: ttnn.from_torch(
        torch.zeros(1, 1, M, w),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    return pc, mk(N), mk(N // nd)


def matmul_reduce_scatter_decode(
    x, weight, tt_ccl, interm_buf, out_buf, progcfg, compute_cfg, topology, rs_offset=(0, 6)
):
    """Fused row-parallel matmul + reduce-scatter(dim=3) for decode (matmul_reduce_scatter_async).

    x: K-sharded [.,M,K_local]; weight: [K_local,N] K-sharded. Matmul runs on progcfg's (reduced)
    grid; RS workers land at rs_offset (disjoint rows) to avoid the collision that deadlocks a
    full-grid fused CCL. Persistent buffers are caller-owned. Returns [.,M,N/nd] (fractured, DRAM)."""
    _, rs_out = ttnn.experimental.matmul_reduce_scatter_async(
        x,
        weight,
        persistent_intermediate_buffer=interm_buf,
        persistent_output_buffer=out_buf,
        dim=3,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(),
        reduce_scatter_core_grid_offset=rs_offset,
        barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(),
        num_links=1,
        memory_config_rs=ttnn.DRAM_MEMORY_CONFIG,
        topology=topology,
        subdevice_id=None,
        memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
        program_config=progcfg,
        compute_kernel_config=compute_cfg,
    )
    # rs_out IS the persistent output buffer; clone so the caller can deallocate its copy while the
    # persistent buffer survives for the next token (else layer.py's deallocate frees it -> corruption).
    return ttnn.clone(rs_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _mmrs_prefill_shared_bufs(tt_ccl, M, N, nd, dtype):
    """Lazily allocate (and cache on tt_ccl) shared persistent buffers for the prefill fused out-proj.

    Prefill M (=chunk seq, e.g. 2048) makes per-layer buffers huge (fp32 [1,1,2048,5120]≈42MB × 64
    layers = infeasible). Prefill runs layers sequentially and each op's output is cloned before the
    next layer reuses the buffer, so ONE shared set per (M,N,nd,dtype) is safe. Allocated during the
    pre-capture warmup forward (eager), reused inside the trace. Keyed so variable M/dtype coexist."""
    cache = getattr(tt_ccl, "_qwen36_mmrs_prefill_bufs", None)
    if cache is None:
        cache = {}
        tt_ccl._qwen36_mmrs_prefill_bufs = cache
    key = (M, N, nd, str(dtype))
    if key not in cache:
        mesh = tt_ccl.mesh_device
        mk = lambda w: ttnn.from_torch(
            torch.zeros(1, 1, M, w),
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        cache[key] = (mk(N), mk(N // nd))
    return cache[key]


def matmul_reduce_scatter_prefill(x, weight, tt_ccl, compute_cfg, topology, nd, dtype, grid=(8, 8), rs_offset=(0, 8)):
    """Fused row-parallel out-proj matmul + reduce-scatter for PREFILL (matmul_reduce_scatter_async).

    Unlike decode (M=1, where the 2D matmul collapses to ~8 cores and this loses), at prefill M>>1 the
    2D matmul fills the grid, so overlapping the RS with the matmul is a WIN (biggest for the fp32
    GDN-out with its large RS). grid=(8,8): matmul rows 0-7, RS workers rows 8-9. x: K-sharded
    [.,M,K_local]; weight [K_local,N]. Returns [1,1,M,N/nd] (cloned; shared buffer survives)."""
    M, K_local = x.shape[-2], x.shape[-1]
    N = weight.shape[-1]
    interm, out_buf = _mmrs_prefill_shared_bufs(tt_ccl, M, N, nd, dtype)
    x4 = ttnn.reshape(x, (1, 1, M, K_local))
    # RS-bound: 2 ethernet links parallelize the fp32 cross-device reduce (P150x4 max; traced_8k win).
    # grid (8,8) leaves rows 8-9 for the 2 RS worker rows.
    num_links = 2
    per_core_N = max(1, math.ceil(N / TILE_SIZE / grid[0]))
    pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=min(4, max(1, K_local // TILE_SIZE // grid[0])),
        out_subblock_h=1,
        # Keep 1x1: op242 is RS-bound and this op is pipelined to overlap the matmul with the RS.
        # Widening the subblock desyncs that overlap and measured net-negative on traced_8k TTFT.
        out_subblock_w=1,
        per_core_M=max(1, math.ceil(M / TILE_SIZE / grid[1])),
        per_core_N=per_core_N,
        out_block_w=max(1, per_core_N // 2),
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
        allowed_worker_cores=_full_grid_crs(grid),
    )
    _, rs = ttnn.experimental.matmul_reduce_scatter_async(
        x4,
        weight,
        persistent_intermediate_buffer=interm,
        persistent_output_buffer=out_buf,
        dim=3,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(),
        reduce_scatter_core_grid_offset=rs_offset,
        barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(),
        num_links=num_links,
        memory_config_rs=ttnn.DRAM_MEMORY_CONFIG,
        topology=topology,
        subdevice_id=None,
        memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
        program_config=pc,
        compute_kernel_config=compute_cfg,
    )
    return ttnn.clone(rs, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def sharded_decode_matmul(
    x,
    weight,
    compute_cfg,
    decode_progcfg,
    act_shard_cfg,
    prefill_progcfg_fn,
    prefill_k,
    decode_out_memory_config=ttnn.DRAM_MEMORY_CONFIG,
):
    """DRAM-WIDTH_SHARDED weight matmul; branches on M (decode vs prefill).

    Decode (M<=32): L1-sharded act + DRAM-sharded kernel. Prefill: 2D matmul.
    Gate on x.shape[-2] (seq/M), not x.shape[1] (Z=1 in both modes). Decode result placement is
    `decode_out_memory_config` (default DRAM-interleaved; pass L1 to keep the small decode
    activation resident). Prefill result is always DRAM-interleaved."""
    seq = x.shape[-2]
    if seq <= TILE_SIZE:
        # Reshard act to L1 if needed; skip dealloc when x already sharded (GDN reuses x).
        already_sharded = x.memory_config() == act_shard_cfg
        x_sh = x if already_sharded else ttnn.to_memory_config(x, act_shard_cfg)
        out = ttnn.linear(
            x_sh,
            weight,
            compute_kernel_config=compute_cfg,
            program_config=decode_progcfg,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        if not already_sharded:
            ttnn.deallocate(x_sh)
        return ttnn.to_memory_config(out, decode_out_memory_config)
    pc = prefill_progcfg_fn(seq, prefill_k, weight.shape[-1])
    return ttnn.linear(
        x, weight, compute_kernel_config=compute_cfg, program_config=pc, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def replicate(torch_tensor, mesh, cache_path, dtype=ttnn.bfloat16):
    """Small tensor (norm/bias) -> replicated on every device."""
    if torch_tensor.dim() == 1:
        torch_tensor = torch_tensor.unsqueeze(0).unsqueeze(0)
    elif torch_tensor.dim() == 2:
        torch_tensor = torch_tensor.unsqueeze(0)
    return ttnn.as_tensor(
        torch_tensor.to(torch.bfloat16),
        dtype=dtype,
        device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=cache_path,
    )


def shard_small(torch_tensor, mesh, cache_path, dim=-1, dtype=ttnn.bfloat16):
    """Small per-head tensor (conv taps, A_log, dt_bias) -> sharded."""
    if torch_tensor.dim() == 1:
        torch_tensor = torch_tensor.unsqueeze(0).unsqueeze(0)
    elif torch_tensor.dim() == 2:
        torch_tensor = torch_tensor.unsqueeze(0)
    return ttnn.as_tensor(
        torch_tensor.to(torch.bfloat16),
        dtype=dtype,
        device=mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=dim),
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=cache_path,
    )


def replicate_kv_weight(weight, n_kv_heads, tp, head_dim):
    """Replicate KV weight so each device gets >=1 head. No-op when tp <= n_kv_heads."""
    if tp <= n_kv_heads:
        return weight
    chunks = weight.reshape(n_kv_heads, head_dim, -1)
    parts = []
    for d in range(tp):
        kv_idx = (d * n_kv_heads) // tp
        parts.append(chunks[kv_idx])
    return torch.cat(parts, dim=0).reshape(tp * head_dim, -1)


# FP8 dequantization
def dequant_fp8_block(weight_fp8, scale_inv, block_size=128):
    """Dequantize a block-wise FP8 weight tensor to bfloat16."""
    out_f, in_f = weight_fp8.shape
    weight_bf16 = weight_fp8.to(torch.bfloat16).reshape(out_f // block_size, block_size, in_f // block_size, block_size)
    weight_bf16 = weight_bf16 * scale_inv[:, None, :, None].to(torch.bfloat16)
    return weight_bf16.reshape(out_f, in_f)


# Weight-prep (reorder HF weights for per-device sharding)
def prepare_attn_qkv(q_w, k_w, v_w, qg_per, kv_per, tp):
    """Fuse attn q+gate/k/v for column-parallel shard: each device gets [qg_d|k_d|v_d].

    q_w: [n_heads*head_dim*2, in]; k_w/v_w: [n_kv_heads*head_dim, in].
    qg_per/kv_per: per-device out block sizes."""
    parts = []
    for d in range(tp):
        parts.append(q_w[d * qg_per : (d + 1) * qg_per, :])
        parts.append(k_w[d * kv_per : (d + 1) * kv_per, :])
        parts.append(v_w[d * kv_per : (d + 1) * kv_per, :])
    return torch.cat(parts, dim=0)


def prepare_attn_qkv_deint(q_w, k_w, v_w, nh_local, hd, kv_per, tp):
    """Like prepare_attn_qkv but de-interleaves [q,g] per head -> [all_q|all_gate|k|v] per device.

    Avoids prefill relayout in _make_heads (column perm only; numerically identical).
    q_w: [nh_total*hd*2, in]; nh_local/kv_per: per-device block sizes."""
    hd2 = hd * 2
    parts = []
    for d in range(tp):
        base = d * nh_local * hd2
        q_rows = [q_w[base + h * hd2 : base + h * hd2 + hd, :] for h in range(nh_local)]
        g_rows = [q_w[base + h * hd2 + hd : base + h * hd2 + hd2, :] for h in range(nh_local)]
        # Per-device layout [all_q | k | v | all_gate]: q/k/v contiguous so _make_heads* can hand
        # the fused q|k|v block straight to nlp_create_qkv_heads (no re-concat); gate trails, applied
        # post-SDPA. (Column perm only; numerically identical to [q|gate|k|v].)
        parts.append(torch.cat(q_rows, dim=0))  # all_q
        parts.append(k_w[d * kv_per : (d + 1) * kv_per, :])
        parts.append(v_w[d * kv_per : (d + 1) * kv_per, :])
        parts.append(torch.cat(g_rows, dim=0))  # all_gate (last)
    return torch.cat(parts, dim=0)


def prepare_gdn_qkv(qkv_w, key_dim, value_dim, nk, dk, nv, dv, tp):
    """Interleave GDN Q/K/V heads for row-parallel shard (contiguous q/k/v block per device).

    qkv_w: [key_dim*2 + value_dim, hidden]."""
    q_part = qkv_w[:key_dim, :]
    k_part = qkv_w[key_dim : 2 * key_dim, :]
    v_part = qkv_w[2 * key_dim :, :]

    q_per = nk // tp
    v_per = nv // tp
    shards = []
    for s in range(tp):
        q_s = q_part[s * q_per * dk : (s + 1) * q_per * dk, :]
        k_s = k_part[s * q_per * dk : (s + 1) * q_per * dk, :]
        v_s = v_part[s * v_per * dv : (s + 1) * v_per * dv, :]
        shards.append(torch.cat([q_s, k_s, v_s], dim=0))
    return torch.cat(shards, dim=0)


def prepare_conv_taps(conv_w, key_dim, nk, dk, nv, dv, kernel_size, tp):
    """Split fused conv1d into kernel taps, reordered to match prepare_gdn_qkv grouping."""
    cw = conv_w.float()
    q_per = nk // tp
    v_per = nv // tp
    taps = []
    for j in range(kernel_size):
        tap = cw[:, 0, j]
        q_tap = tap[:key_dim]
        k_tap = tap[key_dim : 2 * key_dim]
        v_tap = tap[2 * key_dim :]
        shards = []
        for s in range(tp):
            q_s = q_tap[s * q_per * dk : (s + 1) * q_per * dk]
            k_s = k_tap[s * q_per * dk : (s + 1) * q_per * dk]
            v_s = v_tap[s * v_per * dv : (s + 1) * v_per * dv]
            shards.append(torch.cat([q_s, k_s, v_s]))
        taps.append(torch.cat(shards))
    return taps
