# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Single pi0.5 denoise decoder LAYER as a faithful DEVICE-TIME proxy of production.

This is a SELF-CONTAINED reference (deep-plan_1): an INDEPENDENT ttnn op sequence with
no external model-package imports. The production config builders (matmul_pcfg, sharded_rms_norm,
get_sdpa_compute_kernel_config, _denoise_sdpa_pcfg/_denoise_tuned_pcfg) are VENDORED
verbatim below so the config VALUES match the production ``TTNNPi05DenoiseExpertBlock``
op-for-op (same program configs, fidelities, core grids). The layer is measured against
the production single-layer device-time target on a SINGLE Blackhole chip (1x1 mesh,
ZERO D2D), matched within production's measured tracy run-to-run noise band, with
PCC >= 0.99 preserved.

Layer (adaRMS / adaLN-Zero modulated decoder block, width=1024, mlp_dim=4096,
num_heads=8, num_kv_heads=1 (GQA), head_dim=256, M=32 suffix tokens attending to a
1024-token VLM prefix KV, KV=1056):

    h    = rms_norm_nogamma(x) * (1 + scale_a) + shift_a     # FUSED sharded adaRMS, no learned gamma
    qkv  = h @ Wqkv (bf8_b out) ; split q[8]/k[1]/v[1] ; RoPE(q), RoPE(k)
    k,v  = concat(prefix_kv (bf8_b), suffix_kv)              # 1024 + 32 = 1056 keys
    a    = nlp_concat_heads( SDPA(q, k, v, mask) ) @ Wo (bf16 out)
    x    = x + gate_a * a
    h2   = rms_norm_nogamma(x) * (1 + scale_m) + shift_m
    m    = down( gelu_tanh(gate(h2)) * up(h2) )              # GeGLU MLP, FUSED gelu into gate matmul
    x    = x + gate_m * m

Three MLP modes are exercised. ONLY the MLP path varies across them -- the
adaRMS / QKV / RoPE / concat-KV / SDPA / o-proj / gated-residual ops are
byte-identical (this is the whole point of the comparison):

  * ``linear``        -- production default (ttnn.linear gate/up/down, fused
                         tanh-gelu into the gate matmul, 8x8 matmul_pcfg, LoFi default).
  * ``decode``        -- MLP via ttnn.matmul_decode (partial-width-sharded resident-L1
                         weights, reshard-before-down, HiFi2 gate/up, LoFi down).
  * ``decode_fused``  -- same as decode but with ``fused_gelu=True`` on the gate matmul.

PCC gate (eager, no tracy), parametrized over all three modes:
    pytest tests/ttnn/unit_tests/operations/matmul/test_denoise_layer_matmul_decode.py -k pcc -x -s

Timing (tracy traced-replay, opens its OWN 1x1 mesh); mode via PI05_SL_MODE env:
    PI05_SL_MODE=decode python -m tracy -p -r -v --op-support-count 20000 -m \
      "pytest tests/.../test_denoise_layer_matmul_decode.py -k timing -s"
"""

import os

import pytest
import torch
import torch.nn.functional as F

import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_and_get_pcc
from tests.ttnn.utils_for_testing import assert_with_pcc


def _pcc(golden, calculated):
    """Return the numeric PCC value (comp_and_get_pcc returns (passed, str, cal_pcc))."""
    _, _, cal_pcc = comp_and_get_pcc(golden, calculated, pcc=0.0)
    return float(cal_pcc)


# =========================================================================== #
# SELF-CONTAINED production config builders -- no external model-package import.
#
# Vendored VERBATIM from the production pi0.5 forward path (source commit
# c8582f1d2d32fa8f02537bfd45776acfffff1a61) so this test has no external model
# dependency while emitting byte-identical program configs / compute-kernel
# configs to the production ``TTNNPi05DenoiseExpertBlock``. Sources:
#   * build_matmul_pcfg / build_sharded_norm_pcfg / _RMS_NORM_COMPUTE_CONFIG
#         <- models/pi05/modeling_pi05_pcfg.py
#   * matmul_pcfg / sharded_rms_norm / sdpa_program_config
#         <- models/pi05/modeling_pi05_bs.py
#   * sdpa_prefill_chunk_sizes / get_sdpa_math_fidelity / get_sdpa_compute_kernel_config
#         <- models/pi05/modeling_pi05_common.py
#   * _denoise_tuned_pcfg / _denoise_sdpa_pcfg
#         <- models/pipelined_pi05/denoise_block.py
# Env flags are honored identically (PI0_DENOISE_MM_TUNE, PI0_MM_SWEEP_V2,
# LADDER_SDPA_HIFI/FP32/PACKER, PI0_SDPA_DENOISE_K_FORCE). Re-vendor + bump the
# commit if the production builders change.
# =========================================================================== #

_pcfg_cache: dict = {}
_sharded_norm_cache: dict = {}

_RMS_NORM_COMPUTE_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)


def _flag(name):
    return os.environ.get(name, "").lower() in ("1", "true", "yes", "on")


def build_matmul_pcfg(m_tiles, k_tiles, n_tiles, grid_x, grid_y, *, in0_block_w=None, activation=None, dst_budget=8):
    """2D-block / 1D-width-shard matmul program config (or None to fall back)."""
    if m_tiles == 0 or k_tiles == 0 or n_tiles == 0:
        return None
    total_cores = grid_x * grid_y

    # --- 1D width-shard path: small M, big N ---
    if m_tiles * 4 <= grid_y and n_tiles >= total_cores // 4:
        num_cores = min(total_cores, n_tiles)
        while num_cores > total_cores // 2 and n_tiles % num_cores != 0:
            num_cores -= 1
        if n_tiles % num_cores != 0:
            num_cores = total_cores
            per_core_N_1d = (n_tiles + num_cores - 1) // num_cores
        else:
            per_core_N_1d = n_tiles // num_cores

        if in0_block_w is None:
            in0_bw = 16
        else:
            in0_bw = in0_block_w
        while in0_bw > 1 and in0_bw * per_core_N_1d > 32:
            in0_bw //= 2
        while k_tiles % in0_bw != 0 and in0_bw > 1:
            in0_bw //= 2
        if in0_bw < 2:
            in0_bw = 1

        out_subblock_w_1d = min(per_core_N_1d, dst_budget)
        while out_subblock_w_1d > 1 and per_core_N_1d % out_subblock_w_1d != 0:
            out_subblock_w_1d -= 1
        out_subblock_h_1d = max(1, dst_budget // out_subblock_w_1d)
        out_subblock_h_1d = min(m_tiles, out_subblock_h_1d)
        while out_subblock_h_1d > 1 and m_tiles % out_subblock_h_1d != 0:
            out_subblock_h_1d -= 1

        cfg_gx = min(grid_x, num_cores)
        cfg_gy = (num_cores + cfg_gx - 1) // cfg_gx

        key = (m_tiles, k_tiles, n_tiles, grid_x, grid_y, str(activation), dst_budget, "1d")
        if key in _pcfg_cache:
            return _pcfg_cache[key]
        cfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(cfg_gx, cfg_gy),
            in0_block_w=in0_bw,
            out_subblock_h=out_subblock_h_1d,
            out_subblock_w=out_subblock_w_1d,
            per_core_M=m_tiles,
            per_core_N=per_core_N_1d,
            fuse_batch=True,
            fused_activation=activation,
            mcast_in0=True,
        )
        _pcfg_cache[key] = cfg
        return cfg

    # --- 2D block-shard path (default, large M) ---
    per_core_M = (m_tiles + grid_y - 1) // grid_y
    per_core_N = (n_tiles + grid_x - 1) // grid_x
    if per_core_M == 0 or per_core_N == 0:
        return None

    if in0_block_w is None:
        if per_core_N <= 12:
            in0_block_w = 8
        else:
            in0_block_w = 4
    while k_tiles % in0_block_w != 0 and in0_block_w > 1:
        in0_block_w //= 2
    if in0_block_w == 1 and k_tiles > 32:
        return None

    key = (m_tiles, k_tiles, n_tiles, grid_x, grid_y, str(activation), dst_budget)
    if key in _pcfg_cache:
        return _pcfg_cache[key]

    out_subblock_w = min(per_core_N, dst_budget)
    while out_subblock_w > 1 and per_core_N % out_subblock_w != 0:
        out_subblock_w -= 1
    out_subblock_h_budget = max(1, dst_budget // out_subblock_w)
    out_subblock_h = min(per_core_M, out_subblock_h_budget)
    while out_subblock_h > 1 and per_core_M % out_subblock_h != 0:
        out_subblock_h -= 1

    cfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=activation,
    )
    _pcfg_cache[key] = cfg
    return cfg


def build_sharded_norm_pcfg(m_tiles, hidden_tiles, *, max_grid_x=8, max_grid_y=8):
    """(program_config, sharded_memory_config_factory, grid) for sharded RMS/LayerNorm, or None."""
    key = (m_tiles, hidden_tiles, max_grid_x, max_grid_y)
    if key in _sharded_norm_cache:
        return _sharded_norm_cache[key]

    cand_y = [y for y in range(min(max_grid_y, m_tiles), 0, -1) if m_tiles % y == 0]
    cand_x = [x for x in range(min(max_grid_x, hidden_tiles), 0, -1) if hidden_tiles % x == 0]
    if not cand_y or not cand_x:
        _sharded_norm_cache[key] = None
        return None

    best = None
    best_cores = 0
    for gy in cand_y:
        for gx in cand_x:
            cores = gx * gy
            if cores > best_cores or (cores == best_cores and gx > best[0]):
                best = (gx, gy)
                best_cores = cores
    if best is None:
        _sharded_norm_cache[key] = None
        return None

    gx, gy = best
    block_h = m_tiles // gy
    block_w = hidden_tiles // gx
    subblock_w = block_w

    pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        subblock_w=subblock_w,
        block_h=block_h,
        block_w=block_w,
        inplace=False,
    )

    def make_memcfg(b, m_logical, m_physical, hidden):
        return ttnn.create_sharded_memory_config(
            (b, 1, m_physical, hidden),
            core_grid=ttnn.CoreGrid(y=gy, x=gx),
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )

    grid = ttnn.CoreGrid(y=gy, x=gx)
    result = (pc, make_memcfg, grid)
    _sharded_norm_cache[key] = result
    return result


def matmul_pcfg(m_tiles, k_tiles, n_tiles, grid_x, grid_y, **kw):
    try:
        return build_matmul_pcfg(m_tiles, k_tiles, n_tiles, grid_x, grid_y, **kw)
    except Exception:
        return None


def _sharded_norm_pcfg(m_tiles, hidden_tiles, *, max_grid_x=8, max_grid_y=8):
    try:
        return build_sharded_norm_pcfg(m_tiles, hidden_tiles, max_grid_x=max_grid_x, max_grid_y=max_grid_y)
    except Exception:
        return None


def sharded_rms_norm(x, weight, eps, m_padded, hidden, *, batch=1, bias=None):
    """Sharded RMSNorm with pre-offset weight (1+scale) and optional fused bias (shift),
    returning an INTERLEAVED-L1 result. Falls back to plain ttnn.rms_norm when no clean
    grid divides the shape."""
    m_tiles = m_padded // 32
    cfg = _sharded_norm_pcfg(m_tiles, hidden // 32, max_grid_x=8, max_grid_y=min(8, max(1, m_tiles)))
    if cfg is None:
        return ttnn.rms_norm(x, weight=weight, bias=bias, epsilon=eps, memory_config=ttnn.L1_MEMORY_CONFIG)
    pc, memcfg_factory, _grid = cfg
    memcfg = memcfg_factory(batch, m_padded, m_padded, hidden)
    x_sh = ttnn.to_memory_config(x, memcfg)
    normed = ttnn.rms_norm(
        x_sh,
        weight=weight,
        bias=bias,
        epsilon=eps,
        program_config=pc,
        compute_kernel_config=_RMS_NORM_COMPUTE_CONFIG,
        memory_config=memcfg,
    )
    ttnn.deallocate(x_sh)
    out = ttnn.sharded_to_interleaved(normed, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(normed)
    return out


def sdpa_prefill_chunk_sizes(seq_len_q, seq_len_kv, *, tile=32):
    """q_chunk / k_chunk sizes for ttnn SDPA, mirroring the tt_transformers baseline."""
    longest = max(seq_len_q, seq_len_kv)
    if longest >= 2048:
        base_q, base_k = 256, 256
    elif longest >= 512:
        base_q, base_k = 64, 128
    else:
        base_q, base_k = 64, 64
    q_aligned = ((seq_len_q + tile - 1) // tile) * tile if seq_len_q > 0 else tile
    k_aligned = ((seq_len_kv + tile - 1) // tile) * tile if seq_len_kv > 0 else tile
    return max(min(base_q, q_aligned), tile), max(min(base_k, k_aligned), tile)


def get_sdpa_math_fidelity():
    if os.environ.get("LADDER_SDPA_HIFI") == "2":
        return ttnn.MathFidelity.HiFi2
    return ttnn.MathFidelity.HiFi4


def get_sdpa_compute_kernel_config():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=get_sdpa_math_fidelity(),
        math_approx_mode=False,
        fp32_dest_acc_en=(os.environ.get("LADDER_SDPA_FP32") == "1"),
        packer_l1_acc=(os.environ.get("LADDER_SDPA_PACKER", "1") != "0"),
    )


def sdpa_program_config(seq_q, seq_kv, grid_x, grid_y, *, q_chunk=None, k_chunk=None):
    try:
        qc, kc = sdpa_prefill_chunk_sizes(seq_q, seq_kv)
        if q_chunk is not None:
            qc = q_chunk
        if k_chunk is not None:
            kc = k_chunk
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(grid_x, grid_y),
            q_chunk_size=qc,
            k_chunk_size=kc,
            exp_approx_mode=False,
        )
    except Exception:
        return None


_DENOISE_TUNE_TABLE_BASE = {(64, 32): (120, 32), (128, 32): (24, 32)}
_DENOISE_TUNE_TABLE_V2 = {(64, 32): (120, 32), (128, 32): (24, 32), (32, 80): (64, 8)}


def _denoise_tuned_pcfg(m_tiles, k_tiles, n_tiles, grid_x, grid_y, *, activation=None):
    if m_tiles != 1 or not _flag("PI0_DENOISE_MM_TUNE"):
        return None
    table = _DENOISE_TUNE_TABLE_V2 if _flag("PI0_MM_SWEEP_V2") else _DENOISE_TUNE_TABLE_BASE
    override = table.get((k_tiles, n_tiles))
    if override is None:
        return None
    num_cores, in0_bw = override
    if k_tiles % in0_bw != 0:
        return None
    per_core_N = (n_tiles + num_cores - 1) // num_cores if n_tiles % num_cores else n_tiles // num_cores
    eff_budget = 4
    out_sw = min(per_core_N, eff_budget)
    while out_sw > 1 and per_core_N % out_sw != 0:
        out_sw -= 1
    out_sh = max(1, eff_budget // out_sw)
    out_sh = min(m_tiles, out_sh)
    while out_sh > 1 and m_tiles % out_sh != 0:
        out_sh -= 1
    cfg_gx = min(grid_x, num_cores)
    cfg_gy = (num_cores + cfg_gx - 1) // cfg_gx
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(cfg_gx, cfg_gy),
        in0_block_w=in0_bw,
        out_subblock_h=out_sh,
        out_subblock_w=out_sw,
        per_core_M=m_tiles,
        per_core_N=per_core_N,
        fuse_batch=True,
        fused_activation=activation,
        mcast_in0=True,
    )


def _denoise_sdpa_pcfg(q_seq, kv_seq, grid_x, grid_y):
    q_chunk = 32
    k_force = os.environ.get("PI0_SDPA_DENOISE_K_FORCE", "").strip()
    k_chunk = None
    kv_aligned = ((kv_seq + 31) // 32) * 32
    if k_force:
        kf = int(k_force)
        if kv_aligned % kf == 0:
            k_chunk = kf
    if k_chunk is None:
        for cand in (256, 128, 96, 64, 32):
            if kv_aligned % cand == 0:
                k_chunk = cand
                break
        k_chunk = k_chunk or 32
    return sdpa_program_config(q_seq, kv_seq, grid_x, grid_y, q_chunk=q_chunk, k_chunk=k_chunk)


try:
    from tracy import signpost
except Exception:  # pragma: no cover

    def signpost(*a, **k):
        pass


# --------------------------------------------------------------------------- shape constants
W = 1024
MLP_DIM = 4096
NH, NKV, HD = 8, 1, 256
QKV_OUT = NH * HD + 2 * NKV * HD  # 2560
M = 32  # suffix tokens (1 tile)
AH = 10  # real action tokens (rest of M is padding)
PREFIX = 1024  # VLM prefix KV length
KV = PREFIX + M  # 1056
EPS = 1e-6
SCALE = HD**-0.5
PCC = 0.99
SEED = 0
MT = M // 32  # 1

_L1, _DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG

# --------------------------------------------------------------------------- decode-MLP constants (ported from git HEAD)
# These are used ONLY by the decode / decode_fused MLP branch -- the rest of the
# layer is byte-identical to the linear production-matched path above.
K_BLOCKS, N_BLOCKS, RESHARD_CORES = 2, 32, 2
_HIFI2 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False
)
_LOFI = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False
)


# --------------------------------------------------------------------------- weights / inputs
def _make_layer(seed=SEED):
    torch.manual_seed(seed)
    g = lambda *s: torch.randn(*s, dtype=torch.float32)
    return {
        "wqkv": g(QKV_OUT, W) * 0.02,  # (out, in)
        "wo": g(W, NH * HD) * 0.02,
        "gate": g(MLP_DIM, W) * 0.02,
        "up": g(MLP_DIM, W) * 0.02,
        "down": g(W, MLP_DIM) * 0.02,
        # adaLN-Zero modulation (precomputed from the timestep cond once per inference).
        # No learned RMS gamma -- production folds (1+scale)/shift into the fused norm op.
        "scale_a": g(1, W) * 0.1,
        "shift_a": g(1, W) * 0.1,
        "gate_a": g(1, W) * 0.1,
        "scale_m": g(1, W) * 0.1,
        "shift_m": g(1, W) * 0.1,
        "gate_m": g(1, W) * 0.1,
    }


def _inputs(seed=SEED + 1):
    torch.manual_seed(seed)
    x = torch.randn(M, W)
    cos = torch.randn(1, 1, M, HD)  # arbitrary RoPE tables (ttnn applies x*cos+rotate_half(x)*sin)
    sin = torch.randn(1, 1, M, HD)
    prefix_k = torch.randn(1, NKV, PREFIX, HD) * 0.1
    prefix_v = torch.randn(1, NKV, PREFIX, HD) * 0.1
    mask = torch.zeros(1, 1, M, KV)
    mask[:, :, :, PREFIX + AH :] = -1e9  # mask the padded suffix key positions
    return x, cos, sin, prefix_k, prefix_v, mask


# --------------------------------------------------------------------------- torch reference (regenerated golden)
def _rmsnorm_nogamma(x):
    # NO learned gamma -- matches production _apply_ada (fused norm weight = 1+scale, bias = shift).
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _rope(x, cos, sin):
    return x * cos + _rotate_half(x) * sin


def _reference(wts, x, cos, sin, pk, pv, mask):
    c, s = cos[0, 0], sin[0, 0]  # [M, HD]
    # adaRMS #1: no learned gamma, (1+scale_a) and shift_a are the fused norm weight/bias.
    h = _rmsnorm_nogamma(x) * (1 + wts["scale_a"]) + wts["shift_a"]
    qkv = h @ wts["wqkv"].t()  # [M, 2560]
    q = qkv[:, : NH * HD].view(M, NH, HD).permute(1, 0, 2)  # [NH, M, HD]
    k = qkv[:, NH * HD : (NH + NKV) * HD].view(M, NKV, HD).permute(1, 0, 2)
    v = qkv[:, (NH + NKV) * HD :].view(M, NKV, HD).permute(1, 0, 2)
    q = _rope(q, c, s)
    k = _rope(k, c, s)
    # prefix-RoPE parity: prefix KV is fed un-RoPE'd to BOTH device and golden (pin §9.3).
    k = torch.cat([pk[0], k], dim=1)  # [NKV, KV, HD]
    v = torch.cat([pv[0], v], dim=1)
    k = k.repeat_interleave(NH // NKV, dim=0)  # GQA expand -> [NH, KV, HD]
    v = v.repeat_interleave(NH // NKV, dim=0)
    scores = (q @ k.transpose(-1, -2)) * SCALE + mask[0]  # [NH, M, KV]
    a = torch.softmax(scores, dim=-1) @ v  # [NH, M, HD]
    a = a.permute(1, 0, 2).reshape(M, NH * HD)  # [M, 2048]
    x = x + wts["gate_a"] * (a @ wts["wo"].t())
    # adaRMS #2
    h2 = _rmsnorm_nogamma(x) * (1 + wts["scale_m"]) + wts["shift_m"]
    # GeGLU with tanh-approx GELU (production fuses (GELU, True) = tanh-approx into the gate matmul).
    m = (F.gelu(h2 @ wts["gate"].t(), approximate="tanh") * (h2 @ wts["up"].t())) @ wts["down"].t()
    return x + wts["gate_m"] * m


# --------------------------------------------------------------------------- ttnn helpers
def _tt(device, t, dtype=ttnn.bfloat16, mem=_L1):
    return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype, memory_config=mem)


# --------------------------------------------------------------------------- decode-MLP helpers (ported verbatim from git HEAD)
def _crs(device, n):
    return ttnn.num_cores_to_corerangeset(n, device.compute_with_storage_grid_size(), True)


def _pws_B(device, w_kn, n_blocks):
    """Partial-width-sharded resident-L1 weight tensor for matmul_decode."""
    k, n = w_kn.shape
    kc, nc = k // K_BLOCKS, n // n_blocks
    br = w_kn.reshape(K_BLOCKS, kc, n).permute(1, 0, 2).reshape(kc, n * K_BLOCKS)
    mc = ttnn.create_sharded_memory_config(
        (kc, nc),
        core_grid=_crs(device, K_BLOCKS * n_blocks),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    return ttnn.from_torch(br, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc, dtype=ttnn.bfloat8_b)


# --------------------------------------------------------------------------- ttnn layer (Option C: independent ops, imported configs)
def _build_layer(device, wts, ins, mode="linear"):
    """Upload weights once; return a run() closure that does the full layer forward
    using the production config builders. ``mode`` in {linear, decode, decode_fused}
    controls ONLY the MLP path -- every other op is byte-identical across modes."""
    x_t, cos_t, sin_t, pk_t, pv_t, mask_t = ins
    decode = mode != "linear"
    fused = mode == "decode_fused"

    # static tensors (bound OUTSIDE any trace capture)
    x0 = _tt(device, x_t.view(1, 1, M, W))  # bf16 L1
    # fused-adaRMS modulation: weight = (1+scale), bias = shift (bf16, DRAM). gate stays L1-resident.
    sa1 = _tt(device, (1.0 + wts["scale_a"]).view(1, 1, 1, W), mem=_DRAM)
    sha = _tt(device, wts["shift_a"].view(1, 1, 1, W), mem=_DRAM)
    gta = _tt(device, wts["gate_a"].view(1, 1, 1, W))
    sf1 = _tt(device, (1.0 + wts["scale_m"]).view(1, 1, 1, W), mem=_DRAM)
    shf = _tt(device, wts["shift_m"].view(1, 1, 1, W), mem=_DRAM)
    gtm = _tt(device, wts["gate_m"].view(1, 1, 1, W))
    # weights bf8_b, moved to L1 (production move_weights_to_device_impl re-homes these to L1)
    wqkv = ttnn.to_memory_config(_tt(device, wts["wqkv"].t().contiguous(), dtype=ttnn.bfloat8_b), _L1)
    wo = ttnn.to_memory_config(_tt(device, wts["wo"].t().contiguous(), dtype=ttnn.bfloat8_b), _L1)
    # MLP weights: linear keeps bf8_b L1-interleaved; decode uses partial-width-sharded resident-L1.
    if decode:
        gate_b = _pws_B(device, wts["gate"].t().contiguous(), N_BLOCKS)
        up_b = _pws_B(device, wts["up"].t().contiguous(), N_BLOCKS)
        down_b = _pws_B(device, wts["down"].t().contiguous(), N_BLOCKS)
        a_mc = ttnn.create_sharded_memory_config(
            (M, W // RESHARD_CORES),
            core_grid=_crs(device, RESHARD_CORES),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        mid_mc = ttnn.create_sharded_memory_config(
            (M, MLP_DIM // RESHARD_CORES),
            core_grid=_crs(device, RESHARD_CORES),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
    else:
        gw = ttnn.to_memory_config(_tt(device, wts["gate"].t().contiguous(), dtype=ttnn.bfloat8_b), _L1)
        uw = ttnn.to_memory_config(_tt(device, wts["up"].t().contiguous(), dtype=ttnn.bfloat8_b), _L1)
        dw = ttnn.to_memory_config(_tt(device, wts["down"].t().contiguous(), dtype=ttnn.bfloat8_b), _L1)
    cos = _tt(device, cos_t)
    sin = _tt(device, sin_t)
    # prefix KV bf8_b @ L1 (production _kv_dtype default; concat dtype-consistent with bf8_b suffix-K out)
    pk = _tt(device, pk_t, dtype=ttnn.bfloat8_b, mem=_L1)
    pv = _tt(device, pv_t, dtype=ttnn.bfloat8_b, mem=_L1)
    mask = _tt(device, mask_t, mem=_DRAM)

    def _mlp(normed2):
        """GeGLU MLP. Only branch that differs across modes."""
        if decode:
            # --- decode / decode_fused: matmul_decode, partial-width-sharded resident-L1 weights ---
            hw = ttnn.to_memory_config(normed2, a_mc)
            ttnn.deallocate(normed2)
            gate = ttnn.matmul_decode(
                hw, gate_b, partial_width_sharded=True, compute_kernel_config=_HIFI2, fused_gelu=fused
            )
            up = ttnn.matmul_decode(hw, up_b, partial_width_sharded=True, compute_kernel_config=_HIFI2)
            if not fused:
                gate = ttnn.gelu(gate, fast_and_approximate_mode=False)
            hid = ttnn.multiply(gate, up, memory_config=gate.memory_config())
            hid2 = ttnn.to_memory_config(hid, mid_mc)
            ows = ttnn.matmul_decode(hid2, down_b, partial_width_sharded=True, compute_kernel_config=_LOFI)
            out = ttnn.to_memory_config(ows, _L1)
            for t in (hw, gate, up, hid, hid2, ows):
                ttnn.deallocate(t)
            return out
        # --- linear (production default): 8x8 matmul_pcfg, FUSED tanh-gelu into gate matmul, LoFi default ck ---
        gate_pc = matmul_pcfg(MT, W // 32, MLP_DIM // 32, 8, 8, activation=(ttnn.UnaryOpType.GELU, True))
        up_pc = matmul_pcfg(MT, W // 32, MLP_DIM // 32, 8, 8)
        down_pc = matmul_pcfg(MT, MLP_DIM // 32, W // 32, 8, 8)
        gate = ttnn.linear(
            normed2, gw, dtype=None, memory_config=_L1, program_config=gate_pc, compute_kernel_config=None
        )
        up = ttnn.linear(normed2, uw, dtype=None, memory_config=_L1, program_config=up_pc, compute_kernel_config=None)
        ttnn.deallocate(normed2)
        hid = ttnn.multiply(gate, up, memory_config=_L1)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        out = ttnn.linear(hid, dw, dtype=None, memory_config=_L1, program_config=down_pc, compute_kernel_config=None)
        ttnn.deallocate(hid)
        return out

    def run():
        g = device.compute_with_storage_grid_size()
        # --- adaRMS #1 (fused sharded, no learned gamma) ---
        normed = sharded_rms_norm(x0, sa1, EPS, M, W, bias=sha)
        # --- QKV projection (bf8_b out, LoFi default via ck=None) ---
        qpc = _denoise_tuned_pcfg(MT, W // 32, QKV_OUT // 32, g.x, g.y) or matmul_pcfg(
            MT, W // 32, QKV_OUT // 32, g.x, g.y, in0_block_w=8
        )
        qkv = ttnn.linear(
            normed, wqkv, dtype=ttnn.bfloat8_b, memory_config=_L1, program_config=qpc, compute_kernel_config=None
        )
        ttnn.deallocate(normed)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=NH, num_kv_heads=NKV, transpose_k_heads=False, memory_config=_L1
        )
        ttnn.deallocate(qkv)
        q = ttnn.experimental.rotary_embedding(q, cos, sin, memory_config=_L1)
        k = ttnn.experimental.rotary_embedding(k, cos, sin, memory_config=_L1)
        kk = ttnn.concat([pk, k], dim=2, memory_config=_L1)
        vv = ttnn.concat([pv, v], dim=2, memory_config=_L1)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        # --- SDPA (HiFi4 via get_sdpa_compute_kernel_config; q_chunk=32, k_chunk=96) ---
        sdpa_cores = min(g.x, NH * ((q.shape[-2] + 31) // 32))
        spc = _denoise_sdpa_pcfg(q.shape[-2], kk.shape[-2], sdpa_cores, 1)
        a = ttnn.transformer.scaled_dot_product_attention(
            q,
            kk,
            vv,
            attn_mask=mask,
            is_causal=False,
            scale=SCALE,
            compute_kernel_config=get_sdpa_compute_kernel_config(),
            memory_config=_L1,
            **({"program_config": spc} if spc is not None else {}),
        )
        ttnn.deallocate(q)
        ttnn.deallocate(kk)
        ttnn.deallocate(vv)
        a = ttnn.experimental.nlp_concat_heads(a, memory_config=_L1)
        # --- o projection (bf16 out) ---
        opc = _denoise_tuned_pcfg(MT, (NH * HD) // 32, W // 32, g.x, g.y) or matmul_pcfg(
            MT, (NH * HD) // 32, W // 32, g.x, g.y, in0_block_w=8
        )
        o = ttnn.linear(a, wo, dtype=ttnn.bfloat16, memory_config=_L1, program_config=opc, compute_kernel_config=None)
        ttnn.deallocate(a)
        og = ttnn.multiply(gta, o, memory_config=_L1)
        x1 = ttnn.add(x0, og, memory_config=_L1)
        ttnn.deallocate(o)
        ttnn.deallocate(og)
        # --- adaRMS #2 ---
        normed2 = sharded_rms_norm(x1, sf1, EPS, M, W, bias=shf)
        # --- GeGLU MLP (the ONLY mode-varying region; bracketed by an MLP-only signpost pair) ---
        signpost(header="mlp")
        m = _mlp(normed2)
        signpost(header="mlp_end")
        mg = ttnn.multiply(gtm, m, memory_config=_L1)
        out = ttnn.add(x1, mg, memory_config=_L1)
        for t in (x1, m, mg):
            ttnn.deallocate(t)
        return out

    return run


# --------------------------------------------------------------------------- PCC test (eager, no tracy)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 134217728, "l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("mode", ["linear", "decode", "decode_fused"])
def test_denoise_layer_pcc(device, mode):
    """Full denoise layer must match the regenerated (mode-independent) torch golden.

    Only the MLP path varies across modes; the golden is the same math for all three.
    """
    wts = _make_layer()
    ins = _inputs()
    ref = _reference(wts, *ins)  # [M, W]

    run = _build_layer(device, wts, ins, mode)
    out = ttnn.to_torch(run()).float().reshape(M, W)
    pcc_val = _pcc(ref, out)
    print(
        f"\n[denoise LAYER {mode}] PCC = {pcc_val:.8f} (assert >= {PCC}); mask shape "
        f"{tuple(ins[5].shape)} masked idx [{PREFIX + AH}:{KV}]"
    )
    assert_with_pcc(ref, out, PCC)


# --------------------------------------------------------------------------- timing (tracy traced-replay)
def _capture_and_measure(device, reps, mode):
    """Trace-replay the full layer ``reps`` times, signposted by a per-mode region
    header (full layer) AND a nested MLP-only region (``mlp``/``mlp_end``, emitted
    inside run()). reps replays -> reps mlp regions; the parser sums all of them."""
    region = f"layer_{mode}"
    wts = _make_layer()
    ins = _inputs()
    run = _build_layer(device, wts, ins, mode)
    # warm-up / compile (eager, >=5 reps)
    for _ in range(6):
        ttnn.deallocate(run())
        ttnn.synchronize_device(device)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    out = run()  # persistent
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.synchronize_device(device)
    signpost(header=region)
    for _ in range(reps):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    signpost(header=f"{region}_end")
    ttnn.release_trace(device, tid)


def main():
    reps = int(os.environ.get("PI05_SL_REPS", "50"))
    mode = os.environ.get("PI05_SL_MODE", "linear")
    assert mode in ("linear", "decode", "decode_fused"), f"bad PI05_SL_MODE={mode!r}"
    print(f"[timing] mode={mode} reps={reps} region=layer_{mode}")
    dev = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), l1_small_size=24576, trace_region_size=134_217_728)
    try:
        _capture_and_measure(dev, reps, mode)
        print(f"[trace {mode}] capture+replay OK")
    finally:
        ttnn.close_mesh_device(dev)


def test_denoise_layer_linear_timing():
    """tracy entrypoint: opens its OWN 1x1 mesh and does traced-replay timing."""
    main()


if __name__ == "__main__":
    main()
