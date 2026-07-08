# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Block-sharded (BS) Tier-1 optimization layer for the pi0.5 streamed-denoise port.

VENDORED from ``tt_symbiote.models.pi05.modeling_pi05_bs`` with imports rewired to the
local ``.common`` / ``.pcfg`` modules. The BS optimizations are the MAIN (only) path --
always on; the helpers no-op-fall-back (return None -> caller uses core_grid/interleaved)
when no clean grid divides a shape. ZERO tt_symbiote imports.
"""
from __future__ import annotations

import ttnn

from .common import sdpa_prefill_chunk_sizes
from .pcfg import _RMS_NORM_COMPUTE_CONFIG, build_matmul_pcfg, build_sharded_norm_pcfg

TT_METAL_COMMIT = "58672b47cfd304195798bcf34d44f5dbcbcf5189"

_ENABLED = True


def bs_enabled() -> bool:
    return _ENABLED


def _rms_compute():
    return _RMS_NORM_COMPUTE_CONFIG


def matmul_pcfg(m_tiles, k_tiles, n_tiles, grid_x, grid_y, **kw):
    if not bs_enabled():
        return None
    try:
        return build_matmul_pcfg(m_tiles, k_tiles, n_tiles, grid_x, grid_y, **kw)
    except Exception:
        return None


def sharded_norm_pcfg(m_tiles, hidden_tiles, *, max_grid_x=8, max_grid_y=8):
    if not bs_enabled():
        return None
    try:
        return build_sharded_norm_pcfg(m_tiles, hidden_tiles, max_grid_x=max_grid_x, max_grid_y=max_grid_y)
    except Exception:
        return None


def sharded_rms_norm(x, weight, eps, m_padded, hidden, *, batch=1, bias=None, out_block_sharded=False):
    """Sharded RMSNorm with a pre-offset ``weight`` (Gemma ``w+1``), INTERLEAVED-L1 result.
    Optional ``bias`` (adaRMS fused modulation) is added post-norm inside the kernel.

    When ``out_block_sharded`` is True, the trailing ``sharded_to_interleaved`` is SKIPPED and the
    block-sharded ``normed`` (memory_config == ``memcfg``) is returned directly -- the downstream
    consumer (matmul_decode with ``reshard_input=True``) reshards it internally, so the S2I and the
    interleaved intermediate are both eliminated. Falls back to interleaved if no sharded pcfg."""
    m_tiles = m_padded // 32
    cfg = sharded_norm_pcfg(m_tiles, hidden // 32, max_grid_x=8, max_grid_y=min(8, max(1, m_tiles)))
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
        compute_kernel_config=_rms_compute(),
        memory_config=memcfg,
    )
    ttnn.deallocate(x_sh)
    if out_block_sharded:
        return normed  # block-sharded (memcfg); fed straight to matmul_decode(reshard_input=True)
    out = ttnn.sharded_to_interleaved(normed, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(normed)
    return out


def sdpa_program_config(seq_q, seq_kv, grid_x, grid_y, *, q_chunk=None, k_chunk=None):
    """SDPAProgramConfig with tuned (divisor-aware) q/k chunk sizes, or None to fall back."""
    if not _ENABLED:
        return None
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


def sharded_layer_norm(x, weight, bias, eps, m_padded, hidden, *, batch=1):
    """Sharded LayerNorm (affine), INTERLEAVED-L1 result; falls back to plain interleaved."""
    m_tiles = m_padded // 32
    cfg = sharded_norm_pcfg(m_tiles, hidden // 32, max_grid_x=8, max_grid_y=min(8, max(1, m_tiles)))
    if cfg is None:
        return ttnn.layer_norm(x, weight=weight, bias=bias, epsilon=eps, memory_config=ttnn.L1_MEMORY_CONFIG)
    pc, memcfg_factory, _grid = cfg
    memcfg = memcfg_factory(batch, m_padded, m_padded, hidden)
    x_sh = ttnn.to_memory_config(x, memcfg)
    normed = ttnn.layer_norm(
        x_sh,
        weight=weight,
        bias=bias,
        epsilon=eps,
        program_config=pc,
        compute_kernel_config=_rms_compute(),
        memory_config=memcfg,
    )
    ttnn.deallocate(x_sh)
    out = ttnn.sharded_to_interleaved(normed, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(normed)
    return out
