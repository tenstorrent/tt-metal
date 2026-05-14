# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""
TT self-attention for Devstral-2 123B (Hugging Face ``Ministral3Attention``).

Per ``reference/model_structure.txt``, each decoder layer has::

    self_attn: Ministral3Attention(
        q_proj: Linear(12288 -> 12288, bias=False)
        k_proj: Linear(12288 -> 1024, bias=False)
        v_proj: Linear(12288 -> 1024, bias=False)
        o_proj: Linear(12288 -> 12288, bias=False)
    )

That is the same *pattern* as Devstral-Small-2 / Ministral3 (GQA, separate Q/K/V/O linears). HF
``forward`` applies RoPE to Q/K, then multiplies Q by ``get_llama_4_attn_scale`` from
``rope_parameters`` (``llama_4_scaling_beta``, ``original_max_position_embeddings``), then
standard grouped attention and ``o_proj`` — see
``transformers.models.ministral3.modeling_ministral3.Ministral3Attention``.

:class:`models.experimental.devstarl2_small.tt.tt_ministralattn.TtMinistralAttention` already
implements that path on top of :class:`~models.tt_transformers.tt.attention.Attention` (post-RoPE
Q scaling on device, ``position_ids`` on prefill). This module subclasses that class for
``devstral2_large`` bring-up and keeps the **runtime class name** ``Attention`` so meta /
``layers.{i}.attention.*`` weight keys resolve like production Llama-family attention.

On **multi-device** meshes (non-Galaxy) with ~12k hidden, width-sharded DRAM ``TilizeDeviceOperation``
for ``wqkv`` / ``wo`` can exceed static L1 circular-buffer limits on **Blackhole** and **Wormhole T3K**.
:class:`TtDevstral2LargeAttention` forces **interleaved DRAM** for those uploads (same mitigation as
:class:`~models.experimental.devstral2_large.tt.tt_ministralmlp.TtDevstral2LargeMLP` weight tilize).

Prefill ``wo`` (:meth:`~models.tt_transformers.tt.attention.Attention.forward_prefill` ``ttnn.linear``)
uses ``ModelArgs.get_attn_wo_program_config``. For ``MatmulMultiCoreReuseMultiCastProgramConfig``, the
TTNN constructor defaults **omitted** ``out_block_h`` / ``out_block_w`` to ``per_core_M`` /
``per_core_N`` (see ``matmul_nanobind.cpp``). For wide 12k ``wo`` that makes ``out_block_w`` ≈ 48
tiles and overflows L1 for circular buffers on wide multi-device paths — unrelated to weight
**sharding** (DRAM width shard is already correct; the issue is per-op **output block** sizing). This
class wraps ``get_attn_wo_program_config`` and supplies minimal ``out_block_*`` and ``out_subblock_w``
for **prefill** only when :func:`~models.experimental.devstral2_large.tt.device_dram_mitigation.devstral2_large_multi_device_dram_mitigation`
is active (conceptually similar to :class:`TtDevstral2LargeRMSNorm` using a tight multicore norm config).

Activation row count for ``program_config`` still comes from shared ``Attention.forward_prefill``.
"""

from __future__ import annotations

import ttnn

from models.experimental.devstral2_large.tt.device_dram_mitigation import (
    devstral2_large_multi_device_dram_mitigation,
)
from models.experimental.devstarl2_small.tt.tt_ministralattn import TtMinistralAttention as _TtMinistralAttentionBase
from models.tt_transformers.tt.common import Mode


def _tighten_wo_prefill_prog_cfg(cfg):
    """Force minimal output block sizes for WO prefill matmul on wide multi-device paths.

    If ``out_block_h`` / ``out_block_w`` are not passed to ``MatmulMultiCoreReuseMultiCastProgramConfig``,
    TTNN sets them to ``per_core_M`` / ``per_core_N``. For ~12k-wide WO, ``out_block_w`` then matches
    ``per_core_N`` (~48 tiles) and static CBs overlap reserved L1 — **not** fixed by sequence chunking
    alone. Supply explicit 1×1 output blocks and minimal subblock width.
    """
    if cfg is None:
        return cfg
    mc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig
    if not isinstance(cfg, mc):
        return cfg
    return mc(
        compute_with_storage_grid_size=cfg.compute_with_storage_grid_size,
        in0_block_w=cfg.in0_block_w,
        out_subblock_h=cfg.out_subblock_h,
        out_subblock_w=1,
        out_block_h=1,
        out_block_w=1,
        per_core_M=cfg.per_core_M,
        per_core_N=cfg.per_core_N,
        transpose_mcast=cfg.transpose_mcast,
        fused_activation=cfg.fused_activation,
        fuse_batch=cfg.fuse_batch,
    )


def _widen_qkv_prefill_in0_block_w(cfg, in0_block_w: int = 2):
    """Increase in0_block_w for QKV prefill matmul on Devstral-2-123B.

    The shared config defaults to in0_block_w=1 for seq_len<=128. For K=12288 that means
    384 inner-loop iterations per core; doubling to 2 halves the loop overhead. Only touches
    MatmulMultiCoreReuseMultiCastProgramConfig (seq_len<=128 path); MinimalMatmulConfig
    (seq_len>128) is returned unchanged.
    """
    mc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig
    if not isinstance(cfg, mc):
        return cfg
    return mc(
        compute_with_storage_grid_size=cfg.compute_with_storage_grid_size,
        in0_block_w=in0_block_w,
        out_subblock_h=cfg.out_subblock_h,
        out_subblock_w=cfg.out_subblock_w,
        per_core_M=cfg.per_core_M,
        per_core_N=cfg.per_core_N,
        transpose_mcast=cfg.transpose_mcast,
        fused_activation=cfg.fused_activation,
        fuse_batch=cfg.fuse_batch,
    )


def _multi_device_dram_weight_tilize(mesh_device, configuration) -> bool:
    return configuration is not None and devstral2_large_multi_device_dram_mitigation(mesh_device, configuration)


class TtDevstral2LargeAttention(_TtMinistralAttentionBase):
    """
    Ministral3-style attention for Devstral-2 123B on Tenstorrent.

    Pass ``llama_4_scaling_beta`` and ``original_max_position_embeddings`` from
    ``Ministral3Config.rope_parameters`` (same kwargs as :class:`TtMinistralAttention`).
    """

    def __init__(
        self,
        *args,
        llama_4_scaling_beta: float | None = None,
        original_max_position_embeddings: int | None = None,
        **kwargs,
    ):
        mesh_device = args[0] if len(args) > 0 else kwargs.get("mesh_device")
        configuration = kwargs.get("configuration")
        orig_create_dram_sharded = None
        patch_dram_tilize = _multi_device_dram_weight_tilize(mesh_device, configuration)
        if patch_dram_tilize:
            orig_create_dram_sharded = configuration.create_dram_sharded_mem_config

            def _interleaved_dram_for_tilize(_m, _n):
                return ttnn.DRAM_MEMORY_CONFIG

            configuration.create_dram_sharded_mem_config = _interleaved_dram_for_tilize  # type: ignore[method-assign]
        try:
            super().__init__(
                *args,
                llama_4_scaling_beta=llama_4_scaling_beta,
                original_max_position_embeddings=original_max_position_embeddings,
                **kwargs,
            )
        finally:
            if patch_dram_tilize and orig_create_dram_sharded is not None:
                configuration.create_dram_sharded_mem_config = orig_create_dram_sharded  # type: ignore[method-assign]

        # Without this patch, prefill WO keeps TTNN defaults: out_block_h=per_core_M (often 1),
        # out_block_w=per_core_N (~48) → L1 CB clash on wide multi-device (BH or WH T3K).
        if devstral2_large_multi_device_dram_mitigation(mesh_device, getattr(self, "args", None)):
            _base_get_attn_wo = self.args.get_attn_wo_program_config

            def _get_attn_wo_wide_prefill(mode, seq_len=1, prefetcher=None):
                cfg = _base_get_attn_wo(mode, seq_len, prefetcher)
                if mode != Mode.PREFILL:
                    return cfg
                return _tighten_wo_prefill_prog_cfg(cfg)

            self.args.get_attn_wo_program_config = _get_attn_wo_wide_prefill  # type: ignore[method-assign]

        # Widen in0_block_w for QKV prefill: shared config defaults to 1 (K=12288 → 384 loop
        # iterations). in0_block_w=2 halves that, improving compute pipeline utilization.
        # Bind the real ModelArgs method before reassignment; use a distinct closure cell name so
        # we never recurse if this block is edited alongside other get_* patches.
        _base_get_attn_qkv = self.args.get_attn_qkv_program_config

        def _get_attn_qkv_wide_prefill(mode, seq_len=1, prefetcher=None):
            cfg = _base_get_attn_qkv(mode, seq_len, prefetcher)
            if mode != Mode.PREFILL:
                return cfg
            return _widen_qkv_prefill_in0_block_w(cfg)

        self.args.get_attn_qkv_program_config = _get_attn_qkv_wide_prefill  # type: ignore[method-assign]


TtDevstral2LargeAttention.__name__ = "Attention"
TtDevstral2LargeAttention.__qualname__ = "Attention"

TtMinistralAttention = TtDevstral2LargeAttention

__all__ = [
    "TtDevstral2LargeAttention",
    "TtMinistralAttention",
]
