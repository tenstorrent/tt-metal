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

On **Blackhole** with **more than one device** (non-Galaxy), :class:`~models.tt_transformers.tt.attention.Attention`
uploads ``wqkv`` / ``wo`` via ``configuration.create_dram_sharded_mem_config`` → width-sharded DRAM.
That ``TilizeDeviceOperation`` path can exceed static L1 circular-buffer limits (~1.5 MiB) for
12k-class matrices. :class:`TtDevstral2LargeAttention` temporarily forces **interleaved DRAM** for
those uploads (same mitigation as Devstral-2 large MLP weight tilize).

Prefill ``wo`` (:meth:`~models.tt_transformers.tt.attention.Attention.forward_prefill` ``ttnn.linear``)
uses ``ModelArgs.get_attn_wo_program_config``. For ``MatmulMultiCoreReuseMultiCastProgramConfig``, the
TTNN constructor defaults **omitted** ``out_block_h`` / ``out_block_w`` to ``per_core_M`` /
``per_core_N`` (see ``matmul_nanobind.cpp``). For wide 12k ``wo`` that makes ``out_block_w`` ≈ 48
tiles and overflows Blackhole L1 for circular buffers — unrelated to weight **sharding** (DRAM
width shard is already correct; the issue is per-op **output block** sizing). This class wraps
``get_attn_wo_program_config`` on Blackhole and supplies minimal ``out_block_*`` and
``out_subblock_w`` for **prefill** only (conceptually similar to :class:`TtDevstral2LargeRMSNorm`
using a tight multicore norm config instead of the fused default).

Activation row count for ``program_config`` still comes from shared ``Attention.forward_prefill``.
"""

from __future__ import annotations

import ttnn

from models.common.utility_functions import is_blackhole
from models.experimental.devstarl2_small.tt.tt_ministralattn import TtMinistralAttention as _TtMinistralAttentionBase
from models.tt_transformers.tt.common import Mode


def _bh_tighten_wo_prefill_prog_cfg(cfg):
    """Blackhole: force minimal output block sizes for WO prefill matmul.

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


def _bh_multi_dram_weight_tilize(mesh_device, configuration) -> bool:
    return (
        is_blackhole()
        and mesh_device is not None
        and mesh_device.get_num_devices() > 1
        and not getattr(configuration, "is_galaxy", False)
    )


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
        patch_dram_tilize = (
            configuration is not None
            and mesh_device is not None
            and _bh_multi_dram_weight_tilize(mesh_device, configuration)
        )
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
        # out_block_w=per_core_N (~48) → L1 CB clash on Blackhole.
        if is_blackhole() and getattr(self, "args", None) is not None and not getattr(self.args, "is_galaxy", False):
            _orig_get_attn_wo = self.args.get_attn_wo_program_config

            def _get_attn_wo_bh(mode, seq_len=1, prefetcher=None):
                cfg = _orig_get_attn_wo(mode, seq_len, prefetcher)
                if mode != Mode.PREFILL:
                    return cfg
                return _bh_tighten_wo_prefill_prog_cfg(cfg)

            self.args.get_attn_wo_program_config = _get_attn_wo_bh  # type: ignore[method-assign]


TtDevstral2LargeAttention.__name__ = "Attention"
TtDevstral2LargeAttention.__qualname__ = "Attention"

TtMinistralAttention = TtDevstral2LargeAttention

__all__ = [
    "TtDevstral2LargeAttention",
    "TtMinistralAttention",
]
