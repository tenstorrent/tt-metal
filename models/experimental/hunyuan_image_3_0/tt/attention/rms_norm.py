# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# TTNN implementation of HunyuanRMSNorm.
# Thin wrapper around models/common/rmsnorm.py (shared with tt_transformers).

import os

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm
from models.tt_transformers.tt.common import Mode

from ..parallel_utils import rmsnorm_shard_config

# Width-shard the small-M residual-stream norm (input/post-attn/ln_f) so its H-wide
# reduction parallelizes across cores instead of serializing on 1 core (62us -> ~40us
# incl. reshards at Sd=32, measured). Gated in rmsnorm_shard_config to the safe regime;
# set HY_SHARD_NORM=0 to force the plain interleaved kernel (A/B / rollback).
_SHARD_NORM = os.environ.get("HY_SHARD_NORM", "1") != "0"


class HunyuanTtRMSNorm(LightweightModule):
    """
    Single-device TTNN RMSNorm for HunyuanImage-3.0.

    Args:
        device:           TTNN device.
        dim:              Hidden size (e.g. 4096 or head_dim for QK norm).
        state_dict:       Model state_dict (plain torch tensors).
        weight_key:       Key of the weight tensor in state_dict, with or without
                          the ``.weight`` suffix, e.g.
                          ``model.layers.0.input_layernorm`` or
                          ``model.layers.0.input_layernorm.weight``.
        eps:              Variance epsilon (default 1e-5).
        weight_dtype:     TTNN dtype for the weight tensor (default bfloat16).
        weight_cache_path: Optional pathlib.Path for weight caching.
    """

    def __init__(
        self,
        device,
        dim: int,
        state_dict: dict,
        weight_key: str,
        eps: float = 1e-5,
        weight_dtype=ttnn.bfloat16,
        weight_cache_path=None,
    ):
        super().__init__()
        self.device = device
        self.dim = dim
        key = weight_key.removesuffix(".weight")
        # Norm weights are 1-D (loaded ROW_MAJOR) and precision-sensitive; bf8/bf4
        # are TILE-only and inappropriate here. Keep norms in bf16 even when the
        # backbone runs bf8 matmul weights (mixed precision — see MEMORY_FIT_PLAN.md).
        if weight_dtype in (ttnn.bfloat8_b, ttnn.bfloat4_b):
            weight_dtype = ttnn.bfloat16
        self._norm = RMSNorm(
            device=device,
            dim=dim,
            state_dict=state_dict,
            weight_key=key,
            layer_num=None,
            eps=eps,
            weight_dtype=weight_dtype,
            weight_cache_path=weight_cache_path,
            fp32_dest_acc_en=True,
        )

    def forward(self, x: ttnn.Tensor, out_memory_config=None) -> ttnn.Tensor:
        """
        Args:
            x: TTNN tensor, shape [B, S, H] or [B, heads, S, head_dim] in TILE_LAYOUT.
            out_memory_config: optional memory config for the normalised output
                (e.g. ttnn.L1_MEMORY_CONFIG); defaults to the input's placement.
        Returns:
            Normalised tensor, same shape as input.
        """
        # Small-M residual-stream norm: run the H-wide reduction width-sharded across
        # cores instead of serialized on 1 core. rmsnorm_shard_config returns None
        # outside the measured-safe regime (large M, or the narrow QK-norm), so this
        # transparently falls back to the interleaved kernel below.
        cfg = rmsnorm_shard_config(self.device, x.shape[-2], x.shape[-1]) if _SHARD_NORM else None
        if cfg is not None:
            program_config, shard_mc = cfg
            x_sharded = ttnn.interleaved_to_sharded(x, shard_mc)
            y = ttnn.rms_norm(
                x_sharded,
                epsilon=self._norm.eps,
                weight=self._norm.weight,
                program_config=program_config,
                memory_config=shard_mc,
                compute_kernel_config=self._norm.compute_kernel_config_hifi2,
            )
            ttnn.deallocate(x_sharded)
            out = ttnn.sharded_to_interleaved(y, out_memory_config or ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(y)
            return out

        norm_config = {"output_mem_config": out_memory_config} if out_memory_config is not None else None
        return self._norm(x, mode=Mode.PREFILL, norm_config=norm_config)
