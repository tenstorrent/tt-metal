# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TT RMSNorm for Devstral-2 / Ministral3.

Matches ``Ministral3RMSNorm``::

    variance = mean(x ** 2, dim=-1, keepdim=True)
    return weight * x * rsqrt(variance + eps)

Weight is replicated across the mesh. RMS is computed per-token, so the op is local — no
collective is needed even under TP.

Prefill and decode use **width-sharded** RMSNorm (hidden=12288). For prefill with
``seq_len <= kv_block_size``, input and post-attention norms keep WIDTH-sharded L1 for QKV / gate /
up matmuls on the same 8×8 grid; other paths revert to interleaved via ``get_activation_mem_config``.
"""

from __future__ import annotations

from typing import Optional

import torch
import ttnn

from models.experimental.devstral2_123B_instruct.tt.mem_config import (
    get_decode_width_sharded_activation_mem_config,
    get_decode_width_sharded_norm_program_config,
    get_prefill_width_sharded_activation_mem_config,
    get_prefill_width_sharded_norm_program_config,
    get_sharded_norm_compute_kernel_config,
)
from models.experimental.devstral2_123B_instruct.tt.model_args import (
    DEVSTRAL2_LARGE_L1_SMALL_SIZE,
    Devstral2Args,
)
from models.experimental.devstral2_123B_instruct.tt.weight_loading import (
    NORM_WEIGHT_MEM_CONFIG,
    resolve_weight_cache_path,
    upload_replicated_tile,
)

__all__ = ["DEVSTRAL2_LARGE_L1_SMALL_SIZE", "TtRMSNorm"]


def _load_weight(
    state_dict: dict,
    weight_key: str,
    hidden_size: int,
    mesh_device,
    dtype: ttnn.DataType,
    *,
    weight_cache_path: Optional[str] = None,
) -> ttnn.Tensor:
    """Load (or zero-init) the per-feature scale and upload as a replicated, tile-padded tensor."""
    if weight_key in state_dict:
        w = state_dict[weight_key]
    else:
        w = torch.ones(hidden_size, dtype=torch.bfloat16)
    w_padded = w.reshape(1, 1, 1, hidden_size).expand(1, 1, ttnn.TILE_SIZE, hidden_size).contiguous()
    return upload_replicated_tile(
        w_padded,
        mesh_device,
        dtype=dtype,
        memory_config=NORM_WEIGHT_MEM_CONFIG,
        weight_cache_path=weight_cache_path,
        cache_key=weight_key,
    )


def _ensure_memory_config(x: ttnn.Tensor, memory_config: ttnn.MemoryConfig) -> ttnn.Tensor:
    if x.memory_config() == memory_config:
        return x
    return ttnn.to_memory_config(x, memory_config)


class TtRMSNorm:
    """Devstral-2 RMSNorm with width-sharded prefill/decode kernels."""

    def __init__(
        self,
        args: Devstral2Args,
        mesh_device,
        state_dict: dict,
        weight_key: str,
        *,
        dtype: Optional[ttnn.DataType] = None,
        weight_cache_path: Optional[str] = None,
    ) -> None:
        self.args = args
        self.eps = args.rms_norm_eps
        self.mesh_device = mesh_device
        cache_path = resolve_weight_cache_path(weight_cache_path, args)
        self.weight = _load_weight(
            state_dict,
            weight_key,
            args.hidden_size,
            mesh_device,
            dtype or args.weight_dtype,
            weight_cache_path=cache_path,
        )
        self._decode_sharded_mem_config = get_decode_width_sharded_activation_mem_config(args.hidden_size)
        self._decode_sharded_program_config = get_decode_width_sharded_norm_program_config(args.hidden_size)
        self._sharded_compute_kernel_config = get_sharded_norm_compute_kernel_config(mesh_device)

    def __call__(
        self,
        x: ttnn.Tensor,
        memory_config: Optional[ttnn.MemoryConfig] = None,
        *,
        mode: str = "prefill",
    ) -> ttnn.Tensor:
        if mode == "decode":
            return self._forward_decode(x, memory_config=memory_config)
        return self._forward_prefill(x, memory_config=memory_config)

    def _forward_decode(self, x: ttnn.Tensor, *, memory_config: Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor:
        """Width-sharded norm kernel; activations stay L1 interleaved outside this op."""
        out_mem = memory_config if memory_config is not None else ttnn.L1_MEMORY_CONFIG
        x = _ensure_memory_config(x, self._decode_sharded_mem_config)
        out = ttnn.rms_norm(
            x,
            epsilon=self.eps,
            weight=self.weight,
            program_config=self._decode_sharded_program_config,
            memory_config=self._decode_sharded_mem_config,
            compute_kernel_config=self._sharded_compute_kernel_config,
        )
        if out_mem != self._decode_sharded_mem_config:
            out = ttnn.to_memory_config(out, out_mem)
        return out

    def _forward_prefill(self, x: ttnn.Tensor, *, memory_config: Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor:
        """Width-sharded norm for short chunks (seq_len <= kv_block_size); streaming fallback otherwise.

        Streaming (no program_config) processes tile-rows one at a time; its L1 footprint is
        bounded by block_w alone and doesn't grow with seq_len, so it is safe for any length.
        """
        out_mem = memory_config if memory_config is not None else ttnn.L1_MEMORY_CONFIG
        seq_len = max(1, int(x.shape[-2]))
        if seq_len <= self.args.kv_block_size:
            sharded_mem = get_prefill_width_sharded_activation_mem_config(seq_len, self.args.hidden_size)
            program_config = get_prefill_width_sharded_norm_program_config(seq_len, self.args.hidden_size)
            x = _ensure_memory_config(x, sharded_mem)
            out = ttnn.rms_norm(
                x,
                epsilon=self.eps,
                weight=self.weight,
                program_config=program_config,
                memory_config=sharded_mem,
                compute_kernel_config=self._sharded_compute_kernel_config,
            )
            if out_mem != sharded_mem:
                out = ttnn.to_memory_config(out, out_mem)
            return out
        x = _ensure_memory_config(x, out_mem)
        return ttnn.rms_norm(
            x,
            epsilon=self.eps,
            weight=self.weight,
            memory_config=out_mem,
            compute_kernel_config=self._sharded_compute_kernel_config,
        )

    def forward(
        self,
        x: ttnn.Tensor,
        memory_config: Optional[ttnn.MemoryConfig] = None,
        *,
        mode: str = "prefill",
    ) -> ttnn.Tensor:
        return self(x, memory_config=memory_config, mode=mode)
