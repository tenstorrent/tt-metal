# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""tt-symbiote adapter for the native Cosmos3VLTextMoTDecoderLayer.

Swaps each `Cosmos3VLTextMoTDecoderLayer` in the HF transformer for the
native composite (RMSNorms + native joint-attention + native MLPs +
residual adds, all on device). When this swap is active the walk stops
at the layer boundary — the inner attention/MLP/RMSNorm modules are
owned by the native layer, so their individual replacement entries
become inert for this layer (still active for stray Linears elsewhere
in the trunk).

Why this beats the per-module swaps: with attention + MLP replaced
individually, each layer's RMSNorms and residual adds stay on host
PyTorch, forcing ~12 host↔device roundtrips per decoder layer (4
RMSNorms + 2 residual adds, each crossing the boundary going in and
coming out). 64 layers × 12 roundtrips × ~20 denoise steps ≈ 15k
boundary crossings per generate. The fused layer cuts this to 1 per
layer (the tt-symbiote conversion at entry/exit).

Reference forward signature:
    forward(und_seq, gen_seq, rotary_emb) -> (und_out, gen_out)
where rotary_emb = (cos_und, sin_und, cos_gen, sin_gen) and und_seq /
gen_seq are 2D `[N, hidden_size]` torch tensors. tt-symbiote's tree_map
walks into the tuple and the adapter unpacks it into the native
module's individual cos/sin args.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import ttnn
from models.experimental.tt_symbiote.core.module import TTNNModule

if TYPE_CHECKING:
    from torch import nn

    from models.tt_dit.experimental.cosmos3_i2v.model.decoder_layer import Cosmos3VLTextMoTDecoderLayer


class TTNNCosmos3VLTextMoTDecoderLayer(TTNNModule):
    """Drop-in replacement for `Cosmos3VLTextMoTDecoderLayer` under tt-symbiote."""

    def __init__(
        self,
        *,
        hidden_size: int,
        head_dim: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        intermediate_size: int,
        attention_bias: bool,
        rms_norm_eps: float,
    ) -> None:
        super().__init__()
        self._config = {
            "hidden_size": hidden_size,
            "head_dim": head_dim,
            "num_attention_heads": num_attention_heads,
            "num_key_value_heads": num_key_value_heads,
            "intermediate_size": intermediate_size,
            "attention_bias": attention_bias,
            "rms_norm_eps": rms_norm_eps,
        }
        self._captured_state_dict: dict | None = None
        self._inner: Cosmos3VLTextMoTDecoderLayer | None = None

    @classmethod
    def from_torch(cls, layer: nn.Module) -> TTNNCosmos3VLTextMoTDecoderLayer:
        """Build adapter from a `Cosmos3VLTextMoTDecoderLayer` instance.

        Sniffs config off the loaded sub-modules. The reference exposes
        `hidden_size` directly; everything else comes from the attention
        and MLP children.
        """
        attn = layer.self_attn
        mlp = layer.mlp
        attention_bias = attn.to_q.bias is not None
        rms_norm_eps = float(layer.input_layernorm.eps)
        new = cls(
            hidden_size=layer.hidden_size,
            head_dim=attn.head_dim,
            num_attention_heads=attn.num_attention_heads,
            num_key_value_heads=attn.num_key_value_heads,
            intermediate_size=mlp.gate_proj.out_features,
            attention_bias=attention_bias,
            rms_norm_eps=rms_norm_eps,
        )
        new._fallback_torch_layer = layer
        return new

    def preprocess_weights_impl(self) -> None:
        if self.torch_layer is None:
            msg = "TTNNCosmos3VLTextMoTDecoderLayer.preprocess_weights_impl requires a fallback torch layer"
            raise RuntimeError(msg)
        self._captured_state_dict = self.torch_layer.state_dict()

    def move_weights_to_device_impl(self) -> None:
        # Local imports avoid circular imports via the package __init__ during replacement registration.
        from models.tt_dit.experimental.cosmos3_i2v.model.decoder_layer import Cosmos3VLTextMoTDecoderLayer
        from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
        from models.tt_dit.parallel.manager import CCLManager

        if self.device is None:
            msg = "TTNNCosmos3VLTextMoTDecoderLayer.move_weights_to_device_impl requires a device"
            raise RuntimeError(msg)
        if self._captured_state_dict is None:
            msg = "preprocess_weights must run before move_weights_to_device"
            raise RuntimeError(msg)

        mesh_shape = tuple(self.device.shape)
        tp_axis = max(range(len(mesh_shape)), key=lambda i: mesh_shape[i])
        tp_factor = mesh_shape[tp_axis]
        sp_axis = 1 - tp_axis if len(mesh_shape) == 2 else 0
        sp_factor = mesh_shape[sp_axis] if len(mesh_shape) == 2 else 1

        parallel_config = DiTParallelConfig(
            cfg_parallel=ParallelFactor(1, 0),
            sequence_parallel=ParallelFactor(sp_factor, sp_axis),
            tensor_parallel=ParallelFactor(tp_factor, tp_axis),
        )
        ccl_manager = (
            CCLManager(mesh_device=self.device, num_links=1, topology=ttnn.Topology.Linear)
            if tp_factor > 1 or sp_factor > 1
            else None
        )

        self._inner = Cosmos3VLTextMoTDecoderLayer(
            mesh_device=self.device,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
            **self._config,
        )
        self._inner.load_torch_state_dict(self._captured_state_dict)

        self._captured_state_dict = None
        self._fallback_torch_layer = None

    def deallocate_weights_impl(self) -> None:
        if self._inner is not None:
            self._inner.deallocate_weights()
            self._inner = None

    @staticmethod
    def _to_4d(x: ttnn.Tensor) -> tuple[ttnn.Tensor, tuple[int, ...]]:
        shape = tuple(x.shape)
        rank = len(shape)
        if rank == 4:
            return x, shape
        if rank == 2:
            n, h = shape
            return ttnn.reshape(x, (1, 1, n, h)), shape
        if rank == 3:
            b, n, h = shape
            return ttnn.reshape(x, (1, b, n, h)), shape
        msg = f"TTNNCosmos3VLTextMoTDecoderLayer expects rank-2/3/4 input, got rank {rank}"
        raise ValueError(msg)

    @staticmethod
    def _ensure_tile(x: ttnn.Tensor) -> ttnn.Tensor:
        """tt-symbiote's host→device conversion lands tensors in ROW_MAJOR. Downstream matmul
        and RMSNorm kernels require TILE — convert here so the inner native module can assume
        TILE on every input."""
        if x.layout != ttnn.TILE_LAYOUT:
            return ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return x

    def forward(
        self,
        und_seq: ttnn.Tensor,
        gen_seq: ttnn.Tensor,
        rotary_emb: tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor],
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        if self._inner is None:
            msg = "TTNNCosmos3VLTextMoTDecoderLayer.forward called before weights were moved to device"
            raise RuntimeError(msg)

        cos_und, sin_und, cos_gen, sin_gen = rotary_emb

        und_4d, und_orig_shape = self._to_4d(self._ensure_tile(und_seq))
        gen_4d, gen_orig_shape = self._to_4d(self._ensure_tile(gen_seq))
        cos_und_4d, _ = self._to_4d(self._ensure_tile(cos_und))
        sin_und_4d, _ = self._to_4d(self._ensure_tile(sin_und))
        cos_gen_4d, _ = self._to_4d(self._ensure_tile(cos_gen))
        sin_gen_4d, _ = self._to_4d(self._ensure_tile(sin_gen))

        und_out_4d, gen_out_4d = self._inner(
            und_4d,
            gen_4d,
            cos_und_4d,
            sin_und_4d,
            cos_gen_4d,
            sin_gen_4d,
        )

        und_out = ttnn.reshape(und_out_4d, und_orig_shape) if len(und_orig_shape) != 4 else und_out_4d
        gen_out = ttnn.reshape(gen_out_4d, gen_orig_shape) if len(gen_orig_shape) != 4 else gen_out_4d
        return und_out, gen_out
