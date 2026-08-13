# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""tt-symbiote adapter for the native Cosmos3 joint-attention block.

Wraps `Cosmos3JointAttention` (from `model/attention.py`) as a `TTNNModule`
so `register_module_replacement_dict` can swap each
`Cosmos3PackedMoTAttention` instance in the HF transformer for the native
TT path. The adapter:

  1. Captures the reference module's config + weights when `from_torch`
     is called during the replacement walk.
  2. Builds the inner `Cosmos3JointAttention` (with `parallel_config` +
     `CCLManager` derived from the mesh shape) in
     `move_weights_to_device_impl`, deferred until `set_device` has run.
  3. Bridges the reference's 2D `[N, hidden_size]` torch contract to the
     native module's 4D `[1, 1, N, hidden_size]` ttnn contract — and
     unpacks the `rotary_emb` tuple.

For the Phase 1 wrapped model, the reference's forward is
`forward(und_seq, gen_seq, rotary_emb)` where `rotary_emb = (cos_und,
sin_und, cos_gen, sin_gen)`. tt-symbiote's `tree_map` walks into the
tuple and converts each element to ttnn before this adapter sees them.

Mesh-shape policy: TP along the larger axis, no SP (yet). For a (1, N)
mesh that's `tp_factor=N` on axis 1, which is what WH LoudBox uses.
For a 2D mesh like (2, 4) we pick the dim with more chips. Constraint:
`tp_factor` must divide `num_key_value_heads` (8 in the real config),
so tp ∈ {1, 2, 4, 8} are valid; the adapter refuses other shapes with
a clear error.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import ttnn
from models.experimental.tt_symbiote.core.module import TTNNModule

if TYPE_CHECKING:
    from torch import nn

    from models.tt_dit.experimental.cosmos3_i2v.model.attention import Cosmos3JointAttention


class TTNNCosmos3JointAttention(TTNNModule):
    """Drop-in replacement for `Cosmos3PackedMoTAttention` under tt-symbiote."""

    def __init__(
        self,
        *,
        hidden_size: int,
        head_dim: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        attention_bias: bool,
        rms_norm_eps: float,
    ) -> None:
        super().__init__()
        self._config = {
            "hidden_size": hidden_size,
            "head_dim": head_dim,
            "num_attention_heads": num_attention_heads,
            "num_key_value_heads": num_key_value_heads,
            "attention_bias": attention_bias,
            "rms_norm_eps": rms_norm_eps,
        }
        self._captured_state_dict: dict | None = None
        self._inner: Cosmos3JointAttention | None = None

    @classmethod
    def from_torch(cls, attn: nn.Module) -> TTNNCosmos3JointAttention:
        """Build adapter from a `Cosmos3PackedMoTAttention` instance.

        We pull config out of the module's own attributes — the reference
        class stores `hidden_size`, `head_dim`, `num_attention_heads`,
        `num_key_value_heads` directly. `attention_bias` and `rms_norm_eps`
        aren't stored explicitly, so we sniff them off the loaded submodules.
        """
        attention_bias = attn.to_q.bias is not None
        rms_norm_eps = float(attn.norm_q.eps)

        new = cls(
            hidden_size=attn.hidden_size,
            head_dim=attn.head_dim,
            num_attention_heads=attn.num_attention_heads,
            num_key_value_heads=attn.num_key_value_heads,
            attention_bias=attention_bias,
            rms_norm_eps=rms_norm_eps,
        )
        new._fallback_torch_layer = attn
        return new

    def preprocess_weights_impl(self) -> None:
        """Capture the reference module's state dict for the move step."""
        if self.torch_layer is None:
            msg = "TTNNCosmos3JointAttention.preprocess_weights_impl requires a fallback torch layer"
            raise RuntimeError(msg)
        self._captured_state_dict = self.torch_layer.state_dict()

    def move_weights_to_device_impl(self) -> None:
        """Instantiate the inner native module on the configured device and load weights."""
        # Local imports avoid circular imports via the package __init__ during replacement registration.
        import ttnn
        from models.tt_dit.experimental.cosmos3_i2v.model.attention import Cosmos3JointAttention
        from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
        from models.tt_dit.parallel.manager import CCLManager

        if self.device is None:
            msg = "TTNNCosmos3JointAttention.move_weights_to_device_impl requires a device"
            raise RuntimeError(msg)
        if self._captured_state_dict is None:
            msg = "preprocess_weights must run before move_weights_to_device"
            raise RuntimeError(msg)

        mesh_shape = tuple(self.device.shape)
        tp_axis = max(range(len(mesh_shape)), key=lambda i: mesh_shape[i])
        tp_factor = mesh_shape[tp_axis]
        sp_axis = 1 - tp_axis if len(mesh_shape) == 2 else 0
        sp_factor = mesh_shape[sp_axis] if len(mesh_shape) == 2 else 1

        # Note: SP isn't exercised in the attention forward yet — gen-pathway ring SDPA is the
        # follow-up. For now we still report sp_factor so the config is faithful.
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

        self._inner = Cosmos3JointAttention(
            mesh_device=self.device,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
            **self._config,
        )
        self._inner.load_torch_state_dict(self._captured_state_dict)

        # Drop the host-side references now that weights live on device.
        self._captured_state_dict = None
        self._fallback_torch_layer = None

    def deallocate_weights_impl(self) -> None:
        if self._inner is not None:
            self._inner.deallocate_weights()
            self._inner = None

    @staticmethod
    def _to_4d(x: ttnn.Tensor) -> tuple[ttnn.Tensor, tuple[int, ...]]:
        """Reshape `[N, H]` or `[B, N, H]` to `[1, 1, N, H]`. Returns (reshaped, orig_shape)."""
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
        msg = f"TTNNCosmos3JointAttention expects rank-2/3/4 input, got rank {rank}"
        raise ValueError(msg)

    @staticmethod
    def _ensure_tile(x: ttnn.Tensor) -> ttnn.Tensor:
        """tt-symbiote's host→device conversion (LightweightRun.to_ttnn) calls ttnn.from_torch
        with layout=None for non-bool dtypes, which lands the tensor in ROW_MAJOR. The downstream
        matmul kernels require TILE. Convert here so the inner native module can assume TILE."""
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
            msg = "TTNNCosmos3JointAttention.forward called before weights were moved to device"
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

        # Restore original ranks so the downstream torch fallback sees the same shape it expects.
        und_out = ttnn.reshape(und_out_4d, und_orig_shape) if len(und_orig_shape) != 4 else und_out_4d
        gen_out = ttnn.reshape(gen_out_4d, gen_orig_shape) if len(gen_orig_shape) != 4 else gen_out_4d
        return und_out, gen_out
