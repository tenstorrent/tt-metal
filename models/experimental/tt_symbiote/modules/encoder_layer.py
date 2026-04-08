# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""HF ``Qwen3OmniMoeAudioEncoderLayer`` → TTNN (pre-norm attention block + FFN on device)."""

from __future__ import annotations

import ttnn

from models.experimental.tt_symbiote.core.module import TTNNModule
from models.experimental.tt_symbiote.core.run_config import trace_enabled
from models.experimental.tt_symbiote.modules.attention import TTNNQwenAudioAttention
from models.experimental.tt_symbiote.modules.linear import TTNNLinear
from models.experimental.tt_symbiote.modules.normalization import TTNNQwenLayerNorm
from models.experimental.tt_symbiote.modules.qwen_omni_vision_patch import _ensure_ttnn, _replicate_mapper


def _squeeze_leading_one_if_3d(x: ttnn.Tensor) -> ttnn.Tensor:
    if len(x.shape) == 3 and int(x.shape[0]) == 1:
        return ttnn.squeeze(x, 0)
    return x


def _all_gather_hidden_to_full(x: ttnn.Tensor, mesh_device, full_dim: int) -> ttnn.Tensor:
    """``TTNNLinearIReplicatedWColSharded`` (audio ``out_proj``) emits width shards; residual is full-width replicated."""
    if mesh_device is None or mesh_device.get_num_devices() <= 1:
        return x
    w = int(x.shape[-1])
    if w == full_dim:
        return x
    n = int(mesh_device.get_num_devices())
    if w * n != full_dim:
        return x
    return ttnn.all_gather(x, dim=-1, num_links=1, topology=ttnn.Topology.Linear)


def _activation_ttnn(x: ttnn.Tensor, torch_activation_module) -> ttnn.Tensor:
    """Map HF ``ACT2FN`` module to ``ttnn`` (gelu / silu); default to ``ttnn.gelu``."""
    name = type(torch_activation_module).__name__.lower()
    if "silu" in name or "swish" in name:
        return ttnn.silu(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.gelu(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)


@trace_enabled
class TTNNQwen3OmniMoeAudioEncoderLayer(TTNNModule):
    """TTNN audio encoder block: LayerNorm → attention → residual → LayerNorm → fc1 → act → fc2 → residual."""

    @classmethod
    def from_torch(cls, layer):
        m = cls()
        m._fallback_torch_layer = layer
        m.embed_dim = int(layer.embed_dim)
        m.self_attn_layer_norm = TTNNQwenLayerNorm.from_torch(layer.self_attn_layer_norm)
        m.self_attn = TTNNQwenAudioAttention.from_torch(layer.self_attn)
        m.final_layer_norm = TTNNQwenLayerNorm.from_torch(layer.final_layer_norm)
        for ln in (m.self_attn_layer_norm, m.final_layer_norm):
            if isinstance(ln, TTNNQwenLayerNorm):
                # Mesh: hidden states are ``ReplicateTensorToMesh``, not width-sharded (see ``forward``).
                ln._force_replicated_input_layernorm = True
        m.fc1 = TTNNLinear.from_torch(layer.fc1)
        m.fc2 = TTNNLinear.from_torch(layer.fc2)
        m._activation_fn = layer.activation_fn
        return m

    def preprocess_weights_impl(self):
        for mod in (
            self.self_attn_layer_norm,
            self.self_attn,
            self.final_layer_norm,
            self.fc1,
            self.fc2,
        ):
            if isinstance(mod, TTNNModule):
                mod.preprocess_weights()

    def move_weights_to_device_impl(self):
        for mod in (
            self.self_attn_layer_norm,
            self.self_attn,
            self.final_layer_norm,
            self.fc1,
            self.fc2,
        ):
            if isinstance(mod, TTNNModule):
                mod.move_weights_to_device()

    def deallocate_weights_impl(self):
        for mod in (
            self.self_attn_layer_norm,
            self.self_attn,
            self.final_layer_norm,
            self.fc1,
            self.fc2,
        ):
            if isinstance(mod, TTNNModule):
                mod.deallocate_weights()
        super().deallocate_weights_impl()

    def forward(
        self,
        hidden_states,
        cu_seqlens,
        attention_mask=None,
        **kwargs,
    ):
        mapper = _replicate_mapper(self.device)
        residual = _ensure_ttnn(hidden_states, self.device, mesh_mapper=mapper)

        if isinstance(self.self_attn_layer_norm, TTNNQwenLayerNorm):
            hs = self.self_attn_layer_norm(residual)
        else:
            th = ttnn.to_torch(residual)
            th = self.self_attn_layer_norm(th)
            hs = _ensure_ttnn(th, self.device, mesh_mapper=mapper)

        hs = self.self_attn(
            hs,
            attention_mask=attention_mask,
            cu_seqlens=cu_seqlens,
            **kwargs,
        )
        hs = _squeeze_leading_one_if_3d(hs)
        hs = _all_gather_hidden_to_full(hs, self.device, self.embed_dim)
        hidden_states = ttnn.add(residual, hs, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        residual = hidden_states
        if isinstance(self.final_layer_norm, TTNNQwenLayerNorm):
            hidden_states = self.final_layer_norm(hidden_states)
        else:
            th = ttnn.to_torch(hidden_states)
            th = self.final_layer_norm(th)
            hidden_states = _ensure_ttnn(th, self.device, mesh_mapper=mapper)

        hidden_states = self.fc1(hidden_states)
        hidden_states = _activation_ttnn(hidden_states, self._activation_fn)
        hidden_states = self.fc2(hidden_states)
        hidden_states = _squeeze_leading_one_if_3d(hidden_states)
        hidden_states = _all_gather_hidden_to_full(hidden_states, self.device, self.embed_dim)
        hidden_states = ttnn.add(residual, hidden_states, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # HF clamps fp16 extremes; TTNN path uses bfloat16 — optional torch clamp omitted for parity cost.
        return (hidden_states,)
