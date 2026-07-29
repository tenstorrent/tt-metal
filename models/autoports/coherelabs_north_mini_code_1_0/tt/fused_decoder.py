# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Graph-fused single-device CohereLabs/North-Mini-Code-1.0 decoder.

This stage preserves the functional decoder's public and cache contracts while
removing two avoidable MLP operations:

* dense gate/up projections sharing the same input are packed into one linear;
* SiLU is evaluated by the binary multiply kernel instead of a separate unary
  dispatch.

The attention graph in :class:`FunctionalDecoder` is already expressed through
TTNN's dedicated QKV split, RoPE, SDPA, paged-cache, and head-concatenation
operations.  Inheriting those methods is intentional: they are the fused
attention implementation, not a host or functional fallback.
"""

from __future__ import annotations

import math

import ttnn
from models.autoports.coherelabs_north_mini_code_1_0.tt.functional_decoder import (
    ADVERTISED_CONTEXT,
    DEFAULT_PAGE_SIZE,
    FunctionalDecoder,
    _as_device_tensor,
    _load_expert_weights,
    _require_tensor,
    _rope_output_permutation,
)


class FusedDecoder(FunctionalDecoder):
    """North-Mini decoder with graph-fused MLP operations."""

    @classmethod
    def from_state_dict(
        cls,
        state_dict,
        *,
        hf_config,
        layer_idx,
        mesh_device,
        batch=1,
        max_cache_len=ADVERTISED_CONTEXT,
        page_size=DEFAULT_PAGE_SIZE,
        dense_gate_up_variant="packed_slice",
        sparse_gate_up_variant="packed",
        **kwargs,
    ):
        import torch

        if dense_gate_up_variant not in ("decode_only", "packed_all", "packed_slice"):
            raise ValueError(f"unsupported dense_gate_up_variant {dense_gate_up_variant!r}")
        if sparse_gate_up_variant not in ("separate", "packed"):
            raise ValueError(f"unsupported sparse_gate_up_variant {sparse_gate_up_variant!r}")

        if not isinstance(mesh_device, ttnn.MeshDevice) or tuple(mesh_device.shape) != (1, 1):
            raise ValueError("FusedDecoder requires a single-device 1x1 MeshDevice")
        if not 0 <= layer_idx < int(hf_config.num_hidden_layers):
            raise ValueError(f"layer_idx={layer_idx} is outside the configured layer range")
        if batch < 1 or batch > 32:
            raise ValueError(f"fused decode batch must be in [1, 32], got {batch}")
        if not 1 <= max_cache_len <= int(hf_config.max_position_embeddings):
            raise ValueError(f"max_cache_len must be in [1, {hf_config.max_position_embeddings}], got {max_cache_len}")
        if page_size < ttnn.TILE_SIZE or page_size % ttnn.TILE_SIZE:
            raise ValueError(f"page_size must be a positive multiple of {ttnn.TILE_SIZE}, got {page_size}")

        hidden_size = int(hf_config.hidden_size)
        num_heads = int(hf_config.num_attention_heads)
        num_kv_heads = int(hf_config.num_key_value_heads)
        head_dim = int(hf_config.head_dim)
        if (hidden_size, num_heads, num_kv_heads, head_dim) != (2048, 32, 4, 128):
            raise ValueError(
                "North-Mini target dimensions are hidden=2048, heads=32, kv_heads=4, head_dim=128; "
                f"got {(hidden_size, num_heads, num_kv_heads, head_dim)}"
            )

        reserved_blocks = batch * math.ceil(max_cache_len / page_size)
        cache_shape = (reserved_blocks, num_kv_heads, page_size, head_dim)
        reserved_cache = tuple(
            ttnn.zeros(
                cache_shape,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for _ in range(2)
        )

        q = _require_tensor(state_dict, layer_idx, "self_attn.q_proj.weight")
        k = _require_tensor(state_dict, layer_idx, "self_attn.k_proj.weight")
        v = _require_tensor(state_dict, layer_idx, "self_attn.v_proj.weight")
        o = _require_tensor(state_dict, layer_idx, "self_attn.o_proj.weight")
        norm = _require_tensor(state_dict, layer_idx, "input_layernorm.weight")
        q = q.index_select(0, _rope_output_permutation(num_heads, head_dim))
        k = k.index_select(0, _rope_output_permutation(num_kv_heads, head_dim))
        qkv = torch.cat((q, k, v), dim=0).transpose(-2, -1).to(torch.bfloat16)
        weights = {}

        mlp_type = hf_config.mlp_layer_types[layer_idx]
        if mlp_type == "dense":
            gate = _require_tensor(state_dict, layer_idx, "mlp.gate_proj.weight")
            up = _require_tensor(state_dict, layer_idx, "mlp.up_proj.weight")
            gate_up = torch.cat((gate, up), dim=0).transpose(-2, -1).to(torch.bfloat16)
            weights["gate_up"] = _as_device_tensor(gate_up, mesh_device=mesh_device)
            if dense_gate_up_variant == "decode_only":
                weights["gate_proj"] = _as_device_tensor(
                    gate.transpose(-2, -1).to(torch.bfloat16), mesh_device=mesh_device
                )
                weights["up_proj"] = _as_device_tensor(up.transpose(-2, -1).to(torch.bfloat16), mesh_device=mesh_device)
            down = _require_tensor(state_dict, layer_idx, "mlp.down_proj.weight")
            weights["down_proj"] = _as_device_tensor(down.transpose(-2, -1).to(torch.bfloat16), mesh_device=mesh_device)
        elif mlp_type == "sparse":
            gate, up, down = _load_expert_weights(
                state_dict, layer_idx, int(hf_config.num_experts), int(hf_config.intermediate_size)
            )
            if sparse_gate_up_variant == "packed":
                weights["expert_gate_up"] = _as_device_tensor(torch.cat((gate, up), dim=-1), mesh_device=mesh_device)
            else:
                weights["expert_gate"] = _as_device_tensor(gate, mesh_device=mesh_device)
                weights["expert_up"] = _as_device_tensor(up, mesh_device=mesh_device)
            weights["expert_down"] = _as_device_tensor(down, mesh_device=mesh_device)
            weights["router"] = _as_device_tensor(
                _require_tensor(state_dict, layer_idx, "mlp.gate.weight").transpose(-2, -1).to(torch.bfloat16),
                mesh_device=mesh_device,
            )
        else:
            raise ValueError(f"unsupported North-Mini MLP layer kind {mlp_type!r}")

        weights["qkv"] = _as_device_tensor(qkv, mesh_device=mesh_device)
        weights["o"] = _as_device_tensor(o.transpose(-2, -1).to(torch.bfloat16), mesh_device=mesh_device)
        weights["norm"] = _as_device_tensor(
            norm.reshape(1, 1, 1, hidden_size).to(torch.bfloat16),
            mesh_device=mesh_device,
        )

        decoder = cls(
            hf_config=hf_config,
            layer_idx=layer_idx,
            mesh_device=mesh_device,
            batch=batch,
            max_cache_len=max_cache_len,
            page_size=page_size,
            weights=weights,
        )
        decoder.dense_gate_up_variant = dense_gate_up_variant
        decoder.sparse_gate_up_variant = sparse_gate_up_variant
        decoder._reserved_kv_cache = reserved_cache
        decoder._reserved_kv_blocks = reserved_blocks
        return decoder

    def create_paged_kv_cache(self, *, num_blocks: int | None = None):
        """Return setup-reserved K/V storage before making any later allocation."""
        requested_blocks = self._reserved_kv_blocks if num_blocks is None else int(num_blocks)
        if requested_blocks < self._reserved_kv_blocks:
            raise ValueError(f"num_blocks={requested_blocks} cannot cover required {self._reserved_kv_blocks} blocks")
        if self._reserved_kv_cache is not None and requested_blocks == self._reserved_kv_blocks:
            cache = self._reserved_kv_cache
            self._reserved_kv_cache = None
            return cache
        return super().create_paged_kv_cache(num_blocks=requested_blocks)

    @staticmethod
    def _swiglu(gate, up):
        """Execute SiLU(gate) * up in one device binary kernel."""
        return ttnn.multiply(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
        )

    def _dense_mlp(self, normalized):
        # Decode uses the fast two-way split specialization. Prefill uses two
        # device slices because TTNN's split specialization does not compile
        # for every valid non-tile-aligned logical sequence length.
        if normalized.shape[2] == 1 or self.dense_gate_up_variant in ("packed_all", "packed_slice"):
            packed = ttnn.linear(
                normalized,
                self.weights["gate_up"],
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            if self.dense_gate_up_variant == "packed_slice" and normalized.shape[2] != 1:
                gate = ttnn.slice(
                    packed,
                    (0, 0, 0, 0),
                    (packed.shape[0], packed.shape[1], packed.shape[2], self.intermediate_size),
                )
                up = ttnn.slice(
                    packed,
                    (0, 0, 0, self.intermediate_size),
                    (packed.shape[0], packed.shape[1], packed.shape[2], 2 * self.intermediate_size),
                )
            else:
                gate, up = ttnn.split(packed, self.intermediate_size, dim=-1)
        else:
            gate = ttnn.linear(
                normalized,
                self.weights["gate_proj"],
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            up = ttnn.linear(
                normalized,
                self.weights["up_proj"],
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        activated = self._swiglu(gate, up)
        return ttnn.linear(
            activated,
            self.weights["down_proj"],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _attention_decode(
        self,
        normalized,
        *,
        key_cache,
        value_cache,
        page_table,
        current_positions,
        position_cos,
        position_sin,
    ):
        """Decode attention without the baseline's final slice/transpose pair."""
        query, key, value = self._qkv_decode(normalized, position_cos, position_sin)
        ttnn.experimental.paged_update_cache(
            key_cache, key, update_idxs_tensor=current_positions, page_table=page_table
        )
        ttnn.experimental.paged_update_cache(
            value_cache, value, update_idxs_tensor=current_positions, page_table=page_table
        )
        attended = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            query,
            key_cache,
            value_cache,
            page_table_tensor=page_table,
            cur_pos_tensor=current_positions,
            scale=self.scale,
            sliding_window_size=self.sliding_window,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attended = ttnn.to_memory_config(attended, self.decode_concat_memory_config)
        attended = ttnn.experimental.nlp_concat_heads_decode(
            attended,
            num_heads=self.num_heads,
            sub_core_grids=self.decode_sub_core_grids,
        )
        projected = ttnn.linear(
            attended,
            self.weights["o"],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if projected.shape[2] != self.batch:
            projected = ttnn.slice(projected, (0, 0, 0, 0), (1, 1, self.batch, self.hidden_size))
        # A direct reshape wins at batch 1.  At the serving batch, the
        # otherwise redundant transpose gives consistently better traced wall
        # latency (6.026 ms versus 6.051 ms), so retain that measured geometry.
        if self.batch == 32:
            return ttnn.permute(projected, (0, 2, 1, 3))
        return ttnn.reshape(projected, (1, self.batch, 1, self.hidden_size))

    def _sparse_moe_chunk(self, normalized, token_count):
        """Keep exact routed-MoE math while fusing its SiLU/multiply pair."""
        flat = ttnn.reshape(normalized, (token_count, self.hidden_size))
        logits = ttnn.linear(
            flat,
            self.weights["router"],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        top_values, top_indices = ttnn.topk(logits, k=self.top_k, dim=-1, sorted=True)
        top_values = ttnn.sigmoid(top_values)
        routing = ttnn.scatter(ttnn.zeros_like(logits), dim=-1, index=top_indices, src=top_values)

        expert_input = ttnn.reshape(flat, (1, token_count, self.hidden_size))
        expert_input = ttnn.repeat(expert_input, ttnn.Shape((self.num_experts, 1, 1)))
        if self.sparse_gate_up_variant == "packed":
            gate_up = ttnn.matmul(
                expert_input,
                self.weights["expert_gate_up"],
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            gate, up = ttnn.split(gate_up, self.intermediate_size, dim=-1)
        else:
            gate = ttnn.matmul(
                expert_input,
                self.weights["expert_gate"],
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            up = ttnn.matmul(
                expert_input,
                self.weights["expert_up"],
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        activated = self._swiglu(gate, up)
        expert_output = ttnn.matmul(
            activated,
            self.weights["expert_down"],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        routing = ttnn.permute(routing, (1, 0))
        routing = ttnn.reshape(routing, (self.num_experts, token_count, 1))
        expert_output = ttnn.multiply(expert_output, routing)
        return ttnn.sum(expert_output, dim=0)
