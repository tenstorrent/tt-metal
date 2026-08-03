# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Correctness-first single-device Cohere2-MoE decoder layer.

This implements the three decoder kinds in ``CohereLabs/North-Mini-Code-1.0``:

* layer 0: full attention, forced RoPE, dense SwiGLU;
* layers whose index is not divisible by four: sliding attention, RoPE, routed MoE;
* remaining layers: full attention without RoPE, routed MoE.

Public tensor contracts
-----------------------
``prefill_forward`` accepts ``hidden_states`` in logical
``[1, batch, sequence, hidden]`` order and returns the same shape.  It accepts
every positive logical sequence length up to the allocated cache capacity; TTNN
owns any physical tile padding.  ``page_table`` maps each logical block for each
batch element to a physical block in ``key_cache`` and ``value_cache``.

``decode_forward`` accepts one token in ``[1, batch, 1, hidden]`` order.
``current_positions`` is a device INT32 tensor with one zero-based cache
position per batch element.  ``position_cos`` and ``position_sin`` are stable
device input tensors containing the rows selected for those positions.  This
separation makes the entire forward pass trace-capturable: callers update input,
position, and RoPE tensors in place before replay and no Python position enters
the captured graph.

The cache is always paged and has physical shape
``[num_blocks, num_kv_heads, page_size, head_dim]``.  Prefill fills it using the
page table and decode both updates and reads it through the same table.

All host work (state-dict lookup, interleaved-RoPE weight permutation, tensor
conversion, and RoPE table construction) is confined to ``from_state_dict`` or
explicit setup helpers.  Runtime forwards contain no torch conversion or host
fallback.  This functional baseline deliberately leaves matmul program configs,
per-core grids, sharding policy, and non-required precision choices to the
optimization stage.  Only decode QKV/SDPA/head-concat inputs use the minimum
framework-provided L1 height-sharded layout required by those operations.
"""

from __future__ import annotations

import math
from typing import Mapping

import ttnn
from models.common.lightweightmodule import LightweightModule

MODEL_ID = "CohereLabs/North-Mini-Code-1.0"
ADVERTISED_CONTEXT = 500_000
DEFAULT_PAGE_SIZE = 32
PREFILL_MOE_CHUNK = 1024


def _candidate_keys(layer_idx: int, suffix: str) -> tuple[str, ...]:
    return (
        f"model.layers.{layer_idx}.{suffix}",
        f"model.model.layers.{layer_idx}.{suffix}",
        f"layers.{layer_idx}.{suffix}",
        suffix,
    )


def _find_tensor(state_dict: Mapping[str, object], layer_idx: int, suffix: str):
    for key in _candidate_keys(layer_idx, suffix):
        if key in state_dict:
            return state_dict[key]
    return None


def _require_tensor(state_dict: Mapping[str, object], layer_idx: int, suffix: str):
    tensor = _find_tensor(state_dict, layer_idx, suffix)
    if tensor is None:
        tried = ", ".join(_candidate_keys(layer_idx, suffix))
        raise KeyError(f"Missing North-Mini decoder tensor {suffix!r}; tried {tried}")
    return tensor


def _as_device_tensor(
    tensor,
    *,
    mesh_device,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
):
    return ttnn.from_torch(
        tensor.contiguous(),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=dtype,
        layout=layout,
        memory_config=memory_config,
    )


def _rope_output_permutation(num_heads: int, head_dim: int):
    """Return rows that convert Cohere's interleaved RoPE to split-half RoPE.

    Cohere pairs dimensions ``(0, 1), (2, 3), ...``.  TTNN's basic rotary op
    pairs the first half with the second half.  Reordering every projected head
    to ``[even dimensions, odd dimensions]`` makes the two definitions exactly
    equivalent.  Q and K receive the same orthogonal permutation, so attention
    scores are unchanged.
    """
    import torch

    per_head = torch.cat((torch.arange(0, head_dim, 2), torch.arange(1, head_dim, 2)))
    return torch.cat([per_head + head * head_dim for head in range(num_heads)])


def _stack_unpacked_experts(state_dict, layer_idx: int, projection: str, num_experts: int):
    import torch

    tensors = [
        _find_tensor(state_dict, layer_idx, f"mlp.experts.{expert}.{projection}.weight")
        for expert in range(num_experts)
    ]
    if any(tensor is None for tensor in tensors):
        return None
    return torch.stack(tensors)


def _load_expert_weights(state_dict, layer_idx: int, num_experts: int, intermediate_size: int):
    """Accept either Transformers' packed parameters or hub shard key layout."""
    import torch

    fused = _find_tensor(state_dict, layer_idx, "mlp.experts.gate_up_proj")
    down = _find_tensor(state_dict, layer_idx, "mlp.experts.down_proj")
    if fused is not None and down is not None:
        gate, up = fused.split(intermediate_size, dim=1)
    else:
        gate = _stack_unpacked_experts(state_dict, layer_idx, "gate_proj", num_experts)
        up = _stack_unpacked_experts(state_dict, layer_idx, "up_proj", num_experts)
        down = _stack_unpacked_experts(state_dict, layer_idx, "down_proj", num_experts)
        if gate is None or up is None or down is None:
            raise KeyError(
                "Sparse North-Mini layers require packed mlp.experts gate_up_proj/down_proj "
                "or all per-expert gate_proj/up_proj/down_proj tensors"
            )

    expected_gate = (num_experts, intermediate_size, 2048)
    expected_down = (num_experts, 2048, intermediate_size)
    if tuple(gate.shape) != expected_gate or tuple(up.shape) != expected_gate:
        raise ValueError(f"expert gate/up shapes must be {expected_gate}, got {tuple(gate.shape)}, {tuple(up.shape)}")
    if tuple(down.shape) != expected_down:
        raise ValueError(f"expert down shape must be {expected_down}, got {tuple(down.shape)}")
    return (
        gate.transpose(-2, -1).to(torch.bfloat16),
        up.transpose(-2, -1).to(torch.bfloat16),
        down.transpose(-2, -1).to(torch.bfloat16),
    )


class FunctionalDecoder(LightweightModule):
    """Functional North-Mini decoder with paged prefill and traced decode."""

    def __init__(
        self,
        *,
        hf_config,
        layer_idx: int,
        mesh_device,
        batch: int,
        max_cache_len: int,
        page_size: int,
        weights: dict[str, ttnn.Tensor],
    ):
        super().__init__()
        self.hf_config = hf_config
        self.layer_idx = layer_idx
        self.mesh_device = mesh_device
        self.batch = batch
        self.max_cache_len = max_cache_len
        self.page_size = page_size
        self.weights = weights

        self.hidden_size = int(hf_config.hidden_size)
        self.num_heads = int(hf_config.num_attention_heads)
        self.num_kv_heads = int(hf_config.num_key_value_heads)
        self.head_dim = int(hf_config.head_dim)
        self.scale = self.head_dim**-0.5
        self.eps = float(hf_config.rms_norm_eps)
        self.layer_type = hf_config.layer_types[layer_idx]
        self.mlp_type = hf_config.mlp_layer_types[layer_idx]
        self.sliding_window = int(hf_config.sliding_window) if self.layer_type == "sliding_attention" else None
        self.use_rope = self.sliding_window is not None or (
            self.mlp_type == "dense" and int(hf_config.prefix_dense_sliding_window_pattern) == 1
        )
        self.intermediate_size = (
            int(hf_config.prefix_dense_intermediate_size)
            if self.mlp_type == "dense"
            else int(hf_config.intermediate_size)
        )
        self.num_experts = int(hf_config.num_experts)
        self.top_k = int(hf_config.num_experts_per_tok)

        # Decode RoPE and head-concat are among the few TTNN operations that
        # cannot consume interleaved inputs.  This is the minimal
        # workload-derived layout: one tile-height shard for each active batch
        # lane, bounded by the device's available worker cores.
        storage_grid = mesh_device.compute_with_storage_grid_size()
        decode_cores = min(batch, storage_grid.x * storage_grid.y)
        rectangle_x = next(
            (
                x
                for x in range(min(decode_cores, storage_grid.x), 0, -1)
                if decode_cores % x == 0 and decode_cores // x <= storage_grid.y
            ),
            None,
        )
        if rectangle_x is not None:
            rectangle_y = decode_cores // rectangle_x
            decode_grid = ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(rectangle_x - 1, rectangle_y - 1))}
            )
            self.decode_sub_core_grids = None
        else:
            decode_grid = ttnn.num_cores_to_corerangeset(decode_cores, storage_grid, row_wise=True)
            self.decode_sub_core_grids = ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(storage_grid.x - 1, storage_grid.y - 1))}
            )
        self.decode_rope_memory_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, self.head_dim),
            core_grid=decode_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.decode_concat_memory_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE * math.ceil(self.num_heads / ttnn.TILE_SIZE), self.head_dim),
            core_grid=decode_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

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
        **_kwargs,
    ):
        """Load canonical HF weights and perform every host-side transformation."""
        import torch

        if not isinstance(mesh_device, ttnn.MeshDevice) or tuple(mesh_device.shape) != (1, 1):
            raise ValueError("FunctionalDecoder requires a single-device 1x1 MeshDevice")
        if not 0 <= layer_idx < int(hf_config.num_hidden_layers):
            raise ValueError(f"layer_idx={layer_idx} is outside the configured layer range")
        if batch < 1 or batch > 32:
            raise ValueError(f"functional decode batch must be in [1, 32], got {batch}")
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

        q = _require_tensor(state_dict, layer_idx, "self_attn.q_proj.weight")
        k = _require_tensor(state_dict, layer_idx, "self_attn.k_proj.weight")
        v = _require_tensor(state_dict, layer_idx, "self_attn.v_proj.weight")
        o = _require_tensor(state_dict, layer_idx, "self_attn.o_proj.weight")
        norm = _require_tensor(state_dict, layer_idx, "input_layernorm.weight")

        q_perm = _rope_output_permutation(num_heads, head_dim)
        k_perm = _rope_output_permutation(num_kv_heads, head_dim)
        q = q.index_select(0, q_perm)
        k = k.index_select(0, k_perm)
        qkv = torch.cat((q, k, v), dim=0).transpose(-2, -1).to(torch.bfloat16)

        weights = {
            "qkv": _as_device_tensor(qkv, mesh_device=mesh_device),
            "o": _as_device_tensor(o.transpose(-2, -1).to(torch.bfloat16), mesh_device=mesh_device),
            "norm": _as_device_tensor(
                norm.reshape(1, 1, 1, hidden_size).to(torch.bfloat16),
                mesh_device=mesh_device,
            ),
        }

        mlp_type = hf_config.mlp_layer_types[layer_idx]
        if mlp_type == "dense":
            for name in ("gate_proj", "up_proj", "down_proj"):
                weight = _require_tensor(state_dict, layer_idx, f"mlp.{name}.weight")
                weights[name] = _as_device_tensor(weight.transpose(-2, -1).to(torch.bfloat16), mesh_device=mesh_device)
        elif mlp_type == "sparse":
            gate, up, down = _load_expert_weights(
                state_dict,
                layer_idx,
                int(hf_config.num_experts),
                int(hf_config.intermediate_size),
            )
            weights["router"] = _as_device_tensor(
                _require_tensor(state_dict, layer_idx, "mlp.gate.weight").transpose(-2, -1).to(torch.bfloat16),
                mesh_device=mesh_device,
            )
            weights["expert_gate"] = _as_device_tensor(gate, mesh_device=mesh_device)
            weights["expert_up"] = _as_device_tensor(up, mesh_device=mesh_device)
            weights["expert_down"] = _as_device_tensor(down, mesh_device=mesh_device)
        else:
            raise ValueError(f"unsupported North-Mini MLP layer kind {mlp_type!r}")

        return cls(
            hf_config=hf_config,
            layer_idx=layer_idx,
            mesh_device=mesh_device,
            batch=batch,
            max_cache_len=max_cache_len,
            page_size=page_size,
            weights=weights,
        )

    @staticmethod
    def build_rope_rows(position_ids, *, hf_config, decode: bool = False):
        """Build setup-boundary RoPE rows in TTNN split-half ordering.

        This helper intentionally returns a pair of torch tensors.  Callers
        transfer them before a measured pass and update stable decode buffers
        before trace replay.
        """
        import torch

        positions = torch.as_tensor(position_ids, dtype=torch.float32)
        head_dim = int(hf_config.head_dim)
        theta = float(hf_config.rope_parameters["rope_theta"])
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        freqs = positions.unsqueeze(-1) * inv_freq
        embedding = torch.cat((freqs, freqs), dim=-1)
        if decode:
            embedding = embedding.reshape(1, -1, 1, head_dim)
        else:
            embedding = embedding.reshape(1, 1, -1, head_dim)
        return embedding.cos().to(torch.bfloat16), embedding.sin().to(torch.bfloat16)

    def create_paged_kv_cache(self, *, num_blocks: int | None = None):
        """Allocate BF16 paged K/V storage; page-table ownership stays with caller."""
        min_blocks = self.batch * math.ceil(self.max_cache_len / self.page_size)
        num_blocks = min_blocks if num_blocks is None else int(num_blocks)
        if num_blocks < min_blocks:
            raise ValueError(f"num_blocks={num_blocks} cannot cover required {min_blocks} blocks")
        shape = (num_blocks, self.num_kv_heads, self.page_size, self.head_dim)
        key = ttnn.zeros(
            shape,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        value = ttnn.zeros(
            shape,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return key, value

    def _validate_hidden(self, hidden_states, *, decode: bool):
        expected = (1, self.batch, 1 if decode else None, self.hidden_size)
        actual = tuple(hidden_states.shape)
        if len(actual) != 4 or actual[0] != 1 or actual[1] != self.batch or actual[3] != self.hidden_size:
            raise ValueError(f"hidden_states must match {expected}, got {actual}")
        if decode and actual[2] != 1:
            raise ValueError(f"decode hidden_states sequence must be 1, got {actual[2]}")
        if not decode and not 1 <= actual[2] <= self.max_cache_len:
            raise ValueError(f"prefill sequence must be in [1, {self.max_cache_len}], got {actual[2]}")
        return actual[2]

    def _qkv_prefill(self, normalized, seq_len, position_cos, position_sin):
        fused = ttnn.linear(
            normalized,
            self.weights["qkv"],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        fused = ttnn.reshape(fused, (self.batch, seq_len, -1))
        query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
            fused,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            transpose_key=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if self.use_rope:
            query = ttnn.experimental.rotary_embedding(
                query, position_cos, position_sin, None, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            key = ttnn.experimental.rotary_embedding(
                key, position_cos, position_sin, None, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            # RoPE exposes its tile-padded sequence as logical shape.  Restore
            # the caller's valid token count before cache fill and SDPA.
            query = ttnn.slice(query, (0, 0, 0, 0), (self.batch, self.num_heads, seq_len, self.head_dim))
            key = ttnn.slice(key, (0, 0, 0, 0), (self.batch, self.num_kv_heads, seq_len, self.head_dim))
        return query, key, value

    def _attention_prefill(
        self,
        normalized,
        *,
        key_cache,
        value_cache,
        page_table,
        position_cos,
        position_sin,
        seq_len,
    ):
        query, key, value = self._qkv_prefill(normalized, seq_len, position_cos, position_sin)
        for user in range(self.batch):
            key_user = ttnn.slice(key, (user, 0, 0, 0), (user + 1, self.num_kv_heads, seq_len, self.head_dim))
            value_user = ttnn.slice(value, (user, 0, 0, 0), (user + 1, self.num_kv_heads, seq_len, self.head_dim))
            ttnn.experimental.paged_fill_cache(key_cache, key_user, page_table, batch_idx=user)
            ttnn.experimental.paged_fill_cache(value_cache, value_user, page_table, batch_idx=user)

        attended = ttnn.transformer.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=True,
            scale=self.scale,
            sliding_window_size=self.sliding_window,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attended = ttnn.transformer.concatenate_heads(attended, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.linear(
            attended,
            self.weights["o"],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _qkv_decode(self, normalized, position_cos, position_sin):
        fused = ttnn.linear(
            normalized,
            self.weights["qkv"],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        fused = ttnn.reshape(fused, (1, 1, self.batch, -1))
        query, key, value = ttnn.experimental.nlp_create_qkv_heads_decode(
            fused,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        )
        if self.use_rope:
            # rotary_embedding_hf's decode kernel requires all three operands
            # in the same minimal height-sharded L1 layout.
            position_cos = ttnn.interleaved_to_sharded(position_cos, self.decode_rope_memory_config)
            position_sin = ttnn.interleaved_to_sharded(position_sin, self.decode_rope_memory_config)
            query = ttnn.experimental.rotary_embedding_hf(
                query,
                position_cos,
                position_sin,
                is_decode_mode=True,
                memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
            )
            key = ttnn.experimental.rotary_embedding_hf(
                key,
                position_cos,
                position_sin,
                is_decode_mode=True,
                memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
            )
        return query, key, value

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
        # Decode head-concat keeps a tile-padded logical batch dimension.
        projected = ttnn.slice(projected, (0, 0, 0, 0), (1, 1, self.batch, self.hidden_size))
        return ttnn.permute(projected, (0, 2, 1, 3))

    def _dense_mlp(self, normalized):
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
        activated = ttnn.multiply(ttnn.silu(gate), up)
        return ttnn.linear(
            activated,
            self.weights["down_proj"],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _sparse_moe_chunk(self, normalized, token_count):
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
        activated = ttnn.multiply(ttnn.silu(gate), up)
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

    def _sparse_moe(self, normalized, seq_len):
        total_tokens = self.batch * seq_len
        flat = ttnn.reshape(normalized, (1, 1, total_tokens, self.hidden_size))
        if total_tokens <= PREFILL_MOE_CHUNK:
            result = self._sparse_moe_chunk(flat, total_tokens)
        else:
            chunks = ttnn.split(flat, PREFILL_MOE_CHUNK, dim=2)
            outputs = [self._sparse_moe_chunk(chunk, chunk.shape[2]) for chunk in chunks]
            result = ttnn.concat(outputs, dim=0)
        return ttnn.reshape(result, (1, self.batch, seq_len, self.hidden_size))

    def _mlp(self, normalized, seq_len):
        return self._dense_mlp(normalized) if self.mlp_type == "dense" else self._sparse_moe(normalized, seq_len)

    def prefill_forward(
        self,
        hidden_states,
        *,
        key_cache,
        value_cache,
        page_table,
        position_cos=None,
        position_sin=None,
    ):
        """Run paged causal prefill and return ``[1, batch, sequence, hidden]``."""
        seq_len = self._validate_hidden(hidden_states, decode=False)
        if self.use_rope and (position_cos is None or position_sin is None):
            raise ValueError("this layer kind requires position_cos and position_sin")
        normalized = ttnn.rms_norm(hidden_states, epsilon=self.eps, weight=self.weights["norm"])
        attention = self._attention_prefill(
            normalized,
            key_cache=key_cache,
            value_cache=value_cache,
            page_table=page_table,
            position_cos=position_cos,
            position_sin=position_sin,
            seq_len=seq_len,
        )
        mlp = self._mlp(normalized, seq_len)
        return ttnn.add(ttnn.add(hidden_states, attention), mlp)

    def decode_forward(
        self,
        hidden_states,
        *,
        key_cache,
        value_cache,
        page_table,
        current_positions,
        position_cos=None,
        position_sin=None,
    ):
        """Run one fully device-resident, trace-safe paged decode step."""
        self._validate_hidden(hidden_states, decode=True)
        if tuple(current_positions.shape) != (self.batch,):
            raise ValueError(f"current_positions must have shape ({self.batch},), got {tuple(current_positions.shape)}")
        if self.use_rope and (position_cos is None or position_sin is None):
            raise ValueError("this layer kind requires position_cos and position_sin")
        normalized = ttnn.rms_norm(hidden_states, epsilon=self.eps, weight=self.weights["norm"])
        attention = self._attention_decode(
            normalized,
            key_cache=key_cache,
            value_cache=value_cache,
            page_table=page_table,
            current_positions=current_positions,
            position_cos=position_cos,
            position_sin=position_sin,
        )
        mlp = self._mlp(normalized, 1)
        return ttnn.add(ttnn.add(hidden_states, attention), mlp)

    def forward(self, hidden_states, *, mode, **kwargs):
        if mode == "prefill":
            return self.prefill_forward(hidden_states, **kwargs)
        if mode == "decode":
            return self.decode_forward(hidden_states, **kwargs)
        raise ValueError(f"mode must be 'prefill' or 'decode', got {mode!r}")
