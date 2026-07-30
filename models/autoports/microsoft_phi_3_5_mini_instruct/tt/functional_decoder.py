# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Correctness-first, single-device Phi-3.5 decoder layer.

Public tensor contracts:

* ``prefill_forward`` consumes ``hidden_states`` shaped ``[1, batch, S, 3072]``
  and a paged cache/page table. Any logical ``1 < S <= max_context`` is valid;
  TTNN owns physical tile padding.
* ``decode_forward`` consumes ``hidden_states`` shaped ``[1, 1, batch, 3072]``.
  ``current_positions`` is an on-device int32 tensor (one position per batch
  slot) used by RoPE, the paged cache update, and paged SDPA. This makes the
  complete decode pass trace-capture/replay safe.

All Torch work is confined to ``from_state_dict``. Runtime forwards contain
only TTNN operations and keep their outputs on device.
"""

from __future__ import annotations

import math
from typing import Mapping

import ttnn
from models.common.lightweightmodule import LightweightModule

HF_ADVERTISED_CONTEXT = 131_072
DEFAULT_PAGE_SIZE = 32
PCC_ACCEPTANCE = 0.995
PREFILL_SDPA_MAX_SEQ = 32_768


def _layer_key(layer_idx: int, suffix: str) -> tuple[str, ...]:
    return (
        f"model.layers.{layer_idx}.{suffix}",
        f"model.model.layers.{layer_idx}.{suffix}",
        f"layers.{layer_idx}.{suffix}",
        suffix,
    )


def _require(state_dict: Mapping[str, object], layer_idx: int, suffix: str):
    for key in _layer_key(layer_idx, suffix):
        if key in state_dict:
            return state_dict[key]
    raise KeyError(f"Missing Phi-3.5 tensor {suffix!r}; tried {_layer_key(layer_idx, suffix)}")


def _to_device(tensor, mesh_device, *, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        tensor.contiguous(),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=layout,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


class FunctionalDecoder(LightweightModule):
    """One dense Phi-3.5 decoder layer with paged prefill and decode."""

    def __init__(
        self,
        *,
        hf_config,
        layer_idx: int,
        mesh_device,
        batch: int,
        max_context: int,
        page_size: int,
        weights: dict[str, ttnn.Tensor],
        short_cos: ttnn.Tensor,
        short_sin: ttnn.Tensor,
        long_cos: ttnn.Tensor,
        long_sin: ttnn.Tensor,
    ):
        self.hf_config = hf_config
        self.layer_idx = layer_idx
        self.mesh_device = mesh_device
        self.batch = batch
        self.max_context = max_context
        self.page_size = page_size
        self.weights = weights
        self.short_cos = short_cos
        self.short_sin = short_sin
        self.long_cos = long_cos
        self.long_sin = long_sin
        self.hidden_size = int(hf_config.hidden_size)
        self.intermediate_size = int(hf_config.intermediate_size)
        self.num_heads = int(hf_config.num_attention_heads)
        self.num_kv_heads = int(hf_config.num_key_value_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.original_context = int(hf_config.original_max_position_embeddings)
        self.scale = self.head_dim**-0.5
        self.eps = float(hf_config.rms_norm_eps)

    @classmethod
    def from_state_dict(
        cls,
        state_dict,
        *,
        hf_config,
        layer_idx,
        mesh_device,
        batch=1,
        max_context=HF_ADVERTISED_CONTEXT,
        page_size=DEFAULT_PAGE_SIZE,
        **_kwargs,
    ):
        """Load canonical HF weights and precompute both Phi-3.5 LongRoPE tables."""
        import torch

        if tuple(mesh_device.shape) != (1, 1):
            raise ValueError(f"FunctionalDecoder requires a 1x1 mesh, got {tuple(mesh_device.shape)}")
        if not 0 <= layer_idx < int(hf_config.num_hidden_layers):
            raise ValueError(f"layer_idx {layer_idx} is outside the configured layer range")
        if int(hf_config.hidden_size) != 3072 or int(hf_config.intermediate_size) != 8192:
            raise ValueError("This translation targets the real Phi-3.5-mini shape (hidden=3072, intermediate=8192)")
        if int(hf_config.num_attention_heads) != 32 or int(hf_config.num_key_value_heads) != 32:
            raise ValueError("This translation targets Phi-3.5-mini's 32 Q heads and 32 KV heads")
        if not 1 <= max_context <= int(hf_config.max_position_embeddings):
            raise ValueError(f"max_context must be in [1, {hf_config.max_position_embeddings}], got {max_context}")
        if page_size <= 0 or page_size % ttnn.TILE_SIZE:
            raise ValueError(f"page_size must be a positive tile multiple, got {page_size}")

        hidden = int(hf_config.hidden_size)
        heads = int(hf_config.num_attention_heads)
        head_dim = hidden // heads
        inter = int(hf_config.intermediate_size)
        qkv = _require(state_dict, layer_idx, "self_attn.qkv_proj.weight")
        o_proj = _require(state_dict, layer_idx, "self_attn.o_proj.weight")
        gate_up = _require(state_dict, layer_idx, "mlp.gate_up_proj.weight")
        down = _require(state_dict, layer_idx, "mlp.down_proj.weight")
        input_norm = _require(state_dict, layer_idx, "input_layernorm.weight")
        post_norm = _require(state_dict, layer_idx, "post_attention_layernorm.weight")
        expected = {
            "qkv": (3 * hidden, hidden),
            "o_proj": (hidden, hidden),
            "gate_up": (2 * inter, hidden),
            "down": (hidden, inter),
        }
        for name, tensor in (("qkv", qkv), ("o_proj", o_proj), ("gate_up", gate_up), ("down", down)):
            if tuple(tensor.shape) != expected[name]:
                raise ValueError(f"{name} has shape {tuple(tensor.shape)}, expected {expected[name]}")

        rope = hf_config.rope_scaling
        positions = torch.arange(max_context, dtype=torch.float32).unsqueeze(1)
        exponent = torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
        base = float(hf_config.rope_theta)
        amplitude = math.sqrt(
            1
            + math.log(int(hf_config.max_position_embeddings) / int(hf_config.original_max_position_embeddings))
            / math.log(int(hf_config.original_max_position_embeddings))
        )

        def rope_table(factors):
            inv_freq = 1.0 / (torch.tensor(factors, dtype=torch.float32) * base**exponent)
            freqs = positions * inv_freq.unsqueeze(0)
            emb = torch.cat((freqs, freqs), dim=-1)
            return (emb.cos() * amplitude).to(torch.bfloat16), (emb.sin() * amplitude).to(torch.bfloat16)

        short_cos, short_sin = rope_table(rope["short_factor"])
        long_cos, long_sin = rope_table(rope["long_factor"])
        norm_shape = (1, 1, hidden // ttnn.TILE_SIZE, ttnn.TILE_SIZE)
        weights = {
            "input_norm": _to_device(
                input_norm.reshape(norm_shape).to(torch.bfloat16), mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT
            ),
            "post_norm": _to_device(
                post_norm.reshape(norm_shape).to(torch.bfloat16), mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT
            ),
            "qkv": _to_device(qkv.transpose(-2, -1).to(torch.bfloat16), mesh_device),
            "o_proj": _to_device(o_proj.transpose(-2, -1).to(torch.bfloat16), mesh_device),
            "gate_up": _to_device(gate_up.transpose(-2, -1).to(torch.bfloat16), mesh_device),
            "down": _to_device(down.transpose(-2, -1).to(torch.bfloat16), mesh_device),
        }
        return cls(
            hf_config=hf_config,
            layer_idx=layer_idx,
            mesh_device=mesh_device,
            batch=batch,
            max_context=max_context,
            page_size=page_size,
            weights=weights,
            short_cos=_to_device(short_cos, mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT),
            short_sin=_to_device(short_sin, mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT),
            long_cos=_to_device(long_cos, mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT),
            long_sin=_to_device(long_sin, mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT),
        )

    def create_paged_kv_cache(self, *, num_physical_blocks=None):
        blocks_per_user = math.ceil(self.max_context / self.page_size)
        num_physical_blocks = num_physical_blocks or self.batch * blocks_per_user
        shape = (num_physical_blocks, self.num_kv_heads, self.page_size, self.head_dim)
        cache_kwargs = dict(
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return ttnn.zeros(shape, **cache_kwargs), ttnn.zeros(shape, **cache_kwargs)

    def _norm(self, hidden_states, weight):
        return ttnn.rms_norm(hidden_states, epsilon=self.eps, weight=weight)

    def _mlp(self, hidden_states):
        normalized = self._norm(hidden_states, self.weights["post_norm"])
        gate_up = ttnn.linear(normalized, self.weights["gate_up"], dtype=ttnn.bfloat16)
        gate_up_shape = tuple(gate_up.shape)
        gate = ttnn.slice(gate_up, [0, 0, 0, 0], [*gate_up_shape[:-1], self.intermediate_size])
        up = ttnn.slice(
            gate_up,
            [0, 0, 0, self.intermediate_size],
            [*gate_up_shape[:-1], 2 * self.intermediate_size],
        )
        activated = ttnn.multiply(ttnn.silu(gate), up)
        return ttnn.add(hidden_states, ttnn.linear(activated, self.weights["down"], dtype=ttnn.bfloat16))

    def _apply_rope(self, value, cos, sin):
        """HF rotate-half expressed from ordinary TTNN ops (Phi head_dim=96).

        ``ttnn.experimental.rotary_embedding`` requires a width divisible by
        64, whereas Phi-3.5's 96-wide heads split at 48. The explicit topology
        is the exact HF operation and has no host fallback.
        """
        leading = list(tuple(value.shape)[:-1])
        first = ttnn.slice(value, [0] * len(leading) + [0], leading + [self.head_dim // 2])
        second = ttnn.slice(
            value,
            [0] * len(leading) + [self.head_dim // 2],
            leading + [self.head_dim],
        )
        rotated = ttnn.concat((ttnn.neg(second), first), dim=-1)
        return ttnn.add(ttnn.multiply(value, cos), ttnn.multiply(rotated, sin))

    def _prefill_rope(self, query, key, seq_len):
        cos_table = self.long_cos if seq_len > self.original_context else self.short_cos
        sin_table = self.long_sin if seq_len > self.original_context else self.short_sin
        cos = ttnn.slice(cos_table, [0, 0], [seq_len, self.head_dim])
        sin = ttnn.slice(sin_table, [0, 0], [seq_len, self.head_dim])
        cos = ttnn.reshape(cos, [1, 1, seq_len, self.head_dim])
        sin = ttnn.reshape(sin, [1, 1, seq_len, self.head_dim])
        return self._apply_rope(query, cos, sin), self._apply_rope(key, cos, sin)

    def _offset_causal_mask(self, *, chunk_start, query_len, key_len):
        query_positions = ttnn.arange(
            chunk_start,
            chunk_start + query_len,
            device=self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        key_positions = ttnn.arange(
            0,
            key_len,
            device=self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        query_positions = ttnn.reshape(query_positions, [1, 1, query_len, 1])
        key_positions = ttnn.reshape(key_positions, [1, 1, 1, key_len])
        allowed = ttnn.typecast(ttnn.ge(query_positions, key_positions), ttnn.bfloat16)
        mask = ttnn.add(ttnn.multiply(allowed, 1.0e4), -1.0e4)
        return ttnn.to_layout(mask, ttnn.TILE_LAYOUT)

    def prefill_forward(self, hidden_states, *, key_cache, value_cache, page_table, user_id=0):
        """Run paged causal prefill and return ``[1, batch, S, hidden]`` on device."""
        shape = tuple(hidden_states.shape)
        if len(shape) != 4 or shape[:2] != (1, self.batch) or shape[3] != self.hidden_size:
            raise ValueError(f"prefill hidden_states must be [1,{self.batch},S,{self.hidden_size}], got {shape}")
        seq_len = shape[2]
        if not 1 < seq_len <= self.max_context:
            raise ValueError(f"prefill sequence must be in [2,{self.max_context}], got {seq_len}")
        residual = hidden_states
        normalized = self._norm(hidden_states, self.weights["input_norm"])
        fused = ttnn.linear(normalized, self.weights["qkv"], dtype=ttnn.bfloat16)
        fused = ttnn.reshape(fused, [self.batch, seq_len, 3 * self.hidden_size])
        query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
            fused,
            None,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            transpose_key=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        query, key = self._prefill_rope(query, key, seq_len)
        query = ttnn.slice(query, [0, 0, 0, 0], [self.batch, self.num_heads, seq_len, self.head_dim])
        key = ttnn.slice(key, [0, 0, 0, 0], [self.batch, self.num_kv_heads, seq_len, self.head_dim])
        value = ttnn.slice(value, [0, 0, 0, 0], [self.batch, self.num_kv_heads, seq_len, self.head_dim])
        for batch_idx in range(self.batch):
            user_key = ttnn.slice(
                key,
                [batch_idx, 0, 0, 0],
                [batch_idx + 1, self.num_kv_heads, seq_len, self.head_dim],
            )
            user_value = ttnn.slice(
                value,
                [batch_idx, 0, 0, 0],
                [batch_idx + 1, self.num_kv_heads, seq_len, self.head_dim],
            )
            ttnn.experimental.paged_fill_cache(
                key_cache,
                user_key,
                page_table,
                batch_idx=user_id + batch_idx,
                block_size=self.page_size,
            )
            ttnn.experimental.paged_fill_cache(
                value_cache,
                user_value,
                page_table,
                batch_idx=user_id + batch_idx,
                block_size=self.page_size,
            )
        if seq_len <= PREFILL_SDPA_MAX_SEQ:
            attended = ttnn.transformer.scaled_dot_product_attention(
                query, key, value, is_causal=True, scale=self.scale, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
        else:
            # Non-chunked SDPA is not correct beyond 32768 tokens. Query the
            # already-populated paged cache in bounded chunks. The operation's
            # framework defaults select its functional program configuration;
            # optimized-stage tuning is intentionally deferred.
            attended_chunks = []
            chunk_start = 0
            while chunk_start < seq_len:
                # Only the initial causal prefix can safely use the full
                # non-chunked limit. Offset masks for later chunks scale as
                # query_len * total_key_len, so bound them to four tiles.
                chunk_capacity = PREFILL_SDPA_MAX_SEQ if chunk_start == 0 else 4 * ttnn.TILE_SIZE
                chunk_len = min(chunk_capacity, seq_len - chunk_start)
                padded_len = math.ceil(chunk_len / ttnn.TILE_SIZE) * ttnn.TILE_SIZE
                query_chunk = ttnn.slice(
                    query,
                    [0, 0, chunk_start, 0],
                    [self.batch, self.num_heads, chunk_start + chunk_len, self.head_dim],
                )
                if padded_len != chunk_len:
                    query_chunk = ttnn.pad(
                        query_chunk,
                        [(0, 0), (0, 0), (0, padded_len - chunk_len), (0, 0)],
                        value=0.0,
                    )
                if chunk_start == 0 and chunk_len == PREFILL_SDPA_MAX_SEQ:
                    prefix_key = ttnn.slice(
                        key,
                        [0, 0, 0, 0],
                        [self.batch, self.num_kv_heads, chunk_len, self.head_dim],
                    )
                    prefix_value = ttnn.slice(
                        value,
                        [0, 0, 0, 0],
                        [self.batch, self.num_kv_heads, chunk_len, self.head_dim],
                    )
                    output_chunk = ttnn.transformer.scaled_dot_product_attention(
                        query_chunk,
                        prefix_key,
                        prefix_value,
                        is_causal=True,
                        scale=self.scale,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                else:
                    mask = self._offset_causal_mask(
                        chunk_start=chunk_start,
                        query_len=padded_len,
                        key_len=seq_len,
                    )
                    output_chunk = ttnn.transformer.scaled_dot_product_attention(
                        query_chunk,
                        key,
                        value,
                        attn_mask=mask,
                        is_causal=False,
                        scale=self.scale,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        compute_kernel_config=ttnn.types.BlackholeComputeKernelConfig(
                            math_fidelity=ttnn.MathFidelity.HiFi4,
                            math_approx_mode=False,
                            fp32_dest_acc_en=True,
                            packer_l1_acc=False,
                        ),
                    )
                if padded_len != chunk_len:
                    output_chunk = ttnn.slice(
                        output_chunk,
                        [0, 0, 0, 0],
                        [self.batch, self.num_heads, chunk_len, self.head_dim],
                    )
                attended_chunks.append(output_chunk)
                chunk_start += chunk_len
            attended = attended_chunks[0] if len(attended_chunks) == 1 else ttnn.concat(attended_chunks, dim=2)
        attended = ttnn.transformer.concatenate_heads(attended, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        projected = ttnn.linear(attended, self.weights["o_proj"], dtype=ttnn.bfloat16)
        projected = ttnn.reshape(projected, [1, self.batch, seq_len, self.hidden_size])
        return self._mlp(ttnn.add(residual, projected))

    def _decode_rope(self, query, key, current_positions, *, use_long_rope):
        cos_table = self.long_cos if use_long_rope else self.short_cos
        sin_table = self.long_sin if use_long_rope else self.short_sin
        rope_positions = ttnn.typecast(current_positions, ttnn.uint32)
        cos = ttnn.embedding(rope_positions, cos_table, layout=ttnn.TILE_LAYOUT)
        sin = ttnn.embedding(rope_positions, sin_table, layout=ttnn.TILE_LAYOUT)
        cos = ttnn.reshape(cos, [1, 1, self.batch, self.head_dim])
        sin = ttnn.reshape(sin, [1, 1, self.batch, self.head_dim])
        # A 48-wide half cannot be sliced from a tile-sharded tensor. Use the
        # untuned DRAM baseline for the explicit Phi rotate-half, then restore
        # the minimal sharding required by paged cache update/decode SDPA.
        query_memory_config = query.memory_config()
        key_memory_config = key.memory_config()
        query = ttnn.to_memory_config(query, ttnn.DRAM_MEMORY_CONFIG)
        key = ttnn.to_memory_config(key, ttnn.DRAM_MEMORY_CONFIG)
        query = self._apply_rope(query, cos, sin)
        key = self._apply_rope(key, cos, sin)
        return (
            ttnn.to_memory_config(query, query_memory_config),
            ttnn.to_memory_config(key, key_memory_config),
        )

    def _decode_concat_memory_config(self):
        grid = self.mesh_device.compute_with_storage_grid_size()
        # nlp_concat_heads_decode requires its one-core-per-user height shards
        # to occupy a single rectangular CoreRange. On Blackhole's 13-wide
        # compute grid, the generic row-wise helper represents batch 32 as
        # 13+13+6 cores, which the op rejects with "bad optional access".
        # Select the widest exact rectangular factor that fits the device.
        grid_x = min(self.batch, grid.x)
        while self.batch % grid_x != 0 or self.batch // grid_x > grid.y:
            grid_x -= 1
        cores = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, self.batch // grid_x - 1))}
        )
        return ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, self.head_dim),
            core_grid=cores,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def decode_forward(
        self,
        hidden_states,
        *,
        key_cache,
        value_cache,
        page_table,
        current_positions,
        use_long_rope,
    ):
        """Run one fully on-device paged decode step.

        ``current_positions`` is int32/ROW_MAJOR/DRAM with logical shape
        ``[batch]`` and is the sole mutable position input during trace replay.
        ``use_long_rope`` is fixed at trace-capture time; capture separate short
        and long traces if a caller spans Phi-3.5's 4096-token RoPE transition.
        """
        shape = tuple(hidden_states.shape)
        if shape != (1, 1, self.batch, self.hidden_size):
            raise ValueError(f"decode hidden_states must be [1,1,{self.batch},{self.hidden_size}], got {shape}")
        if tuple(current_positions.shape) != (self.batch,):
            raise ValueError(f"current_positions must have shape [{self.batch}], got {tuple(current_positions.shape)}")
        residual = hidden_states
        normalized = self._norm(hidden_states, self.weights["input_norm"])
        fused = ttnn.linear(normalized, self.weights["qkv"], dtype=ttnn.bfloat16)
        query, key, value = ttnn.experimental.nlp_create_qkv_heads_decode(
            fused,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        )
        query, key = self._decode_rope(query, key, current_positions, use_long_rope=use_long_rope)
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
            cur_pos_tensor=current_positions,
            page_table_tensor=page_table,
            scale=self.scale,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attended = ttnn.to_memory_config(attended, self._decode_concat_memory_config())
        attended = ttnn.experimental.nlp_concat_heads_decode(attended, num_heads=self.num_heads)
        if self.batch < ttnn.TILE_SIZE:
            attended = ttnn.slice(attended, [0, 0, 0, 0], [1, 1, self.batch, self.hidden_size])
        projected = ttnn.linear(attended, self.weights["o_proj"], dtype=ttnn.bfloat16)
        projected = ttnn.reshape(projected, [1, 1, self.batch, self.hidden_size])
        return self._mlp(ttnn.add(residual, projected))

    def forward(self, hidden_states, *, mode, **kwargs):
        if mode == "prefill":
            return self.prefill_forward(hidden_states, **kwargs)
        if mode == "decode":
            return self.decode_forward(hidden_states, **kwargs)
        raise ValueError(f"mode must be 'prefill' or 'decode', got {mode!r}")
