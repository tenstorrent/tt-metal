# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Decoder-layer ownership and decode normalization for the four-chip model."""

import os

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.gpt_oss.tt.rms_norm import RMSNorm
from models.tt_transformers.tt.load_checkpoints import convert_hf_qkv_to_meta_format


def _local_layer_state_dict(state_dict, layer_idx: int):
    """Return Meta-RoPE-formatted keys local to one HF decoder layer."""
    prefix = f"model.layers.{layer_idx}."
    if any(key.startswith(prefix) for key in state_dict):
        local = {key[len(prefix) :]: value for key, value in state_dict.items() if key.startswith(prefix)}
    else:
        local = dict(state_dict)
    required = {
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.q_proj.bias",
        "self_attn.k_proj.weight",
        "self_attn.k_proj.bias",
        "self_attn.v_proj.weight",
        "self_attn.v_proj.bias",
        "self_attn.o_proj.weight",
        "self_attn.o_proj.bias",
        "self_attn.sinks",
        "mlp.router.weight",
        "mlp.router.bias",
        "mlp.experts.gate_up_proj",
        "mlp.experts.gate_up_proj_bias",
        "mlp.experts.down_proj",
        "mlp.experts.down_proj_bias",
    }
    missing = sorted(required - local.keys())
    if missing:
        raise KeyError(f"Layer {layer_idx} state_dict is missing required GPT-OSS weights: {missing}")
    return convert_hf_qkv_to_meta_format(local, head_dim=64)


class DecodeRMSNorm(RMSNorm):
    """Keep decode normalization on a legal ten-way L1 width shard."""

    @staticmethod
    def sharding_enabled_for(batch_size):
        """Decode norms run on the ten-way width shard at every batch size.

        Sharding at the full 32-row tile was disabled during bring-up after a
        nondeterministic trace.  On 2026-09-16 the batch-32 layer test replayed
        the traced decode 100 times per layer type with bit-identical output
        and the sharded norms saved 66-70 us per layer (1.33 -> 1.26 ms
        sliding, 1.45 -> 1.38 ms full attention), so the full tile now shards
        too; ``GPT_OSS_120B_DECODE_NORM_SHARD_FULL_TILE=0`` restores the
        interleaved 32-row norms.
        """
        if batch_size < ttnn.TILE_SIZE:
            return True
        return os.environ.get("GPT_OSS_120B_DECODE_NORM_SHARD_FULL_TILE", "1") != "0"

    def __init__(
        self,
        mesh_device,
        hf_config,
        state_dict,
        *,
        tensor_cache_path,
        mesh_config,
        weight_dtype,
        enable_decode_sharding,
    ):
        if weight_dtype != ttnn.bfloat16:
            raise ValueError("normalization requires BF16 weights")
        super().__init__(mesh_device, hf_config, state_dict, tensor_cache_path, mesh_config)
        grid = ttnn.CoreGrid(x=10, y=1)
        self.decode_memory_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, 288),
            core_grid=grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.decode_program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[grid.x, grid.y],
            subblock_w=3,
            block_h=1,
            block_w=9,
            inplace=False,
        )
        self.decode_mode = False
        self.enable_decode_sharding = enable_decode_sharding

    def forward(self, x):
        if not self.decode_mode or not self.enable_decode_sharding:
            return super().forward(x)
        owns_sharded = x.memory_config() != self.decode_memory_config
        sharded = ttnn.to_memory_config(x, self.decode_memory_config) if owns_sharded else x
        normed = ttnn.rms_norm(
            sharded,
            weight=self.tt_weight,
            epsilon=self.eps,
            program_config=self.decode_program_config,
            memory_config=self.decode_memory_config,
        )
        if owns_sharded:
            sharded.deallocate(True)
        return normed


class DecoderLayer(LightweightModule):
    """Borrow decode residuals and own the temporary prefill activations."""

    def __init__(
        self,
        *,
        mesh_device,
        hf_config,
        layer_idx,
        layer_type,
        max_batch_size,
        max_context_length,
        page_size,
        input_layernorm,
        post_attention_layernorm,
        attention,
        mlp,
        calibrated_checkpoint_revision=None,
    ):
        self.mesh_device = mesh_device
        self.hf_config = hf_config
        self.layer_idx = layer_idx
        self.layer_type = layer_type
        self.max_batch_size = max_batch_size
        self.max_context_length = max_context_length
        self.page_size = page_size
        self.input_layernorm = input_layernorm
        self.post_attention_layernorm = post_attention_layernorm
        self.self_attn = attention
        self.mlp = mlp
        self.calibrated_checkpoint_revision = calibrated_checkpoint_revision

    @property
    def kv_cache(self):
        return self.self_attn.kv_cache

    def _prefill_forward(
        self,
        hidden_states,
        *,
        position_embeddings,
        current_position,
        page_table,
        kv_cache,
        is_decode,
        user_id,
        batch_size,
        fill_seq_lens=None,
        chunk_start_idx=None,
        ring_tail_block=None,
        fill_start_idx=None,
    ):
        residual = hidden_states
        normed = self.input_layernorm(hidden_states)
        extra = {}
        if fill_seq_lens is not None:
            extra["fill_seq_lens"] = fill_seq_lens
        if chunk_start_idx is not None:
            extra["chunk_start_idx"] = chunk_start_idx
            extra["ring_tail_block"] = ring_tail_block
        if fill_start_idx is not None:
            extra["fill_start_idx"] = fill_start_idx
        attention_out = self.self_attn(
            normed,
            rope_mats=position_embeddings,
            position_idx=current_position,
            page_table=page_table,
            kv_cache=kv_cache,
            is_decode=is_decode,
            user_id=user_id,
            batch_size=batch_size,
            **extra,
        )
        # The attention prefill implementation consumes its input; decode
        # attention borrows it. Keep ownership explicit at this boundary.
        if is_decode:
            normed.deallocate(True)
        hidden_states = ttnn.add(residual, attention_out, output_tensor=attention_out)
        residual.deallocate(True)
        residual = hidden_states
        normed = self.post_attention_layernorm(hidden_states)
        mlp_out = self.mlp(normed, is_decode=is_decode)
        normed.deallocate(True)
        hidden_states = ttnn.add(residual, mlp_out, output_tensor=mlp_out)
        residual.deallocate(True)
        return hidden_states

    def prefill_forward(
        self,
        hidden_states,
        *,
        position_embeddings,
        page_table,
        kv_cache=None,
        user_id=0,
        batch_size=1,
        fill_seq_lens=None,
        chunk_start_idx=None,
        ring_tail_block=None,
        fill_start_idx=None,
    ):
        if page_table is None:
            raise ValueError("DecoderLayer is paged-only and requires page_table")
        if len(hidden_states.shape) != 4 or hidden_states.shape[0] != 1 or hidden_states.shape[1] != batch_size:
            raise ValueError(
                "prefill requires [1, batch, sequence, hidden] input matching batch_size, "
                f"got {tuple(hidden_states.shape)} and batch_size={batch_size}"
            )
        if batch_size < 1 or batch_size > self.max_batch_size:
            raise ValueError(f"batch_size {batch_size} is outside configured maximum {self.max_batch_size}")
        if hidden_states.shape[-1] != self.hf_config.hidden_size:
            raise ValueError(
                f"prefill hidden dimension must be {self.hf_config.hidden_size}, got {hidden_states.shape[-1]}"
            )
        if page_table.shape[-2] < batch_size:
            raise ValueError(f"page_table has {page_table.shape[-2]} rows for batch_size={batch_size}")
        logical_tokens = hidden_states.shape[-2]
        if logical_tokens < 1 or logical_tokens > self.max_context_length:
            raise ValueError(
                f"prefill logical sequence must be within [1, {self.max_context_length}], got {logical_tokens}"
            )
        padded_tokens = ((logical_tokens + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        if len(position_embeddings) != 2 or any(rope.shape[-2] != logical_tokens for rope in position_embeddings):
            raise ValueError("prefill position_embeddings must be a cosine/sine pair of exact logical length")

        if padded_tokens == logical_tokens:
            working_hidden = ttnn.clone(hidden_states, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            working_rope = position_embeddings
        else:
            padding = [(0, 0), (0, 0), (0, padded_tokens - logical_tokens), (0, 0)]
            owned_hidden = ttnn.clone(hidden_states, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            working_hidden = ttnn.pad(owned_hidden, padding, value=0.0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            working_rope = [
                ttnn.pad(
                    ttnn.clone(rope, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                    [(0, 0), (0, 0), (0, padded_tokens - logical_tokens), (0, 0)],
                    value=0.0,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                for rope in position_embeddings
            ]

        working_hidden = ttnn.reshape(
            working_hidden,
            (1, 1, batch_size * padded_tokens, self.hf_config.hidden_size),
        )
        output = self._forward(
            working_hidden,
            position_embeddings=working_rope,
            current_position=None,
            page_table=page_table,
            kv_cache=self.kv_cache if kv_cache is None else kv_cache,
            is_decode=False,
            user_id=user_id,
            batch_size=batch_size,
            fill_seq_lens=fill_seq_lens,
            chunk_start_idx=chunk_start_idx,
            ring_tail_block=ring_tail_block,
            fill_start_idx=fill_start_idx,
        )
        if padded_tokens != logical_tokens:
            for rope in working_rope:
                rope.deallocate(True)
        output = ttnn.reshape(output, (1, batch_size, padded_tokens, self.hf_config.hidden_size))
        if padded_tokens != logical_tokens:
            output = ttnn.slice(
                output,
                starts=[0, 0, 0, 0],
                ends=[1, batch_size, logical_tokens, self.hf_config.hidden_size],
                steps=[1, 1, 1, 1],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return output

    def forward(self, hidden_states, *, mode, **kwargs):
        if mode == "prefill":
            return self.prefill_forward(hidden_states, **kwargs)
        if mode == "decode":
            return self.decode_forward(hidden_states, **kwargs)
        raise ValueError(f"mode must be 'prefill' or 'decode', got {mode!r}")

    def _forward(self, *args, is_decode, **kwargs):
        self.input_layernorm.decode_mode = is_decode
        self.post_attention_layernorm.decode_mode = is_decode
        if not is_decode:
            return self._prefill_forward(*args, is_decode=False, **kwargs)

        hidden_states = args[0]
        position_embeddings = kwargs["position_embeddings"]
        current_position = kwargs["current_position"]
        page_table = kwargs["page_table"]
        kv_cache = kwargs["kv_cache"]
        batch_size = kwargs["batch_size"]

        borrowed_input = hidden_states
        normed = self.input_layernorm(hidden_states)
        attention_out = self.self_attn(
            normed,
            rope_mats=position_embeddings,
            position_idx=current_position,
            page_table=page_table,
            kv_cache=kv_cache,
            is_decode=True,
            user_id=0,
            batch_size=batch_size,
        )
        normed.deallocate(True)
        hidden_states = ttnn.add(borrowed_input, attention_out, output_tensor=attention_out)

        residual = hidden_states
        normed = self.post_attention_layernorm(hidden_states)
        mlp_out = self.mlp(normed, is_decode=True)
        normed.deallocate(True)
        hidden_states = ttnn.add(residual, mlp_out, output_tensor=mlp_out)
        residual.deallocate(True)
        return hidden_states

    def decode_forward(
        self,
        hidden_states,
        *,
        position_embeddings,
        current_position,
        page_table,
        kv_cache=None,
        batch_size=1,
    ):
        if page_table is None:
            raise ValueError("DecoderLayer is paged-only and requires page_table")
        if len(hidden_states.shape) != 4 or hidden_states.shape[0] != 1 or hidden_states.shape[1] != 1:
            raise ValueError(f"decode requires [1, 1, batch, hidden] input, got {tuple(hidden_states.shape)}")
        if hidden_states.shape[-2] != batch_size or hidden_states.shape[-1] != self.hf_config.hidden_size:
            raise ValueError(
                "decode input must match batch_size and hidden size, "
                f"got {tuple(hidden_states.shape)} and batch_size={batch_size}"
            )
        if batch_size < 1 or batch_size > self.max_batch_size:
            raise ValueError(f"batch_size {batch_size} is outside configured maximum {self.max_batch_size}")
        if current_position is None or current_position.shape[-1] < batch_size:
            raise ValueError("decode requires a device-resident current_position covering the decode batch")
        if len(position_embeddings) != 2 or any(rope.shape[1] < batch_size for rope in position_embeddings):
            raise ValueError("decode position_embeddings must be a cosine/sine pair covering the decode batch")
        if page_table.shape[-2] < batch_size:
            raise ValueError(f"page_table has {page_table.shape[-2]} rows for batch_size={batch_size}")
        return self._forward(
            hidden_states,
            position_embeddings=position_embeddings,
            current_position=current_position,
            page_table=page_table,
            kv_cache=self.kv_cache if kv_cache is None else kv_cache,
            is_decode=True,
            user_id=0,
            batch_size=batch_size,
        )
