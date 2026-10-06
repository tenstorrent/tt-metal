# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Single-device Gemma4 MoE decoder, with explicit device-owned paged state.

Prefill accepts [1, 1, S, 2816] for one request slot and returns every logical
row. It pads bounded chunks internally. Decode accepts [1, 1, B, 2816], a
uint32 RoPE position tensor and an int32 cache position tensor. All tensors
passed to forward methods are on the same 1x1 mesh. Setup and reference
comparison are outside these methods. Validation evidence is in
../doc/functional_decoder/README.md.
"""

from dataclasses import replace

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tt.decode_attention import DecodeAttention
from models.autoports.google_gemma_4_26b_a4b_it.tt.precision_ops import norm_weight, rms_norm
from models.autoports.google_gemma_4_26b_a4b_it.tt.routing_precision import QKVLinear, Router
from models.common.lightweightmodule import LightweightModule
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.tt.layer import Gemma4DecoderLayer
from models.demos.gemma4.tt.model_config import Gemma4ModelArgs


class FunctionalDecoder(LightweightModule):
    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx, mesh_device, chunk_size=1024):
        """Load a complete HF layer state dict (unprefixed keys), once.

        Caller owns BF16 paged K/V, page table, and precomputed HF RoPE tables.
        No checkpoint download, weight conversion, or cache allocation occurs
        in a forward pass. Full and sliding attention use the same decoder.
        """
        import torch
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer

        config = getattr(hf_config, "text_config", hf_config)
        if mesh_device.get_num_devices() != 1:
            raise ValueError("Functional decoder requires a 1x1 mesh")
        if chunk_size <= 0 or chunk_size % 128:
            raise ValueError("Physical chunk size must be positive and aligned to 128-token SDPA chunks")
        if chunk_size > 16384:
            raise ValueError("Physical chunk size must stay below the native prefill SDPA limit")
        if config.max_position_embeddings % chunk_size:
            raise ValueError("Physical chunk size must divide the HF context to keep padding inside RoPE tables")
        if config.layer_types[layer_idx] == "sliding_attention" and chunk_size < config.sliding_window:
            raise ValueError("Physical sliding chunks must cover the attention window")
        with torch.device("meta"):
            expected = Gemma4TextDecoderLayer(config, layer_idx).state_dict()
        if set(state_dict) != set(expected):
            raise ValueError(f"Layer state keys differ: {set(state_dict) ^ set(expected)}")
        for name, tensor in state_dict.items():
            if tensor.shape != expected[name].shape:
                raise ValueError(f"Invalid shape for {name}: {tensor.shape} != {expected[name].shape}")
        self = cls()
        self.config = config
        self.layer_idx = layer_idx
        self.chunk_size = chunk_size
        self.layer = Gemma4DecoderLayer(
            mesh_device=mesh_device,
            hf_config=Gemma4ModelArgs.from_hf_config(config),
            state_dict={f"model.layers.{layer_idx}.{k}": v for k, v in state_dict.items()},
            layer_idx=layer_idx,
            ccl_manager=None,
            dtype=ttnn.bfloat16,
            tensor_cache_path=None,
            mesh_config=MeshConfig(mesh_device.shape, decode=ModeConfig(tp=1)),
            max_seq_len=config.max_position_embeddings,
            max_local_batch_size=32,
        )
        attention = self.layer.self_attn
        attention.weights = replace(
            attention.weights,
            wqkv=QKVLinear(
                attention.weights.wqkv,
                mesh_device,
                math_fidelity=ttnn.MathFidelity.HiFi4,
            ),
        )
        self.layer.self_attn = DecodeAttention(attention, config.max_position_embeddings)
        self.layer.moe.router = Router(self.layer.moe.router, config.rms_norm_eps)
        self.input_norm_weight = norm_weight(self.layer.input_layernorm.tt_weight, config.hidden_size)
        self.post_attention_norm_weight = norm_weight(self.layer.post_attention_layernorm.tt_weight, config.hidden_size)
        positions = torch.arange(config.max_position_embeddings, dtype=torch.int32)[None]
        self.positions_u32 = ttnn.from_torch(
            positions, device=mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        self.positions_i32 = ttnn.from_torch(
            positions, device=mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        return self

    def _forward(self, x, **attention_kwargs):
        # The router consumes the attention residual, while experts consume its
        # separately normalized value. Shared and routed MLP outputs have their
        # own post norms before summation.
        layer = self.layer
        normed = rms_norm(x, self.config.rms_norm_eps, self.input_norm_weight)
        attention = layer.self_attn(normed, **attention_kwargs)
        post_attention = rms_norm(attention, self.config.rms_norm_eps, self.post_attention_norm_weight)
        residual = ttnn.add(ttnn.typecast(x, ttnn.float32), post_attention)
        shared_input = ttnn.typecast(layer.pre_feedforward_layernorm.forward(residual), ttnn.bfloat16)
        shared = layer.shared_mlp(shared_input)
        shared = layer.post_feedforward_layernorm_1.forward(shared)
        expert_input = ttnn.typecast(layer.pre_feedforward_layernorm_2.forward(residual), ttnn.bfloat16)
        routed = layer.moe(residual, expert_input)
        routed = layer.post_feedforward_layernorm_2.forward(routed)
        combined = layer.post_feedforward_layernorm.forward(ttnn.add(shared, routed))
        output = ttnn.add(residual, combined)
        return ttnn.typecast(ttnn.mul(output, layer.layer_scalar), ttnn.bfloat16)

    def prefill_forward(self, hidden_states, *, rope_mats, page_table, kv_cache, user_id=0, start_pos=0):
        """Prefill or prefix continuation, preserving all S logical outputs.

        page_table [slots, pages] contains physical page IDs; K/V are
        [physical_pages, kv_heads, page_size, head_dim]. Pages belonging to the
        request must cover the requested context rounded up to 128 tokens
        (the attention kernel's read padding). RoPE tables are 4D, absolute,
        [1,1,context,head_dim], tile rounded for a single chunk and chunk
        rounded for multi-chunk sliding prefill. Allocating both for the full
        HF context satisfies every logical length. Other slots are untouched. The final
        tile's padded rows must belong to the same request, beyond its valid S.
        start_pos=0 begins a fresh request. For a nonzero start_pos the caller
        must have filled [0,start_pos) in this request's pages; per-token updates
        preserve partially occupied pages and all prefix K/V.
        """
        length = hidden_states.shape[-2]
        if any(cache.shape[-2] != 32 or cache.dtype != ttnn.bfloat16 for cache in kv_cache):
            raise ValueError("Functional decoder requires BF16 caches with 32-token pages")
        if length <= 0 or start_pos < 0 or start_pos + length > self.config.max_position_embeddings:
            raise ValueError("Sequence outside HF context contract")
        if start_pos:
            # A continuation may begin within an occupied page or tile. Using
            # per-token paged updates preserves the prefix and neighboring rows;
            # fresh prompts still use bounded parallel prefill chunks below.
            rope_2d = tuple(ttnn.reshape(r, (r.shape[-2], r.shape[-1])) for r in rope_mats)
            request_table = page_table[user_id : user_id + 1, :]
            outputs = []
            for offset in range(length):
                position = start_pos + offset
                outputs.append(
                    self.decode_forward(
                        hidden_states[:, :, offset : offset + 1, :],
                        rope_mats=rope_2d,
                        current_pos=self.positions_u32[:, position : position + 1],
                        cache_pos=ttnn.reshape(self.positions_i32[:, position : position + 1], (1,)),
                        page_table=request_table,
                        kv_cache=kv_cache,
                    )
                )
            return outputs[0] if length == 1 else ttnn.concat(outputs, dim=2)
        block = kv_cache[0].shape[-2]
        if self.chunk_size % block:
            raise ValueError("Chunk size must be a multiple of the cache page size")
        attention = self.layer.self_attn
        attention._release_sliding_prefill_tail(clear_persistent=True)
        outputs = []
        for start in range(0, length, self.chunk_size):
            valid = min(self.chunk_size, length - start)
            physical = (valid + 31) // 32 * 32
            short_sliding_tail = attention.config.is_sliding and start > 0 and valid < self.config.sliding_window
            if short_sliding_tail:
                physical = self.config.sliding_window
            x = hidden_states[:, :, start : start + valid, :]
            if physical != valid:
                x = ttnn.pad(x, [(0, 0), (0, 0), (0, physical - valid), (0, 0)], 0.0)
            rope = tuple(r[:, :, start : start + physical, :] for r in rope_mats)
            chunk_table = page_table[:, start // block : (start + valid + block - 1) // block]
            chunk_start = start
            if length <= self.chunk_size:
                # Single-chunk attention does not need a left-padded tail stash.
                chunk_start, chunk_table = None, None
            elif short_sliding_tail:
                # Tensor-offset mode retains an unpadded final tail. Together
                # with physical window padding this avoids the imported helper's
                # host-created zero buffers for short Q and K/V tails.
                chunk_start = self.positions_i32[:, start : start + 1]
            out = self._forward(
                x,
                rope_mats=rope,
                page_table=page_table,
                kv_cache=kv_cache,
                position_idx=None,
                is_decode=False,
                user_id=user_id,
                valid_seq_len=valid,
                chunk_start_idx=chunk_start,
                chunk_page_table=chunk_table,
            )
            outputs.append(out[:, :, :valid, :])
        return outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=2)

    def decode_forward(self, hidden_states, *, rope_mats, current_pos, cache_pos, page_table, kv_cache):
        """Device-only decode; tensor positions and page IDs may change on replay.

        rope_mats are 2D [context, head_dim] tables. current_pos is uint32
        [1, padded_batch] for on-device embedding lookup; cache_pos is int32
        [B] of valid nonnegative cache positions. The caller allocates these
        buffers before capture, warms this exact signature, and refreshes their
        contents before execute_trace. Output has the input's logical shape.
        """
        batch = hidden_states.shape[-2]
        if any(cache.shape[-2] != 32 or cache.dtype != ttnn.bfloat16 for cache in kv_cache):
            raise ValueError("Functional decoder requires BF16 caches with 32-token pages")
        if batch > 1:
            # Fixed logical-batch orchestration is recorded by trace capture.
            # Each slot uses its own page-table row and tensor-valued position.
            rows = []
            for slot in range(batch):
                rows.append(
                    self.decode_forward(
                        hidden_states[:, :, slot : slot + 1, :],
                        rope_mats=rope_mats,
                        current_pos=current_pos[:, slot : slot + 1],
                        cache_pos=cache_pos[slot : slot + 1],
                        page_table=page_table[slot : slot + 1, :],
                        kv_cache=kv_cache,
                    )
                )
            return ttnn.concat(rows, dim=2)
        return self._forward(
            hidden_states,
            rope_mats=rope_mats,
            position_idx=current_pos,
            position_idx_cache=cache_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            is_decode=True,
        )
