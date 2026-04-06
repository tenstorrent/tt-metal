# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Gemma4 Model — full on-device implementation following gpt-oss pattern.

Architecture:
- 30 decoder layers with [5 sliding, 1 full] x 5 pattern
- Two RoPE configs: sliding (head_dim=256, theta=10k) and global (head_dim=512, theta=1M)
- Embedding scaled by sqrt(hidden_size)
- final_logit_softcapping = 30.0
- tie_word_embeddings = True

Supports both prefill and decode modes with paged attention.
Compatible with tt_transformers Generator interface.
"""

import torch

import ttnn
from models.demos.gemma4.tt.attention import Gemma4AttentionConfig
from models.demos.gemma4.tt.layer import Gemma4DecoderLayer
from models.demos.gemma4.tt.rms_norm import RMSNorm
from models.demos.gemma4.utils.general_utils import get_cache_file_name
from models.demos.gemma4.utils.substate import substate


def create_rope_caches(mesh_device, hf_config, max_seq_len):
    """Create HF-format cos/sin caches for both sliding and global layer types.

    Returns dict mapping layer_type -> (cos_tt, sin_tt) on device.
    Uses HF Gemma4TextRotaryEmbedding for exact frequency computation.
    """
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    is_mesh = hasattr(mesh_device, "shape")
    replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None

    rope = Gemma4TextRotaryEmbedding(hf_config)
    x_dummy = torch.randn(1, max_seq_len, hf_config.hidden_size)
    pos_ids = torch.arange(max_seq_len).unsqueeze(0)

    caches = {}
    for layer_type in set(hf_config.layer_types):
        cos, sin = rope(x_dummy, pos_ids, layer_type=layer_type)
        # [1, max_seq_len, head_dim] -> [1, 1, max_seq_len, head_dim]
        cos_tt = ttnn.from_torch(
            cos.unsqueeze(0),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=replicate,
        )
        sin_tt = ttnn.from_torch(
            sin.unsqueeze(0),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=replicate,
        )
        caches[layer_type] = (cos_tt, sin_tt)
    return caches


class Gemma4Model:
    def __init__(
        self,
        mesh_device,
        hf_config,
        state_dict,
        ccl_manager,
        dtype=ttnn.bfloat16,
        tensor_cache_path=None,
        mesh_config=None,
        max_seq_len=131072,
        max_local_batch_size=1,
        num_layers=None,
        paged_attention_config=None,
        create_kv_cache=True,
        # Legacy parameters — ignored
        transformation_mats=None,
    ):
        self.mesh_device = mesh_device
        self.hf_config = hf_config
        self.mesh_config = mesh_config
        self.hidden_size = hf_config.hidden_size
        self.vocab_size = hf_config.vocab_size
        self.final_logit_softcapping = hf_config.final_logit_softcapping
        self.embed_scale = hf_config.hidden_size**0.5
        self.ccl_manager = ccl_manager
        self.max_seq_len = max_seq_len
        n_layers = num_layers or hf_config.num_hidden_layers

        # RoPE caches per layer type (sliding vs global)
        self.rope_caches = create_rope_caches(mesh_device, hf_config, max_seq_len)

        # Embedding
        is_mesh = hasattr(mesh_device, "shape")
        replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None

        if state_dict and "model.language_model.embed_tokens.weight" in state_dict:
            embed_key = "model.language_model.embed_tokens.weight"
        elif state_dict and "model.embed_tokens.weight" in state_dict:
            embed_key = "model.embed_tokens.weight"
        else:
            embed_key = None

        if embed_key and state_dict:
            embed_weight = state_dict[embed_key]
            self.embedding_weight = ttnn.as_tensor(
                embed_weight.unsqueeze(0).unsqueeze(0),
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=replicate,
                cache_file_name=get_cache_file_name(tensor_cache_path, "embed_tokens.weight"),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            # LM head (tied with embeddings)
            lm_head_weight = embed_weight.transpose(0, 1).unsqueeze(0).unsqueeze(0)
            self.lm_head_weight = ttnn.as_tensor(
                lm_head_weight,
                device=mesh_device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=replicate,
                cache_file_name=get_cache_file_name(tensor_cache_path, "lm_head.weight"),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            self.embedding_weight = None
            self.lm_head_weight = None

        # Decoder layers (each creates its own KV cache if requested)
        self.layers = []
        for i in range(n_layers):
            layer = Gemma4DecoderLayer(
                mesh_device=mesh_device,
                hf_config=hf_config,
                state_dict=state_dict,
                layer_idx=i,
                ccl_manager=ccl_manager,
                dtype=dtype,
                tensor_cache_path=tensor_cache_path,
                mesh_config=mesh_config,
                max_seq_len=max_seq_len,
                max_local_batch_size=max_local_batch_size,
            )
            # Create KV cache for this layer's attention
            if create_kv_cache:
                from models.demos.gemma4.tt.attention.kv_cache import init_kv_cache

                attn_cfg = Gemma4AttentionConfig(hf_config, i)
                kv_cache = init_kv_cache(
                    mesh_device=mesh_device,
                    config=attn_cfg,
                    max_batch_size=max_local_batch_size,
                    max_seq_len=max_seq_len,
                    paged_attention_config=paged_attention_config,
                    cache_dtype=ttnn.bfloat16,
                )
                layer.self_attn.kv_cache = kv_cache
            self.layers.append(layer)

        # Extract KV caches for external access (Generator interface)
        self.tt_kv_cache = []
        for layer in self.layers:
            self.tt_kv_cache.append(layer.self_attn.kv_cache)

        # Final norm
        if state_dict and "model.language_model.norm.weight" in state_dict:
            norm_state = substate(state_dict, "model.language_model.norm")
        elif state_dict and "model.norm.weight" in state_dict:
            norm_state = substate(state_dict, "model.norm")
        else:
            norm_state = {}

        self.norm = RMSNorm(
            mesh_device=mesh_device,
            hf_config=hf_config,
            state_dict=norm_state,
            tensor_cache_path=f"{tensor_cache_path}/final_norm" if tensor_cache_path else None,
            mesh_config=mesh_config,
        )

    def _get_rope_mats(self, layer_idx, seq_len=None):
        """Get (cos, sin) for a given layer, optionally sliced to seq_len."""
        layer_type = self.hf_config.layer_types[layer_idx]
        cos, sin = self.rope_caches[layer_type]
        if seq_len is not None:
            cos = cos[:, :, :seq_len, :]
            sin = sin[:, :, :seq_len, :]
        return (cos, sin)

    def __call__(
        self,
        hidden_states,
        rope_mats=None,
        position_idx=None,
        page_table=None,
        kv_caches=None,
        is_decode=True,
        token_index=None,
    ):
        """
        Forward pass through decoder layers + final norm + lm_head + softcapping.

        Args:
            hidden_states: [1, 1, seq_len, hidden_size] on device (post-embedding)
            rope_mats: (cos, sin) override — if None, uses per-layer rope_caches
            position_idx: decode position tensor
            page_table: paged attention table
            kv_caches: list of [k, v] per layer, or None (uses self.tt_kv_cache)
            is_decode: True for decode, False for prefill
            token_index: int for decode RoPE slicing
        """
        seq_len = hidden_states.shape[2]
        caches = kv_caches or self.tt_kv_cache

        for i, layer in enumerate(self.layers):
            # Per-layer RoPE: sliding and global layers have different cos/sin
            if rope_mats is not None:
                layer_rope = rope_mats  # Override (for backward compat / tests)
            else:
                layer_rope = self._get_rope_mats(i, seq_len=seq_len if not is_decode else None)

            kv_cache = caches[i] if caches else None
            hidden_states = layer(
                hidden_states,
                rope_mats=layer_rope,
                position_idx=position_idx,
                page_table=page_table,
                kv_cache=kv_cache,
                is_decode=is_decode,
                token_index=token_index,
            )

        # Final norm
        hidden_states = self.norm.forward(hidden_states)

        # LM head
        if self.lm_head_weight is not None:
            logits = ttnn.linear(hidden_states, self.lm_head_weight)
            hidden_states.deallocate(True)
        else:
            logits = hidden_states

        # Softcapping: tanh(logits / cap) * cap
        if self.final_logit_softcapping and self.final_logit_softcapping > 0:
            cap = self.final_logit_softcapping
            logits = ttnn.mul(logits, 1.0 / cap)
            logits = ttnn.tanh(logits)
            logits = ttnn.mul(logits, cap)

        return logits

    def embed_tokens(self, tokens):
        """Embed input tokens and scale by sqrt(hidden_size)."""
        if self.embedding_weight is None:
            raise RuntimeError("Embedding weights not loaded")
        embeds = ttnn.embedding(tokens, self.embedding_weight, dtype=ttnn.bfloat16)
        embeds = ttnn.mul(embeds, self.embed_scale)
        return embeds

    # ── Generator-compatible interface ────────────────────────────────────

    def ttnn_prefill_forward(self, x, user_id=0, page_table=None, get_last_token=-1, kv_cache=None, batch_size=1):
        """Prefill forward — matches tt_transformers Generator interface."""
        seq_len = x.shape[-2]
        logits = self(
            hidden_states=x,
            position_idx=None,
            page_table=page_table,
            kv_caches=kv_cache,
            is_decode=False,
        )

        # Extract last token tile for next-token prediction
        if get_last_token != -1:
            logits_sliced = ttnn.slice(
                logits,
                (0, 0, get_last_token, 0),
                (1, 1, get_last_token + 32, logits.shape[-1]),
            )
            logits.deallocate(True)
            logits = logits_sliced

        return logits

    def ttnn_decode_forward(self, tokens, current_pos, rot_mat_idxs=None, page_table=None, kv_cache=None):
        """Decode forward — matches tt_transformers Generator interface."""
        input_embeds = self.embed_tokens(tokens)
        input_embeds = ttnn.unsqueeze(input_embeds, 0)

        # Get position as int for token_index
        if isinstance(current_pos, ttnn.Tensor):
            is_mesh = hasattr(self.mesh_device, "shape")
            pos_cpu = ttnn.get_device_tensors(current_pos)[0] if is_mesh else current_pos
            token_index = int(ttnn.to_torch(pos_cpu).item())
        else:
            token_index = int(current_pos)

        logits = self(
            hidden_states=input_embeds,
            position_idx=current_pos,
            page_table=page_table,
            kv_caches=kv_cache,
            is_decode=True,
            token_index=token_index,
        )

        return logits, None
