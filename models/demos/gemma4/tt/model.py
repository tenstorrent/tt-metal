# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Gemma4 Model: embedding -> decoder layers -> final norm -> lm_head -> softcapping.

Architecture:
- 30 decoder layers with [5 sliding, 1 full] x 5 pattern
- Each layer has attention + shared_mlp + MoE + 7 norms + layer_scalar
- Embedding scaled by sqrt(hidden_size)
- final_logit_softcapping = 30.0
- vocab_size = 262144
- tie_word_embeddings = True (lm_head reuses embedding weight)
"""


import ttnn
from models.demos.gemma4.tt.layer import Gemma4DecoderLayer
from models.demos.gemma4.tt.rms_norm import RMSNorm
from models.demos.gemma4.utils.general_utils import get_cache_file_name
from models.demos.gemma4.utils.substate import substate


class Gemma4Model:
    def __init__(
        self,
        mesh_device,
        hf_config,
        state_dict,
        ccl_manager,
        dtype,
        tensor_cache_path,
        mesh_config,
        max_seq_len,
        max_local_batch_size,
        num_layers=None,
        transformation_mats=None,  # Legacy — ignored (HF-style RoPE)
    ):
        self.mesh_device = mesh_device
        self.hf_config = hf_config
        self.mesh_config = mesh_config
        self.hidden_size = hf_config.hidden_size
        self.vocab_size = hf_config.vocab_size
        self.final_logit_softcapping = hf_config.final_logit_softcapping
        self.embed_scale = hf_config.hidden_size**0.5
        n_layers = num_layers or hf_config.num_hidden_layers

        # Embedding weights: [vocab_size, hidden_size]
        if state_dict and "model.language_model.embed_tokens.weight" in state_dict:
            embed_key = "model.language_model.embed_tokens.weight"
        elif state_dict and "model.embed_tokens.weight" in state_dict:
            embed_key = "model.embed_tokens.weight"
        else:
            embed_key = None

        if embed_key and state_dict:
            embed_weight = state_dict[embed_key]  # [vocab_size, hidden_size]
            self.embedding_weight = ttnn.as_tensor(
                embed_weight.unsqueeze(0).unsqueeze(0),
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                cache_file_name=get_cache_file_name(tensor_cache_path, "embed_tokens.weight"),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            # LM head: tied with embeddings (transposed for matmul)
            # embed: [vocab, H] -> lm_head: [1, 1, H, vocab]
            lm_head_weight = embed_weight.transpose(0, 1).unsqueeze(0).unsqueeze(0)
            self.lm_head_weight = ttnn.as_tensor(
                lm_head_weight,
                device=mesh_device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                cache_file_name=get_cache_file_name(tensor_cache_path, "lm_head.weight"),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            self.embedding_weight = None
            self.lm_head_weight = None

        # Decoder layers
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
                transformation_mats=transformation_mats,
            )
            self.layers.append(layer)

        # Final layernorm
        # Try both key formats for the final norm
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

    def __call__(self, hidden_states, rope_mats, position_idx, page_table, kv_caches, is_decode):
        """
        Full model forward pass (post-embedding).

        Args:
            hidden_states: [1, 1, seq_len, hidden_size] on TT device (already embedded + scaled)
            rope_mats: precomputed RoPE matrices (cos, sin)
            position_idx: current position index
            page_table: paged attention page table
            kv_caches: list of KV caches (one per layer), or None
            is_decode: True for decode mode

        Returns:
            logits: [1, 1, seq_len, vocab_size] on TT device
        """
        # Pass through decoder layers
        for i, layer in enumerate(self.layers):
            kv_cache = kv_caches[i] if kv_caches else None
            hidden_states = layer(hidden_states, rope_mats, position_idx, page_table, kv_cache, is_decode)

        # Final norm
        hidden_states = self.norm.forward(hidden_states)

        # LM head
        if self.lm_head_weight is not None:
            logits = ttnn.linear(hidden_states, self.lm_head_weight)
            hidden_states.deallocate(True)
        else:
            logits = hidden_states

        # Logit softcapping: tanh(logits / cap) * cap
        if self.final_logit_softcapping and self.final_logit_softcapping > 0:
            cap = self.final_logit_softcapping
            logits = ttnn.mul(logits, 1.0 / cap)
            logits = ttnn.tanh(logits)
            logits = ttnn.mul(logits, cap)

        return logits

    def embed_tokens(self, tokens):
        """
        Embed input tokens and scale by sqrt(hidden_size).

        Args:
            tokens: [1, seq_len] token IDs on TT device

        Returns:
            embeddings: [1, 1, seq_len, hidden_size] on TT device
        """
        if self.embedding_weight is None:
            raise RuntimeError("Embedding weights not loaded")

        embeds = ttnn.embedding(tokens, self.embedding_weight, dtype=ttnn.bfloat16)
        # Scale by sqrt(hidden_size) per Gemma4
        embeds = ttnn.mul(embeds, self.embed_scale)
        return embeds
