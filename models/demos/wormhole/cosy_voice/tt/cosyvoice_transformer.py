# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
CosyVoice Transformer — extends tt_transformers.Transformer for speech synthesis.

Key differences from base Transformer:
  1. Uses a speech decoder head (896 → speech_token_size+200) instead of the
     standard LM head (896 → vocab_size).
  2. Accepts pre-composed embeddings (concatenated text + speech tokens) rather
     than raw token IDs for prefill.
  3. Adds a separate speech embedding table for speech token lookup during
     autoregressive decoding.
"""

import torch
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.wormhole.cosy_voice.tt.model_config import CosyVoiceModelConfig
from models.tt_transformers.tt.model import Transformer


class CosyVoiceSpeechHead(LightweightModule):
    """
    Speech decoder head: projects transformer hidden states to speech token logits.

    This replaces the standard LMHead used in text generation models.
    Maps: (batch, seq, 896) → (batch, seq, speech_token_size + 200)
    """

    def __init__(self, config, mesh_device, state_dict, weight_cache_path, dtype):
        super().__init__()
        self.config = config

        # llm_decoder is a Linear(896, speech_token_size+200, bias=False)
        if "llm_decoder.weight" in state_dict:
            decoder_weight = state_dict["llm_decoder.weight"]
        else:
            logger.warning("llm_decoder.weight not found in state dict, using random init")
            decoder_weight = torch.randn(config.speech_vocab_size, config.dim)

        # Transpose for ttnn.linear: (dim, speech_vocab_size)
        decoder_weight_t = decoder_weight.T.unsqueeze(0).unsqueeze(0).contiguous()

        cache_name = None
        if weight_cache_path:
            cache_name = weight_cache_path / "speech_decoder_head"

        self.weight = ttnn.as_tensor(
            decoder_weight_t,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            cache_file_name=cache_name,
        )

    def forward(self, x):
        """Project hidden states to speech token logits."""
        return ttnn.linear(
            x,
            self.weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.config.compute_kernel_config_hifi2,
            dtype=ttnn.bfloat16,
        )


class CosyVoiceSpeechEmbedding(LightweightModule):
    """
    Speech token embedding table.

    Embeds speech tokens (0..speech_token_size+199) into the LLM hidden dimension.
    Used for:
      - Embedding prompt speech tokens during prefill
      - Embedding the previously generated token during autoregressive decode
      - Looking up special tokens (SOS, EOS, task_id, fill)
    """

    def __init__(self, config, mesh_device, state_dict, weight_cache_path, dtype):
        super().__init__()
        self.config = config

        if "speech_embedding.weight" in state_dict:
            emb_weight = state_dict["speech_embedding.weight"]
        else:
            logger.warning("speech_embedding.weight not found, using random init")
            emb_weight = torch.randn(config.speech_vocab_size, config.dim)

        cache_name = None
        if weight_cache_path:
            cache_name = weight_cache_path / "speech_embedding"

        # Store as host tensor for embedding lookup
        self.weight_torch = emb_weight  # Keep CPU copy for host-side embedding lookups

        self.weight = ttnn.as_tensor(
            emb_weight.unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            cache_file_name=cache_name,
        )

    def get_token_embedding(self, token_id):
        """
        Look up a single token embedding on host (for AR decode step).

        Args:
            token_id: integer token index

        Returns:
            torch.Tensor of shape (1, 1, dim)
        """
        return self.weight_torch[token_id].reshape(1, 1, -1)

    def embed_tokens(self, token_ids):
        """
        Embed a sequence of token IDs on host.

        Args:
            token_ids: torch.Tensor of shape (batch, seq_len)

        Returns:
            torch.Tensor of shape (batch, seq_len, dim)
        """
        return self.weight_torch[token_ids]


class CosyVoiceTransformer(Transformer):
    """
    CosyVoice Transformer for speech token prediction.

    This subclasses tt_transformers.Transformer, reusing TransformerBlock,
    Attention, MLP, RMSNorm, and RoPE for the Qwen2-0.5B backbone. It adds
    CosyVoice-specific speech embedding and overrides the standard LMHead
    with a speech decoder head.

    Architecture:
        Input embeddings (pre-composed on host)
        → 24× TransformerBlock (inherited from tt_transformers)
        → RMSNorm
        → SpeechDecoderHead → speech token logits
    """

    def __init__(
        self,
        config: CosyVoiceModelConfig,
        mesh_device,
        qwen2_state_dict: dict,
        cosyvoice_state_dict: dict,
        dtype=ttnn.bfloat8_b,
    ):
        """
        Args:
            config: CosyVoiceModelConfig with all architecture params
            mesh_device: ttnn mesh device
            qwen2_state_dict: Remapped Qwen2 weights (layers.N.attention.wq.*, etc.)
            cosyvoice_state_dict: CosyVoice-specific weights (speech_embedding.*, llm_decoder.*)
            dtype: Weight data type
        """
        weight_cache_path = config.weight_cache_path(dtype)

        # Initialize the base Transformer (sets up layers, norm, rope, embd)
        super().__init__(
            args=config,
            mesh_device=mesh_device,
            dtype=dtype,
            state_dict=qwen2_state_dict,
            weight_cache_path=weight_cache_path,
        )
        self.config = config

        # Store CPU copies of embeddings for prompt composition
        self.text_embedding_torch = qwen2_state_dict["tok_embeddings.weight"]

        # --- Speech-specific modules ---

        # Add speech token embedding table
        self.speech_embedding = CosyVoiceSpeechEmbedding(
            config, mesh_device, cosyvoice_state_dict, weight_cache_path, dtype
        )

        # Override the standard text LMHead with the speech decoder head
        self.lm_head = CosyVoiceSpeechHead(config, mesh_device, cosyvoice_state_dict, weight_cache_path, dtype)

        logger.info(
            f"CosyVoiceTransformer initialized: "
            f"{config.n_layers} layers, dim={config.dim}, "
            f"heads={config.n_heads}, kv_heads={config.n_kv_heads}, "
            f"speech_vocab={config.speech_vocab_size}"
        )

    def setup_kv_cache(self, batch_size, max_seq_len):
        """
        Returns the KV cache structures for all transformer layers.
        Usually tt_transformers Attention class initializes self.layer_past
        during initialization if use_paged_kv_cache is False.
        """
        return [layer.attention.layer_past for layer in self.layers]

    def prepare_inputs_prefill_embeddings(
        self,
        embeddings: torch.Tensor,
        start_pos=0,
        page_table=None,
        chunk_page_table=None,
        last_token_idx=None,
        batch_size=1,
    ):
        """
        Prepare inputs for prefill when we already have the concatenated embeddings.
        This bypasses self.embd() and directly formats the pre-composed tensor.

        Args:
            embeddings: torch.Tensor of shape (1, 1, 1, seq_len * dim) or (batch, seq, dim)
        """
        device = self.mesh_device

        assert embeddings.dim() in (3, 4), f"embeddings must be 3D or 4D, got {embeddings.dim()}D"

        if embeddings.dim() == 3:
            # (batch, seq_len, dim) -> (1, 1, batch * seq_len, dim) to match ttnn unsqueeze logic
            B, S, D = embeddings.shape
            embeddings = embeddings.reshape(1, 1, B * S, D)
        else:
            S = embeddings.shape[2]  # Shape is likely [1, 1, seq_len, dim]

        # The attention kernel requires seq_len to be a multiple of 128.
        # Also, the LM head slices a TILE (32 elements) starting at seq_len - 1,
        # so we must guarantee padded_S >= seq_len + 31.
        padded_S = ((S + 31 + 127) // 128) * 128
        if padded_S > S:
            import torch.nn.functional as F

            # Pad dim=2 (seq_len): F.pad works on last dims, so we need to pad dim=-2
            embeddings = F.pad(embeddings, (0, 0, 0, padded_S - S))  # (0,0) for dim, (0, pad) for seq

        if self.config.num_devices > 1:
            mesh_mapper = ttnn.ShardTensor2dMesh(
                mesh_device=self.mesh_device, dims=(None, 3), mesh_shape=self.config.cluster_shape
            )
        else:
            mesh_mapper = ttnn.ReplicateTensorToMesh(self.mesh_device)

        # Send to device — must be TILE_LAYOUT for RMSNorm in transformer layers
        tokens_embd = ttnn.from_torch(
            embeddings,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=mesh_mapper,
        )
        tokens_embd = ttnn.unsqueeze_to_4D(tokens_embd)

        # Slice the rot mats to the padded prefill seqlen (must match the
        # padded embedding tensor that the attention kernel will process).
        mat_len = self.rope_setup.cos_matrix_prefill.shape[2]
        seq_len = last_token_idx + 1 if last_token_idx is not None else S
        assert mat_len >= seq_len, f"Sequence length {seq_len} exceeds max seq len {mat_len}"

        required_end = start_pos + padded_S
        pad_len = max(0, required_end - mat_len)

        # We set the end_pos to max_seq_len so that we don't create a new tensor for the whole cos_matrix and sin_matrix
        prefill_start_pos = start_pos
        slice_end = min(mat_len, required_end)

        cos_slice = self.rope_setup.cos_matrix_prefill[:, :, prefill_start_pos:slice_end, :]
        sin_slice = self.rope_setup.sin_matrix_prefill[:, :, prefill_start_pos:slice_end, :]

        if pad_len > 0:
            padding = [(0, 0)] * 4
            padding[2] = (0, pad_len)
            cos_slice = ttnn.pad(cos_slice, padding=padding, value=0.0)
            sin_slice = ttnn.pad(sin_slice, padding=padding, value=0.0)

        tt_rot_mats_prefill_global = [cos_slice, sin_slice]
        tt_rot_mats_prefill_local = None

        if page_table is not None:
            tt_page_table = ttnn.from_torch(
                page_table,
                device=device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        else:
            tt_page_table = None

        tt_chunk_page_table = None

        return (
            tokens_embd,
            tt_rot_mats_prefill_global,
            tt_rot_mats_prefill_local,
            tt_page_table,
            tt_chunk_page_table,
        )

    @classmethod
    def from_pretrained(cls, config: CosyVoiceModelConfig, mesh_device, dtype=ttnn.bfloat8_b):
        """
        Load CosyVoiceTransformer from pretrained CosyVoice weights.

        Args:
            config: CosyVoiceModelConfig with weights_dir set
            mesh_device: ttnn mesh device
            dtype: Weight data type

        Returns:
            CosyVoiceTransformer instance
        """
        qwen2_state_dict, cosyvoice_keys = config.load_llm_weights()

        return cls(
            config=config,
            mesh_device=mesh_device,
            qwen2_state_dict=qwen2_state_dict,
            cosyvoice_state_dict=cosyvoice_keys,
            dtype=dtype,
        )
