# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Qwen3-TTS LM Talker implementation on TT hardware.

The Talker is the core autoregressive component of Qwen3-TTS. It is architecturally
identical to Qwen3-1.7B (28 layers, GQA 16Q/8KV, SwiGLU, RMSNorm) with these
TTS-specific additions:

  1. Dual embedding tables: text_vocab (151936) + codec_vocab (3072)
  2. Codec head: predicts codebook-0 tokens instead of text tokens
  3. Speaker embedding injection: projects speaker_emb into hidden state
  4. MRoPE: Multi-dimensional RoPE with interleaved sections [24, 20, 20]
     (for non-streaming mode with sequential positions, equivalent to standard RoPE)

Subclasses the shared Transformer infrastructure from models/tt_transformers/.
During prefill, text tokens are embedded on the host (torch) and passed as
pre-embedded tensors. During decode, codec tokens use the on-device embedding.
"""

import math

import torch

import ttnn
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.model import Transformer


class TalkerTransformer(Transformer):
    """
    Qwen3-TTS Talker: autoregressive transformer that generates CB0 codec tokens.

    Inherits the full decode/prefill infrastructure from the base Transformer.
    Adds:
      - text_embed_weight (torch, host): for CPU-side text token embedding during prefill
      - text_projection (ttnn, device): 2-layer MLP (Linear→SiLU→Linear) projecting
        text embeddings from text_hidden_size to Talker hidden_size
      - Speaker embedding: directly added to hidden state (same dim, no projection needed)
    """

    def __init__(
        self,
        args,
        dtype,
        mesh_device,
        state_dict,
        weight_cache_path,
        paged_attention_config=None,
        use_paged_kv_cache=False,
    ):
        super().__init__(
            args=args,
            dtype=dtype,
            mesh_device=mesh_device,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            paged_attention_config=paged_attention_config,
            use_paged_kv_cache=use_paged_kv_cache,
        )

        # Text embedding weights kept on host for CPU-side prefill embedding
        # HF key: talker.model.text_embedding.weight → after meta conversion: talker.text_embedding.weight
        text_key = "talker.text_embedding.weight"
        if state_dict is not None and text_key in state_dict:
            self.text_embed_weight = state_dict[text_key].clone()
        else:
            self.text_embed_weight = torch.randn(args.text_vocab_size, args.dim)

        # Codec embedding on host (for Code Predictor's CB0 embedding lookup)
        # Device-side codec embedding is self.embd (inherited from Transformer)
        codec_key = "talker.tok_embeddings.weight"
        if state_dict is not None and codec_key in state_dict:
            self.codec_embed_weight = state_dict[codec_key].clone()
        else:
            self.codec_embed_weight = torch.randn(args.vocab_size, args.dim)

        # Text projection MLP: 2-layer MLP with SiLU (projects text LM hidden → Talker hidden)
        # HF keys: talker.text_projection.linear_fc1/fc2 (.weight + .bias)
        def _load_linear(key_prefix, in_dim, out_dim, has_bias=True):
            w_key = f"{key_prefix}.weight"
            b_key = f"{key_prefix}.bias"
            if state_dict is not None and w_key in state_dict:
                w = state_dict[w_key].T.contiguous().unsqueeze(0).unsqueeze(0)
            else:
                w = torch.randn(1, 1, in_dim, out_dim)

            tt_w = ttnn.as_tensor(
                w, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                cache_file_name=weight_cache_path / f"{w_key}" if weight_cache_path else None,
            )

            tt_b = None
            if has_bias:
                if state_dict is not None and b_key in state_dict:
                    b = state_dict[b_key].unsqueeze(0).unsqueeze(0).unsqueeze(0)
                else:
                    b = torch.zeros(1, 1, 1, out_dim)
                tt_b = ttnn.as_tensor(
                    b, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                    cache_file_name=weight_cache_path / f"{b_key}" if weight_cache_path else None,
                )
            return tt_w, tt_b

        self.text_proj_fc1_w, self.text_proj_fc1_b = _load_linear(
            "talker.text_projection.linear_fc1", args.text_hidden_size, args.dim
        )
        self.text_proj_fc2_w, self.text_proj_fc2_b = _load_linear(
            "talker.text_projection.linear_fc2", args.dim, args.dim
        )

    def embed_text_tokens(self, token_ids):
        """Embed text tokens on the host using the text embedding table.

        Args:
            token_ids: torch.Tensor [batch, seq_len] of text token IDs

        Returns:
            torch.Tensor [batch, seq_len, dim] of embeddings
        """
        return torch.nn.functional.embedding(token_ids, self.text_embed_weight)

    def prepare_inputs_prefill(
        self,
        tokens,
        start_pos=0,
        page_table=None,
        chunk_page_table=None,
        trace_enabled=False,
        last_token_idx=None,
        global_user_id=None,
        batch_size=1,
        user_id=0,
        **kwargs,
    ):
        """Prepare prefill inputs, accepting pre-embedded text tensors.

        For TTS prefill, `tokens` is a pre-embedded torch.Tensor [batch, seq_len, dim]
        (produced by embed_text_tokens). This follows the Qwen3-VL pattern where
        embeddings are computed on the host before being sent to device.

        For decode or codec token input, `tokens` is a standard [batch, seq_len]
        integer tensor, which falls through to the base class embedding.
        """
        if isinstance(tokens, torch.Tensor) and tokens.dim() == 3:
            return self._prepare_preembedded_prefill(
                tokens,
                start_pos=start_pos,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                trace_enabled=trace_enabled,
                last_token_idx=last_token_idx,
                batch_size=batch_size,
                user_id=user_id,
            )

        return super().prepare_inputs_prefill(
            tokens,
            start_pos=start_pos,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            trace_enabled=trace_enabled,
            last_token_idx=last_token_idx,
            global_user_id=global_user_id,
            batch_size=batch_size,
            user_id=user_id,
            **kwargs,
        )

    def _prepare_preembedded_prefill(
        self,
        embeddings,
        start_pos=0,
        page_table=None,
        chunk_page_table=None,
        trace_enabled=False,
        last_token_idx=None,
        batch_size=1,
        user_id=0,
    ):
        """Prepare prefill inputs from pre-embedded tensor [batch, seq_len, dim]."""
        device = None if trace_enabled else self.mesh_device
        S = embeddings.shape[1]

        tokens_embd = embeddings.unsqueeze(1)  # [batch, 1, seq_len, dim]
        tokens_embd = ttnn.from_torch(
            tokens_embd,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        # Slice RoPE cos/sin to the prefill sequence length
        mat_len = self.rope_setup.cos_matrix_prefill.shape[2]
        seq_len = last_token_idx + 1 if last_token_idx is not None else S
        assert mat_len >= seq_len, f"Sequence length {seq_len} exceeds max seq len {mat_len}"

        required_end = start_pos + S
        pad_len = max(0, required_end - mat_len)
        prefill_start_pos = 0 if trace_enabled else start_pos
        slice_end = self.args.max_seq_len if trace_enabled else min(mat_len, required_end)

        cos_slice = self.rope_setup.cos_matrix_prefill[:, :, prefill_start_pos:slice_end, :]
        sin_slice = self.rope_setup.sin_matrix_prefill[:, :, prefill_start_pos:slice_end, :]

        if pad_len > 0:
            padding = [(0, 0)] * 4
            padding[2] = (0, pad_len)
            cos_slice = ttnn.pad(cos_slice, padding=padding, value=0.0)
            sin_slice = ttnn.pad(sin_slice, padding=padding, value=0.0)

        tt_rot_mats_prefill = [cos_slice, sin_slice]
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

        if chunk_page_table is not None:
            tt_chunk_page_table = ttnn.from_torch(
                chunk_page_table,
                device=device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        else:
            tt_chunk_page_table = None

        return (
            tokens_embd,
            tt_rot_mats_prefill,
            tt_rot_mats_prefill_local,
            tt_page_table,
            tt_chunk_page_table,
        )

    def text_projection(self, x):
        """Apply the 2-layer text projection MLP: Linear → SiLU → Linear.

        Projects text embeddings from text_hidden_size to Talker hidden_size.
        Called on pre-embedded text tensors before the main Transformer forward.
        """
        h = ttnn.matmul(x, self.text_proj_fc1_w)
        if self.text_proj_fc1_b is not None:
            h = ttnn.add(h, self.text_proj_fc1_b)
        h = ttnn.silu(h)
        h = ttnn.matmul(h, self.text_proj_fc2_w)
        if self.text_proj_fc2_b is not None:
            h = ttnn.add(h, self.text_proj_fc2_b)
        return h

    def ttnn_prefill_forward(
        self,
        x,
        rot_mats_global=None,
        rot_mats_local=None,
        user_id=0,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        get_last_token=-1,
        kv_cache=None,
        batch_size=1,
        speaker_emb=None,
        pre_projected=False,
        **kwargs,
    ):
        """Prefill forward with text projection and optional speaker embedding.

        Args:
            pre_projected: If True, skip text_projection (input already includes
                projected text + codec embeddings from _build_input_embeds).
        """
        if not pre_projected:
            x = self.text_projection(x)

        # Add speaker embedding if provided (speaker_emb is already 2048-dim = hidden_size)
        if speaker_emb is not None:
            x = ttnn.add(x, speaker_emb)

        return super().ttnn_prefill_forward(
            x,
            rot_mats_global=rot_mats_global,
            rot_mats_local=rot_mats_local,
            user_id=user_id,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start_idx,
            get_last_token=get_last_token,
            kv_cache=kv_cache,
            batch_size=batch_size,
        )

    def ttnn_prefill_forward_with_hidden(
        self,
        x,
        rot_mats_global=None,
        rot_mats_local=None,
        user_id=0,
        page_table=None,
        chunk_page_table=None,
        get_last_token=-1,
        kv_cache=None,
        batch_size=1,
        speaker_emb=None,
        pre_projected=False,
    ):
        """Prefill forward returning both logits and pre-norm hidden state.

        Runs the standard prefill path but also captures the pre-norm hidden
        state at the last-token block, needed by Code Predictor at step 0.

        Returns:
            (tt_logits, tt_hidden_block) — logits from standard path,
            pre-norm hidden at the last-token 32-wide block (untilized on DRAM)
        """
        from models.tt_transformers.tt.model_config import TensorGroup

        if not pre_projected:
            x = self.text_projection(x)
        if speaker_emb is not None:
            x = ttnn.add(x, speaker_emb)

        for i, layer in enumerate(self.layers):
            activation_dtype = self.args.decoders_optimizations.get_tensor_dtype(
                decoder_id=i, tensor=TensorGroup.ACTIVATION
            )
            if activation_dtype is not None and x.dtype != activation_dtype:
                x = ttnn.typecast(x, activation_dtype)
            x = layer(
                x, current_pos=None,
                rot_mats_global=rot_mats_global, rot_mats_local=rot_mats_local,
                user_id=user_id, mode=Mode.PREFILL,
                page_table=page_table, chunk_page_table=chunk_page_table,
                kv_cache=kv_cache[i] if kv_cache is not None else None,
                batch_size=batch_size,
            )

        # Extract last-token block for norm + lm_head
        if get_last_token != -1:
            x_block = ttnn.slice(x, (0, 0, get_last_token, 0), (1, 1, get_last_token + 32, x.shape[-1]))
        else:
            x_block = x

        # Apply norm first — Code Predictor needs POST-NORM hidden (matches HF last_hidden_state)
        x_normed = self.norm(x_block, mode=Mode.PREFILL,
                             norm_config=self.args.get_norm_config("lm_head", Mode.PREFILL, self.prefetcher))

        # Capture POST-NORM hidden for Code Predictor
        tt_hidden_block = ttnn.untilize(x_normed, use_multicore=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        lm_head_input_mem_cfg = self.args.get_lm_head_input_mem_config(Mode.PREFILL, None)
        if lm_head_input_mem_cfg.is_sharded():
            x_normed = ttnn.interleaved_to_sharded(x_normed, lm_head_input_mem_cfg)

        tt_logits = self.lm_head(x_normed)
        tt_logits = ttnn.to_memory_config(tt_logits, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        return tt_logits, tt_hidden_block

    def ttnn_decode_forward_preembedded(self, x_embed, current_pos, rot_mat_idxs=None, page_table=None):
        """Decode forward with pre-embedded input, returning both logits and hidden state.

        Runs the Transformer layers inline (rather than calling self.forward) so we
        can capture the hidden state before norm+lm_head.  The hidden state is needed
        by the Code Predictor to generate CB1-15.

        Args:
            x_embed: ttnn.Tensor [1, 1, batch, dim] pre-embedded decode input on device
            current_pos: ttnn.Tensor position IDs
            rot_mat_idxs: ttnn.Tensor rotation matrix indices
            page_table: optional page table

        Returns:
            (tt_logits, tt_hidden) — logits after lm_head, post-norm hidden state
        """
        from models.tt_transformers.tt.model_config import TensorGroup

        rot_mats_global = self.rope_setup.get_rot_mats(rot_mat_idxs)
        rot_mats_local = self.rope_local_setup.get_rot_mats(rot_mat_idxs) if hasattr(self, "rope_local_setup") else None

        x = x_embed
        for i, layer in enumerate(self.layers):
            activation_dtype = self.args.decoders_optimizations.get_tensor_dtype(
                decoder_id=i, tensor=TensorGroup.ACTIVATION
            )
            if not self.args.is_galaxy:
                x = ttnn.to_memory_config(
                    x,
                    self.args.get_residual_mem_config(Mode.DECODE, self.prefetcher),
                    activation_dtype,
                )
            elif activation_dtype is not None and x.dtype != activation_dtype:
                x = ttnn.typecast(x, activation_dtype)

            x = layer(
                x,
                current_pos,
                rot_mats_global=rot_mats_global,
                rot_mats_local=rot_mats_local,
                mode=Mode.DECODE,
                page_table=page_table,
            )

        # Apply norm — Code Predictor needs POST-NORM hidden (matches HF last_hidden_state)
        x = self.norm(x, mode=Mode.DECODE, norm_config=self.args.get_norm_config("lm_head", Mode.DECODE, self.prefetcher))

        # Capture POST-NORM hidden for Code Predictor
        tt_hidden = ttnn.untilize(x, use_multicore=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        x = self.lm_head(x)
        tt_logits = ttnn.untilize(x, use_multicore=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        return tt_logits, tt_hidden

    def prefill(self, text_tokens, start_pos=0, page_table=None, kv_cache=None, speaker_emb=None):
        """High-level prefill: embed text on host, send to device, run forward.

        Args:
            text_tokens: torch.Tensor [batch, seq_len] text token IDs
            start_pos: starting position in KV cache
            page_table: optional paged attention table
            kv_cache: KV cache tensors
            speaker_emb: optional ttnn.Tensor [1, 1, batch, spk_enc_dim] speaker embedding on device

        Returns:
            logits: ttnn.Tensor of codec token logits
        """
        embeddings = self.embed_text_tokens(text_tokens)
        last_token_idx = text_tokens.shape[1] - 1

        tokens_embd, rot_mats, rot_mats_local, tt_page_table, tt_chunk_page_table = (
            self.prepare_inputs_prefill(
                embeddings,
                start_pos=start_pos,
                page_table=page_table,
                last_token_idx=last_token_idx,
            )
        )

        get_last_token = (last_token_idx // 32) * 32

        return self.ttnn_prefill_forward(
            tokens_embd,
            rot_mats_global=rot_mats,
            rot_mats_local=rot_mats_local,
            page_table=tt_page_table,
            chunk_page_table=tt_chunk_page_table,
            get_last_token=get_last_token,
            kv_cache=kv_cache,
            speaker_emb=speaker_emb,
        )
