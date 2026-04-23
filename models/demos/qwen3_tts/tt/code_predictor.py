# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
CodePredictor model implementation for Qwen3-TTS.

The CodePredictor is a 5-layer transformer decoder that takes hidden states
from the Talker and predicts audio codec tokens for multiple code groups (1-15).

IMPORTANT: CodePredictor generates codes 1-15 AUTOREGRESSIVELY:
- Each code group has its own embedding table and LM head
- Generation: input = [past_hidden, code0_embed] -> predict code 1
              input = code1_embed -> predict code 2
              ...
              input = code14_embed -> predict code 15

The architecture matches the official qwen_tts implementation.
"""

from typing import List, Optional, Tuple

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.qwen3_tts.tt.decoder_layer import DecoderLayer
from models.demos.qwen3_tts.tt.model_config import (
    code_predictor_decode_linear_output_memory_config,
    get_device_core_grid,
    restore_code_predictor_linear_output_to_dram,
)


class CodePredictor(LightweightModule):
    """
    Qwen3-TTS CodePredictor model.

    Architecture:
        - Input projection (from Talker hidden_size to CodePredictor hidden_size)
        - 15 codec embedding tables (one per code group 1-15)
        - 5 decoder layers with standard RoPE
        - 15 LM heads (one per code group 1-15)

    CRITICAL: Codes 1-15 are generated autoregressively, NOT in parallel!

    Args:
        device: TTNN device
        config: CodePredictor configuration (Qwen3TTSCodePredictorConfig)
        talker_hidden_size: Hidden size of the Talker model (for input projection)
        state_dict: Model weights
        weight_cache_path: Optional path for weight caching
    """

    def __init__(
        self,
        device,
        config,
        talker_hidden_size: int,
        state_dict: dict,
        weight_cache_path=None,
    ):
        super().__init__()
        self.device = device
        self.config = config
        self.hidden_size = config.hidden_size
        self.talker_hidden_size = talker_hidden_size
        self.num_layers = config.num_hidden_layers
        self.num_code_groups = config.num_code_groups
        self.vocab_size = config.vocab_size

        is_mesh_device = device.__class__.__name__ == "MeshDevice"
        _mesh_mapper = ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None
        _dram = ttnn.DRAM_MEMORY_CONFIG

        def _linear_weight_to_matmul_4d_ttnn(w_2d) -> ttnn.Tensor:
            """Checkpoint linear weight [out, in] -> device [1, 1, in, out] TILE for ttnn.linear (transpose on DRAM, no host round-trip)."""
            out_f, in_f = int(w_2d.shape[0]), int(w_2d.shape[1])
            w_tt = ttnn.from_torch(
                w_2d,
                device=device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=_dram,
                mesh_mapper=_mesh_mapper,
            )
            w_tx = ttnn.transpose(w_tt, -2, -1, memory_config=_dram)
            ttnn.deallocate(w_tt)
            w_4d = ttnn.reshape(w_tx, [1, 1, in_f, out_f], memory_config=_dram)
            out_w = ttnn.clone(w_4d, memory_config=_dram, dtype=ttnn.bfloat16)
            ttnn.deallocate(w_4d)
            ttnn.deallocate(w_tx)
            return out_w

        # Input projection (if Talker and CodePredictor have different hidden sizes)
        # The projection is called "small_to_mtp_projection" in HuggingFace model
        self.needs_projection = talker_hidden_size != config.hidden_size
        if self.needs_projection:
            # Project from talker hidden size to code predictor hidden size
            proj_key = "talker.code_predictor.small_to_mtp_projection.weight"
            if proj_key in state_dict:
                self.input_proj = _linear_weight_to_matmul_4d_ttnn(state_dict[proj_key])
                # Also load bias if present
                bias_key = "talker.code_predictor.small_to_mtp_projection.bias"
                if bias_key in state_dict:
                    b = state_dict[bias_key]
                    bias_tt = ttnn.from_torch(
                        b,
                        device=device,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        memory_config=_dram,
                        mesh_mapper=_mesh_mapper,
                    )
                    h = int(bias_tt.shape[0])
                    self.input_proj_bias = ttnn.reshape(bias_tt, [1, 1, 1, h], memory_config=_dram)
                else:
                    self.input_proj_bias = None
            else:
                # If no projection weight, assume sizes match or use identity
                self.needs_projection = False

        # Codec embeddings (15 tables, one per code group 1-15); device tensors only
        self.codec_embeddings_tt: List[Optional[ttnn.Tensor]] = []
        for i in range(self.num_code_groups - 1):  # 15 embeddings for codes 1-15
            embed_key = f"talker.code_predictor.model.codec_embedding.{i}.weight"
            if embed_key in state_dict:
                w = state_dict[embed_key]
                vocab_size, emb_dim = int(w.shape[0]), int(w.shape[1])
                embed_tt = ttnn.from_torch(
                    w,
                    device=device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=_dram,
                    mesh_mapper=_mesh_mapper,
                )
                embed_4d = ttnn.reshape(embed_tt, [1, 1, vocab_size, emb_dim], memory_config=_dram)
                self.codec_embeddings_tt.append(embed_4d)
            else:
                print(f"  WARNING: Missing CodePredictor embedding {embed_key}")
                self.codec_embeddings_tt.append(None)

        # Decoder layers
        self.layers = []
        for i in range(self.num_layers):
            layer = DecoderLayer(
                device=device,
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                intermediate_size=config.intermediate_size,
                state_dict=state_dict,
                layer_idx=i,
                layer_prefix="talker.code_predictor.model",
                rms_norm_eps=config.rms_norm_eps,
                weight_dtype=ttnn.bfloat16,
                weight_cache_path=weight_cache_path,
            )
            self.layers.append(layer)

        # LM heads (15 heads, one per code group 1-15)
        # Note: lm_head[0] predicts code 1, lm_head[14] predicts code 15
        self.lm_heads = []
        for g in range(self.num_code_groups - 1):  # 15 LM heads
            lm_head_key = f"talker.code_predictor.lm_head.{g}.weight"
            if lm_head_key in state_dict:
                self.lm_heads.append(_linear_weight_to_matmul_4d_ttnn(state_dict[lm_head_key]))

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self._matmul_core_grid = get_device_core_grid(device)

    def get_codec_embedding(self, code_idx: int, token_ids_tt: ttnn.Tensor) -> ttnn.Tensor:
        """
        Look up codec embeddings for one codebook on device.

        Args:
            code_idx: Codebook index (0-14 for codes 1-15)
            token_ids_tt: Token IDs (TTNN)

        Returns:
            Embeddings [batch, 1, seq_len, hidden_size] (tile layout)
        """
        if code_idx < len(self.codec_embeddings_tt) and self.codec_embeddings_tt[code_idx] is not None:
            return ttnn.embedding(
                token_ids_tt,
                self.codec_embeddings_tt[code_idx],
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        raise ValueError(f"Missing TTNN codec embedding for index {code_idx}")

    def forward_single_step(
        self,
        inputs_embeds: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        transformation_mat: ttnn.Tensor,
        generation_step: int,
        attention_mask: ttnn.Tensor = None,
        kv_caches: Optional[List[Tuple[ttnn.Tensor, ttnn.Tensor]]] = None,
        start_pos: int = 0,
        mode: str = "prefill",
        cur_pos_tensor: Optional[ttnn.Tensor] = None,
        decode_attn_mask: Optional[ttnn.Tensor] = None,
        cp_prefill_mask: Optional[ttnn.Tensor] = None,
        return_hidden_state: bool = False,
    ) -> Tuple[ttnn.Tensor, Optional[List[Tuple[ttnn.Tensor, ttnn.Tensor]]]]:
        """
        Single forward step for autoregressive generation.

        Args:
            inputs_embeds: Input embeddings [batch, 1, seq_len, hidden_size]
            cos: Cosine frequencies for RoPE
            sin: Sine frequencies for RoPE
            transformation_mat: Transformation matrix for RoPE
            generation_step: Which code we're predicting (1-15, so use lm_head[generation_step-1])
            attention_mask: Optional attention mask
            kv_caches: Optional list of (k_cache, v_cache) tuples, one per layer
            start_pos: Starting position in sequence (for KV cache)
            mode: "prefill" for full sequence or "decode" for single token
            cur_pos_tensor: Optional int32 device tensor [1] for trace-compatible decode.
            decode_attn_mask: Optional float32 device tensor [1,1,1,max_seq] for decode.
            cp_prefill_mask: Optional float32 device tensor [1,1,seq,max_seq] for trace-
                compatible CP prefill (writes K/V at constant positions 0,1).
            return_hidden_state: If True, return hidden_states instead of applying lm_head.
                Used for trace paths where the lm_head is applied outside the trace.

        Returns:
            Tuple of (output, updated_kv_caches) where:
            - output: logits [batch, 1, seq_len, vocab_size] when return_hidden_state=False,
                      or hidden_states [batch, 1, seq_len, hidden_size] when True.
            - updated_kv_caches: list of (k_cache, v_cache) tuples or None
        """
        hidden_states = inputs_embeds
        _linear_mem = code_predictor_decode_linear_output_memory_config(mode)

        # Input projection if needed (from Talker's 2048 dim to CodePredictor's 1024 dim)
        if self.needs_projection:
            hidden_states = ttnn.linear(
                hidden_states,
                self.input_proj,
                bias=self.input_proj_bias if hasattr(self, "input_proj_bias") else None,
                compute_kernel_config=self.compute_kernel_config,
                memory_config=_linear_mem,
                core_grid=self._matmul_core_grid,
            )
            if mode == "decode":
                hidden_states = restore_code_predictor_linear_output_to_dram(hidden_states)

        # Apply decoder layers
        updated_kv_caches = [] if kv_caches is not None else None
        for i, layer in enumerate(self.layers):
            layer_kv_cache = kv_caches[i] if kv_caches is not None else None
            hidden_states, updated_kv_cache = layer(
                hidden_states,
                cos,
                sin,
                transformation_mat,
                attention_mask,
                kv_cache=layer_kv_cache,
                start_pos=start_pos,
                mode=mode,
                cur_pos_tensor=cur_pos_tensor,
                decode_attn_mask=decode_attn_mask,
                cp_prefill_mask=cp_prefill_mask,
            )
            if updated_kv_caches is not None:
                updated_kv_caches.append(updated_kv_cache)

        if return_hidden_state:
            return hidden_states, updated_kv_caches

        # Use the appropriate LM head for this generation step
        # generation_step is 1-indexed (1 for code 1, 15 for code 15)
        lm_head_idx = generation_step - 1
        if lm_head_idx < len(self.lm_heads):
            logits = ttnn.linear(
                hidden_states,
                self.lm_heads[lm_head_idx],
                compute_kernel_config=self.compute_kernel_config,
                memory_config=_linear_mem,
                core_grid=self._matmul_core_grid,
            )
            if mode == "decode":
                logits = restore_code_predictor_linear_output_to_dram(logits)
        else:
            raise ValueError(f"Invalid generation_step {generation_step}, only have {len(self.lm_heads)} LM heads")

        return logits, updated_kv_caches

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        transformation_mat: ttnn.Tensor,
        attention_mask: ttnn.Tensor = None,
        kv_caches: Optional[List[Tuple[ttnn.Tensor, ttnn.Tensor]]] = None,
        start_pos: int = 0,
        mode: str = "prefill",
    ) -> Tuple[List[ttnn.Tensor], Optional[List[Tuple[ttnn.Tensor, ttnn.Tensor]]]]:
        """
        Forward pass of the CodePredictor model (parallel version - for prefill/testing).

        NOTE: This applies all 15 LM heads to the same hidden state, which is only
        correct for prefill when the input contains the full [past_hidden, code0, code1, ..., code14]
        sequence. For generation, use forward_single_step() instead.

        Args:
            hidden_states: Input hidden states from Talker [batch, 1, seq_len, talker_hidden_size]
            cos: Cosine frequencies for RoPE
            sin: Sine frequencies for RoPE
            transformation_mat: Transformation matrix for RoPE
            attention_mask: Optional attention mask
            kv_caches: Optional list of (k_cache, v_cache) tuples, one per layer
            start_pos: Starting position in sequence (for KV cache)
            mode: "prefill" for full sequence or "decode" for single token

        Returns:
            Tuple of (logits_list, updated_kv_caches) where:
            - logits_list: List of logits tensors, one per code group [batch, 1, seq_len, vocab_size]
            - updated_kv_caches: list of (k_cache, v_cache) tuples or None
        """
        _linear_mem_pf = code_predictor_decode_linear_output_memory_config(mode)

        # Input projection if needed (from Talker's 2048 dim to CodePredictor's 1024 dim)
        if self.needs_projection:
            hidden_states = ttnn.linear(
                hidden_states,
                self.input_proj,
                bias=self.input_proj_bias if hasattr(self, "input_proj_bias") else None,
                compute_kernel_config=self.compute_kernel_config,
                memory_config=_linear_mem_pf,
                core_grid=self._matmul_core_grid,
            )
            if mode == "decode":
                hidden_states = restore_code_predictor_linear_output_to_dram(hidden_states)

        # Apply decoder layers
        updated_kv_caches = [] if kv_caches is not None else None
        for i, layer in enumerate(self.layers):
            layer_kv_cache = kv_caches[i] if kv_caches is not None else None
            hidden_states, updated_kv_cache = layer(
                hidden_states,
                cos,
                sin,
                transformation_mat,
                attention_mask,
                kv_cache=layer_kv_cache,
                start_pos=start_pos,
                mode=mode,
            )
            if updated_kv_caches is not None:
                updated_kv_caches.append(updated_kv_cache)

        # Compute logits for each code group (parallel - only correct for prefill)
        logits_list = []
        for lm_head in self.lm_heads:
            logits = ttnn.linear(
                hidden_states,
                lm_head,
                compute_kernel_config=self.compute_kernel_config,
                memory_config=_linear_mem_pf,
                core_grid=self._matmul_core_grid,
            )
            if mode == "decode":
                logits = restore_code_predictor_linear_output_to_dram(logits)
            logits_list.append(logits)

        return logits_list, updated_kv_caches
