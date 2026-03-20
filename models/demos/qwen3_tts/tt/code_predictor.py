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

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.qwen3_tts.tt.decoder_layer import DecoderLayer


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

        def get_cache_name(name):
            if weight_cache_path is None:
                return None
            return weight_cache_path / f"code_predictor_{name}".replace(".", "_")

        # Input projection (if Talker and CodePredictor have different hidden sizes)
        # The projection is called "small_to_mtp_projection" in HuggingFace model
        self.needs_projection = talker_hidden_size != config.hidden_size
        if self.needs_projection:
            # Project from talker hidden size to code predictor hidden size
            proj_key = "talker.code_predictor.small_to_mtp_projection.weight"
            if proj_key in state_dict:
                proj_weight = state_dict[proj_key]
                # Shape: [1024, 2048] -> transpose to [2048, 1024] for matmul
                proj_weight = torch.transpose(proj_weight, -2, -1).unsqueeze(0).unsqueeze(0)
                self.input_proj = ttnn.as_tensor(
                    proj_weight,
                    device=device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    cache_file_name=get_cache_name("input_proj"),
                    mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
                )
                # Also load bias if present
                bias_key = "talker.code_predictor.small_to_mtp_projection.bias"
                if bias_key in state_dict:
                    bias = state_dict[bias_key].unsqueeze(0).unsqueeze(0).unsqueeze(0)
                    self.input_proj_bias = ttnn.as_tensor(
                        bias,
                        device=device,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        cache_file_name=get_cache_name("input_proj_bias"),
                        mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
                    )
                else:
                    self.input_proj_bias = None
            else:
                # If no projection weight, assume sizes match or use identity
                self.needs_projection = False

        # Codec embeddings (15 embedding tables, one per code group 1-15)
        # These are stored as PyTorch tensors for embedding lookup
        self.codec_embeddings = []
        for i in range(self.num_code_groups - 1):  # 15 embeddings for codes 1-15
            embed_key = f"talker.code_predictor.model.codec_embedding.{i}.weight"
            if embed_key in state_dict:
                embed_weight = state_dict[embed_key].float()  # [vocab_size, talker_hidden_size]
                self.codec_embeddings.append(embed_weight)
            else:
                print(f"  WARNING: Missing CodePredictor embedding {embed_key}")
                self.codec_embeddings.append(None)

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
                lm_head_weight = state_dict[lm_head_key]
                lm_head_weight = torch.transpose(lm_head_weight, -2, -1).unsqueeze(0).unsqueeze(0)
                lm_head = ttnn.as_tensor(
                    lm_head_weight,
                    device=device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    cache_file_name=get_cache_name(f"lm_head_{g}"),
                    mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
                )
                self.lm_heads.append(lm_head)

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def get_codec_embedding(self, code_idx: int, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Get embedding for a specific codebook.

        Args:
            code_idx: Codebook index (0-14 for codes 1-15)
            token_ids: Token IDs to embed [batch, seq_len]

        Returns:
            Embeddings [batch, seq_len, talker_hidden_size]
        """
        if code_idx < len(self.codec_embeddings) and self.codec_embeddings[code_idx] is not None:
            return torch.nn.functional.embedding(token_ids, self.codec_embeddings[code_idx])
        else:
            raise ValueError(f"Missing codec embedding for index {code_idx}")

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

        # Input projection if needed (from Talker's 2048 dim to CodePredictor's 1024 dim)
        if self.needs_projection:
            hidden_states = ttnn.linear(
                hidden_states,
                self.input_proj,
                bias=self.input_proj_bias if hasattr(self, "input_proj_bias") else None,
                compute_kernel_config=self.compute_kernel_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

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
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
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
        # Input projection if needed (from Talker's 2048 dim to CodePredictor's 1024 dim)
        if self.needs_projection:
            hidden_states = ttnn.linear(
                hidden_states,
                self.input_proj,
                bias=self.input_proj_bias if hasattr(self, "input_proj_bias") else None,
                compute_kernel_config=self.compute_kernel_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

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
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            logits_list.append(logits)

        return logits_list, updated_kv_caches
