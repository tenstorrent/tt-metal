# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
CosyVoice LLM Backbone - Transformer for semantic token generation.

The LLM backbone takes text token embeddings and generates semantic speech
tokens autoregressively. It uses a Qwen2-style transformer architecture
with KV-cache support for efficient decoding.
"""

from typing import Generator, List, Optional, Tuple

import torch
import ttnn
from loguru import logger

from models.demos.wormhole.cosyvoice.tt.transformer.transformer import (
    TtLayerNorm,
    TtTransformerBlock,
)


class TtCosyVoiceLLM:
    """CosyVoice LLM backbone on TT hardware.

    Generates semantic speech tokens from text using a transformer decoder.
    Supports autoregressive decoding with KV-cache.
    """

    def __init__(
        self,
        device: ttnn.Device,
        config,
        state_dict: dict,
        tt_cache_path: Optional[str] = None,
    ):
        self.device = device
        self.config = config
        self.num_layers = config.llm_num_layers
        self.hidden_size = config.llm_hidden_size
        self.num_heads = config.llm_num_heads
        self.head_dim = config.llm_head_dim
        self.intermediate_size = config.llm_intermediate_size
        self.speech_token_size = config.speech_token_size
        self.vocab_size = config.text_token_size

        # Special tokens
        self.sos_token_id = 0       # Start of sequence
        self.task_id = 1            # Task identifier
        self.eos_token_id = config.speech_token_size  # End of sequence

        memory_config = ttnn.DRAM_MEMORY_CONFIG
        dtype = ttnn.bfloat16

        logger.info(f"Initializing CosyVoice LLM backbone: {self.num_layers} layers, "
                     f"{self.hidden_size} hidden, {self.num_heads} heads")

        # Embedding layers
        # Text embedding: Qwen2's embedding table
        text_embed_weight = state_dict["llm.model.model.embed_tokens.weight"]
        self.text_embedding = ttnn.as_tensor(
            text_embed_weight.unsqueeze(0),
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=memory_config,
            dtype=dtype,
        )

        # Speech token embedding
        speech_embed_weight = state_dict.get(
            "llm.speech_embedding.weight",
            torch.zeros(config.speech_token_size + 3, self.hidden_size),
        )
        self.speech_embedding = ttnn.as_tensor(
            speech_embed_weight.unsqueeze(0),
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=memory_config,
            dtype=dtype,
        )

        # LLM special token embeddings (sos, task_id)
        llm_embed_weight = state_dict.get(
            "llm.llm_embedding.weight",
            torch.zeros(2, self.hidden_size),
        )
        self.llm_embedding = ttnn.as_tensor(
            llm_embed_weight.unsqueeze(0),
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=memory_config,
            dtype=dtype,
        )

        # LLM decoder (linear projection to vocab)
        llm_decoder_weight = state_dict.get(
            "llm.llm_decoder.weight",
            torch.zeros(self.hidden_size, config.speech_token_size + 3),
        )
        llm_decoder_bias = state_dict.get(
            "llm.llm_decoder.bias",
            None,
        )
        self.llm_decoder_weight = ttnn.as_tensor(
            llm_decoder_weight.unsqueeze(0).unsqueeze(0),
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=memory_config,
            dtype=dtype,
        )
        if llm_decoder_bias is not None:
            self.llm_decoder_bias = ttnn.as_tensor(
                llm_decoder_bias.unsqueeze(0).unsqueeze(0),
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=memory_config,
                dtype=dtype,
            )
        else:
            self.llm_decoder_bias = None

        # Spk embedding affine layer
        spk_weight = state_dict.get(
            "llm.spk_embed_affine_layer.weight",
            torch.zeros(192, self.hidden_size),
        )
        spk_bias = state_dict.get(
            "llm.spk_embed_affine_layer.bias",
            None,
        )
        self.spk_embed_affine = TtLayerNorm(
            device, self.hidden_size, spk_weight, spk_bias,
            memory_config=memory_config, dtype=dtype,
        ) if False else None  # Using linear projection instead

        self.spk_weight = ttnn.as_tensor(
            spk_weight.T.unsqueeze(0).unsqueeze(0),
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=memory_config,
            dtype=dtype,
        )

        # Text encoder
        # For simplicity in Stage 1, we use a single linear layer as text encoder
        # Full text encoder (Qwen2) will be ported in Stage 2
        text_encoder_weight = state_dict.get(
            "llm.text_encoder_affine_layer.weight",
            torch.zeros(self.hidden_size, self.hidden_size),
        )
        text_encoder_bias = state_dict.get(
            "llm.text_encoder_affine_layer.bias",
            None,
        )
        self.text_encoder_weight = ttnn.as_tensor(
            text_encoder_weight.T.unsqueeze(0).unsqueeze(0),
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=memory_config,
            dtype=dtype,
        )
        if text_encoder_bias is not None:
            self.text_encoder_bias = ttnn.as_tensor(
                text_encoder_bias.unsqueeze(0).unsqueeze(0),
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=memory_config,
                dtype=dtype,
            )
        else:
            self.text_encoder_bias = None

        # Transformer layers
        self.layers = []
        for i in range(self.num_layers):
            layer_prefix = f"llm.llm.model.model.layers.{i}"
            attn_params = {
                "q_weight": state_dict[f"{layer_prefix}.self_attn.q_proj.weight"],
                "k_weight": state_dict.get(f"{layer_prefix}.self_attn.k_proj.weight",
                                           state_dict[f"{layer_prefix}.self_attn.q_proj.weight"]),
                "v_weight": state_dict.get(f"{layer_prefix}.self_attn.v_proj.weight",
                                           state_dict[f"{layer_prefix}.self_attn.q_proj.weight"]),
                "o_weight": state_dict[f"{layer_prefix}.self_attn.o_proj.weight"],
            }
            ffn_params = {
                "gate_weight": state_dict[f"{layer_prefix}.mlp.gate_proj.weight"],
                "up_weight": state_dict[f"{layer_prefix}.mlp.up_proj.weight"],
                "down_weight": state_dict[f"{layer_prefix}.mlp.down_proj.weight"],
            }
            norm_weight = state_dict[f"{layer_prefix}.input_layernorm.weight"]
            norm_bias = state_dict.get(f"{layer_prefix}.input_layernorm.bias", None)
            post_norm_weight = state_dict.get(f"{layer_prefix}.post_attention_layernorm.weight", None)

            layer = TtTransformerBlock(
                device, i, self.hidden_size, self.num_heads, self.head_dim,
                self.intermediate_size, attn_params, ffn_params,
                norm_weight, norm_bias, post_norm_weight,
                memory_config=memory_config, dtype=dtype,
            )
            self.layers.append(layer)

        # Final layer norm
        final_norm_weight = state_dict.get(
            "llm.llm.model.model.norm.weight",
            torch.ones(self.hidden_size),
        )
        self.final_norm = TtLayerNorm(
            device, self.hidden_size, final_norm_weight, None,
            memory_config=memory_config, dtype=dtype,
        )

        # KV cache (initialized on first call)
        self.kv_cache = [None] * self.num_layers

    def _embed_text(self, text_tokens: ttnn.Tensor) -> ttnn.Tensor:
        """Embed text tokens using the text embedding table."""
        return ttnn.embedding(text_tokens, self.text_embedding)

    def _embed_speech(self, speech_tokens: ttnn.Tensor) -> ttnn.Tensor:
        """Embed speech tokens using the speech embedding table."""
        return ttnn.embedding(speech_tokens, self.speech_embedding)

    def _encode_text(self, text_embeds: ttnn.Tensor) -> ttnn.Tensor:
        """Encode text embeddings through the text encoder."""
        x = ttnn.linear(text_embeds, self.text_encoder_weight)
        if self.text_encoder_bias is not None:
            x = ttnn.add(x, self.text_encoder_bias)
        return x

    def _encode_speaker(self, speaker_embedding: ttnn.Tensor) -> ttnn.Tensor:
        """Encode speaker embedding."""
        return ttnn.linear(speaker_embedding, self.spk_weight)

    def _decode_logits(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Project transformer output to vocabulary logits."""
        x = ttnn.linear(x, self.llm_decoder_weight)
        if self.llm_decoder_bias is not None:
            x = ttnn.add(x, self.llm_decoder_bias)
        return x

    def prefill(
        self,
        text_tokens: ttnn.Tensor,
        speaker_embedding: Optional[ttnn.Tensor] = None,
        prompt_speech_tokens: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """Run prefill step: process all input tokens at once.

        Args:
            text_tokens: Text token IDs [batch, seq_len]
            speaker_embedding: Optional speaker embedding [batch, 1, spk_dim]
            prompt_speech_tokens: Optional prompt speech tokens [batch, prompt_len]

        Returns:
            Logits tensor [batch, 1, vocab_size] for the next token
        """
        # Embed text tokens
        x = self._embed_text(text_tokens)

        # Encode text
        x = self._encode_text(x)

        # Add speaker embedding
        if speaker_embedding is not None:
            spk = ttnn.linear(speaker_embedding, self.spk_weight)
            spk = ttnn.unsqueeze(spk, 1)  # [batch, 1, hidden]
            # Pad speaker embedding to match sequence length
            spk = ttnn.repeat(spk, (1, x.shape[1], 1))
            x = ttnn.add(x, spk)

        # Add prompt speech tokens if provided
        if prompt_speech_tokens is not None:
            speech_embeds = self._embed_speech(prompt_speech_tokens)
            x = ttnn.concat((x, speech_embeds), dim=1)

        # Prepend SOS token
        sos = ttnn.embedding(
            ttnn.full((x.shape[0], 1), self.sos_token_id, dtype=ttnn.uint32),
            self.llm_embedding,
        )

        # Append task ID token
        task = ttnn.embedding(
            ttnn.full((x.shape[0], 1), self.task_id, dtype=ttnn.uint32),
            self.llm_embedding,
        )

        x = ttnn.concat((sos, x, task), dim=1)

        # Run through transformer layers
        for i, layer in enumerate(self.layers):
            x = layer(x)

        # Final layer norm
        x = self.final_norm(x)

        # Get logits for the last position
        last_hidden = x[:, -1:, :]
        logits = self._decode_logits(last_hidden)

        return logits

    def decode_step(
        self,
        token_embed: ttnn.Tensor,
        step_idx: int,
    ) -> ttnn.Tensor:
        """Run a single autoregressive decoding step.

        Args:
            token_embed: Embedded token [batch, 1, hidden_size]
            step_idx: Current decoding step index

        Returns:
            Logits tensor [batch, 1, vocab_size]
        """
        x = token_embed

        # Run through transformer layers with KV cache
        for i, layer in enumerate(self.layers):
            kv = self.kv_cache[i]
            x = layer(x, kv_cache=kv)
            # Update KV cache
            if kv is not None:
                self.kv_cache[i] = (ttnn.concat((kv[0], x), dim=1),
                                    ttnn.concat((kv[1], x), dim=1))
            else:
                self.kv_cache[i] = (x, x)

        # Final layer norm
        x = self.final_norm(x)

        # Decode logits
        logits = self._decode_logits(x)

        return logits

    @torch.inference_mode()
    def inference(
        self,
        text: torch.Tensor,
        text_len: torch.Tensor,
        prompt_text: torch.Tensor,
        prompt_text_len: torch.Tensor,
        prompt_speech_token: torch.Tensor,
        prompt_speech_token_len: torch.Tensor,
        embedding: torch.Tensor,
        sampling: int = 25,
        max_token_text_ratio: float = 20,
        min_token_text_ratio: float = 2,
    ) -> Generator[torch.Tensor, None, None]:
        """Run full inference: text -> semantic tokens.

        This is the main entry point for the LLM backbone. It handles
        both prefill and autoregressive decoding.

        Args:
            text: Input text token IDs [batch, seq_len]
            text_len: Input text length [batch]
            prompt_text: Prompt text token IDs [batch, prompt_len]
            prompt_text_len: Prompt text length [batch]
            prompt_speech_token: Prompt speech tokens [batch, prompt_speech_len]
            prompt_speech_token_len: Prompt speech token length [batch]
            embedding: Speaker embedding [batch, 1, spk_dim]
            sampling: Top-k sampling parameter
            max_token_text_ratio: Max generated tokens / text tokens ratio
            min_token_text_ratio: Min generated tokens / text tokens ratio

        Yields:
            Generated token IDs one at a time
        """
        device = text.device

        # Concatenate prompt text and input text
        concat_text = torch.cat([prompt_text, text], dim=1)
        total_text_len = text_len + prompt_text_len

        # Calculate generation length bounds
        min_len = int((total_text_len - prompt_text_len) * min_token_text_ratio)
        max_len = int((total_text_len - prompt_text_len) * max_token_text_ratio)

        # Convert to TTNN tensors
        tt_text = ttnn.from_torch(concat_text, device=self.device, dtype=ttnn.uint32)

        # Prefill: process all input tokens
        logits = self.prefill(
            tt_text,
            speaker_embedding=ttnn.from_torch(embedding, device=self.device, dtype=ttnn.bfloat16)
                if embedding.shape[1] > 0 else None,
            prompt_speech_tokens=ttnn.from_torch(prompt_speech_token, device=self.device, dtype=ttnn.uint32)
                if prompt_speech_token.shape[1] > 0 else None,
        )

        # Get first token
        logits_torch = ttnn.to_torch(logits)
        next_token = self._sample(logits_torch, sampling, ignore_eos=True)

        out_tokens = []
        for step in range(max_len):
            if next_token == self.eos_token_id:
                break

            yield next_token
            out_tokens.append(next_token)

            # Check if we should start paying attention to EOS
            ignore_eos = step < min_len

            # Embed the generated token and run decode step
            token_tensor = torch.tensor([[next_token]], device=device)
            tt_token = ttnn.from_torch(token_tensor, device=self.device, dtype=ttnn.uint32)

            if hasattr(self, 'speech_embedding') and self.speech_embedding is not None:
                token_embed = self._embed_speech(tt_token)
            else:
                token_embed = self._embed_text(tt_token)

            logits = self.decode_step(token_embed, step)
            logits_torch = ttnn.to_torch(logits)
            next_token = self._sample(logits_torch, sampling, ignore_eos)

    def _sample(
        self,
        logits: torch.Tensor,
        top_k: int = 25,
        ignore_eos: bool = True,
    ) -> int:
        """Sample next token from logits using top-k sampling."""
        logits = logits[0, -1, :].float()

        # Mask EOS token if needed
        if ignore_eos:
            logits[self.eos_token_id] = float('-inf')

        # Top-k filtering
        if top_k > 0:
            top_k_values, _ = torch.topk(logits, top_k)
            threshold = top_k_values[-1]
            logits[logits < threshold] = float('-inf')

        # Sample
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()

        return next_token

    def reset_kv_cache(self):
        """Reset the KV cache for a new inference sequence."""
        self.kv_cache = [None] * self.num_layers
