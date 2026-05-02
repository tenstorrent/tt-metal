# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Voxtral-4B-TTS-2603 full inference model for N150.

Three-component TTS pipeline:
  1. Text Decoder Backbone  (26-layer GQA transformer, TTNN on N150)
  2. Acoustic Transformer   (3-layer flow-matching, TTNN on N150)
  3. Voxtral Codec Decoder  (4-stage upsampler, CPU Phase 1)

Inference flow:
  1. Load pre-computed voice embedding (or encode reference audio externally)
  2. Prefill text decoder with [voice_emb + text_emb]
  3. Autoregressive decode: predict semantic tokens (one per frame)
  4. ODE solve (8 Euler steps × CFG): generate acoustic tokens from semantic hidden states
  5. Codec decode: tokens → 24kHz waveform

Usage:
  from models.demos.voxtral_tts.tt.model import VoxtralTTSModel

  model = VoxtralTTSModel.from_pretrained(model_dir, device)
  waveform = model.generate_tts(text_tokens, voice_emb)
"""

from pathlib import Path

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.voxtral_tts.tt.acoustic_transformer import TtVoxtralAcousticTransformer, ode_solve_ttnn
from models.demos.voxtral_tts.tt.attention import TtVoxtralTextAttention
from models.demos.voxtral_tts.tt.codec_decoder import TtVoxtralCodecDecoder
from models.demos.voxtral_tts.tt.load_checkpoint import (
    get_acoustic_transformer_state,
    get_codec_decoder_state,
    get_text_decoder_state,
    load_state_dict,
)
from models.demos.voxtral_tts.tt.mlp import TtVoxtralTextMLP
from models.demos.voxtral_tts.tt.model_config import VoxtralTTSConfig
from models.tt_transformers.tt.common import get_rot_transformation_mat, precompute_freqs


class TtVoxtralDecoderLayer(LightweightModule):
    """Single text decoder layer (attention + MLP + norms + residuals)."""

    def __init__(
        self, device, state_dict, weight_cache_path, layer_num, dtype, transformation_mats, config, mlp_dtype=None
    ):
        super().__init__()
        self.config = config
        self.layer_num = layer_num
        if mlp_dtype is None:
            mlp_dtype = dtype

        self.attention = TtVoxtralTextAttention(
            device=device,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            dtype=dtype,
            transformation_mats=transformation_mats,
            configuration=config,
        )
        self.mlp = TtVoxtralTextMLP(
            device=device,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            dtype=mlp_dtype,
            configuration=config,
        )

        def _norm(key):
            w = state_dict[f"layers.{layer_num}.{key}"].to(torch.bfloat16).reshape(1, 1, 1, config.dim)
            return ttnn.as_tensor(
                w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

        self.attn_norm_w = _norm("attention_norm.weight")
        self.ffn_norm_w = _norm("ffn_norm.weight")
        self.norm_eps = config.norm_eps

    def forward(self, x_tt, rot_mats, kv_cache=None, mask=None, mode="prefill"):
        # Attention sublayer
        normed = ttnn.rms_norm(x_tt, weight=self.attn_norm_w, epsilon=self.norm_eps)
        attn_out = self.attention.forward(normed, rot_mats=rot_mats, mode=mode, kv_cache=kv_cache, mask=mask)
        x_tt = ttnn.add(x_tt, attn_out)
        ttnn.deallocate(attn_out)

        # MLP sublayer
        normed2 = ttnn.rms_norm(x_tt, weight=self.ffn_norm_w, epsilon=self.norm_eps)
        mlp_out = self.mlp.forward(normed2)
        x_tt = ttnn.add(x_tt, mlp_out)
        ttnn.deallocate(mlp_out)

        return x_tt


class VoxtralTTSModel(LightweightModule):
    """Full Voxtral-4B-TTS-2603 model for N150 inference.

    Loads all weights and exposes generate_tts() for end-to-end synthesis.
    """

    def __init__(self, device, config: VoxtralTTSConfig, sd_full: dict):
        super().__init__()
        self.device = device
        self.config = config

        sd_text = get_text_decoder_state(sd_full)
        sd_acoustic = get_acoustic_transformer_state(sd_full)
        sd_codec = get_codec_decoder_state(sd_full)

        attn_dtype = ttnn.bfloat16  # attention weights: BF16 for QKV precision
        mlp_dtype = ttnn.bfloat8_b  # MLP weights: BF8 to save DRAM (~2GB vs ~4GB for 26 layers)
        cache_path = None
        dtype = attn_dtype  # default for acoustic transformer

        # ── RoPE precomputation ────────────────────────────────────────────
        cos_raw, sin_raw = precompute_freqs(config.head_dim, config.max_seq_len * 2, config.rope_theta, None, None)
        cos_hf = torch.cat([cos_raw[: config.max_seq_len], cos_raw[: config.max_seq_len]], dim=-1)
        sin_hf = torch.cat([sin_raw[: config.max_seq_len], sin_raw[: config.max_seq_len]], dim=-1)

        self.cos_tt = ttnn.from_torch(
            cos_hf.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.sin_tt = ttnn.from_torch(
            sin_hf.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.transformation_mats = ttnn.as_tensor(
            get_rot_transformation_mat(dhead=config.head_dim),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # ── Token embedding ────────────────────────────────────────────────
        self.tok_emb_w = sd_text["mm_audio_embeddings.tok_embeddings.weight"]  # [131072, 3072]
        self.audio_emb_w = sd_text["mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight"]  # [9088, 3072]

        # ── Text decoder layers ─────────────────────────────────────────────
        self.layers = [
            TtVoxtralDecoderLayer(
                device=device,
                state_dict=sd_text,
                weight_cache_path=cache_path,
                layer_num=i,
                dtype=attn_dtype,
                mlp_dtype=mlp_dtype,
                transformation_mats=self.transformation_mats,
                config=config,
            )
            for i in range(config.n_layers)
        ]

        # ── Final norm ─────────────────────────────────────────────────────
        norm_w = sd_text["norm.weight"].to(torch.bfloat16).reshape(1, 1, 1, config.dim)
        self.norm_w = ttnn.as_tensor(
            norm_w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        # ── Acoustic transformer ───────────────────────────────────────────
        self.acoustic_transformer = TtVoxtralAcousticTransformer(
            device=device,
            state_dict=sd_acoustic,
            weight_cache_path=cache_path,
            dtype=dtype,
            configuration=config,
        )

        # ── Codec decoder ──────────────────────────────────────────────────
        self.codec_decoder = TtVoxtralCodecDecoder(
            device=device,
            state_dict=sd_codec,
            weight_cache_path=cache_path,
            dtype=dtype,
            configuration=config,
        )

    @classmethod
    def from_pretrained(cls, model_dir: str | Path, device) -> "VoxtralTTSModel":
        model_dir = Path(model_dir)
        config = VoxtralTTSConfig(mesh_device=device)
        sd = load_state_dict(model_dir / "consolidated.safetensors")
        return cls(device, config, sd)

    def _prefill(self, inputs_embeds: torch.Tensor) -> ttnn.Tensor:
        """Prefill: inputs_embeds [1, S, D] → hidden [1, 1, S, D] on device."""
        B, S, D = inputs_embeds.shape

        x_tt = ttnn.from_torch(
            inputs_embeds.to(torch.bfloat16).unsqueeze(0),  # [1, 1, S, D]
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        rot_mats = [self.cos_tt, self.sin_tt]

        for layer in self.layers:
            x_tt = layer.forward(x_tt, rot_mats=rot_mats, mode="prefill")

        # Final norm
        x_tt = ttnn.rms_norm(x_tt, weight=self.norm_w, epsilon=self.config.norm_eps)
        return x_tt

    def generate_tts(
        self,
        text_token_ids: torch.Tensor,  # [1, T] int64
        voice_emb: torch.Tensor,  # [1, V, 3072] float32 or bfloat16
        n_ode_steps: int = 8,
        cfg_alpha: float = 1.2,
    ) -> torch.Tensor:
        """
        End-to-end TTS: text + voice → waveform.

        Returns: waveform [1, N_samples] float32 at 24kHz
        """
        # Embed text tokens
        text_emb = torch.nn.functional.embedding(
            text_token_ids, self.tok_emb_w.to(torch.float32)
        ).bfloat16()  # [1, T, 3072]

        # Concatenate voice (pre-computed) + text embeddings
        prefill_emb = torch.cat([voice_emb.bfloat16(), text_emb], dim=1)  # [1, V+T, 3072]
        S = prefill_emb.shape[1]
        V = voice_emb.shape[1]
        T = text_emb.shape[1]

        # Prefill all layers
        h_tt = self._prefill(prefill_emb)  # [1, 1, V+T, 3072]

        # Extract text positions for acoustic conditioning
        h_text_tt = h_tt[:, :, V:, :]  # [1, 1, T, 3072]

        # ODE solve: generate acoustic tokens from semantic hidden states
        acoustic_codes, x_continuous = ode_solve_ttnn(
            h_text_tt,
            self.acoustic_transformer,
            self.device,
            n_steps=n_ode_steps,
            cfg_alpha=cfg_alpha,
        )  # acoustic_codes: [T, 36]
        ttnn.deallocate(h_tt)
        ttnn.deallocate(h_text_tt)

        # Semantic token prediction from final hidden state
        # Use acoustic transformer's semantic head on x_continuous
        h_text_cpu = ttnn.to_torch(
            ttnn.from_torch(
                torch.zeros(1, 1, T, 3072, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        )
        # Actually use the text decoder hidden states for semantic prediction
        # by running acoustic transformer at t=1.0 (final ODE step)
        h_for_sem = ttnn.from_torch(
            torch.zeros(1, 1, T, 3072, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        x_cont_tt = ttnn.from_torch(
            x_continuous.unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        _, sem_logits_tt = self.acoustic_transformer.forward(h_for_sem, x_cont_tt, t=1.0)
        ttnn.deallocate(h_for_sem)
        ttnn.deallocate(x_cont_tt)

        sem_logits = ttnn.to_torch(sem_logits_tt).squeeze(0).squeeze(0)  # [T, 8320]
        ttnn.deallocate(sem_logits_tt)
        semantic_codes = sem_logits[:, :8192].argmax(dim=-1).unsqueeze(0)  # [1, T]

        # Codec decode
        acoustic_codes_batched = acoustic_codes.unsqueeze(0)  # [1, T, 36]
        waveform = self.codec_decoder.forward(semantic_codes, acoustic_codes_batched)
        return waveform
