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

    def forward(self, x_tt, rot_mats, current_pos=None, kv_cache=None, mask=None, mode="prefill"):
        # Attention sublayer
        normed = ttnn.rms_norm(x_tt, weight=self.attn_norm_w, epsilon=self.norm_eps)
        attn_out = self.attention.forward(
            normed, current_pos=current_pos, rot_mats=rot_mats, mode=mode, kv_cache=kv_cache, mask=mask
        )
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

    # Audio embedding table layout (from vllm-omni MultiVocabEmbeddings):
    #   2 AudioSpecialTokens: EMPTY_AUDIO=0, END_AUDIO=1
    #   Semantic codebook (8192 codes + 2 special = 8194, padded to 8320):
    #     row 0:        EMPTY_AUDIO (special)
    #     row 1:        END_AUDIO (EoA — stop generation when predicted)
    #     rows 2..8193: semantic codes 0..8191
    #   Acoustic codebooks (36 × 23 = 828, offset 8194):
    #     codebook k (0..35): rows 8194 + k*23 to 8194 + k*23 + 22
    #       row +0: EMPTY_ACOUSTIC
    #       row +1: END_ACOUSTIC
    #       rows +2..+22: acoustic levels 0..20
    #   Total: 8194 + 828 = 9022, padded to 9088.
    _N_SPECIAL = 2  # EMPTY_AUDIO, END_AUDIO
    _SEMANTIC_SIZE = 8192
    _ACOUSTIC_STRIDE = 23  # 21 levels + 2 special tokens per codebook
    _ACOUSTIC_OFFSET = 8194  # 8192 semantic + 2 special
    _EOA_TOKEN = 1  # END_AUDIO token position in semantic logits

    def _embed_audio_frame(self, sem_code_pos: int, acoustic_codes: torch.Tensor) -> torch.Tensor:
        """Map (sem_code_pos, acoustic_codes[36]) → summed 3072-dim embedding.

        sem_code_pos: raw argmax position (2..8193), already includes the +2 offset.
        acoustic_codes: [36] int tensor, levels 0..20 (will be offset by +2 for embedding).
        """
        emb = self.audio_emb_w[sem_code_pos].to(torch.float32)
        for k in range(36):
            # Level v → row 8194 + k*23 + (v+2)
            level = int(acoustic_codes[k].item())
            idx = self._ACOUSTIC_OFFSET + k * self._ACOUSTIC_STRIDE + level + self._N_SPECIAL
            if idx < self.audio_emb_w.shape[0]:
                emb = emb + self.audio_emb_w[idx].to(torch.float32)
        return emb.bfloat16()  # [D]

    def _prefill(self, inputs_embeds: torch.Tensor) -> ttnn.Tensor:
        """Prefill all layers; populate KV cache. Returns final norm hidden [1,1,S,D]."""
        _, S, D = inputs_embeds.shape
        x_tt = ttnn.from_torch(
            inputs_embeds.to(torch.bfloat16).unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        rot_mats = [self.cos_tt, self.sin_tt]
        for layer in self.layers:
            x_tt = layer.forward(x_tt, rot_mats=rot_mats, mode="prefill")
        x_tt = ttnn.rms_norm(x_tt, weight=self.norm_w, epsilon=self.config.norm_eps)
        return x_tt

    def _decode_one_step(self, frame_emb: torch.Tensor, current_pos: int) -> ttnn.Tensor:
        """Decode one token. Returns final-norm hidden [1,1,1,D] on device."""
        x_tt = ttnn.from_torch(
            frame_emb.reshape(1, 1, 1, self.config.dim).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # Decode rotation matrices at current_pos (HF concatenated-halves format)
        cos_1 = self.cos_tt[:, :, current_pos : current_pos + 1, :]
        sin_1 = self.sin_tt[:, :, current_pos : current_pos + 1, :]
        rot_mats = [cos_1, sin_1]

        cur_pos_t = ttnn.from_torch(
            torch.tensor([current_pos], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for layer in self.layers:
            x_tt = layer.forward(x_tt, rot_mats=rot_mats, current_pos=cur_pos_t, mode="decode")
        ttnn.deallocate(cur_pos_t)
        x_tt = ttnn.rms_norm(x_tt, weight=self.norm_w, epsilon=self.config.norm_eps)
        return x_tt

    def generate_tts(
        self,
        text_token_ids: torch.Tensor,  # [1, T] int64
        voice_emb: torch.Tensor,  # [1, V, 3072] bfloat16 or float32
        n_ode_steps: int = 8,
        cfg_alpha: float = 1.2,
        max_audio_frames: int = 200,
    ) -> torch.Tensor:
        """
        End-to-end TTS: text + voice → waveform (autoregressive).

        Correct input format (from mistral_common encode_speech_request):
          [BOS=1] [begin_audio=25] [voice_frames × V] [text_to_audio=36]
          [text_tokens × T] [audio_to_text=35] [begin_audio=25]

        Then autoregressive decode: one audio frame per step, fed back as input,
        until EoA (semantic code 8192) or max_audio_frames reached.

        Returns: waveform [1, N_samples] float32 at 24kHz
        """
        import torch.nn.functional as F

        V = voice_emb.shape[1]
        D = self.config.dim
        # text_token_ids must be RAW text tokens (no BOS/EOS/instruction markers).
        # Use mistral_common SpeechRequest tokenizer, not ChatCompletionRequest:
        #   tok.instruct_tokenizer.tokenizer.encode(text, bos=False, eos=False)
        # Passing ChatCompletion tokens (which include BOS=1, marker=3, EOS=4)
        # breaks the TTS format by injecting extra tokens in the text segment.
        text_emb = F.embedding(text_token_ids, self.tok_emb_w.to(torch.float32)).bfloat16()
        T = text_emb.shape[1]

        def _tok(token_id):
            return self.tok_emb_w[token_id].bfloat16().reshape(1, 1, D)

        # Full prefill: [BOS] [begin_audio] [voice×V] [text_to_audio] [text×T] [audio_to_text] [begin_audio]
        prefill_emb = torch.cat(
            [
                _tok(1),  # BOS
                _tok(25),  # begin_audio (marks start of voice reference)
                voice_emb.bfloat16(),  # V pre-computed audio frame embeddings
                _tok(36),  # text_to_audio separator
                text_emb,  # T text token embeddings
                _tok(35),  # audio_to_text separator
                _tok(25),  # begin_audio (marks start of output audio)
            ],
            dim=1,
        )
        S_prefill = prefill_emb.shape[1]

        # Prefill and keep the output: h[S_prefill-1] (last position = begin_audio)
        # predicts the FIRST audio frame — no extra decode step needed for frame 0.
        h_prefill = self._prefill(prefill_emb)
        h_last = h_prefill[:, :, -1:, :]  # [1, 1, 1, D] — hidden at last prefill position
        # (h_prefill contains all positions; h_last is a view, not a copy)

        semantic_codes_list = []
        acoustic_codes_list = []
        current_pos = S_prefill  # next frame goes at this position in the KV cache

        # Bootstrap: use the prefill's last hidden state for frame 0
        h_tt = h_last

        for frame_idx in range(max_audio_frames):
            if current_pos >= self.config.max_seq_len:
                break

            # h_tt is already computed (either from prefill or previous decode step)

            # ODE solve for one audio frame conditioned on h_tt
            acoustic_codes, x_continuous = ode_solve_ttnn(
                h_tt,
                self.acoustic_transformer,
                self.device,
                n_steps=n_ode_steps,
                cfg_alpha=cfg_alpha,
            )  # acoustic_codes: [1, 36]

            # Predict semantic token from (h_tt, x_final, t=1.0)
            x_cont_tt = ttnn.from_torch(
                x_continuous.unsqueeze(0).unsqueeze(0),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            _, sem_tt = self.acoustic_transformer.forward(h_tt, x_cont_tt, t=1.0)
            ttnn.deallocate(x_cont_tt)
            # h_tt kept alive until after frame embedding (used for acoustic transformer)

            sem_logits = ttnn.to_torch(sem_tt).squeeze().float()  # [8320]
            ttnn.deallocate(sem_tt)

            # Semantic prediction: mask EMPTY_AUDIO (row 0) and padding (rows 8194+),
            # then find argmax. Row 1 = END_AUDIO (EoA), rows 2..8193 = codes 0..8191.
            sem_logits[0] = float("-inf")  # mask EMPTY_AUDIO
            sem_logits[self._ACOUSTIC_OFFSET :] = float("-inf")  # mask padding
            sem_code_pos = int(sem_logits.argmax().item())  # raw position: 1=EoA or 2..8193
            ttnn.deallocate(h_tt)

            # Stop on End-of-Audio token
            if sem_code_pos == self._EOA_TOKEN:
                break

            semantic_codes_list.append(sem_code_pos)  # store raw position (used for embedding)
            acoustic_codes_list.append(acoustic_codes.squeeze(0))  # [36], levels 0..20

            # Embed: sem_code_pos is already the correct row in audio_emb_w
            frame_emb = self._embed_audio_frame(sem_code_pos, acoustic_codes.squeeze(0))

            # Decode step: frame_emb at position current_pos → h for next frame
            h_tt = self._decode_one_step(frame_emb, current_pos)
            current_pos += 1

        # Clean up remaining h_tt if loop exited without deallocating it
        try:
            ttnn.deallocate(h_tt)
        except Exception:
            pass
        ttnn.deallocate(h_prefill)

        if not semantic_codes_list:
            return torch.zeros(1, 1920, dtype=torch.float32)

        # Convert raw logit positions (2..8193) to actual semantic codes (0..8191)
        sem_codes = [pos - self._N_SPECIAL for pos in semantic_codes_list]
        sem_codes_batch = torch.tensor(sem_codes).unsqueeze(0)  # [1, N], values 0..8191
        # Acoustic codes are already in [0, 20] range
        aco_codes_batch = torch.stack(acoustic_codes_list, dim=0).unsqueeze(0)  # [1, N, 36]
        return self.codec_decoder.forward(sem_codes_batch, aco_codes_batch)
