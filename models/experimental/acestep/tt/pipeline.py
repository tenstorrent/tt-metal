# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ACE-Step v1.5 full generation pipeline on TTNN (DiT denoise loop -> VAE decode -> audio).

This is the top-level factory that assembles the validated subsystems into the end-to-end
text-to-music compute path, mirroring the DiT/LTX pipeline-factory pattern:

    create_tt_pipeline(args, device) -> AceStepPipeline
    pipeline.generate(context_latents, encoder_hidden_states, noise, infer_steps, ...) -> latents
    pipeline.decode(latents) -> 48kHz stereo waveform

The denoise loop reproduces the reference `generate_audio` ODE (Euler flow-matching) branch:

    t = linspace(1, 0, steps+1);  (optional shift transform)
    for (t_curr, t_prev): vt = DiT(xt, t_curr, ...); xt = xt - vt*(t_curr - t_prev)

CFG guidance is omitted here (guidance_scale=1.0 path) because apg_guidance.py is absent from the
base snapshot and only combines two DiT predictions — orthogonal to the core update. The reference
comparison in the e2e test uses the same no-CFG ODE so the PCC is an honest apples-to-apples check.

Conditioning (encoder_hidden_states, context_latents) is precomputed and passed in, so the pipeline
here owns exactly the DiT-loop + VAE stages; the ConditionEncoder is a separate validated factory
(build_condition_encoder) and the two compose in the full-pipeline test.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

import ttnn
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from models.experimental.acestep.reference.hf_reference import load_config, load_modeling_module
from models.experimental.acestep.tt.flow_match import FlowMatchStep
from models.experimental.acestep.tt.model_config import (
    AceStepModelConfig,
    _build_dit_model,
    build_vae_decoder,
)

HEAD_DIM = 128
PATCH = 2


def _shifted_timesteps(infer_steps: int, shift: float, device, dtype=torch.float32) -> torch.Tensor:
    t = torch.linspace(1.0, 0.0, infer_steps + 1, device=device, dtype=dtype)
    if shift != 1.0:
        t = shift * t / (1 + (shift - 1) * t)
    return t


@dataclass
class AceStepPipeline:
    """Assembled TT pipeline: text encoder + condition encoder + DiT denoiser + Oobleck VAE.

    Customer-facing usage is a single call (tokenization + all encoders + denoise + decode handled
    internally):

        pipe = create_tt_pipeline(AceStepModelConfig.from_hf(), device)   # full text-to-music
        wav  = pipe.generate_song("upbeat synthwave, nostalgic", lyrics="neon city lights", seconds=8)
        # wav: torch [1, 2, samples] @ 48 kHz stereo

    Lower-level stages (generate / decode) remain available for advanced use.
    """

    args: AceStepModelConfig
    dit: object  # AceStepDiTModel
    vae: object  # OobleckDecoder
    vae_config: object
    mesh_device: object
    _rope: Qwen3RotaryEmbedding
    _mod: object  # reference modeling module (for create_4d_mask)
    text_encoder: object = None  # AceStepTextEncoder (prompt/lyrics -> embeddings)
    condition_encoder: object = None  # AceStepConditionEncoder (-> DiT cross-attn context)
    tokenizer: object = None  # HF Qwen3 tokenizer
    _hf_text_cfg: object = None  # text-encoder HF config (for its RoPE)

    # ---- denoise ----
    def _rope_tables(self, t_prime: int):
        pos = torch.arange(t_prime).unsqueeze(0)
        cos, sin = self._rope(torch.zeros(1, t_prime, HEAD_DIM), pos)
        cos_tt = ttnn.from_torch(
            cos.unsqueeze(1), device=self.mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        sin_tt = ttnn.from_torch(
            sin.unsqueeze(1), device=self.mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        return cos_tt, sin_tt

    def _sliding_mask(self, t_prime: int):
        if t_prime <= self.args.sliding_window:
            return None
        mk = self._mod.create_4d_mask(
            seq_len=t_prime,
            dtype=torch.float32,
            device=torch.device("cpu"),
            attention_mask=None,
            sliding_window=self.args.sliding_window,
            is_sliding_window=True,
            is_causal=False,
        )
        return ttnn.from_torch(mk, device=self.mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    def generate(self, hidden_noise, context_latents, encoder_hidden_states, *, infer_steps=30, shift=1.0):
        """Run the ODE denoise loop. Inputs are TT tensors [1,1,T,·]; returns clean latents TT.

        hidden_noise:          [1,1,T,64]  initial noise xt
        context_latents:       [1,1,T,128] (src_latents concat chunk_masks)
        encoder_hidden_states: [1,1,enc,2048]
        """
        seq_len = hidden_noise.shape[2]
        t_prime = seq_len // PATCH
        cos_tt, sin_tt = self._rope_tables(t_prime)
        sliding = self._sliding_mask(t_prime)

        solver = FlowMatchStep(self.mesh_device)
        t = _shifted_timesteps(infer_steps, shift, torch.device("cpu"))

        xt = hidden_noise
        for step_idx in range(infer_steps):
            t_curr = float(t[step_idx].item())
            t_prev = float(t[step_idx + 1].item())
            t_scalar = torch.tensor([t_curr], dtype=torch.float32)
            vt = self.dit.forward(
                xt, context_latents, t_scalar, t_scalar, cos_tt, sin_tt, encoder_hidden_states, sliding_mask=sliding
            )
            xt_new = solver.euler_step(xt, vt, t_curr - t_prev)
            if step_idx > 0:
                ttnn.deallocate(xt)
            xt = xt_new
        return xt  # [1,1,T,64] clean latents

    # ---- high-level customer API ----
    SAMPLE_RATE = 48000
    LATENT_HZ = 25  # 25 latent frames / second

    def generate_song(
        self,
        prompt: str,
        *,
        lyrics: str = "",
        seconds: float = 8.0,
        infer_steps: int = 30,
        shift: float = 1.0,
        seed: int = 0,
    ):
        """Text -> 48 kHz stereo song in one call. Returns torch waveform [1, 2, samples].

        prompt:  style / caption text.  lyrics: optional lyric text.  seconds: song length.
        Tokenization, text encoding, conditioning, denoise (infer_steps ODE) and VAE decode are
        all handled internally. Requires the pipeline built with text/condition encoders
        (create_tt_pipeline(..., with_encoders=True), the default).
        """
        assert (
            self.text_encoder is not None and self.condition_encoder is not None
        ), "generate_song needs the encoders; build with create_tt_pipeline(..., with_encoders=True)"
        assert self.vae is not None, "generate_song needs the VAE; build with with_vae=True"

        enc_hs = self.encode_prompt(prompt, lyrics)
        seq_len = self._latent_len(seconds)
        gen = torch.Generator().manual_seed(seed)
        noise = torch.randn(1, 1, seq_len, self.args.audio_acoustic_hidden_dim, generator=gen)
        # text2music: silence source latents + all-valid chunk mask.
        hidden_ch = self.args.audio_acoustic_hidden_dim
        context = torch.cat([torch.zeros(1, 1, seq_len, hidden_ch), torch.ones(1, 1, seq_len, hidden_ch)], dim=-1)
        noise_tt = ttnn.from_torch(noise, device=self.mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        context_tt = ttnn.from_torch(context, device=self.mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        latents = self.generate(noise_tt, context_tt, enc_hs, infer_steps=infer_steps, shift=shift)
        return self.decode(latents)

    def _latent_len(self, seconds: float) -> int:
        n = max(2, int(round(seconds * self.LATENT_HZ)))
        return n + (n % 2)  # even for patch_size=2

    def encode_prompt(self, prompt: str, lyrics: str = ""):
        """Tokenize + encode prompt/lyrics -> DiT cross-attn context TT tensor [1,1,ctx,hidden].

        text  -> tokenizer -> TT text encoder -> text_hidden_states (projected inside cond-enc)
        lyric -> tokenizer -> text_encoder.embed_tokens lookup (matches the reference)
        timbre-> silence (text2music, no reference audio)
        """
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

        text_ids = torch.tensor([self.tokenizer(prompt, truncation=True, max_length=256).input_ids])
        lyric_ids = torch.tensor([self.tokenizer(lyrics or prompt, truncation=True, max_length=2048).input_ids])
        tlen, llen = text_ids.shape[1], lyric_ids.shape[1]

        # 1. text encoder (its own RoPE: theta from the text-encoder config, head_dim 128).
        te_rope = Qwen3RotaryEmbedding(self._hf_text_cfg)
        tcos, tsin = self._rope_tt(te_rope, tlen)
        text_hs = self.text_encoder.forward(text_ids, tcos, tsin)  # [1,1,tlen,text_hidden]

        # 2. lyric embeddings = text_encoder.embed_tokens lookup only (per reference).
        lyric_hs = self.text_encoder.embed_tokens[lyric_ids.long()].reshape(
            1, 1, llen, self.text_encoder.config.hidden_size
        )
        lyric_tt = ttnn.from_torch(lyric_hs, device=self.mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        # 3. timbre = silence (text2music: no reference audio).
        timbre_len = 96
        timbre = torch.zeros(1, 1, timbre_len, self.args.audio_acoustic_hidden_dim)
        timbre_tt = ttnn.from_torch(timbre, device=self.mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        # 4. condition encoder -> packed cross-attn context.
        lcos, lsin = self._rope_tt(self._rope, llen)
        tbcos, tbsin = self._rope_tt(self._rope, timbre_len)
        return self.condition_encoder.forward(
            text_hs,
            lyric_tt,
            timbre_tt,
            lcos,
            lsin,
            tbcos,
            tbsin,
            lyric_sliding=self._enc_sliding(llen),
            timbre_sliding=self._enc_sliding(timbre_len),
        )

    def _rope_tt(self, rope, seq_len: int):
        pos = torch.arange(seq_len).unsqueeze(0)
        cos, sin = rope(torch.zeros(1, seq_len, HEAD_DIM), pos)
        cos_tt = ttnn.from_torch(
            cos.unsqueeze(1), device=self.mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        sin_tt = ttnn.from_torch(
            sin.unsqueeze(1), device=self.mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        return cos_tt, sin_tt

    def _enc_sliding(self, seq_len: int):
        mk = self._mod.create_4d_mask(
            seq_len=seq_len,
            dtype=torch.float32,
            device=torch.device("cpu"),
            attention_mask=None,
            sliding_window=self.args.sliding_window,
            is_sliding_window=True,
            is_causal=False,
        )
        return ttnn.from_torch(mk, device=self.mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    # ---- decode ----
    def decode(self, latents_tt):
        """VAE-decode clean latents [1,1,T,64] -> 48kHz stereo waveform torch [1,2,T*1920]."""
        seq_len = latents_tt.shape[2]
        lat = ttnn.to_torch(latents_tt).float().reshape(1, seq_len, self.args.audio_acoustic_hidden_dim)
        # VAE decoder wants [B,T,C] ROW_MAJOR fp32.
        lat_tt = ttnn.from_torch(lat, device=self.mesh_device, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)
        wav_tt = self.vae.forward(lat_tt)
        ac = self.vae_config.audio_channels
        wav = ttnn.to_torch(wav_tt).float()[..., :ac].reshape(1, -1, ac).transpose(1, 2)  # [1,2,samples]
        return wav


def create_tt_pipeline(
    args: AceStepModelConfig, mesh_device, *, with_vae: bool = True, with_encoders: bool = True
) -> AceStepPipeline:
    """Assemble the full TT text-to-music pipeline from the genuine checkpoint.

    Single public model-construction entry point. Builds the DiT denoiser and, by default, the
    Oobleck VAE decoder + text encoder + condition encoder + tokenizer — so `generate_song(prompt)`
    works out of the box. Weights load lazily on first forward.

    - with_vae:      build the VAE decoder (needed for audio output / generate_song / decode).
    - with_encoders: build the text + condition encoders + tokenizer (needed for generate_song /
                     encode_prompt). Set False for DiT-only latent tests.
    """
    from models.experimental.acestep.tt.model_config import build_condition_encoder, build_text_encoder

    dit = _build_dit_model(args, mesh_device)
    vae, vae_cfg = (None, None)
    if with_vae:
        vae, vae_cfg = build_vae_decoder(mesh_device)

    text_encoder = condition_encoder = tokenizer = hf_text_cfg = None
    if with_encoders:
        from transformers import AutoTokenizer
        from models.experimental.acestep.reference.weight_utils import pipeline_dir

        text_encoder, hf_text = build_text_encoder(mesh_device)
        hf_text_cfg = hf_text.config
        condition_encoder = build_condition_encoder(args, mesh_device)
        tokenizer = AutoTokenizer.from_pretrained(str(pipeline_dir() / "Qwen3-Embedding-0.6B"))

    hf = load_config()
    rope = Qwen3RotaryEmbedding(hf)
    mod = load_modeling_module()
    return AceStepPipeline(
        args=args,
        dit=dit,
        vae=vae,
        vae_config=vae_cfg,
        mesh_device=mesh_device,
        _rope=rope,
        _mod=mod,
        text_encoder=text_encoder,
        condition_encoder=condition_encoder,
        tokenizer=tokenizer,
        _hf_text_cfg=hf_text_cfg,
    )
