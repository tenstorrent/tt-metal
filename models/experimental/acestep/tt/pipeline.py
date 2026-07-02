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
    """Assembled TT pipeline: DiT denoiser + flow-match solver + Oobleck VAE decoder."""

    args: AceStepModelConfig
    dit: object  # AceStepDiTModel
    vae: object  # OobleckDecoder
    vae_config: object
    mesh_device: object
    _rope: Qwen3RotaryEmbedding
    _mod: object  # reference modeling module (for create_4d_mask)

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


def create_tt_pipeline(args: AceStepModelConfig, mesh_device, *, with_vae: bool = True) -> AceStepPipeline:
    """Assemble the full TT generation pipeline from the genuine checkpoint.

    Builds the DiT denoiser and, if requested, the Oobleck VAE decoder (build_vae_decoder).
    Weights load lazily on first forward. This is the single public model-construction entry point.
    """
    dit = _build_dit_model(args, mesh_device)
    vae, vae_cfg = (None, None)
    if with_vae:
        vae, vae_cfg = build_vae_decoder(mesh_device)

    hf = load_config()
    rope = Qwen3RotaryEmbedding(hf)
    mod = load_modeling_module()
    return AceStepPipeline(
        args=args, dit=dit, vae=vae, vae_config=vae_cfg, mesh_device=mesh_device, _rope=rope, _mod=mod
    )
