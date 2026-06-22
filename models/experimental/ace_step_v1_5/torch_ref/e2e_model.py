# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end ACE-Step v1.5 model (PyTorch reference): text encoder → DiT → VAE decoder.

Default denoise path uses the **official** Hugging Face ``AutoModel.generate_audio()`` sampler
(same as ACE-Step-1.5). Set ``use_dit_ref_sampler=True`` on :class:`E2EConfig` for the lightweight
:class:`~models.experimental.ace_step_v1_5.torch_ref.full_pipeline.AceStepV15TorchPipeline`
(TTNN PCC / module parity only).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch

from .full_pipeline import AceStepV15TorchPipeline
from .hf_generate import (
    build_t_schedule,
    default_guidance_scale,
    ensure_hf_modeling_ready,
    load_hf_ace_model,
    prepare_silence_and_masks,
    run_hf_generate_audio,
)


@dataclass
class E2EConfig:
    """Configuration for the end-to-end pipeline (mirrors ttnn_impl.e2e_model.E2EConfig)."""

    checkpoint_safetensors_path: str
    vae_dir: str
    text_model_dir: str
    silence_latent_path: str

    variant: str = "acestep-v15-turbo"
    ace_step_repo_root: Optional[str] = None
    use_dit_ref_sampler: bool = False

    duration_sec: float = 10.0
    infer_steps: int = 50
    shift: float = 1.0
    guidance_scale: float = 7.0
    cfg_interval_start: float = 0.0
    cfg_interval_end: float = 1.0
    use_adg: Optional[bool] = None
    seed: int = 0
    sample_rate: int = 48000
    qwen_safetensors_path: Optional[str] = None
    vae_chunk_latents: int = 32
    vae_overlap_latents: int = 4

    dcw_enabled: bool = True
    dcw_mode: str = "double"
    dcw_scaler: float = 0.05
    dcw_high_scaler: float = 0.02
    dcw_wavelet: str = "haar"
    sampler_mode: str = "euler"
    infer_method: str = "ode"


def run_torch_denoise_loop(
    pipe: AceStepV15TorchPipeline,
    t_schedule: List[float],
    frames: int,
    enc_hs: torch.Tensor,
    ctx_lat: torch.Tensor,
    null_emb: Optional[torch.Tensor],
    do_cfg: bool,
    seed: int,
    cfg_fn: Optional[Callable[[int, float, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    progress_fn: Optional[Callable[[int, int, float, float], None]] = None,
) -> torch.Tensor:
    """Run the lightweight DiT ref denoising loop (TTNN PCC only — not HF parity)."""
    num_steps = len(t_schedule)

    torch.manual_seed(seed)
    xt = torch.randn((1, frames, 64), dtype=torch.float32)
    xt = xt.to(device=enc_hs.device, dtype=enc_hs.dtype)

    if do_cfg and null_emb is None:
        null_emb = torch.zeros_like(enc_hs)

    if cfg_fn is None:

        def cfg_fn(
            _si: int, _t: float, _xt: torch.Tensor, vt_cond: torch.Tensor, _vt_uncond: torch.Tensor
        ) -> torch.Tensor:
            return vt_cond

    with torch.inference_mode():
        for step_idx in range(num_steps):
            t_curr_f = float(t_schedule[step_idx])
            t_next_f = float(t_schedule[step_idx + 1]) if step_idx < num_steps - 1 else 0.0
            dt = t_curr_f - t_next_f

            if do_cfg:
                enc2 = torch.cat([enc_hs, null_emb.expand_as(enc_hs)], dim=0)
                ctx2 = torch.cat([ctx_lat, ctx_lat], dim=0)
                xt2 = torch.cat([xt, xt], dim=0)

                acoustic = pipe.forward(
                    xt_bt64=xt2,
                    context_latents_bt128=ctx2,
                    timestep_index=step_idx,
                    encoder_hidden_states_btd=enc2,
                )
            else:
                acoustic = pipe.forward(
                    xt_bt64=xt,
                    context_latents_bt128=ctx_lat,
                    timestep_index=step_idx,
                    encoder_hidden_states_btd=enc_hs,
                )

            vt2 = acoustic.to(torch.float32)
            if do_cfg:
                vt = cfg_fn(step_idx, t_curr_f, xt, vt2[0:1], vt2[1:2])
            else:
                vt = vt2
            xt = xt - vt * dt

            if progress_fn is not None:
                progress_fn(step_idx, num_steps, t_curr_f, dt)

    return xt.float().cpu()


def decode_with_vae(
    vae: torch.nn.Module,
    pred_latents: torch.Tensor,
    torch_dev: torch.device,
) -> torch.Tensor:
    """Decode DiT latents to a waveform via the Oobleck VAE."""
    with torch.inference_mode():
        lat = pred_latents.transpose(1, 2).contiguous().to(device=torch_dev, dtype=next(vae.parameters()).dtype)
        wav = vae.decode(lat).sample.float().cpu()
        peak = wav.abs().amax(dim=[1, 2], keepdim=True).clamp(min=1e-8)
        wav = (wav / peak).clamp(-1.0, 1.0)
    return wav


class AceStepE2EModelTorch:
    """End-to-end ACE-Step v1.5 in PyTorch: text → DiT → VAE → waveform."""

    def __init__(
        self,
        config: E2EConfig,
        torch_device: torch.device | str | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.config = config
        self.torch_dev = torch.device(torch_device or "cpu")
        self.dtype = dtype
        self.model_dir = Path(config.checkpoint_safetensors_path).parent

        ckpt_dir = self.model_dir.parent
        self._ref_root = ensure_hf_modeling_ready(
            ckpt_dir=str(ckpt_dir),
            ace_step_repo_root=config.ace_step_repo_root,
        )

        self._load_text_encoder()
        self.frames = int(round(config.duration_sec * 25.0))
        self.silence_latent, self.src_latents, self.chunk_masks = prepare_silence_and_masks(
            config.silence_latent_path,
            frames=self.frames,
        )

        self.gs = default_guidance_scale(variant=config.variant, guidance_scale=config.guidance_scale)
        self.t_schedule = build_t_schedule(
            shift=config.shift,
            infer_steps=config.infer_steps,
            variant=config.variant,
        )
        self.timesteps_host = np.asarray(self.t_schedule + [0.0], dtype=np.float32)

        self.ace_model = None
        self.pipe = None
        self.null_emb = None

        if config.use_dit_ref_sampler:
            self.pipe = AceStepV15TorchPipeline(
                checkpoint_safetensors_path=config.checkpoint_safetensors_path,
                timesteps_host=self.timesteps_host,
                device=self.torch_dev,
                dtype=dtype,
            )
        else:
            self.ace_model = load_hf_ace_model(self.model_dir, device=self.torch_dev, dtype=dtype)
            nc = getattr(self.ace_model, "null_condition_emb", None)
            if nc is None:
                inner = getattr(self.ace_model, "model", None)
                if inner is not None:
                    nc = getattr(inner, "null_condition_emb", None)
            if nc is not None:
                self.null_emb = nc.float().cpu()

        self._load_vae()

    def _load_text_encoder(self) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.text_model_dir)
        self.text_model = AutoModel.from_pretrained(self.config.text_model_dir).eval().to(self.torch_dev)

    def _load_vae(self) -> None:
        from diffusers.models import AutoencoderOobleck

        self.vae = AutoencoderOobleck.from_pretrained(self.config.vae_dir).eval().to(self.torch_dev)

    def encode_text(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a text prompt into hidden states and attention mask."""
        dit_instruction = "Fill the audio semantic mask based on the given conditions:"
        metas = {"caption": prompt, "duration": self.config.duration_sec, "language": "en"}
        text_prompt = f"""# Instruction
{dit_instruction}

# Caption
{prompt}

# Metas
{metas}<|endoftext|>
"""
        tokens = self.tokenizer(text_prompt, padding="max_length", truncation=True, max_length=256, return_tensors="pt")
        input_ids = tokens["input_ids"].to(self.torch_dev)
        attn_mask = tokens["attention_mask"].to(self.torch_dev).to(torch.bool)
        with torch.inference_mode():
            text_out = self.text_model(input_ids=input_ids, attention_mask=attn_mask)
            text_hidden_states = text_out.last_hidden_state
        return text_hidden_states, attn_mask

    def denoise_hf(
        self,
        text_hidden_states: torch.Tensor,
        attn_mask: torch.Tensor,
        *,
        precomputed_lm_hints_25Hz: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Official HF ``generate_audio()`` denoise (matches ACE-Step-1.5)."""
        if self.ace_model is None:
            raise RuntimeError("HF model not loaded (use_dit_ref_sampler=True?)")
        return run_hf_generate_audio(
            self.ace_model,
            text_hidden_states=text_hidden_states,
            text_attention_mask=attn_mask,
            src_latents=self.src_latents,
            silence_latent=self.silence_latent,
            chunk_masks=self.chunk_masks,
            device=self.torch_dev,
            seed=self.config.seed,
            infer_steps=self.config.infer_steps,
            guidance_scale=self.gs,
            shift=self.config.shift,
            variant=self.config.variant,
            cfg_interval_start=self.config.cfg_interval_start,
            cfg_interval_end=self.config.cfg_interval_end,
            use_adg=self.config.use_adg,
            precomputed_lm_hints_25Hz=precomputed_lm_hints_25Hz,
            dcw_enabled=self.config.dcw_enabled,
            dcw_mode=self.config.dcw_mode,
            dcw_scaler=self.config.dcw_scaler,
            dcw_high_scaler=self.config.dcw_high_scaler,
            dcw_wavelet=self.config.dcw_wavelet,
            sampler_mode=self.config.sampler_mode,
            infer_method=self.config.infer_method,
        )

    def denoise_ref(
        self,
        enc_hs: torch.Tensor,
        ctx_lat: torch.Tensor,
    ) -> torch.Tensor:
        """Lightweight ref DiT denoise (TTNN PCC only)."""
        if self.pipe is None or self.null_emb is None:
            raise RuntimeError("Ref pipeline not loaded (use_dit_ref_sampler=False?)")
        return run_torch_denoise_loop(
            pipe=self.pipe,
            t_schedule=self.t_schedule,
            frames=self.frames,
            enc_hs=enc_hs,
            ctx_lat=ctx_lat,
            null_emb=self.null_emb,
            do_cfg=self.gs > 1.0 + 1e-6,
            seed=self.config.seed,
        )

    def prepare_condition_ref(
        self,
        text_hidden_states: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """HF ``prepare_condition`` for ref-sampler path only."""
        if self.ace_model is None:
            self.ace_model = load_hf_ace_model(self.model_dir, device=self.torch_dev, dtype=self.dtype)

        B = 1
        frames = self.frames
        lyric_dim = int(text_hidden_states.shape[-1])
        lyric_hidden_states = torch.zeros((B, 1, lyric_dim), dtype=torch.float32, device=self.torch_dev)
        lyric_attention_mask = torch.ones((B, 1), dtype=torch.bool, device=self.torch_dev)
        refer_audio = torch.zeros((B, 1, 64), dtype=torch.float32, device=self.torch_dev)
        refer_order = torch.zeros((B,), dtype=torch.long, device=self.torch_dev)
        latent_attention_mask = torch.ones((B, frames), dtype=torch.float32, device=self.torch_dev)

        with torch.inference_mode():
            enc_hs, enc_mask, ctx_lat = self.ace_model.prepare_condition(
                text_hidden_states=text_hidden_states.to(dtype=torch.float32),
                text_attention_mask=attn_mask,
                lyric_hidden_states=lyric_hidden_states,
                lyric_attention_mask=lyric_attention_mask,
                refer_audio_acoustic_hidden_states_packed=refer_audio,
                refer_audio_order_mask=refer_order,
                hidden_states=self.src_latents.to(device=self.torch_dev, dtype=torch.float32),
                attention_mask=latent_attention_mask,
                silence_latent=self.silence_latent.to(device=self.torch_dev, dtype=torch.float32),
                src_latents=self.src_latents.to(device=self.torch_dev, dtype=torch.float32),
                chunk_masks=self.chunk_masks.to(device=self.torch_dev, dtype=torch.float32),
                is_covers=torch.zeros((B,), dtype=torch.bool, device=self.torch_dev),
                precomputed_lm_hints_25Hz=None,
            )

        nc = getattr(self.ace_model, "null_condition_emb", None)
        if nc is not None:
            self.null_emb = nc.float().cpu()

        return enc_hs.float().cpu(), enc_mask.float().cpu(), ctx_lat.float().cpu()

    def denoise(
        self,
        text_hidden_states: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        enc_hs: torch.Tensor | None = None,
        enc_mask: torch.Tensor | None = None,
        ctx_lat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run denoise: HF ``generate_audio`` by default, ref sampler if configured."""
        if self.config.use_dit_ref_sampler:
            if enc_hs is None or ctx_lat is None:
                if text_hidden_states is None or attn_mask is None:
                    raise ValueError("ref sampler requires text_hidden_states+attn_mask or enc_hs+ctx_lat")
                enc_hs, enc_mask, ctx_lat = self.prepare_condition_ref(text_hidden_states, attn_mask)
            return self.denoise_ref(enc_hs, ctx_lat)

        if text_hidden_states is None or attn_mask is None:
            raise ValueError("HF denoise requires text_hidden_states and attn_mask")
        return self.denoise_hf(text_hidden_states, attn_mask)

    def decode_vae(self, pred_latents: torch.Tensor) -> torch.Tensor:
        return decode_with_vae(self.vae, pred_latents, self.torch_dev)

    def generate(self, prompt: str) -> torch.Tensor:
        """Full end-to-end: text prompt → waveform tensor."""
        text_hs, attn_mask = self.encode_text(prompt)
        pred_latents = self.denoise(text_hidden_states=text_hs, attn_mask=attn_mask)
        return self.decode_vae(pred_latents)
