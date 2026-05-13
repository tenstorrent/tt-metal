# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end ACE-Step v1.5 model (PyTorch reference): text encoder → DiT → VAE decoder.

Pure-PyTorch counterpart of ``ttnn_impl/e2e_model.py``.  Every stage runs on
the host via PyTorch so the output can be compared against the TTNN
implementation for PCC validation.

Standalone functions :func:`run_torch_denoise_loop` and :func:`decode_with_vae`
are exported for reuse by other scripts (e.g. the prompt-to-wav demo).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch

from .full_pipeline import AceStepV15TorchPipeline


@dataclass
class E2EConfig:
    """Configuration for the end-to-end pipeline (mirrors ttnn_impl.e2e_model.E2EConfig)."""

    checkpoint_safetensors_path: str
    vae_dir: str
    text_model_dir: str
    silence_latent_path: str

    duration_sec: float = 10.0
    infer_steps: int = 50
    shift: float = 1.0
    guidance_scale: float = 7.0
    cfg_interval_start: float = 0.0
    cfg_interval_end: float = 1.0
    use_adg: bool = True
    seed: int = 0
    sample_rate: int = 48000
    qwen_safetensors_path: Optional[str] = None
    vae_chunk_latents: int = 32
    vae_overlap_latents: int = 4


def _build_t_schedule(
    *,
    shift: float,
    infer_steps: int,
) -> List[float]:
    """Build the diffusion timestep schedule."""
    t = [float(i) / float(infer_steps) for i in range(infer_steps, -1, -1)]
    if shift != 1.0:
        s = float(shift)
        t = [s * x / (1.0 + (s - 1.0) * x) for x in t]
    return t


# ---------------------------------------------------------------------------
# Standalone helpers – importable by demo scripts to avoid code duplication.
# ---------------------------------------------------------------------------


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
    """Run the PyTorch DiT denoising loop.

    Args:
        pipe: PyTorch pipeline instance.
        t_schedule: Descending timestep floats.  The function Euler-steps
            toward ``t = 0`` after the last entry.
        frames: Temporal frame count.
        enc_hs: Encoder hidden states ``[B, S, D]``.
        ctx_lat: Context latents ``[B, T, 128]``.
        null_emb: Null-condition embedding for CFG (broadcastable to *enc_hs*).
            Falls back to zeros when ``None`` and *do_cfg* is ``True``.
        do_cfg: Whether classifier-free guidance is active (batch doubles).
        seed: RNG seed for initial noise.
        cfg_fn: ``(step_idx, t_curr, xt, vt_cond, vt_uncond) -> vt`` guidance
            combiner.  Defaults to returning *vt_cond* unchanged.
        progress_fn: ``(step_idx, num_steps, t_curr, dt)`` called after each step.

    Returns:
        Denoised latents ``[B, frames, 64]`` on CPU float32.
    """
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
    """Decode DiT latents to a waveform via the Oobleck VAE.

    Args:
        vae: Loaded ``AutoencoderOobleck`` instance.
        pred_latents: ``[B, frames, 64]`` latents from the denoising loop.
        torch_dev: Device for the VAE forward pass.

    Returns:
        Waveform ``[B, channels, samples]`` normalized to ``[-1, 1]``.
    """
    with torch.inference_mode():
        lat = pred_latents.transpose(1, 2).contiguous().to(device=torch_dev, dtype=next(vae.parameters()).dtype)
        wav = vae.decode(lat).sample.float().cpu()
        peak = wav.abs().amax(dim=[1, 2], keepdim=True).clamp(min=1e-8)
        wav = (wav / peak).clamp(-1.0, 1.0)
    return wav


class AceStepE2EModelTorch:
    """End-to-end ACE-Step v1.5 in pure PyTorch: text → DiT → VAE → waveform.

    Stages:
        1. Text encoding (Qwen3-Embedding on host)
        2. Condition preparation (silence latent, masking)
        3. PyTorch DiT denoising loop
        4. VAE decode (AutoencoderOobleck on host)
    """

    def __init__(
        self,
        config: E2EConfig,
        torch_device: torch.device | str | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.config = config
        self.torch_dev = torch.device(torch_device or "cpu")
        self.dtype = dtype

        self._load_text_encoder()
        self._load_silence_latent()

        self.t_schedule = _build_t_schedule(
            shift=config.shift,
            infer_steps=config.infer_steps,
        )
        self.timesteps_host = np.asarray(self.t_schedule + [0.0], dtype=np.float32)
        self.frames = int(round(config.duration_sec * 25.0))

        self.pipe = AceStepV15TorchPipeline(
            checkpoint_safetensors_path=config.checkpoint_safetensors_path,
            timesteps_host=self.timesteps_host,
            device=self.torch_dev,
            dtype=dtype,
        )

        self._load_vae()

    def _load_text_encoder(self) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.text_model_dir)
        self.text_model = AutoModel.from_pretrained(self.config.text_model_dir).eval().to(self.torch_dev)

    def _load_silence_latent(self) -> None:
        silence = torch.load(self.config.silence_latent_path, map_location="cpu").to(torch.float32)
        if silence.ndim != 3:
            raise RuntimeError(f"Unexpected silence_latent rank: {tuple(silence.shape)}")
        if int(silence.shape[-1]) == 64:
            pass
        elif int(silence.shape[1]) == 64:
            silence = silence.transpose(1, 2).contiguous()
        else:
            raise RuntimeError(f"Unexpected silence_latent shape: {tuple(silence.shape)}")
        self.silence_latent = silence

    def _load_vae(self) -> None:
        from diffusers.models import AutoencoderOobleck

        self.vae = AutoencoderOobleck.from_pretrained(self.config.vae_dir).eval().to(self.torch_dev)

    def encode_text(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a text prompt into hidden states and attention mask.

        Returns:
            (text_hidden_states [1, S, D], attention_mask [1, S] bool)
        """
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

    def prepare_condition(
        self,
        text_hidden_states: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build conditioning tensors for the DiT.

        Returns:
            (enc_hs [1, S, D], enc_mask [1, S], ctx_lat [1, T, 128])
        """
        from transformers import AutoModel

        frames = self.frames
        silence = self.silence_latent
        src_latents = silence[:, :frames, :].contiguous()
        if src_latents.shape[1] < frames:
            rep = (frames + src_latents.shape[1] - 1) // src_latents.shape[1]
            src_latents = src_latents.repeat(1, rep, 1)[:, :frames, :].contiguous()

        chunk_masks = torch.ones((1, frames, 64), dtype=torch.float32)

        ace = (
            AutoModel.from_pretrained(
                str(Path(self.config.checkpoint_safetensors_path).parent),
                trust_remote_code=True,
            )
            .eval()
            .to(self.torch_dev)
        )

        B = 1
        lyric_dim = int(text_hidden_states.shape[-1])
        lyric_hidden_states = torch.zeros((B, 1, lyric_dim), dtype=torch.float32, device=self.torch_dev)
        lyric_attention_mask = torch.ones((B, 1), dtype=torch.bool, device=self.torch_dev)
        refer_audio = torch.zeros((B, 1, 64), dtype=torch.float32, device=self.torch_dev)
        refer_order = torch.zeros((B,), dtype=torch.long, device=self.torch_dev)
        latent_attention_mask = torch.ones((B, frames), dtype=torch.float32, device=self.torch_dev)

        with torch.inference_mode():
            enc_hs, enc_mask, ctx_lat = ace.prepare_condition(
                text_hidden_states=text_hidden_states.to(dtype=torch.float32),
                text_attention_mask=attn_mask,
                lyric_hidden_states=lyric_hidden_states,
                lyric_attention_mask=lyric_attention_mask,
                refer_audio_acoustic_hidden_states_packed=refer_audio,
                refer_audio_order_mask=refer_order,
                hidden_states=src_latents.to(device=self.torch_dev, dtype=torch.float32),
                attention_mask=latent_attention_mask,
                silence_latent=silence.to(device=self.torch_dev, dtype=torch.float32),
                src_latents=src_latents.to(device=self.torch_dev, dtype=torch.float32),
                chunk_masks=chunk_masks.to(device=self.torch_dev, dtype=torch.float32),
                is_covers=torch.zeros((B,), dtype=torch.bool, device=self.torch_dev),
                precomputed_lm_hints_25Hz=None,
            )

        enc_hs = enc_hs.float().cpu()
        enc_mask = enc_mask.float().cpu()
        ctx_lat = ctx_lat.float().cpu()

        nc = getattr(ace, "null_condition_emb", None)
        if nc is None:
            inner = getattr(ace, "model", None)
            if inner is not None:
                nc = getattr(inner, "null_condition_emb", None)
        if nc is None:
            raise RuntimeError("Could not find null_condition_emb on ACE-Step model.")
        self.null_emb = nc.float().cpu()

        del ace
        return enc_hs, enc_mask, ctx_lat

    def denoise(
        self,
        enc_hs: torch.Tensor,
        enc_mask: torch.Tensor,
        ctx_lat: torch.Tensor,
    ) -> torch.Tensor:
        """Run the PyTorch DiT denoising loop.

        Returns:
            pred_latents [1, frames, 64] on CPU.
        """
        return run_torch_denoise_loop(
            pipe=self.pipe,
            t_schedule=self.t_schedule,
            frames=self.frames,
            enc_hs=enc_hs,
            ctx_lat=ctx_lat,
            null_emb=self.null_emb,
            do_cfg=self.config.guidance_scale > 1.0 + 1e-6,
            seed=self.config.seed,
        )

    def decode_vae(self, pred_latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to waveform via the Oobleck VAE.

        Args:
            pred_latents: [1, frames, 64]

        Returns:
            waveform [1, channels, samples] normalized to [-1, 1].
        """
        return decode_with_vae(self.vae, pred_latents, self.torch_dev)

    def generate(self, prompt: str) -> torch.Tensor:
        """Full end-to-end: text prompt → waveform tensor.

        Args:
            prompt: text description of the music.

        Returns:
            waveform [1, channels, samples] normalized to [-1, 1].
        """
        text_hs, attn_mask = self.encode_text(prompt)
        enc_hs, enc_mask, ctx_lat = self.prepare_condition(text_hs, attn_mask)
        pred_latents = self.denoise(enc_hs, enc_mask, ctx_lat)
        wav = self.decode_vae(pred_latents)
        return wav
