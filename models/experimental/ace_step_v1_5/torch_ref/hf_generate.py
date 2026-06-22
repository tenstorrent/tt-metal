# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Official Hugging Face ACE-Step ``generate_audio()`` helpers for torch_ref.

Default torch_ref path: load ``AutoModel`` with ``trust_remote_code`` and run the same
``generate_audio()`` sampler as ACE-Step-1.5 (CFG/ADG, DCW, Euler/Heun, turbo timesteps).

The lightweight :class:`~models.experimental.ace_step_v1_5.torch_ref.full_pipeline.AceStepV15TorchPipeline`
remains for TTNN PCC only (``use_dit_ref_sampler=True``).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List

import torch

# Turbo discrete timesteps (aligned with acestep turbo modeling).
_VALID_TIMESTEPS = [
    1.0,
    0.9545454545454546,
    0.9333333333333333,
    0.9,
    0.875,
    0.8571428571428571,
    0.8333333333333334,
    0.7692307692307693,
    0.75,
    0.6666666666666666,
    0.6428571428571429,
    0.625,
    0.5454545454545454,
    0.5,
    0.4,
    0.375,
    0.3,
    0.25,
    0.2222222222222222,
    0.125,
]

_SHIFT_TIMESTEPS: dict[float, list[float]] = {
    1.0: [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125],
    2.0: [
        1.0,
        0.9333333333333333,
        0.8571428571428571,
        0.7692307692307693,
        0.6666666666666666,
        0.5454545454545454,
        0.4,
        0.2222222222222222,
    ],
    3.0: [1.0, 0.9545454545454546, 0.9, 0.8333333333333334, 0.75, 0.6428571428571429, 0.5, 0.3],
}

_VENDORED_ACESTEP_ROOT = Path(__file__).resolve().parent / "_vendored_acestep"

_WELL_KNOWN_REPO_ROOTS = (
    Path("/home/iguser/tt-ign/ACE-Step-1.5"),
    Path.home() / "ACE-Step-1.5",
    Path.home() / "proj_sdk" / "ACE-Step-1.5",
)


def resolve_ace_step_repo_root(
    *,
    ckpt_dir: str | Path | None = None,
    ace_step_repo_root: str | Path | None = None,
) -> Path | None:
    """Directory containing ``acestep/`` for ``trust_remote_code`` modeling imports."""
    candidates: list[Path] = []
    if ace_step_repo_root:
        candidates.append(Path(ace_step_repo_root).expanduser().resolve())
    env = os.environ.get("ACE_STEP_REPO_ROOT")
    if env:
        candidates.append(Path(env).expanduser().resolve())
    candidates.append(_VENDORED_ACESTEP_ROOT)
    candidates.extend(_WELL_KNOWN_REPO_ROOTS)

    if ckpt_dir:
        cur = Path(ckpt_dir).expanduser().resolve()
        for _ in range(8):
            candidates.append(cur)
            if cur.parent == cur:
                break
            cur = cur.parent

    seen: set[str] = set()
    for c in candidates:
        key = str(c)
        if key in seen:
            continue
        seen.add(key)
        if (c / "acestep" / "__init__.py").is_file():
            return c
    return None


def ensure_hf_modeling_ready(
    *,
    ckpt_dir: str | Path | None = None,
    ace_step_repo_root: str | Path | None = None,
) -> Path | None:
    """Apply transformers compat and add ACE-Step repo to ``sys.path`` when found."""
    from models.experimental.ace_step_v1_5.torch_ref.transformers_cache_compat import apply_transformers_cache_compat

    apply_transformers_cache_compat()
    ref_root = resolve_ace_step_repo_root(ckpt_dir=ckpt_dir, ace_step_repo_root=ace_step_repo_root)
    if ref_root is not None:
        from models.experimental.ace_step_v1_5.demo.ref_decoder_compare import ensure_acestep_repo_on_path

        ensure_acestep_repo_on_path(ref_root)
    return ref_root


def build_t_schedule(
    *,
    shift: float,
    infer_steps: int,
    timesteps: str | None = None,
    variant: str,
) -> List[float]:
    """Inference timestep schedule (t_curr per step; terminal 0 appended separately for ref pipeline)."""
    variant_l = (variant or "").lower()
    is_turbo = "turbo" in variant_l

    if timesteps:
        raw = [float(x.strip()) for x in timesteps.split(",") if x.strip()]
        while raw and raw[-1] == 0.0:
            raw.pop()
        if not raw:
            raise ValueError("timesteps provided but empty after removing zeros")
        if is_turbo:
            mapped = [min(_VALID_TIMESTEPS, key=lambda v, t=t: abs(v - t)) for t in raw]
            out: list[float] = []
            for t in mapped:
                if not out or out[-1] != t:
                    out.append(t)
            return out
        return raw

    infer_steps = int(infer_steps)
    if infer_steps <= 1:
        raise ValueError("infer_steps must be >= 2")

    if is_turbo:
        s = min(_SHIFT_TIMESTEPS.keys(), key=lambda v: abs(v - float(shift)))
        if infer_steps == 8:
            return list(_SHIFT_TIMESTEPS[float(s)])
        lin = [1.0 - (i / float(infer_steps - 1)) for i in range(infer_steps)]
        mapped = [min(_VALID_TIMESTEPS, key=lambda v, t=t: abs(v - t)) for t in lin]
        out = []
        for t in mapped:
            if not out or out[-1] != t:
                out.append(t)
        return sorted(out, reverse=True)

    t = [1.0 - (i / float(infer_steps)) for i in range(infer_steps)]
    if float(shift) != 1.0:
        s = float(shift)
        t = [s * x / (1.0 + (s - 1.0) * x) for x in t]
    return t


def default_guidance_scale(*, variant: str, guidance_scale: float | None) -> float:
    if guidance_scale is not None:
        return float(guidance_scale)
    return 1.0 if "turbo" in str(variant).lower() else 7.0


def default_use_adg(*, variant: str) -> bool:
    variant_l = str(variant).lower()
    return "base" in variant_l and "turbo" not in variant_l


def load_hf_ace_model(model_dir: str | Path, *, device: torch.device, dtype: torch.dtype | None = None):
    """Load full HF ACE-Step ``AutoModel`` from a checkpoint directory."""
    from transformers import AutoModel

    if dtype is None:
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    return AutoModel.from_pretrained(str(model_dir), trust_remote_code=True, torch_dtype=dtype).eval().to(device)


def run_hf_generate_audio(
    ace_model: Any,
    *,
    text_hidden_states: torch.Tensor,
    text_attention_mask: torch.Tensor,
    src_latents: torch.Tensor,
    silence_latent: torch.Tensor,
    chunk_masks: torch.Tensor,
    device: torch.device,
    seed: int,
    infer_steps: int,
    guidance_scale: float,
    shift: float = 1.0,
    variant: str = "acestep-v15-turbo",
    timesteps: str | None = None,
    cfg_interval_start: float = 0.0,
    cfg_interval_end: float = 1.0,
    use_adg: bool | None = None,
    precomputed_lm_hints_25Hz: torch.Tensor | None = None,
    dcw_enabled: bool = True,
    dcw_mode: str = "double",
    dcw_scaler: float = 0.05,
    dcw_high_scaler: float = 0.02,
    dcw_wavelet: str = "haar",
    sampler_mode: str = "euler",
    infer_method: str = "ode",
    use_progress_bar: bool = True,
) -> torch.Tensor:
    """Run official ``ace_model.generate_audio()``; return ``target_latents`` CPU float32."""
    B = int(src_latents.shape[0])
    lyric_dim = int(text_hidden_states.shape[-1])
    lyric_hidden_states = torch.zeros((B, 1, lyric_dim), dtype=torch.float32, device=device)
    lyric_attention_mask = torch.ones((B, 1), dtype=torch.bool, device=device)
    refer_audio = torch.zeros((B, 1, 64), dtype=torch.float32, device=device)
    refer_order = torch.zeros((B,), dtype=torch.long, device=device)
    latent_attention_mask = torch.ones((B, int(src_latents.shape[1])), dtype=torch.float32, device=device)

    timesteps_tensor = None
    if timesteps:
        t_sched = build_t_schedule(
            shift=float(shift),
            infer_steps=int(infer_steps),
            timesteps=timesteps,
            variant=str(variant),
        )
        timesteps_tensor = torch.tensor(t_sched + [0.0], device=device, dtype=torch.float32)

    if use_adg is None:
        use_adg = default_use_adg(variant=variant)

    with torch.inference_mode():
        gen_out = ace_model.generate_audio(
            text_hidden_states=text_hidden_states.to(device=device, dtype=torch.float32),
            text_attention_mask=text_attention_mask,
            lyric_hidden_states=lyric_hidden_states,
            lyric_attention_mask=lyric_attention_mask,
            refer_audio_acoustic_hidden_states_packed=refer_audio,
            refer_audio_order_mask=refer_order,
            src_latents=src_latents.to(device=device, dtype=torch.float32),
            chunk_masks=chunk_masks.to(device=device, dtype=torch.float32),
            is_covers=torch.zeros((B,), dtype=torch.bool, device=device),
            silence_latent=silence_latent.to(device=device, dtype=torch.float32),
            attention_mask=latent_attention_mask,
            seed=int(seed),
            infer_steps=int(infer_steps),
            diffusion_guidance_scale=float(guidance_scale),
            use_adg=bool(use_adg),
            shift=float(shift),
            timesteps=timesteps_tensor,
            use_progress_bar=bool(use_progress_bar),
            infer_method=str(infer_method),
            sampler_mode=str(sampler_mode),
            cfg_interval_start=float(cfg_interval_start),
            cfg_interval_end=float(cfg_interval_end),
            precomputed_lm_hints_25Hz=precomputed_lm_hints_25Hz,
            dcw_enabled=bool(dcw_enabled),
            dcw_mode=str(dcw_mode),
            dcw_scaler=float(dcw_scaler),
            dcw_high_scaler=float(dcw_high_scaler),
            dcw_wavelet=str(dcw_wavelet),
        )
    return gen_out["target_latents"].float().cpu()


def prepare_silence_and_masks(
    silence_latent_path: str | Path,
    *,
    frames: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load silence latent and build ``src_latents`` + ``chunk_masks`` like HF demos."""
    silence = torch.load(str(silence_latent_path), map_location="cpu").to(torch.float32)
    if silence.ndim != 3:
        raise RuntimeError(f"Unexpected silence_latent rank: {tuple(silence.shape)}")
    if int(silence.shape[-1]) == 64:
        pass
    elif int(silence.shape[1]) == 64:
        silence = silence.transpose(1, 2).contiguous()
    else:
        raise RuntimeError(f"Unexpected silence_latent shape: {tuple(silence.shape)}")

    src_latents = silence[:, :frames, :].contiguous()
    if src_latents.shape[1] < frames:
        rep = (frames + src_latents.shape[1] - 1) // src_latents.shape[1]
        src_latents = src_latents.repeat(1, rep, 1)[:, :frames, :].contiguous()
    chunk_masks = torch.ones((1, frames, 64), dtype=torch.float32)
    return silence, src_latents, chunk_masks
