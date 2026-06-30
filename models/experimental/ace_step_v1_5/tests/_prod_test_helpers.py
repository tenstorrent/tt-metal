# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared checkpoint + handler helpers for production-dimension PCC tests."""

from __future__ import annotations

import os
from pathlib import Path


def ckpt_root() -> Path:
    return Path(
        os.environ.get("ACE_STEP_CHECKPOINT_DIR", "~/.cache/huggingface/hub/ACE-Step-1.5-checkpoints")
    ).expanduser()


def base_model_safetensors() -> Path | None:
    p = ckpt_root() / "acestep-v15-base" / "model.safetensors"
    return p if p.is_file() else None


def vae_hf_dir() -> Path | None:
    p = ckpt_root() / "vae"
    return p if (p / "config.json").is_file() else None


def lm_dir(name: str) -> Path | None:
    p = ckpt_root() / name
    return p if (p / "config.json").is_file() else None


def ensure_vendored_acestep_on_path() -> None:
    from models.experimental.ace_step_v1_5.utils.acestep_paths import ensure_host_preprocess_on_path

    ensure_host_preprocess_on_path()


def init_dit_and_lm_handlers(*, lm_variant: str = "acestep-5Hz-lm-1.7B"):
    """CPU handlers with HF checkpoints (matches demo preprocess path)."""
    ensure_vendored_acestep_on_path()
    from acestep.handler import AceStepHandler

    from models.experimental.ace_step_v1_5.torch_ref.five_hz_lm import LocalFiveHzLMHandler

    root = ckpt_root()
    dit = AceStepHandler()
    status, ok = dit.initialize_service(
        project_root=str(root),
        config_path="acestep-v15-base",
        device="cpu",
        use_flash_attention=False,
        preprocess_only=True,
        use_mlx_dit=False,
    )
    if not ok:
        raise RuntimeError(f"AceStepHandler.initialize_service failed: {status}")

    llm = LocalFiveHzLMHandler()
    status, ok = llm.initialize(
        checkpoint_dir=str(root),
        lm_model_path=lm_variant,
        backend="pt",
        device="cpu",
    )
    if not ok:
        raise RuntimeError(f"LocalFiveHzLMHandler.initialize failed: {status}")
    return dit, llm


def build_instrumental_filtered(
    dit_handler,
    llm_handler,
    *,
    duration_sec: float = 15.0,
    seed: int = 42,
    caption: str = "Guitar",
    infer_steps: int = 8,
    guidance_scale: float = 7.0,
):
    from models.experimental.ace_step_v1_5.utils.acestep_preprocess_shim import GenerationConfig, GenerationParams
    from models.experimental.ace_step_v1_5.utils.official_lm_preprocess import build_filtered_dit_kwargs_for_handler

    frames = max(1, int(round(float(duration_sec) * 25.0)))
    params = GenerationParams(
        task_type="text2music",
        caption=caption,
        lyrics="[Instrumental]",
        instrumental=True,
        duration=float(duration_sec),
        inference_steps=int(infer_steps),
        guidance_scale=float(guidance_scale),
        lm_cfg_scale=2.0,
        use_adg=False,
        cfg_interval_start=0.0,
        cfg_interval_end=1.0,
        thinking=True,
        use_constrained_decoding=True,
        seed=int(seed),
    )
    config = GenerationConfig(batch_size=1, use_random_seed=False, seeds=[seed], audio_format="wav")
    return build_filtered_dit_kwargs_for_handler(dit_handler, llm_handler, params, config, progress=None), frames
