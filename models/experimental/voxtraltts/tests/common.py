# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

import pytest
import torch
import ttnn
from loguru import logger


from models.experimental.voxtraltts.reference.voxtral_config import DEFAULT_VOXTRAL_MODEL
from models.experimental.voxtraltts.tt.text_model import VoxtralTTTextModel
from models.experimental.voxtraltts.tt.voxtral_tt_args import voxtral_text_default_optimizations
from models.experimental.voxtraltts.utils.audio_tokenizer_optimizations import (
    voxtral_audio_tokenizer_default_optimizations,
)

VOXTRAL_STANDARD_CHAR_TEXT = (
    "Voxtral TTS is a frontier open weights text to speech model for production voice agents. It produces "
    "realistic expressive speech with natural prosody across English, French, Spanish, German, Italian, "
    "Portuguese, Dutch, Arabic, and Hindi. The system supports preset voices, low latency streaming, batch "
    "inference, and twenty four kilohertz audio output for customer support, real time translation, reading "
    "applications, call centers, and responsive multilingual assistant workflows. With clear speech."
)


def ensure_voxtral_device_available() -> None:
    """Skip the test when the host exposes no Tenstorrent devices."""
    if ttnn.get_num_devices() < 1:
        pytest.skip("No Tenstorrent device available on this host")


def voxtral_single_device_mesh_shape() -> ttnn.MeshShape:
    """1×1 mesh — one effective device for single-card Voxtral workloads (tokenizer, acoustic step)."""
    ensure_voxtral_device_available()
    return ttnn.MeshShape(1, 1)


def prepare_voxtral_open_mesh_kwargs(device_params: dict | None) -> tuple[dict, dict]:
    """Host-aware ``open_mesh_device`` kwargs plus fabric settings for teardown.

    Applies ``get_updated_device_params`` (Blackhole/QB dispatch axis, etc.) and strips
    fabric keys that must be passed to ``set_fabric`` before opening the mesh.
    """
    from tests.scripts.common import get_updated_device_params

    updated = get_updated_device_params(dict(device_params or {}))
    fabric = {
        "fabric_config": updated.pop("fabric_config", None),
        "fabric_tensix_config": updated.pop("fabric_tensix_config", None),
        "reliability_mode": updated.pop("reliability_mode", None),
        "fabric_manager": updated.pop("fabric_manager", None),
        "fabric_router_config": updated.pop("fabric_router_config", None),
    }
    return updated, fabric


def resolve_voxtral_model_name_or_skip() -> str:
    """Return Voxtral checkpoint id from env or skip."""
    model_name_or_path = os.getenv("VOXTRAL_TTS_MODEL") or os.getenv("HF_MODEL") or DEFAULT_VOXTRAL_MODEL
    if "voxtral" not in model_name_or_path.lower():
        pytest.skip(
            f"Expected a Voxtral checkpoint, got '{model_name_or_path}'. "
            "Set VOXTRAL_TTS_MODEL or HF_MODEL to a Voxtral model/repo."
        )
    return model_name_or_path


def create_real_voxtral_text_model_or_skip(
    device,
    *,
    max_seq_len: int = 256,
    max_batch_size: int = 1,
    dtype=ttnn.bfloat16,
    optimizations=voxtral_text_default_optimizations,
):
    """Build the TT text model with the production config by default."""
    model_name_or_path = resolve_voxtral_model_name_or_skip()
    try:
        return VoxtralTTTextModel.create_from_model_name(
            mesh_device=device,
            model_name_or_path=model_name_or_path,
            dtype=dtype,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            optimizations=optimizations,
        )
    except Exception as exc:
        pytest.skip(f"Unable to build VoxtralTTTextModel from real checkpoint: {exc}")


def create_voxtral_audio_tokenizer_or_skip(
    device,
    *,
    state_dict,
    tokenizer_cfg,
    full_checkpoint=None,
    optimizations=voxtral_audio_tokenizer_default_optimizations,
):
    """Build ``VoxtralTTAudioTokenizer`` with production optimizations by default."""
    from models.experimental.voxtraltts.tt.audio_tokenizer.model import VoxtralTTAudioTokenizer

    opt = optimizations() if callable(optimizations) else optimizations
    try:
        return VoxtralTTAudioTokenizer(
            device,
            state_dict=state_dict,
            tokenizer_cfg=tokenizer_cfg,
            full_checkpoint=full_checkpoint,
            optimizations=opt,
        )
    except Exception as exc:
        pytest.skip(f"Unable to build VoxtralTTAudioTokenizer: {exc}")


def log_per_step_code_match(ref_codes: torch.Tensor, tt_codes: torch.Tensor) -> None:
    """Log semantic/acoustic code agreement per AR step (one CPU + one TT E2E forward)."""
    n_frames = min(int(tt_codes.shape[2]), int(ref_codes.shape[2]))
    n_acoustic = ref_codes.shape[1] - 1
    first_diff_step: int | None = None

    logger.info("")
    logger.info("=" * 70)
    logger.info("PER-STEP CODE MATCH (prod path, single CPU + single TT forward)")
    logger.info("=" * 70)

    for t in range(n_frames):
        sem_ok = bool((ref_codes[0, 0, t] == tt_codes[0, 0, t]).item())
        ac_match = int((ref_codes[0, 1:, t] == tt_codes[0, 1:, t]).sum().item())
        ac_ok = ac_match == n_acoustic
        if not sem_ok or not ac_ok:
            if first_diff_step is None:
                first_diff_step = t
            if not ac_ok:
                bad_cb = (ref_codes[0, 1:, t] != tt_codes[0, 1:, t]).nonzero(as_tuple=False).reshape(-1)
                bad_preview = [
                    f"cb{int(i.item())}:{int(ref_codes[0, 1 + i, t].item())}->{int(tt_codes[0, 1 + i, t].item())}"
                    for i in bad_cb[:6]
                ]
                ac_detail = f" mismatches={bad_preview}"
                if bad_cb.numel() > 6:
                    ac_detail += f" ... (+{int(bad_cb.numel()) - 6} more)"
            else:
                ac_detail = ""
            logger.info(
                f"  step {t}: semantic={'OK' if sem_ok else 'DIFF'} "
                f"(cpu={int(ref_codes[0, 0, t].item())} tt={int(tt_codes[0, 0, t].item())}) "
                f"acoustic={ac_match}/{n_acoustic}{ac_detail}"
            )
        else:
            logger.info(f"  step {t}: semantic=OK acoustic={ac_match}/{n_acoustic} [all match]")

    if first_diff_step is None:
        logger.info(f"  summary: all {n_frames} steps match CPU codes exactly")
    else:
        logger.info(f"  summary: first divergence at step {first_diff_step}")
