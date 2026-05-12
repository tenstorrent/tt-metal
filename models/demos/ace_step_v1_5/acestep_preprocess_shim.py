"""
Types and small helpers mirrored from ``acestep.inference`` for TTNN demos.

``acestep.inference`` imports ``acestep.audio_utils`` (``torchaudio``) at module
load time.  The ``run_prompt_to_wav`` path only needs ``GenerationParams``,
``GenerationConfig``, and the repaint/LM metadata helpers — keep them here to
avoid pulling in the full ``acestep.inference`` import chain.

Keep in sync with ACE-Step ``acestep/inference.py`` when upstream adds fields.
"""

from __future__ import annotations

import glob
import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from loguru import logger


@dataclass
class GenerationParams:
    task_type: str = "text2music"
    instruction: str = "Fill the audio semantic mask based on the given conditions:"
    reference_audio: Optional[str] = None
    src_audio: Optional[str] = None
    audio_codes: str = ""
    caption: str = ""
    global_caption: str = ""
    lyrics: str = ""
    instrumental: bool = False
    vocal_language: str = "unknown"
    bpm: Optional[int] = None
    keyscale: str = ""
    timesignature: str = ""
    duration: float = -1.0
    enable_normalization: bool = True
    normalization_db: float = -1.0
    fade_in_duration: float = 0.0
    fade_out_duration: float = 0.0
    latent_shift: float = 0.0
    latent_rescale: float = 1.0
    inference_steps: int = 8
    seed: int = -1
    guidance_scale: float = 7.0
    use_adg: bool = False
    cfg_interval_start: float = 0.0
    cfg_interval_end: float = 1.0
    shift: float = 1.0
    infer_method: str = "ode"
    sampler_mode: str = "euler"
    velocity_norm_threshold: float = 0.0
    velocity_ema_factor: float = 0.0
    dcw_enabled: bool = True
    dcw_mode: str = "double"
    dcw_scaler: float = 0.05
    dcw_high_scaler: float = 0.02
    dcw_wavelet: str = "haar"
    timesteps: Optional[List[float]] = None
    repainting_start: float = 0.0
    repainting_end: float = -1
    chunk_mask_mode: str = "auto"
    repaint_latent_crossfade_frames: int = 10
    repaint_wav_crossfade_sec: float = 0.0
    repaint_mode: str = "balanced"
    repaint_strength: float = 0.5
    retake_seed: Optional[Union[str, int]] = None
    retake_variance: float = 0.0
    flow_edit_morph: bool = False
    flow_edit_source_caption: str = ""
    flow_edit_source_lyrics: str = ""
    flow_edit_n_min: float = 0.0
    flow_edit_n_max: float = 1.0
    flow_edit_n_avg: int = 1
    audio_cover_strength: float = 1.0
    cover_noise_strength: float = 0.0
    thinking: bool = True
    lm_temperature: float = 0.85
    lm_cfg_scale: float = 2.0
    lm_top_k: int = 0
    lm_top_p: float = 0.9
    lm_negative_prompt: str = "NO USER INPUT"
    use_cot_metas: bool = True
    use_cot_caption: bool = True
    use_cot_lyrics: bool = False
    use_cot_language: bool = True
    use_constrained_decoding: bool = True
    cot_bpm: Optional[int] = None
    cot_keyscale: str = ""
    cot_timesignature: str = ""
    cot_duration: Optional[float] = None
    cot_vocal_language: str = "unknown"
    cot_caption: str = ""
    cot_lyrics: str = ""

    def __post_init__(self) -> None:
        if self.shift is not None and self.shift <= 0:
            self.shift = 1.0
        if self.inference_steps is not None and self.inference_steps < 1:
            self.inference_steps = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationConfig:
    batch_size: int = 2
    allow_lm_batch: bool = False
    use_random_seed: bool = True
    seeds: Optional[List[int]] = None
    lm_batch_chunk_size: int = 8
    constrained_decoding_debug: bool = False
    audio_format: str = "flac"
    mp3_bitrate: str = "128k"
    mp3_sample_rate: int = 48000

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CachedRepaintSource:
    latents: torch.Tensor
    source_seed: Optional[int]
    latent_path: str


def _update_metadata_from_lm(
    metadata: Dict[str, Any],
    bpm: Optional[int],
    key_scale: str,
    time_signature: str,
    audio_duration: Optional[float],
    vocal_language: str,
    caption: str,
    lyrics: str,
) -> Tuple[Optional[int], str, str, Optional[float], str, str, str]:
    if bpm is None and metadata.get("bpm"):
        bpm_value = metadata.get("bpm")
        if bpm_value not in ["N/A", ""]:
            try:
                bpm = int(bpm_value)
            except (ValueError, TypeError):
                pass

    if not key_scale and metadata.get("keyscale"):
        key_scale_value = metadata.get("keyscale", metadata.get("key_scale", ""))
        if key_scale_value != "N/A":
            key_scale = key_scale_value

    if not time_signature and metadata.get("timesignature"):
        time_signature_value = metadata.get("timesignature", metadata.get("time_signature", ""))
        if time_signature_value != "N/A":
            time_signature = time_signature_value

    if audio_duration is None or audio_duration <= 0:
        audio_duration_value = metadata.get("duration", -1)
        if audio_duration_value not in ["N/A", ""]:
            try:
                audio_duration = float(audio_duration_value)
            except (ValueError, TypeError):
                pass

    if not vocal_language and metadata.get("vocal_language"):
        vocal_language = metadata.get("vocal_language")
    if not caption and metadata.get("caption"):
        caption = metadata.get("caption")
    if not lyrics and metadata.get("lyrics"):
        lyrics = metadata.get("lyrics")
    return bpm, key_scale, time_signature, audio_duration, vocal_language, caption, lyrics


def _candidate_repaint_sidecars(src_audio: str) -> List[str]:
    expanded_audio = os.path.expanduser(src_audio)
    candidates = [os.path.splitext(expanded_audio)[0] + ".json"]
    basename = os.path.splitext(os.path.basename(expanded_audio))[0]
    if basename:
        results_root = os.path.join(os.getcwd(), "gradio_outputs")
        sidecars = glob.glob(os.path.join(results_root, "batch_*", f"{glob.escape(basename)}.json"))
        candidates.extend(sorted(sidecars, key=os.path.getmtime, reverse=True))
    seen = set()
    unique_candidates = []
    for candidate in candidates:
        normalized = os.path.abspath(candidate)
        if normalized not in seen:
            seen.add(normalized)
            unique_candidates.append(normalized)
    return unique_candidates


def _coerce_seed_value(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, list):
        value = value[0] if value else None
    if value is None:
        return None
    try:
        text = str(value).split(",")[0].strip()
        if not text:
            return None
        seed = int(float(text))
    except (TypeError, ValueError, OverflowError):
        return None
    return seed if seed >= 0 else None


def _resample_matching_source_seeds(seeds: List[int], source_seed: Optional[int]) -> List[int]:
    if source_seed is None:
        return seeds
    resolved = list(seeds)
    for index, seed in enumerate(resolved):
        if seed != source_seed:
            continue
        replacement = random.randint(0, 2**32 - 1)
        while replacement == source_seed:
            replacement = random.randint(0, 2**32 - 1)
        logger.info(
            "[repaint_cache] Replacing repaint seed {} with {} to avoid reusing source seed",
            source_seed,
            replacement,
        )
        resolved[index] = replacement
    return resolved


def _load_cached_repaint_source(src_audio: Optional[str]) -> Optional[CachedRepaintSource]:
    if not src_audio:
        return None
    try:
        audio_path = os.fspath(src_audio)
    except TypeError:
        return None
    sidecars = _candidate_repaint_sidecars(audio_path)
    json_path = next((path for path in sidecars if os.path.exists(path)), None)
    if json_path is None:
        logger.info(
            "[repaint_cache] No cached source latents found for src_audio={} candidates={}",
            audio_path,
            sidecars,
        )
        return None
    try:
        with open(json_path, encoding="utf-8") as file_obj:
            params = json.load(file_obj)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(params, dict):
        return None
    latent_file = str(params.get("repaint_source_latents_file") or "").strip()
    if not latent_file:
        return None
    latent_path = latent_file
    if not os.path.isabs(latent_path):
        latent_path = os.path.join(os.path.dirname(json_path), latent_file)
    latent_path = os.path.expanduser(latent_path)
    if not os.path.exists(latent_path):
        logger.warning("[repaint_cache] Cached repaint latents missing: {}", latent_path)
        return None
    try:
        latents = np.load(latent_path).astype(np.float32)
    except (OSError, ValueError) as exc:
        logger.warning("[repaint_cache] Could not load cached repaint latents: {}", exc)
        return None
    if latents.ndim != 2:
        logger.warning("[repaint_cache] Cached repaint latents must be shaped [T, C]")
        return None
    logger.info("[repaint_cache] Loaded cached repaint source latents from {}", latent_path)
    return CachedRepaintSource(
        latents=torch.from_numpy(latents),
        source_seed=_coerce_seed_value(params.get("seed")),
        latent_path=latent_path,
    )
