# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""VRAM preflight helpers for handler preprocessing (TTNN demo path)."""

from typing import Any, Dict, List, Optional, Union

import torch
from acestep.constants import DEFAULT_DIT_INSTRUCTION
from loguru import logger


class GenerateMusicMixin:
    """Preprocess-time helpers shared with ``official_lm_preprocess``."""

    def _verify_decoder_device_dtype(self) -> None:
        """Ensure decoder parameters are on ``self.device`` and ``self.dtype``."""
        decoder = getattr(self.model, "decoder", None)
        if decoder is None:
            return

        wrong_device = []
        wrong_dtype = []
        for name, param in decoder.named_parameters():
            if not self._is_on_target_device(param, self.device):
                wrong_device.append(name)
            if param.is_floating_point() and param.dtype != self.dtype:
                wrong_dtype.append(name)

        if wrong_device or wrong_dtype:
            if wrong_device:
                logger.warning(
                    f"[generate_music] LoRA sanity check: {len(wrong_device)} decoder "
                    f"parameters on wrong device (expected {self.device}), fixing: "
                    f"{wrong_device[:5]}{'...' if len(wrong_device) > 5 else ''}"
                )
            if wrong_dtype:
                logger.warning(
                    f"[generate_music] LoRA sanity check: {len(wrong_dtype)} decoder "
                    f"parameters have wrong dtype (expected {self.dtype}), fixing: "
                    f"{wrong_dtype[:5]}{'...' if len(wrong_dtype) > 5 else ''}"
                )
            decoder.to(device=self.device, dtype=self.dtype)

        still_wrong = [name for name, p in decoder.named_parameters() if not self._is_on_target_device(p, self.device)]
        if still_wrong:
            logger.warning(
                f"[generate_music] {len(still_wrong)} params still on wrong device "
                f"after decoder.to(), using recursive move"
            )
            self._recursive_to_device(decoder, self.device, self.dtype)

    def _vram_preflight_check(
        self,
        actual_batch_size: int,
        audio_duration: Optional[float],
        guidance_scale: float,
    ) -> Optional[Dict[str, Any]]:
        """VRAM checks are disabled for TTNN host_preprocess (CPU batching only)."""
        _ = actual_batch_size, audio_duration, guidance_scale
        return None

    def generate_music(
        self,
        captions: str,
        global_caption: str = "",
        lyrics: str = "",
        bpm: Optional[int] = None,
        key_scale: str = "",
        time_signature: str = "",
        vocal_language: str = "en",
        inference_steps: int = 8,
        guidance_scale: float = 7.0,
        use_random_seed: bool = True,
        seed: Optional[Union[str, float, int]] = -1,
        reference_audio=None,
        audio_duration: Optional[float] = None,
        batch_size: Optional[int] = None,
        src_audio=None,
        audio_code_string: Union[str, List[str]] = "",
        repainting_start: float = 0.0,
        repainting_end: Optional[float] = None,
        instruction: str = DEFAULT_DIT_INSTRUCTION,
        audio_cover_strength: float = 1.0,
        cover_noise_strength: float = 0.0,
        task_type: str = "text2music",
        use_adg: bool = False,
        cfg_interval_start: float = 0.0,
        cfg_interval_end: float = 1.0,
        shift: float = 1.0,
        infer_method: str = "ode",
        sampler_mode: str = "euler",
        velocity_norm_threshold: float = 0.0,
        velocity_ema_factor: float = 0.0,
        dcw_enabled: bool = True,
        dcw_mode: str = "double",
        dcw_scaler: float = 0.05,
        dcw_high_scaler: float = 0.02,
        dcw_wavelet: str = "haar",
        use_tiled_decode: bool = True,
        timesteps: Optional[List[float]] = None,
        latent_shift: float = 0.0,
        latent_rescale: float = 1.0,
        chunk_mask_mode: str = "auto",
        repaint_latent_crossfade_frames: int = 10,
        repaint_wav_crossfade_sec: float = 0.0,
        repaint_mode: str = "balanced",
        repaint_strength: float = 0.5,
        source_repaint_latents: Optional[torch.Tensor] = None,
        retake_seed: Optional[Union[str, float, int]] = None,
        retake_variance: float = 0.0,
        flow_edit_morph: bool = False,
        flow_edit_source_caption: str = "",
        flow_edit_source_lyrics: str = "",
        flow_edit_n_min: float = 0.0,
        flow_edit_n_max: float = 1.0,
        flow_edit_n_avg: int = 1,
        progress=None,
    ) -> Dict[str, Any]:
        """Full PyTorch generation was removed from this vendored handler subset."""
        raise NotImplementedError(
            "AceStepHandler.generate_music() is not available in the trimmed vendored handler. "
            "Use the TTNN demo path instead."
        )
