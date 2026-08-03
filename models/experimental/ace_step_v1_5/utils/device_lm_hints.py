# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Device-native LM hint helpers (detok → condition encoder without torch round-trip)."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def ace_step_device_native_detok_hints() -> bool:
    """Default: keep TTNN detok hints on device until the condition encoder consumes them."""
    if _env_truthy("ACE_STEP_TORCH_DETOK_HINTS"):
        return False
    if _env_truthy("ACE_STEP_PYTORCH_DETOK"):
        return False
    return True


def pad_lm_hints_tt_to_frames(
    hints_tt: Any,
    *,
    max_frames: int,
    silence_latent: torch.Tensor,
    device: Any,
    mem: Any,
    dtype: Any,
) -> Any:
    """Pad or trim ``[1, T, 64]`` TTNN hints to ``max_frames`` using silence tail on device."""
    import ttnn

    from models.experimental.ace_step_v1_5.ttnn_impl.math_perf_env import ace_step_concat_kwargs

    t = int(hints_tt.shape[1])
    max_f = int(max_frames)
    if t > max_f:
        return ttnn.slice(hints_tt, (0, 0, 0), (1, max_f, 64))
    if t == max_f:
        return hints_tt

    pad_len = max_f - t
    sil = silence_latent
    if sil.dim() == 2:
        sil = sil.unsqueeze(0)
    pad_np = sil[:, :pad_len, :].detach().cpu().numpy().astype(np.float32)
    pad_tt = ttnn.as_tensor(
        np.ascontiguousarray(pad_np),
        device=device,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=mem,
    )
    _ck = ace_step_concat_kwargs(ttnn)
    if hasattr(ttnn, "concat"):
        return ttnn.concat([hints_tt, pad_tt], dim=1, **_ck)
    return ttnn.concatenate([hints_tt, pad_tt], dim=1, **_ck)


def repair_degenerate_lm_hint_tail_tt(
    payload: dict[str, Any],
    code_string: str,
    target_frames: int,
    *,
    device: Any,
    mem: Any,
    dtype: Any,
    min_good_codes: int = 50,
) -> bool:
    """Silence-pad device hint tail when LM stream degenerates (mirror torch repair)."""
    import ttnn

    from models.experimental.ace_step_v1_5.ttnn_impl.audio_code_detokenizer import parse_audio_code_string
    from models.experimental.ace_step_v1_5.ttnn_impl.math_perf_env import ace_step_concat_kwargs
    from models.experimental.ace_step_v1_5.utils.official_lm_preprocess import find_degenerate_code_prefix_len

    hints_tt = payload.get("precomputed_lm_hints_25Hz_tt")
    sil = payload.get("silence_latent")
    if hints_tt is None or sil is None:
        return False

    codes = parse_audio_code_string(str(code_string or ""))
    window = 25
    if len(codes) < min_good_codes + window:
        return False
    prefix_len = find_degenerate_code_prefix_len(codes)
    if prefix_len is None or prefix_len < min_good_codes:
        return False
    if prefix_len >= int(len(codes) * 0.85):
        return False

    target = int(target_frames)
    good_frames = int(prefix_len * 5)
    if good_frames <= 0 or good_frames >= target:
        return False

    pad = sil
    if pad.dim() == 2:
        pad = pad.unsqueeze(0)
    pad_np = pad[:, : target - good_frames, :].detach().cpu().numpy().astype(np.float32)
    pad_tt = ttnn.as_tensor(
        np.ascontiguousarray(pad_np),
        device=device,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=mem,
    )
    good_tt = ttnn.slice(hints_tt, (0, 0, 0), (1, good_frames, 64))
    _ck = ace_step_concat_kwargs(ttnn)
    if hasattr(ttnn, "concat"):
        repaired = ttnn.concat([good_tt, pad_tt], dim=1, **_ck)
    else:
        repaired = ttnn.concatenate([good_tt, pad_tt], dim=1, **_ck)
    payload["precomputed_lm_hints_25Hz_tt"] = repaired
    payload["precomputed_lm_hints_25Hz"] = None
    print(
        f"[ace_step_v1_5] LM code stream degenerates after ~{prefix_len} codes "
        f"(≈{prefix_len / 5.0:.1f}s): device hint tail silence-padded "
        f"({good_frames}→{target} frames) — garbage tail codes sound like noise in ctx",
        flush=True,
    )
    return True


def ace_step_materialize_payload_lm_hints_for_handoff(payload: dict[str, Any]) -> None:
    """Convert device hints to host torch for pickle / cross-device mesh handoff."""
    hints_tt = payload.get("precomputed_lm_hints_25Hz_tt")
    if hints_tt is None:
        return
    if payload.get("precomputed_lm_hints_25Hz") is not None:
        payload.pop("precomputed_lm_hints_25Hz_tt", None)
        return

    import ttnn

    out = ttnn.to_torch(hints_tt).float()
    if out.ndim == 2:
        out = out.unsqueeze(0)
    payload["precomputed_lm_hints_25Hz"] = out.contiguous()
    payload.pop("precomputed_lm_hints_25Hz_tt", None)
    try:
        ttnn.deallocate(hints_tt)
    except Exception:
        pass
