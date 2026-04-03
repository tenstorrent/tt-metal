#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Static checks: **32** video frames + 8-device mesh → four 8-way ViT frame-DP rounds
(text TP=8 on the same mesh; vision parallel width matches device count).

No heavy demo imports (no einops/torch required). Run from ``tt-metal`` root:

    python3 -m models.demos.molmo2.demo.verify_batch32_frame_dp

On T3K (8 devices), also run the real demo, e.g.::

    python -m models.demos.molmo2.demo.demo \\
      --video /path/to/video.mp4 \\
      --max-video-frames 32 \\
      --max-tokens 16

Pass ``--max-video-frames 32`` to the demo for this scenario. Vision ``pixel_values`` batch
equals ``n_frames``; text ``batch_size`` in ``Molmo2Generator`` stays ``1``.
"""
from __future__ import annotations

import os
import sys

# Scenario: 32-frame vision batch (independent of ``demo.VIDEO_MAX_FRAMES`` default).
FRAME_BATCH_SCENARIO = 32


def _effective_video_max_frames(requested: int) -> int:
    """Mirror ``demo.effective_video_max_frames`` without importing ``demo.py``."""
    from models.demos.molmo2.tt.frame_parallel_config import video_align_frames_to_mesh_width

    cap = requested
    raw = os.environ.get("MOLMO2_VIDEO_MAX_FRAMES", "").strip()
    if raw:
        try:
            cap = max(1, int(raw))
        except ValueError:
            pass
    if not video_align_frames_to_mesh_width():
        return cap
    try:
        import ttnn

        n = len(ttnn.get_device_ids())
    except Exception:
        return cap
    if n <= 1:
        return cap
    aligned = (cap // n) * n
    return aligned if aligned > 0 else cap


def _mesh_width() -> int:
    try:
        import ttnn

        return max(1, len(ttnn.get_device_ids()))
    except Exception:
        return 8


def main(argv: list[str] | None = None) -> int:
    from models.demos.molmo2.tt.frame_parallel_config import (
        molmo2_vision_parallel_milestone,
        video_align_frames_to_mesh_width,
        vision_frame_dp_enabled,
        vision_frame_dp_remainder_mode,
    )

    cap = FRAME_BATCH_SCENARIO
    mesh_w = _mesh_width()
    eff = _effective_video_max_frames(cap)

    errors: list[str] = []

    if eff % mesh_w != 0 and mesh_w > 1:
        errors.append(
            f"effective frame cap {eff} is not divisible by mesh width {mesh_w} "
            f"(remainder={vision_frame_dp_remainder_mode()!r} will use tail/pad/gather for leftovers)"
        )

    if not vision_frame_dp_enabled():
        errors.append("Frame DP disabled: set MOLMO2_VISION_PARALLEL_MILESTONE to 1 or unset (default 1)")

    rounds = eff // mesh_w if mesh_w and eff % mesh_w == 0 else None
    print("Molmo2: 32-frame vision batch vs 8-wide frame DP (text TP=8 mesh)")
    print(f"  Frame batch under test:         {FRAME_BATCH_SCENARIO}")
    print(f"  effective_video_max_frames(32): {eff}")
    print(f"  Host mesh width (device ids):   {mesh_w}")
    print(f"  Milestone / frame_dp:           {molmo2_vision_parallel_milestone()} / {vision_frame_dp_enabled()}")
    print(f"  Align frames to mesh:           {video_align_frames_to_mesh_width()}")
    print(f"  Remainder mode:                 {vision_frame_dp_remainder_mode()!r}")
    if rounds is not None:
        print(f"  Full DP rounds at 8 devices:    {eff} / {mesh_w} = {rounds}")
    print("  Text generator batch_size:        1 (single sequence)")

    if errors:
        print("\nWARN / FAIL:")
        for e in errors:
            print(f"  - {e}")
        return 1

    print("\nOK: Default 32 frames align with 8-wide frame DP when the mesh exposes 8 devices.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
