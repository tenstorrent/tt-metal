# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the runnable demo entrypoints in this package."""

from __future__ import annotations

import torch


def save_png(img_denorm, path):
    """Write a [1,3,H,W] tensor in [0,1] to a PNG; falls back to a raw tensor dump if PIL fails."""
    try:
        from PIL import Image

        arr = (img_denorm[0].permute(1, 2, 0).clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
        Image.fromarray(arr).save(path)
        print(f"[demo] wrote {path}")
    except Exception as exc:  # noqa: BLE001
        print(f"[demo] could not write PNG ({exc}); saving tensor instead")
        torch.save(img_denorm, path + ".pt")
