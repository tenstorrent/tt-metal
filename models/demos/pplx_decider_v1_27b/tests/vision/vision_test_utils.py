# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the vision-tower tests: the bf16 HF goldens, the tower build, PCC logging.

Goldens (``reference/hf_vision_reference.py``, bit-exact vs HF ``get_image_features``):
``$PPLX_DECIDER_VISION_GOLDEN/{inputs,tower}/<id>*.safetensors``, 8 images v01..v08 with 256..1024
patches (v02 = 936 patches in the 1024 bucket, the padded-key-mask case).
"""

from __future__ import annotations

import json
import os
import time
from functools import lru_cache
from pathlib import Path

import pytest
import torch
from loguru import logger
from safetensors.torch import load_file

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc

GOLDEN_DIR = Path(
    os.environ.get("PPLX_DECIDER_VISION_GOLDEN", "/local/ttuser/gtobar/artifacts/pplx_decider/goldens/vision")
)
STAGE_DIR = Path(os.environ.get("PPLX_DECIDER_STAGE12A_DIR", "/local/ttuser/gtobar/artifacts/pplx_decider/stage12a"))
PCC_LOG = Path(os.environ.get("PPLX_DECIDER_VISION_PCC_LOG", STAGE_DIR / "pcc_vision.jsonl"))
IMAGES = (
    "v01_dominant_color",
    "v02_count_circles",
    "v03_receipt_total",
    "v04_tallest_bar",
    "v05_red_circle_yes",
    "v06_red_circle_no",
    "v07_brightness_dark",
    "v08_progress_fill",
)
MODULE_THRESHOLD = 0.995  # per module / block / merger, teacher forced
TOWER_THRESHOLD = 0.99  # whole tower end to end from pixels
DEVICE_PARAMS = {"l1_small_size": 24576}


def image_ids(images=IMAGES):
    return [i.split("_")[0] for i in images]


@lru_cache(maxsize=1)
def reader():
    from models.demos.pplx_decider_v1_27b.reference.hf_reference import SnapshotReader

    return SnapshotReader()


def golden_inputs(image: str) -> dict[str, torch.Tensor]:
    path = GOLDEN_DIR / "inputs" / f"{image}.safetensors"
    if not path.exists():
        pytest.fail(f"Missing vision golden {path} (reference/hf_vision_reference.py)")
    return load_file(str(path))


@lru_cache(maxsize=2)
def golden_tower(image: str) -> dict[str, torch.Tensor]:
    path = GOLDEN_DIR / "tower" / f"{image}.bf16.safetensors"
    if not path.exists():
        pytest.fail(f"Missing vision golden {path} (reference/hf_vision_reference.py)")
    return load_file(str(path))


def block_input(golden: dict[str, torch.Tensor], idx: int) -> torch.Tensor:
    """HF input of block ``idx``: the pos-embedded patch embedding for block 0, else block ``idx-1`` output."""
    return golden["embed_with_pos"] if idx == 0 else golden[f"block_{idx - 1:02d}"]


def build_tower(device):
    from models.demos.pplx_decider_v1_27b.tt.vision.tower import PplxVisionTower

    return PplxVisionTower.from_snapshot(device, reader())


def prepare(tower, image: str, **kwargs):
    enc = golden_inputs(image)
    return tower.prepare_inputs(enc["pixel_values"], enc["image_grid_thw"], **kwargs)


def to_host_rows(x: ttnn.Tensor, rows: int) -> torch.Tensor:
    """Device [1, 1, S, D] -> host fp32 [rows, D] (the real rows of a padded bucket)."""
    out = ttnn.to_torch(x).to(torch.float32)
    return out.reshape(-1, out.shape[-1])[:rows]


def pcc_record(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    module: str,
    image: str,
    block: int | None = None,
    threshold: float = MODULE_THRESHOLD,
    bucket: int | None = None,
    extra: dict | None = None,
) -> dict:
    """PCC + max abs diff, appended to the stage-12A PCC log. Does not assert (callers gather, then assert)."""
    reference, candidate = reference.to(torch.float32), candidate.to(torch.float32)
    assert reference.shape == candidate.shape, f"{module}: {tuple(reference.shape)} vs {tuple(candidate.shape)}"
    passing, pcc = comp_pcc(reference, candidate, threshold)
    _, allclose_msg = comp_allclose(reference, candidate)
    record = dict(
        module=module,
        block=block,
        image=image,
        rows=int(reference.shape[0]),
        bucket=bucket,
        pcc=float(pcc),
        max_abs_diff=float((reference - candidate).abs().max()),
        threshold=threshold,
        passed=bool(passing),
        policy="vision_bf16__w_bf16__hifi4",
        time=time.strftime("%Y-%m-%d %H:%M:%S"),
        **(extra or {}),
    )
    PCC_LOG.parent.mkdir(parents=True, exist_ok=True)
    with PCC_LOG.open("a") as f:
        f.write(json.dumps(record) + "\n")
    logger.info(
        f"PCC {module}{'' if block is None else f' B{block}'} {image}: {record['pcc']:.6f} (>= {threshold}); {allclose_msg}"
    )
    return record
