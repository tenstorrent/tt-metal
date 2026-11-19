# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import json
import math
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn

# SC1 is one engine on the whole 4x8 Galaxy with collectives along both mesh axes, so it needs the 2D fabric.
GALAXY_MESH = (4, 8)
GALAXY_DEVICE_PARAMS = {"fabric_config": ttnn.FabricConfig.FABRIC_2D}

# BF16 checkpoint dir (tencent/Hy4-preview); the same variable the prefill reference reads.
HY4_HF_MODEL_ENV = "HY4_HF_MODEL"


def pcc(a, b):
    """Pearson correlation. Raises on NaN: callers take min() over scores, and min() would skip a NaN score."""
    a = a.flatten().double()
    b = b.flatten().double()
    score = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
    assert math.isfinite(score), "PCC is not finite: an input holds NaN/Inf or is constant"
    return score


def check_pcc(name, ref, dev, threshold):
    assert ref.numel() == dev.numel(), f"{name}: reference {tuple(ref.shape)} vs device {tuple(dev.shape)}"
    score = pcc(ref, dev)
    max_diff = (ref.flatten().double() - dev.flatten().double()).abs().max().item()
    logger.info(f"{name}: pcc={score:.6f} | max|Δ|={max_diff:.2e}")
    assert score >= threshold, f"{name}: pcc={score:.6f} | max|Δ|={max_diff:.2e} (threshold {threshold})"


def hy4_tensors(names):
    """Read tensors by name from the BF16 checkpoint in $HY4_HF_MODEL as fp32; skip if it is not there."""
    safetensors = pytest.importorskip("safetensors")
    root = os.environ.get(HY4_HF_MODEL_ENV)
    if not root:
        pytest.skip(f"{HY4_HF_MODEL_ENV} is not set (tencent/Hy4-preview checkpoint dir)")
    index = Path(root) / "model.safetensors.index.json"
    if not index.is_file():
        pytest.skip(f"{index} not found")
    weight_map = json.loads(index.read_text())["weight_map"]
    out = []
    for name in names:
        shard = Path(root) / weight_map[name]
        if not shard.is_file():
            pytest.skip(f"{shard.name} (holds {name}) is not downloaded")
        with safetensors.safe_open(str(shard), framework="pt") as f:
            out.append(f.get_tensor(name).float())
    return out
