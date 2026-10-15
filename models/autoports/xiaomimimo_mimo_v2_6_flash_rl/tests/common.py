# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch

import ttnn
from models.autoports.xiaomimimo_mimo_v2_6_flash_rl.reference.checkpoint import MiMoCheckpoint, default_model_path
from models.autoports.xiaomimimo_mimo_v2_6_flash_rl.reference.config import MiMoConfig

GALAXY_DEVICE_PARAMS = {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 0}
DP = 4
TP = 8


def model_path():
    path = default_model_path()
    if not os.path.exists(os.path.join(path, "config.json")):
        pytest.skip(f"MiMo checkpoint not found at {path} (set MIMO_MODEL_PATH)")
    return path


def load_config_and_ckpt():
    path = model_path()
    cfg = MiMoConfig.from_model_path(path)
    return cfg, MiMoCheckpoint(path, cfg)


def cache_dir(tag):
    root = os.environ.get("MIMO_TT_CACHE", "/home/zni/models/mimo-tt-cache")
    return os.path.join(root, tag)


def pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def dp_submeshes(mesh_device):
    """The 4x8 Galaxy split into its four 1x8 rows: one TP=8 replica per row."""
    assert tuple(mesh_device.shape) == (DP, TP), f"expected a {DP}x{TP} mesh, got {tuple(mesh_device.shape)}"
    subs = mesh_device.create_submeshes(ttnn.MeshShape(1, TP))
    assert len(subs) == DP
    return subs


def block_page_table(max_batch, blocks_per_user, seed=0):
    """A shuffled slot -> block mapping, so the paged path is exercised with non-contiguous blocks."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(max_batch * blocks_per_user, generator=g)
    return perm.reshape(max_batch, blocks_per_user).to(torch.int32)
