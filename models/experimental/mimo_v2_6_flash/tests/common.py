# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch

import ttnn
from models.experimental.mimo_v2_6_flash.reference.checkpoint import MiMoCheckpoint, default_model_path
from models.experimental.mimo_v2_6_flash.reference.config import MiMoConfig
from models.experimental.mimo_v2_6_flash.tt.generator import create_dp_submeshes

# CCL ops keep their semaphores for the program's lifetime. Without an L1_SMALL region they go in the
# main L1 pool, and after enough distinct prefill shapes they fragment it badly enough to clash with
# later ops' static circular buffers. all_gather places them in L1_SMALL when one exists.
L1_SMALL_SIZE = 65536
GALAXY_DEVICE_PARAMS = {
    "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
    "trace_region_size": 0,
    "l1_small_size": L1_SMALL_SIZE,
}
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
    """Converted TTNN weights for ``tag``, kept per checkpoint: MOPD and RL share tensor names, not values."""
    root = os.environ.get("MIMO_TT_CACHE", "/home/zni/models/mimo-tt-cache")
    return os.path.join(root, os.path.basename(os.path.normpath(model_path())), tag)


def golden_path():
    """Golden hidden states / logits of reference/make_golden.py, kept per checkpoint (MIMO_GOLDEN overrides)."""
    ckpt_name = os.path.basename(os.path.normpath(model_path()))
    path = os.environ.get("MIMO_GOLDEN", os.path.join("/home/zni/models/mimo-golden", ckpt_name, "golden_48l.pt"))
    if not os.path.exists(path):
        pytest.skip(f"golden file {path} not found; run reference/make_golden.py")
    return path


def pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def dp_submeshes(mesh_device):
    """The 4x8 Galaxy split into its four 1x8 rows: one TP=8 replica per row."""
    assert tuple(mesh_device.shape) == (DP, TP), f"expected a {DP}x{TP} mesh, got {tuple(mesh_device.shape)}"
    subs = create_dp_submeshes(mesh_device, DP)  # also registers the streaming kernels' bank cores per row
    assert len(subs) == DP
    return subs


def block_page_table(max_batch, blocks_per_user, seed=0):
    """A shuffled slot -> block mapping, so the paged path is exercised with non-contiguous blocks."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(max_batch * blocks_per_user, generator=g)
    return perm.reshape(max_batch, blocks_per_user).to(torch.int32)
