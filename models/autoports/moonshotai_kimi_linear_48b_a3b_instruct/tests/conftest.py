# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Autoport test fixtures (used with -c /dev/null --confcutdir <autoport>, i.e. without tt-metal's root conftest).

Environment contract (set by ~/kimi-linear-bringup/scripts/lib.sh model_env):
  KIMI_MESH_SHAPE      1x1 | 1x2 | 1x4          (default 1x1)
  TT_MESH_PARENT_SHAPE 1x4                      open this parent and take KIMI_MESH_SHAPE as a submesh (QB2: a bare 1x2 cannot init fabric)
  TT_METAL_VISIBLE_DEVICES                      set by the caller for 1x1
  KIMI_FABRIC          FABRIC_1D | FABRIC_1D_RING | DISABLED (default: RING for >= 4 chips, 1D for submeshes, DISABLED for 1x1)
  KIMI_TRACE_REGION    bytes of trace region (default 0)
  KIMI_SNAPSHOT        HF snapshot dir (tests needing real weights skip when unset)
  KIMI_GOLDEN_ROOT     dir with layer_goldens.pt (stage 02)
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

import ttnn
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.reference.config import KimiLinearConfig
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.reference.weights import KimiCheckpoint
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.tt.ccl import KimiCCL


def _parse(s: str) -> tuple[int, int]:
    r, c = s.lower().split("x")
    return int(r), int(c)


def mesh_shape() -> tuple[int, int]:
    return _parse(os.environ.get("KIMI_MESH_SHAPE", "1x1"))


def _fabric_for(n: int, submesh: bool):
    env = os.environ.get("KIMI_FABRIC")
    if env:
        return None if env == "DISABLED" else getattr(ttnn.FabricConfig, env)
    if n == 1 and not submesh:
        return None
    return (
        ttnn.FabricConfig.FABRIC_1D
        if submesh
        else (ttnn.FabricConfig.FABRIC_1D_RING if n >= 4 else ttnn.FabricConfig.FABRIC_1D)
    )


def set_fabric(fabric_config):
    if fabric_config is None:
        return
    ttnn.set_fabric_config(
        fabric_config,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
    )


@pytest.fixture(scope="session")
def mesh_device():
    r, c = mesh_shape()
    parent_env = os.environ.get("TT_MESH_PARENT_SHAPE")
    trace = int(os.environ.get("KIMI_TRACE_REGION", "0"))
    submesh = bool(parent_env) and _parse(parent_env) != (r, c)
    fabric = _fabric_for(r * c, submesh)
    set_fabric(fabric)
    parent = None
    if submesh:
        pr, pc = _parse(parent_env)
        parent = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(pr, pc), trace_region_size=trace)
        mesh = parent.create_submesh(ttnn.MeshShape(r, c))
    else:
        mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(r, c), trace_region_size=trace)
    assert mesh.get_num_devices() == r * c, (mesh.get_num_devices(), r, c)
    yield mesh
    ttnn.close_mesh_device(mesh)
    if parent is not None:
        ttnn.close_mesh_device(parent)
    if fabric is not None:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.fixture(scope="session")
def ccl(mesh_device):
    return KimiCCL(mesh_device)


@pytest.fixture(scope="session")
def snapshot() -> Path:
    s = os.environ.get("KIMI_SNAPSHOT")
    if not s or not (Path(s) / "model.safetensors.index.json").is_file():
        pytest.skip("KIMI_SNAPSHOT not set / weights not downloaded")
    return Path(s)


@pytest.fixture(scope="session")
def hf_config(snapshot) -> KimiLinearConfig:
    cfg = KimiLinearConfig.from_snapshot(snapshot)
    cfg.validate()
    return cfg


@pytest.fixture(scope="session")
def checkpoint(snapshot, hf_config) -> KimiCheckpoint:
    return KimiCheckpoint(snapshot, hf_config)


@pytest.fixture(scope="session")
def goldens():
    root = os.environ.get("KIMI_GOLDEN_ROOT")
    p = Path(root or "") / "layer_goldens.pt"
    if not root or not p.is_file():
        pytest.skip("KIMI_GOLDEN_ROOT/layer_goldens.pt missing (stage 02)")
    return torch.load(p, weights_only=False)


@pytest.fixture(scope="session")
def cache_path(tmp_path_factory) -> Path:
    root = os.environ.get("TT_CACHE_PATH")
    p = Path(root) / "kimi_linear_48b" / f"tp{mesh_shape()[1]}" if root else tmp_path_factory.mktemp("ttcache")
    p.mkdir(parents=True, exist_ok=True)
    return p
