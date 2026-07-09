# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest helpers for Seamless M4T v2 PCC tests."""

from __future__ import annotations

import os
from typing import Any

import pytest
import ttnn

from models.experimental.seamless_m4t_v2_large.scripts.download_weights import ensure_seamless_m4t_v2_large_weights


def weights_dir_or_skip() -> str:
    try:
        return ensure_seamless_m4t_v2_large_weights()
    except ImportError as e:
        pytest.skip(str(e))
    except Exception as e:
        pytest.skip(f"Could not prepare seamless-m4t-v2-large weights: {e}")


def legacy_mesh_device_param() -> tuple[int, int] | int:
    """Env-aware mesh shape for module PCC tests (P150→1×1, BH-QB→1×4, else auto)."""
    mesh_env = os.environ.get("MESH_DEVICE")
    if mesh_env in {"P150": (1, 1), "BH-QB": (1, 4)}:
        return {"P150": (1, 1), "BH-QB": (1, 4)}[mesh_env]
    if "TT_MESH_WIDTH" in os.environ:
        return int(os.environ["TT_MESH_WIDTH"])
    try:
        return (1, 4) if ttnn.get_num_devices() >= 4 else (1, 1)
    except Exception:
        return (1, 1)


def legacy_device_params() -> dict[str, Any]:
    mesh_param = legacy_mesh_device_param()
    params: dict[str, Any] = {"l1_small_size": 32768, "num_command_queues": 2}
    if mesh_param not in ((1, 1), 1):
        params["fabric_config"] = ttnn.FabricConfig.FABRIC_1D
    return params
