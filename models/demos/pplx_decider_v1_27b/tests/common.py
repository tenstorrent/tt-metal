# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
from pathlib import Path

import pytest


def load_example_state(example, ckpt_dir):
    if "state_file" in example:
        return (Path(ckpt_dir) / example["state_file"]).read_text()
    return example["state"]


def ids_sha256(input_ids):
    return hashlib.sha256(json.dumps(input_ids).encode()).hexdigest()


def _parse_shape(text):
    rows, cols = (int(x) for x in text.lower().split("x"))
    return rows, cols


def parametrize_mesh_traced():
    """Like qwen36 parametrize_mesh_tp, with a trace region for capture_prefill_trace.

    PPLX_PARENT_MESH unset (QB2): the MESH_DEVICE mesh with FABRIC_1D.
    PPLX_PARENT_MESH="4x8" / "8x4" (Galaxy): that full mesh with FABRIC_1D_RING; the model runs on a submesh, see
    model_mesh.
    """
    # Local imports keep CPU-only tests free of the device stack.
    import ttnn
    from models.demos.blackhole.qwen36.tests.test_factory import _resolve_mesh_shape
    from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE
    from models.demos.pplx_decider_v1_27b.tt.model import TRACE_REGION_SIZE

    parent = os.environ.get("PPLX_PARENT_MESH")
    # A bare 1x8 / 1x4 mesh on a Galaxy fails the fabric router handshake, so the full mesh is opened (fabric is
    # initialised on every chip). qwen36 CCLs use Topology.Ring on Galaxy, hence FABRIC_1D_RING.
    shape = _parse_shape(parent) if parent else _resolve_mesh_shape()
    device_params = {
        "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING if parent else ttnn.FabricConfig.FABRIC_1D,
        "l1_small_size": GDN_CONV1D_L1_SMALL_SIZE,
        "trace_region_size": TRACE_REGION_SIZE,
    }

    def decorator(fn):
        fn = pytest.mark.parametrize("device_params", [device_params], indirect=True)(fn)
        return pytest.mark.parametrize(
            "mesh_device", [pytest.param(shape, id=f"{shape[0]}x{shape[1]}")], indirect=True
        )(fn)

    return decorator


def model_mesh(mesh_device):
    """The mesh the model runs on: the root mesh, or the first PPLX_SUBMESH ("1x8" default) of PPLX_PARENT_MESH."""
    if not os.environ.get("PPLX_PARENT_MESH"):
        return mesh_device
    from loguru import logger

    import ttnn

    sub_shape = _parse_shape(os.environ.get("PPLX_SUBMESH", "1x8"))
    if sub_shape == tuple(mesh_device.shape):
        logger.info(f"PPLX_SUBMESH {sub_shape[0]}x{sub_shape[1]} is the parent mesh: running on it, no submesh")
        return mesh_device
    sub = mesh_device.create_submeshes(ttnn.MeshShape(*sub_shape))[0]
    logger.info(f"parent mesh {tuple(mesh_device.shape)} device ids: {list(mesh_device.get_device_ids())}")
    logger.info(f"submesh shape: {tuple(sub.shape)} num_devices={sub.get_num_devices()}")
    logger.info(f"submesh physical device ids: {list(sub.get_device_ids())}")
    try:
        fabric_nodes = [sub.get_fabric_node_id(ttnn.MeshCoordinate(0, c)) for c in range(sub.shape[1])]
        logger.info(f"submesh row-0 fabric node ids: {[str(n) for n in fabric_nodes]}")
    except Exception as e:  # informational only
        logger.warning(f"submesh fabric node ids unavailable: {e!r}")
    return sub
