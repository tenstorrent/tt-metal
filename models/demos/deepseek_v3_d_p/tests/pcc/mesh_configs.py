# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Shared mesh configuration parameters for deepseek_v3_d_p tests.

Platform-specific 4-value config groups (mesh_device, device_params, num_links, topology):
  P300_MESH_CONFIGS  – 2-chip topologies
  QB_MESH_CONFIGS    – 4-chip topologies
  LB_MESH_CONFIGS    – 8-chip topologies
  GLX_MESH_CONFIGS   – 32-chip topologies
  ALL_MESH_CONFIGS   – union of P300 + QB + LB (no GLX)
  TP_QB_MESH_CONFIGS – 4-chip TP-only topologies (no fabric_router_config)
  TP_LM_HEAD_MESH_CONFIGS – LM head specific topologies

4-value single-device config (mesh_device, device_params, num_links, topology):
  SINGLE_DEVICE_CONFIG   – single-chip

2-value device config groups (mesh_device, device_params):
  P300_DEVICE_CONFIGS    – 2-chip
  QB_DEVICE_CONFIGS      – 4-chip
  TP_QB_DEVICE_CONFIGS   – 4-chip TP-only
  LB_DEVICE_CONFIGS      – 8-chip
"""

import pytest

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config, get_max_payload_size
from models.demos.deepseek_v3_d_p.tt.moe.tt_prefill_transformer import TT_PREFILL_TRANSFORMER_L1_SMALL

# Short aliases
L1_SMALL = TT_PREFILL_TRANSFORMER_L1_SMALL
FABRIC_1D = ttnn.FabricConfig.FABRIC_1D
FABRIC_1D_RING = ttnn.FabricConfig.FABRIC_1D_RING
DISABLED = ttnn.FabricConfig.DISABLED
LINEAR = ttnn.Topology.Linear
RING = ttnn.Topology.Ring


def _mesh_param(shape, fabric, payload, nlinks, topo, topo_marker, test_id, **device_kwargs):
    """Build a single pytest.param for the mesh_device parametrize axis (4-value)."""
    device_params = {"fabric_config": fabric, **device_kwargs}
    if payload is not None:
        device_params["fabric_router_config"] = create_fabric_router_config(max_payload_size=payload)
    return pytest.param(
        shape,
        device_params,
        nlinks,
        topo,
        marks=pytest.mark.requires_mesh_topology(mesh_shape=shape, topology=topo_marker),
        id=test_id,
    )


def _device_param(shape, fabric, topo_marker, test_id, **device_kwargs):
    """Build a pytest.param for tests that parametrize (mesh_device, device_params) only."""
    device_params = {"fabric_config": fabric, **device_kwargs}
    if isinstance(shape, int):
        return pytest.param(shape, device_params, id=test_id)
    return pytest.param(
        shape,
        device_params,
        marks=pytest.mark.requires_mesh_topology(mesh_shape=shape, topology=topo_marker),
        id=test_id,
    )


def select(configs, *ids):
    """Filter a config list to entries matching the given test IDs."""
    id_set = set(ids)
    result = [c for c in configs if c.id in id_set]
    missing = id_set - {c.id for c in result}
    assert not missing, f"Unknown test IDs: {missing}"
    return result


# ==============================================================================
# 4-value configs (mesh_device, device_params, num_links, topology)
# Used by dispatch/combine and EP tests that need fabric_router_config
# ==============================================================================

# -- 2-chip (P300) ---------------------------------------------------------
P300_MESH_CONFIGS = [
    _mesh_param(
        (2, 1),
        FABRIC_1D,
        get_max_payload_size(),
        1,
        LINEAR,
        "linear",
        "linear-2-1link",
        l1_small_size=L1_SMALL,
    ),
    _mesh_param(
        (2, 1),
        FABRIC_1D,
        get_max_payload_size(),
        2,
        LINEAR,
        "linear",
        "linear-2-2link",
        l1_small_size=L1_SMALL,
    ),
]

# -- 4-chip (QB) ------------------------------------------------------------
QB_MESH_CONFIGS = [
    _mesh_param(
        (4, 1),
        FABRIC_1D,
        get_max_payload_size(),
        1,
        LINEAR,
        "linear",
        "linear-4-1link",
        l1_small_size=L1_SMALL,
    ),
    _mesh_param(
        (4, 1),
        FABRIC_1D,
        get_max_payload_size(),
        2,
        LINEAR,
        "linear",
        "linear-4-2link",
        l1_small_size=L1_SMALL,
    ),
    _mesh_param(
        (4, 1),
        FABRIC_1D_RING,
        get_max_payload_size(),
        1,
        RING,
        "ring",
        "ring-4-1link",
        l1_small_size=L1_SMALL,
    ),
    _mesh_param(
        (4, 1),
        FABRIC_1D_RING,
        get_max_payload_size(),
        2,
        RING,
        "ring",
        "ring-4-2link",
        l1_small_size=L1_SMALL,
    ),
    _mesh_param(
        (2, 2),
        FABRIC_1D,
        get_max_payload_size(),
        1,
        LINEAR,
        "mesh-2x2",
        "mesh-2x2",
        l1_small_size=L1_SMALL,
    ),
]

# -- 8-chip (LB) ------------------------------------------------------------
LB_MESH_CONFIGS = [
    _mesh_param(
        (8, 1),
        FABRIC_1D,
        get_max_payload_size(),
        1,
        LINEAR,
        "linear",
        "linear-8-1link",
        l1_small_size=L1_SMALL,
    ),
    _mesh_param(
        (8, 1),
        FABRIC_1D,
        get_max_payload_size(),
        2,
        LINEAR,
        "linear",
        "linear-8-2link",
        l1_small_size=L1_SMALL,
    ),
    _mesh_param(
        (8, 1),
        FABRIC_1D_RING,
        get_max_payload_size(),
        1,
        RING,
        "ring",
        "ring-8-1link",
        l1_small_size=L1_SMALL,
    ),
    _mesh_param(
        (8, 1),
        FABRIC_1D_RING,
        get_max_payload_size(),
        2,
        RING,
        "ring",
        "ring-8-2link",
        l1_small_size=L1_SMALL,
    ),
    _mesh_param(
        (4, 2),
        FABRIC_1D,
        get_max_payload_size(),
        1,
        LINEAR,
        "mesh-4x2",
        "mesh-4x2",
        l1_small_size=L1_SMALL,
    ),
    _mesh_param(
        (2, 4),
        FABRIC_1D,
        get_max_payload_size(),
        1,
        LINEAR,
        "mesh-4x2",
        "mesh-2x4",
        l1_small_size=L1_SMALL,
    ),
]

# -- 32-chip (Galaxy) -------------------------------------------------------
GLX_MESH_CONFIGS = [
    _mesh_param(
        (8, 4),
        FABRIC_1D,
        get_max_payload_size(),
        1,
        LINEAR,
        "mesh-8x4",
        "mesh-8x4",
        l1_small_size=L1_SMALL,
    ),
]

# Combined (backward-compat): P300 + QB + LB (no Galaxy)
ALL_MESH_CONFIGS = P300_MESH_CONFIGS + QB_MESH_CONFIGS + LB_MESH_CONFIGS

# -- TP 4-value configs -------------------------------------------------------
TP_QB_MESH_CONFIGS = [
    _mesh_param(
        (1, 4),
        FABRIC_1D,
        get_max_payload_size(),
        1,
        LINEAR,
        "linear",
        "linear-4",
        l1_small_size=L1_SMALL,
    ),
    _mesh_param(
        (1, 4),
        FABRIC_1D_RING,
        get_max_payload_size(),
        1,
        RING,
        "ring",
        "ring-4",
        l1_small_size=L1_SMALL,
    ),
]

TP_LM_HEAD_MESH_CONFIGS = [
    _mesh_param(
        (1, 4),
        FABRIC_1D_RING,
        get_max_payload_size(),
        1,
        RING,
        "ring",
        "1x4-ring",
        l1_small_size=L1_SMALL,
    ),
    _mesh_param(
        (2, 2),
        FABRIC_1D_RING,
        get_max_payload_size(),
        1,
        RING,
        "ring",
        "2x2-ring",
        l1_small_size=L1_SMALL,
    ),
    _mesh_param(
        (2, 4),
        FABRIC_1D,
        get_max_payload_size(),
        1,
        LINEAR,
        "linear",
        "2x4-linear",
        l1_small_size=L1_SMALL,
    ),
]

# ==============================================================================
# 2-value configs (mesh_device, device_params)
# Used by tests that don't parametrize num_links / topology
# ==============================================================================

SINGLE_DEVICE_CONFIG = [
    _mesh_param(
        (1, 1),
        DISABLED,
        None,
        1,
        LINEAR,
        "linear",
        "single-chip",
        l1_small_size=L1_SMALL,
    ),
]

P300_DEVICE_CONFIGS = [
    _device_param((1, 2), FABRIC_1D, "linear", "linear-1x2", l1_small_size=L1_SMALL),
]

QB_DEVICE_CONFIGS = [
    _device_param((1, 1), FABRIC_1D, "linear", "single", l1_small_size=L1_SMALL),
    _device_param((1, 4), FABRIC_1D, "linear", "linear-1x4", l1_small_size=L1_SMALL),
    _device_param((2, 2), FABRIC_1D, "linear", "linear-2x2", l1_small_size=L1_SMALL),
    _device_param((4, 1), FABRIC_1D, "linear", "linear-4", l1_small_size=L1_SMALL),
]

TP_QB_DEVICE_CONFIGS = [
    _device_param((1, 4), FABRIC_1D, "linear", "linear-4", l1_small_size=L1_SMALL),
    _device_param((1, 4), FABRIC_1D_RING, "ring", "ring-4", l1_small_size=L1_SMALL),
]

LB_DEVICE_CONFIGS = [
    _device_param((8, 1), FABRIC_1D, "linear", "linear-8", l1_small_size=L1_SMALL),
    _device_param((4, 2), FABRIC_1D, "mesh-4x2", "mesh-4x2", l1_small_size=L1_SMALL),
    _device_param((2, 4), FABRIC_1D, "mesh-4x2", "mesh-2x4", l1_small_size=L1_SMALL),
]
