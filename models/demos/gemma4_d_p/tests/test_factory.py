# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Galaxy prefill test helpers."""

import os

import pytest

import ttnn


def find_layer_idx(hf_text_config, layer_type):
    """Return the first layer index whose type matches.

    layer_type: "sliding_attention" or "full_attention". Raises if no match —
    callers should pytest.skip when a model lacks a layer of the requested
    type (e.g. early-layer-only configs that have no global layer in the
    first N layers).
    """
    for i, lt in enumerate(hf_text_config.layer_types):
        if lt == layer_type:
            return i
    raise ValueError(f"No layer of type {layer_type} in layer_types={hf_text_config.layer_types}")


def _fabric_router_config():
    """Fabric router tuning applied alongside the fabric config."""

    config = ttnn.FabricRouterConfig()
    config.max_packet_payload_size_bytes = 8192
    return config


def parametrize_mesh_with_fabric(mesh_shapes=None, device_params_extra=None):
    """Parametrize full-Galaxy layouts without probing hardware during collection."""
    from ..config import GALAXY_MESH_SHAPES, validate_galaxy_mesh

    params = []
    for shape in mesh_shapes or GALAXY_MESH_SHAPES:
        validate_galaxy_mesh(shape)
        params.append(
            pytest.param(
                shape,
                {
                    # GEMMA4_FABRIC=FABRIC_1D for a Galaxy whose ring (wrap-around) links do not train (pair it with
                    # TT_MESH_GRAPH_DESC_PATH=.../single_bh_galaxy_mesh_graph_descriptor.textproto).
                    "fabric_config": getattr(ttnn.FabricConfig, os.environ.get("GEMMA4_FABRIC", "FABRIC_1D_RING")),
                    "fabric_router_config": _fabric_router_config(),
                    **dict(device_params_extra or {}),
                },
                id=f"{shape[0]}x{shape[1]}",
            )
        )
    return pytest.mark.parametrize("mesh_device, device_params", params, indirect=True)


_HALVES = []  # keeps every half's submesh alive for the test's lifetime


def sp4_half(mesh_device):
    """SP4 x 2 study (GEMMA4_TEST_SP4=1): one CP4 x TP4 half of the 8x4 Galaxy instead of the whole mesh.

    The Galaxy splits into two 4x4 submeshes that each run a full model copy; a single-model test uses the first.
    Needs a TT_CACHE_PATH holding a tensor_cache_bf16_mesh4x4 cache. Otherwise returns mesh_device unchanged."""
    if os.environ.get("GEMMA4_TEST_SP4") != "1":
        return mesh_device
    os.environ["GEMMA4_ALLOW_SUBMESH"] = "1"
    halves = mesh_device.create_submeshes(ttnn.MeshShape(4, 4))
    _HALVES.append(halves)
    return halves[0]
