# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Galaxy prefill test helpers."""

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
                    "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                    "fabric_router_config": _fabric_router_config(),
                    **dict(device_params_extra or {}),
                },
                id=f"{shape[0]}x{shape[1]}",
            )
        )
    return pytest.mark.parametrize("mesh_device, device_params", params, indirect=True)
