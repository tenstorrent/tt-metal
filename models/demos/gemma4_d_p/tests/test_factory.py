# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Galaxy prefill test helpers."""

import pytest
import torch

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


class TestFactory:
    """Reference and device RoPE tables for Galaxy tests."""

    @staticmethod
    def create_hf_rope(hf_text_config, seq_len, layer_idx):
        """Create HF RoPE position embeddings (cos, sin) for torch reference."""
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

        rope = Gemma4TextRotaryEmbedding(hf_text_config)
        x_dummy = torch.randn(1, seq_len, hf_text_config.hidden_size)
        pos_ids = torch.arange(seq_len).unsqueeze(0)
        layer_type = hf_text_config.layer_types[layer_idx]
        cos, sin = rope(x_dummy, pos_ids, layer_type=layer_type)
        return cos, sin

    @staticmethod
    def create_tt_rope_cache(device, hf_text_config, max_seq_len, layer_idx, mesh_config=None):
        """Create HF-format cos/sin cache on TT device using HF Gemma4TextRotaryEmbedding.

        Returns (cos_cache, sin_cache) each [1, 1, max_seq_len, head_dim] on device.
        Matches exactly what HF produces (including identity padding for partial RoPE).

        ``mesh_config`` with a context-parallel degree > 1 shards the sequence
        dimension across the CP axis instead of replicating, so each rank gets the
        cos/sin rows for the positions it actually owns. RoPE is the one place that
        needs absolute positions, and because it is a per-position lookup the
        sharding is all that is required. Omitting mesh_config replicates, which is
        what every non-CP caller wants.
        """
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

        from models.demos.gemma4_d_p.tt.ccl import cp_degree

        rope = Gemma4TextRotaryEmbedding(hf_text_config)
        x_dummy = torch.randn(1, max_seq_len, hf_text_config.hidden_size)
        pos_ids = torch.arange(max_seq_len).unsqueeze(0)
        layer_type = hf_text_config.layer_types[layer_idx]
        cos, sin = rope(x_dummy, pos_ids, layer_type=layer_type)
        # cos, sin: [1, max_seq_len, head_dim] -> [1, 1, max_seq_len, head_dim]
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

        is_mesh = hasattr(device, "shape")
        if is_mesh and mesh_config is not None and cp_degree(mesh_config) > 1:
            shard_dims = (-2, None) if mesh_config.sp_axis == 0 else (None, -2)
            mapper = ttnn.ShardTensor2dMesh(device, device.shape, dims=shard_dims)
        elif is_mesh:
            mapper = ttnn.ReplicateTensorToMesh(device)
        else:
            mapper = None

        cos_tt = ttnn.from_torch(
            cos,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=mapper,
        )
        sin_tt = ttnn.from_torch(
            sin,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=mapper,
        )
        return cos_tt, sin_tt
