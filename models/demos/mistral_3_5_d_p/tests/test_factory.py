# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared test scaffolding for ``mistral_3_5_d_p``. Pattern: ``minimax_m3/tests/test_factory.py``.

The single mesh+fabric parametrization helper every unit test uses, plus the small builders that
keep "same random weights on both sides" from being re-implemented per test.

**The bring-up mesh is the spec's target mesh and nothing smaller.** ``spec.mesh_shape`` is (4, 8) —
TP=8 on the cols, SP=4 on the rows — and every module test runs there, so sharding and collectives
are exercised from the first one. A ``(1, 1)`` case is offered ONLY for the handful of blocks whose
math is genuinely mesh-independent (norm, MLP at TP=1) as a fast local-iteration aid; nothing in the
Testing tables depends on it, because a smaller submesh cannot bring up the fabric on a Galaxy
(carving chips out leaves their ethernet partners outside the submesh with no router kernel).
"""

from __future__ import annotations

import os

import pytest
import torch

import ttnn

from ..spec import SPEC

TARGET_MESH = SPEC.mesh_shape  # (4, 8): SP=4 rows, TP=8 cols


def linear_fabric() -> bool:
    """True -> ``FABRIC_1D`` + ``ttnn.Topology.Linear`` (a plain-grid Galaxy, this pod's wiring).
    False -> ``FABRIC_1D_RING`` + ``ttnn.Topology.Ring`` (a torus-wired Galaxy). See conftest.py."""
    return os.getenv("MISTRAL_LINEAR_FABRIC", "1").strip().lower() in ("1", "true", "yes", "on")


def topology() -> "ttnn.Topology":
    return ttnn.Topology.Linear if linear_fabric() else ttnn.Topology.Ring


def parametrize_mesh(mesh_shapes=None, *, trace_region_size: int = 0):
    """Paired ``(mesh_device, device_params)`` parametrization, one id per shape (``4x8``, ``1x1``).

    Defaults to the spec's target mesh alone. Shapes that do not fit the host are filtered out, and
    if nothing fits the case is emitted as a skip so a smaller box reports "skipped", not "passed".
    """
    num_devices = ttnn.get_num_devices()
    shapes = [TARGET_MESH] if mesh_shapes is None else list(mesh_shapes)
    shapes = [s for s in shapes if s[0] * s[1] <= num_devices]
    if not shapes:
        return pytest.mark.parametrize(
            "mesh_device, device_params",
            [
                pytest.param(
                    TARGET_MESH,
                    {"fabric_config": None},
                    id=f"{TARGET_MESH[0]}x{TARGET_MESH[1]}",
                    marks=pytest.mark.skip(
                        reason=f"needs the spec's target mesh {TARGET_MESH} ({TARGET_MESH[0] * TARGET_MESH[1]} "
                        f"devices); this host has {num_devices}"
                    ),
                )
            ],
            indirect=True,
        )

    multidev_fabric = ttnn.FabricConfig.FABRIC_1D if linear_fabric() else ttnn.FabricConfig.FABRIC_1D_RING
    params = [
        pytest.param(
            shape,
            {
                "fabric_config": (None if shape == (1, 1) else multidev_fabric),
                **({"trace_region_size": trace_region_size} if trace_region_size else {}),
            },
            id=f"{shape[0]}x{shape[1]}",
        )
        for shape in shapes
    ]
    return pytest.mark.parametrize("mesh_device, device_params", params, indirect=True)


def build_mesh_and_ccl(mesh_device):
    """``(MeshConfig, CCLManager)`` for this mesh, at the spec's TP and this pod's topology."""
    from ..tt.ccl import CCLManager
    from ..tt.config import MeshConfig
    from ..utils.general_utils import get_default_num_links

    rows, cols = tuple(mesh_device.shape)
    mesh_config = MeshConfig((rows, cols), tp=cols)
    ccl = CCLManager(mesh_device, num_links=get_default_num_links(mesh_device), topology=topology())
    return mesh_config, ccl


def sp_tp_shard_mapper(mesh_device, *, seq_dim=None, head_dim=None, feature_dim=None):
    """A ``ShardTensor2dMesh`` over (SP rows, TP cols), naming the tensor dims by role.

    Exactly one of ``head_dim`` / ``feature_dim`` may be given (both live on the TP axis); ``None``
    for an axis replicates along it. Keeps the ``dims=(a, b)`` tuples out of the test bodies, where
    a swapped pair is invisible.
    """
    assert head_dim is None or feature_dim is None, "head_dim and feature_dim both shard the TP axis"
    dims = [None, None]
    dims[SPEC.sp_axis] = seq_dim
    dims[SPEC.tp_axis] = head_dim if head_dim is not None else feature_dim
    return ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=tuple(dims))


def to_device(tensor, mesh_device, *, mapper=None, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    """``ttnn.from_torch`` with this package's DRAM/tile defaults; replicated when no mapper given."""
    return ttnn.from_torch(
        tensor,
        device=mesh_device,
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper or ttnn.ReplicateTensorToMesh(mesh_device),
    )


def from_device_replicated(tt_tensor, shape=None):
    """Read back a tensor that is REPLICATED across the mesh: device 0's shard is the whole answer."""
    out = ttnn.to_torch(ttnn.get_device_tensors(tt_tensor)[0]).float()
    return out.reshape(shape) if shape is not None else out


def from_device_sp_sharded(tt_tensor, mesh_device, *, seq_dim=2):
    """Reassemble an SP-sharded (TP-replicated) tensor: concat the SP rows' shards, column 0 only."""
    rows, cols = tuple(mesh_device.shape)
    shards = ttnn.get_device_tensors(tt_tensor)
    return torch.cat([ttnn.to_torch(shards[r * cols]).float() for r in range(rows)], dim=seq_dim)


def from_device_tp_sharded(tt_tensor, mesh_device, *, feature_dim=-1):
    """Reassemble a TP-sharded (SP-replicated) tensor: concat row 0's column shards."""
    _, cols = tuple(mesh_device.shape)
    shards = ttnn.get_device_tensors(tt_tensor)
    return torch.cat([ttnn.to_torch(shards[c]).float() for c in range(cols)], dim=feature_dim)


def meta_swizzle(state_dict: dict, head_dim: int) -> dict:
    """Run q/k projections through the production HF->Meta swizzle.

    The TT attention consumes Meta-interleaved q/k because the on-device rope is
    ``rotary_embedding_indexed``; the torch reference is HF-convention. Tests must apply the SAME
    production helper (``convert_hf_qkv_to_meta_format``) that ``tt/model_config.py`` applies, or
    they measure the swizzle instead of the attention.
    """
    from models.tt_transformers.tt.load_checkpoints import convert_hf_qkv_to_meta_format

    return convert_hf_qkv_to_meta_format(state_dict, head_dim)
