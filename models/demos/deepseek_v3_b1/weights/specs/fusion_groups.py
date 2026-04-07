# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""FusionGroupSpec constants for DeepSeek V3 weight overlapping.

Each constant (e.g. ``Q_AB_KV_A_SPEC``) is a :class:`FusionGroupSpec`
built from the :mod:`~weights.specs.overlap_configs` singletons.  These
are the declarative layouts consumed by the tensor cache and
:func:`create_overlapped_tensor`.
"""

from __future__ import annotations

from dataclasses import replace

import ttnn
from models.demos.deepseek_v3_b1.weights.cache.types import (
    FusionGroupSpec,
    MeshMapperConfig,
    RegionSpec,
    ReplicateMeshMapper,
    Shard2dMeshMapper,
)
from models.demos.deepseek_v3_b1.weights.overlap.spec import OverlappedTensorSpec
from models.demos.deepseek_v3_b1.weights.specs.overlap_configs import (
    GATE_UP_PROJ_SINGLE_DEVICE_OVERLAP_SPEC,
    KVB12_PROJ_SINGLE_DEVICE_OVERLAP_SPEC,
    O_PROJ_GATE_MM_RMSNORM_GAMMA_SINGLE_DEVICE_OVERLAP_SPEC,
    QAB_KVA_PROJ_SINGLE_DEVICE_OVERLAP_SPEC,
)


def _infer_mesh_mapper(lanes: list[list[tuple[str, OverlappedTensorSpec]]]) -> MeshMapperConfig:
    """Derive the mesh mapper config from ``tp_dim`` values across all specs."""
    dims: list[int | None] = [None, None]
    for lane in lanes:
        for spec_name, spec in lane:
            for mesh_dim in range(2):
                td = spec.tp_dim[mesh_dim]
                if td is not None:
                    if dims[mesh_dim] is not None and dims[mesh_dim] != td:
                        raise ValueError(
                            f"Conflicting tp_dim[{mesh_dim}] in {spec_name!r}: "
                            f"previously saw {dims[mesh_dim]}, now {td}"
                        )
                    dims[mesh_dim] = td
    if dims[0] is None and dims[1] is None:
        return ReplicateMeshMapper()
    return Shard2dMeshMapper(dims=(dims[0], dims[1]))


def _build_fusion_group_spec(
    name: str,
    lanes: list[list[tuple[str, OverlappedTensorSpec]]],
    sharding_strategy: ttnn.TensorMemoryLayout,
    mesh_mapper_config: MeshMapperConfig | None = None,
) -> FusionGroupSpec:
    """Derive a :class:`FusionGroupSpec` from named :class:`OverlappedTensorSpec` fields."""
    if mesh_mapper_config is None:
        mesh_mapper_config = _infer_mesh_mapper(lanes)
    regions: list[RegionSpec] = []
    for lane in lanes:
        subtensors = tuple(
            replace(spec, name=n, logical_tensor_shape=spec.logical_tensor_shape or spec.raw_tensor_shape)
            for n, spec in lane
        )
        regions.append(
            RegionSpec(
                core_range_set=lane[0][1].core_range_set,
                subtensors=subtensors,
            )
        )
    return FusionGroupSpec(
        name=name,
        regions=tuple(regions),
        sharding_strategy=sharding_strategy,
        mesh_mapper_config=mesh_mapper_config,
    )


_QAB_SPEC = QAB_KVA_PROJ_SINGLE_DEVICE_OVERLAP_SPEC
Q_AB_KV_A_SPEC = _build_fusion_group_spec(
    "q_ab_kv_a",
    [
        [("q_a_proj", _QAB_SPEC.q_a_shard_spec), ("q_b_proj", _QAB_SPEC.q_b_shard_spec)],
        [("kv_a_proj", _QAB_SPEC.kv_a_shard_spec)],
    ],
    sharding_strategy=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
)

_OV_SPEC = O_PROJ_GATE_MM_RMSNORM_GAMMA_SINGLE_DEVICE_OVERLAP_SPEC
O_PROJ_GATE_MM_NORMS_SPEC = _build_fusion_group_spec(
    "o_proj_gate_mm_norms",
    [
        [("o_proj", _OV_SPEC.o_proj)],
        [("gate_mm", _OV_SPEC.gate_mm)],
        [("attn_norm", _OV_SPEC.attn_norm), ("q_norm", _OV_SPEC.q_norm), ("ffn_norm", _OV_SPEC.ffn_norm)],
        [("kv_norm", _OV_SPEC.kv_norm)],
    ],
    sharding_strategy=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    mesh_mapper_config=Shard2dMeshMapper(dims=(None, 1)),
)

_KVB_SPEC = KVB12_PROJ_SINGLE_DEVICE_OVERLAP_SPEC
KV_B12_SPEC = _build_fusion_group_spec(
    "kv_b12",
    [
        [("kv_b1_proj", _KVB_SPEC.kv_b1_shard_spec)],
        [("kv_b2_proj", _KVB_SPEC.kv_b2_shard_spec)],
    ],
    sharding_strategy=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
)

_GU_SPEC = GATE_UP_PROJ_SINGLE_DEVICE_OVERLAP_SPEC
GATE_UP_SPEC = _build_fusion_group_spec(
    "gate_up",
    [
        [("shared_gate_proj", _GU_SPEC.gate_shard_spec)],
        [("shared_up_proj", _GU_SPEC.up_shard_spec)],
    ],
    sharding_strategy=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
)
