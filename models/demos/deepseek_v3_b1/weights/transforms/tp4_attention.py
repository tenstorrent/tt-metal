# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TP4 decoder attention weight layout: packer, preprocessing, and fusion-group specs.

This module owns the **TP4-sharded attention weight layout** for the
DeepSeek decoder on a 4x2 mesh.  It is expressed as *two independent
per-core fusion artefacts* (not one multi-buffer group):

* :func:`build_merged_main_tp4_spec` — the main merged attention buffer:
  ``o_proj`` (TP4-shuffled) + RMSNorm gammas + ``q_a`` + ``q_b`` +
  ``kv_a``, packed across the ~115-core union of their per-tensor core
  sets (``per_core=True``).
* :func:`build_gate_mm_tp4_spec` — ``gate_mm`` (the MoE router gate) on
  its narrow 8-core slab (``per_core=True``).  Technically not an
  attention weight, but it shares the same TP4 sharding/mesh layout as
  the attention block and runs right after it in the decoder dataflow,
  so its packer/layout naturally belongs next to the attention ones.
  Kept out of the main spec so its 8-core allocation doesn't wait on a
  lockstep reservation across the 115-core main buffer.

Two specs (not one multi-buffer group) let the cache keep a single-blob
schema, invalidate the two artefacts independently, and still benefit
from per-core allocation end to end.  Callers typically invoke both
factories and call :func:`~weights.cache.cache.TensorCache.get_or_create`
twice — once per spec — then merge the resulting view dicts.

The packer :func:`pack_o_proj_weights_tp4_shuffled` used to live on
``O_PROJ_GATE_MM_RMSNORM_GAMMA_SingleDeviceOverlapSpec`` and pulled
``shuffle_q_a`` out of ``QAB_KVA_PROJ_SingleDeviceOverlapSpec``.  That
cross-spec reference made the class hierarchy awkward; this module is
its new home.
"""

from __future__ import annotations

from dataclasses import replace

import torch

import ttnn
from models.demos.deepseek_v3_b1.weights.cache.types import FusionGroupSpec, RegionSpec, Shard2dMeshMapper
from models.demos.deepseek_v3_b1.weights.overlap.spec import OverlappedTensorSpec
from models.demos.deepseek_v3_b1.weights.specs.overlap_configs import (
    O_PROJ_GATE_MM_RMSNORM_GAMMA_SINGLE_DEVICE_OVERLAP_SPEC,
    QAB_KVA_PROJ_SINGLE_DEVICE_OVERLAP_SPEC,
)


def pack_o_proj_weights_tp4_shuffled(o_proj_weights: torch.Tensor) -> torch.Tensor:
    """Pack full-mesh o_proj weights for ``tp_dim=(1, 0)`` plus ``shuffle_q_a`` layout.

    Input ``o_proj_weights`` has global shape ``(16384, 7168)``.  Each mesh device's
    ``(8192, 1792)`` slice is packed with ``shuffle_q_a`` to ``(4096, 3584)`` and
    written to the sub-rectangle that :func:`~weights.overlap.packing.overlap_tensors`
    reads for that device when ``raw_tensor_shape=(8192, 14336)`` and ``tp_dim=(1, 0)``.
    """
    if tuple(o_proj_weights.shape) != (16384, 7168):
        raise ValueError(
            f"pack_o_proj_weights_tp4_shuffled expects shape (16384, 7168), got {tuple(o_proj_weights.shape)}"
        )
    shuffle = QAB_KVA_PROJ_SINGLE_DEVICE_OVERLAP_SPEC.shuffle_q_a
    out = torch.empty((8192, 14336), dtype=o_proj_weights.dtype, device=o_proj_weights.device)
    for mesh_row in range(4):
        for mesh_col in range(2):
            block = o_proj_weights[
                8192 * mesh_col : 8192 * (mesh_col + 1),
                1792 * mesh_row : 1792 * (mesh_row + 1),
            ]
            packed = shuffle(block)
            out[
                4096 * mesh_col : 4096 * (mesh_col + 1),
                3584 * mesh_row : 3584 * (mesh_row + 1),
            ] = packed
    return out


def _named(spec: OverlappedTensorSpec, name: str) -> OverlappedTensorSpec:
    return replace(spec, name=name, logical_tensor_shape=spec.logical_tensor_shape or spec.raw_tensor_shape)


def _region(*lane: tuple[str, OverlappedTensorSpec]) -> RegionSpec:
    named = tuple(_named(spec, name) for name, spec in lane)
    return RegionSpec(core_range_set=named[0].core_range_set, subtensors=named)


def build_merged_main_tp4_spec(
    *,
    name: str = "o_proj_tp4_norms_q_ab_kv_a",
    o_proj_dtype: ttnn.DataType = ttnn.DataType.BFLOAT4_B,
    q_ab_dtype: ttnn.DataType = ttnn.DataType.BFLOAT4_B,
    kv_a_dtype: ttnn.DataType = ttnn.DataType.BFLOAT4_B,
    transform_version: int = 0,
) -> FusionGroupSpec:
    """Main TP4-merged decoder spec: o_proj + norms + q_ab + kv_a, per-core.

    Defaults match upstream's BFP4 attention flip (#41931): every MLA
    matmul weight (``q_a_proj``, ``q_b_proj``, ``kv_a_proj``, ``o_proj``,
    ``kv_b1/kv_b2_proj``) lives in the same dtype + tile so the shared
    circular buffers in :mod:`~fused_ops.attention_block.op` agree.
    Dtypes are still plumbed per-lane in case a future experiment wants to
    revert individual lanes to BFP8.
    """
    o_cfg = O_PROJ_GATE_MM_RMSNORM_GAMMA_SINGLE_DEVICE_OVERLAP_SPEC
    q_cfg = QAB_KVA_PROJ_SINGLE_DEVICE_OVERLAP_SPEC

    o_proj_tp4 = replace(
        o_cfg.o_proj,
        raw_tensor_shape=(8192, 14336),
        dtype=o_proj_dtype,
        tp_dim=(1, 0),
    )
    q_a = replace(q_cfg.q_a_shard_spec, dtype=q_ab_dtype)
    q_b = replace(q_cfg.q_b_shard_spec, dtype=q_ab_dtype)
    kv_a = replace(q_cfg.kv_a_shard_spec, dtype=kv_a_dtype)

    return FusionGroupSpec(
        name=name,
        regions=(
            _region(("o_proj", o_proj_tp4)),
            _region(("attn_norm", o_cfg.attn_norm), ("q_norm", o_cfg.q_norm), ("ffn_norm", o_cfg.ffn_norm)),
            _region(("kv_norm", o_cfg.kv_norm)),
            _region(("q_a_proj", q_a), ("q_b_proj", q_b)),
            _region(("kv_a_proj", kv_a)),
        ),
        sharding_strategy=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        # Per-tensor tp is handled by overlap_tensors independently; this
        # mesh_mapper_config only participates in fingerprinting.
        mesh_mapper_config=Shard2dMeshMapper(dims=(1, 0)),
        transform_version=transform_version,
        per_core=True,
    )


def build_gate_mm_tp4_spec(
    *,
    name: str = "gate_mm_tp4",
    gate_mm_dtype: ttnn.DataType | None = None,
    transform_version: int = 0,
) -> FusionGroupSpec:
    """Standalone per-core fusion spec for ``gate_mm``.

    Shipping ``gate_mm`` as its own artefact means the narrow 8-core slab
    can be allocated independently of the much wider 115-core main buffer.
    """
    o_cfg = O_PROJ_GATE_MM_RMSNORM_GAMMA_SINGLE_DEVICE_OVERLAP_SPEC
    gate_mm = o_cfg.gate_mm if gate_mm_dtype is None else replace(o_cfg.gate_mm, dtype=gate_mm_dtype)

    return FusionGroupSpec(
        name=name,
        regions=(_region(("gate_mm", gate_mm)),),
        sharding_strategy=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        mesh_mapper_config=Shard2dMeshMapper(dims=(1, 0)),
        transform_version=transform_version,
        per_core=True,
    )
