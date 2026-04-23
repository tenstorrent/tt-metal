# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Attention weight preprocessing transforms and fusion helpers.

Shared preprocessing functions are the single source of truth for the
shuffle/TP-concat/mesh-reshape orchestration.  Both ``prepare.py``
and the ``fuse_*`` helpers delegate to these.
"""

from __future__ import annotations

from dataclasses import replace

import torch

import ttnn
from models.demos.deepseek_v3_b1.weights.overlap.packing import OverlapEntry, OverlappedTensor, overlap_tensors
from models.demos.deepseek_v3_b1.weights.specs.overlap_configs import (
    KVB12_PROJ_SINGLE_DEVICE_OVERLAP_SPEC,
    O_PROJ_GATE_MM_RMSNORM_GAMMA_SINGLE_DEVICE_OVERLAP_SPEC,
    QAB_KVA_PROJ_SINGLE_DEVICE_OVERLAP_SPEC,
    QAB_KVA_PROJ_SingleDeviceOverlapSpec,
)
from models.demos.deepseek_v3_b1.weights.transforms.moe import _tp_factors


def preprocess_q_ab_kv_a(
    q_a: torch.Tensor,
    q_b: torch.Tensor,
    kv_a: torch.Tensor,
    mesh_shape: tuple[int, int],
) -> dict[str, torch.Tensor]:
    """Shuffle and TP-concat q_a/q_b/kv_a into fusion-ready tensors.

    Args:
        q_a: Transposed q_a_proj weight ``(K, N)``.
        q_b: Deinterleaved q_b_proj weight ``(K, N)`` (full or TP1-trimmed).
        kv_a: Transposed kv_a_proj weight ``(K, N)``.
        mesh_shape: ``(rows, cols)`` of the device mesh, ``(1, 1)`` for single device.

    Returns:
        Dict with keys ``q_a_proj``, ``q_b_proj``, ``kv_a_proj`` — shuffled,
        TP-concatenated torch tensors ready for tilization.
    """
    cfg = QAB_KVA_PROJ_SINGLE_DEVICE_OVERLAP_SPEC
    q_a_packed = cfg.shuffle_q_a(q_a)
    q_b_tp = cfg.q_b_shard_spec.tp(mesh_shape)
    q_b_slices = [cfg.shuffle_q_b(cfg.get_q_b_slice(q_b, i, mesh_shape)) for i in range(q_b_tp)]
    q_b_pre = torch.cat(q_b_slices, dim=1) if q_b_tp > 1 else q_b_slices[0]
    kv_reordered = cfg.shuffle_kv_a(kv_a)
    return {"q_a_proj": q_a_packed, "q_b_proj": q_b_pre, "kv_a_proj": kv_reordered}


def preprocess_kv_b12(
    kv_b1: torch.Tensor,
    kv_b2: torch.Tensor,
    mla_tp: int,
) -> dict[str, torch.Tensor]:
    """Shuffle and TP-concat kv_b1/kv_b2 into fusion-ready tensors."""
    cfg = KVB12_PROJ_SINGLE_DEVICE_OVERLAP_SPEC
    per_device_b2_w = cfg.kv_b2_proj_shape[1]
    b2_shuffled = [cfg.shuffle_kv_b2(kv_b2[:, i * per_device_b2_w : (i + 1) * per_device_b2_w]) for i in range(mla_tp)]
    kv_b2_pre = torch.cat(b2_shuffled, dim=0) if mla_tp > 1 else b2_shuffled[0]
    return {"kv_b1_proj": kv_b1, "kv_b2_proj": kv_b2_pre}


def _make_q_ab_kv_a_overlap_entries(
    cfg: QAB_KVA_PROJ_SingleDeviceOverlapSpec,
    mesh_shape: tuple[int, int],
    q_a_proj_weights: torch.Tensor,
    q_b_proj_weights: torch.Tensor,
    kv_a_proj_weights: torch.Tensor,
    q_ab_dtype: ttnn.DataType,
    kv_a_dtype: ttnn.DataType,
) -> list[OverlapEntry]:
    """Validate MLA proj weights and build three :class:`OverlapEntry` items (no I/O)."""
    q_b_tp = cfg.q_b_shard_spec.tp(mesh_shape)

    assert (
        q_a_proj_weights.shape == cfg.q_a_proj_shape
    ), f"q_a_proj_weights must be {cfg.q_a_proj_shape}, got {tuple(q_a_proj_weights.shape)}"
    expected_q_b_shape = (cfg.q_b_proj_shape[0], cfg.q_b_proj_shape[1] * q_b_tp)
    assert (
        tuple(q_b_proj_weights.shape) == expected_q_b_shape
    ), f"q_b_proj_weights must be {expected_q_b_shape}, got {tuple(q_b_proj_weights.shape)}"
    assert (
        kv_a_proj_weights.shape == cfg.kv_a_proj_shape
    ), f"kv_a_proj_weights must be {cfg.kv_a_proj_shape}, got {tuple(kv_a_proj_weights.shape)}"

    q_a_packed = cfg.shuffle_q_a(q_a_proj_weights)
    kv_reordered = cfg.shuffle_kv_a(kv_a_proj_weights)

    q_b_shuffled_slices = [
        cfg.shuffle_q_b(cfg.get_q_b_slice(q_b_proj_weights, tp_idx, mesh_shape)) for tp_idx in range(q_b_tp)
    ]
    q_b_preprocessed = torch.cat(q_b_shuffled_slices, dim=1) if q_b_tp > 1 else q_b_shuffled_slices[0]

    return [
        OverlapEntry(
            "q_a_proj",
            q_a_packed,
            replace(cfg.q_a_shard_spec, raw_tensor_shape=tuple(q_a_packed.shape), dtype=q_ab_dtype),
        ),
        OverlapEntry(
            "q_b_proj",
            q_b_preprocessed,
            replace(cfg.q_b_shard_spec, raw_tensor_shape=tuple(q_b_preprocessed.shape), dtype=q_ab_dtype),
        ),
        OverlapEntry(
            "kv_a_proj",
            kv_reordered,
            replace(cfg.kv_a_shard_spec, raw_tensor_shape=tuple(kv_reordered.shape), dtype=kv_a_dtype),
        ),
    ]


def fuse_q_ab_kv_a(
    q_a_proj_weights: torch.Tensor,
    q_b_proj_weights: torch.Tensor,
    kv_a_proj_weights: torch.Tensor,
    device,
    *,
    dtype: ttnn.DataType = ttnn.bfloat4_b,
    move_to_device: bool = True,
) -> dict[str, OverlappedTensor]:
    """Fuse q_a, q_b, and kv_a projection weights into one overlapped buffer."""
    cfg = QAB_KVA_PROJ_SINGLE_DEVICE_OVERLAP_SPEC
    mesh_shape = (device.shape[0], device.shape[1]) if device.get_num_devices() > 1 else (1, 1)
    entries = _make_q_ab_kv_a_overlap_entries(
        cfg, mesh_shape, q_a_proj_weights, q_b_proj_weights, kv_a_proj_weights, dtype, dtype
    )
    return overlap_tensors(entries, device=device, move_to_device=move_to_device)


def fuse_o_proj_gate_mm_norms(
    o_proj_weights: torch.Tensor,
    gate_mm_weights: torch.Tensor,
    attn_norm: torch.Tensor,
    q_norm: torch.Tensor,
    kv_norm: torch.Tensor,
    ffn_norm: torch.Tensor,
    device,
    *,
    o_proj_dtype: ttnn.DataType = ttnn.bfloat4_b,
    move_to_device: bool = True,
) -> dict[str, OverlappedTensor]:
    """Fuse o_proj, gate_mm, and RMSNorm weights into one overlapped buffer."""
    cfg = O_PROJ_GATE_MM_RMSNORM_GAMMA_SINGLE_DEVICE_OVERLAP_SPEC

    return overlap_tensors(
        [
            OverlapEntry(
                "o_proj",
                o_proj_weights,
                replace(cfg.o_proj, raw_tensor_shape=tuple(o_proj_weights.shape), dtype=o_proj_dtype),
            ),
            OverlapEntry("gate_mm", gate_mm_weights, cfg.gate_mm),
            OverlapEntry("attn_norm", attn_norm, cfg.attn_norm),
            OverlapEntry("q_norm", q_norm, cfg.q_norm),
            OverlapEntry("ffn_norm", ffn_norm, cfg.ffn_norm),
            OverlapEntry("kv_norm", kv_norm, cfg.kv_norm),
        ],
        device=device,
        move_to_device=move_to_device,
    )


def fuse_o_proj_tp4_shuffled_gate_mm_norms_q_ab_kv_a(
    o_proj_weights: torch.Tensor,
    gate_mm_weights: torch.Tensor,
    attn_norm: torch.Tensor,
    q_norm: torch.Tensor,
    kv_norm: torch.Tensor,
    ffn_norm: torch.Tensor,
    q_a_proj_weights: torch.Tensor,
    q_b_proj_weights: torch.Tensor,
    kv_a_proj_weights: torch.Tensor,
    device,
    *,
    o_proj_dtype: ttnn.DataType = ttnn.bfloat4_b,
    q_ab_dtype: ttnn.DataType = ttnn.bfloat4_b,
    kv_a_dtype: ttnn.DataType = ttnn.bfloat4_b,
    move_to_device: bool = True,
) -> dict[str, OverlappedTensor]:
    """Fuse TP4 ``shuffle_q_a`` o_proj, norms, and q_a / q_b / kv_a into one per-core L1 buffer.

    ``gate_mm`` is excluded from the overlap and returned as a standalone
    ``OverlappedTensor`` backed by its own per-core allocated tensor, avoiding
    lockstep L1 reservation on the ~115 cores used by the fused buffer.

    Requires a **4×2** mesh.  ``o_proj_weights`` shape ``(16384, 7168)``.
    Must be the first per-core allocation on the device.
    """
    o_cfg = O_PROJ_GATE_MM_RMSNORM_GAMMA_SINGLE_DEVICE_OVERLAP_SPEC
    q_cfg = QAB_KVA_PROJ_SINGLE_DEVICE_OVERLAP_SPEC
    mesh_shape = (device.shape[0], device.shape[1])
    if mesh_shape != (4, 2):
        raise ValueError(
            "fuse_o_proj_tp4_shuffled_gate_mm_norms_q_ab_kv_a requires a 4x2 mesh; "
            f"got {mesh_shape[0]}x{mesh_shape[1]}"
        )
    if tuple(o_proj_weights.shape) != (16384, 7168):
        raise ValueError(
            "o_proj_weights must have shape (16384, 7168) for TP4+shuffle layout; " f"got {tuple(o_proj_weights.shape)}"
        )

    device_grid = device.compute_with_storage_grid_size()
    q_ab_bb = q_cfg.q_a_shard_spec.core_range_set.bounding_box()
    kv_bb = q_cfg.kv_a_shard_spec.core_range_set.bounding_box()
    required_rows = max(q_ab_bb.end.y, kv_bb.end.y) + 1
    required_cols = max(q_ab_bb.end.x, kv_bb.end.x) + 1
    assert device_grid.y >= required_rows, f"Device grid needs at least {required_rows} rows, got {device_grid.y}"
    assert device_grid.x >= required_cols, f"Device grid needs at least {required_cols} cols, got {device_grid.x}"

    o_packed = o_cfg.pack_o_proj_weights_tp4_shuffled(o_proj_weights)
    o_spec = replace(
        o_cfg.o_proj,
        raw_tensor_shape=tuple(o_packed.shape),
        dtype=o_proj_dtype,
        tp_dim=(1, 0),
    )

    q_entries = _make_q_ab_kv_a_overlap_entries(
        q_cfg, mesh_shape, q_a_proj_weights, q_b_proj_weights, kv_a_proj_weights, q_ab_dtype, kv_a_dtype
    )

    result = overlap_tensors(
        [
            OverlapEntry("o_proj", o_packed, o_spec),
            OverlapEntry("attn_norm", attn_norm, o_cfg.attn_norm),
            OverlapEntry("q_norm", q_norm, o_cfg.q_norm),
            OverlapEntry("ffn_norm", ffn_norm, o_cfg.ffn_norm),
            OverlapEntry("kv_norm", kv_norm, o_cfg.kv_norm),
            *q_entries,
        ],
        device=device,
        move_to_device=move_to_device,
        per_core=True,
    )

    gate_mm_result = overlap_tensors(
        [OverlapEntry("gate_mm", gate_mm_weights, o_cfg.gate_mm)],
        device=device,
        move_to_device=move_to_device,
        per_core=True,
    )
    result["gate_mm"] = gate_mm_result["gate_mm"]

    return result


def fuse_kv_b12(
    kv_b1_proj_weights: torch.Tensor,
    kv_b2_proj_weights: torch.Tensor,
    device,
    *,
    dtype: ttnn.DataType = ttnn.bfloat4_b,
    move_to_device: bool = True,
) -> dict[str, OverlappedTensor]:
    """Fuse kv_b1 and kv_b2 projection weights into one overlapped buffer."""
    cfg = KVB12_PROJ_SINGLE_DEVICE_OVERLAP_SPEC
    mla_tp, _ = _tp_factors(device)
    preprocessed = preprocess_kv_b12(kv_b1_proj_weights, kv_b2_proj_weights, mla_tp)
    kv_b1 = preprocessed["kv_b1_proj"]
    kv_b2_pre = preprocessed["kv_b2_proj"]

    return overlap_tensors(
        [
            OverlapEntry(
                "kv_b1_proj",
                kv_b1,
                replace(cfg.kv_b1_shard_spec, raw_tensor_shape=tuple(kv_b1.shape), dtype=dtype),
            ),
            OverlapEntry(
                "kv_b2_proj",
                kv_b2_pre,
                replace(
                    cfg.kv_b2_shard_spec,
                    raw_tensor_shape=tuple(kv_b2_pre.shape),
                    dtype=dtype,
                    logical_tensor_shape=cfg.kv_b2_proj_shape,
                ),
            ),
        ],
        device=device,
        move_to_device=move_to_device,
    )
