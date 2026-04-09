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


def fuse_q_ab_kv_a(
    q_a_proj_weights: torch.Tensor,
    q_b_proj_weights: torch.Tensor,
    kv_a_proj_weights: torch.Tensor,
    device,
    *,
    dtype: ttnn.DataType = ttnn.bfloat8_b,
    move_to_device: bool = True,
) -> dict[str, OverlappedTensor]:
    """Fuse q_a, q_b, and kv_a projection weights into one overlapped buffer."""
    cfg = QAB_KVA_PROJ_SINGLE_DEVICE_OVERLAP_SPEC
    mesh_shape = (device.shape[0], device.shape[1]) if device.get_num_devices() > 1 else (1, 1)
    preprocessed = preprocess_q_ab_kv_a(q_a_proj_weights, q_b_proj_weights, kv_a_proj_weights, mesh_shape)
    q_a_packed = preprocessed["q_a_proj"]
    q_b_pre = preprocessed["q_b_proj"]
    kv_reordered = preprocessed["kv_a_proj"]

    return overlap_tensors(
        [
            [
                OverlapEntry(
                    "q_a_proj",
                    q_a_packed,
                    replace(cfg.q_a_shard_spec, raw_tensor_shape=tuple(q_a_packed.shape), dtype=dtype),
                ),
                OverlapEntry(
                    "q_b_proj",
                    q_b_pre,
                    replace(cfg.q_b_shard_spec, raw_tensor_shape=tuple(q_b_pre.shape), dtype=dtype),
                ),
            ],
            [
                OverlapEntry(
                    "kv_a_proj",
                    kv_reordered,
                    replace(cfg.kv_a_shard_spec, raw_tensor_shape=tuple(kv_reordered.shape), dtype=dtype),
                ),
            ],
        ],
        device=device,
        move_to_device=move_to_device,
    )


def fuse_o_proj_gate_mm_norms(
    o_proj_weights: torch.Tensor,
    gate_mm_weights: torch.Tensor,
    attn_norm: torch.Tensor,
    q_norm: torch.Tensor,
    kv_norm: torch.Tensor,
    ffn_norm: torch.Tensor,
    device,
    *,
    o_proj_dtype: ttnn.DataType = ttnn.bfloat8_b,
    move_to_device: bool = True,
) -> dict[str, OverlappedTensor]:
    """Fuse o_proj, gate_mm, and RMSNorm weights into one overlapped buffer."""
    cfg = O_PROJ_GATE_MM_RMSNORM_GAMMA_SINGLE_DEVICE_OVERLAP_SPEC

    return overlap_tensors(
        [
            [
                OverlapEntry(
                    "o_proj",
                    o_proj_weights,
                    replace(cfg.o_proj, raw_tensor_shape=tuple(o_proj_weights.shape), dtype=o_proj_dtype),
                )
            ],
            [OverlapEntry("gate_mm", gate_mm_weights, cfg.gate_mm)],
            [
                OverlapEntry("attn_norm", attn_norm, cfg.attn_norm),
                OverlapEntry("q_norm", q_norm, cfg.q_norm),
                OverlapEntry("ffn_norm", ffn_norm, cfg.ffn_norm),
            ],
            [OverlapEntry("kv_norm", kv_norm, cfg.kv_norm)],
        ],
        device=device,
        move_to_device=move_to_device,
    )


def fuse_kv_b12(
    kv_b1_proj_weights: torch.Tensor,
    kv_b2_proj_weights: torch.Tensor,
    device,
    *,
    dtype: ttnn.DataType = ttnn.bfloat8_b,
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
            [
                OverlapEntry(
                    "kv_b1_proj",
                    kv_b1,
                    replace(cfg.kv_b1_shard_spec, raw_tensor_shape=tuple(kv_b1.shape), dtype=dtype),
                ),
            ],
            [
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
        ],
        device=device,
        move_to_device=move_to_device,
    )
