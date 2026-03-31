# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Prepare DeepSeek V3 fused (blitz decode) weights from a state dict.

Takes full HuggingFace state dict tensors (full logical shapes for the target
mesh), applies key mapping, transpose, kv_b split, and fuses/shards onto
the device mesh.

When a ``CacheConfig`` is provided, each fusion group / standalone tensor /
routed-expert list is routed through ``TensorCache`` for content-addressed
on-disk caching.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.blitz_overlap_tensors import OverlappedShardSpec, OverlappedTensor, overlap_tensors
from models.demos.deepseek_v3_b1.overlap_specs import (
    DOWN_PROJ_SINGLE_DEVICE_SPEC,
    GATE_UP_PROJ_SINGLE_DEVICE_OVERLAP_SPEC,
    KVB12_PROJ_SINGLE_DEVICE_OVERLAP_SPEC,
    O_PROJ_GATE_MM_RMSNORM_GAMMA_SINGLE_DEVICE_OVERLAP_SPEC,
    QAB_KVA_PROJ_SINGLE_DEVICE_OVERLAP_SPEC,
)
from models.demos.deepseek_v3_b1.tensor_cache import CacheConfig, Fingerprint

# Bump when any weight transform logic changes to invalidate cached artifacts.
_TRANSFORM_VERSION = 1

# MoE sender core: hardcoded grid (13, 10) so cache layout is consistent across slow/fast dispatch.
# Sender core = (grid.x - 1, grid.y - 1) = (12, 9); must match test_moe_mlp create_runtime_tensors.
MOE_SENDER_GRID_SIZE = (13, 10)
_GATE_BIAS_TILE = ttnn.Tile([16, 16])


# ---------------------------------------------------------------------------
# Device topology helpers
# ---------------------------------------------------------------------------


def _compute_tp(device) -> tuple[int, int]:
    """Derive (mla_tp, moe_tp) from the device mesh shape."""
    num_devices = device.get_num_devices()
    if num_devices == 1:
        return 1, 1
    mesh_shape = (device.shape[0], device.shape[1])
    assert mesh_shape == (
        4,
        2,
    ), f"Only single-device or 4x2 mesh supported, got {mesh_shape[0]}x{mesh_shape[1]}"
    return 2, 8


def _mesh_shape(device) -> tuple[int, int]:
    if device.get_num_devices() == 1:
        return (1, 1)
    return (device.shape[0], device.shape[1])


# ---------------------------------------------------------------------------
# Cache fingerprint helpers
# ---------------------------------------------------------------------------


def _make_fp(
    cache_config: CacheConfig,
    mesh_shape: tuple[int, int],
    group_name: str,
    layer_idx: int,
    spec_fingerprints: tuple[str, ...],
) -> Fingerprint:
    return Fingerprint(
        schema_version=1,
        hf_model_id=cache_config.hf_model_id,
        hf_revision=cache_config.hf_revision,
        transform_version=_TRANSFORM_VERSION,
        mesh_shape=mesh_shape,
        group_name=group_name,
        layer_idx=layer_idx,
        spec_fingerprints=spec_fingerprints,
    )


def _qab_kva_spec_fps() -> tuple[str, ...]:
    cfg = QAB_KVA_PROJ_SINGLE_DEVICE_OVERLAP_SPEC
    return (cfg.q_a_shard_spec.fingerprint(), cfg.q_b_shard_spec.fingerprint(), cfg.kv_a_shard_spec.fingerprint())


def _o_proj_norms_spec_fps() -> tuple[str, ...]:
    cfg = O_PROJ_GATE_MM_RMSNORM_GAMMA_SINGLE_DEVICE_OVERLAP_SPEC
    return (
        cfg.o_proj.fingerprint(),
        cfg.gate_mm.fingerprint(),
        cfg.attn_norm.fingerprint(),
        cfg.q_norm.fingerprint(),
        cfg.kv_norm.fingerprint(),
        cfg.ffn_norm.fingerprint(),
    )


def _kv_b12_spec_fps() -> tuple[str, ...]:
    """Fingerprint from the config-level core range sets + known shapes/dtype.

    KVB12 specs are constructed inline (shape depends on mla_tp) but the
    core grids and sharding strategy are fixed.  Combined with mesh_shape
    in the Fingerprint, this uniquely identifies the layout.
    """
    cfg = KVB12_PROJ_SINGLE_DEVICE_OVERLAP_SPEC
    kv_b1_fp = OverlappedShardSpec(
        core_range_set=cfg.kv_b1_core_range_set,
        raw_tensor_shape=cfg.kv_b1_proj_shape,
        dtype=ttnn.DataType.BFLOAT8_B,
        sharding=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ).fingerprint()
    kv_b2_fp = OverlappedShardSpec(
        core_range_set=cfg.kv_b2_core_range_set,
        raw_tensor_shape=cfg.kv_b2_proj_shape,
        dtype=ttnn.DataType.BFLOAT8_B,
        sharding=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ).fingerprint()
    return (kv_b1_fp, kv_b2_fp)


def _gate_up_spec_fps() -> tuple[str, ...]:
    """Fingerprint from the config-level shapes and core grids."""
    cfg = GATE_UP_PROJ_SINGLE_DEVICE_OVERLAP_SPEC
    gate_fp = OverlappedShardSpec(
        core_range_set=cfg.gate_core_range_set,
        raw_tensor_shape=cfg.stacked_shape,
        dtype=ttnn.DataType.BFLOAT4_B,
        sharding=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ).fingerprint()
    up_fp = OverlappedShardSpec(
        core_range_set=cfg.up_core_range_set,
        raw_tensor_shape=cfg.stacked_shape,
        dtype=ttnn.DataType.BFLOAT4_B,
        sharding=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ).fingerprint()
    return (gate_fp, up_fp)


# Cache artifact types returned by layer_fingerprints
CACHE_TYPE_OVERLAPPED = "overlapped"
CACHE_TYPE_TENSOR = "tensor"
CACHE_TYPE_TENSOR_LIST = "tensor_list"


def layer_fingerprints(
    cache_config: CacheConfig,
    mesh_shape: tuple[int, int],
    layer_idx: int,
    is_moe: bool,
) -> dict[str, tuple[Fingerprint, str]]:
    """Build all fingerprints for a single decoder layer.

    Returns ``{group_name: (fingerprint, cache_type)}`` where *cache_type* is
    one of ``CACHE_TYPE_OVERLAPPED``, ``CACHE_TYPE_TENSOR``, or
    ``CACHE_TYPE_TENSOR_LIST``.
    """
    fp = lambda name, specs: _make_fp(cache_config, mesh_shape, name, layer_idx, specs)
    result: dict[str, tuple[Fingerprint, str]] = {
        "q_ab_kv_a": (fp("q_ab_kv_a", _qab_kva_spec_fps()), CACHE_TYPE_OVERLAPPED),
        "o_proj_gate_mm_norms": (fp("o_proj_gate_mm_norms", _o_proj_norms_spec_fps()), CACHE_TYPE_OVERLAPPED),
        "kv_b12": (fp("kv_b12", _kv_b12_spec_fps()), CACHE_TYPE_OVERLAPPED),
        "gate_up": (fp("gate_up", _gate_up_spec_fps()), CACHE_TYPE_OVERLAPPED),
        "shared_down_proj": (fp("shared_down_proj", ()), CACHE_TYPE_TENSOR),
    }
    if is_moe:
        result["gate_bias"] = (fp("gate_bias", ()), CACHE_TYPE_TENSOR)
        result["routed_gate_proj"] = (fp("routed_gate_proj", ()), CACHE_TYPE_TENSOR_LIST)
        result["routed_up_proj"] = (fp("routed_up_proj", ()), CACHE_TYPE_TENSOR_LIST)
        result["routed_down_proj"] = (fp("routed_down_proj", ()), CACHE_TYPE_TENSOR_LIST)
    else:
        result["routed_gate_proj"] = (fp("routed_gate_proj", ()), CACHE_TYPE_TENSOR)
        result["routed_up_proj"] = (fp("routed_up_proj", ()), CACHE_TYPE_TENSOR)
        result["routed_down_proj"] = (fp("routed_down_proj", ()), CACHE_TYPE_TENSOR)
    return result


def embedding_fingerprint(
    cache_config: CacheConfig,
    mesh_shape: tuple[int, int],
) -> Fingerprint:
    """Build the fingerprint for the embedding tensor."""
    return _make_fp(cache_config, mesh_shape, "embedding", 0, ())


def lm_head_fingerprints(
    cache_config: CacheConfig,
    mesh_shape: tuple[int, int],
) -> dict[str, tuple[Fingerprint, str]]:
    """Build fingerprints for LM head + final norm."""
    fp = lambda name: _make_fp(cache_config, mesh_shape, name, 0, ())
    return {
        "lm_head": (fp("lm_head"), CACHE_TYPE_TENSOR),
        "final_norm": (fp("final_norm"), CACHE_TYPE_TENSOR),
    }


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AttentionWeights:
    """Attention fusion groups: q_ab_kv_a + kv_b12 + o_proj_gate_mm_norms."""

    q_a_proj: OverlappedTensor
    q_b_proj: OverlappedTensor
    kv_a_proj: OverlappedTensor
    o_proj: OverlappedTensor
    gate_mm: OverlappedTensor | None  # None for dense layers
    attn_norm: OverlappedTensor
    q_norm: OverlappedTensor
    kv_norm: OverlappedTensor
    ffn_norm: OverlappedTensor
    kv_b1_proj: OverlappedTensor
    kv_b2_proj: OverlappedTensor
    gate_bias: ttnn.Tensor | None  # e_score_correction_bias for MoE only


@dataclass
class SharedExpertWeights:
    """Shared expert gate_up fusion group + standalone shared_down_proj."""

    shared_gate_proj: OverlappedTensor
    shared_up_proj: OverlappedTensor
    shared_down_proj: ttnn.Tensor


@dataclass
class DenseRoutedExpertWeights:
    """Routed expert weights for dense layers (single tensor per proj)."""

    routed_gate_proj: ttnn.Tensor
    routed_up_proj: ttnn.Tensor
    routed_down_proj: ttnn.Tensor


# Must match MoeRoutedExpertOp.setup_dram_matmul(..., num_subblocks_k=4) for stride checks.
_MOE_DRAM_MATMUL_NUM_SUBBLOCKS_K = 4


def _moe_routed_expert_stride_bytes(weights_tensor: ttnn.Tensor) -> int:
    """Packed DRAM size of one routed expert tensor (bytes), per DRAMStreamingMatmul indexing."""
    shard_spec = weights_tensor.memory_config().shard_spec
    if shard_spec is None:
        raise ValueError("MoE routed expert weights must be sharded (WIDTH_SHARDED DRAM).")
    weights_shard_shape = shard_spec.shape
    K = weights_shard_shape[0]
    tile = weights_tensor.tile
    th, tw = tile.tile_shape[0], tile.tile_shape[1]
    per_core_n = weights_shard_shape[1] // tw
    Kt = K // th
    if Kt % _MOE_DRAM_MATMUL_NUM_SUBBLOCKS_K != 0:
        raise AssertionError(
            f"Kt ({Kt}) must be divisible by num_subblocks_k ({_MOE_DRAM_MATMUL_NUM_SUBBLOCKS_K}) "
            "for MoE DRAM matmul stride check."
        )
    weights_tile_size = tile.get_tile_size(weights_tensor.dtype)
    return Kt * per_core_n * weights_tile_size


def _assert_moe_routed_expert_list_contiguous(tensors: list[ttnn.Tensor], name: str) -> None:
    """Experts must be contiguous in DRAM; see MoERoutedExpertWeights.validate_contiguous_dram."""
    if len(tensors) < 2:
        return
    if not ttnn.is_tensor_storage_on_device(tensors[0]):
        return
    stride = _moe_routed_expert_stride_bytes(tensors[0])
    base = tensors[0].buffer_address()
    for i, t in enumerate(tensors):
        expected = base + i * stride
        actual = t.buffer_address()
        if actual != expected:
            raise AssertionError(
                f"{name}[{i}] DRAM layout not contiguous for DRAMStreamingMatmul: "
                f"expected buffer_address {expected}, got {actual} (stride {stride} bytes per expert). "
                "Allocate all experts of one projection in one batch before the next projection."
            )


@dataclass
class MoERoutedExpertWeights:
    """Routed expert weights for MoE layers (list of tensors, one per expert).

    When on device, each of ``routed_gate_proj``, ``routed_up_proj``, and ``routed_down_proj`` must
    be allocated contiguously in DRAM (see :meth:`validate_contiguous_dram`).
    """

    routed_gate_proj: list[ttnn.Tensor]
    routed_up_proj: list[ttnn.Tensor]
    routed_down_proj: list[ttnn.Tensor]

    def validate_contiguous_dram(self) -> None:
        """Assert experts are contiguous in DRAM for each projection (DRAMStreamingMatmul base+stride)."""
        _assert_moe_routed_expert_list_contiguous(self.routed_gate_proj, "routed_gate_proj")
        _assert_moe_routed_expert_list_contiguous(self.routed_up_proj, "routed_up_proj")
        _assert_moe_routed_expert_list_contiguous(self.routed_down_proj, "routed_down_proj")


class ModelWeights:
    """Generic named weight container.

    Holds two parallel dicts keyed by the same weight names:
    one for prepared TTNN device tensors, one for raw torch tensors.

    Access patterns:
        weights["q_a_proj"]              -> TTNN tensor (OverlappedTensor, ttnn.Tensor, or list)
        weights.torch_tensor("q_a_proj") -> torch.Tensor (raw, for golden computation)
        weights.q_a_proj                 -> same as weights["q_a_proj"] (attribute access)
    """

    def __init__(
        self,
        ttnn_weights: dict[str, Any] | None = None,
        torch_weights: dict[str, Any] | None = None,
    ):
        self._ttnn = ttnn_weights or {}
        self._torch = torch_weights or {}

    def __getitem__(self, name: str):
        return self._ttnn[name]

    def __contains__(self, name: str) -> bool:
        return name in self._ttnn

    def get(self, name: str, default=None):
        return self._ttnn.get(name, default)

    def keys(self):
        return self._ttnn.keys()

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._ttnn[name]
        except KeyError:
            raise AttributeError(f"No weight named '{name}'")

    def ttnn_tensor(self, name: str):
        return self._ttnn[name]

    def torch_tensor(self, name: str):
        return self._torch[name]

    def has_torch(self, name: str) -> bool:
        return name in self._torch

    @property
    def torch_keys(self):
        return self._torch.keys()


@dataclass
class DeepSeekV3EmbeddingLayerWeights:
    """Weights for the embedding layer."""

    embedding: ttnn.Tensor


@dataclass
class DeepSeekV3LMHeadWeights:
    """Weights for the LM head and final RMSNorm."""

    lm_head: ttnn.Tensor
    final_norm: ttnn.Tensor  # model.norm.weight, (1, 7168)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NUM_HEADS = 64
NUM_ROUTED_EXPERTS = 256
_QK_NOPE_HEAD_DIM = 128
_QK_ROPE_HEAD_DIM = 64
_V_HEAD_DIM = 128
_KV_LORA_RANK = 512
_KV_B_PROJ_HEAD_DIM = _QK_NOPE_HEAD_DIM + _V_HEAD_DIM  # 256
_Q_HEAD_DIM = _QK_NOPE_HEAD_DIM + _QK_ROPE_HEAD_DIM  # 192

# Per-TP attention tensor dimensions
_MLA_TP1_Q_B_WIDTH = 12288
_MLA_TP1_O_PROJ_HEIGHT = 8192
_MLA_TP1_KV_B1_HEIGHT = 8192
_MLA_TP1_KV_B2_WIDTH = 8192

# Per-TP shared expert dimensions
_MOE_TP1_SHARED_GATE_UP_N = 256
_MOE_TP1_SHARED_DOWN_K = 256


# ---------------------------------------------------------------------------
# Tensor transformation functions (formerly BlitzDecodeWeights methods)
# ---------------------------------------------------------------------------


def _fuse_q_ab_kv_a(
    device,
    q_a_proj_weights: torch.Tensor,
    q_b_proj_weights: torch.Tensor,
    kv_a_proj_weights: torch.Tensor,
    *,
    dtype: ttnn.DataType = ttnn.bfloat8_b,
    move_to_device: bool = True,
) -> dict[str, OverlappedTensor]:
    """Fuse q_a_proj, q_b_proj, and kv_a_proj via ``overlap_tensors``."""
    cfg = QAB_KVA_PROJ_SINGLE_DEVICE_OVERLAP_SPEC
    mesh = _mesh_shape(device)
    q_b_tp = cfg.q_b_shard_spec.tp(mesh)

    device_grid = device.compute_with_storage_grid_size()
    q_ab_bb = cfg.q_a_shard_spec.core_range_set.bounding_box()
    kv_bb = cfg.kv_a_shard_spec.core_range_set.bounding_box()
    required_rows = max(q_ab_bb.end.y, kv_bb.end.y) + 1
    required_cols = max(q_ab_bb.end.x, kv_bb.end.x) + 1
    assert device_grid.y >= required_rows, f"Device grid needs at least {required_rows} rows, got {device_grid.y}"
    assert device_grid.x >= required_cols, f"Device grid needs at least {required_cols} cols, got {device_grid.x}"

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
        cfg.shuffle_q_b(cfg.get_q_b_slice(q_b_proj_weights, tp_idx, mesh)) for tp_idx in range(q_b_tp)
    ]
    q_b_preprocessed = torch.cat(q_b_shuffled_slices, dim=1) if q_b_tp > 1 else q_b_shuffled_slices[0]

    q_ab_cores = cfg.q_a_shard_spec.core_range_set
    kv_cores = cfg.kv_a_shard_spec.core_range_set

    return overlap_tensors(
        [
            [
                (
                    "q_a_proj",
                    q_a_packed,
                    OverlappedShardSpec(
                        core_range_set=q_ab_cores,
                        raw_tensor_shape=tuple(q_a_packed.shape),
                        dtype=dtype,
                    ),
                ),
                (
                    "q_b_proj",
                    q_b_preprocessed,
                    OverlappedShardSpec(
                        core_range_set=q_ab_cores,
                        raw_tensor_shape=tuple(q_b_preprocessed.shape),
                        dtype=dtype,
                        tp_dim=(None, 1),
                    ),
                ),
            ],
            [
                (
                    "kv_a_proj",
                    kv_reordered,
                    OverlappedShardSpec(
                        core_range_set=kv_cores,
                        raw_tensor_shape=tuple(kv_reordered.shape),
                        dtype=dtype,
                    ),
                ),
            ],
        ],
        device=device,
        move_to_device=move_to_device,
    )


def _fuse_o_proj_gate_mm_norms(
    device,
    o_proj_weights: torch.Tensor,
    gate_mm_weights: torch.Tensor,
    attn_norm: torch.Tensor,
    q_norm: torch.Tensor,
    kv_norm: torch.Tensor,
    ffn_norm: torch.Tensor,
    *,
    o_proj_dtype: ttnn.DataType = ttnn.bfloat8_b,
    move_to_device: bool = True,
) -> dict[str, OverlappedTensor]:
    """Fuse o_proj, gate_mm, and 4 RMSNorm gammas into one WIDTH_SHARDED tensor."""
    cfg = O_PROJ_GATE_MM_RMSNORM_GAMMA_SINGLE_DEVICE_OVERLAP_SPEC

    return overlap_tensors(
        [
            [
                (
                    "o_proj",
                    o_proj_weights,
                    replace(cfg.o_proj, raw_tensor_shape=tuple(o_proj_weights.shape), dtype=o_proj_dtype),
                )
            ],
            [("gate_mm", gate_mm_weights, cfg.gate_mm)],
            [
                ("attn_norm", attn_norm, cfg.attn_norm),
                ("q_norm", q_norm, cfg.q_norm),
                ("ffn_norm", ffn_norm, cfg.ffn_norm),
            ],
            [("kv_norm", kv_norm, cfg.kv_norm)],
        ],
        device=device,
        move_to_device=move_to_device,
    )


def _fuse_kv_b12(
    device,
    kv_b1_proj_weights: torch.Tensor,
    kv_b2_proj_weights: torch.Tensor,
    *,
    dtype: ttnn.DataType = ttnn.bfloat8_b,
    move_to_device: bool = True,
) -> dict[str, OverlappedTensor]:
    """Fuse kv_b1_proj and kv_b2_proj via ``overlap_tensors``."""
    cfg = KVB12_PROJ_SINGLE_DEVICE_OVERLAP_SPEC
    mla_tp, _ = _compute_tp(device)

    expected_b1_shape = (cfg.kv_b1_proj_shape[0] * mla_tp, cfg.kv_b1_proj_shape[1])
    assert (
        tuple(kv_b1_proj_weights.shape) == expected_b1_shape
    ), f"kv_b1 expected {expected_b1_shape}, got {tuple(kv_b1_proj_weights.shape)}"
    expected_b2_shape = (cfg.kv_b2_proj_shape[0], cfg.kv_b2_proj_shape[1] * mla_tp)
    assert (
        tuple(kv_b2_proj_weights.shape) == expected_b2_shape
    ), f"kv_b2 expected {expected_b2_shape}, got {tuple(kv_b2_proj_weights.shape)}"

    per_device_b2_w = cfg.kv_b2_proj_shape[1]
    b2_shuffled = []
    for tp_idx in range(mla_tp):
        b2_slice = kv_b2_proj_weights[:, tp_idx * per_device_b2_w : (tp_idx + 1) * per_device_b2_w]
        b2_shuffled.append(cfg.shuffle_kv_b2(b2_slice))
    kv_b2_preprocessed = torch.cat(b2_shuffled, dim=0) if mla_tp > 1 else b2_shuffled[0]

    return overlap_tensors(
        [
            [
                (
                    "kv_b1_proj",
                    kv_b1_proj_weights,
                    OverlappedShardSpec(
                        core_range_set=cfg.kv_b1_core_range_set,
                        raw_tensor_shape=tuple(kv_b1_proj_weights.shape),
                        dtype=dtype,
                        sharding=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                        tp_dim=(None, 0),
                    ),
                ),
            ],
            [
                (
                    "kv_b2_proj",
                    kv_b2_preprocessed,
                    OverlappedShardSpec(
                        core_range_set=cfg.kv_b2_core_range_set,
                        raw_tensor_shape=tuple(kv_b2_preprocessed.shape),
                        dtype=dtype,
                        sharding=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                        tp_dim=(None, 0),
                        logical_tensor_shape=cfg.kv_b2_proj_shape,
                    ),
                ),
            ],
        ],
        device=device,
        move_to_device=move_to_device,
    )


def _fuse_gate_up(
    device,
    gate_proj_weights: torch.Tensor,
    up_proj_weights: torch.Tensor,
    *,
    dtype: ttnn.DataType = ttnn.bfloat4_b,
    move_to_device: bool = True,
) -> dict[str, OverlappedTensor]:
    """Fuse gate and up projections into one HEIGHT_SHARDED tensor."""
    _, moe_tp = _compute_tp(device)
    cfg = GATE_UP_PROJ_SINGLE_DEVICE_OVERLAP_SPEC

    expected_gate_shape = (cfg.gate_proj_shape[0], cfg.gate_proj_shape[1] * moe_tp)
    assert (
        tuple(gate_proj_weights.shape) == expected_gate_shape
    ), f"gate_proj must be {expected_gate_shape}, got {tuple(gate_proj_weights.shape)}"
    expected_up_shape = (cfg.up_proj_shape[0], cfg.up_proj_shape[1] * moe_tp)
    assert (
        tuple(up_proj_weights.shape) == expected_up_shape
    ), f"up_proj must be {expected_up_shape}, got {tuple(up_proj_weights.shape)}"

    mesh_rows = device.shape[0]
    mesh_cols = device.shape[1]
    per_device_n = cfg.gate_proj_shape[1]
    stacked_h, stacked_w = cfg.stacked_shape

    gate_stacked_list = []
    up_stacked_list = []
    for tp_idx in range(moe_tp):
        gate_slice = gate_proj_weights[:, tp_idx * per_device_n : (tp_idx + 1) * per_device_n]
        up_slice = up_proj_weights[:, tp_idx * per_device_n : (tp_idx + 1) * per_device_n]
        gate_stacked_list.append(cfg.reshuffle_block_to_height_sharded(gate_slice, cfg.gate_core_range_set))
        up_stacked_list.append(cfg.reshuffle_block_to_height_sharded(up_slice, cfg.up_core_range_set))

    if moe_tp == 1:
        gate_preprocessed = gate_stacked_list[0]
        up_preprocessed = up_stacked_list[0]
    else:
        gate_preprocessed = (
            torch.stack(gate_stacked_list)
            .reshape(mesh_rows, mesh_cols, stacked_h, stacked_w)
            .permute(0, 2, 1, 3)
            .reshape(mesh_rows * stacked_h, mesh_cols * stacked_w)
            .contiguous()
        )
        up_preprocessed = (
            torch.stack(up_stacked_list)
            .reshape(mesh_rows, mesh_cols, stacked_h, stacked_w)
            .permute(0, 2, 1, 3)
            .reshape(mesh_rows * stacked_h, mesh_cols * stacked_w)
            .contiguous()
        )

    return overlap_tensors(
        [
            [
                (
                    "gate_proj",
                    gate_preprocessed,
                    OverlappedShardSpec(
                        core_range_set=cfg.gate_core_range_set,
                        raw_tensor_shape=tuple(gate_preprocessed.shape),
                        dtype=dtype,
                        sharding=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                        tp_dim=(0, 1),
                        logical_tensor_shape=cfg.gate_proj_shape,
                    ),
                ),
            ],
            [
                (
                    "up_proj",
                    up_preprocessed,
                    OverlappedShardSpec(
                        core_range_set=cfg.up_core_range_set,
                        raw_tensor_shape=tuple(up_preprocessed.shape),
                        dtype=dtype,
                        sharding=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                        tp_dim=(0, 1),
                        logical_tensor_shape=cfg.up_proj_shape,
                    ),
                ),
            ],
        ],
        device=device,
        move_to_device=move_to_device,
    )


def _create_shared_down_proj(
    device,
    down_proj_weights: torch.Tensor,
    *,
    dtype: ttnn.DataType = ttnn.bfloat4_b,
    move_to_device: bool = True,
) -> ttnn.Tensor:
    """Create the down projection as a standalone WIDTH_SHARDED tensor on 112 matmul cores."""
    _, moe_tp = _compute_tp(device)
    dp_spec = DOWN_PROJ_SINGLE_DEVICE_SPEC
    K_down_per_device = 256
    N_per_core = 64
    N_down = N_per_core * dp_spec.NUM_MATMUL_CORES  # 7168

    expected_down_shape = (K_down_per_device * moe_tp, N_down)
    assert (
        tuple(down_proj_weights.shape) == expected_down_shape
    ), f"down_proj_weights must be {expected_down_shape}, got {tuple(down_proj_weights.shape)}"

    matmul_core_grid = dp_spec.build_matmul_core_grid()

    if moe_tp == 1:
        dp_combined = down_proj_weights
        dp_mapper = ttnn.ReplicateTensorToMesh(device)
    else:
        mesh_rows = device.shape[0]
        mesh_cols = device.shape[1]
        dp_combined = (
            down_proj_weights.reshape(mesh_rows, mesh_cols, K_down_per_device, N_down)
            .permute(0, 2, 1, 3)
            .reshape(mesh_rows * K_down_per_device, mesh_cols * N_down)
        )
        dp_mapper = ttnn.ShardTensor2dMesh(device, mesh_shape=(mesh_rows, mesh_cols), dims=(0, 1))
    device_dp = device if move_to_device else None

    dp_shard_spec = ttnn.ShardSpec(matmul_core_grid, (K_down_per_device, N_per_core), ttnn.ShardOrientation.ROW_MAJOR)
    dp_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, dp_shard_spec)

    return ttnn.from_torch(
        dp_combined,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device_dp,
        memory_config=dp_mem,
        tile=ttnn.Tile([32, 32]),
        mesh_mapper=dp_mapper,
    )


def _shuffle_dram_tiles(tensor: torch.Tensor, tile_size: int, num_banks: int) -> torch.Tensor:
    """Reorder tiles within each DRAM bank shard from row-major to column-major.

    WIDTH_SHARDED DRAM layout stores tiles row-major, but the streaming
    matmul kernel expects K tiles contiguous for each N column.  This
    function transposes the tile order within each shard so that the
    kernel can linearly read K tiles at a time.
    """
    orig_shape = tensor.shape
    K, N = orig_shape[-2], orig_shape[-1]

    lcm = tile_size * num_banks
    n_padded = ((N + lcm - 1) // lcm) * lcm
    needs_padding = n_padded != N

    tensor = tensor.reshape(-1, K, N)
    batch_size = tensor.shape[0]

    if needs_padding:
        tensor = torch.nn.functional.pad(tensor, (0, n_padded - N))

    K_tiles = K // tile_size
    per_N = n_padded // num_banks
    per_N_tiles = per_N // tile_size
    num_tiles_per_shard = K_tiles * per_N_tiles

    tensor = tensor.reshape(batch_size, K, num_banks, per_N)
    tensor = tensor.permute(0, 2, 1, 3).contiguous()
    shards = tensor.reshape(-1, K, per_N)

    tiles = shards.reshape(-1, K_tiles, tile_size, per_N_tiles, tile_size)
    tiles = tiles.permute(0, 1, 3, 2, 4).contiguous()
    tiles = tiles.reshape(-1, num_tiles_per_shard, tile_size, tile_size)

    i = torch.arange(num_tiles_per_shard, device=tensor.device)
    source_idx = (i % K_tiles) * per_N_tiles + (i // K_tiles)
    shuffled_tiles = tiles[:, source_idx, :, :]

    shuffled_tiles = shuffled_tiles.reshape(-1, K_tiles, per_N_tiles, tile_size, tile_size)
    shuffled_tiles = shuffled_tiles.permute(0, 1, 3, 2, 4).contiguous()
    shuffled_shards = shuffled_tiles.reshape(-1, K, per_N)

    shuffled = shuffled_shards.reshape(batch_size, num_banks, K, per_N)
    shuffled = shuffled.permute(0, 2, 1, 3).contiguous()
    shuffled = shuffled.reshape(batch_size, K, n_padded)

    if needs_padding:
        shuffled = shuffled[:, :, :N]

    return shuffled.reshape(*orig_shape)


def _create_moe_routed_experts(
    device,
    gate_proj_weights: torch.Tensor,
    up_proj_weights: torch.Tensor,
    down_proj_weights: torch.Tensor,
    *,
    dtype: ttnn.DataType = ttnn.bfloat4_b,
    move_to_device: bool = True,
) -> tuple[list[ttnn.Tensor], list[ttnn.Tensor], list[ttnn.Tensor]]:
    """Create DRAM WIDTH_SHARDED expert weight tensors for routed MoE.

    Each expert projection is uploaded as a separate WIDTH_SHARDED tensor
    across all DRAM banks as ``bfloat4_b``.

    **IMPORTANT (DRAM layout):** For each projection list, experts must be allocated
    contiguously in DRAM. The inner ``upload()`` loop allocates all experts of one
    projection before the next, which guarantees contiguity.
    """
    tile_w = 32
    num_banks = device.dram_grid_size().x
    mesh_mapper = ttnn.ReplicateTensorToMesh(device)
    device_for_torch = device if move_to_device else None

    def upload(expert_weights: torch.Tensor) -> list[ttnn.Tensor]:
        num_experts, K, N = expert_weights.shape
        N_padded = ((N + num_banks * tile_w - 1) // (num_banks * tile_w)) * (num_banks * tile_w)
        per_core_N = N_padded // num_banks

        dram_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(device.dram_grid_size().x - 1, device.dram_grid_size().y - 1),
                )
            }
        )
        shard_spec = ttnn.ShardSpec(dram_grid, [K, per_core_N], ttnn.ShardOrientation.ROW_MAJOR)
        mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)

        tensors = []
        for i in range(num_experts):
            w = expert_weights[i]
            if N_padded != N:
                w = torch.nn.functional.pad(w, (0, N_padded - N))

            w_shuffled = _shuffle_dram_tiles(w.unsqueeze(0), tile_w, num_banks)
            w_shuffled = w_shuffled.reshape(1, 1, K, N_padded)

            tensors.append(
                ttnn.from_torch(
                    w_shuffled.contiguous(),
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=device_for_torch,
                    memory_config=mem_config,
                    mesh_mapper=mesh_mapper,
                )
            )
            if (i + 1) % 32 == 0:
                logger.info(f"  Uploaded {i + 1}/{num_experts} experts")
        return tensors

    return upload(gate_proj_weights), upload(up_proj_weights), upload(down_proj_weights)


def _create_mlp_routed_experts(
    device,
    gate_proj_weights: torch.Tensor,
    up_proj_weights: torch.Tensor,
    down_proj_weights: torch.Tensor,
    *,
    dtype: ttnn.DataType = ttnn.bfloat4_b,
    move_to_device: bool = True,
) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
    """Create MLP per-device routed expert weights (DRAM).

    After the shared expert (first 2048), the remaining ``8 * 2048``
    columns (gate/up) or rows (down) are split into 8 routed experts,
    one per device.
    """
    shared_n = 2048
    num_routed = 8
    expert_n = 2048

    K_gate = gate_proj_weights.shape[0]  # 7168
    N_down = down_proj_weights.shape[1]  # 7168

    gate_experts = (
        gate_proj_weights[:, shared_n:].reshape(K_gate, num_routed, expert_n).permute(1, 0, 2).contiguous()
    )  # (8, 7168, 2048)
    up_experts = (
        up_proj_weights[:, shared_n:].reshape(K_gate, num_routed, expert_n).permute(1, 0, 2).contiguous()
    )  # (8, 7168, 2048)
    down_experts = down_proj_weights[shared_n:, :].reshape(num_routed, expert_n, N_down).contiguous()  # (8, 2048, 7168)

    tile_w = 32
    num_banks = device.dram_grid_size().x
    mesh_rows = device.shape[0]
    mesh_cols = device.shape[1]
    mesh_mapper = ttnn.ShardTensor2dMesh(device, mesh_shape=(mesh_rows, mesh_cols), dims=(0, 1))
    device_for_torch = device if move_to_device else None

    def upload(experts: torch.Tensor) -> ttnn.Tensor:
        n_exp, K, N = experts.shape
        N_padded = ((N + num_banks * tile_w - 1) // (num_banks * tile_w)) * (num_banks * tile_w)
        per_core_N = N_padded // num_banks

        processed = []
        for i in range(n_exp):
            w = experts[i]
            if N_padded != N:
                w = torch.nn.functional.pad(w, (0, N_padded - N))
            w_shuffled = _shuffle_dram_tiles(w.unsqueeze(0), tile_w, num_banks)
            processed.append(w_shuffled.reshape(K, N_padded))

        stacked = torch.stack(processed).reshape(mesh_rows, mesh_cols, K, N_padded)

        dram_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(device.dram_grid_size().x - 1, device.dram_grid_size().y - 1),
                )
            }
        )
        shard_spec = ttnn.ShardSpec(dram_grid, [K, per_core_N], ttnn.ShardOrientation.ROW_MAJOR)
        mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)

        return ttnn.from_torch(
            stacked.contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device_for_torch,
            memory_config=mem_config,
            mesh_mapper=mesh_mapper,
        )

    return upload(gate_experts), upload(up_experts), upload(down_experts)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def deinterleave_q_b_proj(q_b_proj: torch.Tensor, num_heads: int | None = None) -> torch.Tensor:
    """Convert q_b_proj.weight from HF interleaved to [ALL_NOPE | ALL_ROPE] layout."""
    q_b_transposed = q_b_proj.T
    K, N = q_b_transposed.shape
    if num_heads is None:
        num_heads = N // _Q_HEAD_DIM
    heads = q_b_transposed.reshape(K, num_heads, _Q_HEAD_DIM)
    nope = heads[:, :, :_QK_NOPE_HEAD_DIM].reshape(K, -1)
    rope = heads[:, :, _QK_NOPE_HEAD_DIM:].reshape(K, -1)
    return torch.cat([nope, rope], dim=1).contiguous()


def _key(layer_idx: int, suffix: str) -> str:
    """State dict key under model.layers.{layer_idx}."""
    return f"model.layers.{layer_idx}.{suffix}"


def create_gate_bias_tensor(raw_tensor: torch.Tensor, device, *, move_to_device: bool = False) -> ttnn.Tensor:
    """Build gate_bias (e_score_correction_bias) as HEIGHT_SHARDED on sender core, replicated across mesh."""
    sender_core = ttnn.CoreCoord(MOE_SENDER_GRID_SIZE[0] - 1, MOE_SENDER_GRID_SIZE[1] - 1)
    sender_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(sender_core, sender_core)])
    gate_bias_reshaped = raw_tensor.reshape(16, 16).T.contiguous().to(torch.bfloat16)
    gate_bias_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(sender_core_grid, (16, 16), ttnn.ShardOrientation.ROW_MAJOR),
    )
    return ttnn.from_torch(
        gate_bias_reshaped,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device if move_to_device else None,
        memory_config=gate_bias_mem_config,
        tile=_GATE_BIAS_TILE,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )


def split_kv_b_proj(kv_b_proj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split HF kv_b_proj (out_features, in_features) into kv_b1 and kv_b2."""
    out_features, kv_lora_rank = kv_b_proj.shape
    assert kv_lora_rank == _KV_LORA_RANK
    num_heads = out_features // _KV_B_PROJ_HEAD_DIM
    w = kv_b_proj.reshape(num_heads, _KV_B_PROJ_HEAD_DIM, _KV_LORA_RANK).contiguous()
    kv_b1 = w[:, :_QK_NOPE_HEAD_DIM, :].reshape(-1, _KV_LORA_RANK)
    kv_b2 = w[:, _QK_NOPE_HEAD_DIM:, :].reshape(-1, _KV_LORA_RANK).T.contiguous()
    return kv_b1, kv_b2


def _slice_attention_weights_for_mla_tp(
    q_b: torch.Tensor,
    o_proj: torch.Tensor,
    kv_b1: torch.Tensor,
    kv_b2: torch.Tensor,
    mla_tp: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """When state dict has full (2-TP) logical shapes and mla_tp==1, slice to single-TP."""
    if mla_tp > 1:
        return q_b, o_proj, kv_b1, kv_b2
    if q_b.shape[1] == _MLA_TP1_Q_B_WIDTH * 2:
        q_b = q_b[:, :_MLA_TP1_Q_B_WIDTH].contiguous()
    if o_proj.shape[0] == _MLA_TP1_O_PROJ_HEIGHT * 2:
        o_proj = o_proj[:_MLA_TP1_O_PROJ_HEIGHT, :].contiguous()
    if kv_b1.shape[0] == _MLA_TP1_KV_B1_HEIGHT * 2:
        kv_b1 = kv_b1[:_MLA_TP1_KV_B1_HEIGHT, :].contiguous()
    if kv_b2.shape[1] == _MLA_TP1_KV_B2_WIDTH * 2:
        kv_b2 = kv_b2[:, :_MLA_TP1_KV_B2_WIDTH].contiguous()
    return q_b, o_proj, kv_b1, kv_b2


def _slice_shared_expert_weights_for_moe_tp(
    shared_gate: torch.Tensor,
    shared_up: torch.Tensor,
    shared_down: torch.Tensor,
    moe_tp: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """When state dict has full (8-TP) logical shapes and moe_tp==1, slice to single-TP."""
    if moe_tp > 1:
        return shared_gate, shared_up, shared_down
    full_n = _MOE_TP1_SHARED_GATE_UP_N * 8  # 2048
    if shared_gate.shape[1] == full_n:
        shared_gate = shared_gate[:, :_MOE_TP1_SHARED_GATE_UP_N].contiguous()
    if shared_up.shape[1] == full_n:
        shared_up = shared_up[:, :_MOE_TP1_SHARED_GATE_UP_N].contiguous()
    if shared_down.shape[0] == full_n:
        shared_down = shared_down[:_MOE_TP1_SHARED_DOWN_K, :].contiguous()
    return shared_gate, shared_up, shared_down


def get_layer_raw_tensors(
    state_dict: dict[str, torch.Tensor], layer_idx: int
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Extract and transform raw tensors for one layer from the state dict.

    Returns:
        (q_a, q_b, kv_a, kv_b1, kv_b2, o_proj, attn_norm, q_norm, kv_norm, ffn_norm).
    """
    q_a = state_dict[_key(layer_idx, "self_attn.q_a_proj.weight")].T.contiguous()
    q_b = deinterleave_q_b_proj(state_dict[_key(layer_idx, "self_attn.q_b_proj.weight")])
    kv_a = state_dict[_key(layer_idx, "self_attn.kv_a_proj_with_mqa.weight")].T.contiguous()
    kv_b1, kv_b2 = split_kv_b_proj(state_dict[_key(layer_idx, "self_attn.kv_b_proj.weight")])
    o_proj = state_dict[_key(layer_idx, "self_attn.o_proj.weight")].T.contiguous()

    attn_norm = state_dict[_key(layer_idx, "input_layernorm.weight")].unsqueeze(0)
    q_norm = state_dict[_key(layer_idx, "self_attn.q_a_layernorm.weight")].unsqueeze(0)
    kv_norm = state_dict[_key(layer_idx, "self_attn.kv_a_layernorm.weight")].unsqueeze(0)
    ffn_norm = state_dict[_key(layer_idx, "post_attention_layernorm.weight")].unsqueeze(0)

    return q_a, q_b, kv_a, kv_b1, kv_b2, o_proj, attn_norm, q_norm, kv_norm, ffn_norm


_DENSE_SHARED_N = 2048
_DENSE_NUM_ROUTED = 8
_DENSE_EXPERT_N = 2048


def _extract_torch_weights(
    state_dict: dict[str, torch.Tensor],
    layer_idx: int,
    *,
    is_moe: bool,
    num_routed_experts: int = NUM_ROUTED_EXPERTS,
) -> dict[str, Any]:
    """Extract transformed torch tensors for one layer, keyed by canonical weight names.

    Tensors are in matmul-ready layout (transposed, deinterleaved, split) --
    the same transforms that ``get_layer_raw_tensors`` applies for attention,
    extended with FFN / MoE weights.  Suitable for golden computation.
    """
    q_a, q_b, kv_a, kv_b1, kv_b2, o_proj, attn_norm, q_norm, kv_norm, ffn_norm = get_layer_raw_tensors(
        state_dict, layer_idx
    )

    result: dict[str, Any] = {
        "q_a_proj": q_a,
        "q_b_proj": q_b,
        "kv_a_proj": kv_a,
        "kv_b1_proj": kv_b1,
        "kv_b2_proj": kv_b2,
        "o_proj": o_proj,
        "attn_norm": attn_norm,
        "q_norm": q_norm,
        "kv_norm": kv_norm,
        "ffn_norm": ffn_norm,
    }

    if is_moe:
        result["shared_gate_proj"] = state_dict[_key(layer_idx, "mlp.shared_experts.gate_proj.weight")].T.contiguous()
        result["shared_up_proj"] = state_dict[_key(layer_idx, "mlp.shared_experts.up_proj.weight")].T.contiguous()
        result["shared_down_proj"] = state_dict[_key(layer_idx, "mlp.shared_experts.down_proj.weight")].T.contiguous()
        result["gate_mm"] = state_dict[_key(layer_idx, "mlp.gate.weight")].T.contiguous()
        result["gate_bias"] = state_dict[_key(layer_idx, "mlp.gate.e_score_correction_bias")]

        routed_gate: list[torch.Tensor] = []
        routed_up: list[torch.Tensor] = []
        routed_down: list[torch.Tensor] = []
        for e in range(num_routed_experts):
            routed_gate.append(state_dict[_key(layer_idx, f"mlp.experts.{e}.gate_proj.weight")].T.contiguous())
            routed_up.append(state_dict[_key(layer_idx, f"mlp.experts.{e}.up_proj.weight")].T.contiguous())
            routed_down.append(state_dict[_key(layer_idx, f"mlp.experts.{e}.down_proj.weight")].T.contiguous())
        result["routed_gate_proj"] = routed_gate
        result["routed_up_proj"] = routed_up
        result["routed_down_proj"] = routed_down
    else:
        gate_full = state_dict[_key(layer_idx, "mlp.gate_proj.weight")].T.contiguous()
        up_full = state_dict[_key(layer_idx, "mlp.up_proj.weight")].T.contiguous()
        down_full = state_dict[_key(layer_idx, "mlp.down_proj.weight")].T.contiguous()

        result["shared_gate_proj"] = gate_full[:, :_DENSE_SHARED_N].contiguous()
        result["shared_up_proj"] = up_full[:, :_DENSE_SHARED_N].contiguous()
        result["shared_down_proj"] = down_full[:_DENSE_SHARED_N, :].contiguous()

        routed_gate = []
        routed_up = []
        routed_down = []
        for e in range(_DENSE_NUM_ROUTED):
            start = _DENSE_SHARED_N + e * _DENSE_EXPERT_N
            end = start + _DENSE_EXPERT_N
            routed_gate.append(gate_full[:, start:end].contiguous())
            routed_up.append(up_full[:, start:end].contiguous())
            routed_down.append(down_full[start:end, :].contiguous())
        result["routed_gate_proj"] = routed_gate
        result["routed_up_proj"] = routed_up
        result["routed_down_proj"] = routed_down

    return result


# Gate routing constants (bias/indices layout on sender core)
_GATE_BIAS_INDICES_SHAPE = (16, 16)
_GATE_NUM_INDICES = 256


def create_gate_indices_tensor(
    device: Any,
    sender_core_grid: ttnn.CoreRangeSet,
    *,
    mesh_mapper: Any = None,
) -> ttnn.Tensor:
    """Build constant gate indices 0..255 as HEIGHT_SHARDED on sender core."""
    indices = torch.arange(_GATE_NUM_INDICES, dtype=torch.int32).reshape(
        _GATE_BIAS_INDICES_SHAPE[0], _GATE_BIAS_INDICES_SHAPE[1]
    )
    transposed = torch.transpose(indices, 0, 1).contiguous().to(torch.uint16)
    shard_spec = ttnn.ShardSpec(
        sender_core_grid,
        _GATE_BIAS_INDICES_SHAPE,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    kwargs = {"mesh_mapper": mesh_mapper} if mesh_mapper else {}
    return ttnn.from_torch(
        transposed,
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem_config,
        tile=ttnn.Tile([16, 16]),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Cache-aware helpers
# ---------------------------------------------------------------------------


def _cached_or_fuse(cache_config, mesh_shape, group_name, layer_idx, spec_fps_fn, fuse_fn, device):
    if cache_config is not None:
        fp = _make_fp(cache_config, mesh_shape, group_name, layer_idx, spec_fps_fn())
        return cache_config.cache.get_or_create(fp, fuse=fuse_fn, device=device)
    return fuse_fn()


def _cached_or_create(cache_config, mesh_shape, tensor_name, layer_idx, create_fn, device):
    if cache_config is not None:
        fp = _make_fp(cache_config, mesh_shape, tensor_name, layer_idx, ())
        return cache_config.cache.get_or_create_tensor(fp, create=create_fn, device=device)
    return create_fn()


# ---------------------------------------------------------------------------
# Public prepare_* API
# ---------------------------------------------------------------------------


def prepare_attention_weights(
    device,
    state_dict: dict[str, torch.Tensor],
    layer_idx: int,
    *,
    is_moe: bool,
    move_to_device: bool = False,
    cache_config: CacheConfig | None = None,
) -> AttentionWeights:
    """Prepare attention fusion groups for one layer (q_ab_kv_a, kv_b12, o_proj_gate_mm_norms).

    When ``cache_config`` is provided, each fusion group is routed through
    :meth:`TensorCache.get_or_create`.  On a hit the raw tensors are still
    loaded (cheap) but the expensive tilize-pack-fuse step is skipped.
    """
    mesh = _mesh_shape(device)
    mla_tp, _ = _compute_tp(device)

    logger.debug("Loading raw tensors from state dict for layer {}", layer_idx)
    t0 = time.perf_counter()
    q_a, q_b, kv_a, kv_b1, kv_b2, o_proj, attn_norm, q_norm, kv_norm, ffn_norm = get_layer_raw_tensors(
        state_dict, layer_idx
    )
    q_b, o_proj, kv_b1, kv_b2 = _slice_attention_weights_for_mla_tp(q_b, o_proj, kv_b1, kv_b2, mla_tp)
    logger.debug("  load raw tensors: {:.3f}s", time.perf_counter() - t0)
    logger.debug("Converting attention fusion groups for layer {} (q_ab_kv_a, kv_b12, o_proj_gate_mm_norms)", layer_idx)
    t0 = time.perf_counter()

    qab_kva = _cached_or_fuse(
        cache_config,
        mesh,
        "q_ab_kv_a",
        layer_idx,
        _qab_kva_spec_fps,
        lambda: _fuse_q_ab_kv_a(device, q_a, q_b, kv_a, move_to_device=move_to_device),
        device,
    )
    q_a_proj, q_b_proj, kv_a_proj = qab_kva["q_a_proj"], qab_kva["q_b_proj"], qab_kva["kv_a_proj"]

    kv_b12 = _cached_or_fuse(
        cache_config,
        mesh,
        "kv_b12",
        layer_idx,
        _kv_b12_spec_fps,
        lambda: _fuse_kv_b12(device, kv_b1, kv_b2, move_to_device=move_to_device),
        device,
    )
    kv_b1_proj, kv_b2_proj = kv_b12["kv_b1_proj"], kv_b12["kv_b2_proj"]
    logger.debug("  convert q_ab_kv_a + kv_b12: {:.3f}s", time.perf_counter() - t0)

    if is_moe:
        gate_mm = state_dict[_key(layer_idx, "mlp.gate.weight")].T.contiguous()
        o_norms = _cached_or_fuse(
            cache_config,
            mesh,
            "o_proj_gate_mm_norms",
            layer_idx,
            _o_proj_norms_spec_fps,
            lambda: _fuse_o_proj_gate_mm_norms(
                device, o_proj, gate_mm, attn_norm, q_norm, kv_norm, ffn_norm, move_to_device=move_to_device
            ),
            device,
        )
        gate_bias_raw = state_dict[_key(layer_idx, "mlp.gate.e_score_correction_bias")]
        gate_bias_tt = _cached_or_create(
            cache_config,
            mesh,
            "gate_bias",
            layer_idx,
            lambda: create_gate_bias_tensor(gate_bias_raw, device, move_to_device=move_to_device),
            device,
        )
        logger.debug("  convert o_proj_gate_mm_norms (MoE): {:.3f}s", time.perf_counter() - t0)
        return AttentionWeights(
            q_a_proj=q_a_proj,
            q_b_proj=q_b_proj,
            kv_a_proj=kv_a_proj,
            o_proj=o_norms["o_proj"],
            gate_mm=o_norms["gate_mm"],
            attn_norm=o_norms["attn_norm"],
            q_norm=o_norms["q_norm"],
            kv_norm=o_norms["kv_norm"],
            ffn_norm=o_norms["ffn_norm"],
            kv_b1_proj=kv_b1_proj,
            kv_b2_proj=kv_b2_proj,
            gate_bias=gate_bias_tt,
        )
    else:
        gate_mm_dummy = torch.zeros(7168, 256, dtype=torch.bfloat16, device=next(iter(state_dict.values())).device)
        o_norms = _cached_or_fuse(
            cache_config,
            mesh,
            "o_proj_gate_mm_norms",
            layer_idx,
            _o_proj_norms_spec_fps,
            lambda: _fuse_o_proj_gate_mm_norms(
                device, o_proj, gate_mm_dummy, attn_norm, q_norm, kv_norm, ffn_norm, move_to_device=move_to_device
            ),
            device,
        )
        logger.debug("  convert o_proj_gate_mm_norms (dense): {:.3f}s", time.perf_counter() - t0)
        return AttentionWeights(
            q_a_proj=q_a_proj,
            q_b_proj=q_b_proj,
            kv_a_proj=kv_a_proj,
            o_proj=o_norms["o_proj"],
            gate_mm=None,
            attn_norm=o_norms["attn_norm"],
            q_norm=o_norms["q_norm"],
            kv_norm=o_norms["kv_norm"],
            ffn_norm=o_norms["ffn_norm"],
            kv_b1_proj=kv_b1_proj,
            kv_b2_proj=kv_b2_proj,
            gate_bias=None,
        )


def prepare_shared_expert_weights(
    device,
    state_dict: dict[str, torch.Tensor],
    layer_idx: int,
    *,
    is_moe: bool,
    move_to_device: bool = False,
    cache_config: CacheConfig | None = None,
) -> SharedExpertWeights:
    """Prepare shared expert weights (gate_up fusion group + shared_down_proj) for one layer."""
    mesh = _mesh_shape(device)
    _, moe_tp = _compute_tp(device)
    logger.debug("Converting shared expert weights for layer {} (is_moe={})", layer_idx, is_moe)
    t0 = time.perf_counter()

    if is_moe:
        shared_gate = state_dict[_key(layer_idx, "mlp.shared_experts.gate_proj.weight")].T.contiguous()
        shared_up = state_dict[_key(layer_idx, "mlp.shared_experts.up_proj.weight")].T.contiguous()
        shared_down = state_dict[_key(layer_idx, "mlp.shared_experts.down_proj.weight")].T.contiguous()
        shared_gate, shared_up, shared_down = _slice_shared_expert_weights_for_moe_tp(
            shared_gate, shared_up, shared_down, moe_tp
        )
        fuse_gate_up = lambda: _fuse_gate_up(device, shared_gate, shared_up, move_to_device=move_to_device)
        create_down = lambda: _create_shared_down_proj(device, shared_down, move_to_device=move_to_device)
    else:
        shared_n = 2048
        mlp_gate = state_dict[_key(layer_idx, "mlp.gate_proj.weight")].T.contiguous()[:, :shared_n]
        mlp_up = state_dict[_key(layer_idx, "mlp.up_proj.weight")].T.contiguous()[:, :shared_n]
        mlp_down = state_dict[_key(layer_idx, "mlp.down_proj.weight")].T.contiguous()[:shared_n, :]
        fuse_gate_up = lambda: _fuse_gate_up(device, mlp_gate, mlp_up, move_to_device=move_to_device)
        create_down = lambda: _create_shared_down_proj(device, mlp_down, move_to_device=move_to_device)

    gate_up = _cached_or_fuse(cache_config, mesh, "gate_up", layer_idx, _gate_up_spec_fps, fuse_gate_up, device)
    shared_down_proj = _cached_or_create(cache_config, mesh, "shared_down_proj", layer_idx, create_down, device)

    logger.debug("  shared expert weights done in {:.3f}s", time.perf_counter() - t0)
    return SharedExpertWeights(
        shared_gate_proj=gate_up["gate_proj"],
        shared_up_proj=gate_up["up_proj"],
        shared_down_proj=shared_down_proj,
    )


def prepare_routed_expert_weights(
    device,
    state_dict: dict[str, torch.Tensor],
    layer_idx: int,
    *,
    is_moe: bool,
    num_routed_experts: int = NUM_ROUTED_EXPERTS,
    move_to_device: bool = False,
    cache_config: CacheConfig | None = None,
) -> DenseRoutedExpertWeights | MoERoutedExpertWeights:
    """Prepare routed expert weights for one layer (dense: single MLP; MoE: num_routed_experts experts).

    When ``cache_config`` is provided:
    * **MoE** -- each projection list (gate, up, down) is routed through
      :meth:`TensorCache.get_or_create_tensor_list` for atomic, order-preserving
      caching that guarantees contiguous DRAM allocation.
    * **Dense** -- each of the three projections is routed through
      :meth:`TensorCache.get_or_create_tensor`.
    """
    mesh = _mesh_shape(device)

    if is_moe:

        def _create_all_experts():
            """Load raw weights and convert all experts. Returns (gate_list, up_list, down_list)."""
            logger.info(
                "Loading and converting {} routed experts for layer {} (this may be slow)...",
                num_routed_experts,
                layer_idx,
            )
            t0 = time.perf_counter()
            gate_list = []
            up_list = []
            down_list = []
            for e in range(num_routed_experts):
                if e > 0 and e % 64 == 0:
                    logger.debug("  loaded experts 0..{} from state dict", e - 1)
                gate_list.append(state_dict[_key(layer_idx, f"mlp.experts.{e}.gate_proj.weight")].T.contiguous())
                up_list.append(state_dict[_key(layer_idx, f"mlp.experts.{e}.up_proj.weight")].T.contiguous())
                down_list.append(state_dict[_key(layer_idx, f"mlp.experts.{e}.down_proj.weight")].T.contiguous())
            load_elapsed = time.perf_counter() - t0
            logger.info("  loaded {} experts from state dict in {:.3f}s", num_routed_experts, load_elapsed)
            logger.debug("Converting routed experts to device format...")
            t0 = time.perf_counter()
            gate_stacked = torch.stack(gate_list, dim=0)
            up_stacked = torch.stack(up_list, dim=0)
            down_stacked = torch.stack(down_list, dim=0)
            routed_gate, routed_up, routed_down = _create_moe_routed_experts(
                device, gate_stacked, up_stacked, down_stacked, move_to_device=move_to_device
            )
            logger.info("  converted routed experts in {:.3f}s", time.perf_counter() - t0)
            return routed_gate, routed_up, routed_down

        if cache_config is not None:
            _experts_cache: tuple | None = None

            def _ensure_experts():
                nonlocal _experts_cache
                if _experts_cache is None:
                    _experts_cache = _create_all_experts()

            def _create_proj(proj_idx):
                def _create():
                    _ensure_experts()
                    return list(_experts_cache[proj_idx])

                return _create

            fp_gate = _make_fp(cache_config, mesh, "routed_gate_proj", layer_idx, ())
            fp_up = _make_fp(cache_config, mesh, "routed_up_proj", layer_idx, ())
            fp_down = _make_fp(cache_config, mesh, "routed_down_proj", layer_idx, ())
            routed_gate_proj = cache_config.cache.get_or_create_tensor_list(
                fp_gate, create=_create_proj(0), device=device
            )
            routed_up_proj = cache_config.cache.get_or_create_tensor_list(fp_up, create=_create_proj(1), device=device)
            routed_down_proj = cache_config.cache.get_or_create_tensor_list(
                fp_down, create=_create_proj(2), device=device
            )
        else:
            routed_gate_proj, routed_up_proj, routed_down_proj = _create_all_experts()

        routed = MoERoutedExpertWeights(
            routed_gate_proj=routed_gate_proj,
            routed_up_proj=routed_up_proj,
            routed_down_proj=routed_down_proj,
        )
        if move_to_device:
            routed.validate_contiguous_dram()
        return routed
    else:
        mlp_gate = state_dict[_key(layer_idx, "mlp.gate_proj.weight")].T.contiguous()
        mlp_up = state_dict[_key(layer_idx, "mlp.up_proj.weight")].T.contiguous()
        mlp_down = state_dict[_key(layer_idx, "mlp.down_proj.weight")].T.contiguous()

        if cache_config is not None:
            _dense_cache: dict[str, ttnn.Tensor] = {}

            def _ensure_dense_converted():
                if not _dense_cache:
                    g, u, d = _create_mlp_routed_experts(
                        device, mlp_gate, mlp_up, mlp_down, move_to_device=move_to_device
                    )
                    _dense_cache["gate"] = g
                    _dense_cache["up"] = u
                    _dense_cache["down"] = d

            def _cached_or_create_dense(tensor_name, key):
                fp = _make_fp(cache_config, mesh, tensor_name, layer_idx, ())

                def _create():
                    _ensure_dense_converted()
                    return _dense_cache[key]

                return cache_config.cache.get_or_create_tensor(fp, create=_create, device=device)

            routed_gate_proj = _cached_or_create_dense("routed_gate_proj", "gate")
            routed_up_proj = _cached_or_create_dense("routed_up_proj", "up")
            routed_down_proj = _cached_or_create_dense("routed_down_proj", "down")
        else:
            routed_gate_proj, routed_up_proj, routed_down_proj = _create_mlp_routed_experts(
                device, mlp_gate, mlp_up, mlp_down, move_to_device=move_to_device
            )
        return DenseRoutedExpertWeights(
            routed_gate_proj=routed_gate_proj,
            routed_up_proj=routed_up_proj,
            routed_down_proj=routed_down_proj,
        )


def prepare_dense_layer_weights(
    device,
    state_dict: dict[str, torch.Tensor],
    layer_idx: int,
    *,
    move_to_device: bool = False,
    cache_config: CacheConfig | None = None,
    store_torch: bool = False,
) -> ModelWeights:
    """Prepare fused weights for a single dense decoder layer."""
    logger.info("Preparing dense layer {}...", layer_idx)
    t0 = time.perf_counter()
    attn = prepare_attention_weights(
        device, state_dict, layer_idx, is_moe=False, move_to_device=move_to_device, cache_config=cache_config
    )
    shared = prepare_shared_expert_weights(
        device, state_dict, layer_idx, is_moe=False, move_to_device=move_to_device, cache_config=cache_config
    )
    routed = prepare_routed_expert_weights(
        device, state_dict, layer_idx, is_moe=False, move_to_device=move_to_device, cache_config=cache_config
    )
    assert isinstance(routed, DenseRoutedExpertWeights)
    torch_weights = _extract_torch_weights(state_dict, layer_idx, is_moe=False) if store_torch else None
    logger.info("  dense layer {} done in {:.3f}s", layer_idx, time.perf_counter() - t0)
    return ModelWeights(
        {
            "q_a_proj": attn.q_a_proj,
            "q_b_proj": attn.q_b_proj,
            "kv_a_proj": attn.kv_a_proj,
            "o_proj": attn.o_proj,
            "attn_norm": attn.attn_norm,
            "q_norm": attn.q_norm,
            "kv_norm": attn.kv_norm,
            "ffn_norm": attn.ffn_norm,
            "kv_b1_proj": attn.kv_b1_proj,
            "kv_b2_proj": attn.kv_b2_proj,
            "shared_gate_proj": shared.shared_gate_proj,
            "shared_up_proj": shared.shared_up_proj,
            "shared_down_proj": shared.shared_down_proj,
            "routed_gate_proj": [routed.routed_gate_proj],
            "routed_up_proj": [routed.routed_up_proj],
            "routed_down_proj": [routed.routed_down_proj],
        },
        torch_weights,
    )


def prepare_moe_layer_weights(
    device,
    state_dict: dict[str, torch.Tensor],
    layer_idx: int,
    *,
    num_routed_experts: int = NUM_ROUTED_EXPERTS,
    move_to_device: bool = False,
    cache_config: CacheConfig | None = None,
    store_torch: bool = False,
) -> ModelWeights:
    """Prepare fused weights for a single MoE decoder layer."""
    logger.info("Preparing MoE layer {}...", layer_idx)
    t0 = time.perf_counter()
    attn = prepare_attention_weights(
        device, state_dict, layer_idx, is_moe=True, move_to_device=move_to_device, cache_config=cache_config
    )
    shared = prepare_shared_expert_weights(
        device, state_dict, layer_idx, is_moe=True, move_to_device=move_to_device, cache_config=cache_config
    )
    routed = prepare_routed_expert_weights(
        device,
        state_dict,
        layer_idx,
        is_moe=True,
        num_routed_experts=num_routed_experts,
        move_to_device=move_to_device,
        cache_config=cache_config,
    )
    assert isinstance(attn.gate_mm, OverlappedTensor)
    assert attn.gate_bias is not None
    assert isinstance(routed, MoERoutedExpertWeights)
    torch_weights = (
        _extract_torch_weights(state_dict, layer_idx, is_moe=True, num_routed_experts=num_routed_experts)
        if store_torch
        else None
    )
    logger.info("  MoE layer {} done in {:.3f}s", layer_idx, time.perf_counter() - t0)
    return ModelWeights(
        {
            "q_a_proj": attn.q_a_proj,
            "q_b_proj": attn.q_b_proj,
            "kv_a_proj": attn.kv_a_proj,
            "o_proj": attn.o_proj,
            "gate_mm": attn.gate_mm,
            "attn_norm": attn.attn_norm,
            "q_norm": attn.q_norm,
            "kv_norm": attn.kv_norm,
            "ffn_norm": attn.ffn_norm,
            "gate_bias": attn.gate_bias,
            "kv_b1_proj": attn.kv_b1_proj,
            "kv_b2_proj": attn.kv_b2_proj,
            "shared_gate_proj": shared.shared_gate_proj,
            "shared_up_proj": shared.shared_up_proj,
            "shared_down_proj": shared.shared_down_proj,
            "routed_gate_proj": routed.routed_gate_proj,
            "routed_up_proj": routed.routed_up_proj,
            "routed_down_proj": routed.routed_down_proj,
        },
        torch_weights,
    )


def _to_tt_embedding(embedding_torch: torch.Tensor, device, *, move_to_device: bool = False) -> ttnn.Tensor:
    """Convert a torch embedding tensor to TT (DRAM, ROW_MAJOR, ReplicateTensorToMesh)."""
    return ttnn.from_torch(
        embedding_torch.contiguous(),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device if move_to_device else None,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )


def prepare_embedding_weights(
    state_dict: dict[str, torch.Tensor],
    device,
    *,
    move_to_device: bool = False,
    cache_config: CacheConfig | None = None,
) -> DeepSeekV3EmbeddingLayerWeights:
    """Prepare embedding weights from state dict (model.embed_tokens.weight)."""
    logger.info("Preparing embedding weights...")
    w = state_dict["model.embed_tokens.weight"]
    assert w.shape == (129280, 7168), f"Expected embedding shape (129280, 7168), got {w.shape}"

    def create():
        return _to_tt_embedding(w, device, move_to_device=move_to_device)

    if cache_config is not None:
        mesh = _mesh_shape(device)
        fp = _make_fp(cache_config, mesh, "embedding", 0, ())
        embedding_tt = cache_config.cache.get_or_create_tensor(fp, create=create, device=device)
    else:
        embedding_tt = create()
    return DeepSeekV3EmbeddingLayerWeights(embedding=embedding_tt)


# LM head constants
_LM_HEAD_K = 7168
_LM_HEAD_VOCAB_SIZE = 129280
_LM_HEAD_NUM_MATMUL_CORES = 101
_LM_HEAD_MATMUL_CORE_GRID = ttnn.CoreRangeSet(
    [
        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 9)),
        ttnn.CoreRange(ttnn.CoreCoord(10, 0), ttnn.CoreCoord(10, 0)),
    ]
)
_LM_HEAD_B_TILE = ttnn.Tile([32, 32])
_LM_HEAD_A_TILE = ttnn.Tile([1, 32])
_LM_HEAD_N_PER_CORE = 160
_LM_HEAD_MCAST_CORE = ttnn.CoreCoord(10, 9)
_LM_HEAD_MCAST_CORE_GRID = ttnn.CoreRangeSet([ttnn.CoreRange(_LM_HEAD_MCAST_CORE, _LM_HEAD_MCAST_CORE)])


def _to_tt_lm_head_matrix(
    lm_head_torch: torch.Tensor, device, *, mesh_mapper, move_to_device: bool = False
) -> ttnn.Tensor:
    """Convert (K, N) lm_head torch tensor to TT (WIDTH_SHARDED 101 cores, L1)."""
    lm_head_shard_shape = (_LM_HEAD_K, _LM_HEAD_N_PER_CORE)
    lm_head_shard_spec = ttnn.ShardSpec(
        _LM_HEAD_MATMUL_CORE_GRID,
        lm_head_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    lm_head_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        lm_head_shard_spec,
    )
    return ttnn.from_torch(
        lm_head_torch.contiguous(),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device if move_to_device else None,
        memory_config=lm_head_mem_config,
        mesh_mapper=mesh_mapper,
        tile=_LM_HEAD_B_TILE,
    )


def _to_tt_lm_head_final_norm(norm_torch: torch.Tensor, device, *, move_to_device: bool = False) -> ttnn.Tensor:
    """Convert (1, K) final norm torch tensor to TT (HEIGHT_SHARDED on mcast core)."""
    norm_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(_LM_HEAD_MCAST_CORE_GRID, (1, _LM_HEAD_K), ttnn.ShardOrientation.ROW_MAJOR),
    )
    return ttnn.from_torch(
        norm_torch.contiguous(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        tile=_LM_HEAD_A_TILE,
        device=device if move_to_device else None,
        memory_config=norm_mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )


def prepare_lm_head_weights(
    state_dict: dict[str, torch.Tensor],
    device,
    *,
    move_to_device: bool = False,
    cache_config: CacheConfig | None = None,
) -> DeepSeekV3LMHeadWeights:
    """Prepare LM head and final norm weights from state dict."""
    logger.info("Preparing LM head weights...")
    lm_w = state_dict["lm_head.weight"]
    assert lm_w.shape == (
        _LM_HEAD_VOCAB_SIZE,
        _LM_HEAD_K,
    ), f"Expected lm_head shape ({_LM_HEAD_VOCAB_SIZE}, {_LM_HEAD_K}), got {lm_w.shape}"

    logger.info("Preparing LM head norm...")
    norm_w = state_dict["model.norm.weight"]
    assert norm_w.shape == (7168,), f"Expected final norm shape (7168,), got {norm_w.shape}"

    def create_lm_head():
        return _to_tt_lm_head_matrix(
            lm_w.T, device, mesh_mapper=ttnn.ShardTensorToMesh(device, dim=1), move_to_device=move_to_device
        )

    def create_final_norm():
        return _to_tt_lm_head_final_norm(norm_w.unsqueeze(0), device, move_to_device=move_to_device)

    if cache_config is not None:
        mesh = _mesh_shape(device)
        fp_lm = _make_fp(cache_config, mesh, "lm_head", 0, ())
        lm_head_tt = cache_config.cache.get_or_create_tensor(fp_lm, create=create_lm_head, device=device)
        fp_norm = _make_fp(cache_config, mesh, "final_norm", 0, ())
        final_norm_tt = cache_config.cache.get_or_create_tensor(fp_norm, create=create_final_norm, device=device)
    else:
        lm_head_tt = create_lm_head()
        final_norm_tt = create_final_norm()

    return DeepSeekV3LMHeadWeights(lm_head=lm_head_tt, final_norm=final_norm_tt)
