# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Pack preprocessed torch tensors into fused device buffers and OverlappedTensor views.

For production DeepSeek fusion groups, this delegates to :class:`BlitzDecodeWeights`
so layout matches the existing ``get_tt_*`` implementations.

A small ``_unittest_toy`` fusion group is supported for unit tests without full model shapes.
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.deepseek_v3_b1.tensor_cache.types import FusionGroupSpec


def create_overlapped_tensor(
    spec: FusionGroupSpec,
    preprocessed: dict[str, torch.Tensor],
    device,
    *,
    mesh_mapper=None,
    move_to_device: bool = True,
) -> tuple[ttnn.Tensor, dict[str, "OverlappedTensor"]]:
    """Pack preprocessed tensors into one fused buffer and logical views per ``spec``.

    Args:
        spec: Fusion layout (name must be a supported DeepSeek group or ``_unittest_toy``).
        preprocessed: Mapping from sub-tensor name to 2-D torch tensor (already transformed).
        device: Mesh or single device (used for placement and TP inference).
        mesh_mapper: Reserved for future generic paths; DeepSeek delegates ignore this
            (mapping is derived from device topology inside ``BlitzDecodeWeights``).
        move_to_device: If True, fused tensor is placed on ``device``; if False, host tensor
            with mesh metadata (for cache store before ``ttnn.load_tensor``).

    Returns:
        ``(fused_tensor, views)`` where ``views`` maps logical name to :class:`OverlappedTensor`.
    """
    del mesh_mapper  # DeepSeek path uses BlitzDecodeWeights internal mappers

    name = spec.name
    if name == "q_ab_kv_a":
        return _create_q_ab_kv_a(preprocessed, device, move_to_device)
    if name == "o_proj_gate_mm_norms":
        return _create_o_proj_gate_mm_norms(preprocessed, device, move_to_device)
    if name == "kv_b12":
        return _create_kv_b12(preprocessed, device, move_to_device)
    if name == "gate_up":
        return _create_gate_up(preprocessed, device, move_to_device)
    if name == "_unittest_toy":
        return _create_unittest_toy(preprocessed, device, move_to_device)

    raise ValueError(f"Unsupported FusionGroupSpec.name: {name!r}")


def _fused_and_dict(ots: list, keys: list[str]) -> tuple[ttnn.Tensor, dict]:
    fused = ots[0].fused_tensor
    return fused, {k: v for k, v in zip(keys, ots, strict=True)}


def _create_q_ab_kv_a(
    preprocessed: dict[str, torch.Tensor],
    device,
    move_to_device: bool,
) -> tuple[ttnn.Tensor, dict[str, "OverlappedTensor"]]:
    from models.demos.deepseek_v3_b1.blitz_decode_weights import BlitzDecodeWeights

    for k in ("q_a_proj", "q_b_proj", "kv_a_proj"):
        if k not in preprocessed:
            raise KeyError(f"preprocessed missing {k!r} for q_ab_kv_a")
    bw = BlitzDecodeWeights(device)
    ots = bw.get_tt_q_ab_proj_and_kv_a_proj_weights(
        preprocessed["q_a_proj"],
        preprocessed["q_b_proj"],
        preprocessed["kv_a_proj"],
        move_to_device=move_to_device,
    )
    return _fused_and_dict(ots, ["q_a_proj", "q_b_proj", "kv_a_proj"])


def _create_o_proj_gate_mm_norms(
    preprocessed: dict[str, torch.Tensor],
    device,
    move_to_device: bool,
) -> tuple[ttnn.Tensor, dict[str, "OverlappedTensor"]]:
    from models.demos.deepseek_v3_b1.blitz_decode_weights import BlitzDecodeWeights

    keys = ("o_proj", "gate_mm", "attn_norm", "q_norm", "kv_norm", "ffn_norm")
    for k in keys:
        if k not in preprocessed:
            raise KeyError(f"preprocessed missing {k!r} for o_proj_gate_mm_norms")
    bw = BlitzDecodeWeights(device)
    ots = bw.get_tt_o_proj_and_gate_mm_weights(
        preprocessed["o_proj"],
        preprocessed["gate_mm"],
        preprocessed["attn_norm"],
        preprocessed["q_norm"],
        preprocessed["kv_norm"],
        preprocessed["ffn_norm"],
        move_to_device=move_to_device,
    )
    return _fused_and_dict(ots, list(keys))


def _create_kv_b12(
    preprocessed: dict[str, torch.Tensor],
    device,
    move_to_device: bool,
) -> tuple[ttnn.Tensor, dict[str, "OverlappedTensor"]]:
    from models.demos.deepseek_v3_b1.blitz_decode_weights import BlitzDecodeWeights

    for k in ("kv_b1_proj", "kv_b2_proj"):
        if k not in preprocessed:
            raise KeyError(f"preprocessed missing {k!r} for kv_b12")
    bw = BlitzDecodeWeights(device)
    ots = bw.get_tt_kv_b12_proj_weights(
        preprocessed["kv_b1_proj"],
        preprocessed["kv_b2_proj"],
        move_to_device=move_to_device,
    )
    return _fused_and_dict(ots, ["kv_b1_proj", "kv_b2_proj"])


def _create_gate_up(
    preprocessed: dict[str, torch.Tensor],
    device,
    move_to_device: bool,
) -> tuple[ttnn.Tensor, dict[str, "OverlappedTensor"]]:
    from models.demos.deepseek_v3_b1.blitz_decode_weights import BlitzDecodeWeights

    for k in ("shared_gate_proj", "shared_up_proj"):
        if k not in preprocessed:
            raise KeyError(f"preprocessed missing {k!r} for gate_up")
    bw = BlitzDecodeWeights(device)
    moe_tp = bw.moe_tp
    k_down = 256 * moe_tp
    n_down = 7168
    dummy_down = torch.zeros((k_down, n_down), dtype=torch.bfloat16)
    gate_ov, up_ov, _down = bw.get_tt_moe_shared_expert_weights(
        preprocessed["shared_gate_proj"],
        preprocessed["shared_up_proj"],
        dummy_down,
        move_to_device=move_to_device,
    )
    del _down
    fused = gate_ov.fused_tensor
    return fused, {"shared_gate_proj": gate_ov, "shared_up_proj": up_ov}


def _create_unittest_toy(
    preprocessed: dict[str, torch.Tensor],
    device,
    move_to_device: bool,
) -> tuple[ttnn.Tensor, dict[str, "OverlappedTensor"]]:
    """Two 32×32 bfloat16 tiles stacked vertically on one WIDTH_SHARDED core; TILE layout."""
    from models.demos.deepseek_v3_b1.blitz_decode_weights import OverlappedTensor

    for k in ("a", "b"):
        if k not in preprocessed:
            raise KeyError("preprocessed must contain 'a' and 'b' for _unittest_toy")
        t = preprocessed[k]
        if tuple(t.shape) != (32, 32):
            raise ValueError(f"_unittest_toy expects (32,32) tensors, got {k}={tuple(t.shape)}")

    a = preprocessed["a"].to(dtype=torch.bfloat16).contiguous()
    b = preprocessed["b"].to(dtype=torch.bfloat16).contiguous()
    combined = torch.cat([a, b], dim=0)
    assert combined.shape == (64, 32)

    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    shard_spec = ttnn.ShardSpec(crs, (64, 32), ttnn.ShardOrientation.ROW_MAJOR)
    mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        shard_spec,
    )
    mesh_mapper = ttnn.ReplicateTensorToMesh(device)
    device_for_torch = device if move_to_device else None

    fused = ttnn.from_torch(
        combined,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device_for_torch,
        memory_config=mem_config,
        mesh_mapper=mesh_mapper,
        tile=ttnn.Tile((32, 32)),
    )

    tile = fused.get_tile()
    ts = tuple(tile.tile_shape)
    tile_bytes = tile.get_tile_size(ttnn.bfloat16)
    total_one = tile_bytes  # one tile per subtensor

    crs_set = crs
    v_a = OverlappedTensor(
        fused_tensor=fused,
        tensor_shape=(32, 32),
        shard_shape=(32, 32),
        core_range_set=crs_set,
        dtype=ttnn.bfloat16,
        tile_shape=ts,
        byte_offset=0,
        total_size=total_one,
    )
    v_b = OverlappedTensor(
        fused_tensor=fused,
        tensor_shape=(32, 32),
        shard_shape=(32, 32),
        core_range_set=crs_set,
        dtype=ttnn.bfloat16,
        tile_shape=ts,
        byte_offset=total_one,
        total_size=total_one,
    )
    return fused, {"a": v_a, "b": v_b}
