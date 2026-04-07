# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek **pack/fuse** runtime: build fused device buffers and ``OverlappedTensor`` views.

This is model-adapter fusion logic (delegates to :class:`BlitzDecodeWeights`). The generic cache
calls this via the compatibility shim :mod:`models.demos.deepseek_v3_b1.tensor_cache.fuse`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch

import ttnn
from models.demos.deepseek_v3_b1.tensor_cache.types import FusionGroupSpec

if TYPE_CHECKING:
    from models.demos.deepseek_v3_b1.blitz_decode_weights import OverlappedTensor

_FUSION_GROUP_CREATORS: dict[
    str,
    Callable[[dict[str, torch.Tensor], object, bool], tuple[ttnn.Tensor, dict[str, "OverlappedTensor"]]],
] = {}


def _register_fusion_group(
    name: str,
) -> Callable[
    [Callable[[dict[str, torch.Tensor], object, bool], tuple[ttnn.Tensor, dict[str, "OverlappedTensor"]]]],
    Callable[[dict[str, torch.Tensor], object, bool], tuple[ttnn.Tensor, dict[str, "OverlappedTensor"]]],
]:
    """Register a FusionGroupSpec.name -> creator function mapping."""

    def decorator(
        fn: Callable[[dict[str, torch.Tensor], object, bool], tuple[ttnn.Tensor, dict[str, "OverlappedTensor"]]],
    ) -> Callable[[dict[str, torch.Tensor], object, bool], tuple[ttnn.Tensor, dict[str, "OverlappedTensor"]]]:
        _FUSION_GROUP_CREATORS[name] = fn
        return fn

    return decorator


def _validate_views_match_spec(spec: FusionGroupSpec, views: dict[str, OverlappedTensor]) -> None:
    """Assert produced OverlappedTensor views are consistent with the FusionGroupSpec regions.

    Catches drift between BlitzDecodeWeights layout and the FusionGroupSpec used for fingerprinting.
    Skipped when ``spec.regions`` is empty (e.g. test-only specs).
    """
    if not spec.regions:
        return
    for region in spec.regions:
        for st in region.subtensors:
            view = views.get(st.name)
            if view is None:
                raise AssertionError(
                    f"FusionGroupSpec {spec.name!r} declares subtensor {st.name!r} "
                    f"but it is missing from produced views (got {sorted(views.keys())})"
                )
            if tuple(view.tensor_shape) != tuple(st.tensor_shape):
                raise AssertionError(
                    f"FusionGroupSpec {spec.name!r} subtensor {st.name!r}: "
                    f"tensor_shape mismatch: spec={st.tensor_shape}, view={view.tensor_shape}"
                )
            if view.dtype != st.dtype:
                raise AssertionError(
                    f"FusionGroupSpec {spec.name!r} subtensor {st.name!r}: "
                    f"dtype mismatch: spec={st.dtype}, view={view.dtype}"
                )
            if tuple(view.tile_shape) != tuple(st.tile_shape):
                raise AssertionError(
                    f"FusionGroupSpec {spec.name!r} subtensor {st.name!r}: "
                    f"tile_shape mismatch: spec={st.tile_shape}, view={view.tile_shape}"
                )


def create_overlapped_tensor(
    spec: FusionGroupSpec,
    preprocessed: dict[str, torch.Tensor],
    device,
    *,
    mesh_mapper=None,
    move_to_device: bool = True,
) -> tuple[ttnn.Tensor, dict[str, OverlappedTensor]]:
    """Pack preprocessed tensors into one fused buffer and logical views per ``spec``.

    Args:
        spec: Fusion layout (name must be a supported DeepSeek group).
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
    creator = _FUSION_GROUP_CREATORS.get(name)
    if creator is None:
        supported = ", ".join(sorted(_FUSION_GROUP_CREATORS))
        raise ValueError(f"Unsupported FusionGroupSpec.name: {name!r}. Supported names: {supported}")
    fused, views = creator(preprocessed, device, move_to_device)

    _validate_views_match_spec(spec, views)
    return fused, views


def _fused_and_dict(ots: list, keys: list[str]) -> tuple[ttnn.Tensor, dict]:
    fused = ots[0].fused_tensor
    return fused, {k: v for k, v in zip(keys, ots, strict=True)}


@_register_fusion_group("q_ab_kv_a")
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


@_register_fusion_group("o_proj_gate_mm_norms")
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


@_register_fusion_group("kv_b12")
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


@_register_fusion_group("gate_up")
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
    # TODO: Refactor BlitzDecodeWeights.get_tt_moe_shared_expert_weights to accept gate+up
    # without requiring down_proj, so this dummy tensor is unnecessary. The dummy zeros are safe
    # today only because down_proj is independently sharded (not fused with gate/up in the same
    # OverlappedTensor) — its values do not affect gate/up layout or shard placement.
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
