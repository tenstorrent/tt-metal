# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Pack preprocessed torch tensors into fused device buffers and OverlappedTensor views.

Reads the declarative :class:`FusionGroupSpec` regions to build
:class:`OverlapEntry` objects, then delegates to :func:`overlap_tensors`.
No per-group dispatch — the spec *is* the layout recipe.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import torch

import ttnn
from models.demos.deepseek_v3_b1.blitz_overlap_tensors import OverlapEntry, overlap_tensors
from models.demos.deepseek_v3_b1.tensor_cache.types import FusionGroupSpec

if TYPE_CHECKING:
    from models.demos.deepseek_v3_b1.blitz_overlap_tensors import OverlappedTensor


def _validate_views_match_spec(spec: FusionGroupSpec, views: dict[str, OverlappedTensor]) -> None:
    """Assert produced OverlappedTensor views are consistent with the FusionGroupSpec regions.

    Catches drift between the preprocessing output and the FusionGroupSpec used for fingerprinting.
    Skipped when ``spec.regions`` is empty (e.g. test-only specs).

    Note: ``raw_tensor_shape`` is NOT validated here because
    ``create_overlapped_tensor`` intentionally overrides it with the
    actual preprocessed tensor shape, which may differ from the spec's
    single-device default due to TP expansion or shuffle reshaping.
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
            if view.dtype != st.dtype:
                raise AssertionError(
                    f"FusionGroupSpec {spec.name!r} subtensor {st.name!r}: "
                    f"dtype mismatch: spec={st.dtype}, view={view.dtype}"
                )
            if tuple(view.tile_shape) != (st.tile_h, st.tile_w):
                raise AssertionError(
                    f"FusionGroupSpec {spec.name!r} subtensor {st.name!r}: "
                    f"tile_shape mismatch: spec={(st.tile_h, st.tile_w)}, view={view.tile_shape}"
                )


def create_overlapped_tensor(
    spec: FusionGroupSpec,
    preprocessed: dict[str, torch.Tensor],
    device,
    *,
    move_to_device: bool = True,
) -> tuple[ttnn.Tensor, dict[str, "OverlappedTensor"]]:
    """Pack preprocessed tensors into one fused buffer and logical views per ``spec``.

    Reads ``spec.regions`` to build :class:`OverlapEntry` objects from the
    ``preprocessed`` tensors, then calls :func:`overlap_tensors`.  The
    subtensor's ``raw_tensor_shape`` is updated to match the actual tensor
    shape (which may differ from the spec's default due to TP expansion).

    Args:
        spec: Fusion layout whose regions describe lanes, core ranges, dtypes,
            and tile shapes.
        preprocessed: Mapping from sub-tensor name to fully-preprocessed 2-D
            torch tensor (shuffled, TP-concatenated, block-resharded — ready
            for tilization).
        device: Mesh or single device for placement.
        move_to_device: If True, fused tensor is placed on ``device``; if False,
            host tensor with mesh metadata (for cache store).

    Returns:
        ``(fused_tensor, views)`` where ``views`` maps logical name to
        :class:`OverlappedTensor`.
    """
    lanes: list[list[OverlapEntry]] = []
    for region in spec.regions:
        lane: list[OverlapEntry] = []
        for st in region.subtensors:
            tensor = preprocessed.get(st.name)
            if tensor is None:
                raise KeyError(
                    f"preprocessed missing {st.name!r} for FusionGroupSpec {spec.name!r} "
                    f"(available keys: {sorted(preprocessed.keys())})"
                )
            lane.append(
                OverlapEntry(
                    st.name,
                    tensor,
                    replace(st, raw_tensor_shape=tuple(tensor.shape)),
                )
            )
        lanes.append(lane)

    views = overlap_tensors(lanes, device, move_to_device=move_to_device)
    _validate_views_match_spec(spec, views)
    fused = next(iter(views.values())).fused_tensor
    return fused, views
