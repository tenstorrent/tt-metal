# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Pack preprocessed torch tensors into a fused device buffer and OverlappedTensor views.

Reads the declarative :class:`FusionGroupSpec` to build :class:`OverlapEntry`
objects, then delegates to :func:`overlap_tensors`.  The spec's
:attr:`~FusionGroupSpec.per_core` flag is forwarded so a group that needs
per-core (non-lockstep) allocation can opt in without authoring a custom
pipeline.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import torch

import ttnn
from models.demos.deepseek_v3_b1.weights.cache.types import FusionGroupSpec
from models.demos.deepseek_v3_b1.weights.overlap.packing import OverlapEntry, overlap_tensors

if TYPE_CHECKING:
    from models.demos.deepseek_v3_b1.weights.overlap.packing import OverlappedTensor


def _validate_views_match_spec(spec: FusionGroupSpec, views: dict[str, "OverlappedTensor"]) -> None:
    """Assert produced OverlappedTensor views are consistent with the FusionGroupSpec layout.

    Catches drift between the preprocessing output and the FusionGroupSpec used for fingerprinting.
    Skipped when the spec declares no regions (e.g. test-only specs).

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
    """Pack preprocessed tensors into one fused buffer and its logical views.

    ``spec.per_core`` controls whether the buffer is allocated with the
    global/lockstep allocator or with
    :meth:`ttnn.MemoryConfig.experimental_set_per_core_allocation`.

    Args:
        spec: Fusion layout describing the regions of this group.
        preprocessed: Mapping from sub-tensor name to fully-preprocessed 2-D
            torch tensor (shuffled, TP-concatenated, block-resharded — ready
            for tilization).
        device: Mesh or single device for placement.
        move_to_device: If True, the fused tensor is placed on ``device``;
            if False, a host tensor with mesh metadata (for cache store).

    Returns:
        ``(fused, views)`` — the fused :class:`ttnn.Tensor` and a mapping
        from logical sub-tensor name to :class:`~ttnn.OverlappedTensor`.
    """
    if not spec.regions:
        raise ValueError(f"FusionGroupSpec {spec.name!r} declares no regions")

    entries: list[OverlapEntry] = []
    for region in spec.regions:
        for st in region.subtensors:
            tensor = preprocessed.get(st.name)
            if tensor is None:
                raise KeyError(
                    f"preprocessed missing {st.name!r} for FusionGroupSpec {spec.name!r} "
                    f"(available keys: {sorted(preprocessed.keys())})"
                )
            entries.append(
                OverlapEntry(
                    st.name,
                    tensor,
                    replace(st, raw_tensor_shape=tuple(tensor.shape)),
                )
            )

    views = overlap_tensors(
        entries,
        device,
        move_to_device=move_to_device,
        per_core=spec.per_core,
    )
    fused = next(iter(views.values())).fused_tensor
    _validate_views_match_spec(spec, views)
    return fused, views
