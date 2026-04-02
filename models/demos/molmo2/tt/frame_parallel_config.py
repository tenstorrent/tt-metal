# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Environment-driven milestones for Molmo2 **frame-level vision data parallelism** on a mesh.

Frame-level DP: each mesh device runs the ViT on a disjoint subset of frames/crops per
parallel round (e.g. 32 frames on 8 devices → 4 rounds × 8-way parallel). Rounds are
concatenated so downstream pooling and the LM see the same token order as a single
replicated ViT run.

Milestones (``MOLMO2_VISION_PARALLEL_MILESTONE``)
-----------------------------------------------
``0`` / ``legacy`` / ``off``
    Disable mesh frame splitting. Multi-frame ViT uses the legacy path (replicated
    weights; crops processed sequentially on the mesh).

``1`` / ``on`` / ``frame_dp`` (default when unset)
    Enable frame-level data parallelism on ``MeshDevice`` when ``num_crops > 1``.

``2`` / ``aligned``
    Same as ``1``, and the demo **also** aligns ``max_frames`` to a multiple of the
    host device count before extraction (even per-device rounds, no replicated tail).

Additional variables
--------------------
``MOLMO2_FRAME_DP_LOG_ROUNDS``
    If truthy, log each ViT parallel round (frame index range and chunk size).

``MOLMO2_FRAME_DP_LOG_DEVICE_FRAMES``
    If truthy, log how each sampled frame maps to mesh shard slot and physical device ID
    (sharded rounds, partial gather, and replicated tail).

``MOLMO2_VIDEO_ALIGN_FRAMES``
    If truthy, align max video frames to ``len(ttnn.get_device_ids())`` (same effect
    as milestone 2 for alignment only; can combine with milestone 0 to only align).

``MOLMO2_VIDEO_MAX_FRAMES``
    If set, the demo uses this as the frame cap before alignment (overrides CLI default flow
    when the demo applies this in ``effective_video_max_frames``).

``MOLMO2_FRAME_DP_REMAINDER``
    How to run ViT frame-parallel rounds when the frame count is **not** divisible by the mesh
    device count:

    - ``tail`` (default): after full ``D``-way rounds, process leftover frames one-by-one with
      replication (slower tail, no extra compute on dummy frames).
    - ``pad``: zero-pad the batch to the next multiple of ``D``, run only full rounds, then
      strip padded tokens from the sequence before returning (equal device utilization; dummy
      ViT compute on padding).
    - ``gather`` / ``check_gather``: run the last partial round with ``ShardTensorToMesh`` using
      ``chunk = remainder`` (TTNN supports fewer shards than devices on 1D distribution); gather
      with ``ConcatMeshToTensor`` so no sequential tail and no dummy frames.
"""

from __future__ import annotations

import logging
import os

_logger = logging.getLogger(__name__)


def _truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def molmo2_vision_parallel_milestone() -> int:
    """
    Return milestone level 0 (legacy), 1 (frame DP), or 2 (frame DP + demo frame align).
    """
    raw = os.environ.get("MOLMO2_VISION_PARALLEL_MILESTONE", "").strip().lower()
    if raw in ("", "1", "m1", "on", "frame_dp", "true", "yes"):
        return 1
    if raw in ("0", "off", "false", "legacy", "no"):
        return 0
    if raw in ("2", "m2", "aligned", "align"):
        return 2
    try:
        return max(0, min(2, int(raw)))
    except ValueError:
        _logger.warning("Invalid MOLMO2_VISION_PARALLEL_MILESTONE=%r; using 1", raw)
        return 1


def vision_frame_dp_enabled() -> bool:
    """Use mesh sharding for multi-crop / multi-frame ViT when possible."""
    return molmo2_vision_parallel_milestone() >= 1


def vision_frame_dp_log_rounds() -> bool:
    return _truthy("MOLMO2_FRAME_DP_LOG_ROUNDS")


def vision_frame_dp_log_device_frames() -> bool:
    """Log per-device / per-shard frame assignment for ViT frame DP (debug)."""
    return _truthy("MOLMO2_FRAME_DP_LOG_DEVICE_FRAMES")


def video_align_frames_to_mesh_width() -> bool:
    """Trim max frame cap to a multiple of device count (demo / preprocess)."""
    return _truthy("MOLMO2_VIDEO_ALIGN_FRAMES") or molmo2_vision_parallel_milestone() >= 2


def vision_frame_dp_remainder_mode() -> str:
    """
    Return how to handle frames after the last full ``D``-way parallel round.

    Returns one of: ``"tail"``, ``"pad"``, ``"gather"``.
    """
    raw = os.environ.get("MOLMO2_FRAME_DP_REMAINDER", "").strip().lower()
    if raw in ("", "tail", "replicated", "sequential", "default"):
        return "tail"
    if raw in ("pad", "padding", "padded", "zero_pad"):
        return "pad"
    if raw in ("gather", "check_gather", "shard", "partial"):
        return "gather"
    _logger.warning("Invalid MOLMO2_FRAME_DP_REMAINDER=%r; using tail", raw)
    return "tail"
