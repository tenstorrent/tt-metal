# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tensor-parallel (TP) configuration and primitives for ACE-Step v1.5 on multi-chip meshes.

**Default OFF.** When TP is disabled (single chip, or ``ACE_STEP_TP`` unset / ``0`` / ``off``),
every helper here is a pass-through that preserves the existing *replicate-everything*
behaviour. Nothing in the model changes until TP is explicitly turned on AND a call site opts
in by asking for a shard mapper / collective.

Convention mirrors ``models/tt_transformers`` on a 2-D pod (e.g. BH_QB ``cluster_shape=(2,2)``):
    - **column-parallel** weights (Q/K/V, gate/up, patch-embed): shard the OUTPUT-feature dim
      across the TP mesh axis; each chip holds ``out // tp`` features. No reduce needed after
      the matmul — the partial output is consumed head/feature-local.
    - **row-parallel** weights (attn output proj ``wo``, MLP ``w2``/down): shard the INPUT-feature
      dim across the TP mesh axis; each chip produces a partial sum that must be **all-reduced**.

tt_transformers shards heads / w1w3-N across ``cluster_shape[1]`` (cols) and the k-dim across
``cluster_shape[0]`` (rows). We default the TP axis to **cols** (axis 1) so ACE-Step lines up
with that stack; override with ``ACE_STEP_TP_AXIS``.

Environment:
    ACE_STEP_TP        off|0 (default) | on|1 | auto   — enable TP; ``auto`` = on iff mesh>1 chip
    ACE_STEP_TP_AXIS   0|1 (default 1 = cols)          — which mesh axis carries the TP shards

NOTE: the collective wrappers below are written against the ttnn API verified present in this
build (``ttnn.all_reduce`` / ``ttnn.all_gather`` / ``ShardTensor2dMesh`` / ``ConcatMesh2dToTensor``).
Exact keyword signatures for the collectives must still be confirmed on-device — see
``docs/TP4_PLAN.md`` (validation gate G0). Until then they are used only when TP is explicitly on.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Tuple

import torch

import ttnn


# ---------------------------------------------------------------------------
# Enable / config resolution
# ---------------------------------------------------------------------------


def _num_chips(mesh_device: Any) -> int:
    if mesh_device is not None and hasattr(mesh_device, "get_num_devices"):
        try:
            return int(mesh_device.get_num_devices())
        except Exception:
            return 1
    return 1


def _mesh_shape(mesh_device: Any) -> Tuple[int, int]:
    if mesh_device is not None and hasattr(mesh_device, "shape"):
        shp = tuple(int(x) for x in mesh_device.shape)
        if len(shp) == 2:
            return (shp[0], shp[1])
        if len(shp) == 1:
            return (1, shp[0])
    n = _num_chips(mesh_device)
    return (1, n)


def _env_tp_axis() -> int:
    raw = os.environ.get("ACE_STEP_TP_AXIS", "1").strip()
    return 0 if raw == "0" else 1


_TP_ON_VALUES = ("on", "1", "true", "yes", "auto")
_TP_FULL_VALUES = ("4", "full", "all")  # 4-way: shard one dim across ALL chips (cluster_axis=None)


def ace_step_tp_env_requested() -> bool:
    """Env-only TP request check (no device needed) — used to decide fabric-enable *before* the
    mesh is opened. Must agree with :func:`ace_step_tp_enabled` modulo the multi-chip check."""
    return os.environ.get("ACE_STEP_TP", "").strip().lower() in (_TP_ON_VALUES + _TP_FULL_VALUES)


def ace_step_tp_full_requested() -> bool:
    """True when 4-way (all-chip) TP is requested (``ACE_STEP_TP=4|full|all``)."""
    return os.environ.get("ACE_STEP_TP", "").strip().lower() in _TP_FULL_VALUES


def ace_step_tp_enabled(mesh_device: Any) -> bool:
    """True iff TP is switched on for this device.

    ``ACE_STEP_TP`` = ``on``/``1``/``auto`` → per-axis TP (degree = one mesh axis);
    ``4``/``full``/``all`` → shard across ALL chips (degree = num_chips). Requires >1 chip.
    """
    raw = os.environ.get("ACE_STEP_TP", "").strip().lower()
    if _num_chips(mesh_device) <= 1:
        return False
    return raw in (_TP_ON_VALUES + _TP_FULL_VALUES)


@dataclass(frozen=True)
class TPConfig:
    """Resolved TP layout for a mesh device."""

    enabled: bool
    axis: int  # mesh axis carrying the TP shards (0=rows, 1=cols); ignored when full
    rows: int
    cols: int
    full: bool = False  # 4-way: shard one dim across ALL chips (collectives use all devices)

    @property
    def degree(self) -> int:
        """Number of TP shards. Full → all chips; else the size of the TP mesh axis."""
        if not self.enabled:
            return 1
        if self.full:
            return self.rows * self.cols
        return self.cols if self.axis == 1 else self.rows

    @property
    def mesh_shape(self) -> Tuple[int, int]:
        return (self.rows, self.cols)


def resolve_tp_config(mesh_device: Any) -> TPConfig:
    rows, cols = _mesh_shape(mesh_device)
    enabled = ace_step_tp_enabled(mesh_device)
    full = enabled and ace_step_tp_full_requested()
    axis = _env_tp_axis()
    return TPConfig(enabled=enabled, axis=axis, rows=rows, cols=cols, full=full)


# ---------------------------------------------------------------------------
# Weight mesh mappers (shard vs replicate)
# ---------------------------------------------------------------------------


def _replicate_mapper(mesh_device: Any) -> Any | None:
    """Replicate a host tensor to every chip. Mirrors tt_device.ace_step_replicate_mesh_mapper
    (kept here to avoid an import cycle)."""
    if mesh_device is None or not isinstance(mesh_device, ttnn.MeshDevice):
        return None
    if _num_chips(mesh_device) <= 1:
        return None
    rows, cols = _mesh_shape(mesh_device)
    if rows > 1 and cols > 1 and hasattr(ttnn, "ShardTensor2dMesh"):
        # (None, None) on a 2-D mesh = replicate (see tt_device notes: ReplicateTensorToMesh
        # can stall on large Blackhole uploads).
        return ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=(rows, cols))
    return ttnn.ReplicateTensorToMesh(mesh_device)


def tp_weight_mesh_mapper(mesh_device: Any, *, shard_dim: int, cfg: TPConfig | None = None) -> Any | None:
    """Mesh mapper that shards a weight along ``shard_dim`` across the TP mesh axis.

    Returns the *replicate* mapper when TP is disabled, so a call site can unconditionally route
    its weight uploads through this and get legacy behaviour for free until TP is switched on.

    - column-parallel:  shard the output-feature dim  → pass that dim as ``shard_dim``
    - row-parallel:      shard the input-feature dim   → pass that dim as ``shard_dim``
      (the row-parallel matmul output then needs :func:`tp_all_reduce`).
    """
    cfg = cfg or resolve_tp_config(mesh_device)
    if not cfg.enabled or cfg.degree <= 1:
        return _replicate_mapper(mesh_device)
    if cfg.full:
        # 4-way: split `shard_dim` across ALL chips (mesh linearised to num_chips devices).
        return ttnn.ShardTensorToMesh(mesh_device, dim=shard_dim)
    if not hasattr(ttnn, "ShardTensor2dMesh"):
        raise RuntimeError("ttnn.ShardTensor2dMesh unavailable; cannot build TP weight mapper")
    # Shard `shard_dim` across the TP axis, replicate across the other mesh axis.
    dims = (None, shard_dim) if cfg.axis == 1 else (shard_dim, None)
    return ttnn.ShardTensor2dMesh(mesh_device, dims=dims, mesh_shape=cfg.mesh_shape)


# ---------------------------------------------------------------------------
# Collectives (no-op when TP is off)
# ---------------------------------------------------------------------------


def tp_all_reduce(tensor: "ttnn.Tensor", mesh_device: Any, *, cfg: TPConfig | None = None) -> "ttnn.Tensor":
    """Sum partial outputs of a row-parallel matmul across the TP shards.

    Pass-through when TP is off. Signature confirmed on-device at gate G0 (docs/TP4_PLAN.md).
    """
    cfg = cfg or resolve_tp_config(mesh_device)
    if not cfg.enabled or cfg.degree <= 1:
        return tensor
    # Sum across the TP shards; full → all devices (cluster_axis omitted), else the TP axis.
    if cfg.full:
        return ttnn.all_reduce(tensor)
    return ttnn.all_reduce(tensor, cluster_axis=cfg.axis)


def tp_all_gather(tensor: "ttnn.Tensor", mesh_device: Any, *, dim: int, cfg: TPConfig | None = None) -> "ttnn.Tensor":
    """Gather a column-parallel sharded tensor back to a per-axis-replicated full tensor along
    tensor ``dim``. Pass-through when TP is off."""
    cfg = cfg or resolve_tp_config(mesh_device)
    if not cfg.enabled or cfg.degree <= 1:
        return tensor
    # `dim` is a required positional; full → all devices (cluster_axis omitted), else the TP axis.
    if cfg.full:
        return ttnn.all_gather(tensor, dim)
    return ttnn.all_gather(tensor, dim, cluster_axis=cfg.axis)


# ---------------------------------------------------------------------------
# Readback of a sharded tensor to host torch
# ---------------------------------------------------------------------------


def tp_read_replicated_to_torch(tensor: "ttnn.Tensor", *, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Read a mesh-replicated tensor (e.g. a row-parallel matmul output *after* :func:`tp_all_reduce`)
    to host by taking device-0's shard.

    Validated in Phase 2: ``to_torch_auto_compose`` mis-infers the post-CCL topology of an
    all-reduced tensor (``TT_FATAL: dims must be unique``). Since every chip holds the same data,
    device-0's shard is the whole answer.
    """
    out = ttnn.to_torch(ttnn.get_device_tensors(tensor)[0])
    if dtype is not None and out.dtype != dtype:
        out = out.to(dtype)
    return out.contiguous()


def tp_read_sharded_to_torch(
    tensor: "ttnn.Tensor",
    mesh_device: Any,
    *,
    shard_dim: int,
    dtype: torch.dtype | None = None,
    cfg: TPConfig | None = None,
) -> torch.Tensor:
    """Read a TP-sharded tensor back to host, reassembling the sharded ``shard_dim``.

    When TP is off this defers to the standard replicate readback. When on, it gathers the
    shards on-device first (so the existing replicate-aware readback stays valid), avoiding the
    silent-corruption trap of composing a sharded tensor with ``to_torch_auto_compose``.
    """
    from models.experimental.ace_step_v1_5.utils.tt_device import ace_step_ttnn_to_torch

    cfg = cfg or resolve_tp_config(mesh_device)
    if cfg.enabled and cfg.degree > 1:
        tensor = tp_all_gather(tensor, mesh_device, dim=shard_dim, cfg=cfg)
    return ace_step_ttnn_to_torch(tensor, dtype=dtype, mesh_device=mesh_device)
