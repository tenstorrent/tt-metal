# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""MeshConfig — prefill mesh parallelization for Llama-3.1-8B.

TP shards features on one axis (default cols); the other axis (rows) carries SP prefill (SP = #rows).
Both derive from mesh shape + tp_axis, so TP is the only knob.

Target: one 4x8 Blackhole Galaxy, TP=8, SP=4.

Why TP=8 specifically: ``NUM_KEY_VALUE_HEADS = 8`` over 8 columns puts **exactly one KV head on each
chip**, so a KV chunk lives on exactly one chip and the migration layer's DeviceGroup degenerates to
a single node — the cheapest address table this framework can express. TP=8 also divides every width
cleanly: 4096/8 = 512 (16 tiles) and 14336/8 = 1792 (56 tiles), so none of the tile-alignment padding
other models in this fleet carry is needed here.

Adapted from ``gpt_oss_d_p/tt/config.py``, minus the expert-parallel axis (Llama is dense). The
collective helpers are deliberately left out for now — they land with the op that first needs them.

**There is no sequence-parallel mesh mapper here, on purpose.** SP distribution is not a
``ShardTensor2dMesh`` placement: the host reshuffles each chunk into device-major order before it
goes over the H2D socket, and the KV cache is filled block-cyclically by the write kernel. A mapper
that sharded a tensor dim contiguously across the SP rows would look right and place data wrong, so
none is offered. ``gpt_oss_d_p`` has a ``sequence_parallel`` helper, but it shards dim -3 across the
TP axis and its only caller uses it for per-head attention sinks, which Llama does not have.
"""

from loguru import logger

import ttnn
from models.demos.llama_3p1_8b_d_p.tt.prefill_geometry import PREFILL_LAYOUT as layout


class MeshConfig:
    """Prefill mesh parallelization. TP is the only knob; SP follows from the mesh shape."""

    def __init__(self, mesh_shape, tp, tp_axis: int = layout.tp_axis):
        """
        Args:
            mesh_shape: (rows, cols) - any mesh size
            tp: tensor-parallel size (shards features along tp_axis)
            tp_axis: which mesh axis is TP (0=rows, 1=cols, default: 1). The other axis
                carries sequence-parallel prefill (SP = size of that axis).
        """
        self.mesh_shape = tuple(mesh_shape)
        # Validate the shape and the axis BEFORE deriving anything from them. Deriving first lets
        # a bad axis build plausible-looking sharding metadata instead of failing here: tp_axis=-1
        # indexes mesh_shape fine and passes the TP check below, but `1 - tp_axis` would then put
        # SP on axis 2, and anything outside {0, 1} only surfaces later as an unrelated IndexError.
        if len(self.mesh_shape) != 2:
            raise ValueError(f"mesh_shape must be 2-D (rows, cols); got {self.mesh_shape}")
        if tp_axis not in (0, 1):
            raise ValueError(
                f"tp_axis must be 0 (rows) or 1 (cols); got {tp_axis!r}. Negative indices are rejected so TP and SP cannot resolve to the same axis."
            )
        self.tp = tp
        self.tp_axis = tp_axis
        self.sp_axis = 1 - tp_axis
        self.total_devices = self.mesh_shape[0] * self.mesh_shape[1]
        self._validate()

    def _validate(self):
        tp_dim_size = self.mesh_shape[self.tp_axis]
        # shard_mapper always shards a tensor across the ENTIRE tp_axis, so TP must span the whole
        # axis. A smaller TP would build head/feature counts from `tp` while the mapper still splits
        # across all `tp_dim_size` devices, giving inconsistent per-device shapes.
        if self.tp != tp_dim_size:
            raise ValueError(
                f"TP({self.tp}) must equal mesh_{self.tp_axis}_size({tp_dim_size}); "
                f"sub-axis TP is unsupported (shard_mapper shards the full axis)."
            )
        if (self.mesh_shape, self.tp) != (layout.mesh_shape, layout.tp):
            logger.warning(
                f"MeshConfig(mesh_shape={self.mesh_shape}, tp={self.tp}) is untested; only "
                f"mesh_shape={layout.mesh_shape}, tp={layout.tp} (SP=4) is the Llama-3.1-8B target."
            )

    @property
    def sp(self) -> int:
        """Sequence-parallel degree (size of the non-TP axis)."""
        return self.mesh_shape[self.sp_axis]

    def shard_mapper(self, mesh_device, tensor_dim=None, mesh_dims=None):
        """Unified 2D sharding - replaces all individual mappers."""
        if mesh_dims is None:
            # Default: shard along the TP axis, replicate along the other. Built by position rather
            # than a `tp_axis == 1` conditional so the two stay consistent if the axis ever flips.
            # ShardTensor2dMesh reads dims[0] as the placement on mesh rows and dims[1] on cols.
            dims = [None, None]
            dims[self.tp_axis] = tensor_dim
            mesh_dims = tuple(dims)

        return ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, dims=mesh_dims)

    def column_parallel(self, mesh_device):
        """Column-parallel weights (feature dimension sharding) — qkv_proj, gate_proj, up_proj."""
        return self.shard_mapper(mesh_device, tensor_dim=-1)

    def row_parallel(self, mesh_device):
        """Row-parallel weights (input dimension sharding) — o_proj, down_proj."""
        return self.shard_mapper(mesh_device, tensor_dim=-2)

    def shard_size(self, total_size):
        """Size per device for tensor parallel sharding."""
        return total_size // self.tp

    def __repr__(self):
        return f"MeshConfig({self.mesh_shape}, tp={self.tp}, sp={self.sp}, tp_axis={self.tp_axis})"
