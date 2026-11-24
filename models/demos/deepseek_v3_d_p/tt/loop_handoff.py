# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.4 loop handoff: the per-chunk state loop pass 1 (stage G1) forwards to loop pass 2 (stage G2) beside the
activation.

* KVPE rows (Q2.1). Every loop layer's pass-1 KV latent, which pass-2 attention reads as bank A. Each chip forwards
  the ``chunk_size_global / (sp * tp)`` packed cache rows it stored for the chunk, read back from its cache
  (``read_padded_kv_cache``), and G2 writes them into its copy at the same offsets with ``update_padded_kv_cache`` in
  complete-mesh mode, whose row-major chip order is the ``sp * tp`` linearization G1 wrote with. The copy is bit-exact
  by construction and the payload is the stored rows, not the 640-row TP-replicated fields the write consumed (~4x).
  On the wire the rows travel packed, ``KVPE_ROWS_PER_PAGE`` consecutive rows per D2D page: the D2D relay's cost
  follows its page count, and one cache row is a small page (q171.3.20: the whole 5120-token handoff 9.89 -> 5.78 ms
  on a 4-cable relay proxy). Packing is a row-major regrouping (same bytes, same order), undone before the write.
* Top-k sets (Q2.2). The indices of every full indexer layer of the loop, one segment per set; pass-2 layer ``L``
  reuses the set of pass-1 layer ``L``'s reuse group, so G2 runs no indexer.

Both ride as extra segments of the boundary's activation D2D service (``extra_tensors``), in :meth:`LoopHandoff.
segments` order, so the boundary has one service. Everything per chunk is read on device from the chunk's metadata
tensors (slot, start, end), so collect and write are trace-safe once their buffers are persistent.
"""

from dataclasses import dataclass
from typing import Sequence

import ttnn
from models.demos.common.prefill.runners.runner_utils import axis_mapper_config, indexer_indices_global_spec

# Cache rows per D2D page of the KVPE segment (q171.3.20): the smallest packing at the measured plateau (16, 32 and
# 160 rows per page moved the handoff equally fast); it divides a chip's rows per layer.
KVPE_ROWS_PER_PAGE = 16


@dataclass(frozen=True)
class LoopHandoff:
    """The handoff's shape on an ``sp x tp`` stage mesh with SP on mesh axis 0.

    ``kv_layers`` cache layers starting at the stage cache's ``first_kv_layer`` (of ``num_kv_layers`` per slot) are
    forwarded; ``index_shape`` is one top-k set's per-chip shape and ``num_index_sets`` how many sets travel."""

    mesh_shape: tuple
    chunk_size_global: int
    kv_layers: int
    kv_width: int
    kv_dtype: object
    index_shape: tuple
    num_index_sets: int

    def __post_init__(self):
        sp, tp = self.mesh_shape
        assert (
            self.chunk_size_global % (sp * tp * ttnn.TILE_SIZE) == 0
        ), f"chunk {self.chunk_size_global} must give each of the {sp * tp} chips whole 32-row tiles of the cache"
        assert (
            self.chunk_rows % KVPE_ROWS_PER_PAGE == 0
        ), f"{self.chunk_rows} cache rows per chip per layer do not pack into pages of {KVPE_ROWS_PER_PAGE} rows"

    @property
    def kv_wire_dtype(self):
        """The KVPE segment's dtype: the cache's, except FP8 rows travel as their bytes (UINT8), since FP8 has no D2D
        stream path; read_padded_kv_cache returns them so and update_padded_kv_cache takes them so."""
        return ttnn.uint8 if self.kv_dtype == ttnn.fp8_e4m3 else self.kv_dtype

    @property
    def chunk_rows(self) -> int:
        """Cache rows one chip stores per layer per chunk."""
        sp, tp = self.mesh_shape
        return self.chunk_size_global // (sp * tp)

    def kvpe_rows_shape(self) -> list:
        """One chip's KVPE rows of the chunk as the cache stores them: [1, 1, kv_layers * chunk_rows, row]."""
        return [1, 1, self.kv_layers * self.chunk_rows, self.kv_width]

    def kvpe_packed_shape(self) -> list:
        """The same bytes as they travel: KVPE_ROWS_PER_PAGE consecutive rows per page."""
        return [1, 1, self.kv_layers * self.chunk_rows // KVPE_ROWS_PER_PAGE, KVPE_ROWS_PER_PAGE * self.kv_width]

    def kvpe_spec(self) -> ttnn.TensorSpec:
        """Global spec of the KVPE segment: each chip's packed block (kvpe_packed_shape), laid out like the index
        segments (rows over SP, width over TP), so chip (sp, tp) sends its own block to chip (sp, tp)."""
        sp, tp = self.mesh_shape
        _, _, pages, page = self.kvpe_packed_shape()
        return ttnn.TensorSpec(
            shape=ttnn.Shape([1, 1, sp * pages, tp * page]),
            dtype=self.kv_wire_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            buffer_type=ttnn.BufferType.DRAM,
        )

    def segments(self) -> list:
        """``(global spec, mapper config)`` of each segment, in wire order: the KVPE rows, then the top-k sets.
        The index sets travel as today's one-set index segment does: rows over SP, top-k over TP."""
        mapper = self.index_mapper_config()
        index = (indexer_indices_global_spec(self.index_shape, self.mesh_shape), mapper)
        return [(self.kvpe_spec(), mapper)] + [index] * self.num_index_sets

    @staticmethod
    def index_mapper_config() -> ttnn.MeshMapperConfig:
        return axis_mapper_config(ttnn.PlacementShard(2), ttnn.PlacementShard(3), sp_axis=0)


def _free_reshaped(reshaped: ttnn.Tensor, source: ttnn.Tensor) -> None:
    """Free a reshape's output unless it is a view of ``source``'s buffer (a caller's persistent buffer)."""
    if reshaped.buffer_address() != source.buffer_address():
        ttnn.deallocate(reshaped)


def _check_mesh(mesh_device, handoff: LoopHandoff) -> None:
    shape = tuple(mesh_device.shape)
    assert shape == tuple(handoff.mesh_shape), f"mesh {shape} != handoff mesh {handoff.mesh_shape} (SP on axis 0)"


class LoopHandoffSender:
    """G1 side: the KVPE rows buffer the handoff is collected into and the packed buffer it is sent from, with the
    top-k sets ``index_sets`` (the model's persistent sets, which pass 1 fills as it makes them, or a caller's copies),
    at fixed addresses for trace. G2 receives into the same layout: a sender's segments with G2's model sets."""

    def __init__(
        self,
        mesh_device,
        handoff: LoopHandoff,
        *,
        index_sets: Sequence[ttnn.Tensor],
        first_kv_layer: int,
        num_kv_layers: int,
    ):
        _check_mesh(mesh_device, handoff)
        assert (
            len(index_sets) == handoff.num_index_sets
        ), f"{len(index_sets)} top-k sets, the handoff carries {handoff.num_index_sets}"
        self.handoff = handoff
        self.first_kv_layer = first_kv_layer
        self.num_kv_layers = num_kv_layers
        spec = lambda shape: ttnn.TensorSpec(
            shape=ttnn.Shape(shape),
            dtype=handoff.kv_wire_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            buffer_type=ttnn.BufferType.DRAM,
        )
        self.kvpe = ttnn.allocate_tensor_on_device(spec(handoff.kvpe_rows_shape()), mesh_device)
        self.kvpe_packed = ttnn.allocate_tensor_on_device(spec(handoff.kvpe_packed_shape()), mesh_device)
        self.index_sets = list(index_sets)

    def collect(self, kvpe_cache: ttnn.Tensor, metadata: Sequence[ttnn.Tensor]):
        """Gather this chunk's KVPE rows after every loop layer has written its KV: the rows each chip stored at the
        chunk's (slot, start) in ``metadata`` (the same tensors the writes read), packed for the wire. The top-k sets
        are filled by pass 1 as it makes them. Device-only, so it may be captured."""
        self.kvpe.update_tensor_topology(kvpe_cache.tensor_topology())
        ttnn.experimental.deepseek_prefill.read_padded_kv_cache(
            kvpe_cache,
            metadata[0],
            metadata[1],
            first_layer=self.first_kv_layer,
            num_read_layers=self.handoff.kv_layers,
            num_layers=self.num_kv_layers,
            chunk_size_global=self.handoff.chunk_size_global,
            # The stage cache is written SP on mesh axis 0, TP-deduped on axis 1 (the mesh contract above).
            cluster_axis=0,
            tp_axis=1,
            output=self.kvpe,
        )
        packed = ttnn.reshape(self.kvpe, self.handoff.kvpe_packed_shape())
        ttnn.copy(packed, self.kvpe_packed)
        _free_reshaped(packed, self.kvpe)

    def segments(self) -> list:
        """The tensors to send after the activation, in :meth:`LoopHandoff.segments` order."""
        return [self.kvpe_packed, *self.index_sets]


def write_kvpe_copy(
    kvpe_cache: ttnn.Tensor,
    kvpe_rows: ttnn.Tensor,
    metadata: Sequence[ttnn.Tensor],
    handoff: LoopHandoff,
    *,
    first_kv_layer: int,
    num_kv_layers: int,
) -> None:
    """G2 side: write the received (packed) KVPE segment into the stage's copy of the loop layers' cache (bank A), at
    the chunk's (slot, start, end) in ``metadata``. Same per-chip offsets and end clamp as G1's write, so every row G1
    stored for the request lands bit-exact; rows past the chunk's real tokens are left as they were."""
    packed = kvpe_rows
    kvpe_rows = ttnn.reshape(packed, handoff.kvpe_rows_shape())  # unpack: one cache row per page again
    # Each chip holds its own rows of the chunk, distributed like the cache's sequence dim (row-major over the mesh);
    # complete-mesh mode checks that topology, which the wire mapping (rows over SP, width over TP) does not carry.
    kvpe_rows.update_tensor_topology(kvpe_cache.tensor_topology())
    rows = handoff.chunk_rows
    for layer in range(handoff.kv_layers):
        ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
            kvpe_cache,
            kvpe_rows,
            metadata[0],
            metadata[1],
            layer_idx=first_kv_layer + layer,
            num_layers=num_kv_layers,
            cluster_axis=None,
            valid_global=metadata[2],
            input_row_offset=layer * rows,
            input_rows=rows,
        )
    _free_reshaped(kvpe_rows, packed)
