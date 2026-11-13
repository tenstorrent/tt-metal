# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""The persistent TP12 MoE: one ``fused_experts`` launch serving many layers over D2D sockets.

The MoE blocks of every layer it serves are loaded TP12 on ``moe_mesh`` (a 3x4 submesh), each the usual
:class:`~.moe.DeepSeekV41SparseMoeBlock` with the router and shared expert fused into ``fused_experts``. The op
is launched once, on the first layer's tensors (every layer shares their layouts), and loops on the device: it
takes one ``[64-byte header | bf16 row]`` request page at a time from each requester in strict turn, the
header's first word naming a row of the layer table, runs that layer's whole MoE in its three-stage pipeline,
and every MoE chip sends its partial row straight to every chip of the requester, which sums the twelve partials
itself (``persistent_moe_request``), so a request's output is replicated over the requester's mesh.

A requester is a mesh above the MoE whose chips each send down their column to the MoE chips below: a 1x4
submesh in one of rows 1 to 4 of the 8x4 Galaxy when the MoE is the 3x4 at row 5 (the MoE chips' partials
leave them northward). See ``tests/decode/test_moe_persistent.py`` and ``tests/decode/test_decoder_layer_persistent_moe.py``.

Once launched, the MoE mesh's cores belong to the op: nothing else may run there until :meth:`PersistentMoE.stop`,
which every requester must reach (each sends one stop request) before the device closes.
"""

from typing import Mapping, Sequence

import torch
import ttnn

from models.experimental.deepseek_v4_flash.tt.common import CCL_TOPOLOGY, width_sharded_l1_config
from models.experimental.deepseek_v4_flash.tt.decode.hyperconnection import _HC_COLLAPSE_CORES

from .moe import DeepSeekV41PreloadedExperts, DeepSeekV41SparseMoeBlock

TP_SIZE = 12
STOP_LAYER = 0xFFFFFFFF
HEADER_BYTES = 64
# The worker grid and router row the op multicasts each row over: 12 x (8 + 1).
REPLICA_GRID = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 8))])
# The op's router is sharded by expert over the MoE chips: chip m scores the gate's tile column m (experts
# [32m, 32m + 32)), K-split over the router row's first ROUTER_PARTS cores, which hold each layer's slice in L1.
ROUTER_PARTS = 10
ROUTER_PART_CORES = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 8), ttnn.CoreCoord(ROUTER_PARTS - 1, 8))])


def default_socket_core(mesh) -> ttnn.CoreCoord:
    """The bottom-right worker core: outside the 12x9 grid the op multicasts over and away from its reduce
    senders (the bottom row's left end). On a requester it only holds the socket FIFOs and the request op,
    which runs between the layer's other ops."""
    grid = mesh.compute_with_storage_grid_size()
    return ttnn.CoreCoord(grid.x - 1, grid.y - 1)


def build_moe_layers(config, loader, moe_mesh, cache, layer_ids: Sequence[int], *, log=None) -> dict:
    """``{layer: DeepSeekV41SparseMoeBlock}`` at TP12 on ``moe_mesh`` for the checkpoint's ``layer_ids``.

    ``cache`` is the model's :class:`~...weight_cache.WeightCache`; each layer reads ``layers.<i>.ffn``, the
    routed experts K-split over every DRAM bank (``tools/fill_expert_cache.py --tp-size 12 --mesh-rows 3``
    fills them ahead of time)."""
    from models.experimental.deepseek_v41_flash.tt.config import expert_provider, moe_weights

    moes = {}
    for layer in layer_ids:
        layer_cache = cache.sub(f"layers.{layer}.ffn")
        experts = DeepSeekV41PreloadedExperts(
            config,
            expert_provider(loader, layer),
            moe_mesh,
            dtype=ttnn.bfloat4_b,
            cache=layer_cache,
            tp_size=TP_SIZE,
            balance_dram_banks=True,
        )
        moe = DeepSeekV41SparseMoeBlock(
            config, moe_weights(loader, layer), moe_mesh, experts, cache=layer_cache, tp_size=TP_SIZE
        )
        moes[layer] = moe
        if log is not None:
            log(layer)
    return moes


def _as_int32(values):
    return torch.tensor([v - (1 << 32) if v >= 1 << 31 else v for v in values], dtype=torch.int32)


class PersistentMoE:
    """The persistent op over ``moes`` -- ``{layer: block}`` or a list (layer = index), every block at TP12 on
    ``moe_mesh`` -- serving one requester per mesh of ``request_meshes``.

    A layer's row of the table is its rank among the sorted layer ids, so a server may hold any subset of the
    model's layers. Each requester's request socket runs from ``request_core`` on each of its chips to
    ``socket_core`` on the MoE chips of the same column, with a one-page FIFO; its partials buffer (see
    :meth:`partials_buffer`) on the hyper-connection's stream-collapse cores on every one of its chips. The sockets and
    buffers allocate L1, so build this before any trace capture on the requesters.

    ``single_senders`` (per requester, default all False) has a requester's first chip send its every request to
    all the MoE chips -- one multicast, for a requester mesh right above the MoE -- instead of each chip down its
    own column, so the MoE chips start each row together (see the sharded router).

    The op takes requests in strict turn (requester 0, 1, ..., then 0 again), so with several requesters each
    sends its next request only after the others have sent theirs. Use :meth:`request` (or :meth:`upload` and
    :meth:`request` for a host row), and :meth:`stop` once to end the kernels.
    """

    def __init__(
        self,
        moe_mesh,
        request_meshes: Sequence,
        moes,
        socket_core: ttnn.CoreCoord | None = None,
        request_core: ttnn.CoreCoord | None = None,
        single_senders: Sequence[bool] | None = None,
    ):
        if not isinstance(moes, Mapping):
            moes = dict(enumerate(moes))
        if not moes:
            raise ValueError("the persistent MoE needs at least one layer")
        self.moe_mesh = moe_mesh
        self.request_meshes = list(request_meshes)
        self.layer_ids = sorted(moes)
        self.table_rows = {layer: row for row, layer in enumerate(self.layer_ids)}
        self.socket_core = socket_core or default_socket_core(moe_mesh)
        self.request_core = request_core or default_socket_core(self.request_meshes[0])
        first = moes[self.layer_ids[0]]
        self.hidden = first.hidden
        self.row_bytes = self.hidden * 2
        experts = first.experts
        for layer in self.layer_ids:
            moe = moes[layer]
            assert moe.fused_router and moe.fused_shared is not None, f"layer {layer}: the op runs the whole block"
            assert moe.experts.tp_size == TP_SIZE, f"layer {layer}: experts at TP{moe.experts.tp_size}"
        assert experts.fused_all_reduce_axes is not None, "the persistent op reduces over the whole mesh"

        # Each layer's gate, one [D, 32] column per MoE chip in L1 on its partial cores (see ROUTER_PART_CORES).
        self.router_columns = [self.router_column(moes[layer], moe_mesh) for layer in self.layer_ids]
        rows = []
        for row, layer in enumerate(self.layer_ids):
            moe = moes[layer]
            header = [
                row,
                moe.fused_shared.gate_up.buffer_address(),
                moe.fused_shared.down.buffer_address(),
                self.router_columns[row].buffer_address(),
                moe.gate.ranking_bias().buffer_address(),
            ]
            header += [0] * (HEADER_BYTES // 4 - len(header))
            rows.append(
                _as_int32(
                    header
                    + [w.buffer_address() for w in moe.experts.gate_up_weights]
                    + [w.buffer_address() for w in moe.experts.down_weights]
                )
            )
        self.layer_table = ttnn.from_torch(
            torch.stack(rows),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=moe_mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(moe_mesh),
        )
        # The row lands here on every core of the grid, replacing the all_gather_for_matmul replica.
        self.replica = ttnn.from_torch(
            torch.zeros(1, 1, REPLICA_GRID.num_cores(), self.hidden),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=moe_mesh,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(REPLICA_GRID, [1, self.hidden], ttnn.ShardOrientation.ROW_MAJOR),
            ),
            mesh_mapper=ttnn.ReplicateTensorToMesh(moe_mesh),
        )

        moe_rows, moe_cols = tuple(moe_mesh.shape)
        self.request_tx, self.request_rx, self.partials = [], [], []
        for request_mesh in self.request_meshes:
            if tuple(request_mesh.shape)[1] != moe_cols:
                raise ValueError(f"a requester must span the MoE's {moe_cols} columns, got {tuple(request_mesh.shape)}")
            # Each MoE chip starts a row when its page arrives, and the sharded router has every chip wait for the
            # latest one, so pages sent by several requester chips start the MoE at their skew. A single sender (its
            # first chip) starts every MoE chip at once; one multicast reaches them only from the row right above.
            single = single_senders is not None and single_senders[len(self.request_tx)]
            request_connections = [
                ttnn.SocketConnection(
                    ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0 if single else c), self.request_core),
                    ttnn.MeshCoreCoord(ttnn.MeshCoordinate(r, c), self.socket_core),
                )
                for c in range(moe_cols)
                for r in range(moe_rows)
            ]
            tx, rx = ttnn.create_socket_pair(
                request_mesh,
                moe_mesh,
                ttnn.SocketConfig(
                    request_connections, ttnn.SocketMemoryConfig(ttnn.BufferType.L1, HEADER_BYTES + self.row_bytes)
                ),
            )
            self.request_tx.append(tx)
            self.request_rx.append(rx)
            self.partials.append(self.partials_buffer(request_mesh, moe_rows * moe_cols))
        self.inputs = [self.request_input(j) for j in range(len(self.request_meshes))]

        shared = first.fused_shared
        # Kept to relaunch the kernels after a stop (see :meth:`start`): the mesh is time-shared with the prefill MoE.
        self._launch = lambda: ttnn.experimental.deepseek.moe.fused_experts(
            self.replica,
            router_weight=self.router_columns[0],
            router_bias=first.gate.ranking_bias(),
            gate_up_weights=experts.gate_up_weights,
            down_weights=experts.down_weights,
            num_experts=experts.top_k,
            intermediate_size=experts.intermediate,
            swiglu_limit=experts.limit,
            top_k=experts.top_k,
            routed_scaling_factor=experts.routed_scaling_factor,
            routing_eps=experts.routing_eps,
            experts_block_size=experts.experts_block_size,
            shared_gate_up_weights=shared.gate_up,
            shared_down_weights=shared.down,
            shared_swiglu_limit=shared.swiglu_limit,
            memory_config=experts.output_memory_config,
            all_reduce_cluster_axes=experts.fused_all_reduce_axes,
            all_reduce_num_links=2,
            all_reduce_topology=CCL_TOPOLOGY,
            persistent_layer_table=self.layer_table,
            persistent_socket_core=self.socket_core,
            persistent_input_sockets=self.request_rx,
            persistent_partials=self.partials,
        )
        self.scratch = self._launch()
        self.stopped = False

    @staticmethod
    def router_column(moe, moe_mesh):
        """``moe``'s gate split by expert over ``moe_mesh``: chip m (in mesh order, the op's chip index) holds the
        gate's tile column m, ``[D, 32]``, in L1, height-sharded over :data:`ROUTER_PART_CORES` (its K slices). The
        tile columns are the gate's own tiles, so its bfloat8_b blocks carry over exactly."""
        gate = moe.gate.fused_gate_weight
        weight = ttnn.to_torch(ttnn.get_device_tensors(gate)[0])
        weight = weight.reshape(-1, weight.shape[-2], weight.shape[-1])[0]
        d, e = weight.shape
        chips = moe_mesh.get_num_devices()
        assert e == 32 * chips, f"the sharded router needs {32 * chips} experts over {chips} chips, got {e}"
        assert d % (32 * ROUTER_PARTS) == 0, f"the sharded router splits K={d} into {ROUTER_PARTS} tile slices"
        columns = torch.stack([weight[:, 32 * m : 32 * (m + 1)] for m in range(chips)]).unsqueeze(1)
        return ttnn.from_torch(
            columns,
            dtype=gate.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=moe_mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(moe_mesh, dim=0),
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(ROUTER_PART_CORES, [d // ROUTER_PARTS, 32], ttnn.ShardOrientation.ROW_MAJOR),
            ),
        )

    # The partials buffer's parts: one per stream-collapse core, which holds those columns of the residual streams
    # (so the request can fuse the hyper-connection post step); the op's two fabric links send half each.
    PARTS = _HC_COLLAPSE_CORES

    def streams_memory_config(self, request_mesh, num_streams: int = 4) -> ttnn.MemoryConfig:
        """The residual streams' ``[hc, D]`` ROW_MAJOR width-sharded L1 config, whose shards the partials buffer's
        line up with (see :meth:`request`'s ``streams``)."""
        return width_sharded_l1_config(num_streams, self.hidden, request_mesh, num_cores=self.PARTS, tile_height=1)

    def partials_buffer(self, request_mesh, num_chips: int):
        """A requester's partials buffer, zeroed: on every one of its chips, one L1 shard per part of the row, on
        the stream-collapse cores in shard order, of ``num_chips`` slots of that part (the MoE chips' partials, in
        their order) then a flag word per chip."""
        part_bytes = self.row_bytes // self.PARTS
        words = num_chips * (part_bytes + 4) // 4
        cores = self.streams_memory_config(request_mesh).shard_spec.grid
        return ttnn.from_torch(
            torch.zeros(self.PARTS, words, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=request_mesh,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(cores, [1, words], ttnn.ShardOrientation.ROW_MAJOR),
            ),
            mesh_mapper=ttnn.ReplicateTensorToMesh(request_mesh),
        )

    def request_input(self, requester: int):
        """A ``[1, 1, 1, D]`` request row on every chip of ``requester``, zeroed: the DRAM-interleaved one-page
        ROW_MAJOR bf16 row :meth:`request` sends."""
        mesh = self.request_meshes[requester]
        return ttnn.from_torch(
            torch.zeros(1, 1, 1, self.hidden),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

    def upload(self, requester: int, row: torch.Tensor, tensor=None):
        host = ttnn.from_torch(
            row.reshape(1, 1, 1, -1),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.request_meshes[requester]),
        )
        ttnn.copy_host_to_device_tensor(host, self.inputs[requester] if tensor is None else tensor)

    def row_of_layer(self, layer: int) -> int:
        try:
            return self.table_rows[layer]
        except KeyError:
            raise KeyError(f"layer {layer} is not on this persistent MoE (it serves {self.layer_ids})") from None

    def request(self, requester: int, layer: int, tensor=None, streams=None, post=None, comb=None):
        """Dispatches ``requester``'s request for ``layer`` (a model layer id) without waiting for it; returns its
        output, the reduced row in ``tensor``'s spec, replicated over the requester's chips. ``tensor`` is a
        ``[1, 1, 1, D]`` bf16 ROW_MAJOR DRAM-interleaved row (one page), :meth:`request_input`'s by default.

        With ``streams`` ``[1, 1, hc, D]`` (in :meth:`streams_memory_config`), ``post`` ``[1, 1, hc, 1]`` and
        ``comb`` ``[1, 1, hc, hc]`` (bf16 TILE) the output is instead the hyper-connection's new streams,
        ``post[h] * row + sum_g comb[g, h] * streams[g]`` (``mix_streams``), in ``streams``' spec."""
        return self._send(requester, self.row_of_layer(layer), tensor, streams=streams, post=post, comb=comb)

    def _send(self, requester: int, row: int, tensor=None, streams=None, post=None, comb=None):
        return ttnn.experimental.deepseek.moe.persistent_moe_request(
            self.inputs[requester] if tensor is None else tensor,
            layer=row,
            request_socket=self.request_tx[requester],
            partials=self.partials[requester],
            streams=streams,
            post=post,
            comb=comb,
        )

    @staticmethod
    def rows_of(out) -> list:
        """Each request chip's copy of the reduced row, as ``[D]`` bf16."""
        return [ttnn.to_torch(t).reshape(-1) for t in ttnn.get_device_tensors(out)]

    @classmethod
    def row_of(cls, out) -> torch.Tensor:
        """The reduced row of ``out``, which every request chip must hold alike."""
        copies = cls.rows_of(out)
        mismatched = [c for c, copy in enumerate(copies) if not torch.equal(copy, copies[0])]
        assert not mismatched, f"request chips {mismatched} got a different response than chip 0"
        return copies[0]

    def stop(self):
        """Ends the kernels: one stop request from every requester, then waits for every mesh. Idempotent."""
        if self.stopped:
            return
        self.stopped = True
        for j in range(len(self.request_meshes)):
            ttnn.deallocate(self._send(j, STOP_LAYER))
        for mesh in self.request_meshes + [self.moe_mesh]:
            ttnn.synchronize_device(mesh)

    def start(self):
        """Relaunches the kernels after :meth:`stop` (a no-op while they run), on the same sockets, buffers and layer
        table: the kernels leave the request sockets' read state, the partials' flags and their go words as a fresh
        launch expects them. The mesh is free between the two, e.g. for
        :class:`~..prefill.persistent_moe.PersistentPrefillMoE`'s requests."""
        if not self.stopped:
            return
        ttnn.deallocate(self.scratch)
        self.scratch = self._launch()
        self.stopped = False
