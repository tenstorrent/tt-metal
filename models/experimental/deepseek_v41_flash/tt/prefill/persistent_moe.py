# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""The prefill side of the persistent TP12 MoE: whole prefill chunks (up to 1024 tokens) served by the MoE mesh.

The decode :class:`~..decode.persistent_moe.PersistentMoE` serves one token row per request from a kernel that
never leaves the device. A prefill chunk is ``[1024, D]`` bf16, 10 MB, so its requests are host driven: each one
is a few ops enqueued on the requester and on the MoE mesh, joined by D2D sockets, and the cost of the host
dispatch is small next to the chunk's MoE (one pass over every expert's weights).

A request for layer ``L`` with the chunk ``x`` (``[1, 1, T, D]`` bf16 ROW_MAJOR, DRAM, replicated on the requester's
chips, ``T`` a multiple of 32 up to :data:`CHUNK`):

1. each requester chip sends ``x`` down its column to the three MoE chips below it (``send_direct_async``, one
   socket per MoE row), so every MoE chip holds the whole chunk;
2. the MoE mesh runs layer ``L``'s MoE at TP12 (:class:`PrefillMoELayer`): the router (the gate matmul,
   ``sqrt(softplus)``, the bias-corrected ranking row), the routed experts (``fused_experts_prefill`` on the
   decode experts' TP12 weights, read in place: I = 192 per chip, its 3x2 core groups one I tile per core),
   the shared expert (its TP12 slice), and the sum of the twelve partials (``all_reduce`` over both mesh axes);
3. MoE row 0, which ends with the sum like every MoE chip, sends it up its column to the requester's chips, so the
   result (``[1, 1, T, D]`` bf16 TILE, DRAM) is replicated over the requester's mesh like ``x``.

The MoE is TP12 because it shares its weights with decode: the routed experts are the decode blocks' own
(:class:`~..decode.moe.DeepSeekV41PreloadedExperts` at TP12), and the router and shared expert are re-laid from
the decode blocks' fused tensors at build time (a few MB per layer per chip), so no weight is read from the
checkpoint again. While the decode op runs it owns the MoE mesh's cores, so prefill requests go between a
:meth:`PersistentMoE.stop` and :meth:`PersistentMoE.start` -- :meth:`PersistentPrefillMoE.window` does both.

See ``tests/prefill/test_moe_persistent_prefill.py``.
"""

from contextlib import contextmanager
from typing import Mapping, Optional, Sequence

import torch

import ttnn

from models.experimental.deepseek_v4_flash.tt.common import _HIFI4, CCL_TOPOLOGY

# The prefill chunk a request carries at most: ``fused_experts_prefill``'s token cap (``kMaxTokens``), the size at
# which one op call -- one pass over the weights -- covers the whole chunk.
CHUNK = 1024
ALIGNMENT = 32
TP_SIZE = 12
# The sockets carry only the direct ops' handshake pages (the payload is written straight into the receiver's
# tensor), so their FIFOs are small: they sit in L1 under ``fused_experts_prefill``'s circular buffers.
SOCKET_FIFO_BYTES = 1024

_HIFI2 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)


def _local(tensor) -> list:
    """Each chip's shard of a mesh tensor, as torch, in mesh order (the order ``ShardTensorToMesh`` fills)."""
    return [ttnn.to_torch(t) for t in ttnn.get_device_tensors(tensor)]


def _per_chip(tensors: Sequence[torch.Tensor], mesh, dtype) -> ttnn.Tensor:
    """``tensors[m]`` (2D) onto chip ``m`` of ``mesh``: TILE, DRAM interleaved, ``[1, 1, rows, cols]`` per chip."""
    return ttnn.from_torch(
        torch.stack(list(tensors)).unsqueeze(1),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )


class PrefillMoELayer:
    """One layer's MoE at TP12 over a ``[1, 1, T, D]`` chunk, built from its decode
    :class:`~..decode.moe.DeepSeekV41SparseMoeBlock` (the fused router and shared expert, TP12).

    * Router: the decode gate's ``[D, E]`` bfloat8_b weight, replicated as a plain DRAM tensor for ``ttnn.linear``
      (the tiles carry over, so the bf8 blocks are the decode op's own); the ranking adds the decode router's
      (shifted) bias, or with an ``image_mask`` its per-token bias.
    * Routed experts: ``fused_experts_prefill`` on ``moe.experts``' weights in place.
    * Shared expert: the fused shared expert's ``gate_up`` (gate/up interleaved per I tile, ``[D, 64]`` shards) and
      ``down`` (``[32, 64]`` shards), re-laid per chip as one ``[D, 2 Is]`` ``[gate | up]`` weight and an ``[Is, D]``
      down, DRAM interleaved; the tiles are moved whole, so their bf8 blocks are unchanged.

    :meth:`__call__` returns this chip's partial; :meth:`forward` the sum over the mesh.
    """

    def __init__(self, moe, mesh):
        experts = moe.experts
        if experts.tp_size != TP_SIZE or mesh.get_num_devices() != TP_SIZE:
            raise ValueError(f"the prefill MoE runs TP{TP_SIZE} on a {TP_SIZE}-chip mesh, got TP{experts.tp_size}")
        if moe.fused_shared is None or moe.gate.fused_gate_weight is None:
            raise ValueError("the prefill MoE re-lays the decode block's fused router and shared expert")
        self.mesh = mesh
        self.experts = experts
        self.router = moe.gate
        self.hidden = moe.hidden

        gate = moe.gate.fused_gate_weight
        weight = _local(gate)[0]
        weight = weight.reshape(-1, weight.shape[-2], weight.shape[-1])[0]
        self.gate_weight = ttnn.from_torch(
            weight.reshape(1, 1, *weight.shape),
            dtype=gate.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        self.bias = ttnn.to_layout(moe.gate.ranking_bias(), ttnn.TILE_LAYOUT)

        shared = moe.fused_shared
        gate_up, down = [], []
        for gu, dn in zip(_local(shared.gate_up), _local(shared.down)):
            gu = gu.reshape(gu.shape[-2], gu.shape[-1])
            d, two_is = gu.shape
            tiles = two_is // 64
            pairs = gu.reshape(d, tiles, 2, 32)
            gate_up.append(torch.cat([pairs[:, :, 0].reshape(d, -1), pairs[:, :, 1].reshape(d, -1)], dim=1))
            down.append(dn.reshape(dn.shape[-2], dn.shape[-1]))
        self.shared_inter = gate_up[0].shape[1] // 2
        self.shared_gate_up = _per_chip(gate_up, mesh, shared.gate_up.dtype)
        self.shared_down = _per_chip(down, mesh, shared.down.dtype)
        limit = shared.swiglu_limit
        self.shared_limit = None if limit is None or limit == float("inf") else limit

    def route(self, x_tile, image_mask=None):
        """``(scores, ranking)``: the unbiased ``sqrt(softplus)`` scores and the biased row ranked on device."""
        logits = ttnn.linear(x_tile, self.gate_weight, compute_kernel_config=_HIFI4)
        scores = ttnn.sqrt(ttnn.softplus(logits))
        ttnn.deallocate(logits)
        if image_mask is None:
            return scores, ttnn.add(scores, self.bias)
        bias = self.router.ranking_bias(image_mask, layout=ttnn.TILE_LAYOUT)
        ranking = ttnn.add(scores, bias)
        ttnn.deallocate(bias)
        return scores, ranking

    def shared_expert(self, x_tile):
        """This chip's partial of the shared expert, ``down(silu(min(gate, L)) * clamp(up, -L, L))``."""
        gate_up = ttnn.linear(x_tile, self.shared_gate_up, compute_kernel_config=_HIFI2)
        t, i = x_tile.shape[2], self.shared_inter
        gate = ttnn.slice(gate_up, [0, 0, 0, 0], [1, 1, t, i])
        up = ttnn.slice(gate_up, [0, 0, 0, i], [1, 1, t, 2 * i])
        ttnn.deallocate(gate_up)
        if self.shared_limit is not None:
            clamped = ttnn.clamp(gate, max=self.shared_limit)
            ttnn.deallocate(gate)
            gate = clamped
            clamped = ttnn.clamp(up, min=-self.shared_limit, max=self.shared_limit)
            ttnn.deallocate(up)
            up = clamped
        act = ttnn.multiply(ttnn.silu(gate), up)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        out = ttnn.linear(act, self.shared_down, compute_kernel_config=_HIFI2)
        ttnn.deallocate(act)
        return out

    def __call__(self, x_rm, x_tile=None, image_mask=None):
        """This chip's partial ``routed + shared`` of the chunk ``x_rm`` (``[1, 1, T, D]`` bf16 ROW_MAJOR, DRAM),
        ``[1, 1, T, D]`` TILE. ``x_tile`` is the same chunk in TILE, made here when not given."""
        own_tile = x_tile is None
        if own_tile:
            x_tile = ttnn.to_layout(x_rm, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        scores, ranking = self.route(x_tile, image_mask)
        experts = self.experts
        routed = ttnn.experimental.deepseek_prefill.fused_experts_prefill(
            x_rm,
            scores,
            experts.gate_up_weights,
            experts.down_weights,
            experts.intermediate,
            experts.limit,
            experts.top_k,
            experts.routed_scaling_factor,
            experts.routing_eps,
            ranking_scores=ranking,
        )
        ttnn.deallocate(scores)
        ttnn.deallocate(ranking)
        shared = self.shared_expert(x_tile)
        if own_tile:
            ttnn.deallocate(x_tile)
        out = ttnn.add(routed, shared, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(routed)
        ttnn.deallocate(shared)
        return out

    def forward(self, x_rm, x_tile=None, image_mask=None):
        """The layer's MoE of the chunk, summed over the twelve chips and replicated on each, ``[1, 1, T, D]`` TILE."""
        out = self(x_rm, x_tile, image_mask)
        for axis in (1, 0):
            partial = out
            out = ttnn.all_reduce(partial, cluster_axis=axis, num_links=2, topology=CCL_TOPOLOGY)
            ttnn.deallocate(partial)
        return out


def _last_column_core(mesh, y: int) -> ttnn.CoreCoord:
    grid = mesh.compute_with_storage_grid_size()
    return ttnn.CoreCoord(grid.x - 1, y)


class PersistentPrefillMoE:
    """Prefill chunks for the layers of ``moes`` -- ``{layer: block}`` or a list (layer = index), every block the
    decode :class:`~..decode.moe.DeepSeekV41SparseMoeBlock` at TP12 on the 3x4 ``moe_mesh`` -- served to one
    requester per mesh of ``request_meshes`` (each a ``1 x 4`` mesh above the MoE, chip ``c`` over column ``c``).

    ``decode`` is the :class:`~..decode.persistent_moe.PersistentMoE` the mesh is time-shared with, if any: requests
    go inside :meth:`window`, which stops it and relaunches it after.

    Each requester has one socket per MoE row for its chunks (requester chip ``(0, c)`` -> MoE chip ``(r, c)``) and
    one for the results (MoE ``(0, c)`` -> requester ``(0, c)``), on cores of the grid's last column:
    ``x_send_cores[r]`` and ``out_recv_core`` on the requester, ``x_recv_core`` and ``out_send_core`` on the MoE.
    Their FIFOs allocate L1, so build this before any trace capture on the requesters.

    ``moes`` may be left out and the layers added later with :meth:`add_layers`, which touches only the MoE mesh: the
    sockets (built here, on every mesh) come first, then the MoE layers can load on a thread of their own while the
    requesters build theirs (see ``tests/prefill/test_full_model_prefill_verify.py``).
    """

    def __init__(
        self,
        moe_mesh,
        request_meshes: Sequence,
        moes=None,
        decode=None,
        x_send_cores: Optional[Sequence[ttnn.CoreCoord]] = None,
        out_recv_core: Optional[ttnn.CoreCoord] = None,
        x_recv_core: Optional[ttnn.CoreCoord] = None,
        out_send_core: Optional[ttnn.CoreCoord] = None,
    ):
        self.moe_mesh = moe_mesh
        self.request_meshes = list(request_meshes)
        self.decode = decode
        self.layers = {}
        self.layer_ids = []
        self.hidden = None
        # Set while the traced prefill captures a requester's trace: PersistentPrefillMLP then enqueues only its
        # requester half (send_chunk); the MoE half (serve) is dispatched per replay instead.
        self.stage_only = False
        # Opening the sockets writes their configs through every mesh's command queue, which a running persistent
        # decode op never frees on the MoE mesh: open them with it stopped.
        with self.window():
            self._open_sockets(x_send_cores, out_recv_core, x_recv_core, out_send_core)
        if moes:
            self.add_layers(moes)

    def add_layers(self, moes) -> None:
        """Adds the layers of ``moes`` (``{layer: block}`` or a list, layer = index; see the class docstring).

        Reads the decode blocks' router and shared expert back and uploads this server's copies: the MoE mesh's work
        only, so it may run on its own thread while the requesters build. The decode op, if running, is stopped for
        it and relaunched after."""
        if not isinstance(moes, Mapping):
            moes = dict(enumerate(moes))
        if not moes:
            raise ValueError("the prefill MoE needs at least one layer")
        relaunch = self.decode is not None and not self.decode.stopped
        if relaunch:
            self.decode.stop()
        try:
            for layer in sorted(moes):
                self.layers[layer] = PrefillMoELayer(moes[layer], self.moe_mesh)
            ttnn.synchronize_device(self.moe_mesh)
        finally:
            if relaunch:
                self.decode.start()
        self.layer_ids = sorted(self.layers)
        self.hidden = self.layers[self.layer_ids[0]].hidden

    def _open_sockets(self, x_send_cores, out_recv_core, x_recv_core, out_send_core):
        moe_mesh = self.moe_mesh
        moe_rows, moe_cols = tuple(moe_mesh.shape)
        self.moe_rows = moe_rows
        x_send_cores = list(x_send_cores or [_last_column_core(self.request_meshes[0], r) for r in range(moe_rows)])
        if len(x_send_cores) != moe_rows:
            raise ValueError(f"one chunk sender core per MoE row ({moe_rows}), got {len(x_send_cores)}")
        out_recv_core = out_recv_core or _last_column_core(self.request_meshes[0], moe_rows)
        x_recv_core = x_recv_core or _last_column_core(moe_mesh, 0)
        out_send_core = out_send_core or _last_column_core(moe_mesh, 1)
        memory = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, SOCKET_FIFO_BYTES)

        def pair(sender_mesh, receiver_mesh, connections):
            config = ttnn.SocketConfig(
                [
                    ttnn.SocketConnection(
                        ttnn.MeshCoreCoord(ttnn.MeshCoordinate(*src), src_core),
                        ttnn.MeshCoreCoord(ttnn.MeshCoordinate(*dst), dst_core),
                    )
                    for src, src_core, dst, dst_core in connections
                ],
                memory,
            )
            return ttnn.create_socket_pair(sender_mesh, receiver_mesh, config)

        self.x_tx, self.x_rx, self.out_tx, self.out_rx = [], [], [], []
        for request_mesh in self.request_meshes:
            if tuple(request_mesh.shape) != (1, moe_cols):
                raise ValueError(f"a requester is a 1x{moe_cols} mesh over the MoE's columns, got {request_mesh.shape}")
            txs, rxs = [], []
            for r in range(moe_rows):
                tx, rx = pair(
                    request_mesh,
                    moe_mesh,
                    [((0, c), x_send_cores[r], (r, c), x_recv_core) for c in range(moe_cols)],
                )
                txs.append(tx)
                rxs.append(rx)
            self.x_tx.append(txs)
            self.x_rx.append(rxs)
            tx, rx = pair(
                moe_mesh,
                request_mesh,
                [((0, c), out_send_core, (0, c), out_recv_core) for c in range(moe_cols)],
            )
            self.out_tx.append(tx)
            self.out_rx.append(rx)
        # The MoE's copy of a chunk, one per token count, reused by every request (the MoE mesh runs them in order).
        self._moe_inputs = {}

    # ------------------------------------------------------------------ time-sharing with decode
    @contextmanager
    def window(self):
        """Stops the decode op (if it runs) for the requests made inside, and relaunches it after them."""
        relaunch = self.decode is not None and not self.decode.stopped
        if relaunch:
            self.decode.stop()
        # A request that failed part-way leaves a requester waiting on its socket for an output that never comes:
        # synchronizing or relaunching would then hang instead of raising, so both are for a clean exit only.
        yield self
        for mesh in self.request_meshes + [self.moe_mesh]:
            ttnn.synchronize_device(mesh)
        if relaunch:
            self.decode.start()

    def _check_free(self):
        if self.decode is not None and not self.decode.stopped:
            raise RuntimeError("the decode persistent MoE owns the MoE mesh's cores: request inside window()")

    # ------------------------------------------------------------------ requests
    def layer(self, layer: int) -> PrefillMoELayer:
        try:
            return self.layers[layer]
        except KeyError:
            raise KeyError(f"layer {layer} is not on this prefill MoE (it serves {self.layer_ids})") from None

    def _moe_input(self, tokens: int):
        x = self._moe_inputs.get(tokens)
        if x is None:
            x = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, tokens, self.hidden]),
                ttnn.bfloat16,
                ttnn.ROW_MAJOR_LAYOUT,
                self.moe_mesh,
                ttnn.DRAM_MEMORY_CONFIG,
            )
            self._moe_inputs[tokens] = x
        return x

    def _check_chunk(self, x):
        shape = tuple(x.shape)
        if len(shape) != 4 or shape[:2] != (1, 1) or shape[3] != self.hidden:
            raise ValueError(f"a chunk is [1, 1, T, {self.hidden}], got {shape}")
        t = shape[2]
        if t % ALIGNMENT or not 0 < t <= CHUNK:
            raise ValueError(f"a chunk has a multiple of {ALIGNMENT} tokens, at most {CHUNK}, got {t}")
        if (
            x.layout != ttnn.ROW_MAJOR_LAYOUT
            or x.dtype != ttnn.bfloat16
            or x.memory_config() != ttnn.DRAM_MEMORY_CONFIG
        ):
            raise ValueError("a chunk is bf16 ROW_MAJOR, DRAM interleaved (the routed op gathers its rows by page)")
        return t

    def request(self, requester: int, layer: int, x, image_mask=None):
        """``requester``'s chunk ``x`` (``[1, 1, T, D]`` bf16 ROW_MAJOR DRAM, replicated over its chips) through the
        MoE of ``layer`` (a model layer id); returns the result, ``[1, 1, T, D]`` bf16 TILE DRAM, replicated over the
        requester's chips. Enqueues the requester's and the MoE mesh's ops without waiting for them. ``image_mask``
        (``[1, T]`` uint32 ROW_MAJOR on the MoE mesh, 1 for an image token) picks each token's router bias.

        The two halves, :meth:`send_chunk` (the requester's) and :meth:`serve` (the MoE mesh's)."""
        out = self.send_chunk(requester, x)
        self.serve(requester, layer, x.shape[2], image_mask=image_mask)
        return out

    def send_chunk(self, requester: int, x):
        """The requester's half of :meth:`request`: sends ``x`` to every MoE row and receives the result into a new
        ``[1, 1, T, D]`` bf16 TILE DRAM tensor, which it returns. Only the requester's ops, on fixed sockets and the
        MoE mesh's fixed input buffer, so it can be traced on the requester (the traced prefill); the MoE mesh's
        half, :meth:`serve`, then runs once per replay, in the same layer order."""
        if not self.stage_only:  # a capture runs nothing, on the MoE mesh least of all
            self._check_free()
        tokens = self._check_chunk(x)
        for tx in self.x_tx[requester]:
            ttnn.experimental.send_direct_async(x, tx)
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, tokens, self.hidden]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            self.request_meshes[requester],
            ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.experimental.recv_direct_async(out, self.out_rx[requester])
        return out

    def serve(self, requester: int, layer: int, tokens: int, image_mask=None) -> None:
        """The MoE mesh's half of :meth:`request`: receives ``requester``'s ``tokens``-row chunk, runs ``layer`` and
        sends the result back. Enqueues without waiting."""
        self._check_free()
        moe_layer = self.layer(layer)
        x_moe = self._moe_input(tokens)
        for rx in self.x_rx[requester]:
            ttnn.experimental.recv_direct_async(x_moe, rx)
        out_moe = moe_layer.forward(x_moe, image_mask=image_mask)
        ttnn.experimental.send_direct_async(out_moe, self.out_tx[requester])
        ttnn.deallocate(out_moe)

    def request_input(self, requester: int, tokens: int = CHUNK):
        """A zeroed chunk on ``requester``'s chips in the layout :meth:`request` takes."""
        mesh = self.request_meshes[requester]
        return ttnn.from_torch(
            torch.zeros(1, 1, tokens, self.hidden),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

    @staticmethod
    def chunks_of(out) -> list:
        """Each request chip's copy of a result, as ``[T, D]`` torch."""
        return [ttnn.to_torch(t).reshape(-1, t.shape[-1]) for t in ttnn.get_device_tensors(out)]

    @classmethod
    def chunk_of(cls, out) -> torch.Tensor:
        """The result ``out``, which every request chip must hold alike, as ``[T, D]`` torch."""
        copies = cls.chunks_of(out)
        mismatched = [c for c, copy in enumerate(copies) if not torch.equal(copy, copies[0])]
        assert not mismatched, f"request chips {mismatched} got a different result than chip 0"
        return copies[0]


class PersistentPrefillMLP:
    """A prefill decoder layer's MoE served by a :class:`PersistentPrefillMoE`: what
    :class:`~.decoder_layer.DeepSeekV41PrefillDecoderLayer` calls as its ``mlp`` in place of
    :class:`~.moe.DeepSeekV41PrefillMoE`, so the layer's stage holds no MoE weights.

    ``server.request_meshes[requester]`` is the layer's ``1 x 4`` stage and ``layer`` its model layer id. A call takes
    the normed rows ``[1, 1, T, D]`` bf16 TILE (``T`` a multiple of 32, at most :data:`CHUNK`) and returns the MoE's
    output in the same layout, replicated over the stage's chips, as the block's all-reduced output is. The server
    may get ``layer`` (:meth:`PersistentPrefillMoE.add_layers`) after this is built, as long as it is before the call.
    """

    def __init__(self, server: PersistentPrefillMoE, requester: int, layer: int):
        self.server = server
        self.requester = requester
        self.layer_idx = layer

    def __call__(self, x, image_mask=None):
        rows = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        mask = None
        if image_mask is not None:
            # The router runs on the MoE mesh: hand it the stage's mask through the host (eager prefill only).
            mesh = self.server.moe_mesh
            mask = ttnn.from_torch(
                ttnn.to_torch(ttnn.get_device_tensors(image_mask)[0]),
                dtype=image_mask.dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
        try:
            if self.server.stage_only:
                if mask is not None:
                    raise NotImplementedError("the traced persistent prefill routes no image tokens")
                return self.server.send_chunk(self.requester, rows)
            return self.server.request(self.requester, self.layer_idx, rows, image_mask=mask)
        finally:
            ttnn.deallocate(rows)
            if mask is not None:
                ttnn.deallocate(mask)

    def serve(self, tokens: int) -> None:
        """The MoE mesh's half of one call of ``tokens`` rows (see :meth:`PersistentPrefillMoE.serve`): what a replay
        of a trace captured with ``server.stage_only`` needs dispatched, layer by layer in the trace's order."""
        self.server.serve(self.requester, self.layer_idx, tokens)
