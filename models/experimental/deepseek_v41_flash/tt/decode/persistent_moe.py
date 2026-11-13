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

import gc
import json
import os
import random
import time
from typing import Mapping, Sequence

import torch
import ttnn
from loguru import logger

from models.experimental.deepseek_v4_flash.tt.common import CCL_TOPOLOGY, width_sharded_l1_config
from models.experimental.deepseek_v4_flash.tt.decode.hyperconnection import _HC_COLLAPSE_CORES
from models.experimental.deepseek_v4_flash.tt.decode.moe import fused_weight_hook

from .moe import DeepSeekV41PreloadedExperts, DeepSeekV41SparseMoeBlock

TP_SIZE = 12
# The decode layout on the 8x4 Galaxy: the MoE is the 3x4 submesh at row 5, and its one requester -- the TP4 stage
# holding every layer's attention, hyper-connections and norms -- is the 1x4 submesh right above it.
GALAXY_MESH = (8, 4)
MOE_MESH_SHAPE = (3, 4)
MOE_MESH_OFFSET = (5, 0)
REQUESTER_ROW = 4
STOP_LAYER = 0xFFFFFFFF
HEADER_BYTES = 64
# The worker grid and router row the op multicasts each row over: 12 x (8 + 1).
REPLICA_GRID = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 8))])
# The op's router is sharded by expert over the MoE chips: chip m scores the gate's tile column m (experts
# [32m, 32m + 32)), K-split over the router row's first ROUTER_PARTS cores, which hold each layer's slice in L1.
ROUTER_PARTS = 10
# Every layer's router columns stay in L1 while they take at most this much of each core's L1 (an L1 buffer takes its
# address range on every core of the chip); beyond it -- e.g. all 40 layers, ~0.7 MB -- they go to DRAM.
ROUTER_L1_BUDGET = int(os.environ.get("DEEPSEEK_V41_MOE_ROUTER_L1_BUDGET", 192 * 1024))
# DEEPSEEK_V41_MOE_FUSE_HC=0: requests return the plain reduced row (2-part reduce next to the request core) and the
# decoder layer mixes it into the streams itself (FFN_MIX), instead of the request running that post step.
FUSE_HC = os.environ.get("DEEPSEEK_V41_MOE_FUSE_HC", "1") not in ("0", "", "false", "False")
ROUTER_PART_CORES = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 8), ttnn.CoreCoord(ROUTER_PARTS - 1, 8))])


def single_senders(rows: Sequence[int]) -> list:
    """``PersistentMoE``'s ``single_senders`` for requesters on Galaxy rows ``rows``: the row right above the MoE
    sends each request as one multicast to all its chips, so they start the row together; the others keep one send
    per column."""
    return [row == MOE_MESH_OFFSET[0] - 1 for row in rows]


def default_socket_core(mesh) -> ttnn.CoreCoord:
    """The bottom-right worker core: outside the 12x9 grid the op multicasts over and away from its reduce
    senders (the bottom row's left end). On a requester it only holds the socket FIFOs and the request op,
    which runs between the layer's other ops."""
    grid = mesh.compute_with_storage_grid_size()
    return ttnn.CoreCoord(grid.x - 1, grid.y - 1)


# DRAM warm start (see :func:`build_moe_layers`).
WARM_START_ENV = "DEEPSEEK_V41_MOE_WARM_START"
_MANIFEST_VERSION = 1
_CANARY_WORDS = 8
# The canary of the last build, kept allocated for the life of the process: freed, a later buffer could take its
# place and overwrite it, and the next run's warm start would then never match.
_resident_canaries: list = []


def warm_start_requested() -> bool:
    """``DEEPSEEK_V41_MOE_WARM_START=1``: try to reuse the routed experts a previous process left in DRAM."""
    return os.environ.get(WARM_START_ENV, "0") not in ("0", "", "false", "False")


class _WarmMiss(RuntimeError):
    """The DRAM a previous run left cannot be reused as it is; the build falls back to a cold load."""


def _tensorbin(kwargs) -> str | None:
    name = kwargs.get("cache_file_name")
    if not name:
        return None
    dtype = kwargs.get("dtype", ttnn.bfloat4_b)
    return f"{name}_dtype_{dtype.name}_layout_{ttnn.TILE_LAYOUT.name}.tensorbin"


def _manifest_path(cache, layer_ids: Sequence[int]) -> str | None:
    if not getattr(cache, "path", None) or not layer_ids:
        return None
    ids = list(layer_ids)
    return os.path.join(cache.path, f"persistent_moe_warm.layers{ids[0]}-{ids[-1]}x{len(ids)}.json")


def _canary(mesh, words=None) -> ttnn.Tensor:
    """A ``[1, 1, 1, 8]`` uint32 DRAM tensor on every MoE chip: holding ``words``, or, without them, allocated and
    left as DRAM holds it."""
    if words is None:
        return ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, 1, _CANARY_WORDS]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
        )
    return ttnn.from_torch(
        torch.tensor(words, dtype=torch.int32).reshape(1, 1, 1, _CANARY_WORDS),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _build_layers(config, loader, moe_mesh, cache, layer_ids, log) -> dict:
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


def _cold_build(config, loader, moe_mesh, cache, layer_ids, log, manifest_path) -> dict:
    """Load every layer from the cache, recording each fused weight's spec and address, then write a fresh canary and
    the manifest a later process's warm start reads."""
    if manifest_path is not None and os.path.exists(manifest_path):
        os.remove(manifest_path)  # Gone until this load completes: a crash mid-load leaves nothing to trust.
    entries = []
    current = {"layer": None}

    def record(load, tensor, device, nd_config, **kwargs):
        t = load(tensor, device, nd_config, **kwargs)
        path = _tensorbin(kwargs)
        stat = os.stat(path) if path is not None and os.path.exists(path) else None
        entries.append(
            {
                "layer": current["layer"],
                "file": path,
                "size": stat.st_size if stat else None,
                "mtime_ns": stat.st_mtime_ns if stat else None,
                "shape": list(t.shape),
                "dtype": t.dtype.name,
                "layout": t.layout.name,
                "memory_config": t.memory_config().to_json(),
                "address": t.buffer_address(),
            }
        )
        return t

    def log_layer(layer):
        # Called once a layer is built: the fused weights loaded next belong to the layer after it.
        nxt = layer_ids.index(layer) + 1
        current["layer"] = layer_ids[nxt] if nxt < len(layer_ids) else None
        if log is not None:
            log(layer)

    current["layer"] = layer_ids[0] if layer_ids else None
    with fused_weight_hook(record):
        moes = _build_layers(config, loader, moe_mesh, cache, layer_ids, log_layer)
    if manifest_path is None:
        return moes
    words = [random.getrandbits(31) for _ in range(_CANARY_WORDS)]
    canary = _canary(moe_mesh, words)
    _resident_canaries.append(canary)
    ttnn.synchronize_device(moe_mesh)
    manifest = {
        "version": _MANIFEST_VERSION,
        "device_ids": list(moe_mesh.get_device_ids()),
        "layer_ids": list(layer_ids),
        "canary": {"words": words, "address": canary.buffer_address()},
        "entries": entries,
    }
    tmp = f"{manifest_path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f)
    os.replace(tmp, manifest_path)
    logger.info(f"MoE warm-start manifest written: {len(entries)} fused weights ({manifest_path})")
    return moes


def _warm_build(config, loader, moe_mesh, cache, layer_ids, log, manifest_path, held: list) -> dict:
    """Re-attach the fused weights a previous process loaded: allocate each at its recorded address without writing
    it, then check the canary and one routed expert per layer against its cache file. Raises :class:`_WarmMiss` when
    anything does not match; ``held`` collects what it allocated, for the caller to free."""
    if manifest_path is None or not os.path.exists(manifest_path):
        raise _WarmMiss("no manifest from an earlier cold load of these layers")
    with open(manifest_path) as f:
        manifest = json.load(f)
    if manifest.get("version") != _MANIFEST_VERSION:
        raise _WarmMiss(f"manifest version {manifest.get('version')}, expected {_MANIFEST_VERSION}")
    if manifest["device_ids"] != list(moe_mesh.get_device_ids()):
        raise _WarmMiss(f"the manifest is for chips {manifest['device_ids']}, not {list(moe_mesh.get_device_ids())}")
    if manifest["layer_ids"] != list(layer_ids):
        raise _WarmMiss("the manifest is for other layers")
    entries = manifest["entries"]
    cursor = {"i": 0}
    attached = []  # (entry, tensor)

    def reattach(load, tensor, device, nd_config, **kwargs):
        i = cursor["i"]
        cursor["i"] += 1
        if i >= len(entries):
            raise _WarmMiss(f"the build loads more fused weights than the manifest's {len(entries)}")
        e = entries[i]
        path = _tensorbin(kwargs)
        if path != e["file"] or path is None:
            raise _WarmMiss(f"fused weight {i} is {path}, the manifest has {e['file']}")
        stat = os.stat(path)
        if (stat.st_size, stat.st_mtime_ns) != (e["size"], e["mtime_ns"]):
            raise _WarmMiss(f"{path} changed since the cold load")
        # Allocate with the caller's own ND config, the one the cold load used, not MemoryConfig.from_json: to_json
        # cannot serialize an ND shard spec's shard_shape (it writes an "Unsupported type" string), and from_json
        # aborts the process on it. The recorded JSON is informational only.
        t = ttnn.allocate_tensor_on_device(
            ttnn.Shape(e["shape"]),
            getattr(ttnn.DataType, e["dtype"]),
            getattr(ttnn.Layout, e["layout"]),
            device,
            nd_config,
        )
        held.append(t)
        if t.buffer_address() != e["address"]:
            raise _WarmMiss(
                f"{path} allocated at {t.buffer_address():#x}, the cold load put it at {e['address']:#x}: the MoE "
                "mesh's allocations differ from the cold run's"
            )
        attached.append((e, t))
        return t

    with fused_weight_hook(reattach):
        moes = _build_layers(config, loader, moe_mesh, cache, layer_ids, log)
    if cursor["i"] != len(entries):
        raise _WarmMiss(f"the build loaded {cursor['i']} fused weights, the manifest has {len(entries)}")

    canary = _canary(moe_mesh)
    held.append(canary)
    if canary.buffer_address() != manifest["canary"]["address"]:
        raise _WarmMiss("the canary is not at its recorded address")
    for chip, copy in enumerate(ttnn.get_device_tensors(canary)):
        if ttnn.to_torch(copy).reshape(-1).tolist() != manifest["canary"]["words"]:
            raise _WarmMiss(f"MoE chip {chip}'s canary was overwritten: the DRAM was reset or reused")

    # One routed expert per layer, chosen anew each run, read back from all 12 chips and compared bit for bit.
    rng = random.Random()
    composer = ttnn.ConcatMeshToTensor(moe_mesh, dim=0)
    checked = 0
    for layer in layer_ids:
        expert = rng.randrange(moes[layer].experts.num_experts)
        tag = f".experts.{expert}."
        for e, t in attached:
            if e["layer"] != layer or tag not in e["file"]:
                continue
            on_device = ttnn.to_torch(t, mesh_composer=composer)
            on_disk = ttnn.to_torch(ttnn.load_tensor(e["file"]), mesh_composer=composer)
            if not torch.equal(on_device, on_disk):
                raise _WarmMiss(f"layer {layer} expert {expert}: DRAM differs from {e['file']}")
            checked += 1
    if checked < 2 * len(layer_ids):
        raise _WarmMiss(f"spot-checked only {checked} weights of {2 * len(layer_ids)}")
    _resident_canaries.append(canary)
    logger.info(f"MoE warm start: canary intact, {checked} routed-expert weights match their cache files")
    return moes


def build_moe_layers(config, loader, moe_mesh, cache, layer_ids: Sequence[int], *, log=None, warm_start=None) -> dict:
    """``{layer: DeepSeekV41SparseMoeBlock}`` at TP12 on ``moe_mesh`` for the checkpoint's ``layer_ids``.

    ``cache`` is the model's :class:`~...weight_cache.WeightCache`; each layer reads ``layers.<i>.ffn``, the
    routed experts K-split over every DRAM bank (``tools/fill_expert_cache.py --tp-size 12 --mesh-rows 3``
    fills them ahead of time). Build on a freshly opened ``moe_mesh``, before anything else allocates there.

    **DRAM warm start.** Closing the device frees only the allocator's book-keeping: until a reset or another
    program overwrites it, DRAM keeps what the last process loaded, and the allocator puts the same allocations at
    the same addresses. So a cold load records every fused weight's (routed experts', shared expert's) spec and
    address and writes a canary, and a manifest next to the cache. With ``warm_start`` (default
    ``DEEPSEEK_V41_MOE_WARM_START=1``) a later process allocates those weights at the recorded addresses without
    writing them -- skipping ~7.6 GB of reads and writes per layer -- and then checks: every address, the canary
    on every chip, and one randomly chosen routed expert per layer read back and compared bit for bit to its cache
    file. Any mismatch frees what it allocated and falls back to a cold load. The check samples: a program that
    overwrote unsampled experts and not the canary would go unnoticed, so after a reset or other jobs on these
    chips run once with ``DEEPSEEK_V41_MOE_WARM_START=0``. The router and its bias always load cold.
    """
    layer_ids = list(layer_ids)
    if warm_start is None:
        warm_start = warm_start_requested()
    manifest_path = _manifest_path(cache, layer_ids)
    if warm_start:
        start = time.perf_counter()
        held: list = []
        reason = None
        try:
            moes = _warm_build(config, loader, moe_mesh, cache, layer_ids, log, manifest_path, held)
            logger.info(
                f"MoE warm start: layers {layer_ids[0]}..{layer_ids[-1]} reused in {time.perf_counter() - start:.0f}s"
            )
            return moes
        except _WarmMiss as e:
            reason = str(e)
        except RuntimeError as e:  # e.g. a readback the device refused: never trust what was left
            reason = f"{type(e).__name__}: {e}"
        # Free what the attempt allocated so the cold load starts from the same allocator state as the run it records.
        for t in held:
            ttnn.deallocate(t)
        held.clear()
        gc.collect()
        ttnn.synchronize_device(moe_mesh)
        logger.warning(f"MoE warm start missed ({reason}); loading the experts cold")
    return _cold_build(config, loader, moe_mesh, cache, layer_ids, log, manifest_path)


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
        fuse_hc: bool | None = None,
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
        # With the HC post step fused, one partials part per stream-collapse core; without, one per fabric link.
        self.fuse_hc = FUSE_HC if fuse_hc is None else fuse_hc
        self.parts = _HC_COLLAPSE_CORES if self.fuse_hc else 2
        first = moes[self.layer_ids[0]]
        self.hidden = first.hidden
        self.row_bytes = self.hidden * 2
        experts = first.experts
        for layer in self.layer_ids:
            moe = moes[layer]
            assert moe.fused_router and moe.fused_shared is not None, f"layer {layer}: the op runs the whole block"
            assert moe.experts.tp_size == TP_SIZE, f"layer {layer}: experts at TP{moe.experts.tp_size}"
        assert experts.fused_all_reduce_axes is not None, "the persistent op reduces over the whole mesh"

        # Each layer's gate, one [D, 32] column per MoE chip: in L1 on its partial cores (see ROUTER_PART_CORES) while
        # every layer's fits ROUTER_L1_BUDGET, else in DRAM, which the partial cores read per request.
        self.router_in_l1 = len(self.layer_ids) * self.router_column_bytes(moes[self.layer_ids[0]]) <= ROUTER_L1_BUDGET
        self.router_columns = [
            self.router_column(moes[layer], moe_mesh, in_l1=self.router_in_l1) for layer in self.layer_ids
        ]
        logger.info(
            f"persistent MoE: router gate columns in {'L1' if self.router_in_l1 else 'DRAM'} ({len(self.layer_ids)} "
            f"layers), {self.parts} partials parts, HC post step {'fused' if self.fuse_hc else 'not fused'}"
        )
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
    def router_column_bytes(moe) -> int:
        """The L1 one layer's router column takes on each of :data:`ROUTER_PART_CORES` (and, as an L1 buffer, at
        that address range on every core)."""
        gate = moe.gate.fused_gate_weight
        k_tiles = gate.shape[-2] // 32
        tile_bytes = {ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576, ttnn.bfloat16: 2048}.get(gate.dtype, 4096)
        return -(-k_tiles // ROUTER_PARTS) * tile_bytes

    @staticmethod
    def router_column(moe, moe_mesh, in_l1: bool = True):
        """``moe``'s gate split by expert over ``moe_mesh``: chip m (in mesh order, the op's chip index) holds the
        gate's tile column m, ``[D, 32]``: in L1, height-sharded over :data:`ROUTER_PART_CORES` (its K slices), or
        DRAM interleaved. The tile columns are the gate's own tiles, so its bfloat8_b blocks carry over exactly."""
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
            memory_config=(
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(ROUTER_PART_CORES, [d // ROUTER_PARTS, 32], ttnn.ShardOrientation.ROW_MAJOR),
                )
                if in_l1
                else ttnn.DRAM_MEMORY_CONFIG
            ),
        )

    def streams_memory_config(self, request_mesh, num_streams: int = 4) -> ttnn.MemoryConfig:
        """The residual streams' ``[hc, D]`` ROW_MAJOR width-sharded L1 config, whose shards the partials buffer's
        line up with when the HC post step is fused (see :meth:`request`'s ``streams``)."""
        return width_sharded_l1_config(
            num_streams, self.hidden, request_mesh, num_cores=_HC_COLLAPSE_CORES, tile_height=1
        )

    def partials_buffer(self, request_mesh, num_chips: int):
        """A requester's partials buffer, zeroed: on every one of its chips, one L1 shard per part of the row, of
        ``num_chips`` slots of that part (the MoE chips' partials, in their order) then a flag word per chip. Fused,
        the parts sit on the stream-collapse cores in shard order (each holding those columns of the streams);
        otherwise on ``request_core``'s left neighbour (part 0) and ``request_core`` (part 1)."""
        part_bytes = self.row_bytes // self.parts
        words = num_chips * (part_bytes + 4) // 4
        if self.fuse_hc:
            cores = self.streams_memory_config(request_mesh).shard_spec.grid
        else:
            x, y = self.request_core.x, self.request_core.y
            cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x - 1, y), ttnn.CoreCoord(x, y))])
        return ttnn.from_torch(
            torch.zeros(self.parts, words, dtype=torch.int32),
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
