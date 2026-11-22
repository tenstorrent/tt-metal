# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""The persistent TP12 MoE: one ``fused_experts`` launch serving many layers over D2D sockets.

The MoE blocks of every layer it serves are loaded TP12 on ``moe_mesh`` (a 3x4 submesh), each the usual
:class:`~.moe.DeepSeekV41SparseMoeBlock` with the router and shared expert fused into ``fused_experts``. The op
is launched once, on the first layer's tensors (every layer shares their layouts), and loops on the device: it
takes ``[64-byte header | bf16 row]`` request pages from its requesters in any order (each MoE chip routes whichever
requester's page is in; chips may differ in order, see the kernels' persistent.h), the header's first word naming a
row of the layer table, runs that layer's whole MoE in its three-stage pipeline,
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
import threading
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
# Several model instances on one server (launch_instances_server): instance j is one whole TP4 model -- attention,
# norms, lm_head and its DSpark drafter -- on mesh row INSTANCE_ROWS[j], the rows nearest the MoE first, and the
# server's requester j. The five rows above the MoE hold five.
INSTANCE_ROWS = (4, 3, 2, 1, 0)
STOP_LAYER = 0xFFFFFFFF
HEADER_BYTES = 64
# The worker grid and router row the op multicasts each row over: 12 x (8 + 1).
REPLICA_GRID = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 8))])
# The op's router is sharded by expert over the MoE chips: chip m scores the gate's tile column m (experts
# [32m, 32m + 32)), K-split over the router row's first ROUTER_PARTS cores, which hold each layer's slice in L1.
ROUTER_PARTS = 10
# Every layer's router columns stay in L1 while they take at most this much of each core's L1 (an L1 buffer takes its
# address range on every core of the chip); beyond it they go to DRAM. The default holds the whole gate: all 40 layers
# take 40 * 16 bfloat8_b tiles = 680 KiB of each partial core's L1, which the MoE mesh has room for.
ROUTER_L1_BUDGET = int(os.environ.get("DEEPSEEK_V41_MOE_ROUTER_L1_BUDGET", 768 * 1024))
# Keep the router columns in L1 when the server also serves blocks (see PersistentMoE.__init__): off.
ROUTER_L1_WITH_BLOCKS = False
# DEEPSEEK_V41_MOE_FUSE_HC=0: requests return the plain reduced row (2-part reduce next to the request core) and the
# decoder layer mixes it into the streams itself (FFN_MIX), instead of the request running that post step.
FUSE_HC = os.environ.get("DEEPSEEK_V41_MOE_FUSE_HC", "1") not in ("0", "", "false", "False")
# A server with a block mode launched once for blocks, serving one-row requests too (see PersistentMoE).
MIXED_ROWS = os.environ.get("DEEPSEEK_V41_MOE_MIXED_ROWS", "1") not in ("0", "", "false", "False")
# On such a server a one-row request runs natively -- its plain row on the wire, 1x32 tiles through every kernel of the
# op -- rather than padded into a block's row 0 (DEEPSEEK_V41_MOE_NATIVE_ONE_ROW=0). Read by the ops in C++, once per
# process; mirrored here for logging.
NATIVE_ONE_ROW = os.environ.get("DEEPSEEK_V41_MOE_NATIVE_ONE_ROW", "1") != "0"
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


# The DSpark drafter's stages, served as extra layers after the target's 40: stage s is layer DSPARK_LAYER_BASE + s.
DSPARK_LAYER_BASE = 40
# A padded expert's bias: below any real one once the real biases are shifted by their max (<= 0), with scores < ~10.
_PAD_BIAS = -1.0e4


class _ServerGate:
    """The router a server layer exposes: its ``[D, 32 * chips]`` gate and its bias row ``[1, 1, 1, 32 * chips]``."""

    def __init__(self, fused_gate_weight, bias):
        self.fused_gate_weight = fused_gate_weight
        self._bias = bias

    def ranking_bias(self):
        return self._bias


class DSparkServerMoE:
    """DSpark drafter stage ``stage``'s FFN (``mtp.{stage}.ffn``: 128 routed experts, top-3, and the shared expert) as a
    layer of the target's persistent MoE server, at TP12 on ``moe_mesh``.

    The server's router is laid out for 32 experts per chip (384), so the gate is zero-padded to 384 columns and the
    padded experts' bias is far below every real one (the real bias shifted by its max, as the target's router does
    for bf16 ranking): they never reach a row's top-3, so they never run and never enter its weight normalization.
    ``server_top_k`` tells the op to keep 3 of its 6 per row (persistent.h kFeHdrTopK). The routing weights are the
    target's formula, ``route_scale * s / (sum + eps)`` over the row's top-3 unbiased ``sqrt(softplus)`` scores. The
    experts and the shared expert use the target layers' TP12 layouts (the experts are built as for a top-6 layer,
    whose 6-group layout the op runs; the top-k of 3 is the server's).
    """

    server_top_k = 3
    fused_router = True

    def __init__(self, config, loader, stage: int, moe_mesh, cache):
        from types import SimpleNamespace

        from models.experimental.deepseek_v4_flash.tt.decode.moe import FusedSharedExpert
        from models.experimental.deepseek_v41_flash.tt.config import dspark_expert_provider, dspark_weights

        num_experts = config.dspark_n_routed_experts
        moe_config = SimpleNamespace(
            **{**vars(config), "num_local_experts": num_experts, "n_routed_experts": num_experts}
        )
        weights = dspark_weights(loader, stage)
        ffn_cache = cache.sub(f"mtp.{stage}.ffn")
        self.hidden = config.hidden_size
        self.experts = DeepSeekV41PreloadedExperts(
            moe_config,
            dspark_expert_provider(loader, stage),
            moe_mesh,
            dtype=ttnn.bfloat4_b,
            cache=ffn_cache,
            tp_size=TP_SIZE,
            balance_dram_banks=True,
        )
        self.fused_shared = FusedSharedExpert(
            {
                "shared_experts.gate_proj.weight": weights["ffn.shared_experts.w1.weight"],
                "shared_experts.up_proj.weight": weights["ffn.shared_experts.w3.weight"],
                "shared_experts.down_proj.weight": weights["ffn.shared_experts.w2.weight"],
            },
            "shared_experts",
            moe_mesh,
            moe_config,
            dram_banks=self.experts.system_config.moe.fused_dram_banks,
            cache=ffn_cache,
            weight_dtype=ttnn.bfloat8_b,
            clamp_swiglu=True,
            tp_size=TP_SIZE,
        )
        chips = moe_mesh.get_num_devices()
        padded = 32 * chips
        gate = weights["ffn.gate.weight"]().float() / getattr(config, "gate_temp", 1.0)  # [E, D]
        gate_cols = torch.zeros(self.hidden, padded)
        gate_cols[:, :num_experts] = gate.t()
        bias = weights["ffn.gate.bias"]().float()
        bias_row = torch.full((padded,), _PAD_BIAS)
        bias_row[:num_experts] = bias - bias.max()
        rep = ttnn.ReplicateTensorToMesh(moe_mesh)
        self.gate = _ServerGate(
            ttnn.from_torch(gate_cols, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=moe_mesh, mesh_mapper=rep),
            ttnn.from_torch(
                bias_row.reshape(1, 1, 1, padded),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=moe_mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=rep,
            ),
        )
        # For a torch reference (tests): the gate and bias as the server ranks them, and the shared expert.
        self.reference_weights = {
            "gate.weight": gate,
            "gate.e_score_correction_bias": bias,
            "shared_experts.gate_proj.weight": weights["ffn.shared_experts.w1.weight"],
            "shared_experts.up_proj.weight": weights["ffn.shared_experts.w3.weight"],
            "shared_experts.down_proj.weight": weights["ffn.shared_experts.w2.weight"],
        }
        self.expert_provider = dspark_expert_provider(loader, stage)


def build_dspark_server_layers(config, loader, moe_mesh, cache) -> dict:
    """The drafter's stages as server layers: ``{DSPARK_LAYER_BASE + s: DSparkServerMoE}``."""
    return {
        DSPARK_LAYER_BASE + s: DSparkServerMoE(config, loader, s, moe_mesh, cache)
        for s in range(config.num_nextn_predict_layers)
    }


def launch_instances_server(
    full_device,
    config,
    loader,
    cache,
    num_layers: int,
    num_instances: int,
    dspark: bool = True,
    log=None,
    prefill: bool = False,
) -> "PersistentMoE":
    """Launch one persistent MoE for ``num_instances`` independent model instances: layers ``0 .. num_layers - 1``
    (plus, with ``dspark``, the drafter's stages as layers :data:`DSPARK_LAYER_BASE` + s) at TP12 on the 3x4 MoE
    submesh, requester j the 1x4 submesh of mesh row :data:`INSTANCE_ROWS` [j].

    Build it on a freshly opened 8x4 ``full_device``, before any model: the request sockets and partials buffers take
    L1 on every requester row before the models allocate there. Each instance is then
    ``DeepSeekV41Model(..., moe_server=server, moe_requester=j, drafter_on_stage=dspark)``, on
    ``server.request_meshes[j]``. The server takes the requesters in any order, so each instance decodes on its own.
    With ``dspark`` it is launched for 4-row blocks and takes one-row requests on the same launch (:data:`MIXED_ROWS`,
    which must be on): every instance chooses per step between a one-row decode step and a 4-row verify block (or
    draft). The caller stops it (:meth:`PersistentMoE.stop`) once every instance has shut down.

    ``prefill`` also builds the persistent TP12 prefill MoE over the same blocks, every instance a requester
    (:class:`~..prefill.persistent_moe.PersistentPrefillMoE`), as ``server.prefill_moe`` (``None`` without): each
    instance's ``build_prefill`` runs its prefill model's MoE on it, time-shared with this server. Its sockets take L1
    on the requester rows too, so it is built here, before the models. The prefill MoE's routed op needs ~1.1 MB of a
    MoE core's L1, so the router columns stay in DRAM (as a DSpark server's do anyway)."""
    if tuple(full_device.shape) != GALAXY_MESH:
        raise ValueError(f"the instances' layout needs the {GALAXY_MESH} mesh, got {tuple(full_device.shape)}")
    if not 1 <= num_instances <= len(INSTANCE_ROWS):
        raise ValueError(
            f"{len(INSTANCE_ROWS)} mesh rows hold at most {len(INSTANCE_ROWS)} instances, not {num_instances}"
        )
    if dspark and not MIXED_ROWS:
        raise ValueError(
            "independent instances mix one-row and block requests on one launch: DEEPSEEK_V41_MOE_MIXED_ROWS=1"
        )
    rows = list(INSTANCE_ROWS[:num_instances])
    cols = MOE_MESH_SHAPE[1]
    moe_mesh = full_device.create_submesh(ttnn.MeshShape(*MOE_MESH_SHAPE), ttnn.MeshCoordinate(*MOE_MESH_OFFSET))
    request_meshes = [full_device.create_submesh(ttnn.MeshShape(1, cols), ttnn.MeshCoordinate(r, 0)) for r in rows]
    start = time.perf_counter()
    moes = build_moe_layers(config, loader, moe_mesh, cache, range(num_layers), log=log)
    target_moes = dict(moes)
    if dspark:
        moes.update(build_dspark_server_layers(config, loader, moe_mesh, cache))
    global ROUTER_L1_BUDGET
    router_budget = ROUTER_L1_BUDGET
    if prefill:
        ROUTER_L1_BUDGET = 0
    try:
        server = PersistentMoE(
            moe_mesh,
            request_meshes,
            moes,
            single_senders=single_senders(rows),
            rows=1,
            block_rows=4 if dspark else 0,
        )
    finally:
        ROUTER_L1_BUDGET = router_budget
    server.prefill_moe = None
    if prefill:
        from models.experimental.deepseek_v41_flash.tt.prefill.persistent_moe import PersistentPrefillMoE

        server.prefill_moe = PersistentPrefillMoE(moe_mesh, request_meshes, target_moes, decode=server)
    logger.info(
        f"persistent MoE for {num_instances} instance(s) on mesh rows {rows}: layers 0..{num_layers - 1}"
        f"{' + the DSpark drafter stages' if dspark else ''}{', + the prefill MoE' if prefill else ''}, "
        f"request modes {server.modes} "
        f"({time.perf_counter() - start:.0f}s)"
    )
    return server


def tile_rows(block: torch.Tensor) -> torch.Tensor:
    """``block`` ``[rows, D]`` laid out as the op's ``(rows x 32)`` tiles, flat: tile t is columns
    ``[32t, 32t + 32)``, face f its 16-column half, each face row-major (rows x 16). A several-row request's row is
    this (see :class:`PersistentMoE`'s ``rows``), and so is its output."""
    rows, d = block.shape
    return block.reshape(rows, d // 32, 2, 16).permute(1, 2, 0, 3).reshape(-1)


def untile_rows(flat: torch.Tensor, rows: int) -> torch.Tensor:
    """The inverse of :func:`tile_rows`: ``[rows, D]``."""
    d = flat.numel() // rows
    return flat.reshape(d // 32, 2, rows, 16).permute(2, 0, 1, 3).reshape(rows, d)


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

    The op takes requests in any order: each requester has one request in flight at a time and sends whenever it
    likes, however many requests the others send (``TT_FE_FIXED_ORDER=1`` at launch restores the old strict turn,
    requester 0, 1, ..., then 0 again, with ``on_demand``). Use :meth:`request` (or :meth:`upload` and :meth:`request`
    for a host row), and :meth:`stop` once to end the kernels.

    ``rows`` (1, 2 or 4) makes every request a block of that many token rows -- DSpark's verify block -- each
    routed on its own, through one pass of the op: the union of the rows' top-k is fetched once and applied to every
    row (with each row's own routing weight, zero where it did not pick the expert). A block travels as its
    ``(rows x 32)`` tiles (:func:`tile_rows`), in request rows and in outputs alike, ``[1, 1, 1, rows * D]``. With
    the HC post step fused, a block's streams are ``(rows x 32)``-tiled ``[1, 1, hc * rows, D]`` (row ``h * rows +
    j`` is stream h's block row j; see :meth:`streams_memory_config` and :meth:`block_streams`) and its post and comb
    carry a tile per row, ``[1, rows, hc, 1]`` and ``[1, rows, hc, hc]``.

    With ``block_rows`` too, the server is launched once for blocks and takes either kind of request
    (:data:`MIXED_ROWS`, on by default): a one-row request goes on the requester's block socket, the page's header
    saying it is one row (``persistent_moe_request``), and its partial comes back as one row into the front of the
    requester's block partials buffer. The op runs it natively (:data:`NATIVE_ONE_ROW`): the page carries the plain
    row and every kernel takes 1x32 tiles for that request; with ``DEEPSEEK_V41_MOE_NATIVE_ONE_ROW=0`` the row rides
    row 0 of a block's tiles instead (the other rows computed and dropped). :meth:`use_rows` then only picks the
    requester-side request rows -- no relaunch -- so requesters may mix one-row and block requests freely, and no
    one-row sockets or partials are built. ``DEEPSEEK_V41_MOE_MIXED_ROWS=0`` keeps one launch per mode instead (a
    switch stops and relaunches the kernels).
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
        rows: int = 1,
        block_rows: int = 0,
        on_demand: Sequence[int] = (),
    ):
        if not isinstance(moes, Mapping):
            moes = dict(enumerate(moes))
        # Under TT_FE_FIXED_ORDER=1 (strict turn), requesters served whenever they send rather than in turn (the DSpark
        # drafter): see fe_ctrl.cpp. Each may send only while no other requester has a request outstanding. In any
        # order (the default) every requester is served whenever it sends.
        self._on_demand_mask = sum(1 << k for k in on_demand)
        if not moes:
            raise ValueError("the persistent MoE needs at least one layer")
        self.moe_mesh = moe_mesh
        self.request_meshes = list(request_meshes)
        self.layer_ids = sorted(moes)
        self.table_rows = {layer: row for row, layer in enumerate(self.layer_ids)}
        self.socket_core = socket_core or default_socket_core(moe_mesh)
        self.request_core = request_core or default_socket_core(self.request_meshes[0])
        # With the HC post step fused, one partials part per stream-collapse core; without, one per fabric link.
        # 8 rows (up to 8 experts per group) would wrap a down read across the weight CB's end.
        if rows not in (1, 2, 4) or block_rows not in (0, 1, 2, 4):
            raise ValueError(f"a request block takes 1, 2 or 4 rows, got {rows} / {block_rows}")
        self.rows = rows
        # Every request mode the server can switch to (use_rows): its sockets, partials and request rows all exist
        # from the start, since they take L1 on the requesters, which their traces fix; only the MoE mesh's
        # replica and launch change on a switch.
        self.modes = sorted({rows, block_rows} - {0})
        active_rows = rows
        # One launch for blocks serving one-row requests too (see the class docstring).
        self.mixed = MIXED_ROWS and len(self.modes) > 1
        # The rows the kernels are launched for: the block's when mixed, else the active mode's (see _launch_rows).
        self.server_rows = max(self.modes) if self.mixed else rows
        self.fuse_hc = FUSE_HC if fuse_hc is None else fuse_hc
        self.parts = _HC_COLLAPSE_CORES if self.fuse_hc else 2
        first = moes[self.layer_ids[0]]
        self.hidden = first.hidden
        # A request's row: its block's (rows x 32) tiles.
        self.row_bytes = self.hidden * 2 * rows
        self._moes = moes
        experts = first.experts
        for layer in self.layer_ids:
            moe = moes[layer]
            assert moe.fused_router and moe.fused_shared is not None, f"layer {layer}: the op runs the whole block"
            assert moe.experts.tp_size == TP_SIZE, f"layer {layer}: experts at TP{moe.experts.tp_size}"
        assert experts.fused_all_reduce_axes is not None, "the persistent op reduces over the whole mesh"

        # Each layer's gate, one [D, 32] column per MoE chip: in L1 on its partial cores (see ROUTER_PART_CORES) while
        # every layer's fits ROUTER_L1_BUDGET, else in DRAM, which the partial cores read per request.
        self.router_in_l1 = len(self.layer_ids) * self.router_column_bytes(moes[self.layer_ids[0]]) <= ROUTER_L1_BUDGET
        # With a block mode too, the columns stay in DRAM in every mode. Freeing the L1 copy for a block launch and
        # rebuilding it on the way back does not work: the block launch's program keeps L1 of its own (cached with
        # it) at the top of the freed range, so the rebuilt columns land ~100 KB lower, into the one-row launch's
        # circular buffers.
        if not ROUTER_L1_WITH_BLOCKS and max(self.modes) > 1:
            self.router_in_l1 = False
        # A block mode's launch needs the L1 the resident router columns take, so it reads them from DRAM: the
        # server keeps a DRAM copy and frees (and on the way back rebuilds) the L1 one when it switches.
        self._router_dram = None
        self._router_l1 = None
        if not self.router_in_l1 or len(self.modes) > 1:
            self._router_dram = [self.router_column(moes[layer], moe_mesh, in_l1=False) for layer in self.layer_ids]
        if self.router_in_l1:
            self._router_l1 = [self.router_column(moes[layer], moe_mesh, in_l1=True) for layer in self.layer_ids]
        logger.info(
            f"persistent MoE: router gate columns in {'L1' if self.router_in_l1 else 'DRAM'} ({len(self.layer_ids)} "
            f"layers), {self.parts} partials parts, HC post step {'fused' if self.fuse_hc else 'not fused'}, "
            f"request modes {self.modes}"
        )
        self.layer_table = None
        self._single_senders = single_senders
        # Mixed: only the block mode's sockets and partials; a one-row mode only its request rows.
        self._mode_state = {
            r: self._build_mode(r, inputs_only=self.mixed and r != self.server_rows) for r in self.modes
        }
        # Several models on one server switch the default request rows from their own threads (use_rows).
        self._lock = threading.RLock()
        self._set_mode(active_rows)
        self.scratch = self._launch()
        self.stopped = False

    def _build_mode(self, rows: int, inputs_only: bool = False) -> dict:
        """One request mode's requester-side state: per requester its request socket (pages of ``rows`` rows), its
        partials buffer and its default request row; ``inputs_only``, the request rows alone (a mixed server's one-row
        mode, which sends on the block mode's sockets into its partials)."""
        if inputs_only:
            return {
                "tx": [],
                "rx": [],
                "partials": [],
                "inputs": [self.request_input(j, rows=rows) for j in range(len(self.request_meshes))],
            }
        moe_rows, moe_cols = tuple(self.moe_mesh.shape)
        row_bytes = self.hidden * 2 * rows
        state = {"tx": [], "rx": [], "partials": [], "inputs": []}
        for j, request_mesh in enumerate(self.request_meshes):
            if tuple(request_mesh.shape)[1] != moe_cols:
                raise ValueError(f"a requester must span the MoE's {moe_cols} columns, got {tuple(request_mesh.shape)}")
            # Each MoE chip starts a row when its page arrives, and the sharded router has every chip wait for the
            # latest one, so pages sent by several requester chips start the MoE at their skew. A single sender (its
            # first chip) starts every MoE chip at once; one multicast reaches them only from the row right above.
            single = self._single_senders is not None and self._single_senders[j]
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
                self.moe_mesh,
                ttnn.SocketConfig(
                    request_connections, ttnn.SocketMemoryConfig(ttnn.BufferType.L1, HEADER_BYTES + row_bytes)
                ),
            )
            state["tx"].append(tx)
            state["rx"].append(rx)
            state["partials"].append(self.partials_buffer(request_mesh, moe_rows * moe_cols, rows=rows))
            state["inputs"].append(self.request_input(j, rows=rows))
        return state

    def _set_mode(self, rows: int) -> None:
        state = self._mode_state[rows]
        # Mixed: every request goes on the block mode's sockets, its partial into the block mode's buffers.
        block = self._mode_state[self.server_rows] if self.mixed else state
        self.rows = rows
        self.row_bytes = self.hidden * 2 * rows
        self.request_tx, self.request_rx = block["tx"], block["rx"]
        self.partials, self.inputs = block["partials"], state["inputs"]
        if not self.mixed:
            self.server_rows = rows

    def _routers_for(self, rows: int) -> list:
        """The router columns a ``rows``-row launch reads: L1 for one-row requests when they fit, else DRAM (the L1
        copy freed, since a block's circular buffers need that L1)."""
        if rows == 1 and self.router_in_l1:
            if self._router_l1 is None:
                d = self._router_dram[0].shape[-2]
                l1 = ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(ROUTER_PART_CORES, [d // ROUTER_PARTS, 32], ttnn.ShardOrientation.ROW_MAJOR),
                )
                self._router_l1 = [ttnn.to_memory_config(t, l1) for t in self._router_dram]
            return self._router_l1
        if self._router_l1 is not None:
            for t in self._router_l1:
                ttnn.deallocate(t)
            self._router_l1 = None
        return self._router_dram

    def _build_layer_table(self, routers: list) -> ttnn.Tensor:
        rows = []
        for row, layer in enumerate(self.layer_ids):
            moe = self._moes[layer]
            header = [
                row,
                moe.fused_shared.gate_up.buffer_address(),
                moe.fused_shared.down.buffer_address(),
                routers[row].buffer_address(),
                moe.gate.ranking_bias().buffer_address(),
                # kFeHdrTopK: a layer picking fewer than the op's top_k per row (the drafter's).
                getattr(moe, "server_top_k", 0),
                0,
                # kFeHdrOnDemand
                self._on_demand_mask,
            ]
            header += [0] * (HEADER_BYTES // 4 - len(header))
            # A layer with fewer experts than the op's (the drafter's 128 of 384) names expert 0 for the rest: its
            # router never picks them (see DSparkServerMoE), so they are never read.
            num_weights = len(self._moes[self.layer_ids[0]].experts.gate_up_weights)
            gate_up = [w.buffer_address() for w in moe.experts.gate_up_weights]
            down = [w.buffer_address() for w in moe.experts.down_weights]
            gate_up += gate_up[:1] * (num_weights - len(gate_up))
            down += down[:1] * (num_weights - len(down))
            rows.append(_as_int32(header + gate_up + down))
        return ttnn.from_torch(
            torch.stack(rows),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.moe_mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.moe_mesh),
        )

    def _launch(self):
        """Launch the op for the current mode's rows, on a fresh replica and layer table (the MoE mesh runs no
        traces, so its buffers may change between launches)."""
        first = self._moes[self.layer_ids[0]]
        experts, shared = first.experts, first.fused_shared
        old = getattr(self, "replica", None)
        if old is not None and old.is_allocated():
            ttnn.deallocate(old)
        self.replica = None
        self.router_columns = self._routers_for(self.server_rows)
        if self.layer_table is not None:
            ttnn.deallocate(self.layer_table)
        self.layer_table = self._build_layer_table(self.router_columns)
        # The row lands here on every core of the grid, replacing the all_gather_for_matmul replica.
        self.replica = ttnn.from_torch(
            torch.zeros(1, 1, REPLICA_GRID.num_cores() * self.server_rows, self.hidden),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.moe_mesh,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(REPLICA_GRID, [self.server_rows, self.hidden], ttnn.ShardOrientation.ROW_MAJOR),
            ),
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.moe_mesh),
        )
        # A block's partial: its rows' tiles over the same output cores.
        output_memory_config = experts.output_memory_config
        if self.server_rows > 1:
            spec = output_memory_config.shard_spec
            output_memory_config = ttnn.MemoryConfig(
                output_memory_config.memory_layout,
                output_memory_config.buffer_type,
                ttnn.ShardSpec(spec.grid, [self.server_rows, spec.shape[1]], spec.orientation),
            )
        return ttnn.experimental.deepseek.moe.fused_experts(
            self.replica,
            router_weight=self.router_columns[0],
            router_bias=first.gate.ranking_bias(),
            gate_up_weights=experts.gate_up_weights,
            down_weights=experts.down_weights,
            # The union of the block's selections is at most rows * top_k experts.
            num_experts=experts.top_k * self.server_rows,
            intermediate_size=experts.intermediate,
            swiglu_limit=experts.limit,
            top_k=experts.top_k,
            routed_scaling_factor=experts.routed_scaling_factor,
            routing_eps=experts.routing_eps,
            experts_block_size=experts.experts_block_size,
            shared_gate_up_weights=shared.gate_up,
            shared_down_weights=shared.down,
            shared_swiglu_limit=shared.swiglu_limit,
            memory_config=output_memory_config,
            all_reduce_cluster_axes=experts.fused_all_reduce_axes,
            all_reduce_num_links=2,
            all_reduce_topology=CCL_TOPOLOGY,
            persistent_layer_table=self.layer_table,
            persistent_socket_core=self.socket_core,
            persistent_input_sockets=self.request_rx,
            persistent_partials=self.partials,
        )

    def use_rows(self, rows: int) -> bool:
        """Serve requests of ``rows`` rows from now on (one of :attr:`modes`): stop the kernels (every requester's
        queue must reach the stop, so earlier requests finish first) and relaunch them for that mode. Returns whether
        it switched."""
        if rows not in self._mode_state:
            raise ValueError(f"the server was built for {self.modes}-row requests, not {rows}")
        if self.mixed:
            # One launch takes both: only the requester side's request rows change (the default request rows and
            # streams configs; requests that pass their own tensor carry their rows).
            with self._lock:
                self._set_mode(rows)
            return False
        if rows == self.rows and not self.stopped:
            return False
        if not self.stopped:
            self.stop()
        self._set_mode(rows)
        self.start()
        return True

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

    def streams_memory_config(self, request_mesh, num_streams: int = 4, rows: int | None = None) -> ttnn.MemoryConfig:
        """The residual streams' width-sharded L1 config, whose shards the partials buffer's line up with when the HC
        post step is fused (see :meth:`request`'s ``streams``): ``[hc, D]`` ROW_MAJOR for one row; for a block,
        ``[hc * rows, D]`` in ``(rows x 32)`` tiles, so a shard holds each stream's columns of the block in turn."""
        rows = self.rows if rows is None else rows
        return width_sharded_l1_config(
            num_streams * rows, self.hidden, request_mesh, num_cores=_HC_COLLAPSE_CORES, tile_height=rows
        )

    def block_streams(self, request_mesh, streams: torch.Tensor):
        """Host ``streams`` -- ``[hc, D]`` for one row, ``[hc, rows, D]`` for a block -- on every chip of
        ``request_mesh``, as :meth:`request`'s ``streams`` takes them."""
        hc = streams.shape[0]
        rows_kwargs = {} if self.rows == 1 else {"tile": ttnn.Tile((self.rows, 32))}
        return ttnn.from_torch(
            streams.reshape(1, 1, hc * self.rows, self.hidden),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT if self.rows == 1 else ttnn.TILE_LAYOUT,
            device=request_mesh,
            memory_config=self.streams_memory_config(request_mesh, hc),
            mesh_mapper=ttnn.ReplicateTensorToMesh(request_mesh),
            **rows_kwargs,
        )

    def partials_buffer(self, request_mesh, num_chips: int, rows: int | None = None):
        """A requester's partials buffer, zeroed: on every one of its chips, one L1 shard per part of the row, of
        ``num_chips`` slots of that part (the MoE chips' partials, in their order) then a flag word per chip. Fused,
        the parts sit on the stream-collapse cores in shard order (each holding those columns of the streams);
        otherwise on ``request_core``'s left neighbour (part 0) and ``request_core`` (part 1)."""
        part_bytes = self.hidden * 2 * (self.rows if rows is None else rows) // self.parts
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

    def request_input(self, requester: int, rows: int | None = None):
        """A ``[1, 1, 1, rows * D]`` request row on every chip of ``requester``, zeroed: the DRAM-interleaved one-page
        ROW_MAJOR bf16 row :meth:`request` sends (a block's tiles, see :func:`tile_rows`)."""
        mesh = self.request_meshes[requester]
        return ttnn.from_torch(
            torch.zeros(1, 1, 1, self.hidden * (self.rows if rows is None else rows)),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

    def upload(self, requester: int, row: torch.Tensor, tensor=None):
        """``row`` ``[D]``, or a block ``[rows, D]``, into the requester's request row."""
        if self.rows > 1:
            row = tile_rows(row.reshape(self.rows, self.hidden))
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
        ``post[h] * row + sum_g comb[g, h] * streams[g]`` (``mix_streams``), in ``streams``' spec. For a block of
        ``rows`` rows: ``streams`` ``[1, 1, hc * rows, D]`` in ``(rows x 32)`` tiles (row ``h * rows + j``),
        ``post`` ``[1, rows, hc, 1]``, ``comb`` ``[1, rows, hc, hc]``, each row j mixed with its own post and comb."""
        moe = self._moes.get(layer)
        if getattr(moe, "server_top_k", 0) and self.server_rows == 1:
            # A drafter stage takes a block (its union of top-3 runs on the several-row grid); in one-row mode the
            # server has no block socket listening, so refuse rather than send into it.
            raise ValueError(
                f"layer {layer} (a DSpark drafter stage) needs a block mode; the server is in one-row mode"
            )
        # A tensor's width says its rows: on a mixed server either mode's (requesters pick their own), else the mode's.
        allowed = self.modes if self.mixed else [self.rows]
        if tensor is not None and not any(tensor.shape[-1] == self.hidden * r for r in allowed):
            raise ValueError(
                f"a request takes [1, 1, 1, rows * {self.hidden}] for rows in {allowed}, got {list(tensor.shape)}"
            )
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
