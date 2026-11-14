# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import contextvars
import os

from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole


def mesh_tp_axis(mesh_shape=None) -> int:
    """Mesh axis of the plain (non-fractured) tensor-parallel group: ``GEMMA4_TP_AXIS`` when set, else 0 on a
    column mesh (several rows, one column) and 1 (the columns) everywhere else (``mesh_orientation.tp_axis_for``).

    TP over the mesh rows is what a column mesh such as the 4-chip line opened as ``MeshShape(4, 1)`` needs: the
    DeepSeek token-dispatch / combine ops only run along mesh axis 0, so a (4,1) mesh with the TP group on axis 0
    can dispatch its routed-expert prefill while the (1,4) orientation cannot. The Galaxy fracture path picks its
    own axis and ignores this knob.
    """
    from models.demos.gemma4.tt.mesh_orientation import tp_axis_for

    return tp_axis_for(mesh_shape)


def mesh_tp(mesh_device) -> int:
    """Dense TP width of ``mesh_device`` under ``mesh_tp_axis()`` (1 on a single device)."""
    if not hasattr(mesh_device, "shape"):
        return 1
    return int(mesh_device.shape[mesh_tp_axis(tuple(mesh_device.shape))])


def default_num_links():
    """Default TP-collective link count for the current arch.

    Blackhole boards expose 2 ethernet links between adjacent mesh devices, so
    reduce-scatter / all-gather can run at ~2x bandwidth vs a single link — and
    on Gemma4 prefill the per-layer all-reduces are ~31% of device time, so this
    is the single highest-ROI CCL knob. Wormhole (T3K) defaults to 1 link here
    (its multi-link tuning needs a separate sweep).

    Override with ``GEMMA4_CCL_NUM_LINKS``.
    """
    env = os.environ.get("GEMMA4_CCL_NUM_LINKS")
    if env is not None:
        return max(1, int(env))
    return 2 if is_blackhole() else 1


def ccl_chunks_per_sync() -> int:
    """Async RS/AG ``chunks_per_sync`` (fabric packet grouping). Default 10."""
    return max(1, int(os.environ.get("GEMMA4_CCL_CHUNKS_PER_SYNC", "10")))


def ccl_num_workers_per_link() -> int:
    """Async RS/AG workers per link. Default 2."""
    return max(1, int(os.environ.get("GEMMA4_CCL_NUM_WORKERS", "2")))


def ccl_num_buffers_per_channel() -> int:
    """Async RS/AG ``num_buffers_per_channel``. Default 2."""
    return max(1, int(os.environ.get("GEMMA4_CCL_NUM_BUFFERS", "2")))


def ccl_persistent_buffers_enabled() -> bool:
    """Reuse DRAM destination buffers across RS/AG calls (Phase P1).

    Default on for async path; disable with ``GEMMA4_CCL_PERSISTENT_BUF=0``.
    Sync ``ttnn.all_reduce`` ignores this (no persistent buffer API).
    """
    return os.environ.get("GEMMA4_CCL_PERSISTENT_BUF", "1").lower() not in ("0", "false", "no")


def lane_ccl_async_enabled() -> bool:
    """GEMMA4_LANE_CCL_ASYNC=1: the lane-axis gather / scatter run as the async collectives with the CCLManager's
    pre-created global semaphores and a barrier semaphore, like the TP all-reduce, instead of the generic ops that
    create their own semaphores per program. Structural fix for the (2,4) line-fabric race (tt-triage 2026-10-24:
    a Linear lane-axis reduce-scatter completes on the end columns first, the middle columns enter the TP collective
    unaligned and a late remote write of the previous collective corrupts a reused L1 semaphore): the barrier aligns
    the participants before any data write and the pool depth (>= 4 under this flag) keeps consecutive collectives on
    distinct semaphores. Valid inside traces (no host synchronize). Default off until the block kernel gate validates
    it; the served Galaxy keeps the generic ops."""
    return os.environ.get("GEMMA4_LANE_CCL_ASYNC", "0") == "1"


def ccl_semaphore_pool_depth() -> int:
    """Rotation depth of the CCLManager's rs / ag / barrier semaphore pools (GEMMA4_CCL_SEM_POOL_DEPTH, default 2;
    at least 4 with GEMMA4_LANE_CCL_ASYNC=1, where the lane collectives share the pools with the TP all-reduce)."""
    depth = max(2, int(os.environ.get("GEMMA4_CCL_SEM_POOL_DEPTH", "2")))
    return max(depth, 4) if lane_ccl_async_enabled() else depth


def allreduce_gather_sum_enabled() -> bool:
    """GEMMA4_CCL_ALLREDUCE_AG=1: the TP all-reduce of decode rows (<= one row tile, 180 KB per chip on the Galaxy)
    as ONE all_gather_async over the TP axis plus a local sum, instead of reduce_scatter_minimal_async + all_gather_async
    (49 + 44 us per all-reduce, 2 per layer = 5.6 ms of the 32.6 ms step, window-2/3 profiles 2026-10-24). At this
    message size the collectives are latency-bound: one collective's hops instead of two, the sum is ~5 us. Opt-in;
    validated per arm by tests/unit/test_layer_lanes_traced.py (traced one-layer step delta + parity)."""
    return os.environ.get("GEMMA4_CCL_ALLREDUCE_AG", "0") == "1"


# ---------------------------------------------------------------------------------------------------------------
# One-shot decode all-reduce (GEMMA4_CCL_ONESHOT): the row-parallel decode matmuls (attention o_proj, MLP down_proj)
# keep their partial sums in the DRAM-sharded kernel's L1 width-sharded output layout and ONE
# ``ttnn.experimental.all_reduce_async(x, scratch, ...)`` gathers every chip's shards into an L1 scratch tensor and
# reduces them locally (the ``gemma4_12b_qb2`` demo's ``MeshCCL.all_reduce``), instead of the reduce-scatter +
# all-gather composite (``sync_all_reduce``). Two slots, "o" and "down", each with a pair of rotating global
# semaphores created with the manager: consecutive uses of one slot are separated by the other slot's all-reduce
# (o -> down -> next layer's o), which no chip completes before every chip has finished reading the first slot, and
# the receiver resets its semaphore after the wait, so a rotation of two also covers back-to-back uses of one slot.
# The scratch (``[1, 1, 32, width * tp]``, shard ``(32, shard_width * tp)`` on the input's grid: 120 KB per core
# for the 12B o_proj at TP=4, 2 x that at TP=8) is allocated around the op like any intermediate
# (``allocate_tensor_on_device``: no host data, free inside a trace replay) rather than kept resident: a resident
# scratch lowers the L1 top for every program, and the prefill matmul slabs (``dram_sharded.prefill_slab_rows``)
# are sized to the L1 without it (measured: the 12B 8x10-grid prefill matmul's CBs clash with a resident 144 KB).
# The op's constraints (device op validate): input, scratch and output WIDTH_SHARDED in L1 (Blackhole refuses a DRAM
# input), an even ring size, the scratch grid containing the output grid with a shard of at least TP x the output
# shard, and ``num_links`` worker cores outside the output grid's bounding box; the output is bf16 in the input's
# layout and goes back to interleaved for the norms and residual adds. Generic over the TP group: the 4-chip QB2
# mesh (TP=4) and the (1,8) Galaxy DP4 engine (TP=8, a Linear line) take the same path with ``mesh_config.tp``.
ONESHOT_SLOTS = ("o", "down")
# Mesh shapes whose builds take the one-shot all-reduce by default (default_ccl_oneshot); the measured basis is in
# its docstring. A shape joins when a served knob A/B measures a decode win with the PCC gates intact.
# (1,8) is OFF: on the Blackhole Galaxy DP4 engine the one-shot op garbles long-context decode on two of the four
# rows (2026-11-10, 31B, rows TT_VISIBLE_DEVICES 4-7/28-31 and 12-15/20-23: the 32K needle answers with a correct
# first token, then garbage; rows 0 and 3 and every 1K prompt clean; GEMMA4_CCL_ONESHOT=0 alone restores all four
# rows at a 1-3 % decode cost). The prose / PCC / 128-1K gates that qualified it never decode at long context.
_ONESHOT_DEFAULT_ON_SHAPES = ((1, 4),)


def default_ccl_oneshot(mesh_shape=None) -> bool:
    """Default of the one-shot decode all-reduce for a mesh shape (``GEMMA4_CCL_ONESHOT=0|1`` overrides).

    Policy: ON only on Blackhole for the shapes in ``_ONESHOT_DEFAULT_ON_SHAPES``, i.e. where a knob A/B on the
    served model measured a decode gain with every PCC gate intact; OFF everywhere else (unmeasured shapes, the lane
    engine, the fractured Galaxy, and Wormhole: the T3K shares the (1,8) shape and is unmeasured).

    Measured basis for (1,4), 12B on the 4-chip Blackhole QB2 mesh (2026-11-06, tt-metal-port, bf16 payload):
    traced B=1 decode step 17.59 -> 15.67 ms on the Linear line and 16.93 -> 15.31 ms on the ring with 2 links
    (-10.9 / -9.6 %); per all-reduce call 21 -> 38 us (line) and 18 -> 28 us (ring) at the 3840-wide layouts,
    11 -> 28 us at the 1024-wide MTP drafter. Served knob A/B on the ring entry (spec records on): prose c1 / c8 /
    c16 +12 / +7 / +5 %, sampled c1 +17 %, greedy 128/128 c1 +9 / +28 % and 128/1K c1 +33 % (a second boot pair;
    the first pair's greedy was flat) with first-token latency equal or lower (128/128 medians 87.8 / 98.7 ->
    88.8 / 72.2 ms, 128/1K 91.7 -> 78.1 ms; the first pair's +23 ms was boot-to-boot variance of the spec host path,
    whose release -> first-token chain runs no one-shot op), coherence pass, 0 throws. Decode PCC set (shared_mlp,
    layer, 24 batched-attention nodes, full model) vs the composite on the same tree: 28 / 30 nodes equal or higher
    (full-model decode 0.9697 -> 0.9735), two attention nodes 1e-6 and 2e-6 lower (bf16 summation order, below the
    thresholds' precision; with fp32 destination accumulation 29 / 30 and one node 6e-6 lower at the same step
    time); the bfp8 payload lowers 26 / 30 nodes by 1e-5..9e-5 and stays excluded.

    Measured basis for (1,8), the Blackhole Galaxy DP4 engine (2026-11-07, the six port knobs together on the
    BLACKHOLE_GALAXY entries, same seeds, 0 throws): 12B greedy 128/1K c1 68.6 -> 89.0 tok/s/u, sampled c1 +10 %,
    prose c1 / c16 / c32 +8 / +14 / +7 %, TTFT lower at every point, GPQA subset 31/40 = base; 31B greedy c1
    51.8 -> 55.8, sampled c1 +5 %, prose c1 / c16 / c32 +6 / +8 / +9 %, greedy after sampled +21 %, GPQA subset
    65/80 with the six knobs vs 66/80 base over two rounds (five knobs without the wide grid 34/40, without the
    one-shot 36/40: all within noise). The P150x8 Loudbox shares the arch and the shape and is unmeasured; the
    T3K (Wormhole) is excluded by the arch guard.
    """
    if mesh_shape is None or not is_blackhole():
        return False
    return tuple(int(d) for d in mesh_shape) in _ONESHOT_DEFAULT_ON_SHAPES


def ccl_oneshot_enabled(mesh_shape=None) -> bool:
    """Whether the decode TP all-reduce of the row-parallel partial sums runs as the one-shot ``all_reduce_async``
    (see the section comment above): ``GEMMA4_CCL_ONESHOT=1|0`` when set, else ``default_ccl_oneshot(mesh_shape)``.
    Only the plain-TP decode step takes it (``ccl_oneshot_applicable``: TP > 1 and even, no lanes, no fracture, one
    row tile, the matmul's L1 width-sharded output), where it takes precedence over GEMMA4_CCL_ALLREDUCE_AG and
    GEMMA4_CCL_ASYNC; every other all-reduce (prefill, the lane engine, the fractured Galaxy, the drafter) is
    unchanged. The payload dtype is GEMMA4_CCL_ONESHOT_DTYPE; the topology and link count are the CCLManager's."""
    env = os.environ.get("GEMMA4_CCL_ONESHOT", "").strip()
    if env != "":
        return env == "1"
    return default_ccl_oneshot(mesh_shape)


def ccl_oneshot_dtype():
    """GEMMA4_CCL_ONESHOT_DTYPE=bf16|bfp8 (default bf16): the dtype of the one-shot all-reduce payload, i.e. of the
    row-parallel matmul's partial-sum output and of the persistent scratch (the reduced result is always bf16). bfp8
    halves the scratch and the fabric traffic (the reference demo measured -5.4 us per layer) at the partial sums'
    precision; keep it only where the PCC gates hold."""
    name = os.environ.get("GEMMA4_CCL_ONESHOT_DTYPE", "bf16").strip().lower()
    if name in ("bfp8", "bfloat8_b", "bf8"):
        return ttnn.bfloat8_b
    if name in ("bf16", "bfloat16", ""):
        return ttnn.bfloat16
    raise ValueError(f"GEMMA4_CCL_ONESHOT_DTYPE={name!r}: expected bf16 or bfp8")


def ccl_oneshot_fp32_acc() -> bool:
    """GEMMA4_CCL_ONESHOT_FP32_ACC=1: fp32 destination accumulation in the one-shot reduction (the Llama lm_head
    all-reduce's remedy for the arrival-order dependence of a bf16 cross-chip sum). Default off."""
    return os.environ.get("GEMMA4_CCL_ONESHOT_FP32_ACC", "0") == "1"


def oneshot_wanted(mesh_config, ccl_manager) -> bool:
    """Whether the decode row-parallel matmuls of this build should hand their partial sums to the one-shot
    all-reduce (the knob, a manager, a plain TP group of an even width: no lanes, no fracture). Decides the matmul's
    ``keep_sharded`` / payload dtype at the call sites; ``ccl_oneshot_applicable`` then checks the tensor."""
    if ccl_manager is None or mesh_config is None:
        return False
    if not ccl_oneshot_enabled(mesh_shape=getattr(mesh_config, "mesh_shape", None)):
        return False
    if getattr(mesh_config, "lane_sharded", False) or getattr(mesh_config, "weight_fracture", False):
        return False
    tp = int(getattr(mesh_config, "tp", 1))
    return tp > 1 and tp % 2 == 0


def oneshot_matmul_kwargs(mesh_config, ccl_manager) -> dict:
    """The ``DramShardedLinear`` call kwargs that produce the one-shot payload (``keep_sharded=True`` and the payload
    dtype), or ``{}`` when the build does not take the one-shot path (``oneshot_wanted``)."""
    if not oneshot_wanted(mesh_config, ccl_manager):
        return {}
    return {"keep_sharded": True, "dtype": ccl_oneshot_dtype()}


def ccl_oneshot_applicable(tensor, mesh_config, ccl_manager) -> bool:
    """Whether ``tensor`` (a row-parallel matmul's partial sums) can go through the one-shot all-reduce: the build
    takes it (``oneshot_wanted``), one row tile of a 4D activation, L1 WIDTH_SHARDED, and the output grid's bounding
    box leaves the op its ``num_links`` worker cores. Anything else takes the composite path."""
    if not oneshot_wanted(mesh_config, ccl_manager):
        return False
    shape = tensor.shape
    if len(shape) != 4 or int(shape[0]) != 1 or int(shape[1]) != 1 or not (1 <= int(shape[2]) <= ttnn.TILE_SIZE):
        return False
    if int(tensor.padded_shape[-2]) != ttnn.TILE_SIZE or tensor.layout != ttnn.TILE_LAYOUT:
        return False
    mem = tensor.memory_config()
    if (
        not mem.is_sharded()
        or mem.memory_layout != ttnn.TensorMemoryLayout.WIDTH_SHARDED
        or mem.buffer_type != ttnn.BufferType.L1
        or mem.shard_spec is None
    ):
        return False
    grid = ccl_manager.mesh_device.compute_with_storage_grid_size()
    bbox = mem.shard_spec.grid.bounding_box()
    bbox_cores = (bbox.end.x - bbox.start.x + 1) * (bbox.end.y - bbox.start.y + 1)
    return bbox_cores + int(ccl_manager.num_links) <= grid.x * grid.y


def _restore_rows(out, rows):
    """A one-row-tile result presented as its ``rows`` logical rows again (the DRAM-sharded decode kernel runs the
    tile as 32 rows under GEMMA4_DRAM_SHARD_ROW_VIEW; a metadata reshape, as ``DramShardedLinear`` does)."""
    if rows is None or int(out.shape[-2]) == int(rows):
        return out
    n = int(out.shape[-1])
    return ttnn.reshape(out, ttnn.Shape([1, 1, int(rows), n]), ttnn.Shape([1, 1, ttnn.TILE_SIZE, n]))


def oneshot_all_reduce(tensor, slot, axis, memory_config, ccl_manager, tp, rows=None):
    """The one-shot all-reduce of the L1 width-sharded partial sums ``tensor`` over ``tp`` chips on mesh ``axis``
    through a scratch allocated around the op and ``slot``'s semaphore; the bf16 result in ``memory_config``
    (interleaved, for the norms and residual adds) with ``rows`` logical rows. Consumes ``tensor``."""
    in_mem = tensor.memory_config()
    scratch = ccl_manager.oneshot_scratch(in_mem, tensor.dtype, tp)
    reduced = ttnn.experimental.all_reduce_async(
        tensor,
        scratch,
        cluster_axis=axis,
        mesh_device=ccl_manager.mesh_device,
        multi_device_global_semaphore=ccl_manager.oneshot_semaphore(slot),
        dtype=ttnn.bfloat16,
        memory_config=in_mem,
        topology=ccl_manager.topology,
        num_links=ccl_manager.num_links,
        fp32_dest_acc=ccl_oneshot_fp32_acc(),
    )
    tensor.deallocate(True)
    scratch.deallocate(True)
    out = ttnn.to_memory_config(reduced, memory_config)
    reduced.deallocate(True)
    return _restore_rows(out, rows)


def default_ccl_topology(mesh_device=None):
    """Default CCL topology for Gemma4 TP collectives.

    Override with ``GEMMA4_CCL_TOPOLOGY=ring|linear``.

    Policy (when env unset):
      * **Ring** only on **Blackhole** meshes with **≥8 devices** (P150x8 TTFT
        sweep: Ring+sync ~28.8s vs Linear+sync ~31.0s @ 31B/128k).
      * **Linear** everywhere else — including Wormhole T3K 1x8. Ring on WH
        drops 26B-A4B ``test_full_model`` PCC below the TEMP 0.76 gate
        (~0.7505 vs ~0.77/0.94 with Linear / main). Ring on 4-device BH also
        drops 12B full-model PCC (~0.97 → ~0.90).

    Async RS+AG is correct but slower than sync on P150x8 — keep
    ``GEMMA4_CCL_ASYNC=0`` unless re-swept.
    """
    override = os.environ.get("GEMMA4_CCL_TOPOLOGY", "").strip().lower()
    if override in ("ring", "r"):
        return ttnn.Topology.Ring
    if override in ("linear", "line", "l"):
        return ttnn.Topology.Linear

    n = mesh_device.get_num_devices() if mesh_device is not None else 0
    # Ring TTFT win was swept on BH P150x8 only. WH T3K is also n=8 but must
    # stay Linear for MoE PCC (matches main's hardcoded Linear all-reduce).
    if n:
        if n >= 8 and is_blackhole():
            return ttnn.Topology.Ring
        return ttnn.Topology.Linear

    try:
        cluster = ttnn.cluster.get_cluster_type()
    except Exception:
        cluster = None

    # No mesh_device: Ring only on full 8-device BH LoudBox / BH Galaxy.
    # Do not treat WH T3K / Galaxy cluster types as Ring defaults.
    ring_when_unknown_n = ()
    for name in ("P150_X8", "BLACKHOLE_GALAXY"):
        if hasattr(ttnn.cluster.ClusterType, name):
            ring_when_unknown_n += (getattr(ttnn.cluster.ClusterType, name),)
    if cluster in ring_when_unknown_n:
        return ttnn.Topology.Ring
    return ttnn.Topology.Linear


def ccl_async_enabled() -> bool:
    """True when prefill/decode allreduce should use async RS+AG.

    Default off until measured green on the target board; enable with
    ``GEMMA4_CCL_ASYNC=1``.
    """
    return os.environ.get("GEMMA4_CCL_ASYNC", "0").lower() in ("1", "true", "yes")


class CCLManager:
    """CCL manager for Gemma4 tensor parallelism.

    Stores mesh_device, num_links, and topology for CCL operations.
    Semaphores support the async RS+AG path (``GEMMA4_CCL_ASYNC=1``).
    Persistent DRAM buffers (``GEMMA4_CCL_PERSISTENT_BUF``) are keyed by shape
    so repeated collectives of the same activation shape skip realloc+barrier.
    """

    def __init__(self, mesh_device, num_links=None, topology=None):
        if num_links is None:
            num_links = default_num_links()
        if topology is None:
            topology = default_ccl_topology(mesh_device)
        self.mesh_device = mesh_device
        self.num_links = num_links
        self.topology = topology
        self.num_devices = mesh_device.get_num_devices()
        topo_name = "Ring" if topology == ttnn.Topology.Ring else "Linear"
        logger.info(
            f"Gemma4 CCLManager: devices={self.num_devices} num_links={num_links} "
            f"topology={topo_name} async={int(ccl_async_enabled())} "
            f"persistent_buf={int(ccl_persistent_buffers_enabled())}"
        )

        grid = mesh_device.compute_with_storage_grid_size()
        num_cores = grid.x * grid.y
        core_range_set = ttnn.num_cores_to_corerangeset(num_cores, grid, row_wise=True)

        # One semaphore pool PER MESH AXIS. The owner path runs a lane-axis (row) all-reduce immediately
        # followed by a TP-axis (column) one on the same worker cores; a barrier word shared by both axes
        # has two unordered senders (the row and the column neighbour), and the reduce-scatter writer's
        # reset-after-wait wipes the other axis's increment when the rows are skewed -> the column
        # reduce-scatter waits forever (the lane engine's served hangs: tt-triage 2026-10-25 21:07 and
        # 2026-10-26 01:14, column chips 0 and 5 stuck at the pairwise start barrier with the partner's
        # increment sent). Per-axis pools keep one sender per launch; within one axis the data
        # dependencies order the launches. Keyed by the cluster axis; None serves callers that name none.
        depth = ccl_semaphore_pool_depth()

        def _create():
            return ttnn.create_global_semaphore(mesh_device, core_range_set, 0)

        self._sem_pools = {}
        for key in (None, 0, 1):
            self._sem_pools[key] = {
                "rs": [[_create() for _ in range(3)] for _ in range(depth)],
                "ag": [[_create() for _ in range(3)] for _ in range(depth)],
                "barrier": [_create() for _ in range(depth)],
                "rs_idx": 0,
                "ag_idx": 0,
                "barrier_idx": 0,
            }
        # One-shot decode all-reduce (ccl_oneshot_enabled): a semaphore pair per slot here, with the other pools
        # (before any trace capture); the scratch is allocated around each op (oneshot_scratch).
        oneshot = ccl_oneshot_enabled(mesh_shape=tuple(int(d) for d in mesh_device.shape))
        self._oneshot_sems = {name: [_create() for _ in range(2)] for name in ONESHOT_SLOTS} if oneshot else {}
        self._oneshot_idx = {name: 0 for name in self._oneshot_sems}
        self._oneshot_layouts: set = set()  # (shard width, cores, dtype) layouts seen, for one log line each
        ttnn.synchronize_device(mesh_device)
        logger.info(
            f"Gemma4 CCLManager: semaphore pools per axis {sorted(str(k) for k in self._sem_pools)} depth={depth} "
            f"({len(self._sem_pools) * depth * 7} global semaphores)"
        )
        # shape_key -> ttnn.Tensor (DRAM interleaved zeros)
        self._persistent_ag: dict = {}
        self._persistent_rs_out: dict = {}
        self._persistent_rs_inter: dict = {}

    def _pool(self, axis):
        """The semaphore pool of a cluster axis (None for callers that name no axis; an axis without its own
        pool falls back to it)."""
        key = None if axis is None else int(axis)
        return self._sem_pools.get(key, self._sem_pools[None])

    def get_rs_semaphore(self, axis=None):
        """Returns list of 3 semaphores for reduce_scatter on ``axis`` (cycles that axis's pool; depth
        ``ccl_semaphore_pool_depth``)."""
        pool = self._pool(axis)
        sems = pool["rs"][pool["rs_idx"]]
        pool["rs_idx"] = (pool["rs_idx"] + 1) % len(pool["rs"])
        return sems

    def get_ag_semaphore(self, axis=None):
        """Returns list of 3 semaphores for all_gather on ``axis`` (cycles that axis's pool)."""
        pool = self._pool(axis)
        sems = pool["ag"][pool["ag_idx"]]
        pool["ag_idx"] = (pool["ag_idx"] + 1) % len(pool["ag"])
        return sems

    def get_barrier_semaphore(self, axis=None):
        """Returns single barrier semaphore for a collective on ``axis`` (cycles that axis's pool)."""
        pool = self._pool(axis)
        sem = pool["barrier"][pool["barrier_idx"]]
        pool["barrier_idx"] = (pool["barrier_idx"] + 1) % len(pool["barrier"])
        return sem

    def get_all_reduce_semaphores(self, axis=None):
        """(barrier pair, rs triple, ag triple) for ``all_reduce_async``.

        All of them were created in ``__init__``, before any trace capture, so the collective
        allocates no semaphore of its own at call time: ``ttnn.all_reduce`` creates fresh ones per
        cached program, and a program first compiled after a decode trace capture would put them
        in the worker L1 the replay overwrites.
        """
        # The reduce-scatter half takes three semaphores, the default all-gather
        # implementation exactly two (the manager's gather pool holds three for
        # the async path's sharded variant).
        pool = self._pool(axis)
        return list(pool["barrier"]), self.get_rs_semaphore(axis), self.get_ag_semaphore(axis)[:2]

    def _shape_key(self, shape, dtype, memory_config, axis=None):
        # axis distinguishes same-shape buffers used by collectives on different
        # cluster axes (per-device shard layout differs even when shapes match).
        return (tuple(int(x) for x in shape), str(dtype), str(memory_config), axis)

    def _alloc_like(self, ref_tensor, memory_config):
        return ttnn.zeros_like(ref_tensor, device=self.mesh_device, memory_config=memory_config)

    def get_persistent_ag_buffer(self, scattered, memory_config, tp, axis=None):
        """Allocate a persistent AG destination sized by TP group width.

        Disabled by default in ``ccl_allreduce`` / ``ccl_allgather``: the gathered
        result is returned as a normal activation and Gemma4 force-deallocates
        those, which would free a manager-cached buffer. Kept for opt-in / tests.
        """
        if not ccl_persistent_buffers_enabled():
            return None
        if tp <= 1:
            return None
        # All-gather expands dim=3 by the TP group size (cluster_axis width).
        out_shape = list(scattered.shape)
        out_shape[3] = int(out_shape[3]) * tp
        key = self._shape_key(out_shape, scattered.dtype, memory_config, axis)
        buf = self._persistent_ag.get(key)
        if buf is None:
            buf = ttnn.zeros(
                out_shape,
                dtype=scattered.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=memory_config,
            )
            self._persistent_ag[key] = buf
            logger.debug(f"CCL persistent AG buffer allocated shape={out_shape}")
        return buf

    def get_persistent_rs_buffers(self, tensor, memory_config, tp, axis=None):
        if not ccl_persistent_buffers_enabled():
            return None
        if tp <= 1:
            return None
        # Reduce-scatter shrinks dim=3 by TP group size (not full mesh size).
        out_shape = list(tensor.shape)
        out_shape[3] = int(out_shape[3]) // tp
        # Linear topology needs a leading size-2 dim for forward/backward streams.
        inter_shape = list(tensor.shape)
        if self.topology == ttnn.Topology.Linear:
            inter_shape = [2] + inter_shape
        inter_key = self._shape_key(inter_shape, tensor.dtype, ttnn.DRAM_MEMORY_CONFIG, axis)
        out_key = self._shape_key(out_shape, tensor.dtype, memory_config, axis)
        inter = self._persistent_rs_inter.get(inter_key)
        if inter is None:
            inter = ttnn.zeros(
                inter_shape,
                dtype=tensor.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self._persistent_rs_inter[inter_key] = inter
        out = self._persistent_rs_out.get(out_key)
        if out is None:
            out = ttnn.zeros(
                out_shape,
                dtype=tensor.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=memory_config,
            )
            self._persistent_rs_out[out_key] = out
            logger.debug(f"CCL persistent RS buffers allocated out={out_shape} inter={inter_shape}")
        return [inter, out]

    def oneshot_scratch(self, memory_config, dtype, tp):
        """The one-shot all-reduce's scratch for partial sums in the L1 width-sharded ``memory_config`` with payload
        ``dtype`` over ``tp`` chips: ``[1, 1, 32, shard_width * cores * tp]`` on every chip, shard
        ``(32, shard_width * tp)`` on the input's grid (the op gathers every chip's shard of a core into that core's
        scratch shard). A device allocation without host data (trace-safe), freed by the caller after the op; the
        layout is logged once per (shard width, grid, dtype)."""
        spec = memory_config.shard_spec
        shard_w = int(spec.shape[1])
        num_cores = int(spec.grid.num_cores())
        key = (shard_w, num_cores, str(dtype))
        if key not in self._oneshot_layouts:
            self._oneshot_layouts.add(key)
            logger.info(
                f"Gemma4 CCL one-shot scratch: {num_cores} cores x (32, {shard_w} x {tp}) {dtype} "
                f"({shard_w * tp * ttnn.TILE_SIZE * (1 if dtype == ttnn.bfloat8_b else 2) // 1024} KB per core)"
            )
        scratch_mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(spec.grid, (ttnn.TILE_SIZE, shard_w * tp), ttnn.ShardOrientation.ROW_MAJOR),
        )
        return ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, ttnn.TILE_SIZE, shard_w * num_cores * tp]),
            dtype,
            ttnn.TILE_LAYOUT,
            self.mesh_device,
            scratch_mem,
        )

    def oneshot_semaphore(self, name):
        """The next of slot ``name``'s two global semaphores (used in turn; see the one-shot section comment)."""
        sems = self._oneshot_sems.get(name)
        if sems is None:
            # the slots are created with the manager (ccl_oneshot_enabled at that time); a slot missing here means
            # the knob changed after the manager was built, and creating semaphores now could land inside a trace
            raise RuntimeError(
                f"Gemma4 CCL one-shot slot {name!r} has no semaphores: the CCLManager was created with "
                "GEMMA4_CCL_ONESHOT off; set the knob before the model is built"
            )
        i = self._oneshot_idx[name]
        self._oneshot_idx[name] = (i + 1) % len(sems)
        return sems[i]


_L1_SMALL_SEMAPHORES: dict = {}


def ccl_semaphores_in_l1_small(mesh_device) -> bool:
    """True when the mesh was opened with an L1_SMALL region (``l1_small_size`` > 0).

    The generic collectives (``ttnn.reduce_scatter`` / ``ttnn.all_gather``) allocate their global
    semaphores once per cached program, i.e. on the first call of a new shape. With no L1_SMALL region
    they land in worker L1, top-down -- the same addresses a previously captured decode trace used for
    its L1 intermediates -- and the next trace replay overwrites them (observed as a wrong first token
    or a hang on the prefill that follows the first traced decode). With a region they go to L1_SMALL,
    which traced ops never touch. ``ttnn.all_gather`` switches on its own; ``ttnn.reduce_scatter``
    needs the flag, and the all-reduce goes through ``sync_all_reduce`` with the manager's
    pre-created semaphores.
    """
    # the mesh's own id (``MeshDevice.id()``, a process-wide counter that is never reused), never ``id(mesh_device)``:
    # a mesh closed and reopened between two calls can reuse the Python address and would inherit a stale answer
    key = int(mesh_device.id())
    if key not in _L1_SMALL_SEMAPHORES:
        try:
            view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1_SMALL)
            _L1_SMALL_SEMAPHORES[key] = int(view.total_bytes_per_bank) > 0
        except Exception as e:  # noqa: BLE001 - absent view means no region
            logger.warning(f"Gemma4 CCL: could not query L1_SMALL ({e}); semaphores stay in L1")
            _L1_SMALL_SEMAPHORES[key] = False
        if not _L1_SMALL_SEMAPHORES[key]:
            logger.warning(
                "Gemma4 CCL: mesh has no L1_SMALL region; collective semaphores created after a trace capture "
                "can be overwritten by replays (open the mesh with l1_small_size, e.g. 65536)"
            )
    return _L1_SMALL_SEMAPHORES[key]


def ccl_allreduce(
    tensor, mesh_config, ccl_manager, memory_config=None, axis=None, group_size=None, slot=None, rows=None
):
    """All-reduce across TP devices.

    Sync ``ttnn.all_reduce`` by default. With ``GEMMA4_CCL_ASYNC=1``, uses
    reduce_scatter_minimal_async + all_gather_async (tt_transformers composite
    pattern) on ``ccl_manager.topology`` (Ring on P150x8).

    ``axis``/``group_size`` select another mesh axis; the defaults keep the tp-axis behavior.

    ``slot`` ("o" / "down"): the decode row-parallel call sites. With GEMMA4_CCL_ONESHOT=1 and a one-row-tile L1
    width-sharded ``tensor`` (the DRAM-sharded matmul's ``keep_sharded`` output) the one-shot all-reduce runs
    (``oneshot_all_reduce``); a sharded tensor the one-shot cannot take goes interleaved first and then through the
    composite. ``rows``: the logical row count to present the result in (the sharded tile carries 32).
    """
    if mesh_config is None or (axis is None and mesh_config.tp <= 1):
        return tensor
    if axis is None and _DEFER_TP_REDUCE.get():
        # sequence-parallel prefill: the caller reduce-scatters the partial sums over rows itself
        return tensor

    memory_config = memory_config or ttnn.DRAM_MEMORY_CONFIG
    tp_axis = mesh_config.tp_axis if axis is None else axis
    if slot is not None and axis is None:
        if ccl_oneshot_applicable(tensor, mesh_config, ccl_manager):
            return oneshot_all_reduce(tensor, slot, tp_axis, memory_config, ccl_manager, int(mesh_config.tp), rows=rows)
        if tensor.is_sharded():
            interleaved = ttnn.to_memory_config(tensor, memory_config)
            tensor.deallocate(True)
            tensor = _restore_rows(interleaved, rows)
    topology = ccl_manager.topology

    chunks = ccl_chunks_per_sync()
    workers = ccl_num_workers_per_link()
    nbuf = ccl_num_buffers_per_channel()
    if allreduce_gather_sum_enabled() and ccl_manager is not None and int(tensor.shape[-2]) <= 32:
        # one all-gather of the partial sums along a fresh dim, then the local sum (allreduce_gather_sum_enabled)
        gathered = ttnn.experimental.all_gather_async(
            tensor,
            persistent_output_buffer=None,
            dim=1,
            multi_device_global_semaphore=ccl_manager.get_ag_semaphore(axis=tp_axis)[:2],
            num_links=ccl_manager.num_links,
            cluster_axis=tp_axis,
            topology=topology,
            memory_config=memory_config,
            barrier_semaphore=ccl_manager.get_barrier_semaphore(axis=tp_axis),
            chunks_per_sync=chunks,
            num_workers_per_link=workers,
            num_buffers_per_channel=nbuf,
        )
        tensor.deallocate(True)
        summed = ttnn.sum(gathered, dim=1, keepdim=True, memory_config=memory_config)
        ttnn.deallocate(gathered)
        return summed
    if ccl_async_enabled():
        tp = mesh_config.tp if group_size is None else group_size
        rs_bufs = ccl_manager.get_persistent_rs_buffers(tensor, memory_config, tp, axis=tp_axis)
        scattered = ttnn.experimental.reduce_scatter_minimal_async(
            tensor,
            persistent_output_buffers=rs_bufs,
            dim=3,
            multi_device_global_semaphore=ccl_manager.get_rs_semaphore(axis=tp_axis),
            barrier_semaphore=ccl_manager.get_barrier_semaphore(axis=tp_axis),
            num_links=ccl_manager.num_links,
            cluster_axis=tp_axis,
            memory_config=memory_config,
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=topology,
            chunks_per_sync=chunks,
            num_workers_per_link=workers,
            num_buffers_per_channel=nbuf,
        )
        tensor.deallocate(True)
        # Do not pass a persistent AG buffer: the gather result is returned as a
        # normal activation and force-deallocated by callers. Persistent RS out
        # aliases ``scattered`` when rs_bufs is set — do not free it either.
        gathered = ttnn.experimental.all_gather_async(
            scattered,
            persistent_output_buffer=None,
            dim=3,
            # Default async all-gather requires only the first two pre-created semaphores.
            multi_device_global_semaphore=ccl_manager.get_ag_semaphore(axis=tp_axis)[:2],
            num_links=ccl_manager.num_links,
            cluster_axis=tp_axis,
            topology=topology,
            memory_config=memory_config,
            barrier_semaphore=ccl_manager.get_barrier_semaphore(axis=tp_axis),
            chunks_per_sync=chunks,
            num_workers_per_link=workers,
            num_buffers_per_channel=nbuf,
        )
        if rs_bufs is None:
            scattered.deallocate(True)
        return gathered

    result = sync_all_reduce(tensor, tp_axis, memory_config, ccl_manager)
    tensor.deallocate(True)
    return result


def sync_all_reduce(tensor, axis, memory_config, ccl_manager=None):
    """Synchronous all-reduce over ``axis``; with a manager, through ``all_reduce_async`` on the
    manager's pre-created semaphores (the same op ``ttnn.all_reduce`` runs, which would otherwise
    create its semaphores per cached program -- in worker L1 when the program is first compiled
    after a decode trace capture, where the replay overwrites them; see
    ``ccl_semaphores_in_l1_small``). Without a manager: ``ttnn.all_reduce``."""
    memory_config = memory_config or ttnn.DRAM_MEMORY_CONFIG
    if ccl_manager is None:
        return ttnn.all_reduce(tensor, cluster_axis=axis, memory_config=memory_config)
    barriers, rs_sems, ag_sems = ccl_manager.get_all_reduce_semaphores(axis)
    return ttnn.experimental.all_reduce_async(
        tensor,
        cluster_axis=axis,
        mesh_device=ccl_manager.mesh_device,
        barrier_semaphores=barriers,
        rs_global_semaphores=rs_sems,
        ag_global_semaphores=ag_sems,
        math_op=ttnn.ReduceType.Sum,
        memory_config=memory_config,
        topology=ccl_manager.topology,
    )


def ccl_allreduce_fractured(tensor, mesh_config, ccl_manager, memory_config=None):
    """All-reduce along both mesh axes to complete 2D weight-fractured partial sums.

    Uses sync ``ttnn.all_reduce`` per axis, or the async RS+AG composite when ``GEMMA4_CCL_ASYNC=1``.
    """
    if mesh_config is None or not getattr(mesh_config, "weight_fracture", False):
        raise ValueError("ccl_allreduce_fractured needs mesh_config.weight_fracture; use ccl_allreduce")

    memory_config = memory_config or ttnn.DRAM_MEMORY_CONFIG
    out = tensor
    for axis in (0, 1):
        if mesh_config.mesh_shape[axis] <= 1:
            continue
        if ccl_manager is not None and ccl_async_enabled():
            out = ccl_allreduce(
                out,
                mesh_config,
                ccl_manager,
                memory_config,
                axis=axis,
                group_size=mesh_config.mesh_shape[axis],
            )
        else:
            reduced = sync_all_reduce(out, axis, memory_config, ccl_manager)
            out.deallocate(True)
            out = reduced
    return out


def lane_ccl_sync_enabled(mesh_config) -> bool:
    """GEMMA4_LANE_CCL_SYNC=1: synchronize the mesh after every lane-axis collective of an eager single-user prefill.

    Workaround for the (2,4) block on a FABRIC_1D line (tt-triage capture 2026-10-24 06:45, devq 0596): a Linear
    lane-axis reduce-scatter completes on the end columns first; the middle columns then enter the next TP
    all-reduce late and a remote write of the previous collective lands on an L1 address the next op already reuses
    (a program semaphore read 0xA3B6F3D3 on the stuck chip), so both chips of the column wait forever while the end
    columns run 21 ops ahead. A device sync after each lane collective keeps the columns aligned, as the per-phase
    synchronization of the (since removed) prefill phase-timing harness did (passed on all 4 lanes). Eager
    single-user prefill only (the
    owner lane is set): never inside a trace capture, where a synchronize is a TT_FATAL. A ring (the Galaxy) completes
    symmetrically and does not need it.
    """
    return (
        os.environ.get("GEMMA4_LANE_CCL_SYNC", "0") == "1"
        and getattr(mesh_config, "_g4_single_lane_prefill_owner", None) is not None
    )


def ccl_lane_gather_rows(tensor, mesh_config, memory_config=None, consume=True, ccl_manager=None):
    """All-gather the row dim across lanes so each chip's fractured weight chunk sees every lane's rows.

    ``consume`` frees ``tensor`` after the gather; pass False for a tensor the caller still owns (the decoder layer
    frees ``expert_input`` itself after the MoE call, and a pad within the tile padding is only a view of it).
    With ``GEMMA4_LANE_CCL_ASYNC=1`` and a ``ccl_manager`` the gather is ``all_gather_async`` on the manager's
    semaphores with a barrier (``lane_ccl_async_enabled``).
    """
    if mesh_config is None or not getattr(mesh_config, "lane_sharded", False):
        return tensor
    memory_config = memory_config or ttnn.DRAM_MEMORY_CONFIG
    if ccl_manager is not None and lane_ccl_async_enabled():
        gathered = ttnn.experimental.all_gather_async(
            tensor,
            persistent_output_buffer=None,
            dim=2,
            multi_device_global_semaphore=ccl_manager.get_ag_semaphore(axis=mesh_config.sp_axis)[:2],
            num_links=ccl_manager.num_links,
            cluster_axis=mesh_config.sp_axis,
            topology=ccl_manager.topology,
            memory_config=memory_config,
            barrier_semaphore=ccl_manager.get_barrier_semaphore(axis=mesh_config.sp_axis),
            chunks_per_sync=ccl_chunks_per_sync(),
            num_workers_per_link=ccl_num_workers_per_link(),
            num_buffers_per_channel=ccl_num_buffers_per_channel(),
        )
    else:
        gathered = ttnn.all_gather(tensor, dim=2, cluster_axis=mesh_config.sp_axis, memory_config=memory_config)
    if consume:
        tensor.deallocate(True)
    if lane_ccl_sync_enabled(mesh_config):
        ttnn.synchronize_device(gathered.device())
    return gathered


def ccl_lane_scatter_rows(tensor, mesh_config, memory_config=None, ccl_manager=None):
    """Complete a lane-sharded fractured matmul: reduce_scatter rows to their lanes, then all-reduce over tp.

    reduce_scatter sums the cross-lane K-chunks; the tp-axis all-reduce sums the within-column chunks.
    """
    if mesh_config is None or not getattr(mesh_config, "lane_sharded", False):
        return tensor
    memory_config = memory_config or ttnn.DRAM_MEMORY_CONFIG
    if ccl_manager is not None and lane_ccl_async_enabled():
        # the manager's semaphores + barrier: every lane enters aligned, the pool keeps consecutive collectives apart
        scattered = ttnn.experimental.reduce_scatter_minimal_async(
            tensor,
            persistent_output_buffers=None,
            dim=2,
            multi_device_global_semaphore=ccl_manager.get_rs_semaphore(axis=mesh_config.sp_axis),
            barrier_semaphore=ccl_manager.get_barrier_semaphore(axis=mesh_config.sp_axis),
            num_links=ccl_manager.num_links,
            cluster_axis=mesh_config.sp_axis,
            memory_config=memory_config,
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ccl_manager.topology,
            chunks_per_sync=ccl_chunks_per_sync(),
            num_workers_per_link=ccl_num_workers_per_link(),
            num_buffers_per_channel=ccl_num_buffers_per_channel(),
        )
    else:
        scattered = ttnn.reduce_scatter(
            tensor,
            dim=2,
            cluster_axis=mesh_config.sp_axis,
            memory_config=memory_config,
            use_l1_small_for_semaphores=ccl_semaphores_in_l1_small(tensor.device()),
        )
    tensor.deallocate(True)
    if lane_ccl_sync_enabled(mesh_config):
        ttnn.synchronize_device(scattered.device())  # the columns enter the TP all-reduce aligned
    reduced = sync_all_reduce(scattered, mesh_config.tp_axis, memory_config, ccl_manager)
    scattered.deallocate(True)
    return reduced


# GEMMA4_PREFILL_SEQ_PARALLEL=1 (opt-in): a dense single-user prefill keeps the residual stream row-sharded over the
# TP axis (each chip holds S / tp rows). The row-parallel o_proj / down_proj partial sums are reduce-scattered over
# rows instead of all-reduced, the norms and residual adds between them run on the chip's own rows only, and the
# next normed input is all-gathered over rows: the same reduce-scatter + all-gather bytes as the all-reduce composite,
# 1 / tp of the norm and add work (every chip normed all S replicated rows before). Model side: model.py
# (_prefill_seq_parallel_applies, mesh_partition before the layers, the row all-gather after) and layer.py.
PREFILL_SEQ_PARALLEL_ENV = "GEMMA4_PREFILL_SEQ_PARALLEL"
_DEFER_TP_REDUCE = contextvars.ContextVar("gemma4_defer_tp_reduce", default=False)


def prefill_seq_parallel_enabled() -> bool:
    """``GEMMA4_PREFILL_SEQ_PARALLEL``: the row-sharded residual for dense single-user prefill (the env when set, else
    the build profile: on for the 31B (1,8) line, mesh_orientation.GALAXY_LINE_PAIR_ON; off elsewhere)."""
    from models.demos.gemma4.tt.mesh_orientation import decode_flag

    return decode_flag(PREFILL_SEQ_PARALLEL_ENV) == "1"


@contextlib.contextmanager
def deferred_tp_reduce():
    """Inside this block ``ccl_allreduce`` on the TP axis returns its input unreduced (the caller owns the reduction:
    ``ccl_reduce_scatter_rows``)."""
    token = _DEFER_TP_REDUCE.set(True)
    try:
        yield
    finally:
        _DEFER_TP_REDUCE.reset(token)


def ccl_reduce_scatter_rows(tensor, mesh_config, ccl_manager, memory_config=None):
    """Sum the TP partials and keep this chip's block of rows: ``[1, 1, S, W]`` partial -> ``[1, 1, S / tp, W]``.

    Default: reduce-scatter over COLUMNS (dim 3, the all-reduce composite's own reduction, so every sum is bitwise the
    replicated path's) and an all-to-all that trades column blocks for row blocks (each chip sends (tp - 1) / tp of
    an S x W / tp shard, 1 / tp of the all-gather's bytes). ``GEMMA4_PREFILL_SEQ_PARALLEL_RS=rows``: one
    reduce-scatter over rows instead, measured 2026-11-11 on a Galaxy row at 1.05x the all-reduce's relative error
    (4.59e-3 vs 4.36e-3 against an fp64 sum of bf16 partials) and -0.014 top-1 on the 31B / 12B 2K prose gates."""
    memory_config = memory_config or ttnn.DRAM_MEMORY_CONFIG
    l1_small = ccl_semaphores_in_l1_small(tensor.device())
    if os.environ.get("GEMMA4_PREFILL_SEQ_PARALLEL_RS", "a2a") == "rows":
        scattered = ttnn.reduce_scatter(
            tensor,
            dim=2,
            cluster_axis=mesh_config.tp_axis,
            memory_config=memory_config,
            use_l1_small_for_semaphores=l1_small,
        )
        tensor.deallocate(True)
        return scattered
    cols = ttnn.reduce_scatter(
        tensor,
        dim=3,
        cluster_axis=mesh_config.tp_axis,
        memory_config=memory_config,
        use_l1_small_for_semaphores=l1_small,
    )
    tensor.deallocate(True)
    rows = ttnn.experimental.all_to_all_async_generic(
        cols,
        in_dim=3,  # the op grows in_dim by tp and splits out_dim by tp: [S, W / tp] -> [S / tp, W]
        out_dim=2,
        persistent_output_buffer=None,
        num_links=ccl_manager.num_links,
        memory_config=memory_config,
        topology=ccl_manager.topology,
        cluster_axis=mesh_config.tp_axis,
    )
    cols.deallocate(True)
    return rows


def ccl_allgather(tensor, mesh_config, ccl_manager, dim=3, memory_config=None):
    """All-gather across TP devices."""
    if mesh_config is None or mesh_config.tp <= 1:
        return tensor

    memory_config = memory_config or ttnn.DRAM_MEMORY_CONFIG
    tp_axis = mesh_config.tp_axis
    topology = ccl_manager.topology
    chunks = ccl_chunks_per_sync()
    workers = ccl_num_workers_per_link()
    nbuf = ccl_num_buffers_per_channel()

    if ccl_async_enabled():
        # Fresh AG output each call (caller-owned); see ccl_allreduce note.
        gathered = ttnn.experimental.all_gather_async(
            tensor,
            persistent_output_buffer=None,
            dim=dim,
            # Default async all-gather requires only the first two pre-created semaphores.
            multi_device_global_semaphore=ccl_manager.get_ag_semaphore(axis=tp_axis)[:2],
            num_links=ccl_manager.num_links,
            cluster_axis=tp_axis,
            topology=topology,
            memory_config=memory_config,
            barrier_semaphore=ccl_manager.get_barrier_semaphore(axis=tp_axis),
            chunks_per_sync=chunks,
            num_workers_per_link=workers,
            num_buffers_per_channel=nbuf,
        )
        tensor.deallocate(True)
        return gathered

    # Sync all_gather: do not pass deprecated num_links/topology/chunks_* —
    # Fabric config supplies those; passing them only emits Sep-2026 warnings.
    gathered = ttnn.all_gather(
        tensor,
        dim=dim,
        cluster_axis=tp_axis,
        memory_config=memory_config,
    )
    tensor.deallocate(True)
    return gathered
