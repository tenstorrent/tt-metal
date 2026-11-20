# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

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


def ccl_allreduce(tensor, mesh_config, ccl_manager, memory_config=None, axis=None, group_size=None):
    """All-reduce across TP devices.

    Sync ``ttnn.all_reduce`` by default. With ``GEMMA4_CCL_ASYNC=1``, uses
    reduce_scatter_minimal_async + all_gather_async (tt_transformers composite
    pattern) on ``ccl_manager.topology`` (Ring on P150x8).

    ``axis``/``group_size`` select another mesh axis; the defaults keep the tp-axis behavior.
    """
    if mesh_config is None or (axis is None and mesh_config.tp <= 1):
        return tensor

    memory_config = memory_config or ttnn.DRAM_MEMORY_CONFIG
    tp_axis = mesh_config.tp_axis if axis is None else axis
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
