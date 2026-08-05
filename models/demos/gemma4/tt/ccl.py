# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import os

from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole


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


def ccl_sync_split_enabled() -> bool:
    """Run the TP all-reduce as sync ``reduce_scatter`` + sync ``all_gather``
    instead of the fused ``ttnn.all_reduce``. Default ON; ``GEMMA4_CCL_SPLIT=0``
    opts back out.

    ``ttnn.all_reduce`` *is* those two ops -- measured identical to within noise
    (fused 95.5 us vs split 96.1 us) and ``torch.equal`` bit-identical on
    per-device-distinct data. But the fused op exposes only
    {cluster_axis, memory_config, num_links, topology, subdevice_id}, while the
    sync halves also expose ``chunks_per_sync`` / ``num_workers_per_link`` /
    ``num_buffers_per_channel``. Splitting therefore costs nothing and unlocks
    the knobs -- see ``ccl_sync_rs_workers``.

    This is NOT the async path (``GEMMA4_CCL_ASYNC``), which uses
    ``reduce_scatter_minimal_async`` and loses in every arm.
    """
    return os.environ.get("GEMMA4_CCL_SPLIT", "1").lower() not in ("0", "false", "no")


def ccl_sync_rs_workers() -> int:
    """``num_workers_per_link`` for the split all-reduce's reduce-scatter.

    Trace-replay sweep of the Gemma4-31B decode all-reduce ([1,1,32,5376] bf16,
    Ring, num_links=1, 1x8 WH LoudBox), 3 repeats, min-of-rounds, every arm
    checked ``torch.equal`` against the fused ``ttnn.all_reduce`` result:

        fused ttnn.all_reduce (shipping)           95.5 us  -> 11.46 ms/step
        split, both halves default                 96.1 us  -> 11.54 ms/step
        split, RS w=1 c=1                        **88.9 us**->**10.67 ms/step**
        split, RS w=2 c=1                          94.0 us  -> 11.28 ms/step
        split, RS w=1 c=2                          96.1 us  -> 11.54 ms/step
        split, RS w=1 c=4                          99.0 us  -> 11.88 ms/step
        split, RS w=4 c=1                         141.0 us  -> 16.9  ms/step

    So ``w=1, c=1`` is worth -6.6 us/all-reduce = **-0.79 ms/decode step**, and
    it is bit-exact -- the reduction order is unchanged, only the worker/sync
    granularity is. ``num_buffers_per_channel`` is noise here (b=2/4/8 all
    88.8-89.4 us); 4 is taken as the middle.

    Note ``w=4`` is a 1.5x cliff, not a plateau: with a single link, extra
    workers contend. Do not raise this without re-sweeping, and do not confuse
    it with the async path's ``GEMMA4_CCL_NUM_WORKERS`` default of 2.

    The GATHER half was swept over the same knobs (w x c x b, 12 arms) and is
    completely insensitive -- 95.5-95.9 us throughout. It runs on ONE worker core
    (vs the reduce-scatter's 6) and 44.7 us for a 688 KB gather is the num_links=1
    fabric floor, not core starvation. Leave it on defaults.
    """
    return max(1, int(os.environ.get("GEMMA4_CCL_SYNC_RS_WORKERS", "1")))


def ccl_sync_rs_chunks() -> int:
    """``chunks_per_sync`` for the split all-reduce's reduce-scatter. See
    ``ccl_sync_rs_workers`` -- 1 measured 88.9 us, 2 measured 96.1, 4 measured 99.0."""
    return max(1, int(os.environ.get("GEMMA4_CCL_SYNC_RS_CHUNKS", "1")))


def ccl_sync_rs_buffers() -> int:
    """``num_buffers_per_channel`` for the split all-reduce's reduce-scatter.
    Noise across 2/4/8; see ``ccl_sync_rs_workers``."""
    return max(1, int(os.environ.get("GEMMA4_CCL_SYNC_RS_BUFFERS", "4")))


def default_ccl_topology(mesh_device=None, is_moe: bool = False):
    """Default CCL topology for Gemma4 TP collectives.

    Override with ``GEMMA4_CCL_TOPOLOGY=ring|linear``.

    Policy (when env unset):
      * **Ring** on **Blackhole** meshes with **≥8 devices** (P150x8 TTFT
        sweep: Ring+sync ~28.8s vs Linear+sync ~31.0s @ 31B/128k).
      * **Ring** on **Wormhole** meshes with **≥8 devices** for **dense**
        models. Trace-replay sweep of the 31B decode all-reduce
        ([1,1,32,5376] bf16, 2/layer x 60 layers) on a 1x8 WH LoudBox:

            sync all_reduce  Linear  114.4 us  -> 13.73 ms/step  (was default)
            sync all_reduce  Ring     96.2 us  -> 11.55 ms/step  <-- now default
            async RS+AG      Linear  131-142us -> 15.7-17.0 ms/step
            async RS+AG      Ring    101-127us -> 12.2-15.2 ms/step

        i.e. Ring is worth ~2.2 ms/step (~4% of a 50.6 ms decode step) and
        sync beats async in every arm. Opening the mesh with
        ``FABRIC_1D_RING`` instead of ``FABRIC_1D`` buys only a further
        93.1 vs 96.2 us, so the topology is taken under plain ``FABRIC_1D``
        and no harness device_params change is needed. ``num_links=2`` is NOT
        usable here — it raises "Event Order Issue: expected to read back
        completion signal for event 27 but got 14" (see default_num_links).
      * **Linear** for **MoE** models on WH: Ring drops 26B-A4B
        ``test_full_model`` PCC below the TEMP 0.76 gate (~0.7505 vs
        ~0.77/0.94 with Linear / main).
      * **Linear** everywhere else. Ring on 4-device BH drops 12B full-model
        PCC (~0.97 → ~0.90).
    """
    override = os.environ.get("GEMMA4_CCL_TOPOLOGY", "").strip().lower()
    if override in ("ring", "r"):
        return ttnn.Topology.Ring
    if override in ("linear", "line", "l"):
        return ttnn.Topology.Linear

    n = mesh_device.get_num_devices() if mesh_device is not None else 0
    if n:
        if n >= 8 and (is_blackhole() or not is_moe):
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

    def __init__(self, mesh_device, num_links=None, topology=None, is_moe: bool = False):
        if num_links is None:
            num_links = default_num_links()
        if topology is None:
            topology = default_ccl_topology(mesh_device, is_moe=is_moe)
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

        self._rs_semaphores = []
        self._ag_semaphores = []
        self._barrier_semaphores = []
        for _ in range(2):
            self._rs_semaphores.append([ttnn.create_global_semaphore(mesh_device, core_range_set, 0) for _ in range(3)])
            self._ag_semaphores.append([ttnn.create_global_semaphore(mesh_device, core_range_set, 0) for _ in range(2)])
            self._barrier_semaphores.append(ttnn.create_global_semaphore(mesh_device, core_range_set, 0))
        ttnn.synchronize_device(mesh_device)

        self._rs_idx = 0
        self._ag_idx = 0
        self._barrier_idx = 0
        # shape_key -> ttnn.Tensor (DRAM interleaved zeros)
        self._persistent_ag: dict = {}
        self._persistent_rs_out: dict = {}
        self._persistent_rs_inter: dict = {}

    def get_rs_semaphore(self):
        """Returns list of 3 semaphores for reduce_scatter (cycles double-buffer)."""
        sems = self._rs_semaphores[self._rs_idx]
        self._rs_idx = (self._rs_idx + 1) % 2
        return sems

    def get_ag_semaphore(self):
        """Returns list of 2 semaphores for all_gather (cycles double-buffer)."""
        sems = self._ag_semaphores[self._ag_idx]
        self._ag_idx = (self._ag_idx + 1) % 2
        return sems

    def get_barrier_semaphore(self):
        """Returns single barrier semaphore (cycles double-buffer)."""
        sem = self._barrier_semaphores[self._barrier_idx]
        self._barrier_idx = (self._barrier_idx + 1) % 2
        return sem

    def _shape_key(self, shape, dtype, memory_config):
        return (tuple(int(x) for x in shape), str(dtype), str(memory_config))

    def _alloc_like(self, ref_tensor, memory_config):
        return ttnn.zeros_like(ref_tensor, device=self.mesh_device, memory_config=memory_config)

    def get_persistent_ag_buffer(self, scattered, memory_config, tp):
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
        key = self._shape_key(out_shape, scattered.dtype, memory_config)
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

    def get_persistent_rs_buffers(self, tensor, memory_config, tp):
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
        inter_key = self._shape_key(inter_shape, tensor.dtype, ttnn.DRAM_MEMORY_CONFIG)
        out_key = self._shape_key(out_shape, tensor.dtype, memory_config)
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


def ccl_l1_gather_enabled() -> bool:
    """Let the TP all-reduce's gather write width-sharded L1 instead of DRAM.
    Default ON; ``GEMMA4_CCL_L1_GATHER=0`` opts back out."""
    return os.environ.get("GEMMA4_CCL_L1_GATHER", "1").lower() not in ("0", "false", "no")


def _decode_l1_gather_memcfg(tensor, ccl_manager):
    """Width-sharded L1 memory config for the all-gather output, or None to keep DRAM.

    Every ``ccl_allreduce`` call site in the decode path feeds its result straight
    into an ``RMSNorm`` (layer.py: post_attention_layernorm, post_feedforward_
    layernorm{,_1,_2}), and that norm's first act is to width-shard its input. So
    having the gather write that layout directly removes an
    InterleavedToSharded per all-reduce -- measured 47.2 -> 43.5 us
    (-0.45 ms/decode step on 31B), bit-exact (ops_list/tools/sweeps/l1_stream.py).

    The layout comes from ``rms_norm.decode_width_shard_spec``, the same function
    the norm uses, so the two provably agree; ``RMSNorm.forward`` compares
    ``memory_config()`` and only takes the input in place on an exact match, so a
    mismatch degrades to today's behaviour rather than corrupting anything.

    Guarded to the decode shape (a single tile of rows). Prefill activations are
    far too large to sit width-sharded in L1, and their norms use the plain path
    anyway.
    """
    if not ccl_l1_gather_enabled():
        return None
    try:
        shape = tensor.shape
        if len(shape) != 4 or not (1 <= shape[-2] <= ttnn.TILE_SIZE):
            return None
        from models.demos.gemma4.tt.rms_norm import decode_width_shard_memcfg

        return decode_width_shard_memcfg(ccl_manager.mesh_device, shape[-1])
    except Exception as e:  # never let a layout optimization break the model
        logger.debug(f"ccl L1 gather memcfg unavailable ({e}); keeping DRAM")
        return None


def ccl_allreduce(tensor, mesh_config, ccl_manager, memory_config=None):
    """All-reduce across TP devices.

    By default, sync ``ttnn.reduce_scatter`` + sync ``ttnn.all_gather`` with a
    swept reduce-scatter worker config (``GEMMA4_CCL_SPLIT=0`` falls back to the
    fused ``ttnn.all_reduce``, which is bit-identical but 6.6 us/call slower --
    see ``ccl_sync_split_enabled`` / ``ccl_sync_rs_workers``).

    With ``GEMMA4_CCL_ASYNC=1``, uses reduce_scatter_minimal_async +
    all_gather_async (tt_transformers composite pattern) on
    ``ccl_manager.topology`` (Ring on P150x8). That path loses in every measured
    arm; it is kept for the record, not as a default.
    """
    if mesh_config is None or mesh_config.tp <= 1:
        return tensor

    # None means the caller expressed no preference, so we are free to pick the
    # layout its consumer wants. Capture it before the DRAM default is applied.
    caller_memory_config = memory_config
    memory_config = memory_config or ttnn.DRAM_MEMORY_CONFIG
    tp_axis = mesh_config.tp_axis
    topology = ccl_manager.topology
    # Computed while ``tensor`` is still alive -- the split path deallocates it
    # before the all-gather runs.
    gather_memory_config = memory_config
    if caller_memory_config is None:
        gather_memory_config = _decode_l1_gather_memcfg(tensor, ccl_manager) or memory_config

    chunks = ccl_chunks_per_sync()
    workers = ccl_num_workers_per_link()
    nbuf = ccl_num_buffers_per_channel()
    if ccl_async_enabled():
        tp = mesh_config.tp
        rs_bufs = ccl_manager.get_persistent_rs_buffers(tensor, memory_config, tp)
        scattered = ttnn.experimental.reduce_scatter_minimal_async(
            tensor,
            persistent_output_buffers=rs_bufs,
            dim=3,
            multi_device_global_semaphore=ccl_manager.get_rs_semaphore(),
            barrier_semaphore=ccl_manager.get_barrier_semaphore(),
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
            multi_device_global_semaphore=ccl_manager.get_ag_semaphore(),
            num_links=ccl_manager.num_links,
            cluster_axis=tp_axis,
            topology=topology,
            memory_config=memory_config,
            barrier_semaphore=ccl_manager.get_barrier_semaphore(),
            chunks_per_sync=chunks,
            num_workers_per_link=workers,
            num_buffers_per_channel=nbuf,
        )
        if rs_bufs is None:
            scattered.deallocate(True)
        return gathered

    if ccl_sync_split_enabled():
        scattered = ttnn.reduce_scatter(
            tensor,
            dim=3,
            cluster_axis=tp_axis,
            num_links=ccl_manager.num_links,
            topology=topology,
            memory_config=memory_config,
            num_workers_per_link=ccl_sync_rs_workers(),
            chunks_per_sync=ccl_sync_rs_chunks(),
            num_buffers_per_channel=ccl_sync_rs_buffers(),
        )
        tensor.deallocate(True)
        result = ttnn.all_gather(
            scattered,
            dim=3,
            cluster_axis=tp_axis,
            num_links=ccl_manager.num_links,
            topology=topology,
            memory_config=gather_memory_config,
        )
        scattered.deallocate(True)
        return result

    result = ttnn.all_reduce(
        tensor,
        cluster_axis=tp_axis,
        num_links=ccl_manager.num_links,
        topology=topology,
        memory_config=memory_config,
    )
    tensor.deallocate(True)
    return result


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
            multi_device_global_semaphore=ccl_manager.get_ag_semaphore(),
            num_links=ccl_manager.num_links,
            cluster_axis=tp_axis,
            topology=topology,
            memory_config=memory_config,
            barrier_semaphore=ccl_manager.get_barrier_semaphore(),
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
