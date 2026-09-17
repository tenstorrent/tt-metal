# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import os

from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.gemma4.tt.dram_sharded import wh_t3k_decode_enabled

# CCL all_gather allocates barrier semaphores in L1_SMALL when this is > 0
# (see all_gather_multicast_factory.cpp). Demo and unit meshes must open with
# this so those semaphores do not fragment the main L1 pool.
_DEFAULT_L1_SMALL_SIZE = 24576
# Wormhole: three 2048 B CCL tiles per packet (validate_packet_size ideal).
# Do not use 3840 B for 12B — vocab all-gather hung after prefill-trace capture.
_WH_CCL_PACKET_BYTES = 6144


def default_l1_small_size() -> int:
    """L1_SMALL region size so CCL all_gather semaphores skip the main L1 pool.

    Override with ``GEMMA4_L1_SMALL_SIZE``.
    """
    return int(os.environ.get("GEMMA4_L1_SMALL_SIZE", _DEFAULT_L1_SMALL_SIZE))


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


def default_ccl_packet_bytes():
    """Wormhole packet-width override; Blackhole keeps the Fabric default.

    6144 B is three 2048 B tiles (the SharedMLP / all-gather page size).
    Blackhole stays on Fabric's 4352 B default: matching page width was
    slower end-to-end on P150x8. Override with ``GEMMA4_CCL_PACKET_BYTES``.

    Applies to every Wormhole model, not just 31B. The hazard this avoids is a
    packet NARROWER than the page: overriding 12B to 3840 B once hung the vocab
    all-gather after the first prefill-trace capture (4096 B pages). 6144 B is
    wider than both the 4096 B and 5376 B pages seen here, so it is safe where
    3840 B was not.
    """
    if is_blackhole():
        return None
    return _WH_CCL_PACKET_BYTES


def fabric_router_config_from_env():
    """Build the Gemma4 Fabric router override selected before mesh open."""
    pkt_env = os.environ.get("GEMMA4_CCL_PACKET_BYTES")
    if pkt_env is None:
        pkt_bytes = default_ccl_packet_bytes()
    elif pkt_env.strip().lower() in ("0", "none", "default", ""):
        pkt_bytes = None
    else:
        pkt_bytes = max(4352, int(pkt_env))
    if pkt_bytes is None:
        return None
    if not is_blackhole():
        pkt_bytes = min(pkt_bytes, 7616)
    router = ttnn.FabricRouterConfig()
    router.max_packet_payload_size_bytes = pkt_bytes
    return router


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


_CCL_ASYNC_MIN_HEIGHT = 2048


def _physical_tile_padded_height(tensor) -> int:
    """Flatten B*S-style leading dimensions and tile-pad the collective height."""
    height = 1
    for index in range(len(tensor.shape) - 1):
        height *= int(tensor.shape[index])
    return ((height + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE


_PREFILL_RS_TALL_HEIGHT = 2048

# (num_workers_per_link, chunks_per_sync, num_buffers_per_channel) for the sync
# reduce-scatter half. See ``ccl_sync_rs_tuning``.
_RS_TUNING_DEFAULT = (1, 1, 4)
_RS_TUNING_PREFILL_TALL = (2, 2, 4)
_RS_TUNING_WH_T3K_DECODE = (4, 2, 8)


def ccl_sync_rs_tuning(padded_height: int | None = None, mesh_device=None, is_moe: bool = True):
    """``(num_workers_per_link, chunks_per_sync, num_buffers_per_channel)`` for
    the sync reduce-scatter half.

    Tall prefill (``h >= 2048``) is bandwidth-bound and takes the wider arm.
    Everything else is latency-bound and ships the default, except a one-tile
    (decode-height) all-reduce for a dense model on a Wormhole T3K, which was
    swept on a real T3K: the 12B hidden=3840 TP=8 reduce_scatter + all_gather
    pair runs 66.6 us at ``(4, 2, 8)`` against 67.9 us at the default, and every
    ``w=4`` arm beat every ``w<4`` arm.

    The T3K arm is gated the same way as the rest of this branch's T3K work
    (``wh_t3k_decode_enabled``: Wormhole, 8 devices, unharvested 8x8 grid) plus
    dense-only, so N150/N300, Blackhole and 26B-A4B keep the default exactly.
    ``test_full_model_decode`` PCC is bit-identical across the two arms.
    """
    if padded_height is not None and int(padded_height) >= _PREFILL_RS_TALL_HEIGHT:
        return _RS_TUNING_PREFILL_TALL
    if padded_height == ttnn.TILE_SIZE and not is_moe and wh_t3k_decode_enabled(mesh_device):
        return _RS_TUNING_WH_T3K_DECODE
    return _RS_TUNING_DEFAULT


# 31B JSON Linear pin is a 128k-decode numerics fix (Ring loops). Apply it only
# at this length and above; shorter dense decode stays on Ring. 64k Ring was
# coherent on a real WH T3K (text_demo_v2 long-context-64k).
LINEAR_PIN_MIN_SEQ_LEN = 128 * 1024


def effective_pinned_ccl_topology(pinned, *, is_moe: bool, max_seq_len):
    """Drop a dense-model Linear pin on Blackhole, or below 128k on Wormhole.

    ``pinned`` is a ``ttnn.Topology`` or ``None``. MoE keeps Linear at every
    length and on every arch. ``None`` lets ``CCLManager`` take the arch default.

    The dense Linear pin exists for one measured Wormhole problem: Ring's
    reduction order loops 31B's 128k decode on a WH T3K. Blackhole was never
    part of that evidence -- it has 32 GB/ASIC and main ships Ring there for
    n>=8 (its own sweep: Ring ~28.8s vs Linear ~31.0s TTFT at 31B/128k). Left
    ungated, the pin silently moved the P150x8 31B vLLM job (max_model_len
    262144) off Ring and onto Linear, costing ~7% for a fault it does not have.
    Do not extend this pin to Blackhole without a Blackhole measurement.
    """
    if pinned != ttnn.Topology.Linear or is_moe:
        return pinned
    if is_blackhole():
        return None
    if max_seq_len is None or int(max_seq_len) >= LINEAR_PIN_MIN_SEQ_LEN:
        return pinned
    return None


def default_ccl_topology(mesh_device=None, is_moe: bool = True):
    """Default CCL topology for Gemma4 TP collectives.

    Override with ``GEMMA4_CCL_TOPOLOGY=ring|linear``.

    Policy (when env unset):
      * **Ring** on **Blackhole** meshes with **≥8 devices** (P150x8 TTFT
        sweep: Ring+sync ~28.8s vs Linear+sync ~31.0s @ 31B/128k).
      * **Ring** on **Wormhole** meshes with **≥8 devices** for **dense**
        models. Use sync Ring under ``FABRIC_1D``; ``num_links=2`` raises an
        Event Order Issue (see default_num_links).
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


def ccl_async_enabled(padded_height: int | None = None) -> bool:
    """Auto-enable async RS+AG only for bandwidth-bound tall prefill."""
    override = os.environ.get("GEMMA4_CCL_ASYNC")
    if override is not None:
        return override.lower() in ("1", "true", "yes")
    return padded_height is not None and int(padded_height) >= _CCL_ASYNC_MIN_HEIGHT


class CCLManager:
    """CCL manager for Gemma4 tensor parallelism.

    Stores mesh_device, num_links, and topology for CCL operations.
    Semaphores support the async RS+AG path (``GEMMA4_CCL_ASYNC=1``).
    Persistent DRAM buffers (``GEMMA4_CCL_PERSISTENT_BUF``) are keyed by shape
    so repeated collectives of the same activation shape skip realloc+barrier.
    """

    def __init__(self, mesh_device, num_links=None, topology=None, is_moe: bool = True):
        if num_links is None:
            num_links = default_num_links()
        if topology is None:
            topology = default_ccl_topology(mesh_device, is_moe=is_moe)
        self.mesh_device = mesh_device
        self.num_links = num_links
        self.topology = topology
        self.is_moe = bool(is_moe)
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


def _short_seq_l1_gather_memcfg(tensor, ccl_manager):
    """Width-sharded L1 gather layout for decode and short-prefill all-reduce."""
    try:
        shape = tensor.shape
        if len(shape) != 4:
            return None
        from models.demos.gemma4.tt.rms_norm import activation_physical_height, width_shard_input_memcfg

        padded_height = activation_physical_height(shape)
        # Decode always gathers into the one-tile residual island. Gathering at
        # height<=1024 hung T3K warmup, so restrict this to the single tile.
        if padded_height != ttnn.TILE_SIZE:
            return None
        return width_shard_input_memcfg(ccl_manager.mesh_device, shape[-1], padded_height)
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        logger.debug(f"Gemma4 L1 gather unavailable ({error}); using caller layout")
        return None


def ccl_allreduce(tensor, mesh_config, ccl_manager, memory_config=None):
    """All-reduce across TP devices.

    Sync ``ttnn.all_reduce`` by default. With ``GEMMA4_CCL_ASYNC=1``, uses
    reduce_scatter_minimal_async + all_gather_async (tt_transformers composite
    pattern) on ``ccl_manager.topology`` (Ring on P150x8).
    """
    if mesh_config is None or mesh_config.tp <= 1:
        return tensor

    caller_memory_config = memory_config
    memory_config = memory_config or ttnn.DRAM_MEMORY_CONFIG
    tp_axis = mesh_config.tp_axis
    topology = ccl_manager.topology
    gather_memory_config = memory_config
    if caller_memory_config is None:
        gather_memory_config = _short_seq_l1_gather_memcfg(tensor, ccl_manager) or memory_config

    chunks = ccl_chunks_per_sync()
    workers = ccl_num_workers_per_link()
    nbuf = ccl_num_buffers_per_channel()
    padded_height = _physical_tile_padded_height(tensor)
    if ccl_async_enabled(padded_height):
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
            # Default async all-gather requires only the first two pre-created semaphores.
            multi_device_global_semaphore=ccl_manager.get_ag_semaphore()[:2],
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

    rs_workers, rs_chunks, rs_buffers = ccl_sync_rs_tuning(
        padded_height, ccl_manager.mesh_device, is_moe=ccl_manager.is_moe
    )
    scattered = ttnn.reduce_scatter(
        tensor,
        dim=3,
        cluster_axis=tp_axis,
        num_links=ccl_manager.num_links,
        topology=topology,
        memory_config=memory_config,
        num_workers_per_link=rs_workers,
        chunks_per_sync=rs_chunks,
        num_buffers_per_channel=rs_buffers,
    )
    tensor.deallocate(True)
    result = ttnn.all_gather(
        scattered,
        dim=3,
        cluster_axis=tp_axis,
        memory_config=gather_memory_config,
    )
    scattered.deallocate(True)
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

    padded_height = _physical_tile_padded_height(tensor)
    if ccl_async_enabled(padded_height):
        # Fresh AG output each call (caller-owned); see ccl_allreduce note.
        gathered = ttnn.experimental.all_gather_async(
            tensor,
            persistent_output_buffer=None,
            dim=dim,
            # Default async all-gather requires only the first two pre-created semaphores.
            multi_device_global_semaphore=ccl_manager.get_ag_semaphore()[:2],
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
