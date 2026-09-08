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


def default_ccl_packet_bytes():
    """Ideal Fabric packet for dense Gemma4 width-sharded CCL pages.

    Matches ``validate_packet_size`` / runtime guidance (≈3× page):
      WH 31B: 2048 B pages → 6144 (Fabric warns on the 4352 default)
      BH 31B: 1344 B pages → 5376
      12B:    960 B pages  → 3840
    Other models leave Fabric's default (``None``).
    """
    model = os.environ.get("HF_MODEL", "").lower()
    if "31b" in model:
        return 5376 if is_blackhole() else 6144
    if "12b" in model:
        return 3840
    return None


def fabric_router_config_from_env():
    """``FabricRouterConfig`` for mesh open, or ``None`` to keep Fabric defaults.

    ``GEMMA4_CCL_PACKET_BYTES`` overrides: unset → :func:`default_ccl_packet_bytes`;
    ``0`` / ``none`` / ``default`` → Fabric default; else ``max(4352, int)``.
    Shared by demo ``_device_params`` and ``parametrize_mesh_with_fabric``.
    """
    # Wormhole Fabric rejects payloads above this (TT_FATAL); BH keeps its own
    # ceiling via the unset/default path.
    _wh_max_packet_bytes = 7616

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
        pkt_bytes = min(pkt_bytes, _wh_max_packet_bytes)
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


def ccl_sync_split_enabled() -> bool:
    """Run the TP all-reduce as sync ``reduce_scatter`` + sync ``all_gather``
    instead of the fused ``ttnn.all_reduce``. Default ON; ``GEMMA4_CCL_SPLIT=0``
    opts back out.

    ``ttnn.all_reduce`` *is* those two ops -- identical to within noise and
    ``torch.equal`` bit-identical on per-device-distinct data. But the fused op
    exposes only
    {cluster_axis, memory_config, num_links, topology, subdevice_id}, while the
    sync halves also expose ``chunks_per_sync`` / ``num_workers_per_link`` /
    ``num_buffers_per_channel``. Splitting therefore costs nothing and unlocks
    the knobs -- see ``ccl_sync_rs_workers``.

    Tall prefill may take the async path instead (``ccl_async_enabled``); this
    flag only applies when async is off.
    """
    return os.environ.get("GEMMA4_CCL_SPLIT", "1").lower() not in ("0", "false", "no")


# Prefill RS worker/chunk switch: decode and short prefill stay latency-bound
# (w=1,c=1). T3K chunk height 2048 (~22 MB) is bandwidth-bound and wants w=2,c=2.
_PREFILL_RS_TALL_HEIGHT = 2048

# At the T3K chunk height the async path (w=2, c=10) beats the sync split, and is
# torch.equal against the fused op. Decode / short prefill stay sync -- async
# lost on small payloads there, and that path takes the L1 gather.
_CCL_ASYNC_MIN_HEIGHT = 2048


def ccl_sync_rs_workers(padded_height: int | None = None) -> int:
    """``num_workers_per_link`` for the split all-reduce's reduce-scatter.

    At decode / short prefill the winner is ``w=1, c=1``, and it is bit-exact --
    the reduction order is unchanged, only the worker/sync granularity is.
    ``num_buffers_per_channel`` is noise here; 4 is taken as the middle.

    Prefill is height-dependent: a ~1 MB payload still wants ``w=1,c=1``, while
    the T3K chunk height (~22 MB) wants ``w=2,c=2``, still bit-exact. Hence the
    height-aware default below; ``GEMMA4_CCL_SYNC_RS_WORKERS`` overrides.

    Note ``w=4`` is a cliff at decode / short prefill, not a plateau: with a
    single link, extra workers contend. Do not raise this without re-sweeping,
    and do not confuse it with the async path's ``GEMMA4_CCL_NUM_WORKERS``
    default of 2.

    The GATHER half was swept over the same knobs (w x c x b) and is completely
    insensitive. It runs on ONE worker core (vs the reduce-scatter's 6) and sits
    at the ``num_links=1`` fabric floor, not core starvation. Leave it on
    defaults.
    """
    env = os.environ.get("GEMMA4_CCL_SYNC_RS_WORKERS")
    if env is not None and str(env).strip() != "":
        return max(1, int(env))
    if padded_height is not None and int(padded_height) >= _PREFILL_RS_TALL_HEIGHT:
        return 2
    return 1


def ccl_sync_rs_chunks(padded_height: int | None = None) -> int:
    """``chunks_per_sync`` for the split all-reduce's reduce-scatter.

    Decode / short prefill want ``c=1``; raising it only costs time. At prefill
    M=2048, ``c=2`` with ``w=2`` is the isolated winner. See
    ``ccl_sync_rs_workers``.
    """
    env = os.environ.get("GEMMA4_CCL_SYNC_RS_CHUNKS")
    if env is not None and str(env).strip() != "":
        return max(1, int(env))
    if padded_height is not None and int(padded_height) >= _PREFILL_RS_TALL_HEIGHT:
        return 2
    return 1


def ccl_sync_rs_buffers() -> int:
    """``num_buffers_per_channel`` for the split all-reduce's reduce-scatter.
    Insensitive across the swept range; see ``ccl_sync_rs_workers``."""
    return max(1, int(os.environ.get("GEMMA4_CCL_SYNC_RS_BUFFERS", "4")))


def default_ccl_topology(mesh_device=None, is_moe: bool = False):
    """Default CCL topology for Gemma4 TP collectives.

    Override with ``GEMMA4_CCL_TOPOLOGY=ring|linear``.

    Policy (when env unset):
      * **Ring** on **Blackhole** meshes with **≥8 devices** (Ring+sync beat
        Linear+sync on the P150x8 TTFT sweep at 31B/128k).
      * **Linear** on Wormhole, including dense 1×8. Ring remains available via
        the environment override, but is not the default until its full-model
        PCC is revalidated.
      * **Linear** everywhere else. Ring on 4-device BH drops 12B full-model PCC
        well below the Linear result.
    """
    override = os.environ.get("GEMMA4_CCL_TOPOLOGY", "").strip().lower()
    if override in ("ring", "r"):
        return ttnn.Topology.Ring
    if override in ("linear", "line", "l"):
        return ttnn.Topology.Linear

    n = mesh_device.get_num_devices() if mesh_device is not None else 0
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


def ccl_async_enabled(padded_height: int | None = None) -> bool:
    """True when TP all-reduce / all-gather should use async RS+AG.

    ``GEMMA4_CCL_ASYNC=1/0`` forces on/off for every height. When unset, async
    auto-enables only for ``padded_height >= 2048`` (bandwidth-bound prefill
    chunks), where it is bit-exact and beats the sync split. Opt out of
    that auto path with ``GEMMA4_CCL_ASYNC_PREFILL=0`` without enabling decode
    async. Decode and short prefill stay on the sync split + L1-gather path.
    """
    env = os.environ.get("GEMMA4_CCL_ASYNC")
    if env is not None:
        return env.lower() in ("1", "true", "yes")
    if padded_height is not None and int(padded_height) >= _CCL_ASYNC_MIN_HEIGHT:
        return os.environ.get("GEMMA4_CCL_ASYNC_PREFILL", "1").lower() not in ("0", "false", "no")
    return False


class CCLManager:
    """CCL manager for Gemma4 tensor parallelism.

    Stores mesh_device, num_links, and topology for CCL operations.
    Semaphores support the async RS+AG path (forced via ``GEMMA4_CCL_ASYNC=1``
    or auto for tall prefill — see ``ccl_async_enabled``).
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
        async_env = os.environ.get("GEMMA4_CCL_ASYNC")
        if async_env is not None and async_env.lower() in ("1", "true", "yes"):
            async_mode = "force-on"
        elif async_env is not None:
            async_mode = "force-off"
        else:
            async_mode = f"auto(h>={_CCL_ASYNC_MIN_HEIGHT})"
        logger.info(
            f"Gemma4 CCLManager: devices={self.num_devices} num_links={num_links} "
            f"topology={topo_name} async={async_mode} "
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


def _short_seq_l1_gather_memcfg(tensor, ccl_manager):
    """Width-sharded L1 memory config for the all-gather output, or None to keep DRAM.

    Every ``ccl_allreduce`` call site in the decode path feeds its result straight
    into an ``RMSNorm`` (layer.py: post_attention_layernorm, post_feedforward_
    layernorm{,_1,_2}), and that norm's first act is to width-shard its input. So
    having the gather write that layout directly removes an
    InterleavedToSharded per all-reduce, bit-exact.

    Decode: tile-aligned height <= ``TILE_SIZE``. Short prefill (physical
    height ``N*C*H`` <= ``_SHARDED_NORM_MAX_HEIGHT``): same win for post-attn /
    post-MLP LN. Batched prefill is ``[B, 1, S, H]`` — shard height is ``B*S``,
    not ``S``. Prefill may keep the LN/residual island
    (``prefill_mlp_island_enabled``); do *not* feed that shard into gate_up —
    Wormhole 1D prefill CBs clash (SharedMLP S2I's to interleaved for M > TILE).
    """
    if not ccl_l1_gather_enabled():
        return None
    try:
        shape = tensor.shape
        if len(shape) != 4:
            return None
        from models.demos.gemma4.tt.rms_norm import (
            _SHARDED_NORM_MAX_HEIGHT,
            activation_physical_height,
            sharded_norm_enabled,
            width_shard_input_memcfg,
        )

        if not sharded_norm_enabled():
            return None
        padded_height = activation_physical_height(shape)
        if not (1 <= padded_height <= _SHARDED_NORM_MAX_HEIGHT):
            return None
        return width_shard_input_memcfg(ccl_manager.mesh_device, shape[-1], padded_height)
    except Exception as e:  # never let a layout optimization break the model
        logger.debug(f"ccl L1 gather memcfg unavailable ({e}); keeping DRAM")
        return None


def _decode_l1_gather_memcfg(tensor, ccl_manager):
    """Alias for ``_short_seq_l1_gather_memcfg`` (decode and short prefill)."""
    return _short_seq_l1_gather_memcfg(tensor, ccl_manager)


def ccl_allreduce(tensor, mesh_config, ccl_manager, memory_config=None):
    """All-reduce across TP devices.

    By default, sync ``ttnn.reduce_scatter`` + sync ``ttnn.all_gather`` with a
    swept reduce-scatter worker config (``GEMMA4_CCL_SPLIT=0`` falls back to the
    fused ``ttnn.all_reduce``, which is bit-identical but slower -- see
    ``ccl_sync_split_enabled`` / ``ccl_sync_rs_workers``).

    Async RS+AG (``reduce_scatter_minimal_async`` + ``all_gather_async``) is
    auto-selected for tall prefill (``ccl_async_enabled``); force with
    ``GEMMA4_CCL_ASYNC=1``. Decode / short prefill stay on the sync split.
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
        gather_memory_config = _short_seq_l1_gather_memcfg(tensor, ccl_manager) or memory_config

    h = int(tensor.shape[-2])
    padded_h = ((h + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
    chunks = ccl_chunks_per_sync()
    workers = ccl_num_workers_per_link()
    nbuf = ccl_num_buffers_per_channel()
    if ccl_async_enabled(padded_h):
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
        # Prefer gather_memory_config so a future short-seq async path can still
        # L1-gather; at M>=2048 that helper returns DRAM.
        gathered = ttnn.experimental.all_gather_async(
            scattered,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=ccl_manager.get_ag_semaphore(),
            num_links=ccl_manager.num_links,
            cluster_axis=tp_axis,
            topology=topology,
            memory_config=gather_memory_config,
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
            num_workers_per_link=ccl_sync_rs_workers(padded_h),
            chunks_per_sync=ccl_sync_rs_chunks(padded_h),
            num_buffers_per_channel=ccl_sync_rs_buffers(),
        )
        tensor.deallocate(True)
        # num_links/topology are deprecated-and-ignored on the new ttnn.all_gather
        # (tt-metal 3218270556c, "New ttnn.all_gather" #48301): passing them only
        # logs the Sep-2026 removal warning. Links/topology come from the Fabric
        # config now — see ccl_allgather() below, which already omits them.
        result = ttnn.all_gather(
            scattered,
            dim=3,
            cluster_axis=tp_axis,
            memory_config=gather_memory_config,
        )
        scattered.deallocate(True)
        return result

    result = ttnn.all_reduce(
        tensor,
        cluster_axis=tp_axis,
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
    h = int(tensor.shape[-2]) if len(tensor.shape) >= 2 else 0
    padded_h = ((h + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE if h else None

    if ccl_async_enabled(padded_h):
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
