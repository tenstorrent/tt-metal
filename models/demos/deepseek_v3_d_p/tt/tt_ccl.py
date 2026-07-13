# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import ttnn

# NOTE: This file is forked from models/common/modules/tt_ccl.py
#       This is done to include logic for divifing the grid of cores for ring attention
#       One col is taken for the CCL communication of the op

# =============================================================================
# CCL tuning defaults - shared across all TTTv2 modules
# =============================================================================

# Default number of chunks per synchronization barrier in CCL operations.
# Higher values reduce sync overhead but increase latency per chunk.
CCL_CHUNKS_PER_SYNC = 10

# Default number of worker threads per Ethernet link for CCL operations.
CCL_NUM_WORKERS_PER_LINK = 2

# Default number of double-buffered channels per CCL link.
CCL_NUM_BUFFERS_PER_CHANNEL = 2

# =============================================================================
# TT_CCL cache - one instance per mesh_device (semaphores are hardware resources)
# =============================================================================


_tt_ccl_cache: dict[int, "TT_CCL"] = {}


def get_tt_ccl(mesh_device: ttnn.MeshDevice) -> "TT_CCL":
    """Get or create TT_CCL for mesh_device (cached per device id)."""
    mesh_id = mesh_device.id()
    if mesh_id not in _tt_ccl_cache:
        _tt_ccl_cache[mesh_id] = TT_CCL(mesh_device)
    return _tt_ccl_cache[mesh_id]


def clear_tt_ccl_cache():
    """Clear cache (for testing)."""
    _tt_ccl_cache.clear()


# =============================================================================
# TT_CCL class
# =============================================================================


def create_global_semaphores(mesh_device, cores, initial_value):
    # create global semaphore handles
    ccl_semaphore_handles = [ttnn.create_global_semaphore(mesh_device, cores, initial_value) for _ in range(2)]
    return ccl_semaphore_handles


class TT_CCL:
    def __init__(
        self,
        mesh_device,
    ):
        self.mesh_device = mesh_device
        full_compute_grid = self.mesh_device.compute_with_storage_grid_size()
        self.sub_device_crs = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(
                        full_compute_grid.x - 1,
                        full_compute_grid.y - 1,
                    ),
                )
            }
        )

        self.ring_attention_ccl_core_grid_offset = (full_compute_grid.x - 1, 0)

        # create global semaphore handles
        self.ring_attention_ccl_semaphore_handles = create_global_semaphores(mesh_device, self.sub_device_crs, 0)
        # Dedicated pair for the ring-fused indexer_score all-gather (ring_indexer_score_dsa). Kept SEPARATE
        # from ring_attention_ccl_semaphore_handles so the indexer AG cannot interleave semaphore state with
        # the MLA ring_mla / ring_joint SDPA that share ring_attention_ccl_semaphore_handles later in the same
        # forward (same reuse-without-reset lifecycle those ops use across layers).
        self.indexer_ag_semaphore_handles = create_global_semaphores(mesh_device, self.sub_device_crs, 0)

        self.barrier_semaphore_idx = [0, 0, 0]
        self.barrier_semaphore_handles = [[], [], []]

        self.ag_semaphores_idx = [0, 0, 0]
        self.ag_semaphore_handles = [[], [], []]

        self.rs_semaphores_idx = [0, 0, 0]
        self.rs_semaphore_handles = [[], [], []]

        # cluster-axis-0, cluster-axis-1, no-cluster-axis
        for i in range(3):
            # double buffered semaphores
            for _ in range(2):
                self.barrier_semaphore_handles[i].append(
                    ttnn.create_global_semaphore(self.mesh_device, self.sub_device_crs, 0)
                )

                self.ag_semaphore_handles[i].append(
                    [ttnn.create_global_semaphore(self.mesh_device, self.sub_device_crs, 0) for _ in range(2)]
                )

                self.rs_semaphore_handles[i].append(
                    [ttnn.create_global_semaphore(self.mesh_device, self.sub_device_crs, 0) for _ in range(3)]
                )

        # Single, stable-address reduce_scatter INTERMEDIATE accumulator, shared by ALL layers'
        # shared experts. Giving the shared-expert reduce_scatter a persistent, fixed-address
        # intermediate (a) keeps it alive across the shared-expert||dispatch sub-device overlap so
        # the concurrent dispatch can't reuse its freed slot mid-flight, and (b) fixes the DRAM
        # layout every iteration so the op's fabric reduction order is identical -> bit-exact
        # determinism. One buffer for the whole model (layers share the shape and run sequentially)
        # keeps the memory cost flat. See TtSharedExpert.forward.
        self.shared_rs_intermediate = None

        # Persistent ring-attention buffers shared by every layer's MLA, keyed by their shape
        # signature. One set for the whole model. See get_mla_ring_attention_buffers.
        self.mla_ring_attention_buffers: dict[tuple, dict] = {}

        # Persistent chunked-prefill (ring_mla) gathered-KV scratch buffers shared by every layer's
        # MLA, keyed by shape signature. See get_mla_chunked_kv_buffer.
        self.mla_chunked_kv_buffers: dict[tuple, "ttnn.Tensor"] = {}

        # Persistent all-gather OUTPUT buffers for the ring-fused indexer_score op, shared by every
        # layer's DSA indexer, keyed by shape signature. See get_indexer_gathered_buffer.
        self.indexer_gathered_buffers: dict[tuple, "ttnn.Tensor"] = {}

    def get_mla_ring_attention_buffers(
        self,
        *,
        seq_len,
        kv_lora_rank,
        qk_rope_head_dim,
        qk_head_dim,
        v_head_dim,
        num_heads,
        tp_axis,
        dtype=ttnn.bfloat8_b,
    ):
        """Lazily allocate (once per mesh) and return the persistent ring-attention buffers shared by
        every layer's MLA: the all-gather K/V output buffers plus the dummy joint_q/kv/v placeholders
        (seq_len=0) that ring_joint_scaled_dot_product_attention requires. All MLA layers share one
        config + seq_len + mesh, so a single set is reused at a stable address across layers -- layers
        run sequentially (no in-flight overlap), and the fixed address also keeps the op's fabric
        reduction order identical for bit-exact determinism. Cached by shape signature so distinct
        configs/seq_lens on the same mesh get their own set. Returns a dict of ttnn.Tensor."""
        import torch

        key = (seq_len, kv_lora_rank, qk_rope_head_dim, qk_head_dim, v_head_dim, num_heads, tp_axis, dtype)
        if key in self.mla_ring_attention_buffers:
            return self.mla_ring_attention_buffers[key]

        mesh_shape = tuple(self.mesh_device.shape)
        num_heads_local = num_heads // self.mesh_device.shape[tp_axis]
        v_shard_dims = [None, None]
        v_shard_dims[tp_axis] = 1  # TP heads
        k_shard_dims = [None, None]  # replicated across the mesh
        joint_shard_dims = [None, None]
        joint_shard_dims[tp_axis] = 1  # shard on head dimension

        def _alloc(tensor, *, shard_dims=None, replicate=False):
            mapper = (
                ttnn.ReplicateTensorToMesh(self.mesh_device)
                if replicate
                else ttnn.ShardTensor2dMesh(self.mesh_device, mesh_shape=mesh_shape, dims=shard_dims)
            )
            return ttnn.from_torch(
                tensor,
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )

        assert num_heads_local * self.mesh_device.shape[tp_axis] == num_heads
        buffers = {
            "persistent_k_output_buffer": _alloc(
                torch.zeros(1, 1, seq_len, kv_lora_rank + qk_rope_head_dim), shard_dims=k_shard_dims
            ),
            "persistent_v_output_buffer": _alloc(
                torch.zeros(1, num_heads, seq_len, v_head_dim), shard_dims=v_shard_dims
            ),
            "joint_q": _alloc(torch.zeros(1, num_heads, 0, qk_head_dim), shard_dims=joint_shard_dims),
            "joint_kv": _alloc(torch.zeros(1, 1, 0, kv_lora_rank + qk_rope_head_dim), replicate=True),
            "joint_v": _alloc(torch.zeros(1, num_heads, 0, v_head_dim), shard_dims=joint_shard_dims),
        }
        self.mla_ring_attention_buffers[key] = buffers
        return buffers

    def get_mla_chunked_kv_buffer(self, *, cache_batch, seq_len, kvpe_dim, dtype=ttnn.bfloat8_b):
        """Lazily allocate (once per mesh) and return the combined gathered-KV scratch buffer used by
        the chunked-prefill ring_mla op (persistent_output_buffer_kv). It's scratch -- each layer's
        gather overwrites it, it holds no per-layer state -- and uniform across layers (cache_batch =
        slot_num*layer_num, seq_len, mesh are all fixed for a model), so one buffer is shared by every
        layer's MLA instead of re-allocating a full slot_num*layer_num buffer per layer. Replicated
        across the mesh ([None, None]); cached by shape signature."""
        import torch

        key = (cache_batch, seq_len, kvpe_dim, dtype)
        if key not in self.mla_chunked_kv_buffers:
            self.mla_chunked_kv_buffers[key] = ttnn.from_torch(
                torch.zeros(cache_batch, 1, seq_len, kvpe_dim),
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    self.mesh_device, mesh_shape=tuple(self.mesh_device.shape), dims=[None, None]
                ),
            )
        return self.mla_chunked_kv_buffers[key]

    def get_indexer_gathered_buffer(self, *, cache_batch, seq_len_full, head_dim, dtype):
        """Lazily allocate (once per mesh) and return the persistent gathered-K scratch buffer used by
        the ring-fused indexer scorer (ring_indexer_score_dsa's all-gather OUTPUT buffer `k`). Like the
        ring_mla chunked-KV scratch (get_mla_chunked_kv_buffer): each layer's fused gather overwrites it,
        it holds no per-layer state, and it is uniform across layers (cache_batch/seq/dim/mesh are all
        model-fixed), so one buffer is shared by every layer's indexer instead of re-allocating per layer.
        Holds the full gathered seq (seq_len_full = sp * per-chip slab) on EVERY device -- the fused op's
        ring all-gather fills the remote SP bands in place while the local band is dual-sourced from
        k_local -- so it is replicated across the whole mesh. The stale contents between calls are never
        read (remote bands are overwritten by the gather; the local band comes from k_local), so it is not
        re-zeroed. Cached by (cache_batch, seq_len_full, head_dim, dtype)."""
        import torch

        key = (cache_batch, seq_len_full, head_dim, dtype)
        if key not in self.indexer_gathered_buffers:
            self.indexer_gathered_buffers[key] = ttnn.from_torch(
                torch.zeros(cache_batch, 1, seq_len_full, head_dim),
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        return self.indexer_gathered_buffers[key]

    def get_shared_rs_intermediate(self, input_tensor):
        """Lazily allocate (once per mesh) and return the shared reduce_scatter intermediate
        accumulator. Line (Linear) topology needs a double-sized leading dim for the
        forward/backward halves: shape = [2, *input_shape]. Interleaved DRAM, input dtype/layout,
        replicated across the mesh. A single buffer is reused at a stable address by every
        shared-expert reduce_scatter — all layers share the same shape and run sequentially, so one
        buffer for the whole model is safe."""
        import torch

        if self.shared_rs_intermediate is None:
            self.shared_rs_intermediate = ttnn.from_torch(
                torch.zeros([2] + list(input_tensor.shape)),
                device=self.mesh_device,
                layout=input_tensor.layout,
                dtype=input_tensor.dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        return self.shared_rs_intermediate

    def get_and_cycle_barrier_semaphore_handle(self, cluster_axis=None):
        semaphore_index = 2 if cluster_axis is None else cluster_axis
        current_idx = self.barrier_semaphore_idx[semaphore_index]
        self.barrier_semaphore_idx[semaphore_index] = (current_idx + 1) % 2
        return self.barrier_semaphore_handles[semaphore_index][current_idx]

    def get_and_cycle_ag_semaphore_handles(self, cluster_axis=None):
        semaphore_index = 2 if cluster_axis is None else cluster_axis
        current_idx = self.ag_semaphores_idx[semaphore_index]
        self.ag_semaphores_idx[semaphore_index] = (current_idx + 1) % 2
        return self.ag_semaphore_handles[semaphore_index][current_idx]

    def get_and_cycle_rs_semaphore_handles(self, cluster_axis=None):
        semaphore_index = 2 if cluster_axis is None else cluster_axis
        current_idx = self.rs_semaphores_idx[semaphore_index]
        self.rs_semaphores_idx[semaphore_index] = (current_idx + 1) % 2
        return self.rs_semaphore_handles[semaphore_index][current_idx]

    def get_num_links(self, cluster_axis=None):
        """Get the number of available Ethernet links for CCL operations on this mesh device."""
        return get_num_links(self.mesh_device, cluster_axis)


# =============================================================================
# Topology auto-detection
# =============================================================================


# todo)) work with the CCL team to find opportunity to simplify this --> e.g., build into TTNN APIs?
def default_topology(mesh_device: ttnn.MeshDevice) -> Optional[ttnn.Topology]:
    """Auto-detect CCL topology based on cluster type and device count."""
    num_devices = mesh_device.get_num_devices()
    if num_devices == 8 and ttnn.cluster.get_cluster_type() in [
        ttnn.cluster.ClusterType.T3K,
        ttnn.cluster.ClusterType.GALAXY,
    ]:
        # NOTE: we always want to do ring if it is available
        return ttnn.Topology.Ring
    elif num_devices > 1:
        # NOTE: this should be a fallback when the ring is not available
        return ttnn.Topology.Linear
    return None


# =============================================================================
# Device name / link count helpers (copied from TTTv1 ccl.py + model_config.py
# to avoid importing from tt_transformers)
# =============================================================================


def _determine_device_name(mesh_device: ttnn.MeshDevice) -> str:
    """Determine device name based on number of devices and architecture."""
    num_devices = mesh_device.get_num_devices() if mesh_device else 0
    arch_name = ttnn.get_arch_name()
    dram_grid_size = mesh_device.dram_grid_size() if mesh_device else None

    if num_devices == 0:
        return "CPU"

    if "blackhole" in arch_name:
        dict_device_names = {
            1: "P100" if dram_grid_size and dram_grid_size.x == 7 else "P150",
            2: "P300",
            4: "P150x4",
            8: "P150x8",
            32: "BHGLX",
        }
    elif "wormhole_b0" in arch_name:
        dict_device_names = {
            1: "N150",
            2: "N300",
            4: "N150x4",
            8: "T3K",
            32: "TG",
        }
    else:
        raise ValueError(f"Unsupported architecture: {arch_name}")

    if num_devices in dict_device_names:
        return dict_device_names[num_devices]
    raise ValueError(f"Unsupported number of devices: {num_devices} for {arch_name}")


def get_num_links(mesh_device: ttnn.MeshDevice, cluster_axis: int | None = None) -> int:
    """
    Get the number of available Ethernet links for CCL operations.

    Args:
        mesh_device: The mesh device to query.
        cluster_axis: Optional cluster axis to query links for.
            - 0: vertical axis (North-South).
            - 1: horizontal axis (East-West).
            - None: minimum across all axes.

    Returns:
        int: The number of available links.
    """
    device_name = _determine_device_name(mesh_device)
    link_dict = {
        "P100": (0, 0),
        "P150": (0, 0),
        "N150": (0, 0),
        "N300": (1, 1),
        "T3K": (1, 1),
        "P150x4": (2, 2),
        "P150x8": (2, 2),
        "P300": (2, 2),
        "BHGLX": (2, 2),  # NOTE: Possible increase to 4 when it's enabled
        "TG": (4, 4),
        "N150x4": (1, 1),
    }
    device_links = link_dict[device_name]
    if cluster_axis is None:
        return min(device_links)
    if cluster_axis in (0, 1):
        return device_links[cluster_axis]
    return min(device_links)
