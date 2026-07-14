# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Fused MoE operation: Routed Expert + Shared Expert.

Architecture:
  MoeRoutedExpertOp  — context setup & build methods for routed expert
  MoeSharedExpertOp  — context setup & build methods for shared expert
  MoeOp              — top-level orchestrator that composes both

Pipeline (routed expert):
  1. Input mcast → 2. Gate matmul → 3. Gate gather → 4. Gate
  5. Index/Scale mcast → 6. gate_proj → 7. up_proj → 8. Fused mul
  9. down_proj gather → 10. down_proj mcast → 11. down_proj → 12. Eltwise add

Pipeline (shared expert, fused):
  3a. Gate/Up KN-sliced matmul (runs on 128 cores after input mcast)
"""

import math
from dataclasses import dataclass
from typing import Any, Optional

import ttnn
from models.demos.deepseek_v3_b1.circular_buffer_utils import (
    CircularBufferIdManager,
    build_cb_reconfig_tensor,
    cb_descriptor_from_overlapped_tensor,
    record_cb_metadata,
    record_cb_metadata_per_coord,
)
from models.demos.deepseek_v3_b1.fused_ops.face_view_utils import FACE_HEIGHT, FACE_WIDTH, can_use_face_view
from models.demos.deepseek_v3_b1.micro_ops.reduce_to_one_b1.op import MESH_LEAF, MESH_ROOT1, MESH_ROOT2, MESH_ROOT3
from models.demos.deepseek_v3_b1.micro_ops.reduce_to_one_b1.op import get_device_role as get_reduce_device_role
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreCompileTimeDescriptor,
    PerCoreRuntimeArgsDescriptor,
    UnifiedCompileTimeCoreDescriptor,
    UnifiedKernelDescriptor,
)
from models.demos.deepseek_v3_b1.utils import float_to_uint32, get_pinned_optimal_dram_bank_to_logical_worker_assignment


def _fused_base_addr(t):
    """Return a usable base address for ``t`` regardless of allocator mode.

    When ``TT_METAL_ALLOCATOR_MODE_HYBRID=1`` is set and a tensor is per-core
    allocated (``experimental_set_per_core_allocation(True)``), ``buffer_address()``
    returns 0 because the buffer has no single global base. Fall back to
    ``experimental_per_core_buffer_address`` on any core in the shard grid; per-core
    addresses are asserted to be uniform across the grid at allocation time
    (``assert_uniform_per_core_addresses``), so any core returns the same base.
    """
    if hasattr(t, "is_per_core_allocated") and t.is_per_core_allocated():
        core = t.memory_config().shard_spec.grid.bounding_box().start
        return t.experimental_per_core_buffer_address(core)
    return t.buffer_address()


class MoeSem:
    """Global semaphore index constants for the fused MoE op.

    Each index maps to a separate global semaphore created by MoeOp.create_semaphores().
    Previously overlapping IDs (gather/mcast on different cores) are now split.
    """

    MCAST_SENDER = 0
    MCAST_DATA_RECEIVER = 1
    DOWN_PROJ_GATHER = 2
    RESIDUAL_MCAST_RECEIVER = 3
    AG_GATHER = 4
    SHARED_DOWN_MCAST_RECEIVER = 5
    BG_GATHER = 6
    SHARED_OUTPUT_MCAST_RECEIVER = 7
    OUTPUT_GATHER = 8
    EXPERT_SCALE_MCAST_RECEIVER = 9
    INDEX_MCAST_RECEIVER = 10
    DOWN_PROJ_MCAST_RECEIVER = 11
    REDUCE_WORKER_FABRIC_BASE = 12  # Shared worker->fabric ready semaphore (legacy name retained)
    REDUCE_SYNC = 16
    REDUCE_AGG_SYNC = 17
    REDUCE_PERSISTENT_FABRIC_SIGNAL = 18
    # Dedicated semaphores for SRAM gate/up gather. Cannot reuse AG_GATHER/
    # BG_GATHER even though SRAM gather runs sequentially with shared gather:
    # without atomic barriers between rounds, BRISC of fast a_cores can inc
    # the shared sem before NCRISC has reset it after the SRAM round (or
    # vice versa), causing lost increments.
    SRAM_AG_GATHER = 19
    SRAM_BG_GATHER = 20
    # Dedicated SRAM down mcast receiver sem (can't reuse DOWN_PROJ_MCAST_RECEIVER:
    # both mcasts target overlapping receivers and run back-to-back, racey otherwise).
    SRAM_DOWN_MCAST_RECEIVER = 21
    # Per-core sync sem used by sync_riscs_enter/exit inside scan_n_sram_active
    # (sender_scan + receiver_scan share it — sequential within an iter, sync
    # primitives leave the sem at (0,0) after each call).
    SCAN_SYNC = 22
    NUM_SEMAPHORES = 23


@dataclass
class MoeContext:
    """Top-level context for the fused MoE op. Hides routed/shared split from op()."""

    routed_ctx: Any
    shared_ctx: Any

    # Device & mesh
    mesh_device: Any
    full_device_grid: Any
    mesh_rows: int
    mesh_cols: int

    # Flags
    enable_routing: bool
    enable_reduce_to_one: bool
    enable_bcast: bool

    # IO tensors (stored for _build_io_tensors)
    gate_mm_weights_tensor: Any
    gate_bias_tensor: Any
    gate_indices_tensor: Any
    gate_output_scores_tensor: Any
    gate_output_indices_tensor: Any
    gate_proj_weights_tensor: Any
    up_proj_weights_tensor: Any
    down_proj_weights_tensor: Any
    final_output_tensor: Any
    rmsnorm_gamma_tensor: Any
    shared_residual_mcast_src_tensor: Any
    shared_gate_weights_fused_tensor: Any  # shared_gate_weights_overlapped.fused_tensor
    shared_down_weights_tensor: Any
    sdpa_kv_cache_buffer: Any
    sdpa_out_interm_buffer: Any

    # Reduce params (None when reduce disabled)
    reduce_params: dict = None
    reduce_intermediate_tensors: Any = None
    reduce_output_tensor: Any = None

    # RMSNorm runtime args (common across all devices)
    rmsnorm_epsilon_packed: int = 0
    rmsnorm_scalar_packed: int = 0

    # Bcast params
    bcast_params: dict = None
    bcast_input_tensor: Any = None
    bcast_intermediate_tensor: Any = None
    bcast_semaphores: Any = None
    bcast_sender_coord: Any = None
    socket: Any = None

    # CB reconfig
    reconfig_moe_cbs: bool = False
    # SRAM BSPM divergence: when True, per-device SRAM weight CBs have
    # different L1 addresses across the mesh.  Forces the reconfig tensor
    # to be sharded per-device (ShardTensor2dMesh) instead of replicated.
    enable_sram_bspm: bool = False


@dataclass
class _MoeRoutedExpertContext:
    """Holds all computed values needed by MoeRoutedExpertOp helper methods."""

    # Device & mesh
    device: Any
    full_device_grid: Any
    device_grid_size: Any
    mesh_rows: int
    mesh_cols: int

    # Core grids
    sender_core: Any
    input_core_grid: Any
    mcast_grid: Any
    mcast_worker_grid: Any
    gate_proj_core_ranges: Any
    num_gate_proj_cores: int

    # Data format & tiles
    data_format: Any
    tile_1x32_size: int
    num_tiles_k: int

    # CB indices (shared between routed and non-routed paths)
    rmsnorm_output_cb: int
    gate_mm_input_cb: int  # Also used as mcast destination for gate_proj/up_proj input
    gate_proj_cb_in0: int
    gate_proj_cb_in1: int
    gate_proj_cb_out: int
    gate_proj_cb_fmt: int
    # cb_out_silu aliases gate_proj_cb_out's L1 region with a tall tile covering
    # all expert outputs so the silu fast-path processes them as one tile.
    gate_proj_cb_out_silu: int
    gate_proj_silu_tile_h: int
    up_proj_cb_in0: int
    up_proj_cb_in1: int
    up_proj_cb_mm_out: int
    up_proj_cb_fmt: int
    mul_cb_in0: int
    mul_cb_in1: int
    mul_cb_out: int
    down_proj_gather_dst_cb: int
    down_proj_mcast_dst_cb: int
    down_proj_cb_in0: int
    down_proj_cb_in1: int
    down_proj_cb_out: int
    down_proj_cb_fmt: int
    # down_proj internal accumulation CB (aliases down_proj_cb_out's L1 region).
    # Per-expert push/pop bookkeeping routes through this so cb_out's metadata
    # only updates once at end. Used only by down_proj (the accum-experts proj).
    down_proj_cb_internal_acc: int
    add_cb_in0: int
    add_cb_in1: int
    add_cb_out: int
    # Semaphore IDs (shared)
    mcast_data_sender_semaphore_addr: int
    mcast_data_receiver_semaphore_addr: int
    gather_noc0_receiver_semaphore_addr: int
    gather_noc1_receiver_semaphore_addr: int

    # Setup result dicts (shared)
    rmsnorm_mcast_params: dict
    gate_proj_params: dict
    up_proj_params: dict
    mul_params: dict
    down_proj_gather_params: dict
    down_proj_mcast_params: dict
    sram_down_mcast_params: dict
    down_proj_params: dict
    add_params: dict

    # Derived values
    mul_num_tiles: int

    # Pre-built CB descriptors (shared)
    rmsnorm_output_cb_descriptor: Any
    gate_mm_input_cb_descriptor: Any

    # Residual mcast (input → shared expert matmul cores)
    residual_mcast_src_cb: int
    residual_mcast_dst_cb: int
    residual_mcast_receiver_semaphore_addr: int
    residual_mcast_src_cb_descriptor: Any
    residual_mcast_params: dict

    # RMSNorm (sender core: raw input → normalized input before input mcast)
    rmsnorm_gamma_cb: int
    rmsnorm_gamma_cb_descriptor: Any
    rmsnorm_epsilon_packed: int
    rmsnorm_scalar_packed: int
    rmsnorm_num_tiles: int
    rmsnorm_gamma_num_pages: int

    # Per-core values (for core descriptors)
    bank_id_core_values: list
    vc_core_values: list
    sender_idx_core_values: list

    # --- Routing-only fields (unused when enable_routing=False) ---
    enable_routing: bool = True

    # Routing CB indices
    gate_mm_weights_cb: int = 0
    gate_mm_output_cb: int = 0
    gate_input_cb: int = 0
    gate_bias_cb: int = 0
    gate_indices_cb: int = 0
    gate_output_cb: int = 0
    gate_output_indices_cb: int = 0
    gate_proj_cb_index: int = 0
    mul_cb_scalar_src: int = 0
    mul_cb_scalar: int = 0

    # Routing semaphore IDs
    expert_scale_mcast_sender_semaphore_addr: int = 0
    expert_scale_mcast_receiver_semaphore_addr: int = 0
    # Per-core sync sem for scan_n_sram_active (sender_scan + receiver_scan).
    scan_sync_sem_addr: int = 0

    # Routing setup result dicts
    gate_mm_params: dict = None
    gate_mm_gather_params: dict = None
    gate_params: dict = None

    # Index mcast params (routing only)
    index_mcast_sender_semaphore_addr: int = 0
    index_mcast_receiver_semaphore_addr: int = 0
    index_mcast_num_pages: int = 0
    index_mcast_data_size_bytes: int = 0

    # Expert scale mcast params (routing only)
    expert_scale_mcast_num_pages: int = 0
    expert_scale_mcast_data_size_bytes: int = 0

    # Routing CB descriptor
    gate_proj_cb_index_descriptor: Any = None

    # cb_fmt CB descriptors (MatmulExpertCompressedDRAM double-buffered per-expert fmt metadata)
    gate_proj_cb_fmt_descriptor: Any = None
    up_proj_cb_fmt_descriptor: Any = None
    down_proj_cb_fmt_descriptor: Any = None

    # Testing flag (routing only)
    use_hardcoded_expert_index: bool = False

    # ReduceToOne
    enable_reduce_to_one: bool = False
    reduce_local_cb: int = 0
    reduce_received_cb: int = 0
    reduce_output_cb: int = 0
    reduce_scratch_cb: int = 0
    reduce_packet_cb: int = 0
    reduce_params: dict = None
    # Pre-built CB descriptors for reduce (set by _overlap_cbs_with_sdpa_buffer)
    reduce_received_cb_descriptors: list = None  # CB for received (3-page) + output
    reduce_scratch_cb_descriptor: Any = None
    reduce_packet_cb_descriptor: Any = None

    # Broadcast
    enable_bcast: bool = False
    bcast_pkt_cb: int = 0
    bcast_pkt_cb_descriptor: Any = None
    bcast_params: dict = None

    # SRAM routed gate_proj (separate-pipeline plan; see SRAM_SEPARATE_PIPELINE.md)
    # CB IDs always allocated so the kernel's setup_sharded_buffer + Op call have
    # valid IDs even when no SRAM experts are placed (T=0). Activation is gated
    # purely by sram_gate_proj_params being non-empty (= num_sram_experts>0).
    sram_gate_proj_cb_in1: int = 0
    sram_gate_proj_out_cb: int = 0
    sram_gate_proj_params: dict = None
    # CB descriptors built only when SRAM weights are provided.
    # cb_in1 uses CompressedTensor.cb_descriptor_from_compressed_tensor() which
    # returns a LIST of per-core descriptors (per_core_allocation=True). Stored
    # per device since each mesh coord has its own L1 layout.
    # cb_out_descriptor is built in _overlap_cbs_with_sdpa_buffer (kv_buf-overlaid).
    sram_gate_proj_cb_in1_descriptors_per_device: Any = None  # dict[MeshCoordinate -> list of CBDescriptors]
    sram_gate_proj_cb_out_descriptor: Any = None

    # SRAM routed up_proj — mirror of sram_gate_proj_*; runs on shared up cores.
    sram_up_proj_cb_in1: int = 0
    sram_up_proj_out_cb: int = 0
    sram_up_proj_params: dict = None
    sram_up_proj_cb_in1_descriptors_per_device: Any = None
    sram_up_proj_cb_out_descriptor: Any = None

    # SRAM routed down_proj — runs on the 112 shared mcast receiver cores.
    # accum_experts=True so the kernel sums across SRAM-flagged TopK winners.
    # Reads from sram_down_mcast_dst_cb (compact n_sram_active layout).
    sram_down_proj_cb_in1: int = 0
    sram_down_proj_out_cb: int = 0
    sram_down_proj_params: dict = None
    sram_down_proj_cb_in1_descriptors_per_device: Any = None
    sram_down_proj_cb_out_descriptor: Any = None

    # When 1, SRAM matmul kernels take the compressed_custom_mm path (per-tile
    # BSPM format codes). When 0, they take the plain custom_mm path (uniform
    # BFP4). Threaded into kernel CT args as sram_use_compression.
    sram_use_compression: int = 0

    # Merged down output CB — receives either eltwise_add(sram_down, shared_down)
    # when n_sram_active > 0, or a copy of shared_down_matmul_out when n_sram_active = 0.
    # Replaces shared_down_matmul_out_cb as residual_add's in0.
    merged_down_out_cb: int = 0
    merged_down_out_cb_descriptor: Any = None

    # SRAM routed gate/up gather (step 3 of separate-pipeline plan).
    # Each of the 64 a_cores (gate) / b_cores (up) sends num_active_experts
    # tiles to the sender core. Dst CB is on sender, sized 8 experts × 64 tiles
    # = 512 tiles. Layout: expert-major × N-major × K-within-N, so GatedReduce
    # can read tiles_per_k=8 contiguous K-partials per N-tile.
    sram_group1_cb: int = 0
    sram_group2_cb: int = 0
    # gather dst CB descriptors (kv_buf-overlaid in _overlap_cbs_with_sdpa_buffer).
    sram_group1_cb_descriptor: Any = None
    sram_group2_cb_descriptor: Any = None
    # SRAM extended GatedReduce CBs (sender_core only).
    sram_intermed_cb: int = 0
    sram_mcast_src_cb: int = 0
    sram_gr_scalar_cb: int = 0
    sram_down_mcast_dst_cb: int = 0
    sram_intermed_cb_descriptor: Any = None
    sram_mcast_src_cb_descriptor: Any = None
    sram_gr_scalar_cb_descriptor: Any = None
    sram_down_mcast_dst_cb_descriptor: Any = None
    # Face-tile descriptor + size (= [16, 16] face) — shared by all SRAM
    # gather-output and gated-reduce CBs so they all use face_view.
    face_tile_desc: Any = None
    face_tile_size: int = 0
    sram_ag_receiver_data_addr: int = 0
    sram_bg_receiver_data_addr: int = 0
    sram_ag_sender_idx_core_values: list = None  # [(core, sender_idx), ...] for a_cores
    sram_bg_sender_idx_core_values: list = None  # [(core, sender_idx), ...] for b_cores
    sram_gather_data_size_bytes: int = 0  # per-tile bytes (1x32 bf16 = 64)
    sram_gather_expert_dst_stride: int = 0  # 64 tiles × 64 bytes = 4096
    sram_gather_total_tiles: int = 0  # 8 experts × 64 cores = 512
    # Reuse AG_GATHER/BG_GATHER semaphores: SRAM gather runs sequentially
    # before shared expert gather, each gather resets its sem to 0 at end.
    sram_ag_receiver_semaphore_addr: int = 0
    sram_bg_receiver_semaphore_addr: int = 0


@dataclass
class _MoeSharedExpertContext:
    """Holds shared expert values for the fused MoE kernel."""

    # Core grids
    compute_core_grid: Any
    a_compute_grid: Any
    b_compute_grid: Any
    a_cores_list: list
    b_cores_list: list
    matmul_core_grid: Any

    # CB indices (flat for compile-time arg readability)
    gu_weights_cb: int
    gu_out_cb: int
    group1_cb: int  # gate gather dst (on sender)
    group2_cb: int  # up gather dst (on sender)
    intermed_cb: int  # gated reduce intermediate (on sender)
    mcast_src_cb: int  # gated reduce output (on sender)
    residual_cb: int  # residual destination on mcast grid
    down_mcast_dst_cb: int

    # Parallelism
    n_parallel: int
    k_parallel: int
    num_compute_cores: int  # 64 per branch

    # Gather params (shared between A and B gathers)
    gather_dest_noc_core: Any
    gu_gather_data_size_bytes: int
    total_gather_tiles: int

    # Semaphore IDs
    ag_receiver_semaphore_addr: int
    bg_receiver_semaphore_addr: int
    ag_noc1_receiver_semaphore_addr: int
    bg_noc1_receiver_semaphore_addr: int
    shared_mcast_sender_semaphore_addr: int
    shared_mcast_receiver_semaphore_addr: int
    output_gather_noc0_receiver_semaphore_addr: int
    output_gather_noc1_receiver_semaphore_addr: int
    output_mcast_sender_semaphore_addr: int
    output_mcast_receiver_semaphore_addr: int

    # Keep-alive tensors and derived addresses
    ag_dummy_tensor: Any
    bg_dummy_tensor: Any
    ag_receiver_data_addr: int
    bg_receiver_data_addr: int
    down_mcast_dst_dummy_tensor: Any
    output_gather_dst_dummy_tensor: Any

    # Setup result dicts (grouped by operation)
    gu_matmul_params: dict  # from setup_kn_sliced_matmul
    gated_reduce_params: dict  # from setup_gated_reduce
    down_mcast_params: dict  # down mcast dimensions + CB descriptor
    down_matmul_params: dict  # down proj matmul dimensions + CB descriptors
    residual_add_params: dict  # residual add dimensions + CB descriptor
    output_gather_params: dict  # output gather dimensions + CB descriptor
    output_mcast_params: dict  # output mcast dimensions

    # Per-core values
    gu_k_offset_core_values: list
    ag_sender_idx_core_values: list
    bg_sender_idx_core_values: list


class MoeRoutedExpertOp:
    """
    MoE Routed Expert fused operation (refactored).

    Follows the SharedExpertOp pattern: context dataclass + decomposed build methods.
    """

    # Fused activation enum values (must match matmul.hpp FusedActivation enum)
    ACTIVATION_NONE = 0
    ACTIVATION_SIGMOID = 1
    ACTIVATION_SILU = 2

    # ------------------------------------------------------------------
    # Setup APIs (routed-expert-specific)
    # ------------------------------------------------------------------

    @staticmethod
    def setup_matmul_expert_dram(
        mesh_device,
        cts_list,
        num_subblocks_k,
        subblock_n,
        cores_per_dram_bank,
        primary_at_last_offset,
        num_active_experts=8,
        num_total_experts=None,
        accum_experts=0,
        primary_worker_cores=None,
        k_parallel_per_bank=1,
    ):
        """Set up MatmulExpertCompressedDRAM infrastructure for one projection.

        Operates on a list of 256 uniform-bfp4_b ``CompressedTensor`` objects (one per
        expert, TP8-sharded) and produces the per-device meta + fmt tables required
        by ``MatmulExpertCompressedDRAM``.

        Args:
            mesh_device: TP8 mesh device.
            cts_list: list[CompressedTensor] of length ``num_total_experts``.
            num_subblocks_k: splits Kt into this many outer-K subblocks.
            num_active_experts: topk count (e.g. 8 for DeepSeek V3 MoE).
            num_total_experts: global expert count (e.g. 256).
            cores_per_dram_bank: compute cores sharing one DRAM bank. 1 for the MoE
                case (each DRAM streamer core owns a full bank).

        Returns:
            dict with:
              - ``per_core_n``, ``Kt``, ``subblock_k``, ``num_subblocks_k`` (matmul dims)
              - ``in1_backing_tensor`` (replicated L1 CB backing; buffer_address seeds cb_in1)
              - ``meta_tensors`` (dict[MeshCoordinate -> (offset_tensor, bs_tensor)];
                under lockstep refactor, the SAME mesh tensor pair for every coord)
              - ``fmt_dram_tensor`` (replicated DRAM fmt tensor)
              - ``fmt_dram_addr`` / ``fmt_per_expert_bytes`` / ``fmt_per_core_bytes``
              - ``expert_offsets_l1_addr_per_device`` (dict[MeshCoordinate -> list[(core, addr)]])
              - ``block_sizes_l1_addr_per_device`` (dict[MeshCoordinate -> list[(core, addr)]])
              - ``per_core_values_per_device`` (dict[MeshCoordinate -> {bank_id, vc, ...}])
              - ``num_active_experts``, ``num_total_experts``, ``cores_per_dram_bank``

        Note: does NOT allocate per-core index / table_idx / cb_fmt / pipeline_sem —
        those live at the MoE-kernel scope (shared across gate/up/down) and are owned
        by the caller. This helper also does not compute CBs or ReaderCTArgs values;
        the caller assembles the final CTArgs tuple from this dict plus the shared
        per-core addresses.
        """
        # Late-bind to avoid circular imports.
        from models.demos.deepseek_v3_b1.micro_ops.matmul_expert.op import (
            _meta_words_for_tiles,
            create_dram_expert_tensors_multi_device,
        )

        if num_total_experts is None:
            num_total_experts = len(cts_list)
        assert (
            len(cts_list) == num_total_experts
        ), f"cts_list length {len(cts_list)} != num_total_experts {num_total_experts}"

        # Derive matmul tiling from the CompressedTensor's per-device logical shape.
        # Note: ct.get_data_tensors()[0].memory_config().shard_spec.shape is the PACKED
        # byte layout ([1, shard_bytes]), not logical K×N — use ct._per_device_tiles_*.
        ct0 = cts_list[0]
        data0 = ct0.get_data_tensors()[0]
        weights_tile = data0.get_tile()
        Kt = ct0._per_device_tiles_h
        num_banks = mesh_device.dram_grid_size().x
        # per_core_n is the N-tile count PER CORE in the bank, which only depends on
        # n_parallel_per_bank (= cores_per_dram_bank when k_parallel=1; = cores/k_parallel
        # when K-split). For K-split with n_parallel=1, all cores in a bank share full N.
        assert cores_per_dram_bank % k_parallel_per_bank == 0, (
            f"cores_per_dram_bank ({cores_per_dram_bank}) must divide " f"k_parallel_per_bank ({k_parallel_per_bank})"
        )
        n_parallel_per_bank = cores_per_dram_bank // k_parallel_per_bank
        per_core_n = ct0._per_device_tiles_w // (num_banks * n_parallel_per_bank)
        assert ct0._per_device_tiles_w % (num_banks * n_parallel_per_bank) == 0, (
            f"per_device_tiles_w ({ct0._per_device_tiles_w}) must divide "
            f"num_banks*n_parallel ({num_banks * n_parallel_per_bank})"
        )
        assert Kt % num_subblocks_k == 0, f"Kt ({Kt}) must be divisible by num_subblocks_k ({num_subblocks_k})"
        subblock_k = Kt // num_subblocks_k
        assert per_core_n % subblock_n == 0, f"per_core_n ({per_core_n}) must be divisible by subblock_n ({subblock_n})"

        is_dram_flags = [1] * num_total_experts

        dram_results = create_dram_expert_tensors_multi_device(
            mesh_device,
            cts_list,
            subblock_k=subblock_k,
            num_subblocks_k=num_subblocks_k,
            per_core_N=per_core_n,
            n_parallel_per_bank=n_parallel_per_bank,
            k_parallel_per_bank=k_parallel_per_bank,
            num_total_experts=num_total_experts,
            is_dram_flags=is_dram_flags,
            subblock_n=subblock_n,
            primary_worker_cores=primary_worker_cores,
            primary_at_last_offset=primary_at_last_offset,
            # MoE overlays cb_in1 + cb_fmt onto the SDPA kv_buf (visible to CB
            # allocator). The helper's private in1_backing_tensor is invisible
            # to the CB allocator → adjacent CB placements can stomp it.
            allocate_in1_backing=False,
        )

        # dram_results layout (one entry per mesh coordinate) — 17-element tuple:
        #   (dram_backing_tensor, meta_tensors, fmt_dram_info, l1_addrs, per_core_values,
        #    num_in1_buffers, fmt_cb_l1_addr, fmt_sem_addr_0, fmt_sem_addr_1,
        #    fmt_sem_0, fmt_sem_1, partial_sem_addr, pipeline_sem_addr,
        #    partial_sem, pipeline_sem, gather_sync_sem_addr, gather_sync_sem)
        # dram_backing_tensor holds in1 + fmt regions fused; buffer_address() seeds cb_in1.
        # l1_addrs per coord = (expert_offsets_l1_addr_core_values, block_sizes_l1_addr_core_values).
        # Cross-device constants (fmt_cb_l1_addr, sem addrs) match across coords by construction.
        first_coord = next(iter(dram_results))
        (
            in1_backing_tensor,
            _,
            fmt_dram_info,
            _,
            _,
            num_in1_buffers,
            fmt_cb_l1_addr,
            fmt_sem_addr_0,
            fmt_sem_addr_1,
            fmt_sem_0,
            fmt_sem_1,
            partial_sem_addr,
            pipeline_sem_addr,
            partial_sem,
            pipeline_sem,
            gather_sync_sem_addr,
            gather_sync_sem,
        ) = dram_results[first_coord]
        fmt_dram_tensor = fmt_dram_info["fmt_dram_tensor"]

        meta_tensors_per_device = {c: dram_results[c][1] for c in dram_results}
        expert_offsets_l1_addr_per_device = {c: dram_results[c][3][0] for c in dram_results}
        block_sizes_l1_addr_per_device = {c: dram_results[c][3][1] for c in dram_results}
        per_core_values_per_device = {c: dram_results[c][4] for c in dram_results}

        # Per-device compute cores list (same list across devices; from first device's per_core_values).
        first_coord_for_cores = next(iter(per_core_values_per_device))
        compute_cores_list = [c for (c, _) in per_core_values_per_device[first_coord_for_cores]["bank_id"]]

        # Fmt DRAM sizing: recompute here (matches helper's internals).
        # K-split: num_subblocks_k_local = num_subblocks_k / k_parallel; each core's fmt
        # covers only its K-slice. With k_parallel=1, num_subblocks_k_local == num_subblocks_k.
        num_subblocks_k_local = num_subblocks_k // k_parallel_per_bank
        _DRAM_ALIGNMENT = ttnn._ttnn.bfp_utils.get_dram_alignment()
        tiles_per_block = subblock_k * subblock_n
        num_iterations_local = num_subblocks_k_local * (per_core_n // subblock_n)
        fmt_words_per_expert = num_iterations_local * _meta_words_for_tiles(tiles_per_block)
        fmt_bytes_per_expert_raw = fmt_words_per_expert * 4
        fmt_bytes_per_expert = ((fmt_bytes_per_expert_raw + _DRAM_ALIGNMENT - 1) // _DRAM_ALIGNMENT) * _DRAM_ALIGNMENT
        fmt_bytes_per_core = num_total_experts * fmt_bytes_per_expert
        cb_fmt_dram_page_size = fmt_bytes_per_expert  # already aligned
        dram_meta_words_per_block = _meta_words_for_tiles(tiles_per_block)

        # CT-args aliases / derived values (consumed by NCRISC + TRISC lists).
        # Must match create_dram_expert_tensors_multi_device's use of _TILE_SIZES[1]
        # (bfp4_b) so in1_region_bytes == helper's in1_region_bytes — otherwise cb_fmt
        # aliased at address_offset=in1_region_bytes overshoots the backing tensor.
        from models.demos.deepseek_v3_b1.micro_ops.dram_streaming_matmul_compressed.op import _TILE_SIZES

        max_tile_size = _TILE_SIZES[1]
        data0_tile = data0.get_tile()
        # cb_in1_dram sized as subblock_k × subblock_n × num_in1_buffers × max_tile_size.
        cb_in1_dram_total_bytes = num_in1_buffers * subblock_k * subblock_n * max_tile_size
        noc_max_page_size = 16384  # Blackhole NOC max burst

        # in0 page size (activation tile). MoE fused uses face tile [1, 32] bf16
        # for cb_in0 (gate_mm_input_cb, TD_1x32), NOT the weight tile. Using
        # data0_tile here was a bug — weights may report [32, 32] or packed shape,
        # giving a page size the compute kernel strides with incorrectly.
        _in0_tile_h, _in0_tile_w = 1, 32
        in0_page_size = _in0_tile_h * _in0_tile_w * 2  # bf16 activations, face tile

        # Legacy fields that _overlap_cbs_with_sdpa_buffer consumes (total size + tile
        # metadata to rebuild cb_in1 pointed at the SDPA backing buffer instead of the
        # in1_backing_tensor we just allocated in L1 locally). MUST include subblock_n
        # so the L1 CB carved by `_overlap_cbs_with_sdpa_buffer` matches the per-slot
        # bytes the streaming kernel actually writes (subblock_k × subblock_n × tile);
        # otherwise NCRISC overruns the CB and corrupts adjacent L1 → NCRISC hang.
        in1_block_size_bytes = subblock_k * subblock_n * max_tile_size
        in1_total_size = num_in1_buffers * in1_block_size_bytes
        assert in1_total_size == cb_in1_dram_total_bytes, (
            f"in1_total_size ({in1_total_size}) must equal cb_in1_dram_total_bytes "
            f"({cb_in1_dram_total_bytes}); both must size the same CB."
        )

        return {
            "per_core_n": per_core_n,
            "Kt": Kt,
            "num_tiles_k": Kt,
            "subblock_k": subblock_k,
            "subblock_n": subblock_n,
            "num_subblocks_k": num_subblocks_k,
            "k_parallel_per_bank": k_parallel_per_bank,
            "num_subblocks_k_local": num_subblocks_k_local,
            "in1_backing_tensor": in1_backing_tensor,
            "fmt_dram_tensor": fmt_dram_tensor,
            "fmt_dram_addr": _fused_base_addr(fmt_dram_tensor),
            "fmt_per_expert_bytes": fmt_bytes_per_expert,
            "fmt_per_core_bytes": fmt_bytes_per_core,
            "fmt_cb_l1_addr": fmt_cb_l1_addr,
            "fmt_cb_page_size": cb_fmt_dram_page_size,
            "fmt_sem_addr_0": fmt_sem_addr_0,
            "fmt_sem_addr_1": fmt_sem_addr_1,
            "partial_sem_addr": partial_sem_addr,
            "pipeline_sem_addr": pipeline_sem_addr,
            # down_proj uses primary_at_last_offset=1 (2-core bank → primary receives the
            # full bank N); gate/up keep 0 and the kernel skips all gather logic.
            "primary_at_last_offset": 1 if primary_at_last_offset else 0,
            "gather_sync_sem_addr": gather_sync_sem_addr,
            "dram_meta_words_per_block": dram_meta_words_per_block,
            "in0_page_size": in0_page_size,
            # Keep sem objects alive so their L1 allocations aren't reclaimed before
            # kernel launch (their addresses are baked into CT args).
            "_fmt_sem_0": fmt_sem_0,
            "_fmt_sem_1": fmt_sem_1,
            "_partial_sem": partial_sem,
            "_pipeline_sem": pipeline_sem,
            "_gather_sync_sem": gather_sync_sem,
            "meta_tensors_per_device": meta_tensors_per_device,
            "expert_offsets_l1_addr_per_device": expert_offsets_l1_addr_per_device,
            "block_sizes_l1_addr_per_device": block_sizes_l1_addr_per_device,
            "per_core_values_per_device": per_core_values_per_device,
            "compute_cores_list": compute_cores_list,
            "num_active_experts": num_active_experts,
            "num_total_experts": num_total_experts,
            "cores_per_dram_bank": cores_per_dram_bank,
            "cores_per_bank": cores_per_dram_bank,
            "cb_in1_size_bytes": cb_in1_dram_total_bytes,
            "noc_max_page_size": noc_max_page_size,
            "accum_experts": accum_experts,
            # Filled in by caller after routing tensor is known (stub 0 until then).
            "index_l1_addr": 0,
            # Legacy compatibility for _overlap_cbs_with_sdpa_buffer.
            "in1_total_size": in1_total_size,
            "weights_tile": data0_tile,
            "weights_dtype": data0.dtype,
            # Compressed storage is bfp4_b (576 B / 32x32 tile). cb_in1 must match
            # canonical matmul_expert's CB format (micro_ops/matmul_expert/op.py:402-407)
            # — bfp8_b here would cause compute to unpack 1088-B tiles over 576-B pages.
            "cb_page_size": max_tile_size,
            "cb_data_format": ttnn.bfloat4_b,
            "in1_buf_addr": 0,
            # Placeholder descriptors — populated by _overlap_cbs_with_sdpa_buffer.
            "cb_in1_descriptor": None,
            "cb_out_descriptor": None,
        }

    @staticmethod
    def setup_matmul_expert_sram(
        mesh_device,
        sram_cts_list,
        core_grid,
        num_tiles_k,
        per_core_n,
        Kt,
        num_active_experts=8,
        accum_experts=False,
    ):
        """Set up MatmulExpertCompressedSRAM infrastructure for one projection.

        Mirrors ``_build_sram_fmt_data`` from the canonical
        ``test_matmul_expert.py`` (per_core_allocation). The caller is
        responsible for allocating ``sram_cts_list`` as L1-sharded
        CompressedTensors on ``core_grid`` — see
        ``_build_sram_cts_standard`` / ``_build_sram_cts_slice_k`` in the
        canonical test for reference layouts.

        Args:
            mesh_device: TP8 mesh device.
            sram_cts_list: list[CompressedTensor] of length T (hot experts).
                Already L1-allocated on ``core_grid``.
            core_grid: CoreRangeSet hosting the SRAM matmul (e.g. 64 shared
                gate cores).
            num_tiles_k: K tiles per core per expert (this core's K-slice
                length when K-sliced; full K otherwise).
            per_core_n: N tiles per core per expert.
            Kt: total K tiles per expert (full K). When ``num_tiles_k < Kt``,
                K-slicing is active and per-core ``sram_k_offset`` values are
                computed.
            num_active_experts: TopK count threaded into kernel CT args.
            accum_experts: False for gate/up (per-expert outputs), True for
                down (cross-expert accumulation).

        Returns:
            dict with kernel CT-arg-ready fields:
              - ``num_sram_experts`` (= len(sram_cts_list))
              - ``num_tiles_k``, ``out_w``, ``in0_page_size``,
                ``meta_words_per_expert``, ``accum_experts``
              - ``sram_fmt_tensors`` / ``sram_base_addr_tensors`` (kept-alive
                tensor dicts: {coord: {core_idx: tensor}})
              - ``sram_fmt_l1_addr_per_device`` /
                ``sram_base_addrs_l1_addr_per_device`` (per-device per-core
                L1 byte addresses)
              - ``sram_k_offsets`` (per-core K-slice offset values, None when
                no K-slicing)
              - ``cb_in1_size_bytes`` (per-core SRAM weight region size)
              - ``cb_data_format``, ``cb_page_size`` (bfp4_b)
              - ``num_active_experts``
              - ``_sram_cts_list`` (kept alive)
        """
        from models.demos.deepseek_v3_b1.micro_ops.matmul_expert.op import (
            _meta_words_for_tiles,
            create_expert_fmt_tensors,
        )

        num_sram_experts = len(sram_cts_list)
        if num_sram_experts == 0:
            return {
                "num_sram_experts": 0,
                "num_tiles_k": num_tiles_k,
                "out_w": per_core_n,
                "num_active_experts": num_active_experts,
                "accum_experts": 1 if accum_experts else 0,
            }

        # Build fmt + base_addrs tables as TWO mesh tensors (one each), each
        # HEIGHT_SHARDED on core_grid + ShardTensor2dMesh across the mesh +
        # lockstep-allocated.  Lockstep gives a uniform L1 base across the
        # mesh, so per-core L1 addresses are uniform across devices (the
        # CONTENT varies per device via ShardTensor2dMesh; only the ADDRESS is
        # uniform).  Replaces the prior pattern of 130 single-core per_core
        # allocated tensors per projection per device, which (a) exhausted
        # the per-core allocator slot table and (b) produced divergent
        # per-(device, core) addresses under BSPM-SRAM L1 frontier
        # divergence.
        sram_fmt_tensor, sram_base_addr_tensor, sram_fmt_l1_addrs, sram_base_addrs_l1_addrs = create_expert_fmt_tensors(
            sram_cts_list, mesh_device, core_grid, num_tiles_k, per_core_n
        )

        cores = ttnn.corerange_to_cores(core_grid)

        # Per-core K-offset values for K-sliced layouts. None when full K per
        # core (no K-slice). Layout matches HEIGHT_SHARDED K-slicing where
        # consecutive cores own consecutive K-slices in row-major (K-major)
        # order with n_parallel = total_cores * num_tiles_k / Kt.
        sram_k_offsets = None
        if num_tiles_k < Kt:
            n_parallel = len(cores) * num_tiles_k // Kt
            sram_k_offsets = [(cores[i], (i // n_parallel) * num_tiles_k) for i in range(len(cores))]

        # Per-device dicts now hold IDENTICAL per-core lists for every coord
        # (lockstep → uniform addresses across mesh).  Kept as dicts to
        # preserve the existing API consumed by _build_compile_time_args /
        # _setup_per_device_args without churn.
        mesh_shape = mesh_device.shape
        _fmt_addr_tuples = [(cores[i], sram_fmt_l1_addrs[i]) for i in range(len(cores))]
        _base_addr_tuples = [(cores[i], sram_base_addrs_l1_addrs[i]) for i in range(len(cores))]
        sram_fmt_l1_addr_per_device = {
            ttnn.MeshCoordinate(r, c): _fmt_addr_tuples
            for r in range(int(mesh_shape[0]))
            for c in range(int(mesh_shape[1]))
        }
        sram_base_addrs_l1_addr_per_device = {
            ttnn.MeshCoordinate(r, c): _base_addr_tuples
            for r in range(int(mesh_shape[0]))
            for c in range(int(mesh_shape[1]))
        }

        # Per-tile fmt metadata word count. Matches DRAM path's
        # _meta_words_for_tiles convention used by the canonical fmt encoder.
        meta_words_per_expert = _meta_words_for_tiles(num_tiles_k * per_core_n)

        # cb_in1 holds T expert slabs back-to-back. Each slab is
        # K_per_core × N_per_core × bfp4_b tile size.
        ct0 = sram_cts_list[0]
        data0 = ct0.get_data_tensors()[0]
        weights_tile = data0.get_tile()
        from models.demos.deepseek_v3_b1.micro_ops.dram_streaming_matmul_compressed.op import _TILE_SIZES

        bfp4_tile_size = _TILE_SIZES[1]  # bfp4_b 32×32 = 576 B
        cb_in1_size_bytes = num_sram_experts * num_tiles_k * per_core_n * bfp4_tile_size

        # Activation page size: MoE uses 1×32 bf16 face tile, NOT the weight
        # tile. Mirrors setup_matmul_expert_dram's convention (op.py:644-645).
        in0_tile_h, in0_tile_w = 1, 32
        in0_page_size = in0_tile_h * in0_tile_w * 2  # bf16

        return {
            "num_sram_experts": num_sram_experts,
            "num_tiles_k": num_tiles_k,
            "out_w": per_core_n,
            "in0_page_size": in0_page_size,
            "meta_words_per_expert": meta_words_per_expert,
            "accum_experts": 1 if accum_experts else 0,
            "num_active_experts": num_active_experts,
            # Single mesh tensors (one each), kept alive in the params dict so
            # they aren't garbage-collected during MoeOp's lifetime.
            "sram_fmt_tensors": sram_fmt_tensor,
            "sram_base_addr_tensors": sram_base_addr_tensor,
            "sram_fmt_l1_addr_per_device": sram_fmt_l1_addr_per_device,
            "sram_base_addrs_l1_addr_per_device": sram_base_addrs_l1_addr_per_device,
            "sram_k_offsets": sram_k_offsets,
            "cb_in1_size_bytes": cb_in1_size_bytes,
            "cb_data_format": ttnn.bfloat4_b,
            "cb_page_size": bfp4_tile_size,
            "weights_tile": weights_tile,
            "weights_dtype": data0.dtype,
            "_sram_cts_list": sram_cts_list,
            "core_grid": core_grid,
        }

    @staticmethod
    def setup_eltwise_mul(
        cb_in0_index,
        cb_in1_index,
        cb_out_index,
        per_core_n,
        cb_scalar_index,
        cb_scalar_src_index,
    ):
        """
        Set up parameters for element-wise multiply with CB aliasing and scalar multiply.
        CB descriptors for in0/in1/out/scalar are wired by the SDPA overlap function.
        """
        # Inputs (matmul outputs) are 1x32 tiles; scalar broadcast requires 1x32,
        # so the mul runs as per_core_n × 1x32 mul_tiles per expert.
        mul_num_tiles = per_core_n

        return {
            "mul_num_tiles": mul_num_tiles,
            "cb_in0_descriptor": None,
            "cb_in1_descriptor": None,
            "cb_out_descriptor": None,
            "cb_scalar_index": cb_scalar_index,
            "cb_scalar_src_index": cb_scalar_src_index,
            "cb_scalar_descriptor": None,
            "cb_scalar_src_descriptor": None,
        }

    @staticmethod
    def setup_eltwise_add(
        cb_in0_index,
        cb_in1_index,
        cb_out_index,
        width_per_core,
        total_width,
        core_ranges,
        out_tensor=None,
    ):
        """
        Set up parameters and CB descriptors for element-wise add with per-core indexing.

        Used after down_proj to add fused_add tensor. Each core uses sender_index to
        offset into the replicated in1 tensor. cb_in0/cb_in1 descriptors are wired
        by the SDPA overlap function. ``out_tensor`` is provided when the add output
        is the final (non-reduce) tensor; otherwise CB out is also wired by overlap.
        """
        compute_cores_list = ttnn.corerange_to_cores(core_ranges, row_wise=True)

        element_size_bytes = 2  # bfloat16
        slice_size_bytes = width_per_core * element_size_bytes

        cb_tile_h = 32
        cb_tile_w = 32
        cb_tile_size_bytes = cb_tile_h * cb_tile_w * element_size_bytes
        cb_tile = ttnn.Tile([cb_tile_h, cb_tile_w])
        cb_tile_desc = ttnn.TileDescriptor(cb_tile)

        in0_size_bytes = slice_size_bytes
        in1_size_bytes = total_width * element_size_bytes

        in0_wait_tiles = 1
        in1_wait_tiles = in1_size_bytes // in0_size_bytes

        num_tiles = 1  # Single 32x32 CB view tile

        cb_out_descriptor = None
        if out_tensor is not None:
            cb_out_descriptor = ttnn.cb_descriptor_from_sharded_tensor(cb_out_index, out_tensor)
            cb_out_descriptor.total_size = num_tiles * cb_tile_size_bytes
            cb_out_descriptor.format_descriptors[0].tile = cb_tile_desc
            cb_out_descriptor.format_descriptors[0].page_size = cb_tile_size_bytes

        sender_index_core_values = [(core, idx) for idx, core in enumerate(compute_cores_list)]

        return {
            "cb_in0": cb_in0_index,
            "cb_in1": cb_in1_index,
            "cb_out": cb_out_index,
            "num_tiles": num_tiles,
            "slice_size_bytes": slice_size_bytes,
            "cb_in0_wait_tiles": in0_wait_tiles,
            "cb_in1_wait_tiles": in1_wait_tiles,
            "cb_in0_descriptor": None,
            "cb_in1_descriptor": None,
            "cb_out_descriptor": cb_out_descriptor,
            "sender_index_core_values": sender_index_core_values,
        }

    @staticmethod
    def setup_gate(
        input_cb,
        bias_cb,
        indices_cb,
        output_cb,
        output_indices_cb,
        bias_tensor,
        indices_tensor,
        output_scores_tensor,
        output_indices_tensor,
        input_tensor=None,
        eps=1e-20,
        scaling_factor=2.5,
        enable_sigmoid=False,
    ):
        """
        Set up parameters for the MoE gate operation.

        The gate computes top-K expert selection with normalized scores.

        Args:
            input_cb: Input CB index (receives gathered matmul output)
            bias_cb: Bias CB index
            indices_cb: Indices CB index
            output_cb: Output scores CB index
            output_indices_cb: Output indices CB index
            bias_tensor: Bias tensor
            indices_tensor: Indices tensor
            output_scores_tensor: Output scores tensor
            output_indices_tensor: Output indices tensor
            input_tensor: Input tensor (optional; CB descriptor overridden by SDPA overlap)
            eps: Epsilon for numerical stability (default 1e-20)
            scaling_factor: Scaling factor for gate scores (default 2.5)
            enable_sigmoid: Whether to apply sigmoid (default False, already done in matmul)

        Returns:
            Dictionary with gate parameters and CB descriptors
        """
        import struct

        eps_uint32 = int.from_bytes(struct.pack("f", eps), byteorder="little")
        scaling_factor_uint32 = int.from_bytes(struct.pack("f", scaling_factor), byteorder="little")

        input_cb_descriptor = None
        if input_tensor is not None:
            input_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(input_cb, input_tensor)
        bias_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(bias_cb, bias_tensor)
        indices_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(indices_cb, indices_tensor)
        output_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(output_cb, output_scores_tensor)
        output_indices_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(output_indices_cb, output_indices_tensor)

        return {
            "input_cb": input_cb,
            "bias_cb": bias_cb,
            "indices_cb": indices_cb,
            "output_cb": output_cb,
            "output_indices_cb": output_indices_cb,
            "eps": eps_uint32,
            "scaling_factor": scaling_factor_uint32,
            "enable_sigmoid": 1 if enable_sigmoid else 0,
            "input_cb_descriptor": input_cb_descriptor,
            "bias_cb_descriptor": bias_cb_descriptor,
            "indices_cb_descriptor": indices_cb_descriptor,
            "output_cb_descriptor": output_cb_descriptor,
            "output_indices_cb_descriptor": output_indices_cb_descriptor,
        }

    @staticmethod
    def golden(
        input_tensor,
        gate_proj_weights_dict=None,
        up_proj_weights_dict=None,
        down_proj_weights_dict=None,
        fused_add_tensor=None,
        enable_routing=True,
        # Routing-only params (ignored when enable_routing=False)
        routing_weights_tensor=None,
        bias_tensor=None,
        eps=1e-20,
        scaling_factor=2.5,
        use_hardcoded_expert_index=False,
        hardcoded_expert_index=0,
        explicit_expert_scale=None,
    ):
        """
        PyTorch reference implementation for validation.

        When enable_routing=False, uses expert 0 with no routing and no expert scale.

        Args:
            input_tensor: [1, K] torch.Tensor
            gate_proj_weights_dict: Dict[int, Tensor] expert_idx → [1,1,K,N_expert]
            up_proj_weights_dict: Dict[int, Tensor] expert_idx → [1,1,K,N_expert]
            down_proj_weights_dict: Dict[int, Tensor] expert_idx → [1,1,N_expert,K]
            fused_add_tensor: [1,1,1,K] torch.Tensor (optional)
            enable_routing: If True, run routing. If False, use expert 0, no scale.
            routing_weights_tensor: [K, N_routing] (routing only)
            bias_tensor: [1, 8, 32] (routing only)
            eps, scaling_factor, use_hardcoded_expert_index, hardcoded_expert_index,
            explicit_expert_scale: routed expert gate params (routing only)

        Returns:
            When enable_routing=True: (top8_scores, top8_indices, final_output)
            When enable_routing=False: (None, None, final_output)
        """
        import torch

        top8_scores = None
        top8_indices = None

        if enable_routing:
            from models.demos.deepseek_v3_b1.micro_ops.deepseek_moe_gate.op import DeepseekMoeGateSingleCore

            # 1. Routing matmul + sigmoid (truncate to bfloat16 to approximate device accumulation)
            logits = (input_tensor.bfloat16().float() @ routing_weights_tensor.bfloat16().float()).bfloat16().float()
            scores = torch.sigmoid(logits)

            # 2. Gate: top-8 selection with normalized scores
            gate_input = scores.reshape(1, 8, 32)
            top8_scores, top8_indices = DeepseekMoeGateSingleCore.golden(
                gate_input, bias_tensor.float(), eps, scaling_factor, enable_sigmoid=False
            )

        # 3. Expert matmuls (if expert weights provided)
        if gate_proj_weights_dict is not None:
            if not enable_routing:
                selected_expert_idx = 0
            elif use_hardcoded_expert_index:
                selected_expert_idx = hardcoded_expert_index
            else:
                selected_expert_idx = int(top8_indices[0, 0].item())

            # gate_proj: input @ weights + SiLU
            gate_proj_weights = gate_proj_weights_dict[selected_expert_idx]
            input_for_expert = input_tensor.reshape(1, 1, 1, -1).float()
            gate_proj_output = input_for_expert @ gate_proj_weights.float()
            gate_proj_output = torch.nn.functional.silu(gate_proj_output)

            # up_proj: input @ weights (no activation)
            up_proj_weights = up_proj_weights_dict[selected_expert_idx]
            up_proj_output = input_for_expert @ up_proj_weights.float()

            # Expert scale (1.0 when routing disabled)
            if not enable_routing:
                expert_scale = 1.0
            elif explicit_expert_scale is not None:
                expert_scale = explicit_expert_scale
            else:
                expert_scale = top8_scores[0, 0].float()

            # Fused output: silu(gate_proj) * up_proj * expert_scale
            fused_output = gate_proj_output * up_proj_output * expert_scale

            # down_proj
            if down_proj_weights_dict is not None:
                down_proj_weights = down_proj_weights_dict[selected_expert_idx]
                down_proj_output = fused_output @ down_proj_weights.float()

                if fused_add_tensor is not None:
                    final_output = down_proj_output + fused_add_tensor.float()
                    return top8_scores, top8_indices, final_output
                return top8_scores, top8_indices, down_proj_output

            return top8_scores, top8_indices, fused_output

        return top8_scores, top8_indices, None

    # ========================================================================
    # Setup & build methods (SharedExpertOp pattern)
    # ========================================================================

    @staticmethod
    def _setup_dimensions(
        shared_residual_mcast_src_tensor,
        # Routing-only tensors (None when enable_routing=False)
        gate_mm_weights_tensor=None,
        gate_bias_tensor=None,
        gate_indices_tensor=None,
        gate_output_scores_tensor=None,
        gate_output_indices_tensor=None,
        # Expert weights (always required)
        gate_proj_weights_tensor=None,
        up_proj_weights_tensor=None,
        down_proj_weights_tensor=None,
        final_output_tensor=None,
        rmsnorm_gamma_tensor=None,
        epsilon=1e-6,
        enable_routing=True,
        use_hardcoded_expert_index=False,
        reduce_intermediate_tensors=None,
        reduce_output_tensor=None,
        reduce_semaphores=None,
        reduce_root_coord=None,
        # Broadcast parameters
        bcast_input_tensor=None,
        bcast_intermediate_tensor=None,
        bcast_semaphores=None,
        bcast_sender_coord=None,
        # Global semaphores (created by MoeOp.create_semaphores)
        semaphores=None,
        cb_id_context=None,
        # Optional worker-core grid override (used to avoid overlap with external micro-ops).
        worker_core_grid=None,
        # SRAM routed gate_proj weights (list of L1-resident CompressedTensors,
        # one per slot, in slot order matching create_gate_indices_tensor's
        # sram_expert_ids encoding). None or empty = no SRAM experts placed;
        # the kernel's SRAM Op call will be a no-op via sram_gate_proj_active=0.
        # Mirrors gate_proj_weights_tensor's DRAM contract — caller passes the
        # CTs, _setup_dimensions calls setup_matmul_expert_sram internally.
        sram_gate_proj_weights_tensor=None,
        # SRAM routed up_proj weights — same contract as gate_proj. Runs on the
        # shared up compute cores (mirror of shared gate cores). None = no SRAM
        # up_proj; kernel skips via sram_up_proj_active=0.
        sram_up_proj_weights_tensor=None,
        # SRAM routed down_proj weights — runs on the 112 shared mcast receiver
        # cores (= shared_down_proj's matmul cores). Per-expert weight layout
        # mirrors shared_down: per-device shape (K_per_dev=256, N_full=7168)
        # width-sharded on 112 cores → (256, 64) per core. accum_experts=True so
        # the kernel sums across SRAM-flagged TopK winners. None = skipped.
        sram_down_proj_weights_tensor=None,
        # When True, SRAM expert weights carry per-tile BSPM mixed precision and
        # the kernel must use the compressed_custom_mm path (use_compression=1).
        # When False (default), SRAM is uniform BFP4 and the plain custom_mm
        # fast path is used (use_compression=0).
        enable_sram_bspm=False,
    ):
        """Compute all dimensions, grids, setup params, CB descriptors, and per-core values.

        Non-weight tensors (gate_input, mcast_output, expert_index/scale, down_proj_gather_output,
        down_proj_output, fused_add) are no longer needed — their configuration info is derived
        from device properties and weight tensor shapes. Their CB descriptors are overridden by
        _overlap_cbs_with_sdpa_buffer in the production path.
        """
        assert isinstance(gate_proj_weights_tensor, list), (
            "gate_proj_weights_tensor must be a list[CompressedTensor] (TP8). Dense MLPs use "
            "prepare_dense_routed_experts_compressed_tp8; MoE layers use compressed_tp8=True."
        )
        assert isinstance(up_proj_weights_tensor, list), "up_proj_weights_tensor must be a list[CompressedTensor]"
        assert isinstance(down_proj_weights_tensor, list), "down_proj_weights_tensor must be a list[CompressedTensor]"

        # ==================================================================
        # Extract global semaphore addresses
        # ==================================================================
        enable_bcast = (
            bcast_input_tensor is not None
            and bcast_intermediate_tensor is not None
            and bcast_semaphores is not None
            and bcast_sender_coord is not None
        )
        assert semaphores is not None, "semaphores must be provided (use MoeOp.create_semaphores())"
        sem_addrs = [ttnn.get_global_semaphore_address(s) for s in semaphores]
        mcast_data_sender_semaphore_addr = sem_addrs[MoeSem.MCAST_SENDER]
        mcast_data_receiver_semaphore_addr = sem_addrs[MoeSem.MCAST_DATA_RECEIVER]
        gather_noc0_receiver_semaphore_addr = sem_addrs[MoeSem.DOWN_PROJ_GATHER]
        gather_noc1_receiver_semaphore_addr = sem_addrs[MoeSem.DOWN_PROJ_GATHER]
        residual_mcast_receiver_semaphore_addr = sem_addrs[MoeSem.RESIDUAL_MCAST_RECEIVER]
        expert_scale_mcast_receiver_semaphore_addr = sem_addrs[MoeSem.EXPERT_SCALE_MCAST_RECEIVER]
        index_mcast_receiver_semaphore_addr = sem_addrs[MoeSem.INDEX_MCAST_RECEIVER]
        down_proj_mcast_receiver_semaphore_addr = sem_addrs[MoeSem.DOWN_PROJ_MCAST_RECEIVER]
        sram_down_mcast_receiver_semaphore_addr = sem_addrs[MoeSem.SRAM_DOWN_MCAST_RECEIVER]
        scan_sync_sem_addr = sem_addrs[MoeSem.SCAN_SYNC]

        # ==================================================================
        # Derive config from shared_residual_mcast_src_tensor (the actual input activation)
        # ==================================================================
        data_format = shared_residual_mcast_src_tensor.dtype
        TILE_1x32 = ttnn.Tile((1, 32))
        tile_1x32_size = TILE_1x32.get_tile_size(data_format)
        K = shared_residual_mcast_src_tensor.shape[1]
        num_tiles_k = K // TILE_1x32.tile_shape[1]

        input_core_grid = shared_residual_mcast_src_tensor.memory_config().shard_spec.grid
        sender_core = list(input_core_grid.ranges())[0].start

        mesh_device = shared_residual_mcast_src_tensor.device()
        device_grid_size = mesh_device.compute_with_storage_grid_size()
        mesh_shape = mesh_device.shape
        mesh_rows, mesh_cols = mesh_shape[0], mesh_shape[1]
        device = ttnn.get_device_tensors(shared_residual_mcast_src_tensor)[0].device()

        gate_proj_worker_cores = get_pinned_optimal_dram_bank_to_logical_worker_assignment(device, ttnn.NOC.NOC_0)
        gate_proj_core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in gate_proj_worker_cores])
        mcast_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), sender_core)])
        sender_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(sender_core, sender_core)])
        mcast_worker_grid = mcast_grid.subtract(sender_core_grid)
        num_gate_proj_cores = gate_proj_core_ranges.num_cores()

        # Worker core grid: default to full device grid unless caller overrides.
        full_device_grid = worker_core_grid
        if full_device_grid is None:
            full_device_grid = ttnn.CoreRangeSet(
                [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))]
            )

        expert_scale_mcast_sender_semaphore_addr = mcast_data_sender_semaphore_addr  # Reuse sender semaphore

        # ==================================================================
        # TileDescriptors for CB allocation
        # ==================================================================
        TD_1x16 = ttnn.TileDescriptor(ttnn.Tile((1, 16)))
        TD_1x32 = ttnn.TileDescriptor(TILE_1x32)
        TD_16x16 = ttnn.TileDescriptor(ttnn.Tile((FACE_HEIGHT, FACE_WIDTH)))
        TD_32x32 = ttnn.TileDescriptor(ttnn.Tile((32, 32)))

        # ==================================================================
        # CB indices (auto-assigned via cb_id_context)
        # ==================================================================
        assert cb_id_context is not None, "cb_id_context must be provided"

        # 1x32, bfloat16
        gate_mm_input_cb = cb_id_context.get_cb_id(data_format, TD_1x32)
        gate_proj_cb_out = cb_id_context.get_cb_id(data_format, TD_1x32)
        up_proj_cb_mm_out = cb_id_context.get_cb_id(data_format, TD_1x32)
        down_proj_gather_dst_cb = cb_id_context.get_cb_id(data_format, TD_1x32)
        down_proj_mcast_dst_cb = cb_id_context.get_cb_id(data_format, TD_1x32)
        down_proj_cb_out = cb_id_context.get_cb_id(data_format, TD_1x32)
        residual_mcast_dst_cb = cb_id_context.get_cb_id(data_format, TD_1x32)
        # down_proj_cb_internal_acc: kept as a CT arg slot for kernel API compatibility,
        # but now aliased to cb_out (same CB ID). The kernel writes per-expert pack_tile
        # directly to cb_out's L1 via the override-pinned wr_ptr, with NO per-expert
        # cb_push_back — so cb_out's fifo metadata stays at 0 until the single
        # consumer-facing push at end of all experts. Eltwise_add never observes
        # transient state. This saves 1 CB ID slot vs an independent allocation.
        down_proj_cb_internal_acc = down_proj_cb_out

        # EltwiseMul uses 1x32 CBs directly (no 16x16 alias needed).
        # mul_cb_in0/in1 reuse the existing up_proj/gate_proj output CBs so their
        # read pointers advance correctly on each cb_pop_front.
        mul_cb_in0 = up_proj_cb_mm_out  # CB12, 1x32 — reuse, no separate alias
        mul_cb_in1 = gate_proj_cb_out  # CB11, 1x32 — reuse, no separate alias
        mul_cb_out = cb_id_context.get_cb_id(data_format, TD_1x32)  # 1x32 output

        # 32x32, bfloat16
        add_cb_in0 = cb_id_context.get_cb_id(data_format, TD_32x32)
        add_cb_in1 = cb_id_context.get_cb_id(data_format, TD_32x32)
        add_cb_out = cb_id_context.get_cb_id(data_format, TD_32x32)

        # Tensor-backed CBs (format from weight tensors). Probe the first CT's data tensor.
        gate_proj_weights_probe = gate_proj_weights_tensor[0].get_data_tensors()[0]
        down_proj_weights_probe = down_proj_weights_tensor[0].get_data_tensors()[0]
        gate_proj_cb_in1 = cb_id_context.get_cb_id(
            gate_proj_weights_probe.dtype, ttnn.TileDescriptor(gate_proj_weights_probe.get_tile())
        )
        up_proj_cb_in1 = gate_proj_cb_in1  # intentional alias: sequential matmuls share CB slot
        down_proj_cb_in1 = cb_id_context.get_cb_id(
            down_proj_weights_probe.dtype, ttnn.TileDescriptor(down_proj_weights_probe.get_tile())
        )
        # MatmulExpertCompressedDRAM input activation CBs.
        # gate_proj/up_proj in0 = rmsnorm-mcast destination; down_proj in0 = down_mcast destination.
        gate_proj_cb_in0 = gate_mm_input_cb
        up_proj_cb_in0 = gate_mm_input_cb
        down_proj_cb_in0 = down_proj_mcast_dst_cb
        # MatmulExpertCompressedDRAM fmt-metadata CBs (DRAM-streamed, uint32 tiles) — one per proj.
        gate_proj_cb_fmt = cb_id_context.get_cb_id(data_format, TD_1x32)
        up_proj_cb_fmt = cb_id_context.get_cb_id(data_format, TD_1x32)
        down_proj_cb_fmt = cb_id_context.get_cb_id(data_format, TD_1x32)
        # cb_out_silu: aliases gate_proj_cb_out's L1 region with a tall [8, 32] tile so
        # the K-split silu fast-path's copy_tile + silu + pack_tile covers all 8 expert
        # outputs in one tile op. silu_tile_h = pad_to_face_r_dim(num_active_experts *
        # per_core_n * 1) = pad(8 * 1 * 1) = 8 for the MoE gate K-split config. Active
        # when k_parallel_per_bank>1 — kernel's constexpr branch eliminates otherwise.
        TD_8x32 = ttnn.TileDescriptor(ttnn.Tile((8, 32)))
        gate_proj_cb_out_silu = cb_id_context.get_cb_id(data_format, TD_8x32)

        # SRAM routed gate_proj CBs — always allocated. cb_in1 holds T expert
        # slabs (bfp4_b 32×32 tile); cb_out is per-expert per-N-tile output
        # (bf16 1×32 face tile, mirrors gate_proj_cb_out). Default unused
        # (T=0 → kernel treats them as a no-op via sram_gate_proj_active=0).
        TD_32x32_bfp4 = ttnn.TileDescriptor(ttnn.Tile((32, 32)))
        sram_gate_proj_cb_in1 = cb_id_context.get_cb_id(ttnn.bfloat4_b, TD_32x32_bfp4)
        # sram_gate_proj_out_cb is assigned later (after gate_mm_output_cb is
        # set in the routing block) so it can join ID-B's 3-grid disjoint set
        # {gate_mm, sender, a_cores}. The a_cores ⊃ streamer overlap rules out
        # aliasing with gate_proj_cb_out directly.
        # SRAM up_proj weight CB shares an ID with sram_gate_proj_cb_in1
        # (a_cores ⊥ b_cores; same bfp4 32×32 format). Saves 1 slot in (bfp4, 32×32).
        sram_up_proj_cb_in1 = sram_gate_proj_cb_in1
        # SRAM up_proj output CB shares with up_proj_cb_mm_out (DRAM up output).
        # b_cores ⊥ streamer (streamer ⊂ a_cores). Saves 1 slot in (bf16, 1×32).
        sram_up_proj_out_cb = up_proj_cb_mm_out
        # SRAM gather destination CBs (sender core only). Sized for the gather
        # output: num_active_experts × 64 cores = 512 tiles per device.
        # Allocate with face-tile descriptor (16x16) so the dummy CB descriptor
        # built for reconfig=true matches the runtime tile shape used by GR.
        # The L1 layout is the same either way; this only affects the tile-shape
        # the kernel compiles against (mirrors shared_group1/2 — see line 4015).
        # 8×32 tile (bf16, 512 B) — same byte count as a 16×16 face but matches
        # the 8-K-slice × 32-element gather layout exactly (each gather row =
        # one K-partial's 32 elements). Lets these CBs auto-reuse attention's
        # 8×32 IDs (SDPA outputs) via cross-context manager reuse, avoiding the
        # otherwise sender-only-bound 16×16 bucket.
        sram_group1_cb = cb_id_context.get_cb_id(data_format, TD_8x32)
        sram_group2_cb = cb_id_context.get_cb_id(data_format, TD_8x32)
        # SRAM extended GatedReduce intermediate + output CBs (sender core only).
        # 8×32 tile (= 8 K-partials × 32 N-elements; same 512 B as a face).
        sram_intermed_cb = cb_id_context.get_cb_id(data_format, TD_8x32)
        sram_mcast_src_cb = cb_id_context.get_cb_id(data_format, TD_8x32)
        # SRAM extended GatedReduce scalar CB (sender_core only). 8×32 tile
        # for shape consistency with intermed/out CBs (FPU mul-bcast-scalar).
        # BRISC writes scalar at [0,0]; TRISC reads via mul_tiles_bcast_scalar
        # (BroadcastType::SCALAR uses [0,0] only — rest of tile unused).
        sram_gr_scalar_cb = cb_id_context.get_cb_id(data_format, TD_8x32)
        # Dedicated SRAM down mcast destination CB on the 112 shared mcast
        # receiver cores (kv_buf-overlaid like the DRAM dst CB). Separate from
        # down_proj_mcast_dst_cb to avoid layout collisions — DRAM mcast keeps
        # using its own CB at base, SRAM mcast lands here at base.
        sram_down_mcast_dst_cb = cb_id_context.get_cb_id(data_format, TD_1x32)
        # SRAM routed down_proj CBs — runs on the 112 shared mcast receiver cores.
        # cb_in1: T expert weight slabs (bfp4_b 32×32 tile), per-core L1.
        # cb_out: 1×32 bf16, per_core_n=2 tiles per core (mirrors shared_down_matmul_out).
        sram_down_proj_cb_in1 = cb_id_context.get_cb_id(ttnn.bfloat4_b, TD_32x32_bfp4)
        sram_down_proj_out_cb = cb_id_context.get_cb_id(data_format, TD_1x32)
        # Merged down output CB — feeds residual_add, holds either
        #   shared_down + sram_down  (n_sram_active > 0)  or
        #   shared_down              (n_sram_active == 0, copy path)
        # Replaces shared_down_matmul_out_cb as residual_add's in0.
        merged_down_out_cb = cb_id_context.get_cb_id(data_format, TD_1x32)

        # Routing-only CB indices (0 when routing disabled)
        if enable_routing:
            # ID-B: 3-grid disjoint alias chain on
            #   {sender, gate_mm cores (col 12 rows 0-7), a_cores}
            # — sender ⊥ gate_mm ⊥ a_cores (gate_mm at col 12; a_cores at cols
            # 0-3,7-9; sender at (12,9)). Each kernel uses ID-X with its core's
            # L1 layout via separate cb_descriptors. Saves 2 slots in (bf16, 1×32).
            gate_mm_output_cb = down_proj_gather_dst_cb
            sram_gate_proj_out_cb = gate_mm_output_cb
            gate_input_cb = cb_id_context.get_cb_id(data_format, TD_16x16)
            gate_proj_cb_index = cb_id_context.get_cb_id(data_format, TD_1x16)
            mul_cb_scalar_src = cb_id_context.get_cb_id(data_format, TD_1x16)
            gate_output_cb = cb_id_context.get_cb_id(data_format, TD_1x16)
            gate_output_indices_cb = cb_id_context.get_cb_id(ttnn.uint16, TD_1x16)
            mul_cb_scalar = cb_id_context.get_cb_id(data_format, TD_1x32)
            gate_bias_cb = cb_id_context.get_cb_id(
                gate_bias_tensor.dtype, ttnn.TileDescriptor(gate_bias_tensor.get_tile())
            )
            gate_indices_cb = cb_id_context.get_cb_id(
                gate_indices_tensor.dtype, ttnn.TileDescriptor(gate_indices_tensor.get_tile())
            )
            # gate_mm_weights_cb (gate_mm cores at col 12, rows 0-7) shares CB
            # ID with add_cb_in0 (eltwise_add cores = gate_proj DRAM-bank
            # primaries at cols 0,1,7,8). Strict-disjoint cores; per-core
            # descriptors carry their own (addr, dtype, tile) layouts. Saves 1
            # in (bf16, 32×32). Assumes gate_mm_weights_tensor is bf16 32×32.
            gate_mm_weights_cb = add_cb_in0
        else:
            gate_mm_output_cb = 0
            # No routing → no gate_mm anchor available; allocate sram_gate_out's own ID.
            sram_gate_proj_out_cb = cb_id_context.get_cb_id(data_format, TD_1x32)
            gate_input_cb = 0
            gate_proj_cb_index = 0
            mul_cb_scalar_src = 0
            gate_output_cb = 0
            gate_output_indices_cb = 0
            mul_cb_scalar = 0
            gate_bias_cb = 0
            gate_indices_cb = 0
            gate_mm_weights_cb = 0

        # ReduceToOne CBs (32x32, bfloat16)
        reduce_local_cb = add_cb_out  # intentional alias: reduce reads from add output
        reduce_received_cb = cb_id_context.get_cb_id(data_format, TD_32x32)
        reduce_output_cb = cb_id_context.get_cb_id(data_format, TD_32x32)
        reduce_scratch_cb = cb_id_context.get_cb_id(data_format, TD_32x32)
        reduce_packet_cb = cb_id_context.get_cb_id(data_format, TD_32x32)
        if enable_bcast:
            bcast_pkt_cb = cb_id_context.get_cb_id(data_format, TD_32x32)
        else:
            bcast_pkt_cb = None

        # ==================================================================
        # RMSNorm tile reinterpretation (compute kernel needs 32x32 or 16x32 tiles)
        # ==================================================================
        # Reinterpret N 1x32 tiles as full 32x32 or half 16x32 tiles for compute kernel
        # (matching standalone RMSNorm op.py behavior — mcast just sends raw bytes)
        FULL_32x32_TILE = ttnn.Tile((32, 32))
        HALF_16x32_TILE = ttnn.Tile((16, 32))
        is_16x32_tile = (K // FULL_32x32_TILE.tile_shape[1]) % FULL_32x32_TILE.tile_shape[0] != 0
        rmsnorm_interpreted_tile = HALF_16x32_TILE if is_16x32_tile else FULL_32x32_TILE
        rmsnorm_tile_descriptor = ttnn.TileDescriptor(rmsnorm_interpreted_tile)
        rmsnorm_cb_page_size = rmsnorm_interpreted_tile.get_tile_size(data_format)
        rmsnorm_num_tiles = K // (rmsnorm_interpreted_tile.tile_shape[0] * rmsnorm_interpreted_tile.tile_shape[1])

        # Tensor-backed CBs with reinterpreted rmsnorm tile.
        # rmsnorm_output_cb and residual_mcast_src_cb both live on sender core only,
        # which is strictly disjoint from add_cb_in0/add_cb_out's eltwise_add cores
        # (gate_proj_core_ranges = DRAM bank workers) AND from gate_mm cores (col 12,
        # rows 0-7). Overlay descriptors cb0_desc / cb24_desc are restricted to sender
        # in _overlap_cbs_with_sdpa_buffer to enforce the disjoint-cores invariant.
        # rmsnorm_gamma_cb stays its own id: gamma reads from a DRAM-overlapped tensor
        # on streamer cores, which == gate_proj_core_ranges (NOT disjoint from add).
        residual_mcast_src_cb = add_cb_out
        rmsnorm_gamma_cb = cb_id_context.get_cb_id(data_format, rmsnorm_tile_descriptor)
        rmsnorm_output_cb = add_cb_in0

        # ==================================================================
        # Residual Mcast (raw input from sender → residual CB on mcast grid)
        # ==================================================================
        residual_mcast_src_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
            residual_mcast_src_cb, shared_residual_mcast_src_tensor
        )
        # Override tile to 32x32 for RMSNorm compute (mcast just sends raw bytes, doesn't care)
        residual_mcast_src_cb_descriptor.format_descriptors[0].tile = rmsnorm_tile_descriptor
        residual_mcast_src_cb_descriptor.format_descriptors[0].page_size = rmsnorm_cb_page_size

        residual_mcast_data_size_bytes = num_tiles_k * tile_1x32_size
        residual_mcast_params = MoeOp.setup_mcast(
            device=device,
            sender_core=sender_core,
            mcast_grid=mcast_grid,
            src_cb=residual_mcast_src_cb,
            src_tensor=shared_residual_mcast_src_tensor,
            dst_cb=residual_mcast_dst_cb,
            dst_tensor=None,
            sender_semaphore_addr=mcast_data_sender_semaphore_addr,
            receiver_semaphore_addr=residual_mcast_receiver_semaphore_addr,
            data_size_bytes=residual_mcast_data_size_bytes,
        )
        # Override src_num_pages: CB 32 descriptor is now 32x32 tiles, so setup_sharded_buffer
        # and mcast sender need reinterpreted page count. dst_num_pages stays 224 (CB 33 is 1x32).
        residual_mcast_params["src_num_pages"] = rmsnorm_num_tiles

        # ==================================================================
        # RMSNorm (sender core: residual_mcast_src → rmsnorm_output)
        # Reuse residual_mcast_src tensor for CB descriptor (same core, shape, dtype);
        # overridden by _overlap_cbs_with_sdpa_buffer in production.
        # ==================================================================
        rmsnorm_output_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
            rmsnorm_output_cb, shared_residual_mcast_src_tensor
        )
        rmsnorm_output_cb_descriptor.format_descriptors[0].tile = rmsnorm_tile_descriptor
        rmsnorm_output_cb_descriptor.format_descriptors[0].page_size = rmsnorm_cb_page_size

        rmsnorm_gamma_fused_device0 = ttnn.get_device_tensors(rmsnorm_gamma_tensor.fused_tensor)[0]
        rmsnorm_gamma_cb_descriptor = cb_descriptor_from_overlapped_tensor(
            rmsnorm_gamma_cb, rmsnorm_gamma_tensor, rmsnorm_gamma_fused_device0
        )
        rmsnorm_gamma_cb_descriptor.format_descriptors[0].tile = rmsnorm_tile_descriptor
        rmsnorm_gamma_cb_descriptor.format_descriptors[0].page_size = rmsnorm_cb_page_size

        rmsnorm_epsilon_packed = float_to_uint32(epsilon)
        rmsnorm_scalar_packed = float_to_uint32(1.0 / math.sqrt(float(K)))

        # setup_sharded_buffer num_pages for gamma: use reinterpreted 32x32 tile count
        rmsnorm_gamma_num_pages = rmsnorm_num_tiles

        # ==================================================================
        # RMSNorm Mcast (broadcasts normalized input from rmsnorm_output_cb to all cores)
        # src/dst tensors no longer passed — src_num_pages overridden, dst CB overridden by SDPA overlap.
        # ==================================================================
        rmsnorm_mcast_data_size_bytes = num_tiles_k * tile_1x32_size
        rmsnorm_mcast_params = MoeOp.setup_mcast(
            device=device,
            sender_core=sender_core,
            mcast_grid=mcast_grid,
            src_cb=rmsnorm_output_cb,
            src_tensor=None,
            dst_cb=gate_mm_input_cb,
            dst_tensor=None,
            sender_semaphore_addr=mcast_data_sender_semaphore_addr,
            receiver_semaphore_addr=mcast_data_receiver_semaphore_addr,
            data_size_bytes=rmsnorm_mcast_data_size_bytes,
            src_num_pages=num_tiles_k,
        )
        rmsnorm_mcast_params["src_num_pages"] = rmsnorm_num_tiles

        # CB descriptor placeholder — overridden by _overlap_cbs_with_sdpa_buffer
        gate_mm_input_cb_descriptor = None

        # ==================================================================
        # Routing-only setup (gate MM, gate gather, gate, index/expert_scale mcast)
        # ==================================================================
        gate_mm_params = None
        gate_mm_gather_params = None
        gate_params = None
        gate_proj_cb_index_descriptor = None
        index_mcast_sender_semaphore_addr = 0
        index_mcast_num_pages = 0
        index_mcast_data_size_bytes = 0
        expert_scale_mcast_num_pages = 0
        expert_scale_mcast_data_size_bytes = 0

        if enable_routing:
            # Gate MM (SRAM Matmul)
            gate_mm_params = MoeOp.setup_sram_matmul(
                in0_cb=gate_mm_input_cb,
                in1_cb=gate_mm_weights_cb,
                out_cb=gate_mm_output_cb,
                weights_overlapped=gate_mm_weights_tensor,
                k_num_tiles=num_tiles_k,
                fused_activation=MoeRoutedExpertOp.ACTIVATION_SIGMOID,
                fused_activation_approx_mode=False,
            )

            # Gate MM Gather
            gate_mm_output_tile = ttnn.Tile([1, 32])
            gate_mm_output_tile_size = gate_mm_output_tile.get_tile_size(data_format)
            gate_mm_gather_data_size_bytes = gate_mm_params["out_w"] * gate_mm_output_tile_size

            gate_mm_gather_params = MoeOp.setup_gather(
                device=device,
                receiver_core=sender_core,
                sender_core_ranges=gate_mm_params["core_grid"],
                num_senders=gate_mm_params["num_cores"],
                data_size_bytes_per_sender=gate_mm_gather_data_size_bytes,
                src_cb=gate_mm_params["out_cb"],
                src_num_pages=gate_mm_params["out_w"],
                dst_cb=gate_input_cb,
                noc0_receiver_semaphore_addr=gather_noc0_receiver_semaphore_addr,
                noc1_receiver_semaphore_addr=gather_noc1_receiver_semaphore_addr,
                row_major=False,
                use_explicit_sender_index=False,
                dst_num_pages=1,  # gate_input is [16,16] with tile 16x16 → 1 page
            )

            # Gate
            gate_params = MoeRoutedExpertOp.setup_gate(
                input_cb=gate_input_cb,
                bias_cb=gate_bias_cb,
                indices_cb=gate_indices_cb,
                output_cb=gate_output_cb,
                output_indices_cb=gate_output_indices_cb,
                bias_tensor=gate_bias_tensor,
                indices_tensor=gate_indices_tensor,
                output_scores_tensor=gate_output_scores_tensor,
                output_indices_tensor=gate_output_indices_tensor,
                eps=1e-20,
                scaling_factor=2.5,
                enable_sigmoid=False,
            )

            # Index Mcast — tile info hardcoded (1x16 tile, uint16/bfloat16 = 32 bytes)
            TILE_1x16 = ttnn.Tile((1, 16))
            index_tile_size = TILE_1x16.get_tile_size(ttnn.bfloat16)  # 1*16*2 = 32 bytes
            index_mcast_sender_semaphore_addr = mcast_data_sender_semaphore_addr
            index_mcast_num_pages = 1
            index_mcast_data_size_bytes = index_tile_size

            # Expert Scale Mcast — same tile format (1x16, bfloat16, 32 bytes)
            expert_scale_mcast_num_pages = 1
            expert_scale_mcast_data_size_bytes = index_tile_size

        # ==================================================================
        # MatmulExpertCompressedDRAM: gate_proj
        # ==================================================================
        mesh_device_for_matmul_expert = gate_proj_weights_tensor[0].get_data_tensors()[0].device()
        # MoE routes top-8 of N_total experts → num_active_experts=8. Dense MLP has no
        # routing; cts_list contains exactly the experts to run (one CompressedTensor
        # per chunk from prepare_dense_routed_experts_compressed_tp8), so
        # num_active_experts = len(cts_list). The kernel iterates exp_i = 0..N-1 and,
        # because we feed enable_routing=False through to the unified kernel's
        # ``enable_indexing`` constexpr, synthesizes raw_idx = exp_i without touching
        # cb_index or index_l1_addr — every device runs the same [0..N-1] sequence
        # against its own TP-sliced weights.
        gate_up_num_active_experts = 8 if enable_routing else len(gate_proj_weights_tensor)
        # SRAM matmul kernels iterate `num_active_experts` slots.
        #   Routing: 8 (TopK count, matches DRAM; kernel filters via is_sram_expert).
        #   Dense:   len(sram_*_weights_tensor) — kernel synthesizes raw_idx for each
        #            slot (enable_indexing=0). 0 when no SRAM weights are provided.
        if enable_routing:
            sram_num_active_experts = gate_up_num_active_experts
        elif sram_gate_proj_weights_tensor:
            sram_num_active_experts = len(sram_gate_proj_weights_tensor)
        else:
            sram_num_active_experts = 0
        # Dense-MLP workaround for the low-num_active DRAM kernel bug: when SRAM
        # is placed in dense mode, the DRAM weights list stays at full length
        # (8 chunks). The kernel iterates all 8 slots and OR's EXPERT_SRAM_FLAG
        # onto slots >= ``num_dram_experts_pre_selected`` so the existing
        # is_sram_expert filter skips them — same well-tested code path as MoE
        # 1dram-7sram. Only read by the kernel's synthesized-index path
        # (``enable_indexing=0``); routing reads real indices from L1, so the
        # value is dead for MoE.
        num_dram_experts_pre_selected = gate_up_num_active_experts - sram_num_active_experts

        # K-split: 2 cores per bank split K; primary (= K-reducer at k_slice_idx=1)
        # holds the full K matmul output; sender (= k_slice_idx=0) NOC-writes its
        # partial to primary's cb_out and primary PACKs with l1_acc to sum. Shares
        # cb_in1 with up_proj — both must use the same cores_per_bank/subblock_k.
        gate_proj_params = MoeRoutedExpertOp.setup_matmul_expert_dram(
            mesh_device=mesh_device_for_matmul_expert,
            cts_list=gate_proj_weights_tensor,
            num_subblocks_k=4,
            subblock_n=1,
            cores_per_dram_bank=2,
            k_parallel_per_bank=2,
            primary_at_last_offset=True,
            num_active_experts=gate_up_num_active_experts,
            primary_worker_cores=gate_proj_worker_cores,
        )
        gate_proj_params["num_dram_experts_pre_selected"] = num_dram_experts_pre_selected

        # SRAM routed gate_proj setup (mirrors DRAM helper above). Caller
        # passes pre-built L1 CompressedTensors; we derive K/N tiling from
        # the CT shard shape and pass Kt = full K (= num_tiles_k).
        sram_gate_proj_params = None
        sram_gate_proj_cb_in1_descriptors_per_device = None  # dict[coord -> list of per-core CBDescriptors]
        if sram_gate_proj_weights_tensor:
            _ct0 = sram_gate_proj_weights_tensor[0]
            # Derive num_cores from the per-core allocation map (one entry per
            # core in the grid). get_data_tensors()[0] returns ONE core's tensor
            # in per_core_allocation mode, so its shard_spec.grid is single-core.
            _first_coord = next(iter(_ct0._multi_device_data_per_core))
            _num_cores = len(_ct0._multi_device_data_per_core[_first_coord])
            # HEIGHT_SHARDED layout: per_device_tiles_h = num_cores × per_core_K_tiles;
            # per_device_tiles_w = per_core_N (width not sharded).
            _sram_num_tiles_k = _ct0._per_device_tiles_h // _num_cores
            _sram_per_core_n = _ct0._per_device_tiles_w
            # Grid for setup helper: take from the FIRST per-core tensor's bounding box.
            # Actually simpler: rebuild from all per-core single-core ranges.
            _per_core_tensors = list(_ct0._multi_device_data_per_core[_first_coord].values())
            _grid_cores = [t.memory_config().shard_spec.grid.bounding_box().start for t in _per_core_tensors]
            _grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in _grid_cores])
            sram_gate_proj_params = MoeRoutedExpertOp.setup_matmul_expert_sram(
                mesh_device=mesh_device_for_matmul_expert,
                sram_cts_list=sram_gate_proj_weights_tensor,
                core_grid=_grid,
                num_tiles_k=_sram_num_tiles_k,
                per_core_n=_sram_per_core_n,
                Kt=num_tiles_k,
                num_active_experts=sram_num_active_experts,
                accum_experts=False,
            )
            # cb_in1 descriptor: built via CompressedTensor.cb_descriptor_from_compressed_tensor,
            # which returns ONE descriptor per core (since per_core_allocation=True).
            # Each core's descriptor points to that core's individual L1 buffer.
            # Mirrors the canonical pattern in micro_ops/matmul_expert/op.py:372.
            # Per-device: each mesh coord has its own per-core L1 layout.
            sram_gate_proj_cb_in1_descriptors_per_device = {}
            # cb_descriptor_from_compressed_tensor declares cb_in1 as a single
            # bfloat8_b "page" = whole per-core shard — works for the
            # compressed_custom_mm path (meta-driven strides) but breaks the
            # plain custom_mm path (which reads fifo_page_size as the per-tile
            # stride and walks core_size bytes per tile → garbage). Replace
            # format_descriptor with the actual per-tile shape (576B bfloat4_b)
            # so plain custom_mm strides correctly; compressed path ignores these.
            from models.demos.deepseek_v3_b1.micro_ops.dram_streaming_matmul_compressed.op import _TILE_SIZES

            _bfp4_tile_size = _TILE_SIZES[1]  # bfp4_b 32x32 = 576 B
            _tile_32x32 = ttnn.Tile([32, 32])
            for coord in sram_gate_proj_params["sram_fmt_l1_addr_per_device"]:
                _descs = _ct0.cb_descriptor_from_compressed_tensor(sram_gate_proj_cb_in1, device_coord=coord)
                for _desc in _descs:
                    # Override desc to report a uniform 1-page (576 B) cb_in1
                    # extent across (device, core).  The actual L1 buffer is
                    # >= 576 B (per min_shard_bytes=576 in build_sram_expert_
                    # weights for the cb_in1 backing expert), so this lie is
                    # safe — the framework sees a smaller window than the
                    # real buffer.  Kernel reads tile bytes via fmt metadata
                    # (compressed_custom_mm path), so the smaller window
                    # doesn't affect data reads.
                    _desc.total_size = _bfp4_tile_size
                    _desc.format_descriptors = [
                        ttnn.CBFormatDescriptor(
                            buffer_index=sram_gate_proj_cb_in1,
                            data_format=ttnn.bfloat4_b,
                            page_size=_bfp4_tile_size,
                            tile=ttnn.TileDescriptor(_tile_32x32),
                        )
                    ]
                sram_gate_proj_cb_in1_descriptors_per_device[coord] = _descs
            # cb_out descriptor is built in _overlap_cbs_with_sdpa_buffer
            # (kv_buf-overlaid, mirrors gate_proj_cb_out at cb11).

        # ==================================================================
        # MatmulExpertCompressedDRAM: up_proj
        # ==================================================================
        # K-split: matches gate_proj's config exactly (shared cb_in1 + same 16-core grid).
        up_proj_params = MoeRoutedExpertOp.setup_matmul_expert_dram(
            mesh_device=mesh_device_for_matmul_expert,
            cts_list=up_proj_weights_tensor,
            num_subblocks_k=4,
            subblock_n=1,
            cores_per_dram_bank=2,
            k_parallel_per_bank=2,
            primary_at_last_offset=True,
            num_active_experts=gate_up_num_active_experts,
            primary_worker_cores=gate_proj_worker_cores,
        )
        up_proj_params["num_dram_experts_pre_selected"] = num_dram_experts_pre_selected

        # SRAM routed up_proj setup — mirror of SRAM gate_proj. Caller passes
        # pre-built L1 CompressedTensors; we derive K/N tiling and call
        # setup_matmul_expert_sram. cb_in1 descriptors are per-device.
        sram_up_proj_params = None
        sram_up_proj_cb_in1_descriptors_per_device = None
        if sram_up_proj_weights_tensor:
            _ct0_up = sram_up_proj_weights_tensor[0]
            _first_coord_up = next(iter(_ct0_up._multi_device_data_per_core))
            _num_cores_up = len(_ct0_up._multi_device_data_per_core[_first_coord_up])
            _sram_num_tiles_k_up = _ct0_up._per_device_tiles_h // _num_cores_up
            _sram_per_core_n_up = _ct0_up._per_device_tiles_w
            _per_core_tensors_up = list(_ct0_up._multi_device_data_per_core[_first_coord_up].values())
            _grid_cores_up = [t.memory_config().shard_spec.grid.bounding_box().start for t in _per_core_tensors_up]
            _grid_up = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in _grid_cores_up])
            sram_up_proj_params = MoeRoutedExpertOp.setup_matmul_expert_sram(
                mesh_device=mesh_device_for_matmul_expert,
                sram_cts_list=sram_up_proj_weights_tensor,
                core_grid=_grid_up,
                num_tiles_k=_sram_num_tiles_k_up,
                per_core_n=_sram_per_core_n_up,
                Kt=num_tiles_k,
                num_active_experts=sram_num_active_experts,
                accum_experts=False,
            )
            sram_up_proj_cb_in1_descriptors_per_device = {}
            # See note on sram_gate_proj_cb_in1 above — replace format_descriptor
            # with the actual per-tile shape (576B bfloat4_b) for plain custom_mm.
            from models.demos.deepseek_v3_b1.micro_ops.dram_streaming_matmul_compressed.op import _TILE_SIZES

            _bfp4_tile_size = _TILE_SIZES[1]
            _tile_32x32 = ttnn.Tile([32, 32])
            for coord in sram_up_proj_params["sram_fmt_l1_addr_per_device"]:
                _descs_up = _ct0_up.cb_descriptor_from_compressed_tensor(sram_up_proj_cb_in1, device_coord=coord)
                for _desc in _descs_up:
                    # Uniform 576 B cb_in1 extent (= 1 bfp4 page); see gate.
                    _desc.total_size = _bfp4_tile_size
                    _desc.format_descriptors = [
                        ttnn.CBFormatDescriptor(
                            buffer_index=sram_up_proj_cb_in1,
                            data_format=ttnn.bfloat4_b,
                            page_size=_bfp4_tile_size,
                            tile=ttnn.TileDescriptor(_tile_32x32),
                        )
                    ]
                sram_up_proj_cb_in1_descriptors_per_device[coord] = _descs_up
            # cb_out descriptor is built in _overlap_cbs_with_sdpa_buffer.

        # SRAM routed down_proj setup. Per-expert weight layout mirrors shared_down:
        # per-device (K_per_dev=256, N_full=7168) width-sharded on the 112 shared
        # mcast receiver cores, giving (256, 64) per core (8 K-tiles × 2 N-tiles).
        # No K-slicing per core (Kt = num_tiles_k = sram_k_per_core = 8).
        sram_down_proj_params = None
        sram_down_proj_cb_in1_descriptors_per_device = None
        if sram_down_proj_weights_tensor:
            _ct0_dn = sram_down_proj_weights_tensor[0]
            _first_coord_dn = next(iter(_ct0_dn._multi_device_data_per_core))
            _num_cores_dn = len(_ct0_dn._multi_device_data_per_core[_first_coord_dn])
            # SRAM down is WIDTH_SHARDED (K replicated across cores, N split). The
            # per-device logical shape is (K, N_full) so per_device_tiles_h IS the
            # K tile count (replicated, NOT divided by num_cores), and the per-core
            # N tile count is per_device_tiles_w // num_cores. Gate/up are
            # HEIGHT_SHARDED and use the opposite idiom (see _ct0 above).
            _dn_layout = _ct0_dn._memory_config.memory_layout
            if _dn_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
                _sram_num_tiles_k_dn = _ct0_dn._per_device_tiles_h
                _sram_per_core_n_dn = _ct0_dn._per_device_tiles_w // _num_cores_dn
            else:
                _sram_num_tiles_k_dn = _ct0_dn._per_device_tiles_h // _num_cores_dn
                _sram_per_core_n_dn = _ct0_dn._per_device_tiles_w
            _per_core_tensors_dn = list(_ct0_dn._multi_device_data_per_core[_first_coord_dn].values())
            _grid_cores_dn = [t.memory_config().shard_spec.grid.bounding_box().start for t in _per_core_tensors_dn]
            _grid_dn = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in _grid_cores_dn])
            sram_down_proj_params = MoeRoutedExpertOp.setup_matmul_expert_sram(
                mesh_device=mesh_device_for_matmul_expert,
                sram_cts_list=sram_down_proj_weights_tensor,
                core_grid=_grid_dn,
                num_tiles_k=_sram_num_tiles_k_dn,
                per_core_n=_sram_per_core_n_dn,
                Kt=_sram_num_tiles_k_dn,  # no K-slicing per core
                num_active_experts=sram_num_active_experts,
                accum_experts=True,
            )
            sram_down_proj_cb_in1_descriptors_per_device = {}
            # See note on sram_gate_proj_cb_in1 above — replace format_descriptor
            # with the actual per-tile shape (576B bfloat4_b) for plain custom_mm.
            from models.demos.deepseek_v3_b1.micro_ops.dram_streaming_matmul_compressed.op import _TILE_SIZES

            _bfp4_tile_size = _TILE_SIZES[1]
            _tile_32x32 = ttnn.Tile([32, 32])
            for coord in sram_down_proj_params["sram_fmt_l1_addr_per_device"]:
                _descs_dn = _ct0_dn.cb_descriptor_from_compressed_tensor(sram_down_proj_cb_in1, device_coord=coord)
                for _desc in _descs_dn:
                    # Uniform 576 B cb_in1 extent (= 1 bfp4 page); see gate.
                    _desc.total_size = _bfp4_tile_size
                    _desc.format_descriptors = [
                        ttnn.CBFormatDescriptor(
                            buffer_index=sram_down_proj_cb_in1,
                            data_format=ttnn.bfloat4_b,
                            page_size=_bfp4_tile_size,
                            tile=ttnn.TileDescriptor(_tile_32x32),
                        )
                    ]
                sram_down_proj_cb_in1_descriptors_per_device[coord] = _descs_dn
            # cb_out descriptor is built in _overlap_cbs_with_sdpa_buffer.

        # ==================================================================
        # SRAM gate/up gather to sender (step 3 of separate-pipeline plan).
        # Layout: expert-major × N-major × K-within-N (matches GatedReduce's
        # tiles_per_k=8 contiguous-K-partials read pattern). The gather dst
        # CBs (sram_group1/group2_cb), GR output (sram_mcast_src_cb), and GR
        # scratch CBs (sram_intermed_cb, sram_gr_scalar_cb) are all sender-only
        # and kv_buf-overlaid in _overlap_cbs_with_sdpa_buffer; their L1
        # addresses are written to routed_ctx.sram_*_receiver_data_addr there.
        # ==================================================================
        sram_ag_sender_idx_core_values = None
        sram_bg_sender_idx_core_values = None
        sram_gather_data_size_bytes = 0
        sram_gather_expert_dst_stride = 0
        sram_gather_total_tiles = 0
        if sram_gate_proj_weights_tensor:
            sram_gather_data_size_bytes = tile_1x32_size  # 1 tile = 64 bytes
            _sram_total_cores = 64  # a_cores = b_cores = 64
            sram_gather_expert_dst_stride = _sram_total_cores * tile_1x32_size
            # Dense+SRAM: len(sram_*_weights_tensor); routing: TopK count (= 8). Both
            # flow through sram_num_active_experts.
            sram_gather_total_tiles = sram_num_active_experts * _sram_total_cores

            # Per-core sender_idx: offset within one expert's 64-tile slab.
            # SRAM gate/up cores arranged as 8 K-slices × 8 N-slices row-major
            # (lid // 8 = k_idx, lid % 8 = n_idx). Face-view GatedReduce
            # expects K-major × N-within-K so each face packs one K-slice's
            # 8 N-tiles → sender_idx = k_idx * 8 + n_idx = lid.
            from models.demos.deepseek_v3_b1.fused_ops.shared_expert.op import SharedExpertOp as _SEO

            _a_cores_for_gather, _b_cores_for_gather = _SEO.build_ab_grids()
            sram_ag_sender_idx_core_values = [(core, lid) for lid, core in enumerate(_a_cores_for_gather)]
            sram_bg_sender_idx_core_values = [(core, lid) for lid, core in enumerate(_b_cores_for_gather)]

        # Face-tile descriptor shared by all SRAM gather/GR CBs (16x16 bf16).
        face_tile = ttnn.Tile([FACE_HEIGHT, FACE_WIDTH])
        face_tile_desc = ttnn.TileDescriptor(FACE_HEIGHT, FACE_WIDTH, False)
        face_tile_size = face_tile.get_tile_size(data_format)  # 256 * 2 = 512 bytes for bf16

        # ==================================================================
        # Eltwise Mul: silu(gate_proj) * up_proj [* expert_scale if routing]
        # ==================================================================
        mul_params = MoeRoutedExpertOp.setup_eltwise_mul(
            cb_in0_index=mul_cb_in0,
            cb_in1_index=mul_cb_in1,
            cb_out_index=mul_cb_out,
            per_core_n=gate_proj_params["per_core_n"],
            cb_scalar_index=mul_cb_scalar,
            cb_scalar_src_index=mul_cb_scalar_src,
        )
        mul_num_tiles = mul_params["mul_num_tiles"]

        # ==================================================================
        # down_proj Gather — dst_num_pages computed from gate_proj dimensions
        # ==================================================================
        down_proj_gather_num_experts = gate_proj_params["num_active_experts"]
        down_proj_gather_data_size_bytes = gate_proj_params["per_core_n"] * tile_1x32_size
        # gate_proj_N_padded (in 1x32 tiles) = per_core_n * num_gate_proj_cores * num_experts
        down_proj_gather_dst_num_pages = (
            gate_proj_params["per_core_n"] * num_gate_proj_cores * down_proj_gather_num_experts
        )
        # Per-expert src chunk = per_core_n tiles × 64 B/tile (contiguous in sender CB).
        # Per-expert dst block = every sender's chunk concatenated on the receiver.
        down_proj_gather_src_page_size = down_proj_gather_data_size_bytes
        down_proj_gather_expert_dst_stride = num_gate_proj_cores * down_proj_gather_data_size_bytes
        down_proj_gather_params = MoeOp.setup_gather(
            device=device,
            receiver_core=sender_core,
            sender_core_ranges=gate_proj_core_ranges,
            num_senders=num_gate_proj_cores,
            data_size_bytes_per_sender=down_proj_gather_data_size_bytes,
            src_cb=mul_cb_out,
            src_num_pages=mul_num_tiles * down_proj_gather_num_experts,
            dst_cb=down_proj_gather_dst_cb,
            noc0_receiver_semaphore_addr=gather_noc0_receiver_semaphore_addr,
            noc1_receiver_semaphore_addr=gather_noc1_receiver_semaphore_addr,
            row_major=True,
            use_explicit_sender_index=True,
            dst_num_pages=down_proj_gather_dst_num_pages,
        )
        down_proj_gather_params["num_experts"] = down_proj_gather_num_experts
        down_proj_gather_params["src_page_size"] = down_proj_gather_src_page_size
        down_proj_gather_params["expert_dst_stride"] = down_proj_gather_expert_dst_stride

        # ==================================================================
        # down_proj Mcast — dimensions derived from gate_proj params
        # ==================================================================
        down_proj_mcast_num_tiles = down_proj_gather_dst_num_pages
        down_proj_mcast_data_size_bytes = down_proj_mcast_num_tiles * tile_1x32_size
        down_proj_mcast_params = MoeOp.setup_mcast(
            device=device,
            sender_core=sender_core,
            mcast_grid=mcast_grid,
            src_cb=down_proj_gather_dst_cb,
            src_tensor=None,
            dst_cb=down_proj_mcast_dst_cb,
            dst_tensor=None,
            sender_semaphore_addr=mcast_data_sender_semaphore_addr,
            receiver_semaphore_addr=down_proj_mcast_receiver_semaphore_addr,
            data_size_bytes=down_proj_mcast_data_size_bytes,
            src_num_pages=down_proj_mcast_num_tiles,
        )

        # ==================================================================
        # SRAM down Mcast — sender_core's GatedReduce output → all 128 mcast
        # cores (DRAM streamers + shared mcast receivers). dst CB shared with
        # DRAM down_proj input (down_proj_mcast_dst_cb, kv_buf-overlaid on the
        # full mcast grid). data_size_bytes / src_num_pages are CT worst-case
        # (n_active_experts faces); kernel overrides to n_sram_active × face
        # size at runtime.
        # ==================================================================
        # Dense+SRAM uses len(sram_*_weights_tensor) for the worst-case mcast
        # tile count; routing mode uses TopK count. Both flow through
        # sram_num_active_experts.
        sram_down_mcast_num_tiles = sram_num_active_experts if sram_gate_proj_weights_tensor is not None else 0
        sram_down_mcast_data_size_bytes = sram_down_mcast_num_tiles * face_tile_size
        sram_down_mcast_params = MoeOp.setup_mcast(
            device=device,
            sender_core=sender_core,
            mcast_grid=mcast_grid,
            src_cb=sram_mcast_src_cb,
            src_tensor=None,
            dst_cb=sram_down_mcast_dst_cb,
            dst_tensor=None,
            sender_semaphore_addr=mcast_data_sender_semaphore_addr,
            receiver_semaphore_addr=sram_down_mcast_receiver_semaphore_addr,
            data_size_bytes=sram_down_mcast_data_size_bytes,
            src_num_pages=sram_down_mcast_num_tiles,
        )
        # Receiver-side dst_num_pages is in dst CB page units (1×32 tile = 64 B),
        # not face-tile units (512 B). setup_mcast defaults dst_num_pages to
        # src_num_pages (face count) which is wrong for cross-tile-size mcasts.
        # Fix: total dst pages = num_active × face_size / dst_page_size = 8 × 8 = 64.
        # GR pads its mcast_src_cb pushes to num_active each iter so this constant
        # full-size push keeps the receiver's rd/wr ptrs aligned across iters.
        if sram_down_mcast_num_tiles:
            sram_down_mcast_params["dst_num_pages"] = sram_down_mcast_data_size_bytes // tile_1x32_size

        # ==================================================================
        # MatmulExpertCompressedDRAM: down_proj
        # ==================================================================
        # See gate/up note above: derive num_active_experts so dense MLP (one CT,
        # enable_routing=False) runs exactly one expert iteration per device.
        down_num_active_experts = 8 if enable_routing else len(down_proj_weights_tensor)
        down_proj_params = MoeRoutedExpertOp.setup_matmul_expert_dram(
            mesh_device=mesh_device_for_matmul_expert,
            cts_list=down_proj_weights_tensor,
            # 16 streamer cores (2 per bank): each bank's two cores split N
            # (per_core_n=14 halves to 7 per core), then sender NOC-writes
            # its accum onto the primary so downstream sees the same per-bank
            # N output as the 1-core layout — invisible to gate/up.
            num_subblocks_k=1,
            subblock_n=7,
            cores_per_dram_bank=2,
            primary_at_last_offset=True,
            num_active_experts=down_num_active_experts,
            accum_experts=1,
            primary_worker_cores=gate_proj_worker_cores,
        )
        down_proj_params["num_dram_experts_pre_selected"] = num_dram_experts_pre_selected

        # Wire index_l1_addr for MatmulExpertCompressedDRAM. The routing output
        # indices L1 buffer address is the same across mesh devices when allocations
        # are symmetric; use gate_output_indices_tensor.buffer_address(). When
        # enable_routing=False the kernel's ``enable_indexing`` constexpr is also off
        # (threaded via the same ``enable_routing`` named CT arg in moe_kernel.cpp /
        # decoder_block_kernel.cpp), so the kernel skips the L1 read entirely and
        # the index_l1_addr value is irrelevant — leave it sentinel-zeroed.
        if enable_routing and gate_output_indices_tensor is not None:
            index_l1_addr = _fused_base_addr(gate_output_indices_tensor)
            for p in (gate_proj_params, up_proj_params, down_proj_params):
                if "expert_offsets_l1_addr_per_device" in p:
                    p["index_l1_addr"] = index_l1_addr
                    # sender_index_l1_addr: same pre-overlay address, NOT overwritten by
                    # _overlap_cbs_with_sdpa_buffer. Used by the SRAM gather n_sram scan
                    # on sender_core (which reads gate_output_indices's L1, not the
                    # mcast-destination cb_index L1 used on receivers).
                    p["sender_index_l1_addr"] = index_l1_addr
            # Top-K scores L1 base — sender_core's source for the per-expert
            # scale; SRAM GatedReduce BRISC reads scalar[k] from here.
            if gate_output_scores_tensor is not None:
                gate_proj_params["scores_l1_addr"] = _fused_base_addr(gate_output_scores_tensor)

        # cb_fmt descriptors are now created inside _overlap_cbs_with_sdpa_buffer
        # (overlaid on kv_buf so the CB allocator sees the L1 region and won't stomp
        # it). The old in1_backing_tensor-based path was footgun-prone because the
        # private L1 region was invisible to the CB allocator. Stub None here;
        # _overlap_cbs_with_sdpa_buffer overrides each.
        gate_proj_cb_fmt_descriptor = None
        up_proj_cb_fmt_descriptor = None
        down_proj_cb_fmt_descriptor = None

        # ==================================================================
        # Eltwise Add: down_proj + shared_expert_output
        # Dimensions derived from down_proj params — no separate tensors needed.
        # per_core width = down_proj per_core_n tiles * 32 elements, total = per_core * num_cores
        # ==================================================================
        # Full per-primary N width after gather: per_core_n × cores_per_bank tiles.
        # In gather mode the receiver's L1 holds the bank's full N (sender's slice
        # via NOC + receiver's slice via PACK). cores_per_bank=1 for gate/up keeps
        # this equal to per_core_n × 32 (no behavior change).
        down_proj_width_per_core = down_proj_params["per_core_n"] * down_proj_params["cores_per_bank"] * 32
        down_proj_total_width = down_proj_width_per_core * num_gate_proj_cores
        # When reduce is enabled, CB 24 is a working buffer (backed by SDPA overlap).
        # When reduce is disabled, CB 24 is the final output (backed by final_output_tensor).
        add_out_tensor = final_output_tensor if reduce_intermediate_tensors is None else None
        add_params = MoeRoutedExpertOp.setup_eltwise_add(
            cb_in0_index=add_cb_in0,
            cb_in1_index=add_cb_in1,
            cb_out_index=add_cb_out,
            width_per_core=down_proj_width_per_core,
            total_width=down_proj_total_width,
            core_ranges=gate_proj_core_ranges,
            out_tensor=add_out_tensor,
        )

        # ==================================================================
        # ReduceToOne Setup
        # ==================================================================
        enable_reduce_to_one = (
            reduce_intermediate_tensors is not None
            and reduce_output_tensor is not None
            and reduce_semaphores is not None
            and reduce_root_coord is not None
            and mesh_rows == 4
            and mesh_cols == 2
        )

        reduce_params = {}
        if enable_reduce_to_one:
            # Get semaphore addresses
            reduce_sem_round1_addr = ttnn.get_global_semaphore_address(reduce_semaphores[0])
            reduce_sem_round2_addr = ttnn.get_global_semaphore_address(reduce_semaphores[1])
            reduce_sem_round3_addr = ttnn.get_global_semaphore_address(reduce_semaphores[2])
            reduce_sem_exit_addr = ttnn.get_global_semaphore_address(reduce_semaphores[3])

            # Get per-device tensors for reduce
            reduce_intermediate_per_device = ttnn.get_device_tensors(reduce_intermediate_tensors)
            reduce_output_per_device = ttnn.get_device_tensors(reduce_output_tensor)

            # Calculate reduce tensor properties (shard has 3x width for 3 reduction rounds)
            reduce_sample = reduce_intermediate_per_device[0]
            reduce_element_size = 2  # bfloat16

            reduce_shard_spec = reduce_sample.memory_config().shard_spec
            reduce_full_shard_shape = reduce_shard_spec.shape
            # Per-round shard shape is 1/3 of the full shard (3 rounds packed contiguously)
            reduce_shard_shape = [reduce_full_shard_shape[0], reduce_full_shard_shape[1] // 3]

            # Compute tiles use 32x32 format
            reduce_compute_tile_h = 32
            reduce_compute_tile_w = 32
            reduce_compute_tile_size = reduce_compute_tile_h * reduce_compute_tile_w * reduce_element_size
            reduce_shard_elements = reduce_shard_shape[0] * reduce_shard_shape[1]
            reduce_num_tiles = (reduce_shard_elements + (reduce_compute_tile_h * reduce_compute_tile_w) - 1) // (
                reduce_compute_tile_h * reduce_compute_tile_w
            )

            reduce_payload_size_bytes = reduce_shard_elements * reduce_element_size
            reduce_packet_header_size = ttnn.get_tt_fabric_packet_header_size_bytes()
            reduce_slot_size_bytes = reduce_packet_header_size + reduce_payload_size_bytes

            # Worker cores = gate_proj cores (DRAM bank cores)
            reduce_worker_grid = gate_proj_core_ranges
            reduce_worker_cores_list = ttnn.corerange_to_cores(reduce_worker_grid, row_wise=True)
            reduce_num_workers = len(reduce_worker_cores_list)

            # Build core -> column mapping for fabric core assignment
            reduce_column_to_cores = {}
            for core in reduce_worker_cores_list:
                x = core.x
                if x not in reduce_column_to_cores:
                    reduce_column_to_cores[x] = []
                reduce_column_to_cores[x].append(core)

            reduce_sorted_columns = sorted(reduce_column_to_cores.keys())
            for x in reduce_sorted_columns:
                reduce_column_to_cores[x].sort(key=lambda c: c.y)

            reduce_num_columns = len(reduce_sorted_columns)
            reduce_num_workers_per_column = len(reduce_column_to_cores[reduce_sorted_columns[0]])

            # Fabric cores: one per column, placed to the right of bottom core
            # Avoid sender_core to prevent NOC conflicts with bcast fabric
            reserved_cores = {(sender_core.x, sender_core.y)}
            reduce_fabric_cores = []
            reduce_column_to_fabric_core = {}
            for x in reduce_sorted_columns:
                bottom_core = max(reduce_column_to_cores[x], key=lambda c: c.y)
                fabric_core = ttnn.CoreCoord(bottom_core.x + 1, bottom_core.y)
                if (fabric_core.x, fabric_core.y) in reserved_cores:
                    fabric_core = ttnn.CoreCoord(bottom_core.x - 1, bottom_core.y)
                reduce_fabric_cores.append(fabric_core)
                reduce_column_to_fabric_core[x] = fabric_core

            # Core to slot index within column
            reduce_core_to_slot_idx = {}
            for x in reduce_sorted_columns:
                for slot_idx, core in enumerate(reduce_column_to_cores[x]):
                    reduce_core_to_slot_idx[(core.x, core.y)] = slot_idx

            # Core to shard index
            reduce_core_to_shard_idx = {}
            for shard_idx, core in enumerate(reduce_worker_cores_list):
                reduce_core_to_shard_idx[(core.x, core.y)] = shard_idx

            # Get output core from reduce_output_tensor
            reduce_output_sample = reduce_output_per_device[0]
            reduce_output_shard_spec = reduce_output_sample.memory_config().shard_spec
            reduce_output_core = reduce_output_shard_spec.grid.ranges()[0].start

            reduce_params = {
                "sem_round1_addr": reduce_sem_round1_addr,
                "sem_round2_addr": reduce_sem_round2_addr,
                "sem_round3_addr": reduce_sem_round3_addr,
                "sem_exit_addr": reduce_sem_exit_addr,
                "intermediate_per_device": reduce_intermediate_per_device,
                "output_per_device": reduce_output_per_device,
                "num_tiles": reduce_num_tiles,
                "payload_size_bytes": reduce_payload_size_bytes,
                "slot_size_bytes": reduce_slot_size_bytes,
                "compute_tile_size": reduce_compute_tile_size,
                "worker_cores_list": reduce_worker_cores_list,
                "num_workers": reduce_num_workers,
                "num_workers_per_column": reduce_num_workers_per_column,
                "num_columns": reduce_num_columns,
                "fabric_cores": reduce_fabric_cores,
                "column_to_fabric_core": reduce_column_to_fabric_core,
                "core_to_slot_idx": reduce_core_to_slot_idx,
                "core_to_shard_idx": reduce_core_to_shard_idx,
                "output_core": reduce_output_core,
            }

        # ==================================================================
        # Broadcast (CCL broadcast into CB 25)
        # ==================================================================
        bcast_params = None
        if enable_bcast:
            from models.demos.deepseek_v3_b1.micro_ops.host_io.utils import dtype_size

            bcast_out_ready_sem_addr = ttnn.get_global_semaphore_address(bcast_semaphores[0])
            bcast_barrier_sem_addr = ttnn.get_global_semaphore_address(bcast_semaphores[1])
            bcast_secondary_sync_sem_addr = ttnn.get_global_semaphore_address(bcast_semaphores[2])

            bcast_input_sample = ttnn.get_device_tensors(bcast_input_tensor)[0]
            bcast_input_shape = bcast_input_sample.shape
            bcast_element_size = dtype_size(bcast_input_sample.dtype)
            bcast_payload_size_bytes = bcast_input_shape[0] * bcast_input_shape[1] * bcast_element_size
            bcast_page_size_bytes = 32 * 32 * bcast_element_size  # interpret as 32x32 tile
            assert bcast_payload_size_bytes % bcast_page_size_bytes == 0
            bcast_input_num_pages = bcast_payload_size_bytes // bcast_page_size_bytes

            bcast_params = {
                "out_ready_sem_addr": bcast_out_ready_sem_addr,
                "barrier_sem_addr": bcast_barrier_sem_addr,
                "secondary_sync_sem_addr": bcast_secondary_sync_sem_addr,
                "sender_coord": bcast_sender_coord,
                "page_size_bytes": bcast_page_size_bytes,
                "input_num_pages": bcast_input_num_pages,
            }
            bcast_pkt_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(bcast_pkt_cb, bcast_input_tensor)
            bcast_pkt_cb_descriptor.format_descriptors[0].tile = rmsnorm_output_cb_descriptor.format_descriptors[0].tile
            bcast_pkt_cb_descriptor.format_descriptors[0].page_size = rmsnorm_output_cb_descriptor.format_descriptors[
                0
            ].page_size

        # ==================================================================
        # Per-core bank_id, vc, sender_idx
        # ==================================================================
        gate_proj_optimal_workers = get_pinned_optimal_dram_bank_to_logical_worker_assignment(device, ttnn.NOC.NOC_0)
        core_to_bank_id = {}
        for bank_id, core in enumerate(gate_proj_optimal_workers):
            core_to_bank_id[(core.x, core.y)] = bank_id

        bank_id_core_values = []
        vc_core_values = []
        sender_idx_core_values = []
        bank_ids = []
        for idx, core in enumerate(gate_proj_optimal_workers):
            bank_id = core_to_bank_id[(core.x, core.y)]
            vc = bank_id & 0x3
            for j in range(idx):
                prev_core = gate_proj_optimal_workers[j]
                if prev_core.y == core.y and (bank_ids[j] & 0x3) == (bank_id & 0x3):
                    vc = (vc + 1) & 0x3
                    break
            bank_ids.append(bank_id)
            bank_id_core_values.append((core, bank_id))
            vc_core_values.append((core, vc))
            sender_idx_core_values.append((core, idx))

        # Silu fast-path tile-h padded up to next valid face_r_dim ∈ {2, 4, 8, 16}.
        # Covers all num_active_experts * per_core_n output tiles as one tall tile.
        # For MoE gate K-split: num_active_experts=8, per_core_n=1 → silu_tile_h=8 (matches
        # the TD_8x32 used at the cb_id allocation site). The kernel's silu_iterations =
        # silu_tile_h / 2 = 4 covers both faces of the partial-face [silu_tile_h, 32] tile.
        from models.demos.deepseek_v3_b1.micro_ops.matmul_expert.op import _pad_to_face_r_dim

        gate_proj_silu_tile_h = _pad_to_face_r_dim(
            gate_proj_params["num_active_experts"] * gate_proj_params["per_core_n"] * 1
        )

        # ==================================================================
        # Return context
        # ==================================================================
        return _MoeRoutedExpertContext(
            # Device & mesh
            device=device,
            full_device_grid=full_device_grid,
            device_grid_size=device_grid_size,
            mesh_rows=mesh_rows,
            mesh_cols=mesh_cols,
            # Core grids
            sender_core=sender_core,
            input_core_grid=input_core_grid,
            mcast_grid=mcast_grid,
            mcast_worker_grid=mcast_worker_grid,
            gate_proj_core_ranges=gate_proj_core_ranges,
            num_gate_proj_cores=num_gate_proj_cores,
            # Data format & tiles
            data_format=data_format,
            tile_1x32_size=tile_1x32_size,
            num_tiles_k=num_tiles_k,
            # CB indices (shared)
            rmsnorm_output_cb=rmsnorm_output_cb,
            gate_mm_input_cb=gate_mm_input_cb,
            gate_proj_cb_in1=gate_proj_cb_in1,
            gate_proj_cb_out=gate_proj_cb_out,
            up_proj_cb_in1=up_proj_cb_in1,
            up_proj_cb_mm_out=up_proj_cb_mm_out,
            mul_cb_in0=mul_cb_in0,
            mul_cb_in1=mul_cb_in1,
            mul_cb_out=mul_cb_out,
            down_proj_gather_dst_cb=down_proj_gather_dst_cb,
            down_proj_mcast_dst_cb=down_proj_mcast_dst_cb,
            down_proj_cb_in1=down_proj_cb_in1,
            down_proj_cb_out=down_proj_cb_out,
            down_proj_cb_internal_acc=down_proj_cb_internal_acc,
            add_cb_in0=add_cb_in0,
            add_cb_in1=add_cb_in1,
            add_cb_out=add_cb_out,
            # Semaphore IDs (shared)
            mcast_data_sender_semaphore_addr=mcast_data_sender_semaphore_addr,
            mcast_data_receiver_semaphore_addr=mcast_data_receiver_semaphore_addr,
            gather_noc0_receiver_semaphore_addr=gather_noc0_receiver_semaphore_addr,
            gather_noc1_receiver_semaphore_addr=gather_noc1_receiver_semaphore_addr,
            # MatmulExpertCompressedDRAM CB indices
            gate_proj_cb_in0=gate_proj_cb_in0,
            gate_proj_cb_fmt=gate_proj_cb_fmt,
            gate_proj_cb_out_silu=gate_proj_cb_out_silu,
            gate_proj_silu_tile_h=gate_proj_silu_tile_h,
            up_proj_cb_in0=up_proj_cb_in0,
            up_proj_cb_fmt=up_proj_cb_fmt,
            down_proj_cb_in0=down_proj_cb_in0,
            down_proj_cb_fmt=down_proj_cb_fmt,
            # Setup result dicts (shared)
            rmsnorm_mcast_params=rmsnorm_mcast_params,
            gate_proj_params=gate_proj_params,
            up_proj_params=up_proj_params,
            mul_params=mul_params,
            down_proj_gather_params=down_proj_gather_params,
            down_proj_mcast_params=down_proj_mcast_params,
            sram_down_mcast_params=sram_down_mcast_params,
            down_proj_params=down_proj_params,
            add_params=add_params,
            # Derived
            mul_num_tiles=mul_num_tiles,
            # Pre-built CB descriptors (shared)
            rmsnorm_output_cb_descriptor=rmsnorm_output_cb_descriptor,
            gate_mm_input_cb_descriptor=gate_mm_input_cb_descriptor,
            # Residual mcast
            residual_mcast_src_cb=residual_mcast_src_cb,
            residual_mcast_dst_cb=residual_mcast_dst_cb,
            residual_mcast_receiver_semaphore_addr=residual_mcast_receiver_semaphore_addr,
            residual_mcast_src_cb_descriptor=residual_mcast_src_cb_descriptor,
            residual_mcast_params=residual_mcast_params,
            # RMSNorm
            rmsnorm_gamma_cb=rmsnorm_gamma_cb,
            rmsnorm_gamma_cb_descriptor=rmsnorm_gamma_cb_descriptor,
            rmsnorm_epsilon_packed=rmsnorm_epsilon_packed,
            rmsnorm_scalar_packed=rmsnorm_scalar_packed,
            rmsnorm_num_tiles=rmsnorm_num_tiles,
            rmsnorm_gamma_num_pages=rmsnorm_gamma_num_pages,
            # Per-core values
            bank_id_core_values=bank_id_core_values,
            vc_core_values=vc_core_values,
            sender_idx_core_values=sender_idx_core_values,
            # --- Routing-only fields ---
            enable_routing=enable_routing,
            # Routing CB indices
            gate_mm_weights_cb=gate_mm_weights_cb,
            gate_mm_output_cb=gate_mm_output_cb,
            gate_input_cb=gate_input_cb,
            gate_bias_cb=gate_bias_cb,
            gate_indices_cb=gate_indices_cb,
            gate_output_cb=gate_output_cb,
            gate_output_indices_cb=gate_output_indices_cb,
            gate_proj_cb_index=gate_proj_cb_index,
            mul_cb_scalar_src=mul_cb_scalar_src,
            mul_cb_scalar=mul_cb_scalar,
            # Routing semaphore IDs
            expert_scale_mcast_sender_semaphore_addr=expert_scale_mcast_sender_semaphore_addr,
            expert_scale_mcast_receiver_semaphore_addr=expert_scale_mcast_receiver_semaphore_addr,
            scan_sync_sem_addr=scan_sync_sem_addr,
            # Routing setup result dicts
            gate_mm_params=gate_mm_params,
            gate_mm_gather_params=gate_mm_gather_params,
            gate_params=gate_params,
            # Index mcast (routing only)
            index_mcast_sender_semaphore_addr=index_mcast_sender_semaphore_addr,
            index_mcast_receiver_semaphore_addr=index_mcast_receiver_semaphore_addr,
            index_mcast_num_pages=index_mcast_num_pages,
            index_mcast_data_size_bytes=index_mcast_data_size_bytes,
            # Expert scale mcast (routing only)
            expert_scale_mcast_num_pages=expert_scale_mcast_num_pages,
            expert_scale_mcast_data_size_bytes=expert_scale_mcast_data_size_bytes,
            # Routing CB descriptor
            gate_proj_cb_index_descriptor=gate_proj_cb_index_descriptor,
            # cb_fmt descriptors (MatmulExpertCompressedDRAM)
            gate_proj_cb_fmt_descriptor=gate_proj_cb_fmt_descriptor,
            up_proj_cb_fmt_descriptor=up_proj_cb_fmt_descriptor,
            down_proj_cb_fmt_descriptor=down_proj_cb_fmt_descriptor,
            # Testing flag (routing only)
            use_hardcoded_expert_index=use_hardcoded_expert_index,
            # ReduceToOne
            enable_reduce_to_one=enable_reduce_to_one,
            reduce_local_cb=reduce_local_cb,
            reduce_received_cb=reduce_received_cb,
            reduce_output_cb=reduce_output_cb,
            reduce_scratch_cb=reduce_scratch_cb,
            reduce_packet_cb=reduce_packet_cb,
            reduce_params=reduce_params if enable_reduce_to_one else None,
            # Broadcast
            enable_bcast=enable_bcast,
            bcast_pkt_cb=bcast_pkt_cb if enable_bcast else None,
            bcast_pkt_cb_descriptor=bcast_pkt_cb_descriptor if enable_bcast else None,
            bcast_params=bcast_params,
            # SRAM routed gate_proj (always-allocated CBs; params=None = no-op).
            # cb_out_descriptor is built in _overlap_cbs_with_sdpa_buffer.
            sram_gate_proj_cb_in1=sram_gate_proj_cb_in1,
            sram_gate_proj_out_cb=sram_gate_proj_out_cb,
            sram_gate_proj_params=sram_gate_proj_params,
            sram_gate_proj_cb_in1_descriptors_per_device=sram_gate_proj_cb_in1_descriptors_per_device,
            # SRAM routed up_proj (always-allocated CBs; params=None = no-op)
            sram_up_proj_cb_in1=sram_up_proj_cb_in1,
            sram_up_proj_out_cb=sram_up_proj_out_cb,
            sram_up_proj_params=sram_up_proj_params,
            sram_up_proj_cb_in1_descriptors_per_device=sram_up_proj_cb_in1_descriptors_per_device,
            # SRAM routed down_proj (always-allocated CBs; params=None = no-op)
            sram_down_proj_cb_in1=sram_down_proj_cb_in1,
            sram_down_proj_out_cb=sram_down_proj_out_cb,
            sram_down_proj_params=sram_down_proj_params,
            sram_down_proj_cb_in1_descriptors_per_device=sram_down_proj_cb_in1_descriptors_per_device,
            # Merged down output CB (eltwise_add or copy → residual_add input).
            merged_down_out_cb=merged_down_out_cb,
            # SRAM gate/up gather (always-allocated CB IDs; data only when SRAM enabled).
            # cb descriptors + receiver_data_addrs are set in
            # _overlap_cbs_with_sdpa_buffer (kv_buf-overlaid, sender-only region).
            sram_group1_cb=sram_group1_cb,
            sram_group2_cb=sram_group2_cb,
            # SRAM extended GatedReduce CBs.
            sram_intermed_cb=sram_intermed_cb,
            sram_mcast_src_cb=sram_mcast_src_cb,
            sram_gr_scalar_cb=sram_gr_scalar_cb,
            sram_down_mcast_dst_cb=sram_down_mcast_dst_cb,
            face_tile_desc=face_tile_desc,
            face_tile_size=face_tile_size,
            sram_ag_sender_idx_core_values=sram_ag_sender_idx_core_values,
            sram_bg_sender_idx_core_values=sram_bg_sender_idx_core_values,
            sram_gather_data_size_bytes=sram_gather_data_size_bytes,
            sram_gather_expert_dst_stride=sram_gather_expert_dst_stride,
            sram_gather_total_tiles=sram_gather_total_tiles,
            sram_use_compression=1 if enable_sram_bspm else 0,
            sram_ag_receiver_semaphore_addr=sem_addrs[MoeSem.SRAM_AG_GATHER],
            sram_bg_receiver_semaphore_addr=sem_addrs[MoeSem.SRAM_BG_GATHER],
        )

    @staticmethod
    def _build_compile_time_args(ctx, mesh_chip_id):
        """Build NCRISC, BRISC, and TRISC compile-time arg lists for routed expert."""
        ncrisc_named_compile_time_args = [
            # Input mcast (sender sharded buffer + receiver)
            ("moe_mcast_src_cb", ctx.rmsnorm_mcast_params["src_cb"]),
            ("moe_mcast_src_num_pages", ctx.rmsnorm_mcast_params["src_num_pages"]),
            ("moe_mcast_data_receiver_semaphore_addr", ctx.rmsnorm_mcast_params["receiver_semaphore_addr"]),
            ("moe_mcast_dst_cb", ctx.rmsnorm_mcast_params["dst_cb"]),
            ("moe_mcast_dst_num_pages", ctx.rmsnorm_mcast_params["dst_num_pages"]),
            # Residual mcast source (setup_sharded_buffer on sender core)
            ("shared_residual_mcast_src_cb", ctx.residual_mcast_params["src_cb"]),
            ("shared_residual_mcast_src_num_pages", ctx.residual_mcast_params["src_num_pages"]),
            # Residual mcast receiver (input from sender → residual CB on mcast grid)
            ("shared_residual_mcast_data_receiver_semaphore_addr", ctx.residual_mcast_receiver_semaphore_addr),
            ("shared_residual_cb", ctx.residual_mcast_dst_cb),
            ("shared_residual_num_pages", ctx.residual_mcast_params["dst_num_pages"]),
            # RMSNorm (setup_sharded_buffer for gamma on sender core)
            ("moe_rmsnorm_gamma_cb", ctx.rmsnorm_gamma_cb),
            ("moe_rmsnorm_gamma_num_pages", ctx.rmsnorm_gamma_num_pages),
            # Gate matmul reader (routing only — 0 when disabled)
            ("gate_mm_in0", ctx.gate_mm_params["in0_cb"] if ctx.enable_routing else 0),
            ("gate_mm_in1", ctx.gate_mm_params["in1_cb"] if ctx.enable_routing else 0),
            ("gate_mm_k_num_tiles", ctx.gate_mm_params["k_num_tiles"] if ctx.enable_routing else 0),
            ("gate_mm_out_w", ctx.gate_mm_params["out_w"] if ctx.enable_routing else 0),
            # Gate gather receiver (routing only)
            ("gather_noc0_num_senders", ctx.gate_mm_gather_params["noc0_num_senders"] if ctx.enable_routing else 0),
            ("gather_noc1_num_senders", ctx.gate_mm_gather_params["noc1_num_senders"] if ctx.enable_routing else 0),
            (
                "gather_noc0_receiver_semaphore_addr",
                ctx.gate_mm_gather_params["noc0_receiver_semaphore_addr"] if ctx.enable_routing else 0,
            ),
            (
                "gather_noc1_receiver_semaphore_addr",
                ctx.gate_mm_gather_params["noc1_receiver_semaphore_addr"] if ctx.enable_routing else 0,
            ),
            ("gather_dst_cb", ctx.gate_mm_gather_params["dst_cb"] if ctx.enable_routing else 0),
            ("gather_dst_num_pages", ctx.gate_mm_gather_params["dst_num_pages"] if ctx.enable_routing else 0),
            # Gate reader (routing only)
            ("gate_input_cb", ctx.gate_params["input_cb"] if ctx.enable_routing else 0),
            ("gate_bias_cb", ctx.gate_params["bias_cb"] if ctx.enable_routing else 0),
            ("gate_input_indices_cb", ctx.gate_params["indices_cb"] if ctx.enable_routing else 0),
            ("gate_proj_cb_index", ctx.gate_proj_cb_index),
            # NCRISC needs gate_output_indices_cb for the SRAM gather n_sram
            # scan on sender_core (cb_wait_front on the gate's output CB).
            ("gate_output_indices_cb", ctx.gate_params["output_indices_cb"] if ctx.enable_routing else 0),
            # Mul reader (setup mul_in1 buffer)
            ("mul_cb_in1", ctx.mul_cb_in1),
            ("mul_num_tiles", ctx.mul_num_tiles),
            # down_proj gather receiver (MoeGather: receiver on NCRISC)
            ("down_proj_gather_noc0_num_senders", ctx.down_proj_gather_params["noc0_num_senders"]),
            ("down_proj_gather_noc1_num_senders", ctx.down_proj_gather_params["noc1_num_senders"]),
            (
                "down_proj_gather_noc0_receiver_semaphore_addr",
                ctx.down_proj_gather_params["noc0_receiver_semaphore_addr"],
            ),
            (
                "down_proj_gather_noc1_receiver_semaphore_addr",
                ctx.down_proj_gather_params["noc1_receiver_semaphore_addr"],
            ),
            ("down_proj_gather_dst_cb", ctx.down_proj_gather_params["dst_cb"]),
            ("down_proj_gather_dst_num_pages", ctx.down_proj_gather_params["dst_num_pages"]),
            # Eltwise add
            ("add_cb_in0", ctx.add_cb_in0),
            ("add_cb_in1", ctx.add_cb_in1),
            ("add_cb_in0_wait_tiles", ctx.add_params["cb_in0_wait_tiles"]),
            ("add_cb_in1_wait_tiles", ctx.add_params["cb_in1_wait_tiles"]),
            # gate_proj MatmulExpertCompressedDRAM reader
            ("gate_proj_cb_in0", ctx.gate_proj_cb_in0),
            ("gate_proj_cb_in1", ctx.gate_proj_cb_in1),
            ("gate_proj_cb_out", ctx.gate_proj_cb_out),
            ("gate_proj_cb_index", ctx.gate_proj_cb_index),
            ("gate_proj_num_tiles_k", ctx.gate_proj_params["num_tiles_k"]),
            ("gate_proj_subblock_k", ctx.gate_proj_params["subblock_k"]),
            ("gate_proj_subblock_n", ctx.gate_proj_params["subblock_n"]),
            ("gate_proj_num_subblocks_k", ctx.gate_proj_params["num_subblocks_k"]),
            ("gate_proj_per_core_n", ctx.gate_proj_params["per_core_n"]),
            ("gate_proj_cb_in1_size_bytes", ctx.gate_proj_params["cb_in1_size_bytes"]),
            ("gate_proj_noc_max_page_size", ctx.gate_proj_params["noc_max_page_size"]),
            ("gate_proj_pipeline_sem_addr", ctx.gate_proj_params["pipeline_sem_addr"]),
            ("gate_proj_partial_sem_addr", ctx.gate_proj_params["partial_sem_addr"]),
            ("gate_proj_cores_per_bank", ctx.gate_proj_params["cores_per_bank"]),
            ("gate_proj_num_active_experts", ctx.gate_proj_params["num_active_experts"]),
            ("gate_proj_num_dram_experts_pre_selected", ctx.gate_proj_params["num_dram_experts_pre_selected"]),
            ("gate_proj_index_l1_addr", ctx.gate_proj_params["index_l1_addr"]),
            ("gate_proj_cb_fmt", ctx.gate_proj_cb_fmt),
            ("gate_proj_fmt_dram_addr", ctx.gate_proj_params["fmt_dram_addr"]),
            ("gate_proj_fmt_per_expert_bytes", ctx.gate_proj_params["fmt_per_expert_bytes"]),
            ("gate_proj_fmt_per_core_bytes", ctx.gate_proj_params["fmt_per_core_bytes"]),
            ("gate_proj_fmt_cb_l1_addr", ctx.gate_proj_params["fmt_cb_l1_addr"]),
            ("gate_proj_fmt_cb_page_size", ctx.gate_proj_params["fmt_cb_page_size"]),
            ("gate_proj_fmt_sem_addr_0", ctx.gate_proj_params["fmt_sem_addr_0"]),
            ("gate_proj_fmt_sem_addr_1", ctx.gate_proj_params["fmt_sem_addr_1"]),
            ("gate_proj_k_parallel_per_bank", ctx.gate_proj_params["k_parallel_per_bank"]),
            ("gate_proj_num_subblocks_k_local", ctx.gate_proj_params["num_subblocks_k_local"]),
            ("gate_proj_accum_experts", ctx.gate_proj_params["accum_experts"]),
            ("gate_proj_index_offset", 0),
            ("gate_proj_primary_at_last_offset", ctx.gate_proj_params["primary_at_last_offset"]),
            ("gate_proj_gather_sync_sem_addr", ctx.gate_proj_params["gather_sync_sem_addr"]),
            # gate_proj is non-accum — kernel constexpr-eliminates the cb_internal_acc
            # access, so we just pass cb_out's id to satisfy the template.
            ("gate_proj_cb_internal_acc", ctx.gate_proj_cb_out),
            # Physical CB base address — used as CBIn1ResetAddr template param so the
            # kernel's software write-pointer wrap aligns with the HW CB's physical
            # wrap boundary across sequential matmuls sharing cb_in1 (GP → UP).
            ("gate_proj_in1_buf_addr", ctx.gate_proj_params["in1_buf_addr"]),
            ("down_proj_in1_buf_addr", ctx.down_proj_params["in1_buf_addr"]),
            # up_proj MatmulExpertCompressedDRAM reader — shares weight CB with gate_proj
            ("up_proj_cb_in0", ctx.up_proj_cb_in0),
            ("up_proj_cb_in1", ctx.up_proj_cb_in1),
            ("up_proj_cb_mm_out", ctx.up_proj_cb_mm_out),
            ("up_proj_cb_index", ctx.gate_proj_cb_index),
            ("up_proj_num_tiles_k", ctx.up_proj_params["num_tiles_k"]),
            ("up_proj_subblock_k", ctx.up_proj_params["subblock_k"]),
            ("up_proj_subblock_n", ctx.up_proj_params["subblock_n"]),
            ("up_proj_num_subblocks_k", ctx.up_proj_params["num_subblocks_k"]),
            ("up_proj_per_core_n", ctx.up_proj_params["per_core_n"]),
            ("up_proj_cb_in1_size_bytes", ctx.up_proj_params["cb_in1_size_bytes"]),
            ("up_proj_noc_max_page_size", ctx.up_proj_params["noc_max_page_size"]),
            ("up_proj_pipeline_sem_addr", ctx.up_proj_params["pipeline_sem_addr"]),
            ("up_proj_partial_sem_addr", ctx.up_proj_params["partial_sem_addr"]),
            ("up_proj_cores_per_bank", ctx.up_proj_params["cores_per_bank"]),
            ("up_proj_num_active_experts", ctx.up_proj_params["num_active_experts"]),
            ("up_proj_num_dram_experts_pre_selected", ctx.up_proj_params["num_dram_experts_pre_selected"]),
            ("up_proj_index_l1_addr", ctx.up_proj_params["index_l1_addr"]),
            ("up_proj_cb_fmt", ctx.up_proj_cb_fmt),
            ("up_proj_fmt_dram_addr", ctx.up_proj_params["fmt_dram_addr"]),
            ("up_proj_fmt_per_expert_bytes", ctx.up_proj_params["fmt_per_expert_bytes"]),
            ("up_proj_fmt_per_core_bytes", ctx.up_proj_params["fmt_per_core_bytes"]),
            ("up_proj_fmt_cb_l1_addr", ctx.up_proj_params["fmt_cb_l1_addr"]),
            ("up_proj_fmt_cb_page_size", ctx.up_proj_params["fmt_cb_page_size"]),
            ("up_proj_fmt_sem_addr_0", ctx.up_proj_params["fmt_sem_addr_0"]),
            ("up_proj_fmt_sem_addr_1", ctx.up_proj_params["fmt_sem_addr_1"]),
            ("up_proj_k_parallel_per_bank", ctx.up_proj_params["k_parallel_per_bank"]),
            ("up_proj_num_subblocks_k_local", ctx.up_proj_params["num_subblocks_k_local"]),
            ("up_proj_accum_experts", ctx.up_proj_params["accum_experts"]),
            ("up_proj_index_offset", 0),
            ("up_proj_primary_at_last_offset", ctx.up_proj_params["primary_at_last_offset"]),
            ("up_proj_gather_sync_sem_addr", ctx.up_proj_params["gather_sync_sem_addr"]),
            # up_proj is non-accum — same as gate_proj, pass cb_out as a placeholder.
            ("up_proj_cb_internal_acc", ctx.up_proj_cb_mm_out),
            # down_proj MatmulExpertCompressedDRAM reader
            ("down_proj_cb_in0", ctx.down_proj_cb_in0),
            ("down_proj_cb_in1", ctx.down_proj_cb_in1),
            ("down_proj_cb_out", ctx.down_proj_cb_out),
            ("down_proj_cb_index", ctx.gate_proj_cb_index),
            ("down_proj_num_tiles_k", ctx.down_proj_params["num_tiles_k"]),
            ("down_proj_subblock_k", ctx.down_proj_params["subblock_k"]),
            ("down_proj_subblock_n", ctx.down_proj_params["subblock_n"]),
            ("down_proj_num_subblocks_k", ctx.down_proj_params["num_subblocks_k"]),
            ("down_proj_per_core_n", ctx.down_proj_params["per_core_n"]),
            ("down_proj_cb_in1_size_bytes", ctx.down_proj_params["cb_in1_size_bytes"]),
            ("down_proj_noc_max_page_size", ctx.down_proj_params["noc_max_page_size"]),
            ("down_proj_pipeline_sem_addr", ctx.down_proj_params["pipeline_sem_addr"]),
            ("down_proj_partial_sem_addr", ctx.down_proj_params["partial_sem_addr"]),
            ("down_proj_cores_per_bank", ctx.down_proj_params["cores_per_bank"]),
            ("down_proj_num_active_experts", ctx.down_proj_params["num_active_experts"]),
            ("down_proj_num_dram_experts_pre_selected", ctx.down_proj_params["num_dram_experts_pre_selected"]),
            ("down_proj_index_l1_addr", ctx.down_proj_params["index_l1_addr"]),
            ("down_proj_cb_fmt", ctx.down_proj_cb_fmt),
            ("down_proj_fmt_dram_addr", ctx.down_proj_params["fmt_dram_addr"]),
            ("down_proj_fmt_per_expert_bytes", ctx.down_proj_params["fmt_per_expert_bytes"]),
            ("down_proj_fmt_per_core_bytes", ctx.down_proj_params["fmt_per_core_bytes"]),
            ("down_proj_fmt_cb_l1_addr", ctx.down_proj_params["fmt_cb_l1_addr"]),
            ("down_proj_fmt_cb_page_size", ctx.down_proj_params["fmt_cb_page_size"]),
            ("down_proj_fmt_sem_addr_0", ctx.down_proj_params["fmt_sem_addr_0"]),
            ("down_proj_fmt_sem_addr_1", ctx.down_proj_params["fmt_sem_addr_1"]),
            ("down_proj_k_parallel_per_bank", ctx.down_proj_params["k_parallel_per_bank"]),
            ("down_proj_num_subblocks_k_local", ctx.down_proj_params["num_subblocks_k_local"]),
            ("down_proj_accum_experts", ctx.down_proj_params["accum_experts"]),
            ("down_proj_index_offset", 0),
            ("down_proj_primary_at_last_offset", ctx.down_proj_params["primary_at_last_offset"]),
            ("down_proj_gather_sync_sem_addr", ctx.down_proj_params["gather_sync_sem_addr"]),
            ("down_proj_cb_internal_acc", ctx.down_proj_cb_internal_acc),
            # Routing flag
            ("enable_routing", 1 if ctx.enable_routing else 0),
            # ReduceToOne reader args (CB indices + common RT arg base)
            ("reduce_local_cb", ctx.reduce_local_cb),
            ("reduce_received_cb", ctx.reduce_received_cb),
            ("reduce_ncrisc_common_rt_arg_base", 0),
            # Broadcast (base CT args, always present)
            ("bcast_pkt_cb", ctx.bcast_pkt_cb if ctx.enable_bcast else 0),
            ("bcast_ncrisc_common_rt_arg_base", 0),
        ]

        brisc_named_compile_time_args = [
            # Input mcast sender
            ("moe_mcast_dest_noc_start_x", ctx.rmsnorm_mcast_params["dest_noc_start_x"]),
            ("moe_mcast_dest_noc_start_y", ctx.rmsnorm_mcast_params["dest_noc_start_y"]),
            ("moe_mcast_dest_noc_end_x", ctx.rmsnorm_mcast_params["dest_noc_end_x"]),
            ("moe_mcast_dest_noc_end_y", ctx.rmsnorm_mcast_params["dest_noc_end_y"]),
            ("moe_mcast_num_cores", ctx.rmsnorm_mcast_params["num_cores"]),
            ("moe_mcast_data_sender_semaphore_addr", ctx.rmsnorm_mcast_params["sender_semaphore_addr"]),
            ("moe_mcast_data_receiver_semaphore_addr", ctx.rmsnorm_mcast_params["receiver_semaphore_addr"]),
            ("moe_mcast_data_size_bytes", ctx.rmsnorm_mcast_params["data_size_bytes"]),
            ("moe_mcast_src_cb", ctx.rmsnorm_mcast_params["src_cb"]),
            ("moe_mcast_dst_cb", ctx.rmsnorm_mcast_params["dst_cb"]),
            ("moe_mcast_src_num_pages", ctx.rmsnorm_mcast_params["src_num_pages"]),
            ("moe_mcast_is_part_of_receiver_grid", ctx.rmsnorm_mcast_params["is_sender_part_of_receiver_grid"]),
            # Residual mcast sender (input from sender → residual CB on mcast grid)
            ("shared_residual_mcast_data_sender_semaphore_addr", ctx.mcast_data_sender_semaphore_addr),
            ("shared_residual_mcast_data_receiver_semaphore_addr", ctx.residual_mcast_receiver_semaphore_addr),
            ("shared_residual_mcast_data_size_bytes", ctx.residual_mcast_params["data_size_bytes"]),
            ("shared_residual_mcast_src_cb", ctx.residual_mcast_params["src_cb"]),
            ("shared_residual_mcast_src_num_pages", ctx.residual_mcast_params["src_num_pages"]),
            ("shared_residual_mcast_dst_cb", ctx.residual_mcast_dst_cb),
            # Gate gather sender (routing only — 0 when disabled)
            ("gather_dest_noc_x", ctx.gate_mm_gather_params["dest_noc_x"] if ctx.enable_routing else 0),
            ("gather_dest_noc_y", ctx.gate_mm_gather_params["dest_noc_y"] if ctx.enable_routing else 0),
            ("gather_data_size_bytes", ctx.gate_mm_gather_params["data_size_bytes"] if ctx.enable_routing else 0),
            (
                "gather_receiver_semaphore_addr",
                ctx.gate_mm_gather_params["receiver_semaphore_addr"] if ctx.enable_routing else 0,
            ),
            ("gather_src_cb", ctx.gate_mm_gather_params["src_cb"] if ctx.enable_routing else 0),
            ("gather_src_num_pages", ctx.gate_mm_gather_params["src_num_pages"] if ctx.enable_routing else 0),
            (
                "gather_sender_grid_start_x",
                ctx.gate_mm_gather_params["sender_grid_start_x"] if ctx.enable_routing else 0,
            ),
            (
                "gather_sender_grid_start_y",
                ctx.gate_mm_gather_params["sender_grid_start_y"] if ctx.enable_routing else 0,
            ),
            ("gather_sender_grid_end_x", ctx.gate_mm_gather_params["sender_grid_end_x"] if ctx.enable_routing else 0),
            ("gather_sender_grid_end_y", ctx.gate_mm_gather_params["sender_grid_end_y"] if ctx.enable_routing else 0),
            ("gather_row_major", ctx.gate_mm_gather_params["row_major"] if ctx.enable_routing else 0),
            ("gather_receiver_data_addr", ctx.gate_mm_gather_params["receiver_data_addr"] if ctx.enable_routing else 0),
            # Gate writer (routing only)
            ("gate_output_cb", ctx.gate_params["output_cb"] if ctx.enable_routing else 0),
            ("gate_output_indices_cb", ctx.gate_params["output_indices_cb"] if ctx.enable_routing else 0),
            # Index mcast sender (routing only)
            ("index_mcast_sender_semaphore_addr", ctx.index_mcast_sender_semaphore_addr),
            ("index_mcast_receiver_semaphore_addr", ctx.index_mcast_receiver_semaphore_addr),
            ("index_mcast_data_size_bytes", ctx.index_mcast_data_size_bytes),
            ("index_mcast_num_pages", ctx.index_mcast_num_pages),
            ("gate_proj_cb_index", ctx.gate_proj_cb_index),
            # Required for MatmulExpertCompressedDRAM ResetCBIn1 template param (referenced in moe_kernel.cpp outer scope)
            ("gate_proj_in1_buf_addr", ctx.gate_proj_params["in1_buf_addr"]),
            ("down_proj_in1_buf_addr", ctx.down_proj_params["in1_buf_addr"]),
            # Read by moe_kernel.cpp's dense-mode n_dram_active derivation —
            # BRISC needs it to gate the mcast helpers consistently with TRISC/NCRISC.
            ("gate_proj_num_active_experts", ctx.gate_proj_params["num_active_experts"]),
            ("gate_proj_num_dram_experts_pre_selected", ctx.gate_proj_params["num_dram_experts_pre_selected"]),
            # Expert scale mcast sender (routing only)
            ("expert_scale_mcast_sender_semaphore_addr", ctx.expert_scale_mcast_sender_semaphore_addr),
            ("expert_scale_mcast_receiver_semaphore_addr", ctx.expert_scale_mcast_receiver_semaphore_addr),
            ("expert_scale_mcast_data_size_bytes", ctx.expert_scale_mcast_data_size_bytes),
            ("expert_scale_mcast_num_pages", ctx.expert_scale_mcast_num_pages),
            ("mul_cb_scalar_src", ctx.mul_cb_scalar_src),
            ("mul_cb_scalar", ctx.mul_cb_scalar),
            ("mul_scalar_index_offset", 0),
            # Mul writer
            ("mul_cb_out", ctx.mul_cb_out),
            ("mul_num_tiles", ctx.mul_num_tiles),
            # Mul consumes per_core_n × num_active_experts tiles per CB (gate_proj/up_proj
            # each push that many). Use the matmul's own num_active_experts so dense MLP
            # (8 chunks, no routing) and MoE (top-8 of routing) both get 8 here, while a
            # legacy 1-expert path would get 1.
            ("mul_num_experts", ctx.gate_proj_params["num_active_experts"]),
            # down_proj gather sender (MoeGather: sender on BRISC)
            ("down_proj_gather_dest_noc_x", ctx.down_proj_gather_params["dest_noc_x"]),
            ("down_proj_gather_dest_noc_y", ctx.down_proj_gather_params["dest_noc_y"]),
            ("down_proj_gather_data_size_bytes", ctx.down_proj_gather_params["data_size_bytes"]),
            ("down_proj_gather_receiver_semaphore_addr", ctx.down_proj_gather_params["receiver_semaphore_addr"]),
            ("down_proj_gather_src_cb", ctx.down_proj_gather_params["src_cb"]),
            ("down_proj_gather_src_num_pages", ctx.down_proj_gather_params["src_num_pages"]),
            ("down_proj_gather_sender_grid_start_x", ctx.down_proj_gather_params["sender_grid_start_x"]),
            ("down_proj_gather_sender_grid_start_y", ctx.down_proj_gather_params["sender_grid_start_y"]),
            ("down_proj_gather_sender_grid_end_x", ctx.down_proj_gather_params["sender_grid_end_x"]),
            ("down_proj_gather_sender_grid_end_y", ctx.down_proj_gather_params["sender_grid_end_y"]),
            ("down_proj_gather_row_major", ctx.down_proj_gather_params["row_major"]),
            ("down_proj_gather_receiver_data_addr", ctx.down_proj_gather_params["receiver_data_addr"]),
            ("down_proj_gather_num_experts", ctx.down_proj_gather_params.get("num_experts", 1)),
            ("down_proj_gather_src_page_size", ctx.down_proj_gather_params.get("src_page_size", 0)),
            ("down_proj_gather_expert_dst_stride", ctx.down_proj_gather_params.get("expert_dst_stride", 0)),
            # down_proj mcast sender
            ("down_proj_mcast_sender_semaphore_addr", ctx.down_proj_mcast_params["sender_semaphore_addr"]),
            ("down_proj_mcast_receiver_semaphore_addr", ctx.down_proj_mcast_params["receiver_semaphore_addr"]),
            ("down_proj_mcast_data_size_bytes", ctx.down_proj_mcast_params["data_size_bytes"]),
            ("down_proj_mcast_src_cb", ctx.down_proj_mcast_params["src_cb"]),
            ("down_proj_mcast_dst_cb", ctx.down_proj_mcast_params["dst_cb"]),
            ("down_proj_mcast_src_num_pages", ctx.down_proj_mcast_params["src_num_pages"]),
            # down_proj mcast receiver
            ("down_proj_mcast_dst_num_pages", ctx.down_proj_mcast_params["dst_num_pages"]),
            # Per-expert byte/page counts for the [SRAM | DRAM] concat layout.
            # SRAM face byte size = DRAM per-expert byte size, so the same value
            # is used to size the SRAM mcast and to offset the DRAM mcast past
            # the SRAM portion at runtime.
            (
                "down_proj_mcast_per_expert_bytes",
                ctx.down_proj_mcast_params["data_size_bytes"] // ctx.down_proj_params["num_active_experts"],
            ),
            (
                "down_proj_mcast_per_expert_pages",
                ctx.down_proj_mcast_params["src_num_pages"] // ctx.down_proj_params["num_active_experts"],
            ),
            ("down_proj_num_active_experts", ctx.down_proj_params["num_active_experts"]),
            ("down_proj_num_dram_experts_pre_selected", ctx.down_proj_params["num_dram_experts_pre_selected"]),
            # SRAM down mcast (sender + receiver). data_size_bytes / num_pages
            # are CT worst-case (n_active faces); kernel overrides at runtime to
            # n_sram_active × face_tile_size.
            (
                "sram_down_mcast_sender_semaphore_addr",
                ctx.sram_down_mcast_params.get("sender_semaphore_addr", 0),
            ),
            (
                "sram_down_mcast_receiver_semaphore_addr",
                ctx.sram_down_mcast_params.get("receiver_semaphore_addr", 0),
            ),
            (
                "sram_down_mcast_data_size_bytes",
                ctx.sram_down_mcast_params.get("data_size_bytes", 0),
            ),
            ("sram_down_mcast_src_cb", ctx.sram_down_mcast_params.get("src_cb", 0)),
            ("sram_down_mcast_dst_cb", ctx.sram_down_mcast_params.get("dst_cb", 0)),
            ("sram_down_mcast_src_num_pages", ctx.sram_down_mcast_params.get("src_num_pages", 0)),
            ("sram_down_mcast_dst_num_pages", ctx.sram_down_mcast_params.get("dst_num_pages", 0)),
            # Eltwise add CB (needed by output mcast sender for get_write_ptr)
            ("add_cb_in1", ctx.add_cb_in1),
            # Routing flag
            ("enable_routing", 1 if ctx.enable_routing else 0),
            # ReduceToOne writer args (CB indices + RT arg bases)
            ("reduce_local_cb", ctx.reduce_local_cb),
            ("reduce_scratch_cb", ctx.reduce_scratch_cb),
            ("reduce_brisc_rt_arg_base", 0),
            ("reduce_brisc_fabric_rt_arg_base", 0),
            ("reduce_persistent_fabric_rt_arg_base", 18),
            # Broadcast (base CT args, always present)
            ("bcast_pkt_cb", ctx.bcast_pkt_cb if ctx.enable_bcast else 0),
        ]

        trisc_named_compile_time_args = [
            # RMSNorm compute (sender core only)
            ("moe_rmsnorm_input_cb", ctx.residual_mcast_src_cb),
            ("moe_rmsnorm_gamma_cb", ctx.rmsnorm_gamma_cb),
            ("moe_rmsnorm_output_cb", ctx.rmsnorm_output_cb),
            ("moe_rmsnorm_fp32_acc", 0),
            ("moe_rmsnorm_num_tiles", ctx.rmsnorm_num_tiles),
            ("moe_rmsnorm_rsqrt_fast_approx", 0),
            ("moe_rmsnorm_trisc_common_rt_arg_base", 0),
            # Gate matmul compute (routing only — 0 when disabled)
            ("gate_mm_in0", ctx.gate_mm_params["in0_cb"] if ctx.enable_routing else 0),
            ("gate_mm_in1", ctx.gate_mm_params["in1_cb"] if ctx.enable_routing else 0),
            ("gate_mm_out", ctx.gate_mm_params["out_cb"] if ctx.enable_routing else 0),
            ("gate_mm_k_num_tiles", ctx.gate_mm_params["k_num_tiles"] if ctx.enable_routing else 0),
            ("gate_mm_out_w", ctx.gate_mm_params["out_w"] if ctx.enable_routing else 0),
            ("gate_mm_fused_activation", ctx.gate_mm_params["fused_activation"] if ctx.enable_routing else 0),
            (
                "gate_mm_fused_activation_approx_mode",
                ctx.gate_mm_params["fused_activation_approx_mode"] if ctx.enable_routing else 0,
            ),
            # Gate compute (routing only)
            ("gate_input_cb", ctx.gate_params["input_cb"] if ctx.enable_routing else 0),
            ("gate_bias_cb", ctx.gate_params["bias_cb"] if ctx.enable_routing else 0),
            ("gate_input_indices_cb", ctx.gate_params["indices_cb"] if ctx.enable_routing else 0),
            ("gate_output_cb", ctx.gate_params["output_cb"] if ctx.enable_routing else 0),
            ("gate_output_indices_cb", ctx.gate_params["output_indices_cb"] if ctx.enable_routing else 0),
            ("gate_eps", ctx.gate_params["eps"] if ctx.enable_routing else 0),
            ("gate_scaling_factor", ctx.gate_params["scaling_factor"] if ctx.enable_routing else 0),
            ("gate_enable_sigmoid", ctx.gate_params["enable_sigmoid"] if ctx.enable_routing else 0),
            # gate_proj MatmulExpertCompressedDRAM compute
            ("gate_proj_cb_in0", ctx.gate_proj_cb_in0),
            ("gate_proj_cb_in1", ctx.gate_proj_cb_in1),
            ("gate_proj_cb_out", ctx.gate_proj_cb_out),
            ("gate_proj_cb_index", ctx.gate_proj_cb_index),
            ("gate_proj_num_tiles_k", ctx.gate_proj_params["num_tiles_k"]),
            ("gate_proj_subblock_k", ctx.gate_proj_params["subblock_k"]),
            ("gate_proj_subblock_n", ctx.gate_proj_params["subblock_n"]),
            ("gate_proj_num_subblocks_k", ctx.gate_proj_params["num_subblocks_k"]),
            ("gate_proj_per_core_n", ctx.gate_proj_params["per_core_n"]),
            ("gate_proj_num_active_experts", ctx.gate_proj_params["num_active_experts"]),
            ("gate_proj_num_dram_experts_pre_selected", ctx.gate_proj_params["num_dram_experts_pre_selected"]),
            ("gate_proj_index_l1_addr", ctx.gate_proj_params["index_l1_addr"]),
            ("gate_proj_cb_fmt", ctx.gate_proj_cb_fmt),
            ("gate_proj_dram_meta_words_per_block", ctx.gate_proj_params["dram_meta_words_per_block"]),
            ("gate_proj_in0_page_size", ctx.gate_proj_params["in0_page_size"]),
            ("gate_proj_fmt_cb_l1_addr", ctx.gate_proj_params["fmt_cb_l1_addr"]),
            ("gate_proj_fmt_cb_page_size", ctx.gate_proj_params["fmt_cb_page_size"]),
            ("gate_proj_fmt_sem_addr_0", ctx.gate_proj_params["fmt_sem_addr_0"]),
            ("gate_proj_fmt_sem_addr_1", ctx.gate_proj_params["fmt_sem_addr_1"]),
            ("gate_proj_k_parallel_per_bank", ctx.gate_proj_params["k_parallel_per_bank"]),
            ("gate_proj_num_subblocks_k_local", ctx.gate_proj_params["num_subblocks_k_local"]),
            ("gate_proj_partial_sem_addr", ctx.gate_proj_params["partial_sem_addr"]),
            ("gate_proj_accum_experts", ctx.gate_proj_params["accum_experts"]),
            ("gate_proj_fuse_silu", 1),
            # Silu fast-path — only used when fuse_silu=1. gate_proj has it; up/down pass 0.
            ("gate_proj_cb_out_silu", ctx.gate_proj_cb_out_silu),
            ("gate_proj_silu_tile_h", ctx.gate_proj_silu_tile_h),
            ("gate_proj_cores_per_bank", ctx.gate_proj_params["cores_per_bank"]),
            ("gate_proj_primary_at_last_offset", ctx.gate_proj_params["primary_at_last_offset"]),
            ("gate_proj_gather_sync_sem_addr", ctx.gate_proj_params["gather_sync_sem_addr"]),
            ("gate_proj_cb_internal_acc", ctx.gate_proj_cb_out),
            # Required for MatmulExpertCompressedDRAM ResetCBIn1 template param (referenced in moe_kernel.cpp outer scope)
            ("gate_proj_in1_buf_addr", ctx.gate_proj_params["in1_buf_addr"]),
            # up_proj MatmulExpertCompressedDRAM compute — shares weight CB with gate_proj
            ("up_proj_cb_in0", ctx.up_proj_cb_in0),
            ("up_proj_cb_in1", ctx.up_proj_cb_in1),
            ("up_proj_cb_mm_out", ctx.up_proj_cb_mm_out),
            ("up_proj_cb_index", ctx.gate_proj_cb_index),
            ("up_proj_num_tiles_k", ctx.up_proj_params["num_tiles_k"]),
            ("up_proj_subblock_k", ctx.up_proj_params["subblock_k"]),
            ("up_proj_subblock_n", ctx.up_proj_params["subblock_n"]),
            ("up_proj_num_subblocks_k", ctx.up_proj_params["num_subblocks_k"]),
            ("up_proj_per_core_n", ctx.up_proj_params["per_core_n"]),
            ("up_proj_num_active_experts", ctx.up_proj_params["num_active_experts"]),
            ("up_proj_num_dram_experts_pre_selected", ctx.up_proj_params["num_dram_experts_pre_selected"]),
            ("up_proj_index_l1_addr", ctx.up_proj_params["index_l1_addr"]),
            ("up_proj_cb_fmt", ctx.up_proj_cb_fmt),
            ("up_proj_dram_meta_words_per_block", ctx.up_proj_params["dram_meta_words_per_block"]),
            ("up_proj_in0_page_size", ctx.up_proj_params["in0_page_size"]),
            ("up_proj_fmt_cb_l1_addr", ctx.up_proj_params["fmt_cb_l1_addr"]),
            ("up_proj_fmt_cb_page_size", ctx.up_proj_params["fmt_cb_page_size"]),
            ("up_proj_fmt_sem_addr_0", ctx.up_proj_params["fmt_sem_addr_0"]),
            ("up_proj_fmt_sem_addr_1", ctx.up_proj_params["fmt_sem_addr_1"]),
            ("up_proj_k_parallel_per_bank", ctx.up_proj_params["k_parallel_per_bank"]),
            ("up_proj_num_subblocks_k_local", ctx.up_proj_params["num_subblocks_k_local"]),
            ("up_proj_partial_sem_addr", ctx.up_proj_params["partial_sem_addr"]),
            ("up_proj_accum_experts", ctx.up_proj_params["accum_experts"]),
            ("up_proj_fuse_silu", 0),
            ("up_proj_cb_out_silu", 0),
            ("up_proj_silu_tile_h", 0),
            ("up_proj_cores_per_bank", ctx.up_proj_params["cores_per_bank"]),
            ("up_proj_primary_at_last_offset", ctx.up_proj_params["primary_at_last_offset"]),
            ("up_proj_gather_sync_sem_addr", ctx.up_proj_params["gather_sync_sem_addr"]),
            ("up_proj_cb_internal_acc", ctx.up_proj_cb_mm_out),
            # Mul compute
            ("mul_cb_in0", ctx.mul_cb_in0),
            ("mul_cb_in1", ctx.mul_cb_in1),
            ("mul_cb_out", ctx.mul_cb_out),
            ("mul_num_tiles", ctx.mul_num_tiles),
            # Mul consumes per_core_n × num_active_experts tiles per CB (gate_proj/up_proj
            # each push that many). Use the matmul's own num_active_experts so dense MLP
            # (8 chunks, no routing) and MoE (top-8 of routing) both get 8 here, while a
            # legacy 1-expert path would get 1.
            ("mul_num_experts", ctx.gate_proj_params["num_active_experts"]),
            ("mul_cb_scalar", ctx.mul_cb_scalar),
            ("mul_fp32_dest_acc_en", 0),
            # down_proj MatmulExpertCompressedDRAM compute
            ("down_proj_cb_in0", ctx.down_proj_cb_in0),
            ("down_proj_cb_in1", ctx.down_proj_cb_in1),
            ("down_proj_cb_out", ctx.down_proj_cb_out),
            ("down_proj_cb_index", ctx.gate_proj_cb_index),
            ("down_proj_num_tiles_k", ctx.down_proj_params["num_tiles_k"]),
            ("down_proj_subblock_k", ctx.down_proj_params["subblock_k"]),
            ("down_proj_subblock_n", ctx.down_proj_params["subblock_n"]),
            ("down_proj_num_subblocks_k", ctx.down_proj_params["num_subblocks_k"]),
            ("down_proj_per_core_n", ctx.down_proj_params["per_core_n"]),
            ("down_proj_num_active_experts", ctx.down_proj_params["num_active_experts"]),
            ("down_proj_num_dram_experts_pre_selected", ctx.down_proj_params["num_dram_experts_pre_selected"]),
            ("down_proj_index_l1_addr", ctx.down_proj_params["index_l1_addr"]),
            ("down_proj_cb_fmt", ctx.down_proj_cb_fmt),
            ("down_proj_dram_meta_words_per_block", ctx.down_proj_params["dram_meta_words_per_block"]),
            ("down_proj_in0_page_size", ctx.down_proj_params["in0_page_size"]),
            ("down_proj_fmt_cb_l1_addr", ctx.down_proj_params["fmt_cb_l1_addr"]),
            ("down_proj_fmt_cb_page_size", ctx.down_proj_params["fmt_cb_page_size"]),
            ("down_proj_fmt_sem_addr_0", ctx.down_proj_params["fmt_sem_addr_0"]),
            ("down_proj_fmt_sem_addr_1", ctx.down_proj_params["fmt_sem_addr_1"]),
            ("down_proj_k_parallel_per_bank", ctx.down_proj_params["k_parallel_per_bank"]),
            ("down_proj_num_subblocks_k_local", ctx.down_proj_params["num_subblocks_k_local"]),
            ("down_proj_partial_sem_addr", ctx.down_proj_params["partial_sem_addr"]),
            ("down_proj_accum_experts", ctx.down_proj_params["accum_experts"]),
            ("down_proj_fuse_silu", 0),
            ("down_proj_cb_out_silu", 0),
            ("down_proj_silu_tile_h", 0),
            ("down_proj_cores_per_bank", ctx.down_proj_params["cores_per_bank"]),
            ("down_proj_primary_at_last_offset", ctx.down_proj_params["primary_at_last_offset"]),
            ("down_proj_gather_sync_sem_addr", ctx.down_proj_params["gather_sync_sem_addr"]),
            ("down_proj_cb_internal_acc", ctx.down_proj_cb_internal_acc),
            # Required by decoder_block_kernel.cpp's DRAMStreamingMatmul CBIn1ResetAddr template param
            ("down_proj_in1_buf_addr", ctx.down_proj_params["in1_buf_addr"]),
            # Shared matmul-expert compute args (index_offset: 0 for TP8 all-device broadcast)
            ("gate_proj_index_offset", 0),
            ("up_proj_index_offset", 0),
            ("down_proj_index_offset", 0),
            # Routing flag
            ("enable_routing", 1 if ctx.enable_routing else 0),
            # Eltwise add compute
            ("add_cb_in0", ctx.add_cb_in0),
            ("add_cb_in1", ctx.add_cb_in1),
            ("add_cb_out", ctx.add_cb_out),
            ("add_num_tiles", ctx.add_params["num_tiles"]),
            ("add_cb_in0_wait_tiles", ctx.add_params["cb_in0_wait_tiles"]),
            ("add_cb_in1_wait_tiles", ctx.add_params["cb_in1_wait_tiles"]),
            ("add_slice_size_bytes", ctx.add_params["slice_size_bytes"]),
            # ReduceToOne compute args (CB indices)
            ("reduce_local_cb", ctx.reduce_local_cb),
            ("reduce_received_cb", ctx.reduce_received_cb),
            ("reduce_output_cb", ctx.reduce_output_cb),
            ("reduce_scratch_cb", ctx.reduce_scratch_cb),
        ]

        # ---- SRAM routed gate_proj CT args (always emitted; gated at runtime) ----
        # Default values are placeholders; the kernel's Op call is `if constexpr false`
        # no-op when sram_gate_proj_active=0. Real values come in when the caller
        # builds SRAM weights via build_sram_routed_proj_cts + setup_matmul_expert_sram.
        # Per-core values (fmt_l1_addr, base_addrs_l1_addr, k_offset) emitted via
        # _build_core_descriptors with default 0 → all-cores 0 by default.
        sgp = ctx.sram_gate_proj_params or {}
        sram_uniform_args = [
            # cb_in0 = rmsnorm-mcasted activation (already on shared gate cores).
            ("sram_gate_proj_cb_in0", ctx.gate_mm_input_cb),
            ("sram_gate_proj_cb_in1", ctx.sram_gate_proj_cb_in1),
            ("sram_gate_proj_cb_out", ctx.sram_gate_proj_out_cb),
            # Reuse the existing gate_proj index CB when routing is on; 0 otherwise.
            ("sram_gate_proj_cb_index", ctx.gate_proj_cb_index if ctx.enable_routing else 0),
            ("sram_gate_proj_num_tiles_k", sgp.get("num_tiles_k", 0)),
            ("sram_gate_proj_out_w", sgp.get("out_w", 0)),
            ("sram_gate_proj_cb_in0_num_pages", sgp.get("num_tiles_k", 0)),
            ("sram_gate_proj_num_active_experts", sgp.get("num_active_experts", 0)),
            # SRAM kernel reads indices from this L1 address directly (same
            # location as DRAM expert path — index_mcast lands here on all
            # receivers, including shared gate cores after we extended the
            # mcast set).
            (
                "sram_gate_proj_index_l1_addr",
                ctx.gate_proj_params.get("index_l1_addr", 0) if ctx.enable_routing else 0,
            ),
            ("sram_gate_proj_k_per_core", sgp.get("num_tiles_k", 0)),
            ("sram_gate_proj_meta_words_per_expert", sgp.get("meta_words_per_expert", 0)),
            ("sram_gate_proj_in0_page_size", sgp.get("in0_page_size", 0)),
            ("sram_gate_proj_accum_experts", sgp.get("accum_experts", 0)),
            ("sram_gate_proj_cb_out_sram", 0),
            # Shared by gate/up/down kernel CTArgs (use_compression). 1=BSPM
            # compressed_custom_mm path; 0=uniform BFP4 plain custom_mm path.
            ("sram_use_compression", ctx.sram_use_compression),
        ]
        ncrisc_named_compile_time_args += sram_uniform_args
        trisc_named_compile_time_args += sram_uniform_args

        # ---- SRAM gather (gate=A, up=B) CT args ----
        # NCRISC receiver: noc0_num_senders, semaphore (reuses AG_GATHER/BG_GATHER —
        #   SRAM gather runs sequentially before shared expert gather, semaphore resets at end),
        #   dst_cb, dst_num_pages.
        # BRISC sender:    dest_noc, data_size_bytes, semaphore, src_cb, src_num_pages,
        #   receiver_data_addr, num_experts, src_page_size, expert_dst_stride.
        # Per-core sender_idx is emitted via _build_core_descriptors below.
        _sram_total_cores = 64
        # NOTE: num_experts (sender) and dst_num_pages (receiver) are NOT CT args.
        # They're set at runtime from n_sram_active (computed in moe_kernel.cpp
        # right after index_mcast). The struct fields init to 0 and the gather
        # call site overwrites them via apply_n_sram_to_gather_args().
        sram_ag_ncrisc_args = [
            ("sram_ag_noc0_num_senders", _sram_total_cores if ctx.sram_gather_total_tiles else 0),
            ("sram_ag_noc0_receiver_semaphore_addr", ctx.sram_ag_receiver_semaphore_addr),
            ("sram_ag_dst_cb", ctx.sram_group1_cb),
        ]
        sram_bg_ncrisc_args = [
            ("sram_bg_noc0_num_senders", _sram_total_cores if ctx.sram_gather_total_tiles else 0),
            ("sram_bg_noc0_receiver_semaphore_addr", ctx.sram_bg_receiver_semaphore_addr),
            ("sram_bg_dst_cb", ctx.sram_group2_cb),
        ]
        # NOC physical coord of sender_core (gather destination). BRISC's
        # get_noc_addr expects physical, not logical, coords.
        _sender_noc = ctx.device.worker_core_from_logical_core(ctx.sender_core)
        sram_ag_brisc_args = [
            ("sram_ag_dest_noc_x", _sender_noc.x),
            ("sram_ag_dest_noc_y", _sender_noc.y),
            ("sram_ag_data_size_bytes", ctx.sram_gather_data_size_bytes),
            ("sram_ag_receiver_semaphore_addr", ctx.sram_ag_receiver_semaphore_addr),
            ("sram_ag_src_cb", ctx.sram_gate_proj_out_cb),
            (
                "sram_ag_src_num_pages",
                ctx.sram_gather_total_tiles // _sram_total_cores if ctx.sram_gather_total_tiles else 0,
            ),  # 8 pages per core (CB drain count)
            ("sram_ag_receiver_data_addr", ctx.sram_ag_receiver_data_addr),
            ("sram_ag_src_page_size", ctx.sram_gather_data_size_bytes),  # 64 bytes (1 tile)
            ("sram_ag_expert_dst_stride", ctx.sram_gather_expert_dst_stride),  # 4096 bytes
        ]
        sram_bg_brisc_args = [
            ("sram_bg_dest_noc_x", _sender_noc.x),
            ("sram_bg_dest_noc_y", _sender_noc.y),
            ("sram_bg_data_size_bytes", ctx.sram_gather_data_size_bytes),
            ("sram_bg_receiver_semaphore_addr", ctx.sram_bg_receiver_semaphore_addr),
            ("sram_bg_src_cb", ctx.sram_up_proj_out_cb),
            (
                "sram_bg_src_num_pages",
                ctx.sram_gather_total_tiles // _sram_total_cores if ctx.sram_gather_total_tiles else 0,
            ),
            ("sram_bg_receiver_data_addr", ctx.sram_bg_receiver_data_addr),
            ("sram_bg_src_page_size", ctx.sram_gather_data_size_bytes),
            ("sram_bg_expert_dst_stride", ctx.sram_gather_expert_dst_stride),
        ]
        ncrisc_named_compile_time_args += sram_ag_ncrisc_args + sram_bg_ncrisc_args
        brisc_named_compile_time_args += sram_ag_brisc_args + sram_bg_brisc_args

        # Runtime n_sram scan: each RISC reads cb_index post-index_mcast,
        # counts SRAM-flagged entries, uses that to drive gather num_experts /
        # dst_num_pages. All RISCs need access to index_l1_addr + active count.
        sgp_for_scan = ctx.sram_gate_proj_params or {}
        # Sender-core scan uses gate_output_indices_tensor's L1 address (the
        # gate kernel's output, computed in _setup_dimensions and stored on
        # ctx). Receivers scan kv_buf-overlaid gate_proj_cb_index populated
        # by mcast_index.
        sram_gather_common = [
            (
                "sram_gather_index_l1_addr",
                ctx.gate_proj_params.get("index_l1_addr", 0) if ctx.enable_routing else 0,
            ),
            (
                "sram_gather_sender_index_l1_addr",
                ctx.gate_proj_params.get("sender_index_l1_addr", 0) if ctx.enable_routing else 0,
            ),
            (
                "sram_gather_num_active_experts",
                sgp_for_scan.get("num_active_experts", 0),
            ),
            (
                "sram_gather_cb_index",
                ctx.gate_proj_cb_index if ctx.enable_routing else 0,
            ),
            # Face pages per active expert: num_senders / n_parallel = 64 / 8 = 8.
            # 8 sender writes (1×32 tiles) → 1 face (16×16). Used by gather call
            # to compute total_dst_pages = n_sram_active * sram_gather_pages_per_expert.
            ("sram_gather_pages_per_expert", 8),
            # Per-core sync sem for scan_n_sram_active (sender_scan + receiver_scan).
            # Used by sync_riscs_enter/exit so only BRISC waits on the index CB and
            # the other 4 RISCs sync via this sem.
            ("scan_sync_sem_addr", ctx.scan_sync_sem_addr),
        ]
        ncrisc_named_compile_time_args += sram_gather_common
        brisc_named_compile_time_args += sram_gather_common
        trisc_named_compile_time_args += sram_gather_common

        # SRAM extended GatedReduce CT args (TRISC compute on sender_core).
        sram_gated_reduce_args = [
            ("sram_gated_reduce_tiles_per_k", 8),  # 8 K-partial faces per output face
            ("sram_gated_reduce_group1_cb", ctx.sram_group1_cb),
            ("sram_gated_reduce_group2_cb", ctx.sram_group2_cb),
            ("sram_gated_reduce_intermed_cb", ctx.sram_intermed_cb),
            ("sram_gated_reduce_out_cb", ctx.sram_mcast_src_cb),
            ("sram_gated_reduce_scalar_cb", ctx.sram_gr_scalar_cb),
        ]
        trisc_named_compile_time_args += sram_gated_reduce_args

        # SRAM scalar copy on sender_core BRISC: reads top-K scores from
        # gate_output_scores_tensor's L1, writes one scalar per active expert
        # to sram_gr_scalar_cb (at byte 0 of each face tile). Disabled (cb=0)
        # in dense mode — TRISC's enable_scalar also goes off, taking the
        # silu(g1)*g2 path with no scale.
        sram_gr_scalar_brisc_args = [
            ("sram_gr_scalar_cb", ctx.sram_gr_scalar_cb if ctx.enable_routing else 0),
            (
                "sram_gr_scalar_src_l1_addr",
                ctx.gate_proj_params.get("scores_l1_addr", 0) if ctx.enable_routing else 0,
            ),
        ]
        brisc_named_compile_time_args += sram_gr_scalar_brisc_args

        # ---- SRAM routed up_proj CT args (mirror of gate_proj) ----
        sup = ctx.sram_up_proj_params or {}
        sram_up_uniform_args = [
            ("sram_up_proj_cb_in0", ctx.gate_mm_input_cb),
            ("sram_up_proj_cb_in1", ctx.sram_up_proj_cb_in1),
            ("sram_up_proj_cb_out", ctx.sram_up_proj_out_cb),
            ("sram_up_proj_cb_index", ctx.gate_proj_cb_index if ctx.enable_routing else 0),
            ("sram_up_proj_num_tiles_k", sup.get("num_tiles_k", 0)),
            ("sram_up_proj_out_w", sup.get("out_w", 0)),
            ("sram_up_proj_cb_in0_num_pages", sup.get("num_tiles_k", 0)),
            ("sram_up_proj_num_active_experts", sup.get("num_active_experts", 0)),
            (
                "sram_up_proj_index_l1_addr",
                ctx.gate_proj_params.get("index_l1_addr", 0) if ctx.enable_routing else 0,
            ),
            ("sram_up_proj_k_per_core", sup.get("num_tiles_k", 0)),
            ("sram_up_proj_meta_words_per_expert", sup.get("meta_words_per_expert", 0)),
            ("sram_up_proj_in0_page_size", sup.get("in0_page_size", 0)),
            ("sram_up_proj_accum_experts", sup.get("accum_experts", 0)),
            ("sram_up_proj_cb_out_sram", 0),
        ]
        ncrisc_named_compile_time_args += sram_up_uniform_args
        trisc_named_compile_time_args += sram_up_uniform_args

        # ---- SRAM routed down_proj CT args (accum_experts=1 + compact_in0=1) ----
        # Reads from sram_down_mcast_dst_cb (compact n_sram_active layout, indexed
        # by the running sram_idx — set compact_in0=1 so the kernel uses sram_idx
        # instead of exp_i for cb_in0 offsets).
        sdp = ctx.sram_down_proj_params or {}
        sram_down_uniform_args = [
            # cb_in0 = SRAM down mcast destination (already populated by sram_down_mcast).
            ("sram_down_proj_cb_in0", ctx.sram_down_mcast_dst_cb),
            ("sram_down_proj_cb_in1", ctx.sram_down_proj_cb_in1),
            ("sram_down_proj_cb_out", ctx.sram_down_proj_out_cb),
            ("sram_down_proj_cb_index", ctx.gate_proj_cb_index if ctx.enable_routing else 0),
            ("sram_down_proj_num_tiles_k", sdp.get("num_tiles_k", 0)),
            ("sram_down_proj_out_w", sdp.get("out_w", 0)),
            # cb_in0_num_pages here is just the per-expert K-tile count (kernel
            # multiplies by n_sram_active at runtime via compact_in0 path).
            ("sram_down_proj_cb_in0_num_pages", sdp.get("num_tiles_k", 0)),
            ("sram_down_proj_num_active_experts", sdp.get("num_active_experts", 0)),
            (
                "sram_down_proj_index_l1_addr",
                ctx.gate_proj_params.get("index_l1_addr", 0) if ctx.enable_routing else 0,
            ),
            ("sram_down_proj_k_per_core", sdp.get("num_tiles_k", 0)),
            ("sram_down_proj_meta_words_per_expert", sdp.get("meta_words_per_expert", 0)),
            ("sram_down_proj_in0_page_size", sdp.get("in0_page_size", 0)),
            ("sram_down_proj_accum_experts", sdp.get("accum_experts", 0)),
            ("sram_down_proj_cb_out_sram", 0),
            ("sram_down_proj_compact_in0", 1 if sdp.get("accum_experts", 0) else 0),
        ]
        ncrisc_named_compile_time_args += sram_down_uniform_args
        trisc_named_compile_time_args += sram_down_uniform_args
        return ncrisc_named_compile_time_args, brisc_named_compile_time_args, trisc_named_compile_time_args

    @staticmethod
    def _build_cb_descriptors(ctx):
        """Build circular buffer descriptors for routed expert."""
        descriptors = [
            ctx.rmsnorm_output_cb_descriptor,
            ctx.gate_mm_input_cb_descriptor,
        ]

        # Routing-only CB descriptors (gate_mm, gate, gate_proj_cb_index, mul_scalar)
        if ctx.enable_routing:
            descriptors += [
                ctx.gate_mm_params["weights_cb_descriptor"],
                ctx.gate_mm_params["output_cb_descriptor"],
                ctx.gate_params["input_cb_descriptor"],
                ctx.gate_params["bias_cb_descriptor"],
                ctx.gate_params["indices_cb_descriptor"],
                ctx.gate_params["output_cb_descriptor"],
                ctx.gate_params["output_indices_cb_descriptor"],
            ]

        descriptors += [
            ctx.gate_proj_params["cb_in1_descriptor"],  # Shared by gate_proj and up_proj
        ]

        # SRAM routed gate_proj cb_out descriptor — same on all devices.
        # cb_in1 descriptors are per-device (each coord has its own L1 layout)
        # and are appended in _setup_per_device_args.
        if ctx.sram_gate_proj_cb_out_descriptor is not None:
            descriptors.append(ctx.sram_gate_proj_cb_out_descriptor)
        if ctx.sram_up_proj_cb_out_descriptor is not None:
            descriptors.append(ctx.sram_up_proj_cb_out_descriptor)
        if ctx.sram_down_proj_cb_out_descriptor is not None:
            descriptors.append(ctx.sram_down_proj_cb_out_descriptor)
        # SRAM gather dst CBs (sender-only). Built in _overlap_cbs_with_sdpa_buffer
        # with face_tile (16x16) so GatedReduce reads each K-slice's 8 N-tiles as
        # one face (face_view: 8x faster than per-tile reduce on [1,32]).
        if ctx.sram_group1_cb_descriptor is not None:
            descriptors.append(ctx.sram_group1_cb_descriptor)
        if ctx.sram_group2_cb_descriptor is not None:
            descriptors.append(ctx.sram_group2_cb_descriptor)
        # SRAM extended GatedReduce intermediate CB (2 face tiles, sender_core).
        if ctx.sram_intermed_cb_descriptor is not None:
            descriptors.append(ctx.sram_intermed_cb_descriptor)
        # SRAM extended GatedReduce output CB (sender_core, n_active face tiles).
        if ctx.sram_mcast_src_cb_descriptor is not None:
            descriptors.append(ctx.sram_mcast_src_cb_descriptor)
        # SRAM GatedReduce scalar CB (sender_core, n_active face tiles).
        if ctx.sram_gr_scalar_cb_descriptor is not None:
            descriptors.append(ctx.sram_gr_scalar_cb_descriptor)
        # SRAM down mcast dst CB (kv_buf-overlaid on the 112 mcast receiver cores).
        if ctx.sram_down_mcast_dst_cb_descriptor is not None:
            descriptors.append(ctx.sram_down_mcast_dst_cb_descriptor)
        # Merged down output CB (kv_buf-overlaid on the 112 mcast receiver cores).
        if ctx.merged_down_out_cb_descriptor is not None:
            descriptors.append(ctx.merged_down_out_cb_descriptor)

        if ctx.enable_routing:
            descriptors.append(ctx.gate_proj_cb_index_descriptor)

        # cb_fmt CB descriptors (MatmulExpertCompressedDRAM per-expert fmt metadata)
        if ctx.gate_proj_cb_fmt_descriptor is not None:
            descriptors.append(ctx.gate_proj_cb_fmt_descriptor)
        if ctx.up_proj_cb_fmt_descriptor is not None:
            descriptors.append(ctx.up_proj_cb_fmt_descriptor)
        if ctx.down_proj_cb_fmt_descriptor is not None:
            descriptors.append(ctx.down_proj_cb_fmt_descriptor)

        descriptors += [
            ctx.gate_proj_params["cb_out_descriptor"],
            ctx.up_proj_params["cb_out_descriptor"],
            # mul_cb_in0/in1 reuse gate_proj/up_proj out CBs (already listed above).
            ctx.mul_params["cb_out_descriptor"],
        ]
        # Silu fast-path CB aliasing gate_proj_cb_out's L1 with [silu_tile_h, 32] tile
        # — only present in K-split mode (k_parallel_per_bank>1).
        if "cb_out_silu_descriptor" in ctx.gate_proj_params:
            descriptors.append(ctx.gate_proj_params["cb_out_silu_descriptor"])
        if ctx.enable_routing:
            descriptors += [
                ctx.mul_params["cb_scalar_src_descriptor"],
                ctx.mul_params["cb_scalar_descriptor"],
            ]

        descriptors += [
            ctx.down_proj_gather_params["dst_cb_descriptor"],
            ctx.down_proj_mcast_params["dst_cb_descriptor"],
            ctx.down_proj_params["cb_in1_descriptor"],
            ctx.down_proj_params["cb_out_descriptor"],
            ctx.add_params["cb_in0_descriptor"],
            ctx.add_params["cb_in1_descriptor"],
            ctx.add_params["cb_out_descriptor"],
            ctx.residual_mcast_src_cb_descriptor,
            ctx.residual_mcast_params["dst_cb_descriptor"],
            ctx.rmsnorm_gamma_cb_descriptor,
        ]

        # Reduce CBs (39-45)
        if ctx.enable_reduce_to_one and ctx.reduce_received_cb_descriptors:
            descriptors += ctx.reduce_received_cb_descriptors
            descriptors.append(ctx.reduce_scratch_cb_descriptor)
            descriptors.append(ctx.reduce_packet_cb_descriptor)

        # Bcast CB (46)
        if ctx.enable_bcast and ctx.bcast_pkt_cb_descriptor is not None:
            descriptors.append(ctx.bcast_pkt_cb_descriptor)

        return descriptors

    @staticmethod
    def _build_core_descriptors(ctx):
        """Build unified and per-core compile-time core descriptors for routed expert."""
        unified_compile_time_core_descriptors = [
            UnifiedCompileTimeCoreDescriptor(
                named_compile_time_arg="is_sender_core",
                core_range=ctx.sender_core,
                value=1,
                other_value=0,
            ),
            UnifiedCompileTimeCoreDescriptor(
                named_compile_time_arg="is_mcast_grid_core",
                core_range=ctx.mcast_worker_grid,
                value=1,
                other_value=0,
            ),
            UnifiedCompileTimeCoreDescriptor(
                named_compile_time_arg="is_gate_mm_core",
                core_range=ctx.gate_mm_params["core_grid"] if ctx.enable_routing else ttnn.CoreRangeSet([]),
                value=1,
                other_value=0,
            ),
            UnifiedCompileTimeCoreDescriptor(
                named_compile_time_arg="is_gate_proj_core",
                core_range=ctx.gate_proj_core_ranges,
                value=1,
                other_value=0,
            ),
            # gate_proj streamer cores: 16 cores when K-split is on for gate/up
            # (cores_per_dram_bank=2, k_parallel_per_bank=2), otherwise 8 (= primaries).
            # K-senders (= secondaries, k_slice_idx=0) MUST be active so they compute the
            # K-slice partial and NOC-write to the K-reducer (= primary, k_slice_idx=last).
            UnifiedCompileTimeCoreDescriptor(
                named_compile_time_arg="is_gate_proj_streamer_core",
                core_range=ttnn.CoreRangeSet(
                    [ttnn.CoreRange(c, c) for c in ctx.gate_proj_params["compute_cores_list"]]
                ),
                value=1,
                other_value=0,
            ),
            # down_proj streamer cores: 16 cores when primary_at_last_offset=True (8 primary
            # + 8 secondary, each bank's 2 cores split N), otherwise 8 (= primaries).
            # Senders (post-swap = secondaries) MUST be active so they NOC-write their
            # accum onto the receiver/primary; otherwise the receiver waits forever.
            UnifiedCompileTimeCoreDescriptor(
                named_compile_time_arg="is_down_proj_streamer_core",
                core_range=ttnn.CoreRangeSet(
                    [ttnn.CoreRange(c, c) for c in ctx.down_proj_params["compute_cores_list"]]
                ),
                value=1,
                other_value=0,
            ),
            # ReduceToOne core descriptors — will be updated per-device if enabled
            UnifiedCompileTimeCoreDescriptor(
                named_compile_time_arg="is_reduce_worker_core",
                core_range=ctx.gate_proj_core_ranges if ctx.enable_reduce_to_one else ttnn.CoreRangeSet([]),
                value=1,
                other_value=0,
            ),
            UnifiedCompileTimeCoreDescriptor(
                named_compile_time_arg="is_reduce_fabric_core",
                core_range=ttnn.CoreRangeSet([]),  # Set per-device
                value=1,
                other_value=0,
            ),
            UnifiedCompileTimeCoreDescriptor(
                named_compile_time_arg="reduce_persistent_fabric_signal_enable",
                core_range=ttnn.CoreRangeSet([]),  # Set per-device on ROOT1
                value=1,
                other_value=0,
            ),
        ]

        # bank_id / vc for gate/up/down are built in the per-proj loop below — that
        # source has 16-entry pcv lists when cores_per_bank=2 (K-split on gate/up, or
        # gather on down_proj). Static 8-entry descriptors here would shadow those and
        # collapse secondaries to other_value=0.
        per_core_compile_time_descriptors = [
            PerCoreCompileTimeDescriptor(
                named_compile_time_arg="down_proj_gather_sender_idx",
                core_values=ctx.sender_idx_core_values,
                other_value=0,
            ),
            PerCoreCompileTimeDescriptor(
                named_compile_time_arg="add_sender_index",
                core_values=ctx.add_params["sender_index_core_values"],
                other_value=0,
            ),
        ]

        # MatmulExpertCompressedDRAM per-core descriptors.
        # Values populated from first device's per-coord dict as a shared default;
        # `_setup_per_device_args` overwrites per-device for correctness on mesh
        # devices where addresses may differ.
        if "expert_offsets_l1_addr_per_device" in ctx.gate_proj_params:
            for proj in ("gate_proj", "up_proj", "down_proj"):
                params = getattr(ctx, f"{proj}_params")
                first_coord = next(iter(params["expert_offsets_l1_addr_per_device"]))
                expert_offsets_vals = params["expert_offsets_l1_addr_per_device"][first_coord]
                block_sizes_vals = params["block_sizes_l1_addr_per_device"][first_coord]
                pcv = params["per_core_values_per_device"][first_coord]
                per_core_compile_time_descriptors += [
                    PerCoreCompileTimeDescriptor(
                        named_compile_time_arg=f"{proj}_expert_offsets_l1_addr",
                        core_values=expert_offsets_vals,
                        other_value=0,
                    ),
                    PerCoreCompileTimeDescriptor(
                        named_compile_time_arg=f"{proj}_block_sizes_l1_addr",
                        core_values=block_sizes_vals,
                        other_value=0,
                    ),
                    # DRAM-only path: dram_fmt_l1_addr slot is unused (SRAM-only); pass 0.
                    PerCoreCompileTimeDescriptor(
                        named_compile_time_arg=f"{proj}_dram_fmt_l1_addr",
                        core_values=[],
                        other_value=0,
                    ),
                    PerCoreCompileTimeDescriptor(
                        named_compile_time_arg=f"{proj}_core_in_bank_idx",
                        core_values=pcv["core_in_bank_idx"],
                        other_value=0,
                    ),
                    # bank_id / vc per-proj — for down_proj (cores_per_bank=2 in
                    # gather mode) this is 16 entries; for gate/up (cores_per_bank=1)
                    # it is 8 entries. Overrides the static ctx.bank_id_core_values
                    # entries above which only had 8 entries (gate's primaries) and
                    # caused down_proj's secondaries to fall back to other_value=0.
                    PerCoreCompileTimeDescriptor(
                        named_compile_time_arg=f"{proj}_bank_id",
                        core_values=pcv["bank_id"],
                        other_value=0,
                    ),
                    PerCoreCompileTimeDescriptor(
                        named_compile_time_arg=f"{proj}_vc",
                        core_values=pcv["vc"],
                        other_value=0,
                    ),
                    PerCoreCompileTimeDescriptor(
                        named_compile_time_arg=f"{proj}_next_core_noc_x",
                        core_values=pcv["next_core_noc_x"],
                        other_value=0,
                    ),
                    PerCoreCompileTimeDescriptor(
                        named_compile_time_arg=f"{proj}_next_core_noc_y",
                        core_values=pcv["next_core_noc_y"],
                        other_value=0,
                    ),
                    PerCoreCompileTimeDescriptor(
                        named_compile_time_arg=f"{proj}_k_slice_idx",
                        core_values=pcv["k_slice_idx"],
                        other_value=0,
                    ),
                ]

        # ---- SRAM gate_proj per-core CT args (always emitted; default 0) ----
        # Per-core values for fmt_l1_addr, base_addrs_l1_addr, k_offset are
        # populated when sram_gate_proj_params is non-empty. Otherwise empty
        # core_values list with other_value=0 → all cores see 0 → kernel's
        # `if constexpr (sram_gate_proj_active==1)` skips the path entirely.
        sgp = getattr(ctx, "sram_gate_proj_params", None) or {}
        # Per-device dicts (when present) provide the first-coord defaults here;
        # _setup_per_device_args overrides per coord.
        first_coord_fmt = []
        first_coord_base = []
        if sgp.get("num_sram_experts", 0) > 0:
            fmt_per_dev = sgp.get("sram_fmt_l1_addr_per_device", {}) or {}
            base_per_dev = sgp.get("sram_base_addrs_l1_addr_per_device", {}) or {}
            if fmt_per_dev:
                first_coord_fmt = fmt_per_dev[next(iter(fmt_per_dev))]
            if base_per_dev:
                first_coord_base = base_per_dev[next(iter(base_per_dev))]
        sram_k_offsets = sgp.get("sram_k_offsets") or []

        per_core_compile_time_descriptors += [
            PerCoreCompileTimeDescriptor(
                named_compile_time_arg="sram_gate_proj_fmt_l1_addr",
                core_values=first_coord_fmt,
                other_value=0,
            ),
            PerCoreCompileTimeDescriptor(
                named_compile_time_arg="sram_gate_proj_base_addrs_l1_addr",
                core_values=first_coord_base,
                other_value=0,
            ),
            PerCoreCompileTimeDescriptor(
                named_compile_time_arg="sram_gate_proj_k_offset",
                core_values=sram_k_offsets,
                other_value=0,
            ),
        ]

        # ---- SRAM up_proj per-core CT args (mirror of gate_proj) ----
        sup = getattr(ctx, "sram_up_proj_params", None) or {}
        first_coord_fmt_up = []
        first_coord_base_up = []
        if sup.get("num_sram_experts", 0) > 0:
            fmt_per_dev_up = sup.get("sram_fmt_l1_addr_per_device", {}) or {}
            base_per_dev_up = sup.get("sram_base_addrs_l1_addr_per_device", {}) or {}
            if fmt_per_dev_up:
                first_coord_fmt_up = fmt_per_dev_up[next(iter(fmt_per_dev_up))]
            if base_per_dev_up:
                first_coord_base_up = base_per_dev_up[next(iter(base_per_dev_up))]
        sram_k_offsets_up = sup.get("sram_k_offsets") or []

        per_core_compile_time_descriptors += [
            PerCoreCompileTimeDescriptor(
                named_compile_time_arg="sram_up_proj_fmt_l1_addr",
                core_values=first_coord_fmt_up,
                other_value=0,
            ),
            PerCoreCompileTimeDescriptor(
                named_compile_time_arg="sram_up_proj_base_addrs_l1_addr",
                core_values=first_coord_base_up,
                other_value=0,
            ),
            PerCoreCompileTimeDescriptor(
                named_compile_time_arg="sram_up_proj_k_offset",
                core_values=sram_k_offsets_up,
                other_value=0,
            ),
        ]

        # ---- SRAM down_proj per-core CT args (mirror of gate_proj) ----
        sdp_pc = getattr(ctx, "sram_down_proj_params", None) or {}
        first_coord_fmt_dn = []
        first_coord_base_dn = []
        if sdp_pc.get("num_sram_experts", 0) > 0:
            fmt_per_dev_dn = sdp_pc.get("sram_fmt_l1_addr_per_device", {}) or {}
            base_per_dev_dn = sdp_pc.get("sram_base_addrs_l1_addr_per_device", {}) or {}
            if fmt_per_dev_dn:
                first_coord_fmt_dn = fmt_per_dev_dn[next(iter(fmt_per_dev_dn))]
            if base_per_dev_dn:
                first_coord_base_dn = base_per_dev_dn[next(iter(base_per_dev_dn))]
        sram_k_offsets_dn = sdp_pc.get("sram_k_offsets") or []

        per_core_compile_time_descriptors += [
            PerCoreCompileTimeDescriptor(
                named_compile_time_arg="sram_down_proj_fmt_l1_addr",
                core_values=first_coord_fmt_dn,
                other_value=0,
            ),
            PerCoreCompileTimeDescriptor(
                named_compile_time_arg="sram_down_proj_base_addrs_l1_addr",
                core_values=first_coord_base_dn,
                other_value=0,
            ),
            PerCoreCompileTimeDescriptor(
                named_compile_time_arg="sram_down_proj_k_offset",
                core_values=sram_k_offsets_dn,
                other_value=0,
            ),
        ]

        # ---- SRAM gather per-core sender_idx (gate=A, up=B) ----
        # Each of the 64 a_cores (b_cores) has a sender_idx telling MoeGather
        # where to place its tile within an expert's 64-tile slab on the
        # sender. Layout matches GatedReduce's contiguous-K-partials read.
        sram_ag_idx = ctx.sram_ag_sender_idx_core_values or []
        sram_bg_idx = ctx.sram_bg_sender_idx_core_values or []
        per_core_compile_time_descriptors += [
            PerCoreCompileTimeDescriptor(
                named_compile_time_arg="sram_ag_sender_idx",
                core_values=sram_ag_idx,
                other_value=0,
            ),
            PerCoreCompileTimeDescriptor(
                named_compile_time_arg="sram_bg_sender_idx",
                core_values=sram_bg_idx,
                other_value=0,
            ),
        ]
        return unified_compile_time_core_descriptors, per_core_compile_time_descriptors


class MoeSharedExpertOp:
    """
    Shared expert setup for the fused MoE kernel.

    Provides context setup and build methods for shared expert components
    (Gate/Up KN-sliced matmul, etc.). Does not execute on its own;
    used by MoeOp to compose the fused kernel.
    """

    # ========================================================================
    # Setup APIs
    # ========================================================================

    @staticmethod
    def setup_kn_sliced_matmul(
        weights_tensor,
        cb_weights_index,
        cb_out_index,
        num_tiles_k,
        k_parallel,
        n_parallel,
    ):
        """
        Setup KN-sliced matmul: [1, K] x [K, N] distributed across K*N cores.

        Each core holds a shard of shape [K/k_parallel, N/n_parallel] of the weight matrix.
        Each core computes: activation[k_offset:k_offset+k_per_core] @ weights_shard → 1 output tile.
        CB output descriptor is created by the overlap function.

        Args:
            weights_tensor: Gate/Up weights tensor (HEIGHT_SHARDED on K*N compute cores)
            cb_weights_index: CB index for weights
            cb_out_index: CB index for output
            num_tiles_k: Total K dimension in tiles
            k_parallel: K parallelism factor (number of K slices)
            n_parallel: N parallelism factor (number of N slices)

        Returns:
            dict with:
            - k_per_core: K tiles per core
            - act_total_tiles: Total activation tiles (= num_tiles_k)
            - weights_num_pages: Pages in weights CB per core (= k_per_core)
            - cb_weights_descriptor: CB descriptor for weights (tensor-backed)
            - cb_out_descriptor: None (set by overlap function)
        """
        k_per_core = num_tiles_k // k_parallel
        act_total_tiles = num_tiles_k
        weights_num_pages = k_per_core

        return {
            "k_per_core": k_per_core,
            "act_total_tiles": act_total_tiles,
            "weights_num_pages": weights_num_pages,
            "cb_weights_descriptor": None,
            "weights_cb_addr": _fused_base_addr(weights_tensor),
            "cb_out_descriptor": None,
        }

    @staticmethod
    def setup_gated_reduce(
        group1_tensor,
        group2_tensor,
        intermed_tensor,
        output_tensor,
        cb_group1_index,
        cb_group2_index,
        cb_intermed_index,
        cb_out_index,
        input_tile,
        data_format,
        k_parallel,
        n_parallel,
    ):
        """
        Setup gated reduce: SiLU(reduce(group1)) * reduce(group2).

        Requires face-view optimization (asserted). Creates CB descriptors
        with face-view tile aliasing.

        Args:
            group1_tensor: Gate gather destination tensor (tensor-backed on sender)
            group2_tensor: Up gather destination tensor (tensor-backed on sender)
            intermed_tensor: Intermediate tensor for reduce (tensor-backed on sender)
            output_tensor: Gated reduce output / mcast source tensor (tensor-backed on sender)
            cb_group1_index: CB index for gate gather destination
            cb_group2_index: CB index for up gather destination
            cb_intermed_index: CB index for intermediate
            cb_out_index: CB index for gated reduce output
            input_tile: Original tile format (e.g., Tile([1, 32]))
            data_format: Data format (dtype)
            k_parallel: K parallelism factor (= tiles_per_k)
            n_parallel: N parallelism factor (= k_num_tiles)

        Returns:
            dict with:
            - tiles_per_k, k_num_tiles: Parallelism dimensions
            - kernel_tiles_per_k, kernel_k_num_tiles: Kernel-level tile counts (face-view)
            - mcast_src_num_pages: Pages in mcast source CB (1 with face-view)
            - face_tile_desc, face_tile_size: Face tile descriptor and size
            - input_tile_size, reduce_tile_size: Tile sizes
            - cb_group1_descriptor, cb_group2_descriptor: CB descriptors (face-view aliased)
            - cb_intermed_descriptor, cb_out_descriptor: CB descriptors
        """
        tiles_per_k = k_parallel
        k_num_tiles = n_parallel

        tile_h, tile_w = input_tile.tile_shape
        input_tile_size = input_tile.get_tile_size(data_format)
        assert can_use_face_view(tile_h, tile_w, tiles_per_k, k_num_tiles), (
            f"Face-view optimization is required for shared expert gated reduce "
            f"(tile={tile_h}x{tile_w}, tiles_per_k={tiles_per_k}, k_num_tiles={k_num_tiles})"
        )

        face_tile = ttnn.Tile([FACE_HEIGHT, FACE_WIDTH])
        face_tile_desc = ttnn.TileDescriptor(FACE_HEIGHT, FACE_WIDTH, False)
        face_tile_size = face_tile.get_tile_size(data_format)
        kernel_tiles_per_k = tiles_per_k
        kernel_k_num_tiles = 1
        mcast_src_num_pages = 1
        reduce_tile_size = face_tile_size

        if cb_group1_index == cb_group2_index:
            assert kernel_k_num_tiles == 1, "kernel_k_num_tiles must be 1 if cb_group1_index == cb_group2_index"

        cb_group1_descriptor = None
        if group1_tensor is not None:
            cb_group1_descriptor = ttnn.cb_descriptor_from_sharded_tensor(cb_group1_index, group1_tensor)
            cb_group1_descriptor.format_descriptors[0].tile = face_tile_desc
            cb_group1_descriptor.format_descriptors[0].page_size = face_tile_size

        cb_group2_descriptor = None
        if group2_tensor is not None:
            cb_group2_descriptor = ttnn.cb_descriptor_from_sharded_tensor(cb_group2_index, group2_tensor)
            cb_group2_descriptor.format_descriptors[0].tile = face_tile_desc
            cb_group2_descriptor.format_descriptors[0].page_size = face_tile_size

        cb_intermed_descriptor = None
        if intermed_tensor is not None:
            cb_intermed_descriptor = ttnn.cb_descriptor_from_sharded_tensor(cb_intermed_index, intermed_tensor)
        cb_out_descriptor = None
        if output_tensor is not None:
            cb_out_descriptor = ttnn.cb_descriptor_from_sharded_tensor(cb_out_index, output_tensor)

        return {
            "tiles_per_k": tiles_per_k,
            "k_num_tiles": k_num_tiles,
            "kernel_tiles_per_k": kernel_tiles_per_k,
            "kernel_k_num_tiles": kernel_k_num_tiles,
            "mcast_src_num_pages": mcast_src_num_pages,
            "face_tile_desc": face_tile_desc,
            "face_tile_size": face_tile_size,
            "input_tile_size": input_tile_size,
            "reduce_tile_size": reduce_tile_size,
            "cb_group1_descriptor": cb_group1_descriptor,
            "cb_group2_descriptor": cb_group2_descriptor,
            "cb_intermed_descriptor": cb_intermed_descriptor,
            "cb_out_descriptor": cb_out_descriptor,
        }

    @staticmethod
    def setup_residual_add(
        cb_out_index,
        num_matmul_cores,
        out_w_per_core,
    ):
        """
        Setup residual add: matmul_out + residual → residual_add_out on matmul cores.

        A simple element-wise add where both operands are already present on
        each core (matmul output and mcasted residual).
        CB descriptor is created in _overlap_cbs_with_sdpa_buffer.

        Args:
            cb_out_index: CB index for output
            num_matmul_cores: Number of matmul cores performing the add
            out_w_per_core: Output width per core in tiles

        Returns:
            dict with:
            - out_cb: Output CB index
            - total_in1_tiles: Total residual tiles across all cores
            - cb_out_descriptor: None (set by overlap function)
        """
        return {
            "out_cb": cb_out_index,
            "total_in1_tiles": num_matmul_cores * out_w_per_core,
            "cb_out_descriptor": None,
        }

    @staticmethod
    def setup_kn_per_core_values(
        a_cores_list,
        b_cores_list,
        k_per_core,
        n_parallel,
    ):
        """
        Compute per-core values for KN-sliced matmul + gather.

        For each compute core, determines:
        - k_offset: Starting K-tile index for the matmul slice
        - sender_idx: Index into the gather destination CB (face-view layout)

        Args:
            a_cores_list: Gate (group A) compute cores
            b_cores_list: Up (group B) compute cores
            k_per_core: K tiles per core (from setup_kn_sliced_matmul)
            n_parallel: N parallelism factor

        Returns:
            dict with:
            - k_offset_core_values: [(core, k_offset), ...] for A and B cores
            - ag_sender_idx_core_values: [(core, sender_idx), ...] for A cores
            - bg_sender_idx_core_values: [(core, sender_idx), ...] for B cores
        """
        # K-offset for matmul: each row of k_parallel cores shares the same k_offset
        k_offset_core_values = [(core, (i // n_parallel) * k_per_core) for i, core in enumerate(a_cores_list)] + [
            (core, (i // n_parallel) * k_per_core) for i, core in enumerate(b_cores_list)
        ]

        # Gather sender indices (face-view layout: row-major in [k_parallel, n_parallel])
        def _sender_indices(cores_list):
            return [
                (core, (lid // n_parallel) * n_parallel + (lid % n_parallel)) for lid, core in enumerate(cores_list)
            ]

        return {
            "k_offset_core_values": k_offset_core_values,
            "ag_sender_idx_core_values": _sender_indices(a_cores_list),
            "bg_sender_idx_core_values": _sender_indices(b_cores_list),
        }

    @staticmethod
    def setup_output_mcast(
        gather_dst_num_pages,
        tile_size,
        dst_num_pages,
    ):
        """
        Setup output mcast: sender → all cores, DRAM cores receive shared expert output.

        The sender multicasts the full gathered output. DRAM cores receive it
        into their eltwise add input CB (add_cb_in1) which has a different
        page size than the source CB.

        src_num_pages and dst_num_pages differ because the source CB uses
        1×32 tile-sized pages while the destination CB (add_cb_in1) uses
        slice-sized pages (width_per_core elements each). The total bytes
        are the same: gather_dst_num_pages * tile_size == dst_num_pages * slice_size.

        Args:
            gather_dst_num_pages: Total pages in the gather destination (= num_matmul_cores * out_w)
            tile_size: Size of one tile in bytes (1×32 tile)
            dst_num_pages: Pages in the destination CB (= total_width / width_per_core)

        Returns:
            dict with:
            - data_size_bytes: Total mcast data size
            - src_num_pages: Pages to read from source CB (tile-sized pages)
            - dst_num_pages: Pages each receiver pushes (slice-sized pages)
        """
        return {
            "data_size_bytes": gather_dst_num_pages * tile_size,
            "src_num_pages": gather_dst_num_pages,
            "dst_num_pages": dst_num_pages,
        }

    @staticmethod
    def _setup_dimensions(
        device,
        shared_gate_weights_overlapped,
        shared_up_weights_overlapped,
        shared_down_weights_tensor,
        num_tiles_k,
        tile_1x32_size,
        data_format,
        input_tile,
        input_tile_size,
        sender_core,
        sender_core_grid,
        mcast_grid,
        num_dram_worker_cores,
        k_parallel=8,
        n_parallel=8,
        shared_output_tensor=None,  # Deprecated: CB 38 now backed by sdpa_out_interm_buffer
        # Semaphore IDs (overridable, defaults match MoeOp layout)
        ag_receiver_semaphore_addr=2,
        bg_receiver_semaphore_addr=3,
        ag_noc1_receiver_semaphore_addr=2,
        bg_noc1_receiver_semaphore_addr=3,
        shared_mcast_sender_semaphore_addr=0,
        shared_mcast_receiver_semaphore_addr=3,
        output_gather_noc0_receiver_semaphore_addr=4,
        output_gather_noc1_receiver_semaphore_addr=4,
        output_mcast_sender_semaphore_addr=0,
        output_mcast_receiver_semaphore_addr=4,
        cb_id_context=None,
        residual_mcast_dst_cb=None,
    ):
        """
        Compute shared expert dimensions and build _MoeSharedExpertContext.

        Args:
            device: TT device (single chip)
            shared_gate_weights_overlapped: Gate proj OverlappedTensor (shares fused backing tensor with up)
            shared_up_weights_overlapped: Up proj OverlappedTensor (shares fused backing tensor with gate)
            shared_down_weights_tensor: Down projection weights tensor
            shared_output_tensor: Output tensor for shared expert
            num_tiles_k: Number of K tiles (from routed expert context)
            tile_1x32_size: Size of a 1x32 tile in bytes
            data_format: Data format (dtype)
            input_tile: Input tile format (e.g., Tile((1, 32)))
            input_tile_size: Size of input tile in bytes
            sender_core: The mcast/gather core (e.g., CoreCoord(12, 9))
            sender_core_grid: CoreRangeSet for sender core
            mcast_grid: CoreRangeSet for mcast destination grid (same as routed input mcast)
            k_parallel: K parallelism factor
            n_parallel: N parallelism factor
            ag_receiver_semaphore_addr: Gate gather NOC0 receiver sem address
            bg_receiver_semaphore_addr: Up gather NOC0 receiver sem address
            ag_noc1_receiver_semaphore_addr: Gate gather NOC1 receiver sem address
            bg_noc1_receiver_semaphore_addr: Up gather NOC1 receiver sem address
            shared_mcast_sender_semaphore_addr: Shared mcast sender sem address
            shared_mcast_receiver_semaphore_addr: Shared mcast receiver sem address
            output_gather_noc0_receiver_semaphore_addr: Output gather NOC0 receiver sem address
            output_gather_noc1_receiver_semaphore_addr: Output gather NOC1 receiver sem address
            output_mcast_sender_semaphore_addr: Output mcast sender sem address
            output_mcast_receiver_semaphore_addr: Output mcast receiver sem address

        Returns:
            _MoeSharedExpertContext
        """

        # ==================================================================
        # CB indices (auto-assigned via cb_id_context)
        # ==================================================================
        assert cb_id_context is not None, "cb_id_context must be provided"
        TD_1x32 = ttnn.TileDescriptor(ttnn.Tile((1, 32)))
        TD_16x16 = ttnn.TileDescriptor(ttnn.Tile((FACE_HEIGHT, FACE_WIDTH)))
        # 8x32 tile descriptor: byte-equivalent to 16x16 face (256 elements; 512 B for bf16).
        # Used for shared GR CBs to enable cross-context auto-reuse with attention's 8x32
        # SDPA output IDs (same trick as SRAM GR; see line 1474).
        TD_8x32 = ttnn.TileDescriptor(ttnn.Tile((8, 32)))

        # 1x32, bfloat16
        shared_down_mcast_dst_cb = cb_id_context.get_cb_id(data_format, TD_1x32)
        shared_down_matmul_out_cb = cb_id_context.get_cb_id(data_format, TD_1x32)
        shared_residual_add_out_cb = cb_id_context.get_cb_id(data_format, TD_1x32)
        shared_output_gather_dst_cb = cb_id_context.get_cb_id(data_format, TD_1x32)
        # shared_gu_out_cb (a/b shared compute, 128 cores) shares CB ID with
        # shared_output_gather_dst_cb (sender, 1 core). Strict-disjoint: sender
        # is excluded from a/b cores by build. Saves 1 in (bf16, 1×32).
        shared_gu_out_cb = shared_output_gather_dst_cb

        # 8x32, bfloat16 (reinterpreted from 16x16 face for cross-context reuse).
        # Pages are 256 elements / 512 B either way; the GR kernel does element-wise
        # add_tiles, which is shape-agnostic. Allocating with TD_8x32 lets these auto-reuse
        # attention's 8x32 SDPA output IDs, dropping out of the (otherwise sender-bound)
        # 16x16 bucket. Runtime cb_descriptor still uses face_tile_desc (16x16) — that
        # only governs page_size at reconfig time (matches anyway), not the build tile.
        shared_group1_cb = cb_id_context.get_cb_id(data_format, TD_8x32)
        shared_group2_cb = shared_group1_cb
        shared_intermed_cb = cb_id_context.get_cb_id(data_format, TD_8x32)
        shared_mcast_src_cb = cb_id_context.get_cb_id(data_format, TD_8x32)

        # Shared residual reuses the routed expert's residual_mcast_dst CB when provided,
        # otherwise allocates its own.
        shared_residual_cb = (
            residual_mcast_dst_cb
            if residual_mcast_dst_cb is not None
            else cb_id_context.get_cb_id(data_format, TD_1x32)
        )

        # Tensor-backed CBs (format from weight tensors)
        shared_gate_up_weights_tensor = shared_gate_weights_overlapped.fused_tensor
        shared_down_matmul_in1_cb = cb_id_context.get_cb_id(
            shared_down_weights_tensor.dtype, ttnn.TileDescriptor(shared_down_weights_tensor.get_tile())
        )
        shared_gu_weights_cb = shared_down_matmul_in1_cb
        shared_weights_core_ranges = shared_down_weights_tensor.memory_config().shard_spec.grid.merge(
            shared_gate_up_weights_tensor.memory_config().shard_spec.grid
        )

        # ==================================================================
        # Dimensions
        # ==================================================================
        num_compute_cores = 64  # per branch (gate/up)
        assert k_parallel * n_parallel == num_compute_cores
        total_gather_tiles = num_compute_cores  # 64
        gu_gather_data_size_bytes = input_tile_size  # each compute core sends 1 tile

        # ==================================================================
        # Core grids — from OverlappedTensors
        # ==================================================================
        shared_gate_up_weights_tensor = shared_gate_weights_overlapped.fused_tensor
        a_compute_grid = shared_gate_weights_overlapped.core_range_set
        b_compute_grid = shared_up_weights_overlapped.core_range_set
        compute_core_grid = shared_gate_up_weights_tensor.memory_config().shard_spec.grid

        # Per-core values (k_offset, sender_idx) depend on row-major ordering
        # that must match the weight data layout produced by BDW's
        # _crs_shard_permutation.  The OverlappedTensor CRS enumeration order
        # differs from row-major, so we keep build_ab_grids() for the ordered
        # lists while using the OverlappedTensor grids for CoreRangeSet membership.
        from models.demos.deepseek_v3_b1.fused_ops.shared_expert.op import SharedExpertOp

        a_cores_list, b_cores_list = SharedExpertOp.build_ab_grids()
        assert len(a_cores_list) == num_compute_cores
        assert len(b_cores_list) == num_compute_cores
        assert set((c.x, c.y) for c in a_cores_list) == set(
            (c.x, c.y) for c in ttnn.corerange_to_cores(a_compute_grid)
        ), "Gate cores from OverlappedTensor don't match build_ab_grids()"
        assert set((c.x, c.y) for c in b_cores_list) == set(
            (c.x, c.y) for c in ttnn.corerange_to_cores(b_compute_grid)
        ), "Up cores from OverlappedTensor don't match build_ab_grids()"

        # NOC coordinates for gather destination (sender core)
        gather_dest_noc_core = device.worker_core_from_logical_core(sender_core)

        # ==================================================================
        # Gather destination addresses (set by _overlap_cbs_with_sdpa_buffer)
        # ==================================================================
        ag_receiver_data_addr = 0
        bg_receiver_data_addr = 0

        # ==================================================================
        # Setup APIs — KN-sliced matmul and gated reduce
        # ==================================================================
        gu_matmul_params = MoeSharedExpertOp.setup_kn_sliced_matmul(
            weights_tensor=shared_gate_up_weights_tensor,
            cb_weights_index=shared_gu_weights_cb,
            cb_out_index=shared_gu_out_cb,
            num_tiles_k=num_tiles_k,
            k_parallel=k_parallel,
            n_parallel=n_parallel,
        )

        gated_reduce_params = MoeSharedExpertOp.setup_gated_reduce(
            group1_tensor=None,
            group2_tensor=None,
            intermed_tensor=None,
            output_tensor=None,
            cb_group1_index=shared_group1_cb,
            cb_group2_index=shared_group2_cb,
            cb_intermed_index=shared_intermed_cb,
            cb_out_index=shared_mcast_src_cb,
            input_tile=input_tile,
            data_format=data_format,
            k_parallel=k_parallel,
            n_parallel=n_parallel,
        )

        # ==================================================================
        # Down Mcast (gated reduce output → all 130 cores for down proj)
        # ==================================================================
        mcast_src_num_pages = gated_reduce_params["mcast_src_num_pages"]
        reduce_tile_size = gated_reduce_params["reduce_tile_size"]
        down_mcast_data_size_bytes = mcast_src_num_pages * reduce_tile_size

        down_mcast_params = MoeOp.setup_mcast(
            device=device,
            sender_core=sender_core,
            mcast_grid=mcast_grid,
            src_cb=shared_mcast_src_cb,
            src_tensor=None,
            dst_cb=shared_down_mcast_dst_cb,
            dst_tensor=None,
            sender_semaphore_addr=shared_mcast_sender_semaphore_addr,
            receiver_semaphore_addr=shared_mcast_receiver_semaphore_addr,
            data_size_bytes=down_mcast_data_size_bytes,
            src_num_pages=mcast_src_num_pages,
        )

        # ==================================================================
        # Down Proj Matmul (SRAM matmul on non-DRAM cores)
        # ==================================================================
        matmul_core_grid = shared_down_weights_tensor.memory_config().shard_spec.grid
        num_matmul_cores = matmul_core_grid.num_cores()

        from models.demos.deepseek_v3_b1.fused_ops.down_proj.op import DownProj

        assert (
            num_matmul_cores == DownProj.NUM_MATMUL_CORES
        ), f"Down weights tensor has {num_matmul_cores} cores, expected {DownProj.NUM_MATMUL_CORES}"
        assert set((c.x, c.y) for c in ttnn.corerange_to_cores(matmul_core_grid)) == set(
            (c.x, c.y) for c in ttnn.corerange_to_cores(DownProj.build_matmul_core_grid())
        ), "Down weights matmul cores from tensor don't match DownProj.build_matmul_core_grid()"

        down_matmul_params = MoeOp.setup_sram_matmul(
            in0_cb=shared_down_mcast_dst_cb,
            in1_cb=shared_down_matmul_in1_cb,
            out_cb=shared_down_matmul_out_cb,
            weights_overlapped=shared_down_weights_tensor,
            k_num_tiles=n_parallel,
            weights_core_ranges_override=shared_weights_core_ranges,
        )

        # ==================================================================
        # Residual Add (matmul_out + residual → residual_add_out on matmul cores)
        # ==================================================================
        residual_add_params = MoeSharedExpertOp.setup_residual_add(
            cb_out_index=shared_residual_add_out_cb,
            num_matmul_cores=num_matmul_cores,
            out_w_per_core=down_matmul_params["out_w"],
        )

        # ==================================================================
        # Output Gather (matmul cores → sender)
        # ==================================================================
        output_gather_dst_num_pages = num_matmul_cores * down_matmul_params["out_w"]
        output_gather_params = MoeOp.setup_gather(
            device=device,
            receiver_core=sender_core,
            sender_core_ranges=matmul_core_grid,
            num_senders=num_matmul_cores,
            data_size_bytes_per_sender=down_matmul_params["out_w"] * tile_1x32_size,
            src_cb=shared_residual_add_out_cb,
            src_num_pages=down_matmul_params["out_w"],
            dst_cb=shared_output_gather_dst_cb,
            noc0_receiver_semaphore_addr=output_gather_noc0_receiver_semaphore_addr,
            noc1_receiver_semaphore_addr=output_gather_noc1_receiver_semaphore_addr,
            dst_num_pages=output_gather_dst_num_pages,
        )

        # ==================================================================
        # Output Mcast (sender → all cores, DRAM cores receive into add_cb_in1)
        # ==================================================================
        output_mcast_params = MoeSharedExpertOp.setup_output_mcast(
            gather_dst_num_pages=output_gather_params["dst_num_pages"],
            tile_size=tile_1x32_size,
            dst_num_pages=num_dram_worker_cores,
        )

        # ==================================================================
        # Per-core values
        # ==================================================================
        per_core_values = MoeSharedExpertOp.setup_kn_per_core_values(
            a_cores_list=a_cores_list,
            b_cores_list=b_cores_list,
            k_per_core=gu_matmul_params["k_per_core"],
            n_parallel=n_parallel,
        )
        gu_k_offset_core_values = per_core_values["k_offset_core_values"]
        ag_sender_idx_core_values = per_core_values["ag_sender_idx_core_values"]
        bg_sender_idx_core_values = per_core_values["bg_sender_idx_core_values"]

        return _MoeSharedExpertContext(
            # Core grids
            compute_core_grid=compute_core_grid,
            a_compute_grid=a_compute_grid,
            b_compute_grid=b_compute_grid,
            a_cores_list=a_cores_list,
            b_cores_list=b_cores_list,
            matmul_core_grid=matmul_core_grid,
            # CB indices
            gu_weights_cb=shared_gu_weights_cb,
            gu_out_cb=shared_gu_out_cb,
            group1_cb=shared_group1_cb,
            group2_cb=shared_group2_cb,
            intermed_cb=shared_intermed_cb,
            mcast_src_cb=shared_mcast_src_cb,
            residual_cb=shared_residual_cb,
            down_mcast_dst_cb=shared_down_mcast_dst_cb,
            # Parallelism
            n_parallel=n_parallel,
            k_parallel=k_parallel,
            num_compute_cores=num_compute_cores,
            # Gather params
            gather_dest_noc_core=gather_dest_noc_core,
            gu_gather_data_size_bytes=gu_gather_data_size_bytes,
            total_gather_tiles=total_gather_tiles,
            # Semaphore IDs
            ag_receiver_semaphore_addr=ag_receiver_semaphore_addr,
            bg_receiver_semaphore_addr=bg_receiver_semaphore_addr,
            ag_noc1_receiver_semaphore_addr=ag_noc1_receiver_semaphore_addr,
            bg_noc1_receiver_semaphore_addr=bg_noc1_receiver_semaphore_addr,
            shared_mcast_sender_semaphore_addr=shared_mcast_sender_semaphore_addr,
            shared_mcast_receiver_semaphore_addr=shared_mcast_receiver_semaphore_addr,
            output_gather_noc0_receiver_semaphore_addr=output_gather_noc0_receiver_semaphore_addr,
            output_gather_noc1_receiver_semaphore_addr=output_gather_noc1_receiver_semaphore_addr,
            output_mcast_sender_semaphore_addr=output_mcast_sender_semaphore_addr,
            output_mcast_receiver_semaphore_addr=output_mcast_receiver_semaphore_addr,
            # Keep-alive tensors
            ag_dummy_tensor=None,
            bg_dummy_tensor=None,
            ag_receiver_data_addr=ag_receiver_data_addr,
            bg_receiver_data_addr=bg_receiver_data_addr,
            down_mcast_dst_dummy_tensor=None,
            output_gather_dst_dummy_tensor=None,  # CB 38 backed by sdpa_out_interm_buffer
            # Setup result dicts
            gu_matmul_params=gu_matmul_params,
            gated_reduce_params=gated_reduce_params,
            down_mcast_params=down_mcast_params,
            down_matmul_params=down_matmul_params,
            residual_add_params=residual_add_params,
            output_gather_params=output_gather_params,
            output_mcast_params=output_mcast_params,
            # Per-core values
            gu_k_offset_core_values=gu_k_offset_core_values,
            ag_sender_idx_core_values=ag_sender_idx_core_values,
            bg_sender_idx_core_values=bg_sender_idx_core_values,
        )

    @staticmethod
    def _build_compile_time_args(shared_ctx, rmsnorm_mcast_dst_cb, rmsnorm_mcast_params, routed_ctx):
        """
        Build shared expert compile-time args to append to routed expert args.

        Args:
            shared_ctx: _MoeSharedExpertContext
            rmsnorm_mcast_dst_cb: The CB index for the rmsnorm mcast destination (shared with routed)
            rmsnorm_mcast_params: RMSNorm mcast params (reuse NOC coords for residual mcast)
            routed_ctx: _MoeRoutedExpertContext (for SRAM down merge wiring)

        Returns:
            (ncrisc_args, brisc_args, trisc_args) - lists of named compile-time arg tuples
        """

        ncrisc_args = [
            # Gate/Up weights setup
            ("shared_gu_weights_cb", shared_ctx.gu_weights_cb),
            ("shared_gu_weights_num_pages", shared_ctx.gu_matmul_params["weights_num_pages"]),
            # Gate gather (A) receiver (MoeGather: receiver on NCRISC)
            ("shared_ag_noc0_num_senders", shared_ctx.num_compute_cores),
            ("shared_ag_noc0_receiver_semaphore_addr", shared_ctx.ag_receiver_semaphore_addr),
            ("shared_ag_noc1_receiver_semaphore_addr", shared_ctx.ag_noc1_receiver_semaphore_addr),
            ("shared_ag_dst_cb", shared_ctx.group1_cb),
            ("shared_ag_dst_num_pages", shared_ctx.gated_reduce_params["kernel_tiles_per_k"]),
            # Up gather (B) receiver (MoeGather: receiver on NCRISC)
            ("shared_bg_noc0_num_senders", shared_ctx.num_compute_cores),
            ("shared_bg_noc0_receiver_semaphore_addr", shared_ctx.bg_receiver_semaphore_addr),
            ("shared_bg_noc1_receiver_semaphore_addr", shared_ctx.bg_noc1_receiver_semaphore_addr),
            ("shared_bg_dst_cb", shared_ctx.group2_cb),
            ("shared_bg_dst_num_pages", shared_ctx.gated_reduce_params["kernel_tiles_per_k"]),
            # Down mcast receiver
            ("shared_down_mcast_data_receiver_semaphore_addr", shared_ctx.shared_mcast_receiver_semaphore_addr),
            ("shared_down_mcast_dst_cb", shared_ctx.down_mcast_dst_cb),
            ("shared_down_mcast_dst_num_pages", shared_ctx.down_matmul_params["k_num_tiles"]),
            # Down proj weights (setup_sharded_buffer on 112 matmul cores)
            ("shared_down_matmul_in1", shared_ctx.down_matmul_params["in1_cb"]),
            ("shared_down_matmul_k_num_tiles", shared_ctx.down_matmul_params["k_num_tiles"]),
            ("shared_down_matmul_out_w_per_core", shared_ctx.down_matmul_params["out_w"]),
            # Output gather receiver (MoeGather: receiver on NCRISC)
            ("shared_og_noc0_num_senders", shared_ctx.output_gather_params["noc0_num_senders"]),
            ("shared_og_noc1_num_senders", shared_ctx.output_gather_params["noc1_num_senders"]),
            ("shared_og_noc0_receiver_semaphore_addr", shared_ctx.output_gather_params["noc0_receiver_semaphore_addr"]),
            ("shared_og_noc1_receiver_semaphore_addr", shared_ctx.output_gather_params["noc1_receiver_semaphore_addr"]),
            ("shared_og_dst_cb", shared_ctx.output_gather_params["dst_cb"]),
            ("shared_og_dst_num_pages", shared_ctx.output_gather_params["dst_num_pages"]),
        ]
        brisc_args = [
            # Gate gather (A) sender (MoeGather: sender on BRISC)
            ("shared_ag_dest_noc_x", shared_ctx.gather_dest_noc_core.x),
            ("shared_ag_dest_noc_y", shared_ctx.gather_dest_noc_core.y),
            ("shared_ag_data_size_bytes", shared_ctx.gu_gather_data_size_bytes),
            ("shared_ag_receiver_semaphore_addr", shared_ctx.ag_receiver_semaphore_addr),
            ("shared_ag_src_cb", shared_ctx.gu_out_cb),
            ("shared_ag_src_num_pages", 1),
            ("shared_ag_receiver_data_addr", shared_ctx.ag_receiver_data_addr),
            # Up gather (B) sender (MoeGather: sender on BRISC)
            ("shared_bg_dest_noc_x", shared_ctx.gather_dest_noc_core.x),
            ("shared_bg_dest_noc_y", shared_ctx.gather_dest_noc_core.y),
            ("shared_bg_data_size_bytes", shared_ctx.gu_gather_data_size_bytes),
            ("shared_bg_receiver_semaphore_addr", shared_ctx.bg_receiver_semaphore_addr),
            ("shared_bg_src_cb", shared_ctx.gu_out_cb),
            ("shared_bg_src_num_pages", 1),
            ("shared_bg_receiver_data_addr", shared_ctx.bg_receiver_data_addr),
            # Down mcast sender (CTArgs reused from routed mcast; only need semaphores, CBs, sizes)
            ("shared_down_mcast_data_sender_semaphore_addr", shared_ctx.shared_mcast_sender_semaphore_addr),
            ("shared_down_mcast_data_receiver_semaphore_addr", shared_ctx.shared_mcast_receiver_semaphore_addr),
            ("shared_down_mcast_data_size_bytes", shared_ctx.down_mcast_params["data_size_bytes"]),
            ("shared_down_mcast_src_cb", shared_ctx.mcast_src_cb),  # gated reduce output (CB 31)
            ("shared_down_mcast_src_num_pages", shared_ctx.down_mcast_params["src_num_pages"]),
            ("shared_down_mcast_dst_cb", shared_ctx.down_mcast_dst_cb),
            # Output gather sender (MoeGather: sender on BRISC)
            ("shared_og_dest_noc_x", shared_ctx.output_gather_params["dest_noc_x"]),
            ("shared_og_dest_noc_y", shared_ctx.output_gather_params["dest_noc_y"]),
            ("shared_og_data_size_bytes", shared_ctx.output_gather_params["data_size_bytes"]),
            ("shared_og_receiver_semaphore_addr", shared_ctx.output_gather_params["receiver_semaphore_addr"]),
            ("shared_og_src_cb", shared_ctx.output_gather_params["src_cb"]),
            ("shared_og_src_num_pages", shared_ctx.output_gather_params["src_num_pages"]),
            ("shared_og_receiver_data_addr", shared_ctx.output_gather_params["receiver_data_addr"]),
            # Output mcast sender (sender core → 130 cores) — separate semaphores to avoid race
            ("shared_output_mcast_data_sender_semaphore_addr", shared_ctx.output_mcast_sender_semaphore_addr),
            ("shared_output_mcast_data_receiver_semaphore_addr", shared_ctx.output_mcast_receiver_semaphore_addr),
            ("shared_output_mcast_data_size_bytes", shared_ctx.output_mcast_params["data_size_bytes"]),
            ("shared_output_mcast_src_cb", shared_ctx.output_gather_params["dst_cb"]),  # read from output gather dst
            ("shared_output_mcast_src_num_pages", shared_ctx.output_mcast_params["src_num_pages"]),
            # Output mcast receiver
            ("shared_output_mcast_dst_num_pages", shared_ctx.output_mcast_params["dst_num_pages"]),
        ]
        trisc_args = [
            # Gate/Up matmul
            ("shared_gu_act_cb", rmsnorm_mcast_dst_cb),
            ("shared_gu_weights_cb", shared_ctx.gu_weights_cb),
            ("shared_gu_weights_cb_addr", shared_ctx.gu_matmul_params["weights_cb_addr"]),
            ("shared_gu_out_cb", shared_ctx.gu_out_cb),
            ("shared_gu_k_per_core", shared_ctx.gu_matmul_params["k_per_core"]),
            ("shared_gu_act_total_tiles", shared_ctx.gu_matmul_params["act_total_tiles"]),
            # Gated reduce
            ("shared_gated_reduce_group1_cb", shared_ctx.group1_cb),
            ("shared_gated_reduce_group2_cb", shared_ctx.group2_cb),
            ("shared_gated_reduce_intermed_cb", shared_ctx.intermed_cb),
            ("shared_gated_reduce_mcast_src_cb", shared_ctx.mcast_src_cb),
            ("shared_gated_reduce_tiles_per_k", shared_ctx.gated_reduce_params["kernel_tiles_per_k"]),
            ("shared_gated_reduce_k_num_tiles", shared_ctx.gated_reduce_params["kernel_k_num_tiles"]),
            # Down proj matmul
            ("shared_down_matmul_in0", shared_ctx.down_mcast_dst_cb),  # mcast'd activation
            ("shared_down_matmul_in1", shared_ctx.down_matmul_params["in1_cb"]),
            ("shared_down_matmul_out", shared_ctx.down_matmul_params["out_cb"]),
            ("shared_down_matmul_k_num_tiles", shared_ctx.down_matmul_params["k_num_tiles"]),
            ("shared_down_matmul_out_w_per_core", shared_ctx.down_matmul_params["out_w"]),
            ("shared_down_matmul_weights_cb_addr", shared_ctx.down_matmul_params["weights_cb_addr"]),
            # Residual add — reads from merged_down_out_cb (= shared_down + sram_down
            # when n_sram_active>0, or copy of shared_down when n_sram_active=0).
            # The shared+sram merge or copy happens between shared_down_matmul and
            # residual_add in the kernel order (see moe_kernel.cpp SRAM_DOWN_MERGE).
            ("shared_residual_add_in0", routed_ctx.merged_down_out_cb),
            ("shared_residual_add_in1", shared_ctx.residual_cb),  # residual (pre-loaded bias)
            ("shared_residual_add_out", shared_ctx.residual_add_params["out_cb"]),
            ("shared_residual_add_out_w", shared_ctx.down_matmul_params["out_w"]),
            ("shared_residual_add_total_in1_tiles", shared_ctx.residual_add_params["total_in1_tiles"]),
            # Sram down merge (eltwise_add when n_sram_active>0, copy when 0).
            # Both paths read from shared_down_matmul_out + maybe sram_down → merged_down_out.
            ("sram_down_merge_in0", routed_ctx.sram_down_proj_out_cb),  # SRAM down output
            ("sram_down_merge_in1", shared_ctx.down_matmul_params["out_cb"]),  # shared down output
            ("sram_down_merge_out", routed_ctx.merged_down_out_cb),  # merged → residual_add
            ("sram_down_merge_num_tiles", shared_ctx.down_matmul_params["out_w"]),  # 2 tiles per core
        ]
        return ncrisc_args, brisc_args, trisc_args

    @staticmethod
    def _build_cb_descriptors(shared_ctx):
        """Build CB descriptors for shared expert."""
        gr_group1 = shared_ctx.gated_reduce_params["cb_group1_descriptor"]
        gr_out = shared_ctx.gated_reduce_params["cb_out_descriptor"]
        descs = [
            shared_ctx.gu_matmul_params["cb_out_descriptor"],
            gr_group1,
            shared_ctx.gated_reduce_params["cb_intermed_descriptor"],
        ]
        if gr_out is not gr_group1:
            descs.append(gr_out)
        descs += [
            shared_ctx.down_mcast_params["dst_cb_descriptor"],
            shared_ctx.down_matmul_params["weights_cb_descriptor"],
            shared_ctx.down_matmul_params["output_cb_descriptor"],
            shared_ctx.residual_add_params["cb_out_descriptor"],
            shared_ctx.output_gather_params["dst_cb_descriptor"],
        ]
        return descs

    @staticmethod
    def _build_core_descriptors(shared_ctx, sender_core_grid):
        """
        Build core descriptors for shared expert.

        Args:
            shared_ctx: _MoeSharedExpertContext
            sender_core_grid: CoreRangeSet for sender/gated_reduce core

        Returns:
            (unified_descs, per_core_descs) - lists to append to routed expert descriptors
        """
        unified = [
            UnifiedCompileTimeCoreDescriptor(
                named_compile_time_arg="is_shared_compute_core",
                core_range=shared_ctx.compute_core_grid,
                value=1,
                other_value=0,
            ),
            UnifiedCompileTimeCoreDescriptor(
                named_compile_time_arg="is_shared_gate_compute_core",
                core_range=shared_ctx.a_compute_grid,
                value=1,
                other_value=0,
            ),
            UnifiedCompileTimeCoreDescriptor(
                named_compile_time_arg="is_shared_up_compute_core",
                core_range=shared_ctx.b_compute_grid,
                value=1,
                other_value=0,
            ),
            UnifiedCompileTimeCoreDescriptor(
                named_compile_time_arg="is_shared_gated_reduce_core",
                core_range=sender_core_grid,
                value=1,
                other_value=0,
            ),
            UnifiedCompileTimeCoreDescriptor(
                named_compile_time_arg="is_shared_mcast_receiver_core",
                core_range=shared_ctx.matmul_core_grid,
                value=1,
                other_value=0,
            ),
        ]
        per_core = [
            PerCoreCompileTimeDescriptor(
                named_compile_time_arg="shared_gu_k_offset",
                core_values=shared_ctx.gu_k_offset_core_values,
                other_value=0,
            ),
            PerCoreCompileTimeDescriptor(
                named_compile_time_arg="shared_ag_sender_idx",
                core_values=shared_ctx.ag_sender_idx_core_values,
                other_value=0,
            ),
            PerCoreCompileTimeDescriptor(
                named_compile_time_arg="shared_bg_sender_idx",
                core_values=shared_ctx.bg_sender_idx_core_values,
                other_value=0,
            ),
            PerCoreCompileTimeDescriptor(
                named_compile_time_arg="shared_residual_add_core_idx",
                core_values=[
                    (core, idx) for idx, core in enumerate(ttnn.corerange_to_cores(shared_ctx.matmul_core_grid))
                ],
                other_value=0,
            ),
        ]
        return unified, per_core


class MoeOp:
    """
    Top-level fused MoE operation.

    Composes MoeRoutedExpertOp and MoeSharedExpertOp, merging their
    CB descriptors, compile-time args, and core descriptors into a
    single unified kernel invocation.
    """

    # ------------------------------------------------------------------
    # Semaphore creation (global semaphores, like pre-sdpa pattern)
    # ------------------------------------------------------------------

    @staticmethod
    def create_semaphores(device):
        """Create global semaphores for the fused MoE op.

        Args:
            device: TT device or mesh device

        Returns:
            List of global semaphore objects (length MoeSem.NUM_SEMAPHORES)
        """
        device_grid_size = device.compute_with_storage_grid_size()
        available_cores = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))]
        )
        return [ttnn.create_global_semaphore(device, available_cores, 0) for _ in range(MoeSem.NUM_SEMAPHORES)]

    # ------------------------------------------------------------------
    # Shared utility setup APIs (used by both routed and shared experts)
    # ------------------------------------------------------------------

    @staticmethod
    def get_max_page_size_and_num_pages(device, num_tiles, tile_size):
        """
        Calculate optimal page size and number of pages for NOC transfers.

        The NOC has a maximum burst size that varies by architecture:
        - Wormhole: 8192 bytes
        - Blackhole: 16384 bytes
        """
        total_size = num_tiles * tile_size

        arch = device.arch()
        if arch == ttnn.device.Arch.WORMHOLE_B0:
            noc_max_page_size = 8192
        elif arch == ttnn.device.Arch.BLACKHOLE:
            noc_max_page_size = 16384
        else:
            raise ValueError(f"Unsupported architecture: {arch}")

        page_size = (noc_max_page_size // tile_size) * tile_size
        while total_size % page_size != 0 and page_size >= tile_size:
            page_size -= tile_size

        num_pages = total_size // page_size
        return page_size, num_pages

    @staticmethod
    def setup_mcast(
        device,
        sender_core,
        mcast_grid,
        src_cb,
        src_tensor,
        dst_cb,
        dst_tensor,
        sender_semaphore_addr,
        receiver_semaphore_addr,
        data_size_bytes,
        src_num_pages=None,
    ):
        """
        Set up parameters for a multicast operation.

        Mcast broadcasts data from a sender core to all cores in the mcast grid.

        Args:
            device: TT device
            sender_core: Logical CoreCoord of the sender (single core)
            mcast_grid: CoreRangeSet of destination cores (rectangular grid)
            src_cb: Source CB index on sender core
            src_tensor: Source tensor on sender core (for num_pages calculation), or None if src_num_pages is provided
            dst_cb: Destination CB index on receiver cores
            dst_tensor: Destination tensor on receiver cores (for CB descriptor)
            sender_semaphore_addr: Global semaphore address for sender
            receiver_semaphore_addr: Global semaphore address for receivers
            data_size_bytes: Total data size to mcast in bytes
            src_num_pages: Number of source pages (if provided, src_tensor is not used for this)

        Returns:
            Dictionary with mcast parameters for compile-time args
        """
        mcast_ranges = list(mcast_grid.ranges())
        mcast_grid_range = mcast_ranges[0]
        mcast_grid_start = mcast_grid_range.start
        mcast_grid_end = mcast_grid_range.end

        dest_noc_start_core = device.worker_core_from_logical_core(mcast_grid_start)
        dest_noc_end_core = device.worker_core_from_logical_core(mcast_grid_end)

        num_cores = (mcast_grid_end.x - mcast_grid_start.x + 1) * (mcast_grid_end.y - mcast_grid_start.y + 1)

        sender_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(sender_core, sender_core)])
        is_sender_part_of_receiver_grid = mcast_grid.contains(sender_core_grid)

        if src_num_pages is None:
            src_shard_shape = src_tensor.memory_config().shard_spec.shape
            src_tile = src_tensor.get_tile()
            src_num_pages = (src_shard_shape[0] * src_shard_shape[1]) // (
                src_tile.tile_shape[0] * src_tile.tile_shape[1]
            )

        dst_cb_descriptor = None
        if dst_tensor is not None:
            dst_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(dst_cb, dst_tensor)

        return {
            "dest_noc_start_x": dest_noc_start_core.x,
            "dest_noc_start_y": dest_noc_start_core.y,
            "dest_noc_end_x": dest_noc_end_core.x,
            "dest_noc_end_y": dest_noc_end_core.y,
            "num_cores": num_cores,
            "sender_semaphore_addr": sender_semaphore_addr,
            "receiver_semaphore_addr": receiver_semaphore_addr,
            "data_size_bytes": data_size_bytes,
            "src_cb": src_cb,
            "src_num_pages": src_num_pages,
            "is_sender_part_of_receiver_grid": is_sender_part_of_receiver_grid,
            "dst_cb": dst_cb,
            "dst_num_pages": src_num_pages,
            "dst_cb_descriptor": dst_cb_descriptor,
        }

    @staticmethod
    def setup_gather(
        device,
        receiver_core,
        sender_core_ranges,
        num_senders,
        data_size_bytes_per_sender,
        src_cb,
        src_num_pages,
        dst_cb,
        dst_tensor=None,
        noc0_receiver_semaphore_addr=0,
        noc1_receiver_semaphore_addr=0,
        row_major=True,
        use_explicit_sender_index=False,
        dst_num_pages=None,
    ):
        """
        Set up parameters for a gather operation.

        Gather collects data from multiple sender cores to a single receiver core.

        Args:
            device: TT device
            receiver_core: Logical CoreCoord of the receiver (single core)
            sender_core_ranges: CoreRangeSet of sender cores
            num_senders: Total number of sender cores
            data_size_bytes_per_sender: Bytes each sender sends
            src_cb: Source CB index on sender cores
            src_num_pages: Number of pages to wait for in source CB
            dst_cb: Destination CB index on receiver core
            dst_tensor: Destination tensor on receiver core (optional; CB descriptor created if provided)
            noc0_receiver_semaphore_addr: Global semaphore address for NOC0 senders
            noc1_receiver_semaphore_addr: Global semaphore address for NOC1 senders
            row_major: Grid traversal order (True=row-major, False=column-major)
            use_explicit_sender_index: If True, use explicit per-core sender index (for scattered cores)
            dst_num_pages: Override for destination page count (used when dst_tensor is None)

        Returns:
            Dictionary with gather parameters for NCRISC and BRISC compile-time args
        """
        receiver_core_noc = device.worker_core_from_logical_core(receiver_core)

        sender_ranges = list(sender_core_ranges.ranges())
        sender_min_x = min(r.start.x for r in sender_ranges)
        sender_min_y = min(r.start.y for r in sender_ranges)
        sender_max_x = max(r.end.x for r in sender_ranges)
        sender_max_y = max(r.end.y for r in sender_ranges)

        noc0_num_senders = num_senders
        noc1_num_senders = 0

        dst_cb_descriptor = None
        receiver_data_addr = 0
        if dst_tensor is not None:
            dst_shard_shape = dst_tensor.memory_config().shard_spec.shape
            dst_tile = dst_tensor.get_tile()
            dst_num_pages = (dst_shard_shape[0] * dst_shard_shape[1]) // (
                dst_tile.tile_shape[0] * dst_tile.tile_shape[1]
            )
            dst_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(dst_cb, dst_tensor)
            receiver_data_addr = _fused_base_addr(dst_tensor)

        assert dst_num_pages is not None, "dst_num_pages must be provided (either via dst_tensor or explicit override)"

        return {
            "dest_noc_x": receiver_core_noc.x,
            "dest_noc_y": receiver_core_noc.y,
            "data_size_bytes": data_size_bytes_per_sender,
            "receiver_semaphore_addr": noc0_receiver_semaphore_addr,
            "src_cb": src_cb,
            "src_num_pages": src_num_pages,
            "sender_grid_start_x": sender_min_x,
            "sender_grid_start_y": sender_min_y,
            "sender_grid_end_x": sender_max_x,
            "sender_grid_end_y": sender_max_y,
            "row_major": 1 if row_major else 0,
            "receiver_data_addr": receiver_data_addr,
            "noc0_num_senders": noc0_num_senders,
            "noc1_num_senders": noc1_num_senders,
            "noc0_receiver_semaphore_addr": noc0_receiver_semaphore_addr,
            "noc1_receiver_semaphore_addr": noc1_receiver_semaphore_addr,
            "dst_cb": dst_cb,
            "dst_num_pages": dst_num_pages,
            "dst_cb_descriptor": dst_cb_descriptor,
            "use_explicit_sender_index": use_explicit_sender_index,
        }

    @staticmethod
    def setup_sram_matmul(
        in0_cb,
        in1_cb,
        out_cb,
        weights_overlapped,
        output_tensor=None,
        k_num_tiles=0,
        fused_activation=0,
        fused_activation_approx_mode=True,
        weights_core_ranges_override=None,
    ):
        """
        Set up parameters for an SRAM matmul operation.

        SRAM matmul computes: output = input @ weights with optional fused activation.
        Weights and output are sharded in L1 (SRAM).

        Args:
            in0_cb: Input CB index (receives mcasted input)
            in1_cb: Weights CB index
            out_cb: Output CB index
            weights_overlapped: OverlappedTensor or plain ttnn.Tensor (WIDTH_SHARDED in L1)
            output_tensor: Output tensor (WIDTH_SHARDED in L1), or None if CB descriptor
                           will be created by the overlap function.
            k_num_tiles: K dimension in tiles
            fused_activation: Activation to fuse (0=none, 1=sigmoid, 2=silu)
            fused_activation_approx_mode: Whether to use approximate activation (default False)
        Returns:
            Dictionary with matmul parameters and CB descriptors
        """
        if hasattr(weights_overlapped, "fused_tensor"):
            fused_tensor_device0 = ttnn.get_device_tensors(weights_overlapped.fused_tensor)[0]
            out_w = weights_overlapped.shard_shape[1] // weights_overlapped.tile_shape[1]
            core_grid = weights_overlapped.core_range_set
            weights_cb_descriptor = cb_descriptor_from_overlapped_tensor(
                in1_cb, weights_overlapped, fused_tensor_device0
            )
            if weights_core_ranges_override is not None:
                weights_cb_descriptor.core_ranges = weights_core_ranges_override
        else:
            weights_device0 = ttnn.get_device_tensors(weights_overlapped)[0]
            tile = weights_device0.get_tile()
            shard_shape = weights_overlapped.memory_config().shard_spec.shape
            out_w = shard_shape[1] // tile.tile_shape[1]
            core_grid = weights_overlapped.memory_config().shard_spec.grid
            weights_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(in1_cb, weights_device0)
            if weights_core_ranges_override is not None:
                weights_cb_descriptor.core_ranges = weights_core_ranges_override
        output_cb_descriptor = None
        if output_tensor is not None:
            output_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(out_cb, output_tensor)

        return {
            "in0_cb": in0_cb,
            "in1_cb": in1_cb,
            "out_cb": out_cb,
            "k_num_tiles": k_num_tiles,
            "out_w": out_w,
            "fused_activation": fused_activation,
            "fused_activation_approx_mode": fused_activation_approx_mode,
            "core_grid": core_grid,
            "num_cores": core_grid.num_cores(),
            "weights_cb_descriptor": weights_cb_descriptor,
            "weights_cb_addr": ttnn.get_cb_address(weights_cb_descriptor),
            "output_cb_descriptor": output_cb_descriptor,
        }

    @staticmethod
    def golden_single_device(
        input_tensor,
        shared_gate_weights,
        shared_up_weights,
        shared_down_weights,
        gate_proj_weights_dict=None,
        up_proj_weights_dict=None,
        down_proj_weights_dict=None,
        rmsnorm_gamma=None,
        rmsnorm_epsilon=1e-6,
        enable_routing=True,
        # Routing-only params (ignored when enable_routing=False)
        routing_weights_tensor=None,
        bias_tensor=None,
        eps=1e-20,
        scaling_factor=2.5,
        use_hardcoded_expert_index=False,
        hardcoded_expert_index=0,
        explicit_expert_scale=None,
        include_residual=True,
    ):
        """
        PyTorch reference for the full fused MoE (routed + shared expert + eltwise add).

        When enable_routing=False, operates as dense MLP: single expert (index 0),
        no routing matmul, no expert scale.

        Args:
            input_tensor: [1, K] — raw input (pre-norm)
            shared_gate_weights: [K, K_down] shared expert gate weights
            shared_up_weights: [K, K_down] shared expert up weights
            shared_down_weights: [K_down, N] shared expert down weights
            gate_proj_weights_dict: Dict[int, Tensor] expert gate weights
            up_proj_weights_dict: Dict[int, Tensor] expert up weights
            down_proj_weights_dict: Dict[int, Tensor] expert down weights
            rmsnorm_gamma: [1, K] RMSNorm gamma weights
            rmsnorm_epsilon: RMSNorm epsilon
            enable_routing: If True, run full MoE routing. If False, dense MLP.
            routing_weights_tensor: [K, N_routing] (routing only)
            bias_tensor: [1, 8, 32] gate bias (routing only)
            eps, scaling_factor, use_hardcoded_expert_index, hardcoded_expert_index,
            explicit_expert_scale: routed expert gate params (routing only)
            include_residual: If True, add residual (raw input) in shared expert.
                Set to False for non-root devices in multi-device reduce.

        Returns:
            (top8_scores, top8_indices, final_output) — scores/indices are None when enable_routing=False
        """
        import torch

        from models.demos.deepseek_v3_b1.fused_ops.shared_expert.op import SharedExpertOp

        # Apply RMSNorm: raw input → normalized input (truncate to bfloat16 to match device)
        x = input_tensor.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        normalized_input = ((x * torch.rsqrt(variance + rmsnorm_epsilon)) * rmsnorm_gamma.float()).bfloat16().float()

        # Shared expert: normalized input for compute, residual (raw input) added when include_residual=True
        residual = input_tensor.float() if include_residual else torch.zeros_like(input_tensor.float())
        shared_output = SharedExpertOp.golden(
            normalized_input.float(),
            shared_gate_weights.float(),
            shared_up_weights.float(),
            shared_down_weights.float(),
            residual,
        ).bfloat16()

        # Reshape to match routed golden's fused_add_tensor expectation [1,1,1,N]
        shared_for_add = shared_output.float().reshape(1, 1, 1, -1)

        return MoeRoutedExpertOp.golden(
            normalized_input,
            gate_proj_weights_dict=gate_proj_weights_dict,
            up_proj_weights_dict=up_proj_weights_dict,
            down_proj_weights_dict=down_proj_weights_dict,
            fused_add_tensor=shared_for_add,
            enable_routing=enable_routing,
            routing_weights_tensor=routing_weights_tensor,
            bias_tensor=bias_tensor,
            eps=eps,
            scaling_factor=scaling_factor,
            use_hardcoded_expert_index=use_hardcoded_expert_index,
            hardcoded_expert_index=hardcoded_expert_index,
            explicit_expert_scale=explicit_expert_scale,
        )

    @staticmethod
    def golden(
        input_tensor,
        shared_gate_weights,
        shared_up_weights,
        shared_down_weights,
        gate_proj_weights_dict,
        up_proj_weights_dict,
        down_proj_weights_dict,
        rmsnorm_gamma,
        routing_weights_tensor,
        bias_tensor,
        rmsnorm_epsilon=1e-6,
        routing_mode=True,
        eps=1e-20,
        scaling_factor=2.5,
        include_residual=True,
        top_k=8,
        topk_groups=4,
    ):
        """
        Clean PyTorch golden for post-attention MoE: y = h + MoE(h).

        routing_mode controls all modes:
          - True: normal routed MoE (grouped noaux gate + top-k weighted experts)
          - False: dense MLP mode (single expert 0 with unit scale)
          - list[int]: hardcoded expert indices; weights come from gate scores for those experts
        """
        import torch

        def _as_2d(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(1, -1).to(torch.bfloat16)

        def _reshape_weight(w: torch.Tensor, in_dim: int):
            # Supports [K, N] and [1,1,K,N] tensors.
            w2 = w.to(torch.bfloat16)
            if w2.ndim == 2:
                return w2
            return w2.reshape(in_dim, -1)

        def _compute_grouped_topk(scores_flat: torch.Tensor, bias_flat: torch.Tensor):
            n_routed = scores_flat.shape[-1]
            n_groups = int(bias_tensor.shape[-2])
            experts_per_group = n_routed // n_groups
            scores_for_choice = scores_flat + bias_flat

            grouped = scores_for_choice.reshape(1, n_groups, experts_per_group)
            summed_experts_per_group = min(2, experts_per_group)
            group_scores = torch.topk(grouped, k=summed_experts_per_group, dim=-1, sorted=True)[0].sum(dim=-1)
            sel_topk_groups = min(topk_groups, n_groups)
            group_idx = torch.topk(group_scores, k=sel_topk_groups, dim=-1, sorted=True)[1]

            group_mask = torch.zeros_like(group_scores)
            group_mask.scatter_(1, group_idx, 1)
            score_mask = group_mask.unsqueeze(-1).expand(1, n_groups, experts_per_group).reshape(1, n_routed)
            masked_scores = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))

            sel_top_k = min(top_k, n_routed)
            topk_idx = torch.topk(masked_scores, k=sel_top_k, dim=-1, sorted=True)[1]
            topk_weight = scores_flat.gather(1, topk_idx)
            if sel_top_k > 1:
                topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + eps)
            topk_weight = topk_weight * scaling_factor
            return topk_weight, topk_idx

        # RMSNorm(h) for MoE input.
        x = _as_2d(input_tensor)
        variance = x.pow(2).mean(-1, keepdim=True)
        norm_x = (x * torch.rsqrt(variance + rmsnorm_epsilon)) * _as_2d(rmsnorm_gamma)

        # Shared expert branch: (SiLU(hWg) * (hWu))Wd + residual.
        sh_gate = _reshape_weight(shared_gate_weights, norm_x.shape[-1])
        sh_up = _reshape_weight(shared_up_weights, norm_x.shape[-1])
        sh_down = _reshape_weight(shared_down_weights, sh_gate.shape[-1])
        shared_hidden = torch.nn.functional.silu(norm_x @ sh_gate) * (norm_x @ sh_up)
        shared_output = shared_hidden @ sh_down
        if include_residual:
            shared_output = shared_output + x

        # Routed branch selection/weights.
        topk_scores = None
        topk_indices = None
        if routing_mode is False:
            selected_experts = sorted(gate_proj_weights_dict.keys())
            selected_scales = [torch.tensor(1.0, dtype=torch.bfloat16)] * len(selected_experts)
        else:
            logits = norm_x @ routing_weights_tensor.to(norm_x.dtype)
            scores = torch.sigmoid(logits)
            scores_flat = scores.reshape(1, -1)
            bias_flat = bias_tensor.reshape(1, -1).to(scores_flat.dtype)
            # Hardware gate produces top-k slots; both expert-index selection and
            # expert-scale mcast are indexed by per-device slot offset.
            # Keep this available for hardcoded mode parity.
            gate_topk_scores, gate_topk_indices = _compute_grouped_topk(scores_flat, bias_flat)

            topk_scores, topk_indices = gate_topk_scores, gate_topk_indices
            selected_experts = [int(i) for i in topk_indices[0].tolist()]
            selected_scales = [topk_scores[0, i] for i in range(len(selected_experts))]

        routed_sum = torch.zeros_like(shared_output)
        for expert_idx, expert_scale in zip(selected_experts, selected_scales, strict=True):
            gate_w = _reshape_weight(gate_proj_weights_dict[expert_idx], norm_x.shape[-1])
            up_w = _reshape_weight(up_proj_weights_dict[expert_idx], norm_x.shape[-1])
            down_w = _reshape_weight(down_proj_weights_dict[expert_idx], gate_w.shape[-1])
            gate_out = torch.nn.functional.silu(norm_x @ gate_w)
            up_out = norm_x @ up_w
            routed_hidden = gate_out * up_out * expert_scale
            routed_sum = routed_sum + (routed_hidden @ down_w)

        final_output = (shared_output + routed_sum).reshape(1, 1, 1, -1)
        return topk_scores, topk_indices, final_output

    @staticmethod
    def _overlap_cbs_with_sdpa_buffer(
        routed_ctx, shared_ctx, sdpa_kv_cache_buffer, sdpa_out_interm_buffer, reduce_all_cores_set=None
    ):
        """
        Override working-buffer CB descriptors to overlap with SDPA buffers.

        Packs all routed-expert and shared-expert intermediate CBs into
        sdpa_kv_cache_buffer (156,672 B) starting at offset 0, including the
        DRAM matmul in1 CBs (9/18 shared at the same offset). CBs 26, 30, 31
        overflow into sdpa_out_interm_buffer (43,520 B). Aliased CBs (13→12,
        14→11, 22→19, 18→9) share offsets with their source CB.
        """
        kv_buf = sdpa_kv_cache_buffer
        kv_addr = _fused_base_addr(kv_buf)
        kv_offset = 0

        out_buf = sdpa_out_interm_buffer
        out_addr = _fused_base_addr(out_buf)
        out_offset = 0

        # ── Routed Expert CBs → sdpa_kv_cache_buffer ──

        # CB 0: rmsnorm_output (total_size=14336, page_size=2048, tile=32x32, bfloat16)
        cb0_cb_id = routed_ctx.rmsnorm_output_cb
        cb0_total_size = 14336
        cb0_desc = ttnn.cb_descriptor_from_sharded_tensor(
            cb0_cb_id,
            kv_buf,
            address_offset=kv_offset,
            total_size=cb0_total_size,
        )
        cb0_fmt = ttnn.CBFormatDescriptor(
            buffer_index=cb0_cb_id,
            data_format=ttnn.bfloat16,
            page_size=2048,
            tile=ttnn.TileDescriptor(ttnn.Tile([32, 32])),
        )
        cb0_desc.format_descriptors = [cb0_fmt]
        # Restrict to sender; aliased with add_cb_in0 (eltwise_add cores) and
        # gate_mm_weights_cb (gate_mm cores at col 12). All three core sets are
        # strictly disjoint: sender at (12,9) ⊥ gate_proj DRAM-bank workers ⊥
        # gate_mm col-12 rows 0-7.
        cb0_desc.core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(routed_ctx.sender_core, routed_ctx.sender_core)])
        routed_ctx.rmsnorm_output_cb_descriptor = cb0_desc
        kv_offset += cb0_total_size

        # CB 1: gate_mm_input (total_size=14336, page_size=64, tile=1x32, bfloat16)
        cb1_cb_id = routed_ctx.gate_mm_input_cb
        cb1_total_size = 14336
        cb1_desc = ttnn.cb_descriptor_from_sharded_tensor(
            cb1_cb_id,
            kv_buf,
            address_offset=kv_offset,
            total_size=cb1_total_size,
        )
        cb1_fmt = ttnn.CBFormatDescriptor(
            buffer_index=cb1_cb_id,
            data_format=ttnn.bfloat16,
            page_size=64,
            tile=ttnn.TileDescriptor(ttnn.Tile([1, 32])),
        )
        cb1_desc.format_descriptors = [cb1_fmt]
        routed_ctx.gate_mm_input_cb_descriptor = cb1_desc
        kv_offset += cb1_total_size

        # Routing-only CBs (gate_mm_output, gate_input, expert_index)
        if routed_ctx.enable_routing:
            # CB 3: gate_mm_output (total_size=64, page_size=64, tile=1x32, bfloat16)
            cb3_cb_id = routed_ctx.gate_mm_output_cb
            cb3_total_size = 64
            cb3_desc = ttnn.cb_descriptor_from_sharded_tensor(
                cb3_cb_id,
                kv_buf,
                address_offset=kv_offset,
                total_size=cb3_total_size,
            )
            cb3_fmt = ttnn.CBFormatDescriptor(
                buffer_index=cb3_cb_id,
                data_format=ttnn.bfloat16,
                page_size=64,
                tile=ttnn.TileDescriptor(ttnn.Tile([1, 32])),
            )
            cb3_desc.format_descriptors = [cb3_fmt]
            # Restrict to gate_mm cores; aliased with down_proj_gather_dst_cb
            # (sender). Disjoint: gate_mm is in col 0 rows 0-7; sender is (12,9).
            cb3_desc.core_ranges = routed_ctx.gate_mm_params["core_grid"]
            routed_ctx.gate_mm_params["output_cb_descriptor"] = cb3_desc
            kv_offset += cb3_total_size

            # CB 4: gate_input (total_size=512, page_size=512, tile=16x16, bfloat16)
            cb4_offset = kv_offset
            cb4_cb_id = routed_ctx.gate_input_cb
            cb4_total_size = 512
            cb4_desc = ttnn.cb_descriptor_from_sharded_tensor(
                cb4_cb_id,
                kv_buf,
                address_offset=kv_offset,
                total_size=cb4_total_size,
            )
            cb4_fmt = ttnn.CBFormatDescriptor(
                buffer_index=cb4_cb_id,
                data_format=ttnn.bfloat16,
                page_size=512,
                tile=ttnn.TileDescriptor(ttnn.Tile([16, 16])),
            )
            cb4_desc.format_descriptors = [cb4_fmt]
            routed_ctx.gate_params["input_cb_descriptor"] = cb4_desc
            kv_offset += cb4_total_size
            routed_ctx.gate_mm_gather_params["receiver_data_addr"] = kv_addr + cb4_offset

            # CB 10: expert_index (total_size=32, page_size=32, tile=1x16, bfloat16)
            cb10_cb_id = routed_ctx.gate_proj_cb_index
            cb10_total_size = 32
            cb10_offset = kv_offset
            cb10_desc = ttnn.cb_descriptor_from_sharded_tensor(
                cb10_cb_id,
                kv_buf,
                address_offset=cb10_offset,
                total_size=cb10_total_size,
            )
            cb10_fmt = ttnn.CBFormatDescriptor(
                buffer_index=cb10_cb_id,
                data_format=ttnn.bfloat16,
                page_size=32,
                tile=ttnn.TileDescriptor(ttnn.Tile([1, 16])),
            )
            cb10_desc.format_descriptors = [cb10_fmt]
            routed_ctx.gate_proj_cb_index_descriptor = cb10_desc
            # MatmulExpertCompressedDRAM reads indices from this L1 address directly.
            routed_ctx.gate_proj_params["index_l1_addr"] = kv_addr + cb10_offset
            routed_ctx.up_proj_params["index_l1_addr"] = kv_addr + cb10_offset
            routed_ctx.down_proj_params["index_l1_addr"] = kv_addr + cb10_offset
            kv_offset += cb10_total_size

        # CB 11: gate_proj_output (aliases CB 14) — hardcoded descriptor
        cb11_offset = kv_offset
        cb11_cb_id = routed_ctx.gate_proj_cb_out
        cb11_total_size = 512
        cb11_desc = ttnn.cb_descriptor_from_sharded_tensor(
            cb11_cb_id,
            kv_buf,
            address_offset=kv_offset,
            total_size=cb11_total_size,
        )
        cb11_fmt = ttnn.CBFormatDescriptor(
            buffer_index=cb11_cb_id,
            data_format=ttnn.bfloat16,
            page_size=64,
            tile=ttnn.TileDescriptor(ttnn.Tile([1, 32])),
        )
        cb11_desc.format_descriptors = [cb11_fmt]
        # Restrict to streamer compute cores so the aliased SRAM gate cb_out
        # (same ID, on a_cores) has unambiguous per-core cb_metadata.
        cb11_desc.core_ranges = ttnn.CoreRangeSet(
            [ttnn.CoreRange(c, c) for c in routed_ctx.gate_proj_params["compute_cores_list"]]
        )
        routed_ctx.gate_proj_params["cb_out_descriptor"] = cb11_desc

        # SRAM gate_proj cb_out shares CB ID with gate_proj_cb_out (a_cores ⊥
        # streamer; per-core descriptors split the L1 layout between grids).
        sram_gate_params = routed_ctx.sram_gate_proj_params
        if sram_gate_params and sram_gate_params.get("num_sram_experts", 0) > 0:
            sram_gate_out_total = sram_gate_params["out_w"] * sram_gate_params["num_active_experts"] * 64
            sram_gate_out_desc = ttnn.cb_descriptor_from_sharded_tensor(
                routed_ctx.sram_gate_proj_out_cb,
                kv_buf,
                address_offset=kv_offset,
                total_size=sram_gate_out_total,
            )
            sram_gate_out_desc.format_descriptors = [
                ttnn.CBFormatDescriptor(
                    buffer_index=routed_ctx.sram_gate_proj_out_cb,
                    data_format=ttnn.bfloat16,
                    page_size=64,
                    tile=ttnn.TileDescriptor(ttnn.Tile([1, 32])),
                )
            ]
            sram_gate_out_desc.core_ranges = sram_gate_params["core_grid"]
            routed_ctx.sram_gate_proj_cb_out_descriptor = sram_gate_out_desc
            kv_offset += sram_gate_out_total

        # cb_out_silu: aliases CB 11's L1 region (same offset, same total_size) but
        # with a fat [silu_tile_h, tile_w] tile shape so the kernel's silu fast-path
        # copy_tile + silu + pack_tile covers all expert outputs in one tile op.
        # Active for K-split gate_proj (k_parallel_per_bank>1).
        silu_tile_h = routed_ctx.gate_proj_silu_tile_h
        cb_out_silu_id = routed_ctx.gate_proj_cb_out_silu
        silu_tile_desc = ttnn.Tile([silu_tile_h, 32])
        silu_tile_bytes = silu_tile_desc.get_tile_size(ttnn.bfloat16)
        cb_out_silu_desc = ttnn.cb_descriptor_from_sharded_tensor(
            cb_out_silu_id,
            kv_buf,
            address_offset=cb11_offset,
            total_size=cb11_total_size,
        )
        cb_out_silu_desc.format_descriptors = [
            ttnn.CBFormatDescriptor(
                buffer_index=cb_out_silu_id,
                data_format=ttnn.bfloat16,
                page_size=silu_tile_bytes,
                tile=ttnn.TileDescriptor(silu_tile_desc),
            )
        ]
        # cb_out_silu shares L1 with cb11 on streamer cores by design (silu
        # fast-path). Restrict core_ranges so it doesn't overlap on a_cores
        # where SRAM gate cb_out (aliased to cb11's CB ID) lives.
        cb_out_silu_desc.core_ranges = ttnn.CoreRangeSet(
            [ttnn.CoreRange(c, c) for c in routed_ctx.gate_proj_params["compute_cores_list"]]
        )
        routed_ctx.gate_proj_params["cb_out_silu_descriptor"] = cb_out_silu_desc
        kv_offset += cb11_total_size

        # CB 12: up_proj_mm_out (aliases CB 13) — hardcoded descriptor
        cb12_offset = kv_offset
        cb12_cb_id = routed_ctx.up_proj_cb_mm_out
        cb12_total_size = 512
        cb12_desc = ttnn.cb_descriptor_from_sharded_tensor(
            cb12_cb_id,
            kv_buf,
            address_offset=kv_offset,
            total_size=cb12_total_size,
        )
        cb12_fmt = ttnn.CBFormatDescriptor(
            buffer_index=cb12_cb_id,
            data_format=ttnn.bfloat16,
            page_size=64,
            tile=ttnn.TileDescriptor(ttnn.Tile([1, 32])),
        )
        cb12_desc.format_descriptors = [cb12_fmt]
        # Restrict to streamer compute cores; aliased SRAM up cb_out lives on b_cores.
        cb12_desc.core_ranges = ttnn.CoreRangeSet(
            [ttnn.CoreRange(c, c) for c in routed_ctx.up_proj_params["compute_cores_list"]]
        )
        routed_ctx.up_proj_params["cb_out_descriptor"] = cb12_desc
        kv_offset += cb12_total_size

        # SRAM up_proj cb_out shares CB ID with up_proj_cb_mm_out (b_cores ⊥ streamer).
        sram_up_params = routed_ctx.sram_up_proj_params
        if sram_up_params and sram_up_params.get("num_sram_experts", 0) > 0:
            sram_up_out_total = sram_up_params["out_w"] * sram_up_params["num_active_experts"] * 64
            sram_up_out_desc = ttnn.cb_descriptor_from_sharded_tensor(
                routed_ctx.sram_up_proj_out_cb,
                kv_buf,
                address_offset=kv_offset,
                total_size=sram_up_out_total,
            )
            sram_up_out_desc.format_descriptors = [
                ttnn.CBFormatDescriptor(
                    buffer_index=routed_ctx.sram_up_proj_out_cb,
                    data_format=ttnn.bfloat16,
                    page_size=64,
                    tile=ttnn.TileDescriptor(ttnn.Tile([1, 32])),
                )
            ]
            sram_up_out_desc.core_ranges = sram_up_params["core_grid"]
            routed_ctx.sram_up_proj_cb_out_descriptor = sram_up_out_desc
            kv_offset += sram_up_out_total

        # CB 15: mul_out (fused output) — hardcoded descriptor
        cb15_cb_id = routed_ctx.mul_cb_out
        cb15_total_size = 512
        cb15_desc = ttnn.cb_descriptor_from_sharded_tensor(
            cb15_cb_id,
            kv_buf,
            address_offset=kv_offset,
            total_size=cb15_total_size,
        )
        cb15_fmt = ttnn.CBFormatDescriptor(
            buffer_index=cb15_cb_id,
            data_format=ttnn.bfloat16,
            page_size=64,
            tile=ttnn.TileDescriptor(ttnn.Tile([1, 32])),
        )
        cb15_desc.format_descriptors = [cb15_fmt]
        routed_ctx.mul_params["cb_out_descriptor"] = cb15_desc
        kv_offset += cb15_total_size

        # CB 16: down_proj_gather_dst (total_size=4096, page_size=64, tile=1x32, bfloat16)
        cb16_offset = kv_offset
        cb16_cb_id = routed_ctx.down_proj_gather_dst_cb
        cb16_total_size = 4096
        cb16_desc = ttnn.cb_descriptor_from_sharded_tensor(
            cb16_cb_id,
            kv_buf,
            address_offset=kv_offset,
            total_size=cb16_total_size,
        )
        cb16_fmt = ttnn.CBFormatDescriptor(
            buffer_index=cb16_cb_id,
            data_format=ttnn.bfloat16,
            page_size=64,
            tile=ttnn.TileDescriptor(ttnn.Tile([1, 32])),
        )
        cb16_desc.format_descriptors = [cb16_fmt]
        # Restrict to sender_core; aliased with gate_mm_output_cb (gate_mm cores).
        cb16_desc.core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(routed_ctx.sender_core, routed_ctx.sender_core)])
        routed_ctx.down_proj_gather_params["dst_cb_descriptor"] = cb16_desc
        kv_offset += cb16_total_size
        routed_ctx.down_proj_gather_params["receiver_data_addr"] = kv_addr + cb16_offset

        # CB 17: down_proj_mcast_dst (total_size=4096, page_size=64, tile=1x32, bfloat16)
        cb17_cb_id = routed_ctx.down_proj_mcast_dst_cb
        cb17_total_size = 4096
        cb17_desc = ttnn.cb_descriptor_from_sharded_tensor(
            cb17_cb_id,
            kv_buf,
            address_offset=kv_offset,
            total_size=cb17_total_size,
        )
        cb17_fmt = ttnn.CBFormatDescriptor(
            buffer_index=cb17_cb_id,
            data_format=ttnn.bfloat16,
            page_size=64,
            tile=ttnn.TileDescriptor(ttnn.Tile([1, 32])),
        )
        cb17_desc.format_descriptors = [cb17_fmt]
        routed_ctx.down_proj_mcast_params["dst_cb_descriptor"] = cb17_desc
        kv_offset += cb17_total_size

        # CB 19: down_proj_output (total_size=1792, page_size=64, tile=1x32, bfloat16)
        cb19_offset = kv_offset
        cb19_cb_id = routed_ctx.down_proj_cb_out
        cb19_total_size = 1792
        cb19_desc = ttnn.cb_descriptor_from_sharded_tensor(
            cb19_cb_id,
            kv_buf,
            address_offset=kv_offset,
            total_size=cb19_total_size,
        )
        cb19_fmt = ttnn.CBFormatDescriptor(
            buffer_index=cb19_cb_id,
            data_format=ttnn.bfloat16,
            page_size=64,
            tile=ttnn.TileDescriptor(ttnn.Tile([1, 32])),
        )
        cb19_desc.format_descriptors = [cb19_fmt]
        routed_ctx.down_proj_params["cb_out_descriptor"] = cb19_desc
        kv_offset += cb19_total_size

        # SRAM down_proj cb_out (112 mcast receiver cores). accum_experts=True so
        # the kernel pushes per_core_n tiles ONCE per call (cross-expert sum
        # already accumulated). Per-core size = out_w × tile_1x32 (= 2 × 64 = 128 B).
        sram_down_params = routed_ctx.sram_down_proj_params
        if sram_down_params and sram_down_params.get("num_sram_experts", 0) > 0:
            sram_down_out_total = sram_down_params["out_w"] * 64
            sram_down_out_desc = ttnn.cb_descriptor_from_sharded_tensor(
                routed_ctx.sram_down_proj_out_cb,
                kv_buf,
                address_offset=kv_offset,
                total_size=sram_down_out_total,
            )
            sram_down_out_desc.format_descriptors = [
                ttnn.CBFormatDescriptor(
                    buffer_index=routed_ctx.sram_down_proj_out_cb,
                    data_format=ttnn.bfloat16,
                    page_size=64,
                    tile=ttnn.TileDescriptor(ttnn.Tile([1, 32])),
                )
            ]
            routed_ctx.sram_down_proj_cb_out_descriptor = sram_down_out_desc
            kv_offset += sram_down_out_total

        # cb_internal_acc shares cb_out's CB ID (down_proj_cb_internal_acc ==
        # down_proj_cb_out), so we don't build a separate descriptor — cb19_desc above
        # already covers the same L1 region with the same format.

        # Routing-only CBs (expert_scale, scalar working buffer)
        if routed_ctx.enable_routing:
            # CB 20: expert_scale (total_size=32, page_size=32, tile=1x16, bfloat16)
            cb20_cb_id = routed_ctx.mul_cb_scalar_src
            cb20_total_size = 32
            cb20_desc = ttnn.cb_descriptor_from_sharded_tensor(
                cb20_cb_id,
                kv_buf,
                address_offset=kv_offset,
                total_size=cb20_total_size,
            )
            cb20_fmt = ttnn.CBFormatDescriptor(
                buffer_index=cb20_cb_id,
                data_format=ttnn.bfloat16,
                page_size=32,
                tile=ttnn.TileDescriptor(ttnn.Tile([1, 16])),
            )
            cb20_desc.format_descriptors = [cb20_fmt]
            routed_ctx.mul_params["cb_scalar_src_descriptor"] = cb20_desc
            kv_offset += cb20_total_size

            # CB 21: scalar working buffer (1x32 tile to match mul in0/in1; only [0,0] used via SCALAR bcast)
            # Sized for all topk scalars (8) so the Mul can wait for all experts' scalars upfront
            # and do one flattened tile_regs cycle.
            TILE_1x32 = ttnn.Tile((1, 32))
            tile_1x32_size = TILE_1x32.get_tile_size(ttnn.bfloat16)
            tile_1x32_desc = ttnn.TileDescriptor(TILE_1x32)
            scalar_num_pages = 8  # = mul_num_experts when routing is on
            scalar_total_size = scalar_num_pages * tile_1x32_size
            cb21_desc = ttnn.cb_descriptor_from_sharded_tensor(
                routed_ctx.mul_cb_scalar,
                kv_buf,
                address_offset=kv_offset,
                total_size=scalar_total_size,
            )
            cb21_fmt = ttnn.CBFormatDescriptor(
                buffer_index=routed_ctx.mul_cb_scalar,
                data_format=ttnn.bfloat16,
                page_size=tile_1x32_size,
                tile=tile_1x32_desc,
            )
            cb21_desc.format_descriptors = [cb21_fmt]
            routed_ctx.mul_params["cb_scalar_descriptor"] = cb21_desc
            kv_offset += scalar_total_size

        # CB 23: add_cb_in1 (total_size=14336, page_size=1792, tile=32x32, bfloat16)
        cb23_cb_id = routed_ctx.add_cb_in1
        cb23_total_size = 14336
        cb23_desc = ttnn.cb_descriptor_from_sharded_tensor(
            cb23_cb_id,
            kv_buf,
            address_offset=kv_offset,
            total_size=cb23_total_size,
        )
        cb23_fmt = ttnn.CBFormatDescriptor(
            buffer_index=cb23_cb_id,
            data_format=ttnn.bfloat16,
            page_size=1792,
            tile=ttnn.TileDescriptor(ttnn.Tile([32, 32])),
        )
        cb23_desc.format_descriptors = [cb23_fmt]
        routed_ctx.add_params["cb_in1_descriptor"] = cb23_desc
        kv_offset += cb23_total_size

        # ── cb_fmt for gate/up/down: overlay on kv_buf BEFORE cb_in1 so the CB
        # allocator sees the fmt L1 region (no separate in1_backing_tensor needed).
        # fmt_cb_l1_addr in each proj_params is rewritten to kv_addr+offset; CT-args
        # emit reads it. Two slots (kernel toggles fmt_slot ^= 1), so kv_offset
        # advances by 2 * fmt_page_size.
        gate_proj_params = routed_ctx.gate_proj_params
        up_proj_params = routed_ctx.up_proj_params
        down_proj_params = routed_ctx.down_proj_params

        def _overlay_cb_fmt(params, cb_id):
            fmt_page_size = params["fmt_cb_page_size"]
            fmt_region = 2 * fmt_page_size  # double-buffered slots
            fmt_offset = kv_offset_ref[0]
            desc = ttnn.cb_descriptor_from_sharded_tensor(
                cb_id,
                kv_buf,
                address_offset=fmt_offset,
                total_size=fmt_page_size,
            )
            desc.format_descriptors = [
                ttnn.CBFormatDescriptor(
                    buffer_index=cb_id,
                    data_format=ttnn.uint8,
                    page_size=fmt_page_size,
                ),
            ]
            params["fmt_cb_l1_addr"] = kv_addr + fmt_offset
            kv_offset_ref[0] += fmt_region
            return desc

        kv_offset_ref = [kv_offset]  # mutable boxed ref so closure can advance it
        routed_ctx.gate_proj_cb_fmt_descriptor = _overlay_cb_fmt(gate_proj_params, routed_ctx.gate_proj_cb_fmt)
        routed_ctx.up_proj_cb_fmt_descriptor = _overlay_cb_fmt(up_proj_params, routed_ctx.up_proj_cb_fmt)
        routed_ctx.down_proj_cb_fmt_descriptor = _overlay_cb_fmt(down_proj_params, routed_ctx.down_proj_cb_fmt)
        kv_offset = kv_offset_ref[0]

        # ── CB 9/18: DRAM matmul in1 (gate/up_proj and down_proj share same offset) ──
        cb9_total_size = gate_proj_params["in1_total_size"]
        cb18_total_size = down_proj_params["in1_total_size"]
        shared_in1_size = max(cb9_total_size, cb18_total_size)

        cb9_offset = kv_offset

        cb9_desc = ttnn.cb_descriptor_from_sharded_tensor(
            routed_ctx.gate_proj_cb_in1,
            kv_buf,
            address_offset=cb9_offset,
            total_size=cb9_total_size,
        )
        gate_weights_tile = gate_proj_params["weights_tile"]
        gate_cb_page_size = gate_proj_params.get(
            "cb_page_size",
            gate_weights_tile.get_tile_size(gate_proj_params["weights_dtype"]),
        )
        gate_cb_data_format = gate_proj_params.get("cb_data_format", gate_proj_params["weights_dtype"])
        # Use canonical 32x32 tile for cb_in1 regardless of CompressedTensor's packed
        # uint8 storage (matches CompressedTensor.cb_descriptor_from_compressed_tensor).
        tile_32x32_desc = ttnn.TileDescriptor(ttnn.Tile([32, 32]))
        cb9_fmt = ttnn.CBFormatDescriptor(
            buffer_index=routed_ctx.gate_proj_cb_in1,
            data_format=gate_cb_data_format,
            page_size=gate_cb_page_size,
            tile=tile_32x32_desc,
        )
        cb9_desc.format_descriptors = [cb9_fmt]
        gate_proj_params["cb_in1_descriptor"] = cb9_desc
        gate_proj_params["in1_buf_addr"] = kv_addr + cb9_offset

        # CB 18 aliases CB 9 (same offset, different total_size)
        cb18_desc = ttnn.cb_descriptor_from_sharded_tensor(
            routed_ctx.down_proj_cb_in1,
            kv_buf,
            address_offset=cb9_offset,
            total_size=cb18_total_size,
        )
        down_weights_tile = down_proj_params["weights_tile"]
        down_cb_page_size = down_proj_params.get(
            "cb_page_size",
            down_weights_tile.get_tile_size(down_proj_params["weights_dtype"]),
        )
        down_cb_data_format = down_proj_params.get("cb_data_format", down_proj_params["weights_dtype"])
        cb18_fmt = ttnn.CBFormatDescriptor(
            buffer_index=routed_ctx.down_proj_cb_in1,
            data_format=down_cb_data_format,
            page_size=down_cb_page_size,
            tile=tile_32x32_desc,
        )
        cb18_desc.format_descriptors = [cb18_fmt]
        down_proj_params["cb_in1_descriptor"] = cb18_desc
        down_proj_params["in1_buf_addr"] = kv_addr + cb9_offset

        kv_offset += shared_in1_size

        # ── Shared Expert CBs → sdpa_kv_cache_buffer (continued) ──

        # CB 29: shared_gu_out (total_size=64, page_size=64, tile=1x32, bfloat16)
        cb29_cb_id = shared_ctx.gu_out_cb
        cb29_total_size = 64
        cb29_desc = ttnn.cb_descriptor_from_sharded_tensor(
            cb29_cb_id,
            kv_buf,
            address_offset=kv_offset,
            total_size=cb29_total_size,
        )
        cb29_fmt = ttnn.CBFormatDescriptor(
            buffer_index=cb29_cb_id,
            data_format=ttnn.bfloat16,
            page_size=64,
            tile=ttnn.TileDescriptor(ttnn.Tile([1, 32])),
        )
        cb29_desc.format_descriptors = [cb29_fmt]
        # Restrict to a/b shared compute cores; aliased with shared_output_gather_dst_cb (sender).
        cb29_desc.core_ranges = shared_ctx.compute_core_grid
        shared_ctx.gu_matmul_params["cb_out_descriptor"] = cb29_desc
        kv_offset += cb29_total_size

        # CB 32: shared_intermed (total_size=1024, page_size=512, face tile 16x16, bfloat16)
        # Face-view: pack_tile writes 1 face (256 bf16 = 512 bytes) per call. 2 pages (silu_sum, plain_sum).
        cb32_cb_id = shared_ctx.intermed_cb
        cb32_total_size = 1024
        cb32_desc = ttnn.cb_descriptor_from_sharded_tensor(
            cb32_cb_id,
            kv_buf,
            address_offset=kv_offset,
            total_size=cb32_total_size,
        )
        cb32_fmt = ttnn.CBFormatDescriptor(
            buffer_index=cb32_cb_id,
            data_format=ttnn.bfloat16,
            page_size=shared_ctx.gated_reduce_params["face_tile_size"],
            tile=shared_ctx.gated_reduce_params["face_tile_desc"],
        )
        cb32_desc.format_descriptors = [cb32_fmt]
        shared_ctx.gated_reduce_params["cb_intermed_descriptor"] = cb32_desc
        kv_offset += cb32_total_size

        # CB 33: shared_mcast_src (total_size=512, page_size=512, face tile 16x16, bfloat16)
        # Face-view: pack_tile writes 1 face (256 bf16 = 512 bytes) per call.
        cb33_cb_id = shared_ctx.mcast_src_cb
        cb33_total_size = 512
        cb33_desc = ttnn.cb_descriptor_from_sharded_tensor(
            cb33_cb_id,
            kv_buf,
            address_offset=kv_offset,
            total_size=cb33_total_size,
        )
        cb33_fmt = ttnn.CBFormatDescriptor(
            buffer_index=cb33_cb_id,
            data_format=ttnn.bfloat16,
            page_size=shared_ctx.gated_reduce_params["face_tile_size"],
            tile=shared_ctx.gated_reduce_params["face_tile_desc"],
        )
        cb33_desc.format_descriptors = [cb33_fmt]
        shared_ctx.gated_reduce_params["cb_out_descriptor"] = cb33_desc
        kv_offset += cb33_total_size

        # CB 34: shared_down_mcast_dst (total_size=512, page_size=64, tile=1x32, bfloat16)
        cb34_cb_id = shared_ctx.down_mcast_dst_cb
        cb34_total_size = 512
        cb34_desc = ttnn.cb_descriptor_from_sharded_tensor(
            cb34_cb_id,
            kv_buf,
            address_offset=kv_offset,
            total_size=cb34_total_size,
        )
        cb34_fmt = ttnn.CBFormatDescriptor(
            buffer_index=cb34_cb_id,
            data_format=ttnn.bfloat16,
            page_size=64,
            tile=ttnn.TileDescriptor(ttnn.Tile([1, 32])),
        )
        cb34_desc.format_descriptors = [cb34_fmt]
        shared_ctx.down_mcast_params["dst_cb_descriptor"] = cb34_desc
        kv_offset += cb34_total_size

        # CB 36: shared_down_matmul_out (total_size=128, page_size=64, tile=1x32, bfloat16)
        cb36_cb_id = shared_ctx.down_matmul_params["out_cb"]
        cb36_total_size = 128
        cb36_desc = ttnn.cb_descriptor_from_sharded_tensor(
            cb36_cb_id,
            kv_buf,
            address_offset=kv_offset,
            total_size=cb36_total_size,
        )
        cb36_fmt = ttnn.CBFormatDescriptor(
            buffer_index=cb36_cb_id,
            data_format=ttnn.bfloat16,
            page_size=64,
            tile=ttnn.TileDescriptor(ttnn.Tile([1, 32])),
        )
        cb36_desc.format_descriptors = [cb36_fmt]
        shared_ctx.down_matmul_params["output_cb_descriptor"] = cb36_desc
        kv_offset += cb36_total_size

        # CB merged_down_out: feeds residual_add (replaces shared_down_matmul_out_cb).
        # Holds shared_down + sram_down (or just shared_down when n_sram=0). Same
        # layout/size as cb36 — 2 1×32 tiles per core on the 112 mcast receivers.
        merged_cb_id = routed_ctx.merged_down_out_cb
        merged_total_size = 128
        merged_desc = ttnn.cb_descriptor_from_sharded_tensor(
            merged_cb_id,
            kv_buf,
            address_offset=kv_offset,
            total_size=merged_total_size,
        )
        merged_fmt = ttnn.CBFormatDescriptor(
            buffer_index=merged_cb_id,
            data_format=ttnn.bfloat16,
            page_size=64,
            tile=ttnn.TileDescriptor(ttnn.Tile([1, 32])),
        )
        merged_desc.format_descriptors = [merged_fmt]
        routed_ctx.merged_down_out_cb_descriptor = merged_desc
        kv_offset += merged_total_size

        # CB 37: shared_residual_add_out (total_size=128, page_size=64, tile=1x32, bfloat16)
        cb37_cb_id = shared_ctx.residual_add_params["out_cb"]
        cb37_total_size = 128
        cb37_desc = ttnn.cb_descriptor_from_sharded_tensor(
            cb37_cb_id,
            kv_buf,
            address_offset=kv_offset,
            total_size=cb37_total_size,
        )
        cb37_tile = ttnn.Tile([1, 32])
        cb37_fmt = ttnn.CBFormatDescriptor(
            buffer_index=cb37_cb_id,
            data_format=ttnn.bfloat16,
            page_size=64,
            tile=ttnn.TileDescriptor(cb37_tile),
        )
        cb37_desc.format_descriptors = [cb37_fmt]
        shared_ctx.residual_add_params["cb_out_descriptor"] = cb37_desc
        kv_offset += cb37_total_size

        # ── Sender-only routed SRAM CBs → overlaid onto cb_in1 (cb9/18) region ──
        # All five CBs live exclusively on sender_core. cb_in1 lives on the
        # gate/down DRAM streamer cores (disjoint from sender) and is not a
        # mcast destination, so overlaying sender-only CBs at cb9_offset is
        # safe per the overlap rules (cores disjoint AND neither is mcast dst).
        # cb_in1 region is shared_in1_size bytes (~96 KB); we use ~70 KB.
        if sram_gate_params and sram_gate_params.get("num_sram_experts", 0) > 0:
            # SRAM gather/GR CBs use 8×32 tiles (256 elements = 512 B, same byte
            # count as 16×16 face but matches the 8-K-slice × 32-element gather
            # layout). 8×32 IDs auto-reuse attention's SDPA 8×32 IDs via
            # cross-context manager reuse, freeing 5 slots in the 16×16 bucket.
            sram_gr_tile = ttnn.Tile([8, 32])
            sram_gr_tile_desc = ttnn.TileDescriptor(sram_gr_tile)
            sram_gr_tile_size = sram_gr_tile.get_tile_size(routed_ctx.data_format)
            num_active_local = sram_gate_params["num_active_experts"]

            def _make_face_cb_desc(cb_id, total_size, offset):
                desc = ttnn.cb_descriptor_from_sharded_tensor(
                    cb_id, kv_buf, address_offset=offset, total_size=total_size
                )
                desc.format_descriptors[0].tile = sram_gr_tile_desc
                desc.format_descriptors[0].page_size = sram_gr_tile_size
                return desc

            sram_offset = cb9_offset  # local cursor inside the cb_in1 region

            # sram_group1_cb / sram_group2_cb: gather destinations. Each
            # (expert, n_idx) slot is one 8×32 tile (8 K-partials × 32 N
            # elements = 512 B). Total = num_active × 8 N-tiles × 512 B.
            sram_group1_total = num_active_local * 8 * sram_gr_tile_size  # 32768 B
            routed_ctx.sram_group1_cb_descriptor = _make_face_cb_desc(
                routed_ctx.sram_group1_cb, sram_group1_total, sram_offset
            )
            routed_ctx.sram_ag_receiver_data_addr = kv_addr + sram_offset
            sram_offset += sram_group1_total

            sram_group2_total = sram_group1_total
            routed_ctx.sram_group2_cb_descriptor = _make_face_cb_desc(
                routed_ctx.sram_group2_cb, sram_group2_total, sram_offset
            )
            routed_ctx.sram_bg_receiver_data_addr = kv_addr + sram_offset
            sram_offset += sram_group2_total

            # sram_mcast_src_cb: extended GR output, n_active 8×32 tiles.
            sram_mcast_src_total = num_active_local * sram_gr_tile_size
            routed_ctx.sram_mcast_src_cb_descriptor = _make_face_cb_desc(
                routed_ctx.sram_mcast_src_cb, sram_mcast_src_total, sram_offset
            )
            sram_offset += sram_mcast_src_total

            # sram_intermed_cb: GR scratch (cap=2 8×32 tiles).
            sram_intermed_total = 2 * sram_gr_tile_size
            routed_ctx.sram_intermed_cb_descriptor = _make_face_cb_desc(
                routed_ctx.sram_intermed_cb, sram_intermed_total, sram_offset
            )
            sram_offset += sram_intermed_total

            # sram_gr_scalar_cb: GR scalar broadcast slot (cap=2 8×32 tiles).
            sram_gr_scalar_total = 2 * sram_gr_tile_size
            routed_ctx.sram_gr_scalar_cb_descriptor = _make_face_cb_desc(
                routed_ctx.sram_gr_scalar_cb, sram_gr_scalar_total, sram_offset
            )
            sram_offset += sram_gr_scalar_total

            assert sram_offset - cb9_offset <= shared_in1_size, (
                f"SRAM sender-only CBs ({sram_offset - cb9_offset} B) overflow "
                f"cb_in1 region ({shared_in1_size} B); need a larger overlay or "
                f"separate L1 region."
            )

        # ── Aliased CBs (share offset with source) → sdpa_kv_cache_buffer ──

        # CB 22: add_cb_in0 → same memory as CB 19 (total_size=1792, page_size=1792, tile=32x32, bfloat16)
        cb22_cb_id = routed_ctx.add_cb_in0
        cb22_total_size = 1792
        cb22_desc = ttnn.cb_descriptor_from_sharded_tensor(
            cb22_cb_id,
            kv_buf,
            address_offset=cb19_offset,
            total_size=cb22_total_size,
        )
        cb22_fmt = ttnn.CBFormatDescriptor(
            buffer_index=cb22_cb_id,
            data_format=ttnn.bfloat16,
            page_size=1792,
            tile=ttnn.TileDescriptor(ttnn.Tile([32, 32])),
        )
        cb22_desc.format_descriptors = [cb22_fmt]
        # Restrict to eltwise_add cores; aliased with gate_mm_weights_cb (gate_mm cores).
        # gate_mm and gate_proj_core_ranges are strictly disjoint (col 12 vs col 0,7).
        cb22_desc.core_ranges = routed_ctx.gate_proj_core_ranges
        routed_ctx.add_params["cb_in0_descriptor"] = cb22_desc

        # ── CBs → sdpa_out_interm_buffer ──

        # CB 26: residual_mcast_dst (total_size=14336, page_size=64, tile=1x32, bfloat16)
        cb26_cb_id = routed_ctx.residual_mcast_dst_cb
        cb26_total_size = 14336
        cb26_desc = ttnn.cb_descriptor_from_sharded_tensor(
            cb26_cb_id,
            out_buf,
            address_offset=out_offset,
            total_size=cb26_total_size,
        )
        cb26_fmt = ttnn.CBFormatDescriptor(
            buffer_index=cb26_cb_id,
            data_format=ttnn.bfloat16,
            page_size=64,
            tile=ttnn.TileDescriptor(ttnn.Tile([1, 32])),
        )
        cb26_desc.format_descriptors = [cb26_fmt]
        routed_ctx.residual_mcast_params["dst_cb_descriptor"] = cb26_desc
        out_offset += cb26_total_size

        # SRAM down mcast dst CB on the 112 shared mcast receiver cores. Lives in
        # out_buf since kv_buf is full — and the 112-core grid is a superset of
        # cb26's grid (residual_mcast_dst is on the same shared receivers), so
        # out_buf has the right per-core L1 backing here.
        sram_dst_cb_id = routed_ctx.sram_down_mcast_dst_cb
        sram_dst_total_size = 4096
        sram_dst_desc = ttnn.cb_descriptor_from_sharded_tensor(
            sram_dst_cb_id,
            out_buf,
            address_offset=out_offset,
            total_size=sram_dst_total_size,
        )
        sram_dst_fmt = ttnn.CBFormatDescriptor(
            buffer_index=sram_dst_cb_id,
            data_format=ttnn.bfloat16,
            page_size=64,
            tile=ttnn.TileDescriptor(ttnn.Tile([1, 32])),
        )
        sram_dst_desc.format_descriptors = [sram_dst_fmt]
        routed_ctx.sram_down_mcast_dst_cb_descriptor = sram_dst_desc
        out_offset += sram_dst_total_size

        # Sender-core-only region starts here (CB 30, 31, 38 are only on sender core).
        # Reduce scratch CBs (43-45) can reuse these offsets since they live on
        # disjoint cores (DRAM worker + fabric cores).
        sender_only_offset = out_offset

        # CB 30: shared_group1 (ag gather dst) (total_size=8192, page_size=512, face tile 16x16, bfloat16)
        # Aliased with group2: group1 at +0 (4096 bytes = 8 face-pages), group2 at +4096 (8 face-pages).
        # Face-view matches setup_gated_reduce intent: add_tiles reads 256 bf16 per "tile" slot.
        cb30_cb_id = shared_ctx.group1_cb
        cb30_total_size = 4096 * 2
        cb30_desc = ttnn.cb_descriptor_from_sharded_tensor(
            cb30_cb_id,
            out_buf,
            address_offset=out_offset,
            total_size=cb30_total_size,
        )
        cb30_fmt = ttnn.CBFormatDescriptor(
            buffer_index=cb30_cb_id,
            data_format=ttnn.bfloat16,
            page_size=shared_ctx.gated_reduce_params["face_tile_size"],
            tile=shared_ctx.gated_reduce_params["face_tile_desc"],
        )
        cb30_desc.format_descriptors = [cb30_fmt]
        shared_ctx.gated_reduce_params["cb_group1_descriptor"] = cb30_desc
        shared_ctx.ag_receiver_data_addr = out_addr + out_offset
        out_offset += cb30_total_size

        shared_ctx.bg_receiver_data_addr = shared_ctx.ag_receiver_data_addr + 4096

        # CB 38: shared_output_gather_dst (total_size=14336, page_size=64, tile=1x32, bfloat16)
        cb38_offset = out_offset
        cb38_cb_id = shared_ctx.output_gather_params["dst_cb"]
        cb38_total_size = 14336
        cb38_desc = ttnn.cb_descriptor_from_sharded_tensor(
            cb38_cb_id,
            out_buf,
            address_offset=out_offset,
            total_size=cb38_total_size,
        )
        cb38_fmt = ttnn.CBFormatDescriptor(
            buffer_index=cb38_cb_id,
            data_format=ttnn.bfloat16,
            page_size=64,
            tile=ttnn.TileDescriptor(ttnn.Tile([1, 32])),
        )
        cb38_desc.format_descriptors = [cb38_fmt]
        # Restrict to sender; aliased with shared_gu_out_cb (a/b shared compute).
        cb38_desc.core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(routed_ctx.sender_core, routed_ctx.sender_core)])
        shared_ctx.output_gather_params["dst_cb_descriptor"] = cb38_desc
        shared_ctx.output_gather_params["receiver_data_addr"] = out_addr + cb38_offset
        out_offset += cb38_total_size

        # ── Reduce: CB 24 (add_cb_out / reduce_local_cb) → sdpa_out_interm_buffer ──
        # When reduce is enabled, CB 24 is a working buffer (not final output).
        # Placed after CB 38 on DRAM bank cores; CB 30/31/38 are on sender core (disjoint).
        if reduce_all_cores_set is not None:
            cb24_cb_id = routed_ctx.add_cb_out
            cb24_total_size = 2048  # 1 tile of 32x32 bfloat16
            cb24_desc = ttnn.cb_descriptor_from_sharded_tensor(
                cb24_cb_id,
                out_buf,
                address_offset=out_offset,
                total_size=cb24_total_size,
            )
            cb24_fmt = ttnn.CBFormatDescriptor(
                buffer_index=cb24_cb_id,
                data_format=ttnn.bfloat16,
                page_size=2048,
                tile=ttnn.TileDescriptor(ttnn.Tile([32, 32])),
            )
            cb24_desc.format_descriptors = [cb24_fmt]
            # Restrict to reduce/add cores; aliased with residual_mcast_src_cb (sender).
            # reduce_all_cores_set covers eltwise_add + reduce workers; sender at (12,9)
            # is disjoint from those.
            cb24_desc.core_ranges = reduce_all_cores_set
            routed_ctx.add_params["cb_out_descriptor"] = cb24_desc
            out_offset += cb24_total_size

        # ── Reduce scratch CBs (43-45) → reuse sender-core-only offsets ──
        # These CBs live on DRAM worker + fabric cores, which are disjoint from the
        # sender core where CB 30/31/38 live. Safe to share the same buffer offsets.
        if reduce_all_cores_set is not None:
            reduce_offset = sender_only_offset
            reduce_tile_desc = ttnn.TileDescriptor(32, 32)

            # CB 43: reduce_scratch_cb (compute_tile_size * num_tiles)
            reduce_scratch_size = routed_ctx.reduce_params["compute_tile_size"] * routed_ctx.reduce_params["num_tiles"]
            reduce_cb_scratch_desc = ttnn.cb_descriptor_from_sharded_tensor(
                routed_ctx.reduce_scratch_cb,
                out_buf,
                address_offset=reduce_offset,
                total_size=reduce_scratch_size,
            )
            reduce_cb_scratch_desc.core_ranges = reduce_all_cores_set
            reduce_cb_scratch_desc.format_descriptors = [
                ttnn.CBFormatDescriptor(
                    buffer_index=routed_ctx.reduce_scratch_cb,
                    data_format=ttnn.bfloat16,
                    page_size=routed_ctx.reduce_params["compute_tile_size"],
                    tile=reduce_tile_desc,
                )
            ]
            routed_ctx.reduce_scratch_cb_descriptor = reduce_cb_scratch_desc
            reduce_offset += reduce_scratch_size

            # CB 44: reduce_packet_cb (slot_size_bytes * num_workers_per_column)
            reduce_packet_size = (
                routed_ctx.reduce_params["slot_size_bytes"] * routed_ctx.reduce_params["num_workers_per_column"]
            )
            reduce_cb_packet_desc = ttnn.cb_descriptor_from_sharded_tensor(
                routed_ctx.reduce_packet_cb,
                out_buf,
                address_offset=reduce_offset,
                total_size=reduce_packet_size,
            )
            reduce_cb_packet_desc.core_ranges = reduce_all_cores_set
            reduce_cb_packet_desc.format_descriptors = [
                ttnn.CBFormatDescriptor(
                    buffer_index=routed_ctx.reduce_packet_cb,
                    data_format=ttnn.bfloat16,
                    page_size=routed_ctx.reduce_params["slot_size_bytes"],
                )
            ]
            routed_ctx.reduce_packet_cb_descriptor = reduce_cb_packet_desc
            reduce_offset += reduce_packet_size
            # reduce received CB: single 3-page CB backed by intermediate tensor
            reduce_payload = routed_ctx.reduce_params["payload_size_bytes"]
            routed_ctx.reduce_received_cb_descriptors = []

            received_desc = ttnn.cb_descriptor_from_sharded_tensor(
                routed_ctx.reduce_received_cb, routed_ctx.reduce_params["intermediate_per_device"][0]
            )
            received_desc.core_ranges = reduce_all_cores_set
            received_desc.total_size = 3 * reduce_payload
            received_desc.format_descriptors = [
                ttnn.CBFormatDescriptor(
                    buffer_index=routed_ctx.reduce_received_cb,
                    data_format=ttnn.bfloat16,
                    page_size=reduce_payload,
                    tile=reduce_tile_desc,
                )
            ]
            routed_ctx.reduce_received_cb_descriptors.append(received_desc)

            # reduce output CB: backed by output tensor
            output_desc = ttnn.cb_descriptor_from_sharded_tensor(
                routed_ctx.reduce_output_cb, routed_ctx.reduce_params["output_per_device"][0]
            )
            output_desc.core_ranges = reduce_all_cores_set
            output_desc.total_size = reduce_payload
            output_desc.format_descriptors = [
                ttnn.CBFormatDescriptor(
                    buffer_index=routed_ctx.reduce_output_cb,
                    data_format=ttnn.bfloat16,
                    page_size=reduce_payload,
                    tile=reduce_tile_desc,
                )
            ]
            routed_ctx.reduce_received_cb_descriptors.append(output_desc)

    def _build_reduce_per_device(self, reduce_root_coord, coord, row, col, chip_id):
        """Apply reduce-to-one modifications to per-device state (self.device_*). No-op when reduce disabled."""
        ctx = self.ctx
        if not ctx.enable_reduce_to_one:
            return

        routed_ctx = ctx.routed_ctx
        reduce_params = ctx.reduce_params
        mesh_device = ctx.mesh_device
        use_torus = self.is_torus and reduce_root_coord[0] in [0, 3]
        device_role = get_reduce_device_role(coord, reduce_root_coord, use_torus)

        # Determine destination coordinate based on role
        if device_role == MESH_LEAF:
            if use_torus:
                dest_coord = ttnn.MeshCoordinate(row - 1, col) if row == 1 else ttnn.MeshCoordinate(row + 1, col)
            else:
                dest_coord = ttnn.MeshCoordinate(row + 1, col) if row == 0 else ttnn.MeshCoordinate(row - 1, col)
        elif device_role == MESH_ROOT3:
            dest_coord = ttnn.MeshCoordinate(reduce_root_coord[0], col)
        else:
            dest_coord = reduce_root_coord

        dest_fabric_node_id = mesh_device.get_fabric_node_id(dest_coord)

        # Per-device tensors
        intermediate_tensor = reduce_params["intermediate_per_device"][chip_id]
        out_tensor = reduce_params["output_per_device"][chip_id]
        payload_size_bytes = reduce_params["payload_size_bytes"]

        # Destination L1 address: offset within the single intermediate tensor
        # Page 0 = round 1 (LEAF data), page 1 = round 2 (ROOT3), page 2 = round 3 (ROOT2)
        intermediate_base = _fused_base_addr(intermediate_tensor)
        if device_role == MESH_LEAF:
            dst_l1_addr = intermediate_base  # page 0
            dst_sem_addr = reduce_params["sem_round1_addr"]
        elif device_role == MESH_ROOT3:
            dst_l1_addr = intermediate_base + payload_size_bytes  # page 1
            dst_sem_addr = reduce_params["sem_round2_addr"]
        elif device_role == MESH_ROOT2:
            dst_l1_addr = intermediate_base + 2 * payload_size_bytes  # page 2
            dst_sem_addr = reduce_params["sem_round3_addr"]
        else:
            dst_l1_addr = 0
            dst_sem_addr = reduce_params["sem_exit_addr"]

        output_core_phys = routed_ctx.device.worker_core_from_logical_core(reduce_params["output_core"])

        # Compile-time args
        self.ncrisc_args.extend([("reduce_device_role", device_role), ("reduce_num_tiles", reduce_params["num_tiles"])])
        self.brisc_args.extend(
            [
                ("reduce_device_role", device_role),
                ("reduce_num_tiles", reduce_params["num_tiles"]),
                ("reduce_payload_size_bytes", reduce_params["payload_size_bytes"]),
                ("reduce_num_hops", 1),
                ("reduce_dst_fabric_node_chip_id", dest_fabric_node_id.chip_id),
                ("reduce_dst_fabric_node_mesh_id", int(dest_fabric_node_id.mesh_id)),
                ("reduce_output_core_noc_x", output_core_phys.x),
                ("reduce_output_core_noc_y", output_core_phys.y),
                ("reduce_num_workers", reduce_params["num_workers_per_column"]),
                ("reduce_total_num_workers", reduce_params["num_workers"]),
                ("reduce_agg_output_size_bytes", routed_ctx.num_tiles_k * 32 * 2 if self.downstream_sockets else 0),
                ("reduce_forward_metadata_size_bytes", self._forward_metadata_size_bytes),
                ("reduce_packet_cb", routed_ctx.reduce_packet_cb),
                ("reduce_enable_downstream_socket", 1 if self.downstream_sockets else 0),
            ]
        )
        self.trisc_args.extend([("reduce_device_role", device_role), ("reduce_num_tiles", reduce_params["num_tiles"])])

        # Update fabric core descriptor
        fabric_core_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in reduce_params["fabric_cores"]])
        for i, desc in enumerate(self.device_unified_core_descs):
            if desc.named_compile_time_arg == "is_reduce_fabric_core":
                self.device_unified_core_descs[i] = UnifiedCompileTimeCoreDescriptor(
                    named_compile_time_arg="is_reduce_fabric_core",
                    core_range=fabric_core_set,
                    value=1,
                    other_value=0,
                )
                break

        # NCRISC common runtime args (semaphore addresses)
        self.ncrisc_common_rt_args.extend(
            [
                reduce_params["sem_round1_addr"],
                reduce_params["sem_round2_addr"],
                reduce_params["sem_round3_addr"],
            ]
        )

        # Reduce-to-sender sync CT args (reduce fabric cores → sender core NCRISC)
        sender_core_physical = routed_ctx.device.worker_core_from_logical_core(routed_ctx.sender_core)
        num_reduce_fabric_cores = len(reduce_params["fabric_cores"])
        reduce_sync_sem_addr = self.sem_addrs[MoeSem.REDUCE_SYNC]
        self.ncrisc_args.extend(
            [
                ("reduce_sync_sem_addr", reduce_sync_sem_addr),
                ("reduce_sync_num_fabric_cores", num_reduce_fabric_cores),
            ]
        )
        self.brisc_args.extend(
            [
                ("reduce_sync_sem_addr", reduce_sync_sem_addr),
                ("reduce_sync_noc_x", sender_core_physical.x),
                ("reduce_sync_noc_y", sender_core_physical.y),
            ]
        )

        # Shared worker→fabric ready semaphore consumed by the ready-mask drain loop.
        reduce_worker_fabric_ready_sem_addr = self.sem_addrs[MoeSem.REDUCE_WORKER_FABRIC_BASE]

        # Per-core runtime args for reduce worker and fabric cores
        # Persistent-signal sync: first worker (shard_idx==0) coordinates persistent signaling
        agg_sem_addr = self.sem_addrs[MoeSem.REDUCE_AGG_SYNC]
        persistent_core_noc_x = 0
        persistent_core_noc_y = 0
        if device_role == MESH_ROOT1:
            persistent_core_phys = routed_ctx.device.worker_core_from_logical_core(
                reduce_params["worker_cores_list"][0]
            )
            persistent_core_noc_x = persistent_core_phys.x
            persistent_core_noc_y = persistent_core_phys.y

        # Persistent signal: on ROOT1, aggregator worker signals a fabric core via local NOC,
        # then the fabric core sends a fabric atomic inc to the bcast sender on the entry device.
        persistent_enable_root1 = device_role == MESH_ROOT1 and self.persistent_next_iter_sem_addr != 0
        persistent_dst_noc_x = 0
        persistent_dst_noc_y = 0
        persistent_dst_sem_addr = 0
        self._persistent_fabric_core = None
        self._persistent_target_node = None
        persistent_fabric_signal_sem_addr = 0
        if persistent_enable_root1:
            persistent_fabric_core = reduce_params["fabric_cores"][0]
            persistent_fabric_core_phys = routed_ctx.device.worker_core_from_logical_core(persistent_fabric_core)
            persistent_fabric_signal_sem_addr = self.sem_addrs[MoeSem.REDUCE_PERSISTENT_FABRIC_SIGNAL]
            persistent_dst_noc_x = persistent_fabric_core_phys.x
            persistent_dst_noc_y = persistent_fabric_core_phys.y
            persistent_dst_sem_addr = persistent_fabric_signal_sem_addr
            self._persistent_fabric_core = persistent_fabric_core
            bcast_sender_coord = ctx.bcast_sender_coord
            self._persistent_target_node = mesh_device.get_fabric_node_id(bcast_sender_coord)
            self._persistent_bcast_dst_noc_x = sender_core_physical.x
            self._persistent_bcast_dst_noc_y = sender_core_physical.y
            self._persistent_bcast_dst_mesh_id = int(self._persistent_target_node.mesh_id)
            self._persistent_bcast_dst_chip_id = int(self._persistent_target_node.chip_id)
            self._persistent_bcast_dst_sem_addr = self.persistent_next_iter_sem_addr

            # Set the CTA descriptor so the chosen fabric core gets its own kernel group
            for i, desc in enumerate(self.device_unified_core_descs):
                if desc.named_compile_time_arg == "reduce_persistent_fabric_signal_enable":
                    self.device_unified_core_descs[i] = UnifiedCompileTimeCoreDescriptor(
                        named_compile_time_arg="reduce_persistent_fabric_signal_enable",
                        core_range=ttnn.CoreRangeSet([ttnn.CoreRange(persistent_fabric_core, persistent_fabric_core)]),
                        value=1,
                        other_value=0,
                    )
                    break

        reduce_brisc_per_core_args = []
        for core_idx, core in enumerate(reduce_params["worker_cores_list"]):
            fabric_core = reduce_params["column_to_fabric_core"][core.x]
            fabric_core_phys = routed_ctx.device.worker_core_from_logical_core(fabric_core)
            slot_idx = reduce_params["core_to_slot_idx"][(core.x, core.y)]
            shard_idx = reduce_params["core_to_shard_idx"][(core.x, core.y)]

            socket_config_addr = 0
            worker_agg_sem_addr = 0
            worker_agg_noc_x = 0
            worker_agg_noc_y = 0
            worker_metadata_addr = 0

            if device_role == MESH_ROOT1:
                worker_agg_sem_addr = agg_sem_addr
                worker_agg_noc_x = persistent_core_noc_x
                worker_agg_noc_y = persistent_core_noc_y
                if self.downstream_sockets is not None:
                    socket_config_addr = self.downstream_sockets[shard_idx].get_config_buffer_address()
                if self._metadata_l1_addr != 0:
                    worker_metadata_addr = self._metadata_l1_addr

            is_persistent_agg = persistent_enable_root1 and shard_idx == 0

            reduce_brisc_per_core_args.append(
                (
                    core,
                    [
                        fabric_core_phys.x,
                        fabric_core_phys.y,
                        slot_idx,
                        reduce_worker_fabric_ready_sem_addr,
                        dst_l1_addr,
                        dst_sem_addr,
                        _fused_base_addr(out_tensor),
                        shard_idx,
                        socket_config_addr,
                        worker_metadata_addr,
                        worker_agg_sem_addr,
                        worker_agg_noc_x,
                        worker_agg_noc_y,
                        int(is_persistent_agg),
                        persistent_dst_noc_x if is_persistent_agg else 0,
                        persistent_dst_noc_y if is_persistent_agg else 0,
                        0,  # persistent_dst_mesh_id (unused, local signal)
                        0,  # persistent_dst_chip_id (unused, local signal)
                        persistent_dst_sem_addr if is_persistent_agg else 0,
                    ],
                )
            )

        # Fabric core per-core args start with the shared ready semaphore address.
        for fc_idx, fc in enumerate(reduce_params["fabric_cores"]):
            fc_args = [reduce_worker_fabric_ready_sem_addr]
            if persistent_enable_root1 and fc_idx == 0:
                fc_args.extend(
                    [
                        persistent_fabric_signal_sem_addr,
                        self._persistent_bcast_dst_noc_x,
                        self._persistent_bcast_dst_noc_y,
                        self._persistent_bcast_dst_mesh_id,
                        self._persistent_bcast_dst_chip_id,
                        self._persistent_bcast_dst_sem_addr,
                    ]
                )
            reduce_brisc_per_core_args.append((fc, fc_args))

        self.device_rt_args_desc = PerCoreRuntimeArgsDescriptor(brisc_args=reduce_brisc_per_core_args)

    def _build_bcast_per_device(self, coord, row, col, chip_id):
        """Apply broadcast per-device modifications. No-op when bcast disabled."""
        ctx = self.ctx
        if not ctx.enable_bcast:
            return

        routed_ctx = ctx.routed_ctx
        bp = ctx.bcast_params
        sender_row, sender_col = bp["sender_coord"]
        bcast_is_sender = (row == sender_row) and (col == sender_col)
        bcast_socket_page_size = bp["page_size_bytes"] * bp["input_num_pages"]
        bcast_use_socket = ctx.socket is not None and bcast_is_sender
        if bcast_use_socket:
            self.device_kernel_defines += [("ENABLE_SOCKET_READER", "1")]
        bcast_is_secondary_sender = routed_ctx.mesh_cols > 1 and (row == sender_row) and (col != sender_col)
        bcast_has_secondary_target = bcast_is_sender and routed_ctx.mesh_cols > 1

        bcast_ring_size = routed_ctx.mesh_rows
        bcast_ring_index = row
        bcast_enable_torus = (sender_row == 0) or (sender_row == routed_ctx.mesh_rows - 1 and self.is_torus)

        if bcast_enable_torus:
            bcast_num_targets_forward = (bcast_ring_size - 1) // 2
            bcast_num_targets_backward = bcast_ring_size - 1 - bcast_num_targets_forward
        else:
            bcast_num_targets_forward = bcast_ring_size - bcast_ring_index - 1
            bcast_num_targets_backward = bcast_ring_index

        bcast_data_core_physical = routed_ctx.device.worker_core_from_logical_core(routed_ctx.sender_core)
        bcast_core_noc_x = bcast_data_core_physical.x
        bcast_core_noc_y = bcast_data_core_physical.y

        bcast_intermediate_per_device = ttnn.get_device_tensors(ctx.bcast_intermediate_tensor)
        bcast_intermediate_device = bcast_intermediate_per_device[chip_id]

        mesh_device = ctx.mesh_device
        mesh_rows = routed_ctx.mesh_rows
        self.bcast_fabric_node_id = mesh_device.get_fabric_node_id(coord)
        self.bcast_dst_nodes = []
        if bcast_num_targets_forward > 0:
            if bcast_enable_torus and sender_row == mesh_rows - 1 and row == sender_row:
                forward_coord = ttnn.MeshCoordinate(0, col)
            else:
                forward_coord = ttnn.MeshCoordinate((row + 1) % mesh_rows, col)
            self.bcast_dst_nodes.append(mesh_device.get_fabric_node_id(forward_coord))
        if bcast_num_targets_backward > 0:
            if bcast_enable_torus and sender_row == 0 and row == sender_row:
                backward_coord = ttnn.MeshCoordinate(mesh_rows - 1, col)
            else:
                backward_coord = ttnn.MeshCoordinate((row - 1 + mesh_rows) % mesh_rows, col)
            self.bcast_dst_nodes.append(mesh_device.get_fabric_node_id(backward_coord))
        if bcast_has_secondary_target:
            secondary_coord = ttnn.MeshCoordinate(row, 1 - sender_col)
            self.bcast_dst_nodes.append(mesh_device.get_fabric_node_id(secondary_coord))
        bcast_num_neighbors = len(self.bcast_dst_nodes)

        bcast_tensor_size_bytes = bp["page_size_bytes"] * bp["input_num_pages"]
        bcast_max_payload = int(ttnn.get_tt_fabric_max_payload_size_bytes())
        bcast_chunk_size_bytes = min(bcast_tensor_size_bytes, bcast_max_payload)
        bcast_last_chunk_size_bytes = bcast_tensor_size_bytes % bcast_chunk_size_bytes or bcast_chunk_size_bytes
        bcast_num_chunks = (bcast_tensor_size_bytes + bcast_chunk_size_bytes - 1) // bcast_chunk_size_bytes

        bcast_num_links = 1

        self.ncrisc_args.extend(
            [
                ("bcast_num_pages_to_read", bp["input_num_pages"]),
                ("bcast_tensor0_page_size", bp["page_size_bytes"]),
                ("bcast_num_neighbors", bcast_num_neighbors),
                ("bcast_num_links", bcast_num_links),
                ("bcast_is_sender", int(bcast_is_sender)),
                ("bcast_chunk_size_bytes", bcast_chunk_size_bytes),
                ("bcast_last_chunk_size_bytes", bcast_last_chunk_size_bytes),
                ("bcast_num_chunks", bcast_num_chunks),
                ("bcast_use_socket", int(bcast_use_socket)),
            ]
        )

        self.brisc_args.extend(
            [
                ("bcast_num_pages_to_read", bp["input_num_pages"]),
                ("bcast_is_sender", int(bcast_is_sender)),
                ("bcast_use_socket", int(bcast_use_socket)),
            ]
        )

        self.brisc_common_rt_args = [
            int(ctx.socket.get_config_buffer_address()) if bcast_use_socket else 0,
            int(bcast_socket_page_size) if bcast_use_socket else 0,
            1 if bcast_use_socket else 0,
        ]

        bcast_ncrisc_common_rt_args = [
            int(_fused_base_addr(bcast_intermediate_device)),
            bcast_core_noc_x,
            bcast_core_noc_y,
            int(bp["out_ready_sem_addr"]),
            0,
        ]

        bcast_ncrisc_base = len(self.ncrisc_common_rt_args)
        for i, (name, _val) in enumerate(self.ncrisc_args):
            if name == "bcast_ncrisc_common_rt_arg_base":
                self.ncrisc_args[i] = ("bcast_ncrisc_common_rt_arg_base", bcast_ncrisc_base)
                break
        self.ncrisc_common_rt_args.extend(bcast_ncrisc_common_rt_args)

        if bcast_num_neighbors > 0:
            bcast_ncrisc_per_core = [(routed_ctx.sender_core, [])]
            if self.device_rt_args_desc is not None:
                self.device_rt_args_desc = PerCoreRuntimeArgsDescriptor(
                    brisc_args=self.device_rt_args_desc.brisc_args,
                    ncrisc_args=bcast_ncrisc_per_core,
                )
            else:
                self.device_rt_args_desc = PerCoreRuntimeArgsDescriptor(
                    ncrisc_args=bcast_ncrisc_per_core,
                )

    def _setup_fabric_connections(self, coord, row, col, reduce_root_coord, kernel_result, program):
        """Setup fabric connections for reduce, broadcast, and D2D0 fabric cores."""
        ctx = self.ctx

        # Reduce fabric connections
        if ctx.enable_reduce_to_one:
            use_torus = self.is_torus and reduce_root_coord[0] in [0, 3]
            device_role = get_reduce_device_role(coord, reduce_root_coord, use_torus)
            if device_role != MESH_ROOT1:
                reduce_params = ctx.reduce_params
                mesh_device = ctx.mesh_device
                if device_role == MESH_LEAF:
                    if use_torus:
                        dest_coord = (
                            ttnn.MeshCoordinate(row - 1, col) if row == 1 else ttnn.MeshCoordinate(row + 1, col)
                        )
                    else:
                        dest_coord = (
                            ttnn.MeshCoordinate(row + 1, col) if row == 0 else ttnn.MeshCoordinate(row - 1, col)
                        )
                elif device_role == MESH_ROOT3:
                    dest_coord = ttnn.MeshCoordinate(reduce_root_coord[0], col)
                else:
                    dest_coord = reduce_root_coord

                fabric_node_id = mesh_device.get_fabric_node_id(coord)
                dest_fabric_node_id = mesh_device.get_fabric_node_id(dest_coord)
                num_columns = reduce_params["num_columns"]

                for fc_idx, fc in enumerate(reduce_params["fabric_cores"]):
                    link_idx = 0 if fc_idx < num_columns // 2 else 1
                    fc_kernel_idx = None
                    for group in kernel_result.groups:
                        if group.compile_time_arg_values.get(
                            "is_reduce_fabric_core"
                        ) == 1 and group.core_range_set.contains(fc):
                            fc_kernel_idx = group.brisc_kernel_index
                            break
                    fabric_rt_args_ref = program.kernels[fc_kernel_idx].runtime_args[fc.x][fc.y]
                    fabric_conn_args = ttnn.setup_fabric_connection(
                        fabric_node_id, dest_fabric_node_id, link_idx, program, fc
                    )
                    fabric_rt_args_ref.extend(fabric_conn_args)

        # Persistent next-iteration signal: a fabric core on ROOT1 sends fabric atomic inc
        # to bcast sender on entry device (aggregator signals the fabric core via local NOC)
        if ctx.enable_reduce_to_one and self._persistent_fabric_core is not None:
            mesh_device = ctx.mesh_device
            fc_core = self._persistent_fabric_core
            src_fabric_node_id = mesh_device.get_fabric_node_id(coord)
            dst_fabric_node_id = self._persistent_target_node

            fc_kernel_idx = None
            for group in kernel_result.groups:
                if group.compile_time_arg_values.get(
                    "reduce_persistent_fabric_signal_enable"
                ) == 1 and group.core_range_set.contains(fc_core):
                    fc_kernel_idx = group.brisc_kernel_index
                    break
            if fc_kernel_idx is not None:
                persistent_fabric_rt_args = ttnn.setup_fabric_connection(
                    src_fabric_node_id, dst_fabric_node_id, 0, program, fc_core
                )
                program.kernels[fc_kernel_idx].runtime_args[fc_core.x][fc_core.y].extend(persistent_fabric_rt_args)

        # Broadcast fabric connections
        if ctx.enable_bcast and len(self.bcast_dst_nodes) > 0:
            routed_ctx = ctx.routed_ctx
            bcast_worker_core = routed_ctx.sender_core
            for idx, kernel in enumerate(program.kernels):
                if kernel.core_ranges.contains(bcast_worker_core) and (
                    isinstance(kernel.config, ttnn.ReaderConfigDescriptor)
                    or (
                        isinstance(kernel.config, ttnn.DataMovementConfigDescriptor)
                        and kernel.config.processor == ttnn.DataMovementProcessor.RISCV_1
                    )
                ):
                    bcast_writer_rt_args_ref = kernel.runtime_args[bcast_worker_core.x][bcast_worker_core.y]
                    payload = []
                    for dst_node in self.bcast_dst_nodes:
                        setup_args = ttnn.setup_fabric_connection(
                            self.bcast_fabric_node_id, dst_node, 0, program, bcast_worker_core
                        )
                        payload.extend(setup_args)
                    for dst_node in self.bcast_dst_nodes:
                        payload.append(int(dst_node.mesh_id))
                        payload.append(int(dst_node.chip_id))
                    bcast_writer_rt_args_ref.extend([len(payload)] + payload)
                    break

    # ==================================================================
    # Unified MoeOp helpers — hide routed/shared split from op()
    # ==================================================================

    def __init__(
        self,
        shared_residual_mcast_src_tensor,
        gate_mm_weights_tensor=None,
        gate_bias_tensor=None,
        gate_indices_tensor=None,
        gate_output_scores_tensor=None,
        gate_output_indices_tensor=None,
        gate_proj_weights_tensor=None,
        up_proj_weights_tensor=None,
        down_proj_weights_tensor=None,
        final_output_tensor=None,
        rmsnorm_gamma_tensor=None,
        shared_gate_weights_overlapped=None,
        shared_up_weights_overlapped=None,
        shared_down_weights_tensor=None,
        shared_k_parallel=None,
        shared_n_parallel=None,
        epsilon=1e-6,
        enable_routing=True,
        use_hardcoded_expert_index=False,
        sdpa_kv_cache_buffer=None,
        sdpa_out_interm_buffer=None,
        reduce_intermediate_tensors=None,
        reduce_output_tensor=None,
        reduce_semaphores=None,
        reduce_root_coord=None,
        bcast_input_tensor=None,
        bcast_intermediate_tensor=None,
        bcast_semaphores=None,
        bcast_sender_coord=None,
        socket=None,
        reconfig_moe_cbs=False,
        enable_sram_bspm=False,
        semaphores=None,
        noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC,
        cb_id_context=None,
        worker_core_grid=None,
        is_torus=False,
        downstream_sockets=None,
        persistent_next_iter_semaphore=None,
        persistent_mode=False,
        termination_semaphore=None,
        forward_metadata_size_bytes=0,
        metadata_l1_addr=0,
        sram_gate_proj_weights_tensor=None,
        sram_up_proj_weights_tensor=None,
        sram_down_proj_weights_tensor=None,
    ):
        """Setup both routed and shared expert contexts, then overlap CBs with SDPA buffers."""
        self.noc_mode = noc_mode
        self.is_torus = is_torus
        self.downstream_sockets = downstream_sockets
        self._forward_metadata_size_bytes = forward_metadata_size_bytes
        self._metadata_l1_addr = metadata_l1_addr
        if semaphores is None:
            semaphores = MoeOp.create_semaphores(shared_residual_mcast_src_tensor.device())
        self.sem_addrs = [ttnn.get_global_semaphore_address(s) for s in semaphores]
        sem_addrs = self.sem_addrs
        self.persistent_mode = persistent_mode
        self.persistent_next_iter_sem_addr = (
            int(ttnn.get_global_semaphore_address(persistent_next_iter_semaphore))
            if persistent_next_iter_semaphore is not None
            else 0
        )
        self.termination_sem_addr = (
            int(ttnn.get_global_semaphore_address(termination_semaphore)) if termination_semaphore is not None else 0
        )

        if cb_id_context is None:
            self.cb_id_manager = CircularBufferIdManager()
            cb_id_context = self.cb_id_manager.create_context()
        else:
            self.cb_id_manager = cb_id_context._manager

        routed_ctx = MoeRoutedExpertOp._setup_dimensions(
            shared_residual_mcast_src_tensor,
            gate_mm_weights_tensor=gate_mm_weights_tensor,
            gate_bias_tensor=gate_bias_tensor,
            gate_indices_tensor=gate_indices_tensor,
            gate_output_scores_tensor=gate_output_scores_tensor,
            gate_output_indices_tensor=gate_output_indices_tensor,
            gate_proj_weights_tensor=gate_proj_weights_tensor,
            up_proj_weights_tensor=up_proj_weights_tensor,
            down_proj_weights_tensor=down_proj_weights_tensor,
            final_output_tensor=final_output_tensor,
            rmsnorm_gamma_tensor=rmsnorm_gamma_tensor,
            epsilon=epsilon,
            enable_routing=enable_routing,
            use_hardcoded_expert_index=use_hardcoded_expert_index,
            reduce_intermediate_tensors=reduce_intermediate_tensors,
            reduce_output_tensor=reduce_output_tensor,
            reduce_semaphores=reduce_semaphores,
            reduce_root_coord=reduce_root_coord,
            bcast_input_tensor=bcast_input_tensor,
            bcast_intermediate_tensor=bcast_intermediate_tensor,
            bcast_semaphores=bcast_semaphores,
            bcast_sender_coord=bcast_sender_coord,
            semaphores=semaphores,
            cb_id_context=cb_id_context,
            worker_core_grid=worker_core_grid,
            sram_gate_proj_weights_tensor=sram_gate_proj_weights_tensor,
            sram_up_proj_weights_tensor=sram_up_proj_weights_tensor,
            sram_down_proj_weights_tensor=sram_down_proj_weights_tensor,
            enable_sram_bspm=enable_sram_bspm,
        )

        device_tensor = ttnn.get_device_tensors(shared_residual_mcast_src_tensor)[0]
        input_tile = device_tensor.get_tile()
        input_tile_size = input_tile.get_tile_size(routed_ctx.data_format)
        shared_ctx = MoeSharedExpertOp._setup_dimensions(
            device=routed_ctx.device,
            shared_gate_weights_overlapped=shared_gate_weights_overlapped,
            shared_up_weights_overlapped=shared_up_weights_overlapped,
            shared_down_weights_tensor=shared_down_weights_tensor,
            num_tiles_k=routed_ctx.num_tiles_k,
            tile_1x32_size=routed_ctx.tile_1x32_size,
            data_format=routed_ctx.data_format,
            input_tile=input_tile,
            input_tile_size=input_tile_size,
            sender_core=routed_ctx.sender_core,
            sender_core_grid=routed_ctx.input_core_grid,
            mcast_grid=routed_ctx.mcast_grid,
            num_dram_worker_cores=routed_ctx.gate_proj_core_ranges.num_cores(),
            k_parallel=shared_k_parallel,
            n_parallel=shared_n_parallel,
            ag_receiver_semaphore_addr=sem_addrs[MoeSem.AG_GATHER],
            bg_receiver_semaphore_addr=sem_addrs[MoeSem.BG_GATHER],
            ag_noc1_receiver_semaphore_addr=sem_addrs[MoeSem.AG_GATHER],
            bg_noc1_receiver_semaphore_addr=sem_addrs[MoeSem.BG_GATHER],
            shared_mcast_sender_semaphore_addr=sem_addrs[MoeSem.MCAST_SENDER],
            shared_mcast_receiver_semaphore_addr=sem_addrs[MoeSem.SHARED_DOWN_MCAST_RECEIVER],
            output_gather_noc0_receiver_semaphore_addr=sem_addrs[MoeSem.OUTPUT_GATHER],
            output_gather_noc1_receiver_semaphore_addr=sem_addrs[MoeSem.OUTPUT_GATHER],
            output_mcast_sender_semaphore_addr=sem_addrs[MoeSem.MCAST_SENDER],
            output_mcast_receiver_semaphore_addr=sem_addrs[MoeSem.SHARED_OUTPUT_MCAST_RECEIVER],
            cb_id_context=cb_id_context,
            residual_mcast_dst_cb=routed_ctx.residual_mcast_dst_cb,
        )

        if sdpa_kv_cache_buffer is not None and sdpa_out_interm_buffer is not None:
            sdpa_kv_cache_buffer_device = ttnn.get_device_tensors(sdpa_kv_cache_buffer)[0]
            sdpa_out_interm_buffer_device = ttnn.get_device_tensors(sdpa_out_interm_buffer)[0]

            reduce_all_cores_set = None
            if routed_ctx.enable_reduce_to_one and routed_ctx.reduce_params:
                reduce_all_cores_set = ttnn.CoreRangeSet(
                    [ttnn.CoreRange(c, c) for c in routed_ctx.reduce_params["worker_cores_list"]]
                    + [ttnn.CoreRange(c, c) for c in routed_ctx.reduce_params["fabric_cores"]]
                )
            MoeOp._overlap_cbs_with_sdpa_buffer(
                routed_ctx,
                shared_ctx,
                sdpa_kv_cache_buffer_device,
                sdpa_out_interm_buffer_device,
                reduce_all_cores_set=reduce_all_cores_set,
            )
        self.ctx = MoeContext(
            routed_ctx=routed_ctx,
            shared_ctx=shared_ctx,
            mesh_device=shared_residual_mcast_src_tensor.device(),
            full_device_grid=routed_ctx.full_device_grid,
            mesh_rows=routed_ctx.mesh_rows,
            mesh_cols=routed_ctx.mesh_cols,
            enable_routing=routed_ctx.enable_routing,
            enable_reduce_to_one=routed_ctx.enable_reduce_to_one,
            enable_bcast=routed_ctx.enable_bcast,
            reconfig_moe_cbs=reconfig_moe_cbs,
            enable_sram_bspm=enable_sram_bspm,
            # IO tensors
            gate_mm_weights_tensor=gate_mm_weights_tensor,
            gate_bias_tensor=gate_bias_tensor,
            gate_indices_tensor=gate_indices_tensor,
            gate_output_scores_tensor=gate_output_scores_tensor,
            gate_output_indices_tensor=gate_output_indices_tensor,
            gate_proj_weights_tensor=gate_proj_weights_tensor,
            up_proj_weights_tensor=up_proj_weights_tensor,
            down_proj_weights_tensor=down_proj_weights_tensor,
            final_output_tensor=final_output_tensor,
            rmsnorm_gamma_tensor=rmsnorm_gamma_tensor,
            shared_residual_mcast_src_tensor=shared_residual_mcast_src_tensor,
            shared_gate_weights_fused_tensor=shared_gate_weights_overlapped.fused_tensor,
            shared_down_weights_tensor=shared_down_weights_tensor,
            sdpa_kv_cache_buffer=sdpa_kv_cache_buffer,
            sdpa_out_interm_buffer=sdpa_out_interm_buffer,
            # Reduce
            reduce_params=routed_ctx.reduce_params,
            reduce_intermediate_tensors=reduce_intermediate_tensors,
            reduce_output_tensor=reduce_output_tensor,
            # RMSNorm
            rmsnorm_epsilon_packed=routed_ctx.rmsnorm_epsilon_packed,
            rmsnorm_scalar_packed=routed_ctx.rmsnorm_scalar_packed,
            # Broadcast
            bcast_params=routed_ctx.bcast_params,
            bcast_input_tensor=bcast_input_tensor,
            bcast_intermediate_tensor=bcast_intermediate_tensor,
            bcast_semaphores=bcast_semaphores,
            bcast_sender_coord=bcast_sender_coord,
            socket=socket,
        )

        # Shared descriptors (populated by _build_descriptors)
        self.cb_descriptors = []
        self.unified_core_descs = []
        self.per_core_descs = []
        self.semaphore_descriptors = []
        self.io_tensors = []
        self.kernel_defines = []

        # Per-device state (populated by _setup_device_args per mesh iteration)
        self.ncrisc_args = []
        self.brisc_args = []
        self.trisc_args = []
        self.device_cb_descs = []
        self.device_sem_descs = []
        self.device_unified_core_descs = []
        self.device_per_core_descs = []
        self.device_rt_args_desc = None
        self.ncrisc_common_rt_args = []

    def _build_cb_descriptors(self):
        """Build combined CB descriptors for routed + shared expert + reduce."""
        cb_descriptors = []
        cb_descriptors += MoeRoutedExpertOp._build_cb_descriptors(self.ctx.routed_ctx)
        cb_descriptors += MoeSharedExpertOp._build_cb_descriptors(self.ctx.shared_ctx)
        if self.ctx.reconfig_moe_cbs:
            # Include per-device SRAM weight CB descriptors in the metadata so
            # the reconfig mask covers them too. Without this, NCRISC's
            # reset_stream_regs skips these CBs and setup_sharded_buffer's iter-2
            # cb_reserve_back blocks because iter-1's push (received=1, acked=0,
            # cap=1) was never reset.
            #
            # Replicated path (uniform SRAM): all coords share the same per-core
            # addresses, so flattening to a single list is safe.
            #
            # Per-coord path (BSPM SRAM, enable_sram_bspm=True): each device has
            # different SRAM weight CB addresses, so we keep the coord key and
            # build a per-device reconfig tensor via ShardTensor2dMesh.
            rctx = self.ctx.routed_ctx
            sram_descs_per_coord = {}
            if self.ctx.enable_sram_bspm:
                for per_dev_attr in (
                    "sram_gate_proj_cb_in1_descriptors_per_device",
                    "sram_up_proj_cb_in1_descriptors_per_device",
                    "sram_down_proj_cb_in1_descriptors_per_device",
                ):
                    per_dev = getattr(rctx, per_dev_attr, None) or {}
                    for coord, descs in per_dev.items():
                        if descs:
                            sram_descs_per_coord.setdefault(coord, []).extend(descs)
            # Take per-coord path only if we actually have per-(device, core) SRAM
            # descs.  Defensive: if enable_sram_bspm=True was set but the SRAM
            # weights came from the auto-fit path (uniform across devices), we
            # have no per-coord descs and must fall back to the replicated path.
            if self.ctx.enable_sram_bspm and sram_descs_per_coord:
                # cb_in1 descriptors already carry uniform (total_size=576,
                # page_size=576) — set at desc creation time in setup_matmul_
                # expert_sram for gate/up/down.  No reconfig-time override
                # needed; record_cb_metadata_per_coord reads desc.total_size
                # directly and gets the uniform value.
                self.cb_metadata = None
                self.cb_metadata_per_coord = record_cb_metadata_per_coord(cb_descriptors, sram_descs_per_coord)
                # BSPM-SRAM diagnostic: surface (device, core, cb) combinations
                # where the placeholder buffer kicked in (num_pages=0 or
                # total_size < page_size).  These are the all-bfp0 cores from
                # min_shard_bytes — if the kernel does any cb_wait/cb_reserve
                # on those CBs, num_pages=0 hangs.  Hang is non-deterministic
                # across runs, suggesting an underlying race or uninitialized
                # memory rather than a fixed-core bug.
                from loguru import logger as _diag_logger

                _sram_cb_ids = set()
                for _descs in sram_descs_per_coord.values():
                    for _desc in _descs:
                        for _fmt in _desc.format_descriptors:
                            _sram_cb_ids.add(_fmt.buffer_index)
                _sram_cb_ids = sorted(_sram_cb_ids)
                _placeholder_hits = []
                for coord in sorted(self.cb_metadata_per_coord.keys(), key=lambda c: (c[0], c[1])):
                    per_cb = self.cb_metadata_per_coord[coord]
                    for cb_id in _sram_cb_ids:
                        for addr, total_size, num_pages, page_size, core_ranges in per_cb.get(cb_id, []):
                            if num_pages == 0 or total_size < page_size:
                                for c in ttnn.corerange_to_cores(core_ranges, row_wise=True):
                                    _placeholder_hits.append(
                                        (coord, (c.x, c.y), cb_id, addr, total_size, num_pages, page_size)
                                    )
                if _placeholder_hits:
                    _diag_logger.warning(
                        "[BSPM-SRAM diag] {} placeholder-buffer (num_pages=0) SRAM CB entries detected:",
                        len(_placeholder_hits),
                    )
                    for coord, core, cb_id, addr, total_size, num_pages, page_size in _placeholder_hits:
                        _diag_logger.warning(
                            "  coord={} core{} cb{}: addr=0x{:x} size={} pages={} ps={}",
                            coord,
                            core,
                            cb_id,
                            addr,
                            total_size,
                            num_pages,
                            page_size,
                        )
                else:
                    _diag_logger.info("[BSPM-SRAM diag] No placeholder-buffer SRAM CB entries (all cores have data).")
            else:
                sram_metadata_descs = []
                for per_dev_attr in (
                    "sram_gate_proj_cb_in1_descriptors_per_device",
                    "sram_up_proj_cb_in1_descriptors_per_device",
                    "sram_down_proj_cb_in1_descriptors_per_device",
                ):
                    per_dev = getattr(rctx, per_dev_attr, None) or {}
                    for descs in per_dev.values():
                        if descs:
                            sram_metadata_descs.extend(descs)
                self.cb_metadata = record_cb_metadata(cb_descriptors + sram_metadata_descs)
                self.cb_metadata_per_coord = None
        return cb_descriptors

    def _build_dummy_cb_descs(self):
        """Build dummy CB descriptors from the ID manager's allocation table.

        Everything is dummied out. No buffer pointer, no address offset,
        minimal total_size (one page), full device grid for core_ranges.
        The reconfig tensor provides the real config at kernel start.
        """
        return self.cb_id_manager.build_dummy_cb_descriptors(self.ctx.full_device_grid)

    def _build_cb_reconfig_tensor(self):
        """Build L1-sharded CB reconfig tensor using shared utility.

        Dispatches between replicated and per-device modes based on whether
        SRAM BSPM is enabled — see :meth:`_build_cb_descriptors`.
        """
        if self.cb_metadata_per_coord is not None:
            self.reconfig_tensor = build_cb_reconfig_tensor(
                full_device_grid=self.ctx.full_device_grid,
                mesh_device=self.ctx.mesh_device,
                cb_metadata_per_coord=self.cb_metadata_per_coord,
            )
        else:
            self.reconfig_tensor = build_cb_reconfig_tensor(
                self.cb_metadata, self.ctx.full_device_grid, self.ctx.mesh_device
            )

    def _build_core_descriptors(self):
        """Build combined core descriptors for routed + shared expert."""
        unified_core_descs = []
        per_core_descs = []
        routed_unified, routed_per_core = MoeRoutedExpertOp._build_core_descriptors(self.ctx.routed_ctx)
        unified_core_descs += routed_unified
        per_core_descs += routed_per_core
        shared_unified, shared_per_core = MoeSharedExpertOp._build_core_descriptors(
            self.ctx.shared_ctx, sender_core_grid=self.ctx.routed_ctx.input_core_grid
        )
        unified_core_descs += shared_unified
        per_core_descs += shared_per_core
        return unified_core_descs, per_core_descs

    def _append_compile_time_args(
        self,
        chip_id,
        num_iterations,
        ncrisc_args,
        brisc_args,
        trisc_args,
    ):
        """Append MoE compile-time args (routed + shared) to existing arg lists."""
        routed_ncrisc, routed_brisc, routed_trisc = MoeRoutedExpertOp._build_compile_time_args(
            self.ctx.routed_ctx, chip_id
        )
        ncrisc_args += routed_ncrisc
        brisc_args += routed_brisc
        trisc_args += routed_trisc

        rmsnorm_mcast_dst_cb = self.ctx.routed_ctx.rmsnorm_mcast_params["dst_cb"]
        shared_ncrisc, shared_brisc, shared_trisc = MoeSharedExpertOp._build_compile_time_args(
            self.ctx.shared_ctx,
            rmsnorm_mcast_dst_cb,
            self.ctx.routed_ctx.rmsnorm_mcast_params,
            self.ctx.routed_ctx,
        )
        ncrisc_args += shared_ncrisc
        brisc_args += shared_brisc
        trisc_args += shared_trisc

        ncrisc_args += [("num_iterations", num_iterations)]
        brisc_args += [("num_iterations", num_iterations)]
        trisc_args += [("num_iterations", num_iterations)]
        ncrisc_args += [("persistent_mode", self.persistent_mode)]
        brisc_args += [("persistent_mode", self.persistent_mode)]
        trisc_args += [("persistent_mode", self.persistent_mode)]
        ncrisc_args += [("persistent_next_iter_sem_addr", self.persistent_next_iter_sem_addr)]
        brisc_args += [("persistent_next_iter_sem_addr", self.persistent_next_iter_sem_addr)]
        trisc_args += [("persistent_next_iter_sem_addr", self.persistent_next_iter_sem_addr)]
        ncrisc_args += [("termination_semaphore_addr", self.termination_sem_addr)]
        brisc_args += [("termination_semaphore_addr", self.termination_sem_addr)]
        trisc_args += [("termination_semaphore_addr", self.termination_sem_addr)]

    def _build_semaphore_descriptors(self):
        """Build semaphore descriptors — empty, global semaphores are used instead."""
        return []

    def _build_io_tensors(self):
        """Build IO tensor list from MoeContext."""
        ctx = self.ctx
        io_tensors = []
        if ctx.enable_routing:
            gate_mm_backing = getattr(ctx.gate_mm_weights_tensor, "fused_tensor", ctx.gate_mm_weights_tensor)
            io_tensors += [
                gate_mm_backing,
                ctx.gate_bias_tensor,
                ctx.gate_indices_tensor,
                ctx.gate_output_scores_tensor,
                ctx.gate_output_indices_tensor,
            ]
        # Expert tensors kept alive through the backing/fmt/meta tensors added below
        # (per-projection). The CT data tensors themselves are already uploaded to L1
        # as CompressedTensors.
        for wt in [ctx.gate_proj_weights_tensor, ctx.up_proj_weights_tensor, ctx.down_proj_weights_tensor]:
            for ct in wt:
                for data_t in ct.get_data_tensors():
                    io_tensors += [data_t]
        # MatmulExpertCompressedDRAM backing + meta/fmt/table_idx tensors
        for proj_params in (
            self.ctx.routed_ctx.gate_proj_params,
            self.ctx.routed_ctx.up_proj_params,
            self.ctx.routed_ctx.down_proj_params,
        ):
            if "expert_offsets_l1_addr_per_device" not in proj_params:
                continue
            if proj_params["in1_backing_tensor"] is not None:
                io_tensors.append(proj_params["in1_backing_tensor"])
            io_tensors.append(proj_params["fmt_dram_tensor"])
            # Lockstep refactor: meta_tensors_per_device[coord] = (offset_tensor,
            # bs_tensor) — the SAME mesh tensor for every coord (shared across
            # devices).  Add to io_tensors once for keep-alive.
            _meta_map = proj_params["meta_tensors_per_device"]
            if _meta_map:
                _first_coord = next(iter(_meta_map))
                _offset_t, _bsize_t = _meta_map[_first_coord]
                io_tensors.append(_offset_t)
                io_tensors.append(_bsize_t)
        if ctx.final_output_tensor is not None:
            io_tensors += [ctx.final_output_tensor]
        io_tensors += [
            ctx.rmsnorm_gamma_tensor.fused_tensor,
            ctx.shared_residual_mcast_src_tensor,
            ctx.shared_gate_weights_fused_tensor,
            ctx.shared_down_weights_tensor,
        ]
        if ctx.sdpa_kv_cache_buffer is not None:
            io_tensors += [ctx.sdpa_kv_cache_buffer]
        if ctx.sdpa_out_interm_buffer is not None:
            io_tensors += [ctx.sdpa_out_interm_buffer]
        if ctx.reconfig_moe_cbs:
            io_tensors += [self.reconfig_tensor]
        if ctx.enable_reduce_to_one:
            io_tensors += [
                ctx.reduce_intermediate_tensors,
                ctx.reduce_output_tensor,
            ]
        if ctx.bcast_input_tensor is not None:
            io_tensors += [ctx.bcast_input_tensor]
        if ctx.bcast_intermediate_tensor is not None:
            io_tensors += [ctx.bcast_intermediate_tensor]
        return io_tensors

    def _build_kernel_defines(self):
        """Build base kernel preprocessor defines (shared across devices).

        ENABLE_SOCKET_READER is NOT included here — it's per-device,
        added only on the sender device in _build_bcast_per_device.
        """
        defines = []
        if self.ctx.enable_routing:
            defines += [("ENABLE_ROUTING", "1")]
        if self.ctx.enable_reduce_to_one:
            defines += [("ENABLE_REDUCE_TO_ONE", "1")]
        if self.ctx.enable_bcast:
            defines += [("ENABLE_BCAST", "1")]
        if self.ctx.reconfig_moe_cbs:
            defines += [("RECONFIG_MOE_CBS", "1")]
        return defines

    def _build_descriptors(self):
        """Build all shared (non-per-device) descriptors and store on self."""
        self.cb_descriptors = self._build_cb_descriptors()
        if self.ctx.reconfig_moe_cbs:
            self._build_cb_reconfig_tensor()
            self.dummy_cb_descs = self._build_dummy_cb_descs()
        self.unified_core_descs, self.per_core_descs = self._build_core_descriptors()
        self.semaphore_descriptors = self._build_semaphore_descriptors()
        self.io_tensors = self._build_io_tensors()
        self.kernel_defines = self._build_kernel_defines()

    def _setup_per_device_args(
        self,
        chip_id,
        num_iterations,
        reduce_root_coord,
        coord,
        row,
        col,
    ):
        """Build all per-device state: compile-time args, descriptor copies, and reduce modifications."""
        self._persistent_fabric_core = None
        self._persistent_target_node = None
        # Start from shared descriptors
        self.ncrisc_args = []
        self.brisc_args = []
        self.trisc_args = []
        self._append_compile_time_args(
            chip_id,
            num_iterations,
            self.ncrisc_args,
            self.brisc_args,
            self.trisc_args,
        )

        if self.ctx.reconfig_moe_cbs:
            addr = _fused_base_addr(self.reconfig_tensor)
            self.ncrisc_args.append(("reconfig_cb_config_l1_addr", addr))
            self.brisc_args.append(("reconfig_cb_config_l1_addr", addr))
            self.trisc_args.append(("reconfig_cb_config_l1_addr", addr))

        self.device_cb_descs = list(self.cb_descriptors)
        self.device_sem_descs = list(self.semaphore_descriptors)
        self.device_unified_core_descs = list(self.unified_core_descs)
        self.device_per_core_descs = list(self.per_core_descs)
        self.device_rt_args_desc = None
        self.ncrisc_common_rt_args = []
        self.brisc_common_rt_args = []
        self.bcast_fabric_node_id = None
        self.bcast_dst_nodes = []
        self.device_kernel_defines = list(self.kernel_defines)

        # MatmulExpertCompressedDRAM per-device descriptor override.
        # Replaces the first-coord defaults written by _build_core_descriptors with
        # values for this specific mesh coordinate.
        routed_ctx = self.ctx.routed_ctx
        if "expert_offsets_l1_addr_per_device" in routed_ctx.gate_proj_params:
            proj_name_to_params = {
                "gate_proj": routed_ctx.gate_proj_params,
                "up_proj": routed_ctx.up_proj_params,
                "down_proj": routed_ctx.down_proj_params,
            }
            overrides = {}
            for proj, params in proj_name_to_params.items():
                pcv = params["per_core_values_per_device"][coord]
                overrides[f"{proj}_expert_offsets_l1_addr"] = params["expert_offsets_l1_addr_per_device"][coord]
                overrides[f"{proj}_block_sizes_l1_addr"] = params["block_sizes_l1_addr_per_device"][coord]
                overrides[f"{proj}_core_in_bank_idx"] = pcv["core_in_bank_idx"]
                overrides[f"{proj}_next_core_noc_x"] = pcv["next_core_noc_x"]
                overrides[f"{proj}_next_core_noc_y"] = pcv["next_core_noc_y"]
                overrides[f"{proj}_k_slice_idx"] = pcv["k_slice_idx"]
            for i, desc in enumerate(self.device_per_core_descs):
                name = desc.named_compile_time_arg
                if name in overrides:
                    self.device_per_core_descs[i] = PerCoreCompileTimeDescriptor(
                        named_compile_time_arg=name,
                        core_values=overrides[name],
                        other_value=0,
                    )

        # SRAM routed gate_proj per-device per-core overrides. Each device has
        # its own SRAM weight + fmt + base_addrs L1 allocations (independent
        # per-device addresses), so per-coord override is required for TP8.
        sgp = routed_ctx.sram_gate_proj_params
        if sgp and sgp.get("num_sram_experts", 0) > 0:
            sram_overrides = {
                "sram_gate_proj_fmt_l1_addr": sgp["sram_fmt_l1_addr_per_device"][coord],
                "sram_gate_proj_base_addrs_l1_addr": sgp["sram_base_addrs_l1_addr_per_device"][coord],
            }
            # Host-side diag: dump per-(device, core) values fed into the kernel
            # CT-arg override.  Cross-reference with kernel DPRINT
            # "[SRAM-CT-arg cb_in1=...] fmt=0x... base=0x..." — mismatch
            # means PerCoreCompileTimeDescriptor isn't routing per-core values.
            if self.ctx.enable_sram_bspm:
                from loguru import logger as _diag_logger

                _gate_cb_in1 = routed_ctx.sram_gate_proj_cb_in1
                for (_core, _fmt_addr), (_, _base_addr) in zip(
                    sram_overrides["sram_gate_proj_fmt_l1_addr"],
                    sram_overrides["sram_gate_proj_base_addrs_l1_addr"],
                ):
                    _diag_logger.info(
                        "[SRAM-CT-arg host cb_in1={}] coord={} core=({}, {}) fmt=0x{:x} base=0x{:x}",
                        _gate_cb_in1,
                        coord,
                        _core.x,
                        _core.y,
                        _fmt_addr,
                        _base_addr,
                    )
            for i, desc in enumerate(self.device_per_core_descs):
                name = desc.named_compile_time_arg
                if name in sram_overrides:
                    self.device_per_core_descs[i] = PerCoreCompileTimeDescriptor(
                        named_compile_time_arg=name,
                        core_values=sram_overrides[name],
                        other_value=0,
                    )

            # Append per-device cb_in1 CB descriptors (one per core, per_core_allocation=True).
            # Each device has its own L1 layout, so per-coord descriptors are added here
            # rather than in the shared _build_cb_descriptors.
            cb_in1_descs = routed_ctx.sram_gate_proj_cb_in1_descriptors_per_device.get(coord)
            if cb_in1_descs:
                self.device_cb_descs.extend(cb_in1_descs)

        # SRAM up_proj per-device per-core overrides — mirror of gate_proj.
        sup = routed_ctx.sram_up_proj_params
        if sup and sup.get("num_sram_experts", 0) > 0:
            sram_up_overrides = {
                "sram_up_proj_fmt_l1_addr": sup["sram_fmt_l1_addr_per_device"][coord],
                "sram_up_proj_base_addrs_l1_addr": sup["sram_base_addrs_l1_addr_per_device"][coord],
            }
            for i, desc in enumerate(self.device_per_core_descs):
                name = desc.named_compile_time_arg
                if name in sram_up_overrides:
                    self.device_per_core_descs[i] = PerCoreCompileTimeDescriptor(
                        named_compile_time_arg=name,
                        core_values=sram_up_overrides[name],
                        other_value=0,
                    )

            cb_in1_descs_up = routed_ctx.sram_up_proj_cb_in1_descriptors_per_device.get(coord)
            if cb_in1_descs_up:
                self.device_cb_descs.extend(cb_in1_descs_up)

        # SRAM down_proj per-device per-core overrides — mirror of gate_proj.
        sdp_pd = routed_ctx.sram_down_proj_params
        if sdp_pd and sdp_pd.get("num_sram_experts", 0) > 0:
            sram_down_overrides = {
                "sram_down_proj_fmt_l1_addr": sdp_pd["sram_fmt_l1_addr_per_device"][coord],
                "sram_down_proj_base_addrs_l1_addr": sdp_pd["sram_base_addrs_l1_addr_per_device"][coord],
            }
            for i, desc in enumerate(self.device_per_core_descs):
                name = desc.named_compile_time_arg
                if name in sram_down_overrides:
                    self.device_per_core_descs[i] = PerCoreCompileTimeDescriptor(
                        named_compile_time_arg=name,
                        core_values=sram_down_overrides[name],
                        other_value=0,
                    )

            cb_in1_descs_dn = routed_ctx.sram_down_proj_cb_in1_descriptors_per_device.get(coord)
            if cb_in1_descs_dn:
                self.device_cb_descs.extend(cb_in1_descs_dn)

        # Apply reduce-to-one modifications (no-op when reduce disabled)
        self._build_reduce_per_device(reduce_root_coord, coord, row, col, chip_id)

        # Apply broadcast modifications (no-op when bcast disabled)
        self._build_bcast_per_device(coord, row, col, chip_id)

    @staticmethod
    def op(
        shared_residual_mcast_src_tensor,
        # Routing-only tensors (None when enable_routing=False)
        gate_mm_weights_tensor=None,
        gate_bias_tensor=None,
        gate_indices_tensor=None,
        gate_output_scores_tensor=None,
        gate_output_indices_tensor=None,
        # Expert weights (always required)
        gate_proj_weights_tensor=None,
        up_proj_weights_tensor=None,
        down_proj_weights_tensor=None,
        final_output_tensor=None,
        # RMSNorm gamma weights (sender core)
        rmsnorm_gamma_tensor=None,
        # Shared expert tensors
        shared_gate_weights_overlapped=None,
        shared_up_weights_overlapped=None,
        shared_down_weights_tensor=None,
        shared_k_parallel=None,
        shared_n_parallel=None,
        epsilon=1e-6,
        enable_routing=True,
        use_hardcoded_expert_index=False,
        sdpa_kv_cache_buffer=None,
        sdpa_out_interm_buffer=None,
        num_iterations=1,
        persistent_mode=False,
        persistent_next_iter_semaphore=None,
        termination_semaphore=None,
        # ReduceToOne parameters
        reduce_intermediate_tensors: Optional[list] = None,
        reduce_output_tensor: Optional[ttnn.Tensor] = None,
        reduce_semaphores: Optional[list] = None,
        reduce_root_coord: Optional[ttnn.MeshCoordinate] = None,
        # Broadcast parameters
        bcast_input_tensor=None,
        bcast_intermediate_tensor=None,
        bcast_semaphores=None,
        bcast_sender_coord=None,
        # Socket: when provided, sender device BRISC receives the bcast payload over socket
        # instead of reading from a pre-loaded bcast_input_tensor.
        socket=None,
        # CB reconfig for fusion with preceding op
        reconfig_moe_cbs=False,
        # Global semaphores (created by MoeOp.create_semaphores)
        semaphores=None,
        noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC,
        # Optional worker-core grid override (used to avoid overlap with external micro-ops).
        worker_core_grid=None,
        # Torus topology support
        is_torus=False,
        # Per-worker downstream sockets for reduce workers to send reduced output
        downstream_sockets=None,
        cb_id_context=None,
        # SRAM routed gate_proj weights — list of L1-resident CompressedTensors,
        # one per slot, in slot order matching the sram_expert_ids encoding in
        # create_gate_indices_tensor. None or empty = DRAM-only.
        sram_gate_proj_weights_tensor=None,
        # SRAM routed up_proj weights — same contract as gate_proj.
        sram_up_proj_weights_tensor=None,
        # SRAM routed down_proj weights — runs on the 112 shared mcast receiver
        # cores. None = no SRAM down_proj; kernel skips uniformly.
        sram_down_proj_weights_tensor=None,
        enable_sram_bspm=False,
    ):
        """
        Execute the full fused MoE operation (routed + shared expert).

        When enable_routing=False, operates as a dense MLP without routing logic
        (no gate MM, gate gather, gate/TopK, index mcast, expert scale mcast).

        Args:
            shared_residual_mcast_src_tensor: Input activation on sender core (provides device, dtype, K)
            gate_proj_weights_tensor, up_proj_weights_tensor, down_proj_weights_tensor: Expert weights
            final_output_tensor: Final output tensor
            rmsnorm_gamma_tensor: RMSNorm gamma weights on sender core
            shared_gate_weights_overlapped: Gate proj OverlappedTensor
            shared_up_weights_overlapped: Up proj OverlappedTensor
            shared_down_weights_tensor: Shared expert down weights
            shared_k_parallel, shared_n_parallel: Shared expert parallelism factors
            enable_routing: If True, run full MoE with routing. If False, run as dense MLP.
            gate_mm_weights_tensor, gate_bias_tensor, gate_indices_tensor: Gate weights (routing only)
            gate_output_scores_tensor, gate_output_indices_tensor: Gate output tensors (routing only)
            sdpa_kv_cache_buffer, sdpa_out_interm_buffer: SDPA buffers for CB memory overlap
            num_iterations: Number of iterations to loop inside the kernel (default 1)
            reduce_intermediate_tensors: (Optional) Single intermediate tensor with 3x shard width for reduce rounds
            reduce_output_tensor: (Optional) Final reduced output tensor on ROOT1 device
            reduce_semaphores: (Optional) List of 4 global semaphores for reduce synchronization
            reduce_root_coord: (Optional) MeshCoordinate of ROOT1 device
            reconfig_moe_cbs: If True, create a per-core CB config tensor and reconfigure
                CBs at kernel start (for fusion with a preceding op)

        Returns:
            (gate_output_scores_tensor, gate_output_indices_tensor, final_output_tensor or reduce_output_tensor)
        """
        # ==================================================================
        # Setup
        # ==================================================================
        moe = MoeOp(
            shared_residual_mcast_src_tensor,
            gate_mm_weights_tensor=gate_mm_weights_tensor,
            gate_bias_tensor=gate_bias_tensor,
            gate_indices_tensor=gate_indices_tensor,
            gate_output_scores_tensor=gate_output_scores_tensor,
            gate_output_indices_tensor=gate_output_indices_tensor,
            gate_proj_weights_tensor=gate_proj_weights_tensor,
            up_proj_weights_tensor=up_proj_weights_tensor,
            down_proj_weights_tensor=down_proj_weights_tensor,
            final_output_tensor=final_output_tensor,
            rmsnorm_gamma_tensor=rmsnorm_gamma_tensor,
            shared_gate_weights_overlapped=shared_gate_weights_overlapped,
            shared_up_weights_overlapped=shared_up_weights_overlapped,
            shared_down_weights_tensor=shared_down_weights_tensor,
            shared_k_parallel=shared_k_parallel,
            shared_n_parallel=shared_n_parallel,
            epsilon=epsilon,
            enable_routing=enable_routing,
            use_hardcoded_expert_index=use_hardcoded_expert_index,
            sdpa_kv_cache_buffer=sdpa_kv_cache_buffer,
            sdpa_out_interm_buffer=sdpa_out_interm_buffer,
            reduce_intermediate_tensors=reduce_intermediate_tensors,
            reduce_output_tensor=reduce_output_tensor,
            reduce_semaphores=reduce_semaphores,
            reduce_root_coord=reduce_root_coord,
            bcast_input_tensor=bcast_input_tensor,
            bcast_intermediate_tensor=bcast_intermediate_tensor,
            bcast_semaphores=bcast_semaphores,
            bcast_sender_coord=bcast_sender_coord,
            socket=socket,
            reconfig_moe_cbs=reconfig_moe_cbs,
            semaphores=semaphores,
            noc_mode=noc_mode,
            worker_core_grid=worker_core_grid,
            is_torus=is_torus,
            downstream_sockets=downstream_sockets,
            cb_id_context=cb_id_context,
            persistent_next_iter_semaphore=persistent_next_iter_semaphore,
            persistent_mode=persistent_mode,
            termination_semaphore=termination_semaphore,
            sram_gate_proj_weights_tensor=sram_gate_proj_weights_tensor,
            sram_up_proj_weights_tensor=sram_up_proj_weights_tensor,
            sram_down_proj_weights_tensor=sram_down_proj_weights_tensor,
            enable_sram_bspm=enable_sram_bspm,
        )

        # ==================================================================
        # Build descriptors
        # ==================================================================
        moe._build_descriptors()

        # ==================================================================
        # Create per-device programs (mesh loop)
        # ==================================================================
        ctx = moe.ctx
        mesh_program_descriptor = ttnn.MeshProgramDescriptor()
        for row in range(ctx.mesh_rows):
            for col in range(ctx.mesh_cols):
                coord = ttnn.MeshCoordinate(row, col)
                chip_id = row * ctx.mesh_cols + col

                moe._setup_per_device_args(
                    chip_id,
                    num_iterations,
                    reduce_root_coord,
                    coord,
                    row,
                    col,
                )

                unified_kernel = UnifiedKernelDescriptor(
                    kernel_source="models/demos/deepseek_v3_b1/fused_ops/moe/moe_kernel.cpp",
                    core_ranges=ctx.full_device_grid,
                    ncrisc_named_compile_time_args=moe.ncrisc_args,
                    brisc_named_compile_time_args=moe.brisc_args,
                    trisc_named_compile_time_args=moe.trisc_args,
                    ncrisc_common_runtime_args=moe.ncrisc_common_rt_args,
                    brisc_common_runtime_args=moe.brisc_common_rt_args,
                    trisc_common_runtime_args=[ctx.rmsnorm_epsilon_packed, ctx.rmsnorm_scalar_packed],
                    trisc_compute_config=ttnn.ComputeConfigDescriptor(
                        math_fidelity=ttnn.MathFidelity.LoFi,
                        math_approx_mode=False,
                        fp32_dest_acc_en=False,
                        dst_full_sync_en=False,
                    ),
                    unified_compile_time_core_descriptors=moe.device_unified_core_descs,
                    per_core_compile_time_descriptors=moe.device_per_core_descs,
                    per_core_runtime_args_descriptor=moe.device_rt_args_desc,
                    defines=moe.device_kernel_defines,
                    noc_mode=moe.noc_mode,
                )
                kernel_result = unified_kernel.get_kernel_descriptors()

                kernels = kernel_result.kernels
                cb_descs = moe.dummy_cb_descs if ctx.reconfig_moe_cbs else moe.device_cb_descs
                sem_descs = moe.device_sem_descs

                program = ttnn.ProgramDescriptor(
                    kernels=kernels,
                    cbs=cb_descs,
                    semaphores=sem_descs,
                )

                moe._setup_fabric_connections(coord, row, col, reduce_root_coord, kernel_result, program)
                mesh_program_descriptor[ttnn.MeshCoordinateRange(coord, coord)] = program

        # Execute
        ttnn.generic_op(moe.io_tensors, mesh_program_descriptor)

        # Return appropriate output based on reduce mode
        if ctx.enable_reduce_to_one:
            if ctx.enable_routing:
                return ctx.gate_output_scores_tensor, ctx.gate_output_indices_tensor, ctx.reduce_output_tensor
            return ctx.reduce_output_tensor
        if ctx.enable_routing:
            return ctx.gate_output_scores_tensor, ctx.gate_output_indices_tensor, ctx.final_output_tensor
        return ctx.final_output_tensor
