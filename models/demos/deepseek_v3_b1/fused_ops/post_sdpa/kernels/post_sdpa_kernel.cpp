// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Post SDPA unified kernel with SDPA Reduce-to-All + CCL All-Reduce
// Single kernel file, compiles correctly for all RISC cores
// Each RISC has its own CTArgs struct with different compile-time arg layout
//
// Implements: SDPA Reduce-to-All + Matmul1 + Gather1 + Mcast + Matmul2 + Gather2 + CCL All-Reduce
//
// SDPA Reduce-to-All Phase:
// - SDPA Workers (8 cores): Reduce L/MS tensors across devices, scatter [1,512] to matmul1 cores
// - SDPA Forwarders (2 cores): Forward fabric packets for SDPA CCL
//
// Post-SDPA Phases:
// - Matmul1: [1, 512] x [512, 128] -> [1, 128] on 64 cores (8x8) - waits for scatter data
// - Gather1: Collect [1, 128] from 64 cores to [1, 8192] on gather core
// - Mcast: Broadcast [1, 8192] to 130 cores (13x10 grid, rectangular)
// - Matmul2: [1, 8192] x [8192, 64] -> [1, 64] on 112 active cores (rows 0-8 full 12 + row 9 cols 0-3)
// - Gather2: Collect [1, 64] from 112 active cores to [1, 7168] on gather core
// - CCL All-Reduce: Exchange [1, 7168] between devices, reduce (local + remote + residual)
//
// Note: Mcast grid (13x10=130) includes 18 inactive cores (col 12 rows 0-8 + row 9 cols 4-11)
// which receive mcast data but skip matmul2 via is_matmul2_core=false
//
// SDPA Core Layout:
// - SDPA Workers: (2,8)-(5,8), (2,9)-(5,9) = 8 cores
// - SDPA Forwarders: (6,9), (7,9) = 2 cores
// Note: Some SDPA cores overlap with matmul2 grid - they run SDPA first, then matmul2
//
// CCL Core Layout:
// - CCL Receiver = Gather core (12, 9): already has local data after Gather2
// - CCL Sender = Adjacent core (11, 9): reads from gather core, sends via fabric

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"
#include "../../../unified_kernels/matmul.hpp"
#include "../../../unified_kernels/gather.hpp"
#include "../../../unified_kernels/mcast.hpp"
#ifndef SKIP_CCL
#include "../../../unified_kernels/all_reduce_sender.hpp"
#include "../../../unified_kernels/all_reduce_receiver.hpp"
#endif

// SDPA Reduce-to-All includes (only needed for SDPA cores)
#ifndef SKIP_SDPA
#if defined(COMPILE_FOR_TRISC)
// Compute-only includes for SDPA (no fabric headers - they conflict with compute headers)
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "models/demos/deepseek_v3_b1/kernel_includes/tt_metal/include/compute_kernel_api/sdpa.h"
#endif
#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
// Dataflow-only includes for SDPA (fabric headers for packet sending/forwarding)
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
// tt_memmove for local memory copy (MS data from recv buffer to CB)
// Include common.hpp when:
// - CCL is skipped (common.hpp won't be included via all_reduce_sender.hpp), OR
// - Compiling for NCRISC (all_reduce_sender.hpp only includes common.hpp for BRISC)
#if defined(SKIP_CCL) || defined(COMPILE_FOR_NCRISC)
#include "cpp/ttnn/operations/data_movement/common/kernels/common.hpp"
#endif
using tt::data_movement::common::tt_memmove;
#endif
#endif

// Compile-time role flags for dead code elimination via if constexpr
struct Core {
    // SDPA Reduce-to-All cores (only when SDPA is enabled)
#ifndef SKIP_SDPA
    // SDPA worker cores (2,8)-(5,8), (2,9)-(5,9) = 8 cores - run SDPA reduction and scatter
    static constexpr bool is_sdpa_worker_core = get_named_compile_time_arg_val("is_sdpa_worker_core") == 1;
    // SDPA forwarder cores (6,9), (7,9) = 2 cores - forward fabric packets for SDPA CCL
    static constexpr bool is_sdpa_forwarder_core = get_named_compile_time_arg_val("is_sdpa_forwarder_core") == 1;
#else
    // When SDPA is disabled, these are always false
    static constexpr bool is_sdpa_worker_core = false;
    static constexpr bool is_sdpa_forwarder_core = false;
#endif

    // Post-SDPA cores
    // First matmul on 8x8 grid - receives scatter data from SDPA workers
    static constexpr bool is_matmul1_core = get_named_compile_time_arg_val("is_matmul1_core") == 1;
    // Gather core (12, 9) - receives gather1, sends mcast, receives gather2, CCL receiver
    static constexpr bool is_gather_receiver_core = get_named_compile_time_arg_val("is_gather_receiver_core") == 1;
    // Mcast receiver grid (13x10 = 130 cores) - receives mcast data
    static constexpr bool is_mcast_receiver_core = get_named_compile_time_arg_val("is_mcast_receiver_core") == 1;
    // Active matmul2 cores (112 cores: rows 0-8 full 12 + row 9 cols 0-3)
    static constexpr bool is_matmul2_core = get_named_compile_time_arg_val("is_matmul2_core") == 1;
    // CCL sender core (11, 9) - reads from gather core, sends via fabric
    static constexpr bool is_ccl_sender_core = get_named_compile_time_arg_val("is_ccl_sender_core") == 1;
    // CCL receiver core = gather core (12, 9) - receives remote data, performs reduction
    static constexpr bool is_ccl_receiver_core = get_named_compile_time_arg_val("is_ccl_receiver_core") == 1;
};

void kernel_main() {
// ============================================================================
// NCRISC (Reader)
// SDPA Phase:
// - SDPA worker reader (8 cores): push local L/MS, prepare R1/R2 neighbor data
// - SDPA forwarder NCRISC (2 cores): forward BWD fabric packets
// Post-SDPA Phase:
// - Matmul1 reader (8x8 grid): setup sharded buffers (after scatter arrival)
// - Gather1 sender (8x8 grid): send matmul1 output to gather core
// - Mcast receiver (13x10 grid = 130 cores): receive mcast data
// - Matmul2 reader (112 active cores): setup weights buffer
// - Gather2 sender (112 active cores): send matmul2 output to gather core
// - CCL sender (11, 9): read gather2 output from gather core
// - CCL receiver (12, 9): wait for remote data, push to compute
// ============================================================================
#if defined(COMPILE_FOR_NCRISC)
#ifndef SKIP_SDPA
    // SDPA Reader compile-time args (for SDPA worker cores)
    constexpr uint32_t sdpa_cb_local_l = get_named_compile_time_arg_val("sdpa_cb_local_l");
    constexpr uint32_t sdpa_cb_local_ms = get_named_compile_time_arg_val("sdpa_cb_local_ms");
    constexpr uint32_t sdpa_cb_r1_neighbor_l = get_named_compile_time_arg_val("sdpa_cb_r1_neighbor_l");
    constexpr uint32_t sdpa_cb_r1_neighbor_ms = get_named_compile_time_arg_val("sdpa_cb_r1_neighbor_ms");
    constexpr uint32_t sdpa_cb_r2_neighbor_l = get_named_compile_time_arg_val("sdpa_cb_r2_neighbor_l");
    constexpr uint32_t sdpa_cb_r2_neighbor_ms = get_named_compile_time_arg_val("sdpa_cb_r2_neighbor_ms");
    constexpr uint32_t sdpa_ms_tile_size_bytes = get_named_compile_time_arg_val("sdpa_ms_tile_size_bytes");
    constexpr uint32_t sdpa_l_chunk_size_bytes = get_named_compile_time_arg_val("sdpa_l_chunk_size_bytes");
    constexpr uint32_t sdpa_num_l_chunks = get_named_compile_time_arg_val("sdpa_num_l_chunks");
    constexpr uint32_t sdpa_tiles_per_l_chunk = get_named_compile_time_arg_val("sdpa_tiles_per_l_chunk");
    constexpr uint32_t sdpa_out_tiles = sdpa_num_l_chunks * sdpa_tiles_per_l_chunk;
    constexpr uint32_t sdpa_total_l_bytes = sdpa_num_l_chunks * sdpa_l_chunk_size_bytes;
    // Semaphore thresholds for SDPA reader
    constexpr uint32_t SDPA_MS_SEM_THRESHOLD = 1;
    constexpr uint32_t SDPA_L_SEM_BASE_THRESHOLD = 2;
#endif

    // Matmul1 CTArgs
    using Matmul1CTArgs = deepseek_b1_ops::Matmul::ReaderCTArgs;
    deepseek_b1_ops::Matmul::ReaderArgs matmul1_args{};

    // Gather1 sender args
    deepseek_b1_ops::Gather::SenderArgs gather1_args{
        get_named_compile_time_arg_val("gather1_dest_noc_x"),
        get_named_compile_time_arg_val("gather1_dest_noc_y"),
        get_named_compile_time_arg_val("gather1_data_size_bytes"),
        get_named_compile_time_arg_val("gather1_receiver_semaphore_id"),
        get_named_compile_time_arg_val("gather1_src_cb"),
        get_named_compile_time_arg_val("gather1_src_num_pages"),
        get_named_compile_time_arg_val("gather1_sender_grid_start_x"),
        get_named_compile_time_arg_val("gather1_sender_grid_start_y"),
        get_named_compile_time_arg_val("gather1_sender_grid_end_x"),
        get_named_compile_time_arg_val("gather1_sender_grid_end_y"),
        get_named_compile_time_arg_val("gather1_row_major"),
        get_named_compile_time_arg_val("gather1_receiver_data_addr"),
    };

    // Mcast receiver args
    using McastCTArgs = deepseek_b1_ops::Mcast::ReceiverCTArgs;
    deepseek_b1_ops::Mcast::ReceiverArgs mcast_args{
        get_named_compile_time_arg_val("mcast_data_receiver_semaphore"),
        get_named_compile_time_arg_val("mcast_dst_cb"),
        get_named_compile_time_arg_val("mcast_dst_num_pages"),
    };

    // Matmul2 CTArgs
    using Matmul2CTArgs = deepseek_b1_ops::Matmul::ReaderCTArgs;
    deepseek_b1_ops::Matmul::ReaderArgs matmul2_args{};

    // Gather2 sender args
    deepseek_b1_ops::Gather::SenderArgs gather2_args{
        get_named_compile_time_arg_val("gather2_dest_noc_x"),
        get_named_compile_time_arg_val("gather2_dest_noc_y"),
        get_named_compile_time_arg_val("gather2_data_size_bytes"),
        get_named_compile_time_arg_val("gather2_receiver_semaphore_id"),
        get_named_compile_time_arg_val("gather2_src_cb"),
        get_named_compile_time_arg_val("gather2_src_num_pages"),
        get_named_compile_time_arg_val("gather2_sender_grid_start_x"),
        get_named_compile_time_arg_val("gather2_sender_grid_start_y"),
        get_named_compile_time_arg_val("gather2_sender_grid_end_x"),
        get_named_compile_time_arg_val("gather2_sender_grid_end_y"),
        get_named_compile_time_arg_val("gather2_row_major"),
        get_named_compile_time_arg_val("gather2_receiver_data_addr"),
    };
#ifndef SKIP_CCL
    // CCL Sender NCRISC CTArgs (reads from gather core)
    using CCLSenderReaderCTArgs = deepseek_b1_ops::AllReduceSender::ReaderCTArgs<
        get_named_compile_time_arg_val("ccl_sender_cb0_id"),
        get_named_compile_time_arg_val("ccl_sender_num_tiles"),
        get_named_compile_time_arg_val("ccl_sender_tensor_page_size"),
        get_named_compile_time_arg_val("ccl_sender_data_noc_x"),
        get_named_compile_time_arg_val("ccl_sender_data_noc_y")>;

    // CCL Receiver NCRISC CTArgs (waits for remote data)
    // Note: skip_local_push=1 because gather2 already pushed to CB7 (gather2_dst_cb)
    using CCLReceiverReaderCTArgs = deepseek_b1_ops::AllReduceReceiver::ReaderCTArgs<
        get_named_compile_time_arg_val("ccl_receiver_packet_header_cb_id"),
        get_named_compile_time_arg_val("ccl_receiver_cb_in1"),
        get_named_compile_time_arg_val("ccl_receiver_l1_alignment"),
        get_named_compile_time_arg_val("ccl_receiver_cb_in2"),
        get_named_compile_time_arg_val("ccl_receiver_remote_sender_noc_x"),
        get_named_compile_time_arg_val("ccl_receiver_remote_sender_noc_y"),
        get_named_compile_time_arg_val("ccl_receiver_num_standard_tiles"),
        get_named_compile_time_arg_val("ccl_receiver_cb_residual"),
        get_named_compile_time_arg_val("ccl_receiver_has_residual"),
        get_named_compile_time_arg_val("ccl_receiver_skip_local_push")>;
#endif
// ============================================================================
// BRISC (Writer)
// - Gather1 receiver (gather core): receive from 8x8 grid
// - Mcast sender (gather core): broadcast to 13x10 grid (130 cores)
// - Gather2 receiver (gather core): receive from 112 active matmul2 cores
// - CCL sender (11, 9): send gather2 output via fabric
// ============================================================================
#elif defined(COMPILE_FOR_BRISC)
    // Matmul1/2 CTArgs (BRISC is no-op for matmul)
    using Matmul1CTArgs = deepseek_b1_ops::Matmul::WriterCTArgs;
    using Matmul2CTArgs = deepseek_b1_ops::Matmul::WriterCTArgs;
    deepseek_b1_ops::Matmul::WriterArgs matmul1_args{};
    deepseek_b1_ops::Matmul::WriterArgs matmul2_args{};

    // Gather1 receiver args
    deepseek_b1_ops::Gather::ReceiverArgs gather1_args{
        get_named_compile_time_arg_val("gather1_noc0_num_senders"),
        get_named_compile_time_arg_val("gather1_noc1_num_senders"),
        get_named_compile_time_arg_val("gather1_noc0_receiver_semaphore_id"),
        get_named_compile_time_arg_val("gather1_noc1_receiver_semaphore_id"),
        get_named_compile_time_arg_val("gather1_dst_cb"),
        get_named_compile_time_arg_val("gather1_dst_num_pages"),
    };

    // Mcast sender args
    using McastCTArgs = deepseek_b1_ops::Mcast::SenderCTArgs<
        get_named_compile_time_arg_val("mcast_num_cores"),
        get_named_compile_time_arg_val("mcast_is_part_of_receiver_grid") == 1,
        false>;  // loopback = false

    constexpr uint32_t mcast_src_cb = get_named_compile_time_arg_val("mcast_src_cb");
    constexpr uint32_t mcast_dst_cb = get_named_compile_time_arg_val("mcast_dst_cb");
    deepseek_b1_ops::Mcast::SenderArgs mcast_args{
        get_named_compile_time_arg_val("mcast_dest_noc_start_x"),
        get_named_compile_time_arg_val("mcast_dest_noc_start_y"),
        get_named_compile_time_arg_val("mcast_dest_noc_end_x"),
        get_named_compile_time_arg_val("mcast_dest_noc_end_y"),
        get_named_compile_time_arg_val("mcast_data_sender_semaphore"),
        get_named_compile_time_arg_val("mcast_data_receiver_semaphore"),
        get_named_compile_time_arg_val("mcast_data_size_bytes"),
        mcast_src_cb,
        get_named_compile_time_arg_val("mcast_src_num_pages"),
        get_read_ptr(mcast_src_cb),
        get_write_ptr(mcast_dst_cb),  // CB 4 address (now allocated on gather core too)
    };

    // Gather2 receiver args
    deepseek_b1_ops::Gather::ReceiverArgs gather2_args{
        get_named_compile_time_arg_val("gather2_noc0_num_senders"),
        get_named_compile_time_arg_val("gather2_noc1_num_senders"),
        get_named_compile_time_arg_val("gather2_noc0_receiver_semaphore_id"),
        get_named_compile_time_arg_val("gather2_noc1_receiver_semaphore_id"),
        get_named_compile_time_arg_val("gather2_dst_cb"),
        get_named_compile_time_arg_val("gather2_dst_num_pages"),
    };

#ifndef SKIP_CCL
    // CCL Sender BRISC CTArgs (sends via fabric)
    using CCLSenderWriterCTArgs = deepseek_b1_ops::AllReduceSender::WriterCTArgs<
        get_named_compile_time_arg_val("ccl_sender_packet_header_cb_id"),
        get_named_compile_time_arg_val("ccl_sender_packet_cb_id"),
        get_named_compile_time_arg_val("ccl_sender_l1_alignment"),
        get_named_compile_time_arg_val("ccl_sender_input_num_tiles"),
        get_named_compile_time_arg_val("ccl_sender_page_size_bytes"),
        get_named_compile_time_arg_val("ccl_sender_payload_size_bytes"),
        get_named_compile_time_arg_val("ccl_sender_data_noc_x"),
        get_named_compile_time_arg_val("ccl_sender_data_noc_y"),
        get_named_compile_time_arg_val("ccl_sender_remote_receiver_noc_x"),
        get_named_compile_time_arg_val("ccl_sender_remote_receiver_noc_y"),
        get_named_compile_time_arg_val("ccl_sender_dst_num_hops"),
        get_named_compile_time_arg_val("ccl_sender_num_connections")>;
#endif
// ============================================================================
// TRISC (Compute)
// - Matmul1 compute (8x8 grid)
// - Matmul2 compute (112 active cores)
// - CCL receiver compute (gather core): reduction (local + remote + residual)
// ============================================================================
#elif defined(COMPILE_FOR_TRISC)
    // Matmul1 CTArgs
    using Matmul1CTArgs =
        deepseek_b1_ops::Matmul::ComputeCTArgs<get_named_compile_time_arg_val("matmul1_out_w_per_core")>;
    deepseek_b1_ops::Matmul::ComputeArgs matmul1_args{
        get_named_compile_time_arg_val("matmul1_in0"),
        get_named_compile_time_arg_val("matmul1_in1"),
        get_named_compile_time_arg_val("matmul1_out"),
        get_named_compile_time_arg_val("matmul1_k_num_tiles"),
    };

    // Gather1 compute args (no-op)
    deepseek_b1_ops::Gather::ComputeArgs gather1_args{};

    // Mcast CTArgs (no-op)
    using McastCTArgs = deepseek_b1_ops::Mcast::ComputeCTArgs;
    deepseek_b1_ops::Mcast::ComputeArgs mcast_args{};

    // Matmul2 CTArgs
    using Matmul2CTArgs =
        deepseek_b1_ops::Matmul::ComputeCTArgs<get_named_compile_time_arg_val("matmul2_out_w_per_core")>;
    deepseek_b1_ops::Matmul::ComputeArgs matmul2_args{
        get_named_compile_time_arg_val("matmul2_in0"),
        get_named_compile_time_arg_val("matmul2_in1"),
        get_named_compile_time_arg_val("matmul2_out"),
        get_named_compile_time_arg_val("matmul2_k_num_tiles"),
    };

    // Gather2 compute args (no-op)
    deepseek_b1_ops::Gather::ComputeArgs gather2_args{};

#ifndef SKIP_CCL
    // CCL Receiver compute CTArgs (reduction)
    using CCLReceiverComputeCTArgs = deepseek_b1_ops::AllReduceReceiver::ComputeCTArgs<
        get_named_compile_time_arg_val("ccl_receiver_cb_in0"),
        get_named_compile_time_arg_val("ccl_receiver_cb_in1"),
        get_named_compile_time_arg_val("ccl_receiver_cb_out0"),
        get_named_compile_time_arg_val("ccl_receiver_cb_residual"),
        get_named_compile_time_arg_val("ccl_receiver_cb_temp"),
        get_named_compile_time_arg_val("ccl_receiver_has_residual"),
        get_named_compile_time_arg_val("ccl_receiver_num_tiles")>;
#endif
    // Full init, CBs don't matter
    compute_kernel_hw_startup(0, 0, 0);
#endif

    // ========================================================================
    // Setup sharded buffers (NCRISC only)
    // ========================================================================
#if defined(COMPILE_FOR_NCRISC)
    // Matmul1 buffers (8x8 grid)
    // NOTE: When SDPA is enabled, matmul1 waits for scatter data arrival before setup
    if constexpr (Core::is_matmul1_core) {
#ifndef SKIP_SDPA
        // Wait for SDPA scatter to deliver data to matmul1_in0
        // Each SDPA worker signals this semaphore after scatter write completes
        constexpr uint32_t scatter_arrival_semaphore_id =
            get_named_compile_time_arg_val("scatter_arrival_semaphore_id");
        volatile tt_l1_ptr uint32_t* scatter_arrival_sem_addr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(scatter_arrival_semaphore_id));
        noc_semaphore_wait(scatter_arrival_sem_addr, 1);
        noc_semaphore_set(scatter_arrival_sem_addr, 0);
#endif

        constexpr uint32_t matmul1_in0 = get_named_compile_time_arg_val("matmul1_in0");
        constexpr uint32_t matmul1_k_num_tiles = get_named_compile_time_arg_val("matmul1_k_num_tiles");
        unified_kernels::setup_sharded_buffer(matmul1_in0, matmul1_k_num_tiles);

        constexpr uint32_t matmul1_in1 = get_named_compile_time_arg_val("matmul1_in1");
        constexpr uint32_t matmul1_out_w_per_core = get_named_compile_time_arg_val("matmul1_out_w_per_core");
        unified_kernels::setup_sharded_buffer(matmul1_in1, matmul1_k_num_tiles * matmul1_out_w_per_core);
    }

    // Matmul2 buffers (112 active cores) - weights only, input comes from mcast
    if constexpr (Core::is_matmul2_core) {
        constexpr uint32_t matmul2_in1 = get_named_compile_time_arg_val("matmul2_in1");
        constexpr uint32_t matmul2_k_num_tiles = get_named_compile_time_arg_val("matmul2_k_num_tiles");
        constexpr uint32_t matmul2_out_w_per_core = get_named_compile_time_arg_val("matmul2_out_w_per_core");
        unified_kernels::setup_sharded_buffer(matmul2_in1, matmul2_k_num_tiles * matmul2_out_w_per_core);
    }
#endif

#ifndef SKIP_SDPA
    // ========================================================================
    // SDPA REDUCE-TO-ALL PHASE
    // SDPA worker cores (8): reduce L/MS across devices, scatter to matmul1 cores
    // SDPA forwarder cores (2): forward fabric packets for SDPA CCL
    // ========================================================================
    {
        DeviceZoneScopedN("SDPA_REDUCE_TO_ALL");

#if defined(COMPILE_FOR_NCRISC)
        // SDPA Reader: push local input, prepare neighbor data for compute
        if constexpr (Core::is_sdpa_worker_core) {
            // Runtime args for SDPA reader
            size_t sdpa_rt_arg_idx = 0;
            const uint32_t sdpa_r1_neighbor_sem_addr = get_arg_val<uint32_t>(sdpa_rt_arg_idx++);
            const uint32_t sdpa_r2_neighbor_sem_addr = get_arg_val<uint32_t>(sdpa_rt_arg_idx++);
            const uint32_t sdpa_r1_recv_buffer_addr = get_arg_val<uint32_t>(sdpa_rt_arg_idx++);
            const uint32_t sdpa_r2_recv_buffer_addr = get_arg_val<uint32_t>(sdpa_rt_arg_idx++);

            // Push local input (aliased CBs, no copy needed)
            cb_reserve_back(sdpa_cb_local_l, sdpa_out_tiles);
            cb_push_back(sdpa_cb_local_l, sdpa_out_tiles);
            cb_reserve_back(sdpa_cb_local_ms, 1);
            cb_push_back(sdpa_cb_local_ms, 1);

            // Helper lambda to prepare neighbor data for compute
            auto prepare_sdpa_data = [&](uint32_t cb_l, uint32_t cb_ms, uint32_t sem_addr, uint32_t recv_buffer_addr) {
                volatile tt_l1_ptr uint32_t* sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);

                DPRINT << "SDPA Reader: sem_addr=" << sem_addr << " recv_buffer=" << recv_buffer_addr << ENDL();
                DPRINT << "SDPA Reader: waiting for MS, current sem=" << *sem_ptr << ENDL();

                // MS first (sem >= 1)
                cb_reserve_back(cb_ms, 1);
                noc_semaphore_wait_min(sem_ptr, SDPA_MS_SEM_THRESHOLD);
                // MS is at end of buffer (offset = total_l_bytes)
                // Use tt_memmove for local memory copy (same core L1 to L1)
                uint32_t ms_src_addr = recv_buffer_addr + sdpa_total_l_bytes;
                tt_memmove<true, false, false, 0>(get_write_ptr(cb_ms), ms_src_addr, sdpa_ms_tile_size_bytes);
                cb_push_back(cb_ms, 1);

                // L chunks (sem >= 2, 3, 4, ...)
                for (uint32_t i = 0; i < sdpa_num_l_chunks; i++) {
                    cb_reserve_back(cb_l, sdpa_tiles_per_l_chunk);
                    noc_semaphore_wait_min(sem_ptr, SDPA_L_SEM_BASE_THRESHOLD + i);
                    // L CB is aliased to buffer, just push (zero-copy)
                    cb_push_back(cb_l, sdpa_tiles_per_l_chunk);
                }
                noc_semaphore_set(sem_ptr, 0);
            };

            // Prepare R1 neighbor data
            prepare_sdpa_data(
                sdpa_cb_r1_neighbor_l, sdpa_cb_r1_neighbor_ms, sdpa_r1_neighbor_sem_addr, sdpa_r1_recv_buffer_addr);

            // Prepare R2 neighbor data
            prepare_sdpa_data(
                sdpa_cb_r2_neighbor_l, sdpa_cb_r2_neighbor_ms, sdpa_r2_neighbor_sem_addr, sdpa_r2_recv_buffer_addr);
        }

        // SDPA Forwarder NCRISC: forward BWD direction fabric packets
        if constexpr (Core::is_sdpa_forwarder_core) {
            DPRINT << "SDPA Forwarder NCRISC: Start" << ENDL();
            // Forwarder logic handled separately - runs on BRISC for FWD, NCRISC for BWD
            constexpr uint32_t sdpa_fwd_slots_per_round = get_named_compile_time_arg_val("sdpa_fwd_slots_per_round");
            constexpr uint32_t sdpa_fwd_slot_size = get_named_compile_time_arg_val("sdpa_fwd_slot_size");
            constexpr uint32_t sdpa_fwd_r2_buffer_offset = get_named_compile_time_arg_val("sdpa_fwd_r2_buffer_offset");
            constexpr uint32_t sdpa_fwd_all_sent_mask =
                (sdpa_fwd_slots_per_round == 32) ? 0xFFFFFFFFu : ((1u << sdpa_fwd_slots_per_round) - 1u);

            size_t sdpa_fwd_rt_arg_idx = 0;
            const uint32_t sdpa_fwd_buffer_base = get_arg_val<uint32_t>(sdpa_fwd_rt_arg_idx++);
            const uint32_t sdpa_fwd_buffer_offset = get_arg_val<uint32_t>(sdpa_fwd_rt_arg_idx++);
            const uint32_t sdpa_fwd_r1_sem_addr = get_semaphore(get_arg_val<uint32_t>(sdpa_fwd_rt_arg_idx++));
            const uint32_t sdpa_fwd_r2_sem_addr = get_semaphore(get_arg_val<uint32_t>(sdpa_fwd_rt_arg_idx++));

            const uint32_t my_buffer_base = sdpa_fwd_buffer_base + sdpa_fwd_buffer_offset;

            auto sdpa_fabric_connection =
                tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(
                    sdpa_fwd_rt_arg_idx);
            sdpa_fabric_connection.open();

            volatile tt_l1_ptr uint32_t* r1_sem_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sdpa_fwd_r1_sem_addr);
            volatile tt_l1_ptr uint32_t* r2_sem_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sdpa_fwd_r2_sem_addr);

            const uint32_t r1_buffer_base = my_buffer_base;
            const uint32_t r2_buffer_base = my_buffer_base + sdpa_fwd_r2_buffer_offset;

            uint32_t r1_sent_mask = 0;
            uint32_t r2_sent_mask = 0;
            // Forward packets as they arrive
            do {
                invalidate_l1_cache();
                // Process R1 slots
                uint32_t r1_sem_value = *r1_sem_ptr;
                uint32_t r1_pending = r1_sem_value & ~r1_sent_mask;
                while (r1_pending != 0) {
                    uint32_t slot = __builtin_ctz(r1_pending);
                    uint32_t slot_addr = r1_buffer_base + (slot * sdpa_fwd_slot_size);
                    auto* packet_header = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(slot_addr);
                    uint32_t actual_packet_size = packet_header->get_payload_size_including_header();
                    sdpa_fabric_connection.wait_for_empty_write_slot();
                    sdpa_fabric_connection.send_payload_flush_non_blocking_from_address(slot_addr, actual_packet_size);
                    r1_sent_mask |= (1u << slot);
                    r1_pending &= ~(1u << slot);
                }

                // Process R2 slots
                uint32_t r2_sem_value = *r2_sem_ptr;
                uint32_t r2_pending = r2_sem_value & ~r2_sent_mask;
                while (r2_pending != 0) {
                    uint32_t slot = __builtin_ctz(r2_pending);
                    uint32_t slot_addr = r2_buffer_base + (slot * sdpa_fwd_slot_size);
                    auto* packet_header = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(slot_addr);
                    uint32_t actual_packet_size = packet_header->get_payload_size_including_header();
                    sdpa_fabric_connection.wait_for_empty_write_slot();
                    sdpa_fabric_connection.send_payload_flush_non_blocking_from_address(slot_addr, actual_packet_size);
                    r2_sent_mask |= (1u << slot);
                    r2_pending &= ~(1u << slot);
                }
            } while (r1_sent_mask != sdpa_fwd_all_sent_mask || r2_sent_mask != sdpa_fwd_all_sent_mask);

            sdpa_fabric_connection.close();
            noc_async_full_barrier();
            DPRINT << "SDPA Forwarder NCRISC: End" << ENDL();
        }
#endif  // COMPILE_FOR_NCRISC

#if defined(COMPILE_FOR_BRISC)
        // SDPA Writer: send R1/R2 packets, scatter to matmul1 cores
        if constexpr (Core::is_sdpa_worker_core) {
            constexpr uint32_t sdpa_cb_r1_result_l = get_named_compile_time_arg_val("sdpa_cb_r1_result_l");
            constexpr uint32_t sdpa_cb_r1_result_ms = get_named_compile_time_arg_val("sdpa_cb_r1_result_ms");
            constexpr uint32_t sdpa_cb_packet_slot = get_named_compile_time_arg_val("sdpa_cb_packet_slot");
            constexpr uint32_t sdpa_l1_alignment = get_named_compile_time_arg_val("sdpa_l1_alignment");
            constexpr uint32_t sdpa_page_size_bytes = get_named_compile_time_arg_val("sdpa_page_size_bytes");
            constexpr uint32_t sdpa_slot_size = get_named_compile_time_arg_val("sdpa_slot_size");
            constexpr uint32_t sdpa_cb_l_out = get_named_compile_time_arg_val("sdpa_cb_l_out");
            constexpr uint32_t sdpa_scatter_num_tiles = get_named_compile_time_arg_val("sdpa_scatter_num_tiles");
            constexpr uint32_t sdpa_scatter_src_tile_size =
                get_named_compile_time_arg_val("sdpa_scatter_src_tile_size");
            constexpr uint32_t sdpa_scatter_dst_tile_size =
                get_named_compile_time_arg_val("sdpa_scatter_dst_tile_size");
            constexpr uint32_t sdpa_scatter_face_size = get_named_compile_time_arg_val("sdpa_scatter_face_size");
            constexpr uint32_t sdpa_scatter_row_face_size =
                get_named_compile_time_arg_val("sdpa_scatter_row_face_size");
            constexpr uint32_t sdpa_scatter_num_rows = get_named_compile_time_arg_val("sdpa_scatter_num_rows");
            constexpr uint32_t scatter_arrival_semaphore_id =
                get_named_compile_time_arg_val("scatter_arrival_semaphore_id");
            // Additional compile-time args needed for R1/R2 packet sending
            constexpr uint32_t sdpa_cb_local_l = get_named_compile_time_arg_val("sdpa_cb_local_l");
            constexpr uint32_t sdpa_cb_local_ms = get_named_compile_time_arg_val("sdpa_cb_local_ms");
            constexpr uint32_t sdpa_ms_tile_size_bytes = get_named_compile_time_arg_val("sdpa_ms_tile_size_bytes");
            constexpr uint32_t sdpa_l_chunk_size_bytes = get_named_compile_time_arg_val("sdpa_l_chunk_size_bytes");
            constexpr uint32_t sdpa_num_l_chunks = get_named_compile_time_arg_val("sdpa_num_l_chunks");
            constexpr uint32_t sdpa_tiles_per_l_chunk = get_named_compile_time_arg_val("sdpa_tiles_per_l_chunk");
            constexpr uint32_t sdpa_total_l_bytes = sdpa_num_l_chunks * sdpa_l_chunk_size_bytes;

            static constexpr size_t packet_header_size_bytes = sizeof(PACKET_HEADER_TYPE);

            // Runtime args
            size_t sdpa_wr_rt_arg_idx = 0;
            const uint32_t r1_dst_mesh_id = get_arg_val<uint32_t>(sdpa_wr_rt_arg_idx++);
            const uint32_t r1_dst_chip_id = get_arg_val<uint32_t>(sdpa_wr_rt_arg_idx++);
            const uint32_t r1_neighbor_dst_addr = get_arg_val<uint32_t>(sdpa_wr_rt_arg_idx++);
            const uint32_t r1_neighbor_sem_addr = get_arg_val<uint32_t>(sdpa_wr_rt_arg_idx++);

            const uint32_t r2_dst_mesh_id = get_arg_val<uint32_t>(sdpa_wr_rt_arg_idx++);
            const uint32_t r2_dst_chip_id = get_arg_val<uint32_t>(sdpa_wr_rt_arg_idx++);
            const uint32_t r2_neighbor_dst_addr = get_arg_val<uint32_t>(sdpa_wr_rt_arg_idx++);
            const uint32_t r2_neighbor_sem_addr = get_arg_val<uint32_t>(sdpa_wr_rt_arg_idx++);
            const uint32_t current_core_x = get_arg_val<uint32_t>(sdpa_wr_rt_arg_idx++);
            const uint32_t current_core_y = get_arg_val<uint32_t>(sdpa_wr_rt_arg_idx++);
            const uint32_t fwd_core_x = get_arg_val<uint32_t>(sdpa_wr_rt_arg_idx++);
            const uint32_t fwd_core_y = get_arg_val<uint32_t>(sdpa_wr_rt_arg_idx++);
            const uint32_t r1_fwd_slot_addr = get_arg_val<uint32_t>(sdpa_wr_rt_arg_idx++);
            const uint32_t r1_fwd_sem_addr = get_semaphore(get_arg_val<uint32_t>(sdpa_wr_rt_arg_idx++));
            const uint32_t r1_base_slot_idx = get_arg_val<uint32_t>(sdpa_wr_rt_arg_idx++);
            const uint32_t r2_fwd_slot_addr = get_arg_val<uint32_t>(sdpa_wr_rt_arg_idx++);
            const uint32_t r2_fwd_sem_addr = get_semaphore(get_arg_val<uint32_t>(sdpa_wr_rt_arg_idx++));
            const uint32_t r2_base_slot_idx = get_arg_val<uint32_t>(sdpa_wr_rt_arg_idx++);

            // Scatter runtime args
            const uint32_t scatter_dest_l1_addr = get_arg_val<uint32_t>(sdpa_wr_rt_arg_idx++);
            uint32_t scatter_dest_noc_x[sdpa_scatter_num_rows];
            uint32_t scatter_dest_noc_y[sdpa_scatter_num_rows];
            for (uint32_t i = 0; i < sdpa_scatter_num_rows; i++) {
                scatter_dest_noc_x[i] = get_arg_val<uint32_t>(sdpa_wr_rt_arg_idx++);
                scatter_dest_noc_y[i] = get_arg_val<uint32_t>(sdpa_wr_rt_arg_idx++);
            }

            // Helper: send packet via forwarder
            auto send_sdpa_packet = [&](uint32_t src_addr,
                                        uint32_t payload_size,
                                        uint32_t dst_addr,
                                        uint32_t sem_addr,
                                        uint32_t dst_mesh_id,
                                        uint32_t dst_chip_id,
                                        uint32_t fwd_slot_addr,
                                        uint32_t fwd_sem_addr,
                                        uint32_t slot_idx) {
                cb_reserve_back(sdpa_cb_packet_slot, 1);
                uint32_t header_addr = get_write_ptr(sdpa_cb_packet_slot);

                auto* header = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(header_addr);
                (void)fabric_set_unicast_route(header, dst_chip_id, dst_mesh_id);

                uint64_t dst_noc = get_noc_addr(current_core_x, current_core_y, dst_addr);
                uint64_t sem_noc = get_noc_addr(current_core_x, current_core_y, sem_addr);
                header->to_noc_fused_unicast_write_atomic_inc(
                    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst_noc, sem_noc, 1, false},
                    align(payload_size, sdpa_l1_alignment));

                uint64_t fwd_slot_noc = get_noc_addr(fwd_core_x, fwd_core_y, fwd_slot_addr);
                noc_async_write(header_addr, fwd_slot_noc, packet_header_size_bytes);
                noc_async_write(src_addr, fwd_slot_noc + packet_header_size_bytes, payload_size);
                noc_async_writes_flushed();

                uint64_t fwd_sem_noc = get_noc_addr(fwd_core_x, fwd_core_y, fwd_sem_addr);
                noc_semaphore_inc(fwd_sem_noc, 1u << slot_idx);

                cb_push_back(sdpa_cb_packet_slot, 1);
                cb_pop_front(sdpa_cb_packet_slot, 1);
            };

            // Round 1: Send local input to R1 neighbor
            // MS first (at end of buffer)
            send_sdpa_packet(
                get_read_ptr(sdpa_cb_local_ms),
                sdpa_ms_tile_size_bytes,
                r1_neighbor_dst_addr + sdpa_total_l_bytes,
                r1_neighbor_sem_addr,
                r1_dst_mesh_id,
                r1_dst_chip_id,
                r1_fwd_slot_addr,
                r1_fwd_sem_addr,
                r1_base_slot_idx);
            // L chunks
            for (uint32_t i = 0; i < sdpa_num_l_chunks; i++) {
                uint32_t src_addr = get_read_ptr(sdpa_cb_local_l) + i * sdpa_l_chunk_size_bytes;
                uint32_t dst_addr = r1_neighbor_dst_addr + i * sdpa_l_chunk_size_bytes;
                send_sdpa_packet(
                    src_addr,
                    sdpa_l_chunk_size_bytes,
                    dst_addr,
                    r1_neighbor_sem_addr,
                    r1_dst_mesh_id,
                    r1_dst_chip_id,
                    r1_fwd_slot_addr + (1 + i) * sdpa_slot_size,
                    r1_fwd_sem_addr,
                    r1_base_slot_idx + 1 + i);
            }

            // Round 2: Send R1 result to R2 neighbor (streaming - wait for compute)
            // MS first
            cb_wait_front(sdpa_cb_r1_result_ms, 1);
            send_sdpa_packet(
                get_read_ptr(sdpa_cb_r1_result_ms),
                sdpa_ms_tile_size_bytes,
                r2_neighbor_dst_addr + sdpa_total_l_bytes,
                r2_neighbor_sem_addr,
                r2_dst_mesh_id,
                r2_dst_chip_id,
                r2_fwd_slot_addr,
                r2_fwd_sem_addr,
                r2_base_slot_idx);
            // L chunks (streaming)
            for (uint32_t i = 0; i < sdpa_num_l_chunks; i++) {
                cb_wait_front(sdpa_cb_r1_result_l, (i + 1) * sdpa_tiles_per_l_chunk);
                uint32_t src_addr = get_read_ptr(sdpa_cb_r1_result_l) + i * sdpa_l_chunk_size_bytes;
                uint32_t dst_addr = r2_neighbor_dst_addr + i * sdpa_l_chunk_size_bytes;
                send_sdpa_packet(
                    src_addr,
                    sdpa_l_chunk_size_bytes,
                    dst_addr,
                    r2_neighbor_sem_addr,
                    r2_dst_mesh_id,
                    r2_dst_chip_id,
                    r2_fwd_slot_addr + (1 + i) * sdpa_slot_size,
                    r2_fwd_sem_addr,
                    r2_base_slot_idx + 1 + i);
            }

            // Pop R1 result MS now that we've sent it.
            // TRISC R2 finalize deliberately does NOT pop this CB to avoid a race
            // where TRISC pops it before BRISC reads it.
            cb_pop_front(sdpa_cb_r1_result_ms, 1);
            noc_async_full_barrier();

            // SCATTER PHASE: Distribute output rows to matmul1 cores
            if constexpr (sdpa_scatter_num_rows > 0) {
                // Wait for all compute output
                cb_wait_front(sdpa_cb_l_out, sdpa_scatter_num_tiles);
                uint32_t src_base = get_read_ptr(sdpa_cb_l_out);
                uint32_t temp_base = get_read_ptr(sdpa_cb_r1_result_l);  // Reuse as scratch

                constexpr uint32_t row_face_words = sdpa_scatter_row_face_size / sizeof(uint32_t);
                constexpr uint32_t scatter_payload_bytes = sdpa_scatter_num_tiles * sdpa_scatter_dst_tile_size;

                for (uint32_t row = 0; row < sdpa_scatter_num_rows; row++) {
                    // Reorder: extract row from each source tile
                    for (uint32_t t = 0; t < sdpa_scatter_num_tiles; t++) {
                        uint32_t src_tile = src_base + t * sdpa_scatter_src_tile_size;
                        uint32_t dst_tile = temp_base + t * sdpa_scatter_dst_tile_size;

                        // Copy Face 0 row
                        volatile tt_l1_ptr uint32_t* src_f0 =
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_tile + row * sdpa_scatter_row_face_size);
                        volatile tt_l1_ptr uint32_t* dst_f0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_tile);
                        for (uint32_t w = 0; w < row_face_words; w++) {
                            dst_f0[w] = src_f0[w];
                        }

                        // Copy Face 1 row
                        volatile tt_l1_ptr uint32_t* src_f1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                            src_tile + sdpa_scatter_face_size + row * sdpa_scatter_row_face_size);
                        volatile tt_l1_ptr uint32_t* dst_f1 =
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_tile + sdpa_scatter_row_face_size);
                        for (uint32_t w = 0; w < row_face_words; w++) {
                            dst_f1[w] = src_f1[w];
                        }
                    }

                    // Write reordered row to matmul1 core
                    uint64_t dest_noc_addr =
                        get_noc_addr(scatter_dest_noc_x[row], scatter_dest_noc_y[row], scatter_dest_l1_addr);
                    noc_async_write(temp_base, dest_noc_addr, scatter_payload_bytes);
                    noc_async_writes_flushed();

                    // Signal matmul1 core that scatter data arrived
                    uint64_t matmul1_sem_addr = get_noc_addr(
                        scatter_dest_noc_x[row], scatter_dest_noc_y[row], get_semaphore(scatter_arrival_semaphore_id));
                    noc_semaphore_inc(matmul1_sem_addr, 1);
                }

                noc_async_write_barrier();
            }
        }

        // SDPA Forwarder BRISC: forward FWD direction fabric packets
        if constexpr (Core::is_sdpa_forwarder_core) {
            DPRINT << "SDPA Forwarder BRISC: Start" << ENDL();
            constexpr uint32_t sdpa_fwd_slots_per_round = get_named_compile_time_arg_val("sdpa_fwd_slots_per_round");
            constexpr uint32_t sdpa_fwd_slot_size = get_named_compile_time_arg_val("sdpa_fwd_slot_size");
            constexpr uint32_t sdpa_fwd_r2_buffer_offset = get_named_compile_time_arg_val("sdpa_fwd_r2_buffer_offset");
            constexpr uint32_t sdpa_fwd_all_sent_mask =
                (sdpa_fwd_slots_per_round == 32) ? 0xFFFFFFFFu : ((1u << sdpa_fwd_slots_per_round) - 1u);

            size_t sdpa_fwd_rt_arg_idx = 0;
            const uint32_t sdpa_fwd_buffer_base = get_arg_val<uint32_t>(sdpa_fwd_rt_arg_idx++);
            const uint32_t sdpa_fwd_buffer_offset = get_arg_val<uint32_t>(sdpa_fwd_rt_arg_idx++);
            const uint32_t sdpa_fwd_r1_sem_addr = get_semaphore(get_arg_val<uint32_t>(sdpa_fwd_rt_arg_idx++));
            const uint32_t sdpa_fwd_r2_sem_addr = get_semaphore(get_arg_val<uint32_t>(sdpa_fwd_rt_arg_idx++));

            const uint32_t my_buffer_base = sdpa_fwd_buffer_base + sdpa_fwd_buffer_offset;

            auto sdpa_fabric_connection =
                tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(
                    sdpa_fwd_rt_arg_idx);
            sdpa_fabric_connection.open();

            volatile tt_l1_ptr uint32_t* r1_sem_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sdpa_fwd_r1_sem_addr);
            volatile tt_l1_ptr uint32_t* r2_sem_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sdpa_fwd_r2_sem_addr);

            const uint32_t r1_buffer_base = my_buffer_base;
            const uint32_t r2_buffer_base = my_buffer_base + sdpa_fwd_r2_buffer_offset;

            uint32_t r1_sent_mask = 0;
            uint32_t r2_sent_mask = 0;

            do {
                invalidate_l1_cache();
                // Process R1 slots
                uint32_t r1_sem_value = *r1_sem_ptr;
                uint32_t r1_pending = r1_sem_value & ~r1_sent_mask;
                while (r1_pending != 0) {
                    uint32_t slot = __builtin_ctz(r1_pending);
                    uint32_t slot_addr = r1_buffer_base + (slot * sdpa_fwd_slot_size);
                    auto* packet_header = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(slot_addr);
                    uint32_t actual_packet_size = packet_header->get_payload_size_including_header();
                    sdpa_fabric_connection.wait_for_empty_write_slot();
                    sdpa_fabric_connection.send_payload_flush_non_blocking_from_address(slot_addr, actual_packet_size);
                    r1_sent_mask |= (1u << slot);
                    r1_pending &= ~(1u << slot);
                }

                // Process R2 slots
                uint32_t r2_sem_value = *r2_sem_ptr;
                uint32_t r2_pending = r2_sem_value & ~r2_sent_mask;
                while (r2_pending != 0) {
                    uint32_t slot = __builtin_ctz(r2_pending);
                    uint32_t slot_addr = r2_buffer_base + (slot * sdpa_fwd_slot_size);
                    auto* packet_header = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(slot_addr);
                    uint32_t actual_packet_size = packet_header->get_payload_size_including_header();
                    sdpa_fabric_connection.wait_for_empty_write_slot();
                    sdpa_fabric_connection.send_payload_flush_non_blocking_from_address(slot_addr, actual_packet_size);
                    r2_sent_mask |= (1u << slot);
                    r2_pending &= ~(1u << slot);
                }
            } while (r1_sent_mask != sdpa_fwd_all_sent_mask || r2_sent_mask != sdpa_fwd_all_sent_mask);

            sdpa_fabric_connection.close();
            noc_async_full_barrier();
            DPRINT << "SDPA Forwarder BRISC: End" << ENDL();
        }
#endif  // COMPILE_FOR_BRISC

#if defined(COMPILE_FOR_TRISC)
        // SDPA Compute: R1 and R2 reductions
        if constexpr (Core::is_sdpa_worker_core) {
            DPRINT << "SDPA Compute TRISC: Start" << ENDL();
            // Compile-time args for SDPA compute
            constexpr uint32_t sdpa_scale_fp32 = get_named_compile_time_arg_val("sdpa_scale_fp32");
            constexpr uint32_t sdpa_cb_local_l = get_named_compile_time_arg_val("sdpa_cb_local_l");
            constexpr uint32_t sdpa_cb_local_ms = get_named_compile_time_arg_val("sdpa_cb_local_ms");
            constexpr uint32_t sdpa_cb_r1_neighbor_l = get_named_compile_time_arg_val("sdpa_cb_r1_neighbor_l");
            constexpr uint32_t sdpa_cb_r1_neighbor_ms = get_named_compile_time_arg_val("sdpa_cb_r1_neighbor_ms");
            constexpr uint32_t sdpa_cb_r1_result_l = get_named_compile_time_arg_val("sdpa_cb_r1_result_l");
            constexpr uint32_t sdpa_cb_r1_result_ms = get_named_compile_time_arg_val("sdpa_cb_r1_result_ms");
            constexpr uint32_t sdpa_cb_r2_neighbor_l = get_named_compile_time_arg_val("sdpa_cb_r2_neighbor_l");
            constexpr uint32_t sdpa_cb_r2_neighbor_ms = get_named_compile_time_arg_val("sdpa_cb_r2_neighbor_ms");
            constexpr uint32_t sdpa_cb_l_out = get_named_compile_time_arg_val("sdpa_cb_l_out");
            constexpr uint32_t sdpa_cb_ms_out = get_named_compile_time_arg_val("sdpa_cb_ms_out");
            constexpr uint32_t sdpa_num_l_chunks = get_named_compile_time_arg_val("sdpa_num_l_chunks");
            constexpr uint32_t sdpa_tiles_per_l_chunk = get_named_compile_time_arg_val("sdpa_tiles_per_l_chunk");

            constexpr int vector_mode = VectorMode::RC_custom;
            constexpr bool SDPA_EXP_APPROX_MODE = false;

            binary_op_init_common(sdpa_cb_local_l, sdpa_cb_local_l, sdpa_cb_l_out);
            exp_tile_init<SDPA_EXP_APPROX_MODE, false>();

            // R1: reduce(local, r1_neighbor) -> r1_result (non-final, outputs L and MS)
            // MS reduction + L block processing with streaming
            ckernel::
                sdpa_tail_ms_reduce<SDPA_EXP_APPROX_MODE, false, sdpa_tiles_per_l_chunk, sdpa_scale_fp32, vector_mode>(
                    sdpa_cb_r1_neighbor_ms, sdpa_cb_local_ms, sdpa_cb_r1_result_ms, sdpa_cb_r1_neighbor_l);

            for (uint32_t chunk = 0; chunk < sdpa_num_l_chunks; chunk++) {
                cb_wait_front(sdpa_cb_r1_neighbor_l, (chunk + 1) * sdpa_tiles_per_l_chunk);
                cb_wait_front(sdpa_cb_local_l, (chunk + 1) * sdpa_tiles_per_l_chunk);
                cb_reserve_back(sdpa_cb_r1_result_l, sdpa_tiles_per_l_chunk);

                uint32_t tile_index = chunk * sdpa_tiles_per_l_chunk;
                // R1 is non-normalizing: MS reduce releases regs, so always acquire
                bool acquire_regs = true;
                ckernel::sdpa_tail_l_block<sdpa_tiles_per_l_chunk>(
                    sdpa_cb_r1_neighbor_l, sdpa_cb_local_l, sdpa_cb_r1_result_l, tile_index, acquire_regs);

                cb_push_back(sdpa_cb_r1_result_l, sdpa_tiles_per_l_chunk);
            }
            ckernel::sdpa_tail_finalize(sdpa_cb_r1_neighbor_ms, sdpa_cb_local_ms);

            // R2: reduce(r1_result, r2_neighbor) -> final output (normalized L)
            ckernel::
                sdpa_tail_ms_reduce<SDPA_EXP_APPROX_MODE, true, sdpa_tiles_per_l_chunk, sdpa_scale_fp32, vector_mode>(
                    sdpa_cb_r2_neighbor_ms, sdpa_cb_r1_result_ms, sdpa_cb_ms_out, sdpa_cb_r2_neighbor_l);

            for (uint32_t chunk = 0; chunk < sdpa_num_l_chunks; chunk++) {
                cb_wait_front(sdpa_cb_r2_neighbor_l, (chunk + 1) * sdpa_tiles_per_l_chunk);
                cb_wait_front(sdpa_cb_r1_result_l, (chunk + 1) * sdpa_tiles_per_l_chunk);
                cb_reserve_back(sdpa_cb_l_out, sdpa_tiles_per_l_chunk);

                uint32_t tile_index = chunk * sdpa_tiles_per_l_chunk;
                bool acquire_regs = !(chunk == 0);  // First chunk reuses regs from MS phase when normalize=true
                ckernel::sdpa_tail_l_block<sdpa_tiles_per_l_chunk>(
                    sdpa_cb_r2_neighbor_l, sdpa_cb_r1_result_l, sdpa_cb_l_out, tile_index, acquire_regs);

                cb_push_back(sdpa_cb_l_out, sdpa_tiles_per_l_chunk);
            }
            // Inline R2 finalize: DON'T pop sdpa_cb_r1_result_ms here!
            // BRISC needs to read sdpa_cb_r1_result_ms (to send R2 data to neighbor).
            // If we pop it before BRISC reads it, BRISC hangs on cb_wait_front.
            // BRISC will pop sdpa_cb_r1_result_ms after sending R2 data.
            ckernel::sdpa_bcast_col_reuse_postamble();
            cb_pop_front(sdpa_cb_r2_neighbor_ms, 1);
            DPRINT << "SDPA Compute TRISC: Done" << ENDL();
        }
#endif  // COMPILE_FOR_TRISC
    }
#endif  // SKIP_SDPA

    // ========================================================================
    // Matmul1: [1, 512] x [512, 128] -> [1, 128] per core (8x8 grid)
    // ========================================================================
    {
        DeviceZoneScopedN("MATMUL1");
        deepseek_b1_ops::Matmul::Op<Matmul1CTArgs, Core::is_matmul1_core, true, false> matmul1;
        matmul1(matmul1_args);
    }

    // ========================================================================
    // Gather1: 8x8 matmul cores -> gather core (12, 9)
    // Collects [1, 128] * 64 = [1, 8192]
    // ========================================================================
    {
        DeviceZoneScopedN("GATHER1");
        deepseek_b1_ops::Gather::Op<Core::is_matmul1_core, Core::is_gather_receiver_core, true> gather1;
        gather1(gather1_args);
    }

    // ========================================================================
    // Mcast: gather core -> 13x10 mcast grid (130 cores)
    // Broadcasts [1, 8192] to each core in mcast grid
    // Source: gather1_dst_cb (CB 3), Destination: mcast_dst_cb = matmul2_in0 (CB 4)
    // Note: is_mcast_receiver_core (130 cores) includes 18 inactive cores that receive but skip matmul
    // ========================================================================
    constexpr bool is_mcast_receiver = Core::is_mcast_receiver_core && !Core::is_gather_receiver_core;
    deepseek_b1_ops::Mcast::Op<McastCTArgs, Core::is_gather_receiver_core, is_mcast_receiver, is_mcast_receiver, true>
        mcast;
    mcast.init(mcast_args);
    {
        DeviceZoneScopedN("MCAST");
        mcast(mcast_args);
    }
    mcast.teardown();

    // ========================================================================
    // Matmul2: [1, 8192] x [8192, 64] -> [1, 64] per core (112 active cores)
    // Input: mcast_dst_cb (CB 4), Weights: matmul2_in1 (CB 5), Output: matmul2_out (CB 6)
    // Only runs on 112 active cores (is_matmul2_core=true), 8 inactive cores skip
    // ========================================================================
    {
        DeviceZoneScopedN("MATMUL2");
        // pop_in0 = true (mcast output consumed), pop_in1 = false (weights persistent)
        deepseek_b1_ops::Matmul::Op<Matmul2CTArgs, Core::is_matmul2_core, true, false> matmul2;
        matmul2(matmul2_args);
    }

    // ========================================================================
    // Gather2: 112 active matmul2 cores -> gather core (12, 9)
    // Collects [1, 64] * 112 = [1, 7168]
    // ========================================================================
    {
        DeviceZoneScopedN("GATHER2");
        deepseek_b1_ops::Gather::Op<Core::is_matmul2_core, Core::is_gather_receiver_core, true> gather2;
        gather2(gather2_args);
    }

#ifndef SKIP_CCL
#if defined(COMPILE_FOR_BRISC)
    // Signal CCL sender that gather2 is complete (gather receiver only)
    if constexpr (Core::is_gather_receiver_core && Core::is_ccl_receiver_core) {
        constexpr uint32_t gather2_completion_semaphore_id =
            get_named_compile_time_arg_val("gather2_completion_semaphore_id");
        constexpr uint32_t ccl_sender_noc_x = get_named_compile_time_arg_val("ccl_sender_noc_x");
        constexpr uint32_t ccl_sender_noc_y = get_named_compile_time_arg_val("ccl_sender_noc_y");
        uint64_t ccl_sender_semaphore_addr =
            get_noc_addr(ccl_sender_noc_x, ccl_sender_noc_y, get_semaphore(gather2_completion_semaphore_id));
        noc_semaphore_inc(ccl_sender_semaphore_addr, 1);
    }
#endif

    // ========================================================================
    // CCL All-Reduce: Exchange [1, 7168] between devices
    // - CCL Sender (11, 9): Reads gather2 output from gather core, sends via fabric
    // - CCL Receiver (12, 9): Receives remote data, performs reduction
    //
    // Note: skip_local_push=1 is set for CCLReceiverReaderCTArgs because
    // gather2 already pushed to CB7 (gather2_dst_cb). The receiver just
    // needs to wait for remote data and perform the reduction.
    // ========================================================================
#if defined(COMPILE_FOR_NCRISC)
    if constexpr (Core::is_ccl_sender_core) {
        DeviceZoneScopedN("CCL_SENDER_READ");

        // Wait for gather2 to complete before reading from gather core
        constexpr uint32_t gather2_completion_semaphore_id =
            get_named_compile_time_arg_val("ccl_sender_gather2_completion_semaphore_id");
        volatile tt_l1_ptr uint32_t* gather2_completion_semaphore_addr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(gather2_completion_semaphore_id));
        noc_semaphore_wait(gather2_completion_semaphore_addr, 1);
        noc_semaphore_set(gather2_completion_semaphore_addr, 0);

        // Dummy WriterCTArgs - not used by NCRISC but needed for Op template
        using DummyWriterCTArgs = deepseek_b1_ops::AllReduceSender::WriterCTArgs<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>;

        deepseek_b1_ops::AllReduceSender::RTArgs ccl_sender_args{};
        ccl_sender_args.tensor_address = get_common_arg_val<uint32_t>(0);
        size_t fabric_arg_idx = 0;

        deepseek_b1_ops::AllReduceSender::Op<CCLSenderReaderCTArgs, DummyWriterCTArgs> ccl_sender_reader;
        ccl_sender_reader(ccl_sender_args, fabric_arg_idx);
    }

    if constexpr (Core::is_ccl_receiver_core) {
        DeviceZoneScopedN("CCL_RECEIVER_WAIT");
        // Dummy ComputeCTArgs - not used by NCRISC but needed for Op template
        using DummyComputeCTArgs = deepseek_b1_ops::AllReduceReceiver::ComputeCTArgs<0, 0, 0, 0, 0, 0, 0>;

        deepseek_b1_ops::AllReduceReceiver::RTArgs ccl_receiver_args{};
        ccl_receiver_args.sender_semaphore_addr = get_common_arg_val<uint32_t>(0);
        size_t fabric_arg_idx = 0;

        deepseek_b1_ops::AllReduceReceiver::Op<CCLReceiverReaderCTArgs, DummyComputeCTArgs> ccl_receiver_reader;
        ccl_receiver_reader(ccl_receiver_args, fabric_arg_idx);
    }

#elif defined(COMPILE_FOR_BRISC)
    if constexpr (Core::is_ccl_sender_core) {
        DeviceZoneScopedN("CCL_SENDER_SEND");
        // Dummy ReaderCTArgs - not used by BRISC but needed for Op template
        using DummyReaderCTArgs = deepseek_b1_ops::AllReduceSender::ReaderCTArgs<0, 0, 0, 0, 0>;

        deepseek_b1_ops::AllReduceSender::RTArgs ccl_sender_args{};
        ccl_sender_args.receiver_base_address = get_common_arg_val<uint32_t>(0);
        ccl_sender_args.receive_semaphore_addr = get_common_arg_val<uint32_t>(1);
        size_t fabric_arg_idx = 0;

        deepseek_b1_ops::AllReduceSender::Op<DummyReaderCTArgs, CCLSenderWriterCTArgs> ccl_sender_writer;
        ccl_sender_writer(ccl_sender_args, fabric_arg_idx);
    }
    // CCL Receiver BRISC is no-op

#elif defined(COMPILE_FOR_TRISC)
    if constexpr (Core::is_ccl_receiver_core) {
        DeviceZoneScopedN("CCL_RECEIVER_COMPUTE");
        // Dummy ReaderCTArgs - not used by TRISC but needed for Op template
        using DummyReaderCTArgs = deepseek_b1_ops::AllReduceReceiver::ReaderCTArgs<0, 0, 0, 0, 0, 0, 0, 0, 0, 0>;

        deepseek_b1_ops::AllReduceReceiver::RTArgs ccl_receiver_args{};
        size_t fabric_arg_idx = 0;

        deepseek_b1_ops::AllReduceReceiver::Op<DummyReaderCTArgs, CCLReceiverComputeCTArgs> ccl_receiver_compute;
        ccl_receiver_compute(ccl_receiver_args, fabric_arg_idx);
    }
    // CCL Sender TRISC is no-op
#endif
#endif  // SKIP_CCL
}
