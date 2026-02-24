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

// SDPA Reduce-to-All unified headers (replaces inlined SDPA code)
#ifndef SKIP_SDPA
#include "../../../unified_kernels/sdpa_reduce_worker.hpp"
#include "../../../unified_kernels/sdpa_reduce_forwarder.hpp"
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
    deepseek_compute_kernel_init();
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
    // SDPA REDUCE-TO-ALL PHASE (using unified ops from sdpa_reduce_worker.hpp
    // and sdpa_reduce_forwarder.hpp)
    //
    // SDPA worker cores (8): reduce L/MS across devices, scatter to matmul1 cores
    // SDPA forwarder cores (2): forward fabric packets for SDPA CCL
    //
    // The unified SdpaReduceWorker::Op handles all three RISC processors:
    //   NCRISC: Reader - pushes local input, prepares neighbor data
    //   BRISC: Writer - sends packets via forwarders, scatters output
    //   TRISC: Compute - streaming SDPA tail reduction (R1 + R2)
    //
    // The unified SdpaReduceForwarder::Op handles BRISC (FWD) and NCRISC (BWD)
    // ========================================================================
    {
        DeviceZoneScopedN("SDPA_REDUCE_TO_ALL");

        // SDPA Worker cores: use unified SdpaReduceWorker::Op
        if constexpr (Core::is_sdpa_worker_core) {
            using Worker = deepseek_b1_ops::SdpaReduceWorker;

#if defined(COMPILE_FOR_NCRISC)
            using ReaderCTArgs = Worker::ReaderCTArgs<
                get_named_compile_time_arg_val("sdpa_cb_local_l"),
                get_named_compile_time_arg_val("sdpa_cb_local_ms"),
                get_named_compile_time_arg_val("sdpa_cb_r1_neighbor_l"),
                get_named_compile_time_arg_val("sdpa_cb_r1_neighbor_ms"),
                get_named_compile_time_arg_val("sdpa_cb_r2_neighbor_l"),
                get_named_compile_time_arg_val("sdpa_cb_r2_neighbor_ms"),
                get_named_compile_time_arg_val("sdpa_ms_tile_size_bytes"),
                get_named_compile_time_arg_val("sdpa_l_chunk_size_bytes"),
                get_named_compile_time_arg_val("sdpa_num_l_chunks"),
                get_named_compile_time_arg_val("sdpa_tiles_per_l_chunk"),
                get_named_compile_time_arg_val("sdpa_position_enabled"),
                get_named_compile_time_arg_val("sdpa_per_device_chunk_size")>;

            // Dummy WriterCT and ComputeCT - not used by NCRISC but needed for Op template
            using WriterCTArgs = Worker::WriterCTArgs<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>;
            using ComputeCTArgs = Worker::ComputeCTArgs<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>;

            Worker::Op<ReaderCTArgs, WriterCTArgs, ComputeCTArgs> sdpa_worker;
            sdpa_worker();

#elif defined(COMPILE_FOR_BRISC)
            // Dummy ReaderCT - not used by BRISC
            using ReaderCTArgs = Worker::ReaderCTArgs<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>;

            using WriterCTArgs = Worker::WriterCTArgs<
                get_named_compile_time_arg_val("sdpa_cb_local_l"),
                get_named_compile_time_arg_val("sdpa_cb_local_ms"),
                get_named_compile_time_arg_val("sdpa_cb_r1_result_l"),
                get_named_compile_time_arg_val("sdpa_cb_r1_result_ms"),
                get_named_compile_time_arg_val("sdpa_cb_packet_slot"),
                get_named_compile_time_arg_val("sdpa_l1_alignment"),
                get_named_compile_time_arg_val("sdpa_page_size_bytes"),
                get_named_compile_time_arg_val("sdpa_slot_size"),
                get_named_compile_time_arg_val("sdpa_ms_tile_size_bytes"),
                get_named_compile_time_arg_val("sdpa_l_chunk_size_bytes"),
                get_named_compile_time_arg_val("sdpa_num_l_chunks"),
                get_named_compile_time_arg_val("sdpa_tiles_per_l_chunk"),
                get_named_compile_time_arg_val("sdpa_cb_l_out"),
                get_named_compile_time_arg_val("sdpa_scatter_num_tiles"),
                get_named_compile_time_arg_val("sdpa_scatter_src_tile_size"),
                get_named_compile_time_arg_val("sdpa_scatter_dst_tile_size"),
                get_named_compile_time_arg_val("sdpa_scatter_face_size"),
                get_named_compile_time_arg_val("sdpa_scatter_row_face_size"),
                get_named_compile_time_arg_val("sdpa_scatter_num_rows"),
                1,  // scatter_arrival_enabled=1 (signal matmul1 cores after each scatter row)
                get_named_compile_time_arg_val("scatter_arrival_semaphore_id")>;

            // Dummy ComputeCT - not used by BRISC
            using ComputeCTArgs = Worker::ComputeCTArgs<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>;

            Worker::Op<ReaderCTArgs, WriterCTArgs, ComputeCTArgs> sdpa_worker;
            sdpa_worker();

#elif defined(COMPILE_FOR_TRISC)
            // Dummy ReaderCT and WriterCT - not used by TRISC
            using ReaderCTArgs = Worker::ReaderCTArgs<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>;
            using WriterCTArgs = Worker::WriterCTArgs<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>;

            using ComputeCTArgs = Worker::ComputeCTArgs<
                get_named_compile_time_arg_val("sdpa_cb_local_l"),
                get_named_compile_time_arg_val("sdpa_cb_local_ms"),
                get_named_compile_time_arg_val("sdpa_cb_r1_neighbor_l"),
                get_named_compile_time_arg_val("sdpa_cb_r1_neighbor_ms"),
                get_named_compile_time_arg_val("sdpa_cb_r1_result_l"),
                get_named_compile_time_arg_val("sdpa_cb_r1_result_ms"),
                get_named_compile_time_arg_val("sdpa_cb_r2_neighbor_l"),
                get_named_compile_time_arg_val("sdpa_cb_r2_neighbor_ms"),
                get_named_compile_time_arg_val("sdpa_cb_l_out"),
                get_named_compile_time_arg_val("sdpa_cb_ms_out"),
                get_named_compile_time_arg_val("sdpa_scale_fp32"),
                get_named_compile_time_arg_val("sdpa_tiles_per_l_chunk"),
                get_named_compile_time_arg_val("sdpa_num_l_chunks"),
                get_named_compile_time_arg_val("sdpa_position_enabled"),
                get_named_compile_time_arg_val("sdpa_per_device_chunk_size"),
                1>;  // final_reduction=1 (always normalize in post_sdpa, untilize constraint)

            // Note: compute_kernel_hw_startup already called at top of TRISC block
            Worker::Op<ReaderCTArgs, WriterCTArgs, ComputeCTArgs> sdpa_worker;
            sdpa_worker();
#endif
        }

        // SDPA Forwarder cores: use unified SdpaReduceForwarder::Op
        // Forwarders are dataflow-only (BRISC/NCRISC), TRISC is no-op
#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
        if constexpr (Core::is_sdpa_forwarder_core) {
            using Fwd = deepseek_b1_ops::SdpaReduceForwarder;
            using FwdCTArgs = Fwd::CTArgs<
                get_named_compile_time_arg_val("sdpa_fwd_slots_per_round"),
                get_named_compile_time_arg_val("sdpa_fwd_slot_size"),
                get_named_compile_time_arg_val("sdpa_fwd_r2_buffer_offset")>;

            Fwd::Op<FwdCTArgs> sdpa_forwarder;
            sdpa_forwarder();
        }
#endif
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
