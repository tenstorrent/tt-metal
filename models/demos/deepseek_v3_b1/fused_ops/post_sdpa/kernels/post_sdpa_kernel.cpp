// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Post SDPA unified kernel with SDPA Reduce-to-All + CCL All-Reduce
// Single kernel file, compiles correctly for all RISC cores
// Each RISC has its own CTArgs struct with different compile-time arg layout
//
// Implements: SDPA Reduce-to-All + Matmul4 + Gather2 + Mcast3 + Matmul5 + Gather3 + CCL All-Reduce
//
// SDPA Reduce-to-All Phase:
// - SDPA Workers (8 cores): Reduce L/MS tensors across devices, scatter [1,512] to matmul4 cores
// - SDPA Forwarders (2 cores): Forward fabric packets for SDPA CCL
//
// Post-SDPA Phases:
// - Matmul4: [1, 512] x [512, 128] -> [1, 128] on 64 cores (kv_b2 grid: 5x8 + 12x2) - waits for scatter data
// - Gather2: Collect [1, 128] from 64 cores to [1, 8192] on gather core
// - Mcast3: Broadcast [1, 8192] to 130 cores (13x10 grid, rectangular)
// - Matmul5: [1, 8192] x [8192, 64] -> [1, 64] on 112 active cores (o_proj grid: 12x8 + 8x2)
// - Gather3: Collect [1, 64] from 112 active cores to [1, 7168] on gather core
// - CCL All-Reduce: Exchange [1, 7168] between devices, reduce (local + remote + residual)
//
// Note: Mcast3 grid (13x10=130) includes 18 inactive cores
// which receive mcast3 data but skip matmul5 via is_matmul5_core=false
//
// SDPA Core Layout:
// - SDPA Workers: (2,8)-(5,8), (2,9)-(5,9) = 8 cores
// - SDPA Forwarders: (6,9), (7,9) = 2 cores
// Note: Some SDPA cores overlap with matmul5 grid - they run SDPA first, then matmul5
//
// CCL Core Layout:
// - CCL sender core (11, 9): gather3 receiver + dual fabric writers (NCRISC link 1, BRISC link 0)
// - CCL receiver core (12, 9): fabric reader (BRISC) + reduction compute (TRISC)

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"
#include "../../../unified_kernels/matmul.hpp"
#include "../../../unified_kernels/gather.hpp"
#include "../../../unified_kernels/mcast.hpp"
#ifndef SKIP_CCL
#include "../../../unified_kernels/all_reduce.hpp"
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
    // First matmul on kv_b2 grid (5x8 + 12x2 = 64 cores) - receives scatter data from SDPA workers
    static constexpr bool is_matmul4_core = get_named_compile_time_arg_val("is_matmul4_core") == 1;
    // Gather core (12, 9) - receives gather2, sends mcast3
    static constexpr bool is_gather_receiver_core = get_named_compile_time_arg_val("is_gather_receiver_core") == 1;
    // Mcast3 receiver grid (13x10 = 130 cores) - receives mcast3 data
    static constexpr bool is_mcast3_receiver_core = get_named_compile_time_arg_val("is_mcast3_receiver_core") == 1;
    // Active matmul5 cores (112 cores: o_proj grid 12x8 + 8x2)
    static constexpr bool is_matmul5_core = get_named_compile_time_arg_val("is_matmul5_core") == 1;
    // CCL sender core (11, 9) - receives gather3, sends via fabric
    static constexpr bool is_allreduce_sender_core = get_named_compile_time_arg_val("is_allreduce_sender_core") == 1;
    // CCL receiver core (12, 9) - receives remote data via fabric, runs reduction
    static constexpr bool is_allreduce_receiver_core =
        get_named_compile_time_arg_val("is_allreduce_receiver_core") == 1;
};

void kernel_main() {
// ============================================================================
// NCRISC (Reader)
// SDPA Phase:
// - SDPA worker reader (8 cores): push local L/MS, prepare R1/R2 neighbor data
// - SDPA forwarder NCRISC (2 cores): forward BWD fabric packets
// Post-SDPA Phase:
// - Matmul4 reader (kv_b2 grid): setup sharded buffers (after scatter arrival)
// - Gather2 sender (kv_b2 grid): send matmul4 output to gather core
// - Mcast3 receiver (13x10 grid = 130 cores): receive mcast3 data
// - Matmul5 reader (112 active cores): setup weights buffer
// - Gather3 sender (112 active cores): send matmul5 output to gather core
// - CCL sender core (11, 9): NCRISC writer (fabric link 1)
// ============================================================================
#if defined(COMPILE_FOR_NCRISC)
    // Matmul4 CTArgs
    using Matmul4CTArgs = deepseek_b1_ops::Matmul::ReaderCTArgs;
    deepseek_b1_ops::Matmul::ReaderArgs matmul4_args{};

    // Gather2 sender args (UsePerCoreSenderIdx: each core gets a contiguous index
    // via gather2_sender_idx, avoiding gaps from the non-rectangular kv_b2 grid)
    deepseek_b1_ops::Gather::SenderArgs gather2_args{
        get_named_compile_time_arg_val("gather2_dest_noc_x"),
        get_named_compile_time_arg_val("gather2_dest_noc_y"),
        get_named_compile_time_arg_val("gather2_data_size_bytes"),
        get_semaphore(get_named_compile_time_arg_val("gather2_receiver_semaphore_id")),
        get_named_compile_time_arg_val("gather2_src_cb"),
        get_named_compile_time_arg_val("gather2_src_num_pages"),
        get_named_compile_time_arg_val("gather2_sender_grid_start_x"),
        get_named_compile_time_arg_val("gather2_sender_grid_start_y"),
        get_named_compile_time_arg_val("gather2_sender_grid_end_x"),
        get_named_compile_time_arg_val("gather2_sender_grid_end_y"),
        get_named_compile_time_arg_val("gather2_row_major"),
        get_named_compile_time_arg_val("gather2_receiver_data_addr"),
        get_named_compile_time_arg_val("gather2_sender_idx"),
    };

    // Mcast3 receiver args
    using Mcast3CTArgs = deepseek_b1_ops::Mcast::ReceiverCTArgs;
    deepseek_b1_ops::Mcast::ReceiverArgs mcast3_args{
        get_semaphore(get_named_compile_time_arg_val("mcast3_data_receiver_semaphore")),
        get_named_compile_time_arg_val("mcast3_dst_cb"),
        get_named_compile_time_arg_val("mcast3_dst_num_pages"),
    };

    // Matmul5 CTArgs
    using Matmul5CTArgs = deepseek_b1_ops::Matmul::ReaderCTArgs;
    deepseek_b1_ops::Matmul::ReaderArgs matmul5_args{};

    // Gather3 sender args (UsePerCoreSenderIdx: each core gets a contiguous index
    // via gather3_sender_idx, avoiding gaps from the non-rectangular o_proj grid)
    deepseek_b1_ops::Gather::SenderArgs gather3_args{
        get_named_compile_time_arg_val("gather3_dest_noc_x"),
        get_named_compile_time_arg_val("gather3_dest_noc_y"),
        get_named_compile_time_arg_val("gather3_data_size_bytes"),
        get_semaphore(get_named_compile_time_arg_val("gather3_receiver_semaphore_id")),
        get_named_compile_time_arg_val("gather3_src_cb"),
        get_named_compile_time_arg_val("gather3_src_num_pages"),
        get_named_compile_time_arg_val("gather3_sender_grid_start_x"),
        get_named_compile_time_arg_val("gather3_sender_grid_start_y"),
        get_named_compile_time_arg_val("gather3_sender_grid_end_x"),
        get_named_compile_time_arg_val("gather3_sender_grid_end_y"),
        get_named_compile_time_arg_val("gather3_row_major"),
        get_named_compile_time_arg_val("gather3_receiver_data_addr"),
        get_named_compile_time_arg_val("gather3_sender_idx"),
    };
#ifndef SKIP_CCL
    using AllReduceWriterCTArgs = deepseek_b1_ops::AllReduce::WriterLinkCTArgs<
        get_named_compile_time_arg_val("allreduce_local_data_cb_id"),
        get_named_compile_time_arg_val("allreduce_input_num_tiles"),
        get_named_compile_time_arg_val("allreduce_page_size_bytes"),
        get_named_compile_time_arg_val("allreduce_tiles_per_chunk"),
        get_named_compile_time_arg_val("allreduce_last_chunk_tiles"),
        get_named_compile_time_arg_val("allreduce_num_chunks"),
        get_named_compile_time_arg_val("allreduce_num_links"),
        get_named_compile_time_arg_val("allreduce_writer_link_index"),
        get_named_compile_time_arg_val("allreduce_writer_signal_local_ready"),
        get_named_compile_time_arg_val("allreduce_skip_local_push")>;
#endif
// ============================================================================
// BRISC (Writer)
// - Gather2 receiver (gather core): receive from kv_b2 grid
// - Mcast3 sender (gather core): broadcast to 13x10 grid (130 cores)
// - Gather3 receiver (sender core 11,9): receive from 112 active matmul5 cores
// - CCL sender core (11, 9): BRISC writer (fabric link 0)
// - CCL receiver core (12, 9): BRISC reader
// ============================================================================
#elif defined(COMPILE_FOR_BRISC)
    // Matmul4/2 CTArgs (BRISC is no-op for matmul)
    using Matmul4CTArgs = deepseek_b1_ops::Matmul::WriterCTArgs;
    using Matmul5CTArgs = deepseek_b1_ops::Matmul::WriterCTArgs;
    deepseek_b1_ops::Matmul::WriterArgs matmul4_args{};
    deepseek_b1_ops::Matmul::WriterArgs matmul5_args{};

    // Gather2 receiver args
    deepseek_b1_ops::Gather::ReceiverArgs gather2_args{
        get_named_compile_time_arg_val("gather2_noc0_num_senders"),
        get_named_compile_time_arg_val("gather2_noc1_num_senders"),
        get_semaphore(get_named_compile_time_arg_val("gather2_noc0_receiver_semaphore_id")),
        get_semaphore(get_named_compile_time_arg_val("gather2_noc1_receiver_semaphore_id")),
        get_named_compile_time_arg_val("gather2_dst_cb"),
        get_named_compile_time_arg_val("gather2_dst_num_pages"),
    };

    // Mcast3 sender args
    using Mcast3CTArgs = deepseek_b1_ops::Mcast::SenderCTArgs<
        get_named_compile_time_arg_val("mcast3_num_cores"),
        get_named_compile_time_arg_val("mcast3_is_part_of_receiver_grid") == 1,
        false>;  // loopback = false

    constexpr uint32_t mcast3_src_cb = get_named_compile_time_arg_val("mcast3_src_cb");
    constexpr uint32_t mcast3_dst_cb = get_named_compile_time_arg_val("mcast3_dst_cb");
    deepseek_b1_ops::Mcast::SenderArgs mcast3_args{
        get_named_compile_time_arg_val("mcast3_dest_noc_start_x"),
        get_named_compile_time_arg_val("mcast3_dest_noc_start_y"),
        get_named_compile_time_arg_val("mcast3_dest_noc_end_x"),
        get_named_compile_time_arg_val("mcast3_dest_noc_end_y"),
        get_semaphore(get_named_compile_time_arg_val("mcast3_data_sender_semaphore")),
        get_semaphore(get_named_compile_time_arg_val("mcast3_data_receiver_semaphore")),
        get_named_compile_time_arg_val("mcast3_data_size_bytes"),
        mcast3_src_cb,
        get_named_compile_time_arg_val("mcast3_src_num_pages"),
        get_read_ptr(mcast3_src_cb),
        get_write_ptr(mcast3_dst_cb),  // CB 4 address (now allocated on gather core too)
    };

    // Gather3 receiver args
    deepseek_b1_ops::Gather::ReceiverArgs gather3_args{
        get_named_compile_time_arg_val("gather3_noc0_num_senders"),
        get_named_compile_time_arg_val("gather3_noc1_num_senders"),
        get_semaphore(get_named_compile_time_arg_val("gather3_noc0_receiver_semaphore_id")),
        get_semaphore(get_named_compile_time_arg_val("gather3_noc1_receiver_semaphore_id")),
        get_named_compile_time_arg_val("gather3_dst_cb"),
        get_named_compile_time_arg_val("gather3_dst_num_pages"),
    };

#ifndef SKIP_CCL
    // BRISC has dual role: WriterSingleLink on sender core + Reader on receiver core.
    // Both CT arg sets are available; if constexpr guards execution.
    using AllReduceBriscWriterCTArgs = deepseek_b1_ops::AllReduce::WriterLinkCTArgs<
        get_named_compile_time_arg_val("allreduce_local_data_cb_id"),
        get_named_compile_time_arg_val("allreduce_input_num_tiles"),
        get_named_compile_time_arg_val("allreduce_page_size_bytes"),
        get_named_compile_time_arg_val("allreduce_tiles_per_chunk"),
        get_named_compile_time_arg_val("allreduce_last_chunk_tiles"),
        get_named_compile_time_arg_val("allreduce_num_chunks"),
        get_named_compile_time_arg_val("allreduce_num_links"),
        get_named_compile_time_arg_val("allreduce_writer_link_index"),
        get_named_compile_time_arg_val("allreduce_writer_signal_local_ready"),
        get_named_compile_time_arg_val("allreduce_skip_local_push")>;

    using AllReduceReaderCTArgs = deepseek_b1_ops::AllReduce::ReaderCTArgs<
        get_named_compile_time_arg_val("allreduce_recv_local_data_cb_id"),
        get_named_compile_time_arg_val("allreduce_remote_data_cb_id"),
        get_named_compile_time_arg_val("allreduce_residual_cb_id"),
        get_named_compile_time_arg_val("allreduce_has_residual"),
        get_named_compile_time_arg_val("allreduce_total_num_tiles"),
        get_named_compile_time_arg_val("allreduce_page_size_bytes"),
        get_named_compile_time_arg_val("allreduce_tiles_per_chunk"),
        get_named_compile_time_arg_val("allreduce_last_chunk_tiles"),
        get_named_compile_time_arg_val("allreduce_num_chunks"),
        get_named_compile_time_arg_val("allreduce_num_links")>;
#endif
// ============================================================================
// TRISC (Compute)
// - Matmul4 compute (kv_b2 grid)
// - Matmul5 compute (112 active cores)
// - CCL receiver core (12, 9): TRISC reduction (local + remote + residual)
// ============================================================================
#elif defined(COMPILE_FOR_TRISC)
    // Matmul4 CTArgs
    using Matmul4CTArgs =
        deepseek_b1_ops::Matmul::ComputeCTArgs<get_named_compile_time_arg_val("matmul4_out_w_per_core")>;
    deepseek_b1_ops::Matmul::ComputeArgs matmul4_args{
        get_named_compile_time_arg_val("matmul4_in0"),
        get_named_compile_time_arg_val("matmul4_in1"),
        get_named_compile_time_arg_val("matmul4_out"),
        get_named_compile_time_arg_val("matmul4_k_num_tiles"),
    };

    // Gather2 compute args (no-op)
    deepseek_b1_ops::Gather::ComputeArgs gather2_args{};

    // Mcast3 CTArgs (no-op)
    using Mcast3CTArgs = deepseek_b1_ops::Mcast::ComputeCTArgs;
    deepseek_b1_ops::Mcast::ComputeArgs mcast3_args{};

    // Matmul5 CTArgs
    using Matmul5CTArgs =
        deepseek_b1_ops::Matmul::ComputeCTArgs<get_named_compile_time_arg_val("matmul5_out_w_per_core")>;
    deepseek_b1_ops::Matmul::ComputeArgs matmul5_args{
        get_named_compile_time_arg_val("matmul5_in0"),
        get_named_compile_time_arg_val("matmul5_in1"),
        get_named_compile_time_arg_val("matmul5_out"),
        get_named_compile_time_arg_val("matmul5_k_num_tiles"),
    };

    // Gather3 compute args (no-op)
    deepseek_b1_ops::Gather::ComputeArgs gather3_args{};

#ifndef SKIP_CCL
    using AllReduceComputeCTArgs = deepseek_b1_ops::AllReduce::ComputeCTArgs<
        get_named_compile_time_arg_val("allreduce_cb_remote"),
        get_named_compile_time_arg_val("allreduce_cb_local"),
        get_named_compile_time_arg_val("allreduce_cb_out"),
        get_named_compile_time_arg_val("allreduce_cb_residual"),
        get_named_compile_time_arg_val("allreduce_has_residual"),
        get_named_compile_time_arg_val("allreduce_num_tiles")>;
#endif
    deepseek_compute_kernel_init();
#endif

#ifndef SKIP_SDPA
    // ========================================================================
    // SDPA REDUCE-TO-ALL PHASE (using unified ops from sdpa_reduce_worker.hpp
    // and sdpa_reduce_forwarder.hpp)
    //
    // SDPA worker cores (8): reduce L/MS across devices, scatter to matmul4 cores
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
                get_named_compile_time_arg_val("sdpa_cb_neighbor_l"),
                get_named_compile_time_arg_val("sdpa_cb_neighbor_ms"),
                get_named_compile_time_arg_val("sdpa_ms_tile_size_bytes"),
                get_named_compile_time_arg_val("sdpa_l_chunk_size_bytes"),
                get_named_compile_time_arg_val("sdpa_num_l_chunks"),
                get_named_compile_time_arg_val("sdpa_tiles_per_l_chunk"),
                get_named_compile_time_arg_val("sdpa_position_enabled"),
                get_named_compile_time_arg_val("sdpa_per_device_chunk_size")>;

            uint32_t per_core_rta_arg_idx = 0;
            Worker::ReaderArgs reader_args{
                .r1_neighbor_sem_addr = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
                .r2_neighbor_sem_addr = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
                .r1_recv_buffer_addr = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
                .r2_recv_buffer_addr = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
            };
            if constexpr (ReaderCTArgs::position_enabled) {
                reader_args.pos_addr = get_arg_val<uint32_t>(per_core_rta_arg_idx++);
                reader_args.r1_neighbor_device_idx = get_arg_val<uint32_t>(per_core_rta_arg_idx++);
                reader_args.r2_neighbor_device_idx = get_arg_val<uint32_t>(per_core_rta_arg_idx++);
                reader_args.r2_neighbor_r1_neighbor_idx = get_arg_val<uint32_t>(per_core_rta_arg_idx++);
            }

            unified_kernels::setup_sharded_buffer(ReaderCTArgs::cb_local_l, ReaderCTArgs::out_tiles);
            unified_kernels::setup_sharded_buffer(ReaderCTArgs::cb_local_ms, 1);

            Worker::Op<ReaderCTArgs> sdpa_worker;
            sdpa_worker(reader_args);

#elif defined(COMPILE_FOR_BRISC)
            using WriterCTArgs = Worker::WriterCTArgs<
                get_named_compile_time_arg_val("sdpa_cb_local_l"),
                get_named_compile_time_arg_val("sdpa_cb_local_ms"),
                get_named_compile_time_arg_val("sdpa_cb_r1_result_l"),
                get_named_compile_time_arg_val("sdpa_cb_r1_result_ms"),
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
                1>;  // scatter_arrival_enabled=1 (signal matmul4 cores after each scatter row)

            uint32_t per_core_rta_arg_idx = 0;
            Worker::WriterArgs writer_args{
                .r1_dst_mesh_id = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
                .r1_dst_chip_id = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
                .r1_neighbor_dst_addr = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
                .r1_neighbor_sem_addr = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
                .r2_dst_mesh_id = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
                .r2_dst_chip_id = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
                .r2_neighbor_dst_addr = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
                .r2_neighbor_sem_addr = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
                .current_core_x = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
                .current_core_y = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
                .fwd_core_x = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
                .fwd_core_y = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
                .r1_fwd_slot_addr = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
                .r1_fwd_sem_addr = get_semaphore(get_arg_val<uint32_t>(per_core_rta_arg_idx++)),
                .r1_base_slot_idx = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
                .r2_fwd_slot_addr = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
                .r2_fwd_sem_addr = get_semaphore(get_arg_val<uint32_t>(per_core_rta_arg_idx++)),
                .r2_base_slot_idx = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
                .scatter_dest_l1_addr = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
                .scatter_dest_coords_addr = get_arg_addr(per_core_rta_arg_idx),
                // scatter_arrival_enabled=1, so we need to pass the semaphore address
                .scatter_arrival_sem_addr =
                    get_semaphore(get_named_compile_time_arg_val("scatter_arrival_semaphore_id")),
            };
            per_core_rta_arg_idx += WriterCTArgs::scatter_num_rows * 2;  // x, y value per dest
            Worker::Op<WriterCTArgs> sdpa_worker;
            sdpa_worker(writer_args);

#elif defined(COMPILE_FOR_TRISC)
            using ComputeCTArgs = Worker::ComputeCTArgs<
                get_named_compile_time_arg_val("sdpa_cb_local_l"),
                get_named_compile_time_arg_val("sdpa_cb_local_ms"),
                get_named_compile_time_arg_val("sdpa_cb_neighbor_l"),
                get_named_compile_time_arg_val("sdpa_cb_neighbor_ms"),
                get_named_compile_time_arg_val("sdpa_cb_r1_result_l"),
                get_named_compile_time_arg_val("sdpa_cb_r1_result_ms"),
                get_named_compile_time_arg_val("sdpa_cb_l_out"),
                get_named_compile_time_arg_val("sdpa_scale_fp32"),
                get_named_compile_time_arg_val("sdpa_tiles_per_l_chunk"),
                get_named_compile_time_arg_val("sdpa_num_l_chunks"),
                get_named_compile_time_arg_val("sdpa_position_enabled"),
                get_named_compile_time_arg_val("sdpa_per_device_chunk_size"),
                1>;  // final_reduction=1 (always normalize in post_sdpa, untilize constraint)

            // Note: compute_kernel_hw_startup already called at top of TRISC block
            Worker::ComputeArgs compute_args;
            if constexpr (ComputeCTArgs::position_enabled) {
                uint32_t per_core_rta_arg_idx = 0;
                compute_args.pos_addr = get_arg_val<uint32_t>(per_core_rta_arg_idx++);
                compute_args.device_idx = get_arg_val<uint32_t>(per_core_rta_arg_idx++);
                compute_args.r1_neighbor_device_idx = get_arg_val<uint32_t>(per_core_rta_arg_idx++);
                compute_args.r2_neighbor_device_idx = get_arg_val<uint32_t>(per_core_rta_arg_idx++);
                compute_args.r2_neighbor_r1_neighbor_idx = get_arg_val<uint32_t>(per_core_rta_arg_idx++);
                compute_args.swap_r1_reduction_order = get_arg_val<uint32_t>(per_core_rta_arg_idx++);
                compute_args.swap_r2_reduction_order = get_arg_val<uint32_t>(per_core_rta_arg_idx++);
            }
            Worker::Op<ComputeCTArgs> sdpa_worker;
            sdpa_worker(compute_args);
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

            uint32_t per_core_rta_arg_idx = 0;
            Fwd::ForwarderArgs fwd_args{
                .buffer_base = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
                .buffer_offset = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
                .r1_sem_addr = get_semaphore(get_arg_val<uint32_t>(per_core_rta_arg_idx++)),
                .r2_sem_addr = get_semaphore(get_arg_val<uint32_t>(per_core_rta_arg_idx++)),
                .rta_offset = per_core_rta_arg_idx,
            };
            Fwd::Op<FwdCTArgs> sdpa_forwarder;
            sdpa_forwarder(fwd_args);
        }
#endif
    }
#endif  // SKIP_SDPA

    // ========================================================================
    // Setup sharded buffers (NCRISC only)
    // ========================================================================
#if defined(COMPILE_FOR_NCRISC)
    // Matmul4 buffers (kv_b2 grid)
    // NOTE: When SDPA is enabled, matmul4 waits for scatter data arrival before setup
    if constexpr (Core::is_matmul4_core) {
#ifndef SKIP_SDPA
        // Wait for SDPA scatter to deliver data to matmul4_in0
        // Each SDPA worker signals this semaphore after scatter write completes
        constexpr uint32_t scatter_arrival_semaphore_id =
            get_named_compile_time_arg_val("scatter_arrival_semaphore_id");
        volatile tt_l1_ptr uint32_t* scatter_arrival_sem_addr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(scatter_arrival_semaphore_id));
        noc_semaphore_wait(scatter_arrival_sem_addr, 1);
        noc_semaphore_set(scatter_arrival_sem_addr, 0);
#endif

        constexpr uint32_t matmul4_in0 = get_named_compile_time_arg_val("matmul4_in0");
        constexpr uint32_t matmul4_k_num_tiles = get_named_compile_time_arg_val("matmul4_k_num_tiles");
        unified_kernels::setup_sharded_buffer(matmul4_in0, matmul4_k_num_tiles);

        constexpr uint32_t matmul4_in1 = get_named_compile_time_arg_val("matmul4_in1");
        constexpr uint32_t matmul4_out_w_per_core = get_named_compile_time_arg_val("matmul4_out_w_per_core");
        unified_kernels::setup_sharded_buffer(matmul4_in1, matmul4_k_num_tiles * matmul4_out_w_per_core);
    }

    // Matmul5 buffers (112 active cores) - weights only, input comes from mcast3
    if constexpr (Core::is_matmul5_core) {
        constexpr uint32_t matmul5_in1 = get_named_compile_time_arg_val("matmul5_in1");
        constexpr uint32_t matmul5_k_num_tiles = get_named_compile_time_arg_val("matmul5_k_num_tiles");
        constexpr uint32_t matmul5_out_w_per_core = get_named_compile_time_arg_val("matmul5_out_w_per_core");
        unified_kernels::setup_sharded_buffer(matmul5_in1, matmul5_k_num_tiles * matmul5_out_w_per_core);
    }
#endif

    // ========================================================================
    // Matmul4: [1, 512] x [512, 128] -> [1, 128] per core (kv_b2 grid)
    // ========================================================================
    {
        DeviceZoneScopedN("MATMUL4");
        deepseek_b1_ops::Matmul::Op<Matmul4CTArgs, Core::is_matmul4_core, true, false> matmul4;
        matmul4(matmul4_args);
    }

    // ========================================================================
    // Gather2: matmul4 cores (kv_b2 grid) -> gather core (12, 9)
    // Collects [1, 128] * 64 = [1, 8192]
    // ========================================================================
    {
        DeviceZoneScopedN("GATHER2");
        deepseek_b1_ops::Gather::Op<Core::is_matmul4_core, Core::is_gather_receiver_core, true, true> gather2;
        gather2(gather2_args);
    }

    // ========================================================================
    // Mcast3: gather core -> 13x10 mcast3 grid (130 cores)
    // Broadcasts [1, 8192] to each core in mcast3 grid
    // Source: gather2_dst_cb (CB 3), Destination: mcast3_dst_cb = matmul5_in0 (CB 4)
    // Note: 18 inactive grid cores only do semaphore handshake; only matmul5 cores do full CB receive
    // ========================================================================
    constexpr bool is_mcast3_grid_core = Core::is_mcast3_receiver_core && !Core::is_gather_receiver_core;
    deepseek_b1_ops::Mcast::
        Op<Mcast3CTArgs, Core::is_gather_receiver_core, is_mcast3_grid_core, Core::is_matmul5_core, true>
            mcast3;
    mcast3.init(mcast3_args);
    {
        DeviceZoneScopedN("MCAST3");
        mcast3(mcast3_args);
    }
    mcast3.teardown(mcast3_args);

    // ========================================================================
    // Matmul5: [1, 8192] x [8192, 64] -> [1, 64] per core (112 active cores)
    // Input: mcast3_dst_cb (CB 4), Weights: matmul5_in1 (CB 5), Output: matmul5_out (CB 6)
    // Only runs on 112 active cores (is_matmul5_core=true), 18 inactive cores skip
    // ========================================================================
    {
        DeviceZoneScopedN("MATMUL5");
        // pop_in0 = true (mcast3 output consumed), pop_in1 = false (weights persistent)
        deepseek_b1_ops::Matmul::Op<Matmul5CTArgs, Core::is_matmul5_core, true, false> matmul5;
        matmul5(matmul5_args);
    }

    // ========================================================================
    // Gather3: 112 active matmul5 cores -> sender core (11, 9)
    // Collects [1, 64] * 112 = [1, 7168]
    // ========================================================================
    {
        DeviceZoneScopedN("GATHER3");
        deepseek_b1_ops::Gather::Op<Core::is_matmul5_core, Core::is_allreduce_sender_core, true, true> gather3;
        gather3(gather3_args);
    }

#ifndef SKIP_CCL
    // ========================================================================
    // CCL All-Reduce: Exchange [1, 7168] between devices
    // Sender core (11,9): NCRISC writer (link 1) + BRISC writer (link 0)
    // Receiver core (12,9): BRISC reader + TRISC compute
    // ========================================================================
#if defined(COMPILE_FOR_NCRISC)
    if constexpr (Core::is_allreduce_sender_core) {
        DeviceZoneScopedN("CCL_SENDER_WRITER");
        deepseek_b1_ops::AllReduce::SenderFabricArgs args{};
        args.intermediate_buffer_address = get_common_arg_val<uint32_t>(0);
        args.dest_noc_x = get_common_arg_val<uint32_t>(1);
        args.dest_noc_y = get_common_arg_val<uint32_t>(2);
        args.per_core_rta_start_idx = 0;
        deepseek_b1_ops::AllReduce::WriterSingleLink<AllReduceWriterCTArgs> writer;
        writer(args);
    }

#elif defined(COMPILE_FOR_BRISC)
    if constexpr (Core::is_allreduce_sender_core) {
        DeviceZoneScopedN("CCL_SENDER_WRITER");
        deepseek_b1_ops::AllReduce::SenderFabricArgs args{};
        args.intermediate_buffer_address = get_common_arg_val<uint32_t>(0);
        args.dest_noc_x = get_common_arg_val<uint32_t>(1);
        args.dest_noc_y = get_common_arg_val<uint32_t>(2);
        args.per_core_rta_start_idx = 0;
        deepseek_b1_ops::AllReduce::WriterSingleLink<AllReduceBriscWriterCTArgs> writer;
        writer(args);
    }
    if constexpr (Core::is_allreduce_receiver_core) {
        DeviceZoneScopedN("CCL_READER");
        deepseek_b1_ops::AllReduce::ReceiverArgs args{};
        args.sem_bank_addr_0 = get_common_arg_val<uint32_t>(0);
        args.sem_bank_addr_1 = get_common_arg_val<uint32_t>(1);
        args.sender_noc_x = get_common_arg_val<uint32_t>(2);
        args.sender_noc_y = get_common_arg_val<uint32_t>(3);
        args.sender_local_data_l1_addr = get_common_arg_val<uint32_t>(4);
        args.local_ready_sem_bank_addr = get_common_arg_val<uint32_t>(5);
        deepseek_b1_ops::AllReduce::Reader<AllReduceReaderCTArgs> reader;
        reader(args);
    }

#elif defined(COMPILE_FOR_TRISC)
    if constexpr (Core::is_allreduce_receiver_core) {
        DeviceZoneScopedN("CCL_COMPUTE");
        deepseek_b1_ops::AllReduce::ComputeArgs args{};
        deepseek_b1_ops::AllReduce::Compute<AllReduceComputeCTArgs> compute;
        compute(args);
    }
#endif
#endif  // SKIP_CCL
}
