// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"
#include "../../../unified_kernels/all_reduce.hpp"
#include "../../../unified_kernels/all_gather.hpp"
#include "../../../unified_kernels/gather_reduce.hpp"

#if defined(COMPILE_FOR_NCRISC)
FORCE_INLINE void run_gather_reduce_sender() {
    constexpr uint32_t src_cb = get_named_compile_time_arg_val("gather_reduce_src_cb");
    constexpr uint32_t src_num_pages = get_named_compile_time_arg_val("gather_reduce_src_num_pages");
    unified_kernels::setup_sharded_buffer(src_cb, src_num_pages);

    deepseek_b1_ops::GatherReduce::SenderArgs args{
        get_named_compile_time_arg_val("gather_reduce_dest_noc_x"),
        get_named_compile_time_arg_val("gather_reduce_dest_noc_y"),
        get_named_compile_time_arg_val("gather_reduce_data_size_bytes"),
        get_named_compile_time_arg_val("gather_reduce_receiver_semaphore_addr"),
        src_cb,
        src_num_pages,
        get_named_compile_time_arg_val("gather_reduce_grid_start_x"),
        get_named_compile_time_arg_val("gather_reduce_grid_start_y"),
        get_named_compile_time_arg_val("gather_reduce_grid_end_x"),
        get_named_compile_time_arg_val("gather_reduce_grid_end_y"),
        get_named_compile_time_arg_val("gather_reduce_half_num_cores"),
        get_named_compile_time_arg_val("gather_reduce_dst_cb"),
        get_named_compile_time_arg_val("gather_reduce_half_size_bytes"),
        0,
        noc_index,
    };

    deepseek_b1_ops::GatherReduce::Op<true, false, false, true, false> gather_reduce;
    gather_reduce(args);
}
#elif defined(COMPILE_FOR_BRISC)
FORCE_INLINE void run_gather_reduce_receiver() {
    deepseek_b1_ops::GatherReduce::ReceiverArgs args{
        get_named_compile_time_arg_val("gather_reduce_noc0_num_senders"),
        get_named_compile_time_arg_val("gather_reduce_noc1_num_senders"),
        get_named_compile_time_arg_val("gather_reduce_noc0_receiver_semaphore_addr"),
        get_named_compile_time_arg_val("gather_reduce_noc1_receiver_semaphore_addr"),
        get_named_compile_time_arg_val("gather_reduce_dst_cb"),
        get_named_compile_time_arg_val("gather_reduce_dst_num_tiles"),
    };

    deepseek_b1_ops::GatherReduce::Op<false, true, false, true, false> gather_reduce;
    gather_reduce(args);
}
#elif defined(COMPILE_FOR_TRISC)
FORCE_INLINE void run_gather_reduce_compute() {
    deepseek_compute_kernel_init();

    deepseek_b1_ops::GatherReduce::ComputeArgs args{
        get_named_compile_time_arg_val("gather_reduce_scratch_cb"),
        get_named_compile_time_arg_val("gather_reduce_out_cb"),
        get_named_compile_time_arg_val("gather_reduce_dst_num_tiles"),
    };

    deepseek_b1_ops::GatherReduce::Op<false, false, true, true, false> gather_reduce;
    gather_reduce(args);
}
#endif

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
template <typename WriterCT>
FORCE_INLINE deepseek_b1_ops::AllReduce::SenderArgs make_allreduce_sender_args() {
    deepseek_b1_ops::AllReduce::SenderArgs args{};
    args.intermediate_buffer_address = get_common_arg_val<uint32_t>(0);
    args.dest_noc_x = get_common_arg_val<uint32_t>(1);
    args.dest_noc_y = get_common_arg_val<uint32_t>(2);
    args.per_core_rta_start_idx = 0;
    return args;
}

template <typename TransportCT>
FORCE_INLINE deepseek_b1_ops::AllGather::TransportArgs make_allgather_transport_args() {
    deepseek_b1_ops::AllGather::TransportArgs args{};
    args.scratch_base_addr = get_named_compile_time_arg_val("allgather_scratch_base_addr");
    args.handoff_sem_bank_addr = get_named_compile_time_arg_val("allgather_handoff_sem_addr");
    args.dest_output_base_addr = get_named_compile_time_arg_val("allgather_dest_output_base_addr");
    args.r1_dest_slot_index = get_named_compile_time_arg_val("allgather_r1_dest_slot_index");
    args.dest_noc_x = get_named_compile_time_arg_val("allgather_dest_noc_x");
    args.dest_noc_y = get_named_compile_time_arg_val("allgather_dest_noc_y");
    args.dest_recv_sem_addr = get_named_compile_time_arg_val("allgather_dest_recv_sem_addr");
    args.r2_dest_slot_index = get_named_compile_time_arg_val("allgather_r2_dest_slot_index");
    args.per_core_rta_start_idx = get_common_arg_val<uint32_t>(3);
    return args;
}
#endif

void kernel_main() {
    constexpr bool is_gather_sender = get_named_compile_time_arg_val("is_gather_sender_core") == 1;
    constexpr bool is_gather_receiver = get_named_compile_time_arg_val("is_gather_receiver_core") == 1;
    constexpr bool is_allreduce_sender = get_named_compile_time_arg_val("is_allreduce_sender_core") == 1;
    constexpr bool is_allreduce_receiver = get_named_compile_time_arg_val("is_allreduce_receiver_core") == 1;
    constexpr bool is_ccl_sync_producer = get_named_compile_time_arg_val("is_ccl_sync_producer_core") == 1;
    constexpr bool is_ccl_sync2_producer = get_named_compile_time_arg_val("is_ccl_sync2_producer_core") == 1;
    constexpr bool enable_allgather_transport = get_named_compile_time_arg_val("allgather_transport_enabled") == 1;
    constexpr bool enable_allgather_gather = get_named_compile_time_arg_val("allgather_gather_enabled") == 1;
    constexpr bool open_allgather_after_allreduce =
        get_named_compile_time_arg_val("allgather_open_after_allreduce") == 1;

#if defined(COMPILE_FOR_NCRISC)
    if constexpr (is_ccl_sync_producer) {
        constexpr uint32_t ccl_sync_sem_addr = get_named_compile_time_arg_val("ccl_sync_semaphore_addr");
        constexpr uint32_t ccl_sync_dest_noc_x = get_named_compile_time_arg_val("ccl_sync_dest_noc_x");
        constexpr uint32_t ccl_sync_dest_noc_y = get_named_compile_time_arg_val("ccl_sync_dest_noc_y");
        uint64_t ccl_sync_sem_noc_addr = get_noc_addr(ccl_sync_dest_noc_x, ccl_sync_dest_noc_y, ccl_sync_sem_addr);
        noc_semaphore_inc(ccl_sync_sem_noc_addr, 2);
        noc_async_atomic_barrier();
    }
#endif
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
    if constexpr (is_ccl_sync2_producer) {
        constexpr uint32_t ccl_sync_sem_addr = get_named_compile_time_arg_val("ccl_sync_semaphore2_addr");
        constexpr uint32_t ccl_sync_dest_noc_x = get_named_compile_time_arg_val("ccl_sync2_dest_noc_x");
        constexpr uint32_t ccl_sync_dest_noc_y = get_named_compile_time_arg_val("ccl_sync2_dest_noc_y");
        uint64_t ccl_sync_sem_noc_addr = get_noc_addr(ccl_sync_dest_noc_x, ccl_sync_dest_noc_y, ccl_sync_sem_addr);
        noc_semaphore_inc(ccl_sync_sem_noc_addr, 1);
        noc_async_atomic_barrier();
    }
#endif

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
    if constexpr (is_allreduce_sender) {
        using WriterCT = deepseek_b1_ops::AllReduce::WriterCTArgs<
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

        deepseek_b1_ops::AllReduce::SenderArgs args = make_allreduce_sender_args<WriterCT>();
        deepseek_b1_ops::AllReduce::WriterSingleLink<WriterCT> writer;
        using AllGatherTransportCT = deepseek_b1_ops::AllGather::TransportCTArgs<
            get_named_compile_time_arg_val("allgather_slice_size_bytes"),
            get_named_compile_time_arg_val("allgather_num_chunks"),
            get_named_compile_time_arg_val("allgather_chunk_size_bytes"),
            get_named_compile_time_arg_val("allgather_last_chunk_bytes"),
            get_named_compile_time_arg_val("allgather_num_links"),
            get_named_compile_time_arg_val("allgather_recv_sem_bits_per_slot"),
            get_named_compile_time_arg_val("allgather_r2_active")>;
        deepseek_b1_ops::AllGather::TransportArgs allgather_transport_args =
            make_allgather_transport_args<AllGatherTransportCT>();
        deepseek_b1_ops::AllGather::TransportSender<AllGatherTransportCT> allgather_sender;
        PacketHeaderPool::reset();
        volatile tt_l1_ptr uint32_t* ccl_sync_sem =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_named_compile_time_arg_val("ccl_sync_semaphore_addr"));
        noc_semaphore_wait_min(ccl_sync_sem, 1);
        unified_kernels::semaphore_dec(ccl_sync_sem, 1);
        writer.open_connections(args, false);
        ccl_sync_sem =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_named_compile_time_arg_val("ccl_sync_semaphore2_addr"));
        noc_semaphore_wait(ccl_sync_sem, 2 * get_named_compile_time_arg_val("sdpa_fwd_num_cores"));
        if constexpr (enable_allgather_transport && !open_allgather_after_allreduce) {
            allgather_sender.open_connections(allgather_transport_args, false);
            if constexpr (!enable_allgather_gather) {
                volatile tt_l1_ptr uint32_t* allgather_handoff_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    get_named_compile_time_arg_val("allgather_handoff_sem_addr"));
                noc_semaphore_set(allgather_handoff_sem, 1);
            }
        }

#if defined(COMPILE_FOR_NCRISC)
        if constexpr (is_gather_sender) {
            run_gather_reduce_sender();
        }
#elif defined(COMPILE_FOR_BRISC)
        if constexpr (is_gather_receiver) {
            run_gather_reduce_receiver();
        }
#endif

        writer(args);
        if constexpr (enable_allgather_transport) {
            if constexpr (open_allgather_after_allreduce) {
                allgather_sender.open_connections(allgather_transport_args, false);
                if constexpr (!enable_allgather_gather) {
                    volatile tt_l1_ptr uint32_t* allgather_handoff_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                        get_named_compile_time_arg_val("allgather_handoff_sem_addr"));
                    noc_semaphore_set(allgather_handoff_sem, 1);
                }
            }
            allgather_sender(allgather_transport_args);
        }
    } else {
#if defined(COMPILE_FOR_NCRISC)
        if constexpr (is_gather_sender) {
            run_gather_reduce_sender();
        }
#elif defined(COMPILE_FOR_BRISC)
        if constexpr (is_gather_receiver) {
            run_gather_reduce_receiver();
        }
#endif
    }
#elif defined(COMPILE_FOR_TRISC)
    if constexpr (is_gather_receiver) {
        run_gather_reduce_compute();
    }
#endif

#if defined(COMPILE_FOR_NCRISC)
    if constexpr (is_allreduce_receiver) {
        using ReaderCT = deepseek_b1_ops::AllReduce::ReaderCTArgs<
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

        deepseek_b1_ops::AllReduce::ReceiverArgs args{};
        args.sem_bank_addr_0 = get_common_arg_val<uint32_t>(0);
        args.sem_bank_addr_1 = get_common_arg_val<uint32_t>(1);
        args.sender_noc_x = get_common_arg_val<uint32_t>(2);
        args.sender_noc_y = get_common_arg_val<uint32_t>(3);
        args.sender_local_data_l1_addr = get_common_arg_val<uint32_t>(4);
        args.local_ready_sem_bank_addr = get_common_arg_val<uint32_t>(5);

        deepseek_b1_ops::AllReduce::Reader<ReaderCT> reader;
        reader(args);

        if constexpr (enable_allgather_gather) {
            using AllGatherGatherCT = deepseek_b1_ops::AllGather::GatherCTArgs<
                get_named_compile_time_arg_val("allgather_gather_slice_size_bytes"),
                get_named_compile_time_arg_val("allgather_gather_num_chunks"),
                get_named_compile_time_arg_val("allgather_ring_size"),
                get_named_compile_time_arg_val("allgather_recv_sem_bits_per_slot")>;

            constexpr uint32_t out_cb = get_named_compile_time_arg_val("output_cb_id");
            constexpr uint32_t out_num_tiles = get_named_compile_time_arg_val("output_num_tiles");
            cb_wait_front(out_cb, out_num_tiles);

            deepseek_b1_ops::AllGather::GatherArgs allgather_gather_args{};
            allgather_gather_args.local_input_addr = get_read_ptr(out_cb);
            allgather_gather_args.output_buffer_addr =
                get_named_compile_time_arg_val("allgather_dest_output_base_addr");
            allgather_gather_args.self_slot_index = get_named_compile_time_arg_val("allgather_self_slot_index");
            allgather_gather_args.transport_scratch_base_addr =
                get_named_compile_time_arg_val("allgather_scratch_base_addr");
            allgather_gather_args.transport_noc_x = get_named_compile_time_arg_val("allgather_transport_noc_x");
            allgather_gather_args.transport_noc_y = get_named_compile_time_arg_val("allgather_transport_noc_y");
            allgather_gather_args.handoff_sem_bank_addr = get_named_compile_time_arg_val("allgather_handoff_sem_addr");
            allgather_gather_args.recv_sem_addr = get_named_compile_time_arg_val("allgather_recv_sem_addr");
            allgather_gather_args.r2_src_slot_index = get_named_compile_time_arg_val("allgather_r2_src_slot_index");

            deepseek_b1_ops::AllGather::GatherController<AllGatherGatherCT> allgather_controller;
            allgather_controller(allgather_gather_args);
            cb_pop_front(out_cb, out_num_tiles);
        }
    }
#endif

#if defined(COMPILE_FOR_TRISC)
    if constexpr (is_allreduce_receiver) {
        using ComputeCT = deepseek_b1_ops::AllReduce::ComputeCTArgs<
            get_named_compile_time_arg_val("allreduce_cb_remote"),
            get_named_compile_time_arg_val("allreduce_cb_local"),
            get_named_compile_time_arg_val("allreduce_cb_out"),
            get_named_compile_time_arg_val("allreduce_cb_residual"),
            get_named_compile_time_arg_val("allreduce_has_residual"),
            get_named_compile_time_arg_val("allreduce_num_tiles")>;

        deepseek_compute_kernel_init();

        deepseek_b1_ops::AllReduce::ComputeArgs args{};
        deepseek_b1_ops::AllReduce::Compute<ComputeCT> compute;
        compute(args);
    }
#endif
}
