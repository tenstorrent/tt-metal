// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"
#include "../../../unified_kernels/all_gather.hpp"

struct Core {
    static constexpr bool is_gather_core = get_named_compile_time_arg_val("is_allgather_gather_core") == 1;
    static constexpr bool is_transport_core = get_named_compile_time_arg_val("is_allgather_transport_core") == 1;
};

void kernel_main() {
    // ====================================================================
    // Arg setup
    // ====================================================================

#if defined(COMPILE_FOR_NCRISC)
    using GatherCT = deepseek_b1_ops::AllGather::GatherCTArgs<
        get_named_compile_time_arg_val("allgather_gather_slice_size_bytes"),
        get_named_compile_time_arg_val("allgather_gather_num_chunks"),
        get_named_compile_time_arg_val("allgather_ring_size"),
        get_named_compile_time_arg_val("allgather_recv_sem_bits_per_slot")>;

    deepseek_b1_ops::AllGather::GatherArgs gather_args{};
    if constexpr (Core::is_gather_core) {
        gather_args.local_input_addr = get_common_arg_val<uint32_t>(0);
        gather_args.output_buffer_addr = get_common_arg_val<uint32_t>(1);
        gather_args.self_slot_index = get_common_arg_val<uint32_t>(2);
        gather_args.transport_scratch_base_addr = get_common_arg_val<uint32_t>(3);
        gather_args.transport_noc_x = get_common_arg_val<uint32_t>(4);
        gather_args.transport_noc_y = get_common_arg_val<uint32_t>(5);
        gather_args.handoff_sem_bank_addr = get_common_arg_val<uint32_t>(6);
        gather_args.recv_sem_addr = get_common_arg_val<uint32_t>(7);
        gather_args.r2_src_slot_index = get_common_arg_val<uint32_t>(8);
    }
#endif

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
    using TransportCT = deepseek_b1_ops::AllGather::TransportCTArgs<
        get_named_compile_time_arg_val("allgather_slice_size_bytes"),
        get_named_compile_time_arg_val("allgather_num_chunks"),
        get_named_compile_time_arg_val("allgather_chunk_size_bytes"),
        get_named_compile_time_arg_val("allgather_last_chunk_bytes"),
        get_named_compile_time_arg_val("allgather_num_links"),
        get_named_compile_time_arg_val("allgather_recv_sem_bits_per_slot"),
        get_named_compile_time_arg_val("allgather_r2_active")>;

    deepseek_b1_ops::AllGather::TransportArgs transport_args{};
    if constexpr (Core::is_transport_core) {
        transport_args.scratch_base_addr = get_common_arg_val<uint32_t>(0);
        transport_args.handoff_sem_bank_addr = get_common_arg_val<uint32_t>(1);
        transport_args.dest_output_base_addr = get_common_arg_val<uint32_t>(2);
        transport_args.r1_dest_slot_index = get_common_arg_val<uint32_t>(3);
        transport_args.dest_noc_x = get_common_arg_val<uint32_t>(4);
        transport_args.dest_noc_y = get_common_arg_val<uint32_t>(5);
        transport_args.dest_recv_sem_addr = get_common_arg_val<uint32_t>(6);
        transport_args.r2_dest_slot_index = get_common_arg_val<uint32_t>(7);
        transport_args.per_core_rta_start_idx = 0;
    }
#endif

    // ====================================================================
    // Op invocation
    // ====================================================================

#if defined(COMPILE_FOR_NCRISC)
    if constexpr (Core::is_gather_core) {
        DeviceZoneScopedN("ALLGATHER_GATHER");
        deepseek_b1_ops::AllGather::GatherController<GatherCT> controller;
        controller(gather_args);
    }
#endif

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
    if constexpr (Core::is_transport_core) {
        DeviceZoneScopedN("ALLGATHER_TRANSPORT");
        deepseek_b1_ops::AllGather::TransportSender<TransportCT> sender;
        sender(transport_args);
    }
#endif
}
