// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Unified Forward kernel — standalone wrapper around forward.hpp
// - BRISC: Socket reader (entry column) or no-op (non-entry column)
// - NCRISC: Tensor writer + optional cross-column fabric send
// - TRISC: No-op (dataflow only)

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"
#include "../../../unified_kernels/forward.hpp"

void kernel_main() {
    using Forward = deepseek_b1_ops::Forward;

#if defined(COMPILE_FOR_BRISC)
    using FwdCTArgs = Forward::ReaderCTArgs<
        get_named_compile_time_arg_val("forward_cb_id"),
        get_named_compile_time_arg_val("forward_num_pages"),
        get_named_compile_time_arg_val("forward_is_entry_column")>;

    Forward::ReaderArgs fwd_args{
        get_common_arg_val<uint32_t>(0),  // socket_config_addr
        get_common_arg_val<uint32_t>(1),  // socket_page_size
        get_common_arg_val<uint32_t>(2),  // socket_num_pages
    };

#elif defined(COMPILE_FOR_NCRISC)
    using FwdCTArgs = Forward::WriterCTArgs<
        get_named_compile_time_arg_val("forward_cb_id"),
        get_named_compile_time_arg_val("forward_num_pages"),
        get_named_compile_time_arg_val("forward_page_size"),
        get_named_compile_time_arg_val("forward_is_entry_column"),
        get_named_compile_time_arg_val("forward_fabric_max_payload"),
        get_named_compile_time_arg_val("forward_num_fabric_packets"),
        get_named_compile_time_arg_val("forward_cross_column_payload")>;

    Forward::WriterArgs fwd_args{
        get_common_arg_val<uint32_t>(0),  // tensor_address
        get_common_arg_val<uint32_t>(1),  // my_noc_x
        get_common_arg_val<uint32_t>(2),  // my_noc_y
        get_common_arg_val<uint32_t>(3),  // cross_col_sem_addr
        get_common_arg_val<uint32_t>(4),  // partner_tensor_addr
        get_common_arg_val<uint32_t>(5),  // partner_noc_x
        get_common_arg_val<uint32_t>(6),  // partner_noc_y
        get_common_arg_val<uint32_t>(7),  // partner_chip_id
        get_common_arg_val<uint32_t>(8),  // partner_mesh_id
    };

#elif defined(COMPILE_FOR_TRISC)
    using FwdCTArgs = Forward::ComputeCTArgs;
    Forward::ComputeArgs fwd_args{};
#endif

    constexpr uint32_t num_iterations = get_named_compile_time_arg_val("forward_num_iterations");

    for (uint32_t i = 0; i < num_iterations; i++) {
        DeviceZoneScopedN("FORWARD");
        Forward::Op<FwdCTArgs, true> fwd;
        fwd(fwd_args);
    }
}
