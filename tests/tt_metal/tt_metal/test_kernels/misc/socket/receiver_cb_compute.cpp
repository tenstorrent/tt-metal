// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "socket_api.h"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t socket_config_addr = get_compile_time_arg_val(0);
    constexpr uint32_t input_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t output_cb_index = get_compile_time_arg_val(2);
    constexpr uint32_t page_size = get_compile_time_arg_val(3);
    constexpr uint32_t data_size = get_compile_time_arg_val(4);
    constexpr uint32_t num_tiles_per_page = get_compile_time_arg_val(5);
    constexpr uint32_t num_pages = data_size / page_size;

    SocketReceiverInterface socket = create_receiver_socket_interface(socket_config_addr);
    set_receiver_socket_page_size(socket, page_size);
    assign_local_cb_to_socket(socket, input_cb_index);

    unary_op_init_common(input_cb_index, output_cb_index);
    copy_tile_init(input_cb_index);
    for (uint32_t p = 0; p < num_pages; ++p) {
        socket_wait_for_pages(socket, 1);
        cb_reserve_back(output_cb_index, num_tiles_per_page);
        tile_regs_acquire();
        tile_regs_wait();
        for (uint32_t i = 0; i < num_tiles_per_page; ++i) {
            copy_tile(input_cb_index, i, i);
            pack_tile(i, output_cb_index);
        }
        tile_regs_commit();
        tile_regs_release();
        cb_push_back(output_cb_index, num_tiles_per_page);
        cb_pop_front(input_cb_index, num_tiles_per_page);
        socket_pop_pages(socket, 1);
    }
}
}  // namespace NAMESPACE
