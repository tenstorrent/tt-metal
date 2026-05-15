// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

//
// Multi-upstream D2H sender (BRISC-only).
//
// Per iteration, this kernel assembles a single D2H page from N upstream
// receiver sockets (read sequentially in order 0..N-1), writes the assembled
// payload directly to PCIe via the D2H sender socket, then advances the D2H
// FIFO. There is no NCRISC half and no inter-RISC handshake — a single BRISC
// kernel owns the D2H sender socket and all N upstream receiver sockets.
//
// Layout of one D2H page in PCIe:
//   bytes [i * upstream_page_size, (i+1) * upstream_page_size)
//     for i in [0, N-1)
//   bytes [(N-1) * upstream_page_size,
//          N * upstream_page_size + forward_metadata_size_bytes)
//     for the last upstream (extra metadata bytes appended).
//
// Host-side preconditions:
//   - All upstream sender cores must share a device with this kernel
//     (no fabric on the receiver side).
//   - d2h_page_size == N * upstream_page_size + forward_metadata_size_bytes.
//

#include <array>
#include <cstddef>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "api/socket_api.h"
#include "pcie_noc_utils.h"
#include "../../../unified_kernels/termination.hpp"

constexpr uint32_t send_socket_config_addr = get_compile_time_arg_val(0);
constexpr uint32_t termination_semaphore_addr = get_compile_time_arg_val(1);
constexpr uint32_t d2h_page_size = get_compile_time_arg_val(2);
constexpr uint32_t upstream_page_size = get_compile_time_arg_val(3);
constexpr uint32_t num_upstream_sockets = get_compile_time_arg_val(4);
constexpr uint32_t forward_metadata_size_bytes = get_compile_time_arg_val(5);

constexpr uint32_t upstream_socket_addrs_start_idx = 6;

template <size_t START_IDX, size_t COUNT, size_t I = 0>
struct CTAArrayFiller {
    static constexpr void fill(std::array<uint32_t, COUNT>& arr) {
        arr[I] = get_compile_time_arg_val(START_IDX + I);
        if constexpr (I + 1 < COUNT) {
            CTAArrayFiller<START_IDX, COUNT, I + 1>::fill(arr);
        }
    }
};

template <size_t START_IDX, size_t COUNT>
constexpr std::array<uint32_t, COUNT> fill_ct_args_array() {
    std::array<uint32_t, COUNT> arr{};
    if constexpr (COUNT > 0) {
        CTAArrayFiller<START_IDX, COUNT>::fill(arr);
    }
    return arr;
}

constexpr auto receiver_socket_config_addrs =
    fill_ct_args_array<upstream_socket_addrs_start_idx, num_upstream_sockets>();

void kernel_main() {
    DPRINT << "Starting d2h multi-upstream sender kernel" << ENDL();
    DEVICE_PRINT("Starting d2h multi-upstream sender kernel\n");

    SocketSenderInterface sender_socket = create_sender_socket_interface(send_socket_config_addr);
    set_sender_socket_page_size(sender_socket, d2h_page_size);

    SocketReceiverInterface receiver_sockets[num_upstream_sockets];
    for (uint32_t i = 0; i < num_upstream_sockets; ++i) {
        receiver_sockets[i] = create_receiver_socket_interface(receiver_socket_config_addrs[i]);
        const uint32_t rx_page_size =
            (i == num_upstream_sockets - 1) ? upstream_page_size + forward_metadata_size_bytes : upstream_page_size;
        set_receiver_socket_page_size(receiver_sockets[i], rx_page_size);
    }

    const uint32_t write_addr_hi = sender_socket.d2h.data_addr_hi;
    const uint32_t pcie_xy_enc = sender_socket.d2h.pcie_xy_enc;

    noc_write_init_state<write_cmd_buf>(NOC_INDEX, NOC_UNICAST_WRITE_VC);

    volatile tt_l1_ptr uint32_t* termination_semaphore =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(termination_semaphore_addr);

    bool terminated = false;
    while (!terminated) {
        socket_reserve_pages(sender_socket, 1);

        const uint64_t d2h_base_addr =
            ((static_cast<uint64_t>(write_addr_hi) << 32) | sender_socket.downstream_fifo_addr) +
            sender_socket.write_ptr;

        for (uint32_t i = 0; i < num_upstream_sockets; ++i) {
            if (!deepseek_b1_ops::socket_wait_for_pages_with_termination(
                    receiver_sockets[i], 1, termination_semaphore)) {
                terminated = true;
                break;
            }
            noc_async_wide_write_any_len_with_state(
                NOC_INDEX,
                receiver_sockets[i].read_ptr,
                pcie_xy_enc,
                d2h_base_addr + i * upstream_page_size,
                receiver_sockets[i].page_size);
            socket_pop_pages(receiver_sockets[i], 1);
            noc_async_writes_flushed();
            socket_notify_sender(receiver_sockets[i]);
        }

        if (terminated) {
            break;
        }

        socket_push_pages(sender_socket, 1);
        socket_notify_receiver(sender_socket);
        invalidate_l1_cache();
    }

    update_socket_config(sender_socket);
    socket_barrier(sender_socket);
    for (uint32_t i = 0; i < num_upstream_sockets; ++i) {
        update_socket_config(receiver_sockets[i]);
    }

    noc_async_write_barrier();
    noc_async_read_barrier();
}
