// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include "dataflow_api.h"
#include "debug/dprint.h"
// All of these includes are needed for fabric
#include "fabric_host_interface.h"
#include "risc_common.h"
#include "tt_metal/fabric/hw/inc/tt_fabric.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_interface.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"

using namespace tt::tt_fabric;

struct FabricInternals {
    fabric_client_interface_t client_interface;
    packet_header_t header;
    uint32_t data;
    uint32_t sync_sem;
};

constexpr bool i_am_downstream = get_compile_time_arg_val(0);
constexpr uint32_t other_device_mesh_id = get_compile_time_arg_val(1);
constexpr uint32_t other_device_logical_device_id = get_compile_time_arg_val(2);
constexpr uint32_t fabric_router_xy = get_compile_time_arg_val(3);  // XY routing
constexpr uint32_t eth_chan = get_compile_time_arg_val(4);          // Eth Chan
constexpr uint32_t cb_base = get_compile_time_arg_val(5);           // where we store our data
constexpr uint32_t fabric_interface_router_addr = get_compile_time_arg_val(6);
constexpr uint32_t other_x = get_compile_time_arg_val(7);
constexpr uint32_t other_y = get_compile_time_arg_val(8);

auto fabric_state = reinterpret_cast<volatile FabricInternals*>(cb_base);

void wait_for_data() {
    auto data = &fabric_state->data;
    // Wait for data
    DPRINT << "Wait for data at 0x" << HEX() << (uint32_t)data << "\n";
    while (1) {
        if (data[0] == 0xcaf3caf3) {
            DPRINT << "Data changed to 0x" << HEX() << data[0] << ENDL();
            return;
        }
    };
}

void wait_for_sem() {
    auto sem = &fabric_state->sync_sem;
    while (*sem == 0);
    DPRINT << "Sem set to 0x" << HEX() << *sem << ENDL();
}

void test_async_write() {
    if constexpr (i_am_downstream) {
        wait_for_data();
    } else {
        // Send data
        auto header = &fabric_state->header;
        auto data = &fabric_state->data;
        data[0] = 0xcaf3caf3;
        auto dst_l1_addr = get_noc_addr(other_x, other_y, (uint32_t)data);
        DPRINT << "fabric_async_write\n";
        fabric_async_write(
            (volatile tt_l1_ptr fabric_client_interface_t*)&fabric_state->client_interface,
            fabric_router_xy,
            (uint32_t)header,
            0,
            1,
            dst_l1_addr,
            sizeof(uint32_t) + sizeof(packet_header_t));
        fabric_wait_for_pull_request_flushed(
            (volatile tt_l1_ptr fabric_client_interface_t*)&fabric_state->client_interface);
        DPRINT << "fabric_async_write flushed\n";
    }
}

void test_atomic_inc() {
    if constexpr (i_am_downstream) {
        wait_for_sem();
    } else {
        auto header = &fabric_state->header;
        auto dst_sync_addr = get_noc_addr(other_x, other_y, (uint32_t)&fabric_state->sync_sem);
        DPRINT << "fabric_atomic_inc\n";
        fabric_atomic_inc(
            (volatile tt_l1_ptr fabric_client_interface_t*)&fabric_state->client_interface,
            fabric_router_xy,
            (uint32_t)header,
            0,
            1,
            dst_sync_addr,
            0xbeef,
            31);
        fabric_wait_for_pull_request_flushed(
            (volatile tt_l1_ptr fabric_client_interface_t*)&fabric_state->client_interface);
        DPRINT << "fabric_atomic_inc flushed\n";
    }
}

void test_async_write_and_inc() {
    constexpr uint32_t k_Pattern = 0xabababab;
    if constexpr (i_am_downstream) {
        auto data = &fabric_state->data;
        auto sync = &fabric_state->sync_sem;
        DPRINT << "Wait for sync sem == 5\n";
        // wait for 5
        while (*sync != 5) {
        }
        // check data
        if (data[0] == k_Pattern) {
            DPRINT << "Data check passed 0x" << HEX() << data[0] << ENDL();
        } else {
            DPRINT << "Data mismatch 0x" << HEX() << data[0] << ENDL();
        }
    } else {
        auto header = &fabric_state->header;
        auto data = &fabric_state->data;
        auto sync = &fabric_state->sync_sem;
        auto dst_l1_addr = get_noc_addr(other_x, other_y, (uint32_t)data);
        auto dst_sync_addr = get_noc_addr(other_x, other_y, (uint32_t)sync);
        data[0] = k_Pattern;
        DPRINT << "fabric_async_write_atomic_inc\n";
        fabric_async_write_atomic_inc(
            (volatile tt_l1_ptr fabric_client_interface_t*)&fabric_state->client_interface,
            fabric_router_xy,
            (uint32_t)header,
            0,
            1,
            dst_l1_addr,                                 // write addr
            dst_sync_addr,                               // atomic addr
            sizeof(uint32_t) + sizeof(packet_header_t),  // size
            5                                            // atomic inc
        );
        fabric_wait_for_pull_request_flushed(
            (volatile tt_l1_ptr fabric_client_interface_t*)&fabric_state->client_interface);
        DPRINT << "fabric_async_write_atomic_inc flushed\n";
    }
}

void kernel_main() {
    auto client_interface = (volatile tt_l1_ptr fabric_client_interface_t*)&(fabric_state->client_interface);
    fabric_endpoint_init(client_interface, eth_chan);
    test_async_write();
    test_atomic_inc();
    test_async_write_and_inc();
}
