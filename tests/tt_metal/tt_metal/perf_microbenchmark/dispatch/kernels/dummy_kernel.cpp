// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include "dataflow_api.h"
#include "debug/dprint.h"
// All of these includes are needed for fabric
#include "tt_metal/fabric/hw/inc/tt_fabric.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_interface.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"

using namespace tt::tt_fabric;

struct FabricInternals {
    fabric_client_interface_t client_interface;
    fabric_router_l1_config_t routing_table[2];
};

// fabric_endpoint init assumes routing table is after interface
static_assert(
    offsetof(FabricInternals, client_interface) + sizeof(fabric_client_interface_t) ==
    offsetof(FabricInternals, routing_table));

constexpr bool i_am_downstream = get_compile_time_arg_val(0);
constexpr uint32_t other_device_mesh_id = get_compile_time_arg_val(1);
constexpr uint32_t other_device_logical_device_id = get_compile_time_arg_val(2);
constexpr uint32_t fabric_router_xy = get_compile_time_arg_val(3);  // XY routing
constexpr uint32_t eth_chan = get_compile_time_arg_val(4);          // Eth Chan
constexpr uint32_t cb_base = get_compile_time_arg_val(5);           // where we store our data
constexpr uint32_t fabric_interface_router_addr = get_compile_time_arg_val(6);

packet_header_t packet_header __attribute__((aligned(16)));

void kernel_main() {
    auto fabric_state = reinterpret_cast<volatile FabricInternals*>(fabric_interface_router_addr);
    auto client_interface_ptr = (volatile tt_l1_ptr fabric_client_interface_t*)&(fabric_state->client_interface);

    zero_l1_buf((uint32_t*)&packet_header, sizeof(packet_header_t));

    uint32_t outbound_eth_chan;
    if (i_am_downstream) {  // device 1
        outbound_eth_chan = 1;
    } else {  // device 0
        outbound_eth_chan = 8;
    }

    fabric_endpoint_init(client_interface_ptr, outbound_eth_chan);

    auto fabric_header_addr = cb_base;
    auto sync_sem = reinterpret_cast<volatile uint32_t*>((uint32_t)fabric_header_addr + sizeof(packet_header_t));
    auto data_ptr = reinterpret_cast<volatile uint32_t*>((uint32_t)sync_sem + sizeof(uint32_t));

    DPRINT << "Initial data = 0x" << HEX() << data_ptr[0] << " sync = 0x" << sync_sem[0] << " | note: data addr = 0x"
           << (uint32_t)data_ptr << " sync addr = 0x" << (uint32_t)sync_sem << " cb base = 0x" << cb_base << ENDL();
    if (i_am_downstream) {
        // Wait for data
        while (data_ptr[0] == 0 && sync_sem[0] == 0);
    } else {
        // Send data
        data_ptr[0] = 0xdeadbeef;
        fabric_async_write(
            client_interface_ptr,
            fabric_router_xy,
            (uint32_t)&packet_header,
            other_device_mesh_id,
            other_device_logical_device_id,
            (uint32_t)data_ptr,
            sizeof(uint32_t));
        fabric_wait_for_pull_request_flushed(client_interface_ptr);
    }
}
