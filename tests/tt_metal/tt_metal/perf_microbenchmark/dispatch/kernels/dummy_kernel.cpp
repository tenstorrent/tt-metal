// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include "dataflow_api.h"
#include "debug/dprint.h"
// All of these includes are needed for fabric
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

packet_header_t packet_header __attribute__((aligned(16)));

void kernel_main() {
    auto fabric_state = reinterpret_cast<volatile FabricInternals*>(cb_base);
    auto client_interface = (volatile tt_l1_ptr fabric_client_interface_t*)&(fabric_state->client_interface);
    auto header = (volatile tt_l1_ptr uint32_t*)&(fabric_state->header);
    auto data = (volatile tt_l1_ptr uint32_t*)&(fabric_state->data);
    auto sync_sem = (volatile tt_l1_ptr uint32_t*)&(fabric_state->client_interface);

    fabric_endpoint_init(client_interface, eth_chan);

    if constexpr (i_am_downstream) {
        // Wait for data
        DPRINT << "Wait for data at 0x" << HEX() << (uint32_t)data << "\n";
        while (1) {
            if (data[0] == 0xcaf3caf3) {
                DPRINT << "Data changed to 0x" << HEX() << data[0] << ENDL();
                return;
            }
        };
    } else {
        // Send data
        data[0] = 0xcaf3caf3;
        auto dst_l1_addr = get_noc_addr(other_x, other_y, (uint32_t)data);
        fabric_async_write(
            client_interface,
            fabric_router_xy,
            (uint32_t)header,
            0,
            1,
            dst_l1_addr,
            sizeof(uint32_t) + sizeof(packet_header_t));
        fabric_wait_for_pull_request_flushed(client_interface);
        noc_async_writes_flushed();
        DPRINT << "Sender flushed\n";
    }
}
