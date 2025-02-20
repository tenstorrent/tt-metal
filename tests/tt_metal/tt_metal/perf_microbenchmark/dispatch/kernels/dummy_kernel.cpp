// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include "debug/dprint.h"
#include "fabric/hw/inc/tt_fabric_interface.h"
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

void kernel_main() {
    auto fabric_state = reinterpret_cast<volatile FabricInternals*>(0x184A0);
    auto client_interface_ptr = (volatile tt_l1_ptr fabric_client_interface_t*)&(fabric_state->client_interface);

    fabric_endpoint_init<RoutingType::ROUTING_TABLE>(client_interface_ptr, 8);

    auto sync_sem = reinterpret_cast<volatile uint32_t*>(0x80000);
    auto data_ptr = reinterpret_cast<volatile uint32_t*>(0x80004);
    sync_sem[0] = 0;
    data_ptr[0] = 0;

    if (i_am_downstream) {
        // Wait for data
        DPRINT << "Waiting for data\n";
        while (data_ptr[0] != 0xdeadbeef);
        DPRINT << "Received deadbeef\n";
    } else {
        // Send data
        uint32_t my_src_data = 0xdeadbeef;
        fabric_async_write_atomic_inc(
            client_interface_ptr,
            0,
            (uint32_t)data_ptr,
            1,
            1,
            (uint32_t)data_ptr,
            (uint32_t)sync_sem,
            sizeof(uint32_t),
            1);
        DPRINT << "Sent data\n";
    }
}
