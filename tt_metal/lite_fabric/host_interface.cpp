// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/lite_fabric/hw/inc/lf_dev_mem_map.hpp"
#include "tt_metal/lite_fabric/hw/inc/host_interface.hpp"

namespace lite_fabric {

uint32_t FabricLiteMemoryMap::get_address() {
    auto addr = LITE_FABRIC_CONFIG_START;
    return addr;
}

uint32_t FabricLiteMemoryMap::get_host_interface_addr() {
    return get_address() + offsetof(lite_fabric::FabricLiteMemoryMap, host_interface);
}

uint32_t FabricLiteMemoryMap::get_send_channel_addr() {
    return get_address() + offsetof(lite_fabric::FabricLiteMemoryMap, sender_channel_buffer);
}

uint32_t FabricLiteMemoryMap::get_receiver_channel_addr() {
    return get_address() + offsetof(lite_fabric::FabricLiteMemoryMap, receiver_channel_buffer);
}

uint32_t FabricLiteMemoryMap::get_service_channel_func_addr() {
    return get_address() + offsetof(lite_fabric::FabricLiteMemoryMap, service_lite_fabric_addr);
}

template struct HostToFabricLiteInterface<lite_fabric::SENDER_NUM_BUFFERS_ARRAY[0], lite_fabric::CHANNEL_BUFFER_SIZE>;

}  // namespace lite_fabric
