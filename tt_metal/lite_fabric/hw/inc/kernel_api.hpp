// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/lite_fabric/hw/inc/host_interface.hpp"
#include "tt_metal/lite_fabric/hw/inc/lf_dev_mem_map.hpp"

namespace lite_fabric {

inline void service_lite_fabric_channels() {
#if defined(ARCH_BLACKHOLE) && defined(COMPILE_FOR_ERISC) && defined(ROUTING_FW_ENABLED)
    auto config = reinterpret_cast<volatile lite_fabric::FabricLiteMemoryMap*>(LITE_FABRIC_CONFIG_START);
    void (*service_routing)() = (void (*)())((uint32_t*)config->service_lite_fabric_addr);
    service_routing();
#endif
}

}  // namespace lite_fabric
