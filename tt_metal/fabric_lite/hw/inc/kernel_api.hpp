// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "host_interface.hpp"

namespace fabric_lite {

inline void service_fabric_lite_channels() {
#if defined(FABRIC_LITE_CONFIG_START) && FABRIC_LITE_CONFIG_START != 0 && defined(COMPILE_FOR_ERISC)
    auto config = reinterpret_cast<volatile fabric_lite::FabricLiteMemoryMap*>(FABRIC_LITE_CONFIG_START);
    void (*service_routing)() = (void (*)())((uint32_t*)config->service_fabric_lite_addr);
    service_routing();
#endif
}

}  // namespace fabric_lite
