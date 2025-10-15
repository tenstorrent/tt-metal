// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hal_types.hpp"
#include "impl/context/metal_context.hpp"
#include "kernel_types.hpp"

namespace tt::tt_metal::eth_test_common {

inline void set_arch_specific_eth_config(tt_metal::EthernetConfig& config) {
    if (!tt::tt_metal::MetalContext::instance().hal().get_eth_fw_is_cooperative()) {
        if (tt::tt_metal::MetalContext::instance().hal().get_num_risc_processors(HalProgrammableCoreType::ACTIVE_ETH) >
            1) {
            // Ensure noc index == processor index to not cross eachother
            config.noc = static_cast<tt_metal::NOC>(config.processor);
        } else {
            // Ensure it's using NOC1 as Base FW is using NOC0
            config.noc = tt_metal::NOC::NOC_1;
        }
    }
}

}  // namespace tt::tt_metal::eth_test_common
