// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mock_device_util.hpp"

namespace tt::tt_metal::experimental {

std::optional<std::string> get_mock_cluster_desc_name(tt::ARCH arch, uint32_t num_chips) {
    switch (arch) {
        case tt::ARCH::WORMHOLE_B0:
            switch (num_chips) {
                case 1: return "wormhole_N150.yaml";
                case 2: return "wormhole_N300.yaml";
                case 4: return "2x2_n300_cluster_desc.yaml";
                case 8: return "t3k_cluster_desc.yaml";
                default: return std::nullopt;
            }
        case tt::ARCH::BLACKHOLE:
            switch (num_chips) {
                case 1: return "blackhole_P150.yaml";
                case 2: return "blackhole_P300_both_mmio.yaml";
                default: return std::nullopt;
            }
        default: return std::nullopt;
    }
}

}  // namespace tt::tt_metal::experimental
