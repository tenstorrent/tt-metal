// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mock_device_util.hpp"

namespace tt::tt_metal::experimental {

std::optional<std::string> get_mock_cluster_desc_name(tt::ARCH arch, uint32_t num_chips) {
    // Each entry maps to a cluster descriptor YAML in
    // tt_metal/third_party/umd/tests/cluster_descriptor_examples/. New entries must
    // reference an existing file there; otherwise mock mode will fail to load the
    // cluster at runtime.
    switch (arch) {
        case tt::ARCH::WORMHOLE_B0:
            switch (num_chips) {
                case 1: return "wormhole_N150.yaml";
                case 2: return "wormhole_N300.yaml";
                case 4: return "2x2_n300_cluster_desc.yaml";
                case 8: return "t3k_cluster_desc.yaml";
                case 32: return "6u_cluster_desc.yaml";  // Galaxy 6U, 4x8 grid
                default: return std::nullopt;
            }
        case tt::ARCH::BLACKHOLE:
            switch (num_chips) {
                case 1: return "blackhole_P150.yaml";
                case 2: return "blackhole_P300_both_mmio.yaml";
                case 4: return "blackhole_4xP150.yaml";
                case 8: return "blackhole_8xP150.yaml";
                case 32: return "blackhole_galaxy.yaml";  // Blackhole Galaxy (UBB), 32 chips
                default: return std::nullopt;
            }
        case tt::ARCH::QUASAR:
            switch (num_chips) {
                case 1: return "quasar_Q1.yaml";
                default: return std::nullopt;
            }
        default: return std::nullopt;
    }
}

}  // namespace tt::tt_metal::experimental
