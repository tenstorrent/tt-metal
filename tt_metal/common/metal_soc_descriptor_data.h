// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tt {
namespace metal_soc_descriptor_data {

// Structure to hold dram_view data for each architecture
struct DramView {
    size_t channel;
    std::vector<int> eth_endpoint;
    std::vector<int> worker_endpoint;
    size_t address_offset;
};

// Structure to hold all dram metadata for an architecture
struct DramMetadata {
    std::vector<DramView> dram_views;
    uint64_t dram_view_size;
};

// GRAYSKULL data
const DramMetadata GRAYSKULL_DRAM_METADATA = {
    .dram_views = {
        {.channel = 0, .eth_endpoint = {0}, .worker_endpoint = {0}, .address_offset = 0},
        {.channel = 1, .eth_endpoint = {0}, .worker_endpoint = {0}, .address_offset = 0},
        {.channel = 2, .eth_endpoint = {0}, .worker_endpoint = {0}, .address_offset = 0},
        {.channel = 3, .eth_endpoint = {0}, .worker_endpoint = {0}, .address_offset = 0},
        {.channel = 4, .eth_endpoint = {0}, .worker_endpoint = {0}, .address_offset = 0},
        {.channel = 5, .eth_endpoint = {0}, .worker_endpoint = {0}, .address_offset = 0},
        {.channel = 6, .eth_endpoint = {0}, .worker_endpoint = {0}, .address_offset = 0},
        {.channel = 7, .eth_endpoint = {0}, .worker_endpoint = {0}, .address_offset = 0}
    },
    .dram_view_size = 1073741824
};

// WORMHOLE_B0 data
const DramMetadata WORMHOLE_B0_DRAM_METADATA = {
    .dram_views = {
        {.channel = 0, .eth_endpoint = {0, 0}, .worker_endpoint = {2, 2}, .address_offset = 0},
        {.channel = 0, .eth_endpoint = {0, 0}, .worker_endpoint = {1, 1}, .address_offset = 1073741824},
        {.channel = 1, .eth_endpoint = {1, 1}, .worker_endpoint = {0, 0}, .address_offset = 0},
        {.channel = 1, .eth_endpoint = {1, 1}, .worker_endpoint = {2, 2}, .address_offset = 1073741824},
        {.channel = 2, .eth_endpoint = {0, 0}, .worker_endpoint = {1, 1}, .address_offset = 0},
        {.channel = 2, .eth_endpoint = {0, 0}, .worker_endpoint = {2, 2}, .address_offset = 1073741824},
        {.channel = 3, .eth_endpoint = {2, 2}, .worker_endpoint = {0, 0}, .address_offset = 0},
        {.channel = 3, .eth_endpoint = {2, 2}, .worker_endpoint = {1, 1}, .address_offset = 1073741824},
        {.channel = 4, .eth_endpoint = {1, 1}, .worker_endpoint = {2, 2}, .address_offset = 0},
        {.channel = 4, .eth_endpoint = {1, 1}, .worker_endpoint = {0, 0}, .address_offset = 1073741824},
        {.channel = 5, .eth_endpoint = {1, 1}, .worker_endpoint = {0, 0}, .address_offset = 0},
        {.channel = 5, .eth_endpoint = {1, 1}, .worker_endpoint = {2, 2}, .address_offset = 1073741824}
    },
    .dram_view_size = 1073741824
};

// BLACKHOLE data
const DramMetadata BLACKHOLE_DRAM_METADATA = {
    .dram_views = {
        {.channel = 0, .eth_endpoint = {2, 1}, .worker_endpoint = {2, 1}, .address_offset = 0},
        {.channel = 1, .eth_endpoint = {0, 1}, .worker_endpoint = {0, 1}, .address_offset = 0},
        {.channel = 2, .eth_endpoint = {0, 1}, .worker_endpoint = {0, 1}, .address_offset = 0},
        {.channel = 3, .eth_endpoint = {0, 1}, .worker_endpoint = {0, 1}, .address_offset = 0},
        {.channel = 4, .eth_endpoint = {2, 1}, .worker_endpoint = {2, 1}, .address_offset = 0},
        {.channel = 5, .eth_endpoint = {2, 1}, .worker_endpoint = {2, 1}, .address_offset = 0},
        {.channel = 6, .eth_endpoint = {2, 1}, .worker_endpoint = {2, 1}, .address_offset = 0},
        {.channel = 7, .eth_endpoint = {2, 1}, .worker_endpoint = {2, 1}, .address_offset = 0}
    },
    .dram_view_size = 4278190080
};

// QUASAR data
const DramMetadata QUASAR_DRAM_METADATA = {
    .dram_views = {
        {.channel = 0, .eth_endpoint = {0, 0}, .worker_endpoint = {0, 0}, .address_offset = 0}
    },
    .dram_view_size = 1073741824
};

}  // namespace metal_soc_descriptor_data
}  // namespace tt
