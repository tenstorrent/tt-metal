// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <unordered_map>
#include <variant>
#include <vector>

#include "tt_metal/common/core_coord.hpp"
#include "tt_metal/common/tt_backend_api_types.hpp"

namespace tt::tt_metal {

using transfer_info_cores = std::variant<CoreCoord, CoreRange>;

struct transfer_info {
    std::uint32_t dst_base_addr;
    std::vector<std::pair<transfer_info_cores, std::uint32_t>> dst_noc_info;  // noc_encoding, num_mcast_dests
    bool linked;
    std::vector<std::uint32_t> data;
};

struct kernel_bins_transfer_info {
    std::vector<std::uint32_t> dst_base_addrs;  // BRISC, NCRISC, TRISC etc..
    std::vector<std::uint32_t> page_offsets;    // offsets into paged buffer in DRAM
    std::vector<std::uint32_t> lengths;         // WriteLinear lengths
    std::vector<tt::RISCV> riscvs;              // RISC that each span is targeted for, for binaries
};

struct ProgramTransferInfo {
    std::uint32_t num_active_cores;
    std::unordered_map<std::uint32_t, std::vector<transfer_info>> multicast_semaphores;  // WritePacked, sorted by dst
    std::unordered_map<std::uint32_t, std::vector<transfer_info>> unicast_semaphores;    // WritePacked, sorted by dst
    std::vector<std::tuple<transfer_info_cores, std::uint32_t, kernel_bins_transfer_info>>
        kernel_bins;                         // noc_encoding, num_mcast_dests, transfer_info
    std::vector<std::uint32_t> binary_data;  // Holds binary data for all program kernels
};

}  // namespace tt::tt_metal
