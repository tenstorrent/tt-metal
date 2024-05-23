// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <unordered_map>
#include <cstdint>

// TODO: AL delete?
//
struct transfer_info {
    std::uint32_t size_in_bytes;
    std::uint32_t dst;
    std::uint32_t dst_noc_encoding;
    std::uint32_t num_receivers;
    bool last_transfer_in_group;
    bool linked;
};

struct transfer_info_2 {
    std::uint32_t dst_base_addr;
    vector<pair<uint32_t, uint32_t>> dst_noc_info;  // noc_encoding, num_mcast_dests
    bool linked;
    vector<std::uint32_t> data;
};
struct kernel_bins_transfer_info {
    vector<std::uint32_t> dst_base_addrs;           // BRISC, NCRISC, TRISC etc..
    vector<std::uint32_t> page_offsets;             // offsets into paged buffer in DRAM
    vector<std::uint32_t> lengths;                  // WriteLinear lengths
    vector<pair<uint32_t, uint32_t>> dst_noc_info;  // noc_encoding, num_mcast_dests
    bool linked;
    vector<std::uint32_t> data;                     // all binaries' data for kernel group
};

enum class PageTransferType { MULTICAST, UNICAST };

struct ProgramTransferInfo {
    std::uint32_t num_active_cores;
    std::unordered_map<uint32_t, vector<transfer_info_2>> multicast_semaphores;    // WritePacked, sorted by dst
    std::unordered_map<uint32_t, vector<transfer_info_2>> unicast_semaphores;      // WritePacked, sorted by dst
    vector<kernel_bins_transfer_info> kernel_bins;                                 // RelayPaged, WriteLinear
};
