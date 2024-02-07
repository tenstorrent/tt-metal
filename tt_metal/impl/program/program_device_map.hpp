// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <unordered_map>
#include <cstdint>


struct transfer_info {
    std::uint32_t size_in_bytes;
    std::uint32_t dst;
    std::uint32_t dst_noc_encoding;
    std::uint32_t num_receivers;
    bool last_transfer_in_group;
    bool linked;
};

enum class PageTransferType { MULTICAST, UNICAST };

struct ProgramDeviceMap {
    std::uint32_t num_workers;
    vector<std::uint32_t> program_pages;
    std::unordered_map<PageTransferType, vector<transfer_info>> program_page_transfers;
    std::unordered_map<PageTransferType, vector<transfer_info>> runtime_arg_page_transfers;
    std::unordered_map<PageTransferType, vector<transfer_info>> cb_config_page_transfers;
    std::unordered_map<PageTransferType, vector<transfer_info>> go_signal_page_transfers;
    std::unordered_map<PageTransferType, vector<std::uint32_t>> num_transfers_in_program_pages;
    std::unordered_map<PageTransferType, vector<std::uint32_t>> num_transfers_in_runtime_arg_pages;
    std::unordered_map<PageTransferType, vector<std::uint32_t>> num_transfers_in_cb_config_pages;
    std::unordered_map<PageTransferType, vector<std::uint32_t>> num_transfers_in_go_signal_pages;
};
