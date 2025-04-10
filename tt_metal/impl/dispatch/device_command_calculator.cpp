// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device_command_calculator.hpp"

#include <algorithm>

#include "dispatch/kernels/cq_commands.hpp"
#include "hal.hpp"

namespace tt::tt_metal {

template <typename PackedSubCmd>
uint32_t DeviceCommandCalculator::get_max_write_packed_sub_cmds(
    uint32_t data_size,
    uint32_t max_prefetch_cmd_size,
    uint32_t packed_write_max_unicast_sub_cmds,
    bool no_stride) const {
    static_assert(
        std::is_same<PackedSubCmd, CQDispatchWritePackedUnicastSubCmd>::value or
        std::is_same<PackedSubCmd, CQDispatchWritePackedMulticastSubCmd>::value);
    constexpr bool is_unicast = std::is_same<PackedSubCmd, CQDispatchWritePackedUnicastSubCmd>::value;
    uint32_t sub_cmd_sizeB =
        is_unicast ? sizeof(CQDispatchWritePackedUnicastSubCmd) : sizeof(CQDispatchWritePackedMulticastSubCmd);
    // Approximate calculation due to alignment
    uint32_t max_prefetch_size = max_prefetch_cmd_size - sizeof(CQPrefetchCmd) - this->pcie_alignment -
                                 sizeof(CQDispatchCmd) - this->l1_alignment;
    uint32_t max_prefetch_num_packed_cmds =
        no_stride ? (max_prefetch_size - tt::align(data_size * sizeof(uint32_t), l1_alignment)) / sub_cmd_sizeB
                  : max_prefetch_size / (tt::align(data_size * sizeof(uint32_t), l1_alignment) + sub_cmd_sizeB);

    uint32_t packed_write_max_multicast_sub_cmds =
        get_packed_write_max_multicast_sub_cmds(packed_write_max_unicast_sub_cmds);
    return std::min(
        max_prefetch_num_packed_cmds,
        is_unicast ? packed_write_max_unicast_sub_cmds : packed_write_max_multicast_sub_cmds);
};

// Explicit template instantiations
template uint32_t DeviceCommandCalculator::get_max_write_packed_sub_cmds<CQDispatchWritePackedMulticastSubCmd>(
    uint32_t, uint32_t, uint32_t, bool) const;
template uint32_t DeviceCommandCalculator::get_max_write_packed_sub_cmds<CQDispatchWritePackedUnicastSubCmd>(
    uint32_t, uint32_t, uint32_t, bool) const;

template <typename PackedSubCmd>
void DeviceCommandCalculator::insert_write_packed_payloads(
    const uint32_t num_sub_cmds,
    const uint32_t sub_cmd_sizeB,
    const uint32_t max_prefetch_command_size,
    const uint32_t packed_write_max_unicast_sub_cmds,
    std::vector<std::pair<uint32_t, uint32_t>>& packed_cmd_payloads) {
    uint32_t l1_alignment = MetalContext::instance().hal().get_alignment(HalMemType::L1);
    const uint32_t aligned_sub_cmd_sizeB = tt::align(sub_cmd_sizeB, l1_alignment);
    const uint32_t max_packed_sub_cmds_per_cmd = get_max_write_packed_sub_cmds<PackedSubCmd>(
        aligned_sub_cmd_sizeB, max_prefetch_command_size, packed_write_max_unicast_sub_cmds, false);
    uint32_t rem_num_sub_cmds = num_sub_cmds;
    while (rem_num_sub_cmds != 0) {
        const uint32_t num_sub_cmds_in_cmd = std::min(max_packed_sub_cmds_per_cmd, rem_num_sub_cmds);
        const uint32_t aligned_data_sizeB = aligned_sub_cmd_sizeB * num_sub_cmds_in_cmd;
        const uint32_t dispatch_cmd_sizeB =
            tt::align(sizeof(CQDispatchCmd) + num_sub_cmds_in_cmd * sizeof(PackedSubCmd), l1_alignment);
        packed_cmd_payloads.emplace_back(num_sub_cmds_in_cmd, dispatch_cmd_sizeB + aligned_data_sizeB);
        rem_num_sub_cmds -= num_sub_cmds_in_cmd;
        this->add_dispatch_write_packed<PackedSubCmd>(
            num_sub_cmds_in_cmd, sub_cmd_sizeB, packed_write_max_unicast_sub_cmds);
    }
}

// Explicit template instantiations
template void DeviceCommandCalculator::insert_write_packed_payloads<CQDispatchWritePackedMulticastSubCmd>(
    uint32_t, uint32_t, uint32_t, uint32_t, std::vector<std::pair<uint32_t, uint32_t>>&);

template void DeviceCommandCalculator::insert_write_packed_payloads<CQDispatchWritePackedUnicastSubCmd>(
    uint32_t, uint32_t, uint32_t, uint32_t, std::vector<std::pair<uint32_t, uint32_t>>&);

}  // namespace tt::tt_metal
