// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt_stl/span.hpp>

#include "tt_metal/impl/dispatch/device_command.hpp"
#include "tt_metal/impl/dispatch/kernels/cq_commands.hpp"

namespace tt::tt_metal {

class CircularBufferImpl;
class DeviceCommandCalculator;
struct ProgramCommandSequence;
struct RuntimeArgsData;

namespace experimental::dfb::detail {
struct DataflowBufferImpl;
}

namespace program_dispatch {

enum DispatchWriteOffsets {
    DISPATCH_WRITE_OFFSET_ZERO = 0,
    DISPATCH_WRITE_OFFSET_TENSIX_L1_CONFIG_BASE = 1,
    DISPATCH_WRITE_OFFSET_TENSIX_BINARY_L1_CONFIG_BASE = 2,
    DISPATCH_WRITE_OFFSET_ETH_L1_CONFIG_BASE = 3,
};

struct Transfer {
    uint32_t start;
    ttsl::Span<const uint8_t> data;
    // Retain contributors so later program updates can locate their serialized payloads.
    std::vector<std::shared_ptr<CircularBufferImpl>> cbs;
    std::vector<std::shared_ptr<experimental::dfb::detail::DataflowBufferImpl>> dfbs;
    // Runtime arguments are retargeted to their serialized storage after assembly.
    RuntimeArgsData* rta_data = nullptr;
    // Cross-node configuration pages are refreshed before each cached enqueue.
    std::optional<std::pair<CoreCoord, uint8_t>> cross_node_config;

    std::size_t end() const { return start + data.size(); }
};

struct NocTransferKeyHash {
    std::size_t operator()(const std::pair<uint32_t, uint32_t>& noc_transfer_key) const {
        return std::hash<uint32_t>()(noc_transfer_key.first) ^ std::hash<uint32_t>()(noc_transfer_key.second);
    }
};

// Each map entry contains the address-ordered transfers sent to one NOC destination set.
using BatchedTransfers = std::unordered_map<
    std::pair</*noc_xy_addr*/ uint32_t, /*num_mcast_dests*/ uint32_t>,
    std::map</*start_addr*/ uint32_t, std::vector<Transfer>>,
    NocTransferKeyHash>;

struct ProgramConfigCommandOptions {
    uint32_t pcie_alignment;
    uint32_t l1_alignment;
    uint32_t max_prefetch_command_size;
    bool watcher_assert_enabled;
};

class BatchedTransferGenerator {
public:
    explicit BatchedTransferGenerator(ProgramConfigCommandOptions options);

    void construct_commands(BatchedTransfers& batched_transfers, DeviceCommandCalculator& calculator);

    void assemble_commands(
        ProgramCommandSequence& program_command_sequence,
        std::vector<HostMemDeviceCommand>& device_command_sequences,
        DispatchWriteOffsets write_offset = DISPATCH_WRITE_OFFSET_TENSIX_L1_CONFIG_BASE);

    uint32_t command_count() const;
    uint32_t command_size_bytes(uint32_t command_index) const;

private:
    const ProgramConfigCommandOptions options;
    std::vector<std::vector<Transfer>> batched_cmd_data;
    std::vector<std::vector<CQDispatchWritePackedLargeSubCmd>> batched_dispatch_subcmds;
};

}  // namespace program_dispatch
}  // namespace tt::tt_metal
