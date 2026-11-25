// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include <tt-metalium/vocabulary/core_coord.hpp>
#include <tt-metalium/vocabulary/mesh_coord.hpp>

namespace tt::tt_metal::distributed {
class MeshDevice;
}  // namespace tt::tt_metal::distributed

// Ethernet-core queries for host code that builds Ethernet kernels through ProgramDescriptor
// (EthernetConfigDescriptor), e.g. from Python.
namespace tt::tt_metal::experimental::ethernet {

struct EthernetCore {
    CoreCoord logical;       // logical Ethernet coordinate, as KernelDescriptor core_ranges take it
    CoreCoord virtual_core;  // coordinate a NOC transaction addresses
    bool active = false;     // linked core: kernels on it need slow dispatch and an EthernetKernelMode SENDER/RECEIVER
    int channel = -1;        // Ethernet channel
};

// Every Ethernet core (active and idle) of the chip at device_coord, sorted by channel.
std::vector<EthernetCore> get_ethernet_cores(
    const distributed::MeshDevice& mesh_device, const distributed::MeshCoordinate& device_coord);

// [start, end) of the unreserved L1 that both active and idle Ethernet cores leave to kernels.
struct EthernetL1Range {
    uint32_t start = 0;
    uint32_t end = 0;
};
EthernetL1Range get_common_unreserved_l1_range();

// What an Ethernet kernel may use: with non-cooperative base firmware and several active-Ethernet RISCs, the NOC must
// equal the processor index; with one, the kernel must use NOC1 (base firmware owns NOC0).
struct EthernetRiscInfo {
    bool active_fw_cooperative = false;
    uint32_t active_eth_riscs = 0;
    uint32_t idle_eth_riscs = 0;
};
EthernetRiscInfo get_risc_info();

// Read size_bytes of L1 at address from a core of the chip at device_coord (Ethernet or Tensix, by logical coordinate).
std::vector<uint32_t> read_l1(
    const distributed::MeshDevice& mesh_device,
    const distributed::MeshCoordinate& device_coord,
    const CoreCoord& logical_core,
    bool is_ethernet,
    uint32_t address,
    uint32_t size_bytes);

// Write words to L1 at address on a core of the chip at device_coord (Ethernet or Tensix, by logical coordinate).
void write_l1(
    const distributed::MeshDevice& mesh_device,
    const distributed::MeshCoordinate& device_coord,
    const CoreCoord& logical_core,
    bool is_ethernet,
    uint32_t address,
    std::vector<uint32_t> words);

}  // namespace tt::tt_metal::experimental::ethernet
