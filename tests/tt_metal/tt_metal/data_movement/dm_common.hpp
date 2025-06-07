// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef DM_COMMON_HPP
#define DM_COMMON_HPP

#include <cstdint>
#include "device_fixture.hpp"
#include <tuple>
namespace tt::tt_metal::unit_tests::dm {

// Unique id for each test run
extern uint32_t runtime_host_id;

// Struct to hold L1 address information
struct L1AddressInfo {
    uint64_t base_address;
    uint64_t size;
};

// Function to get L1 address and size
L1AddressInfo get_l1_address_and_size(const IDevice* device, const CoreCoord& core_coord = {0, 0});

// Function to compute physical constraints
std::tuple<uint32_t, uint32_t, uint32_t> compute_physical_constraints(const tt::ARCH arch, const IDevice* device);

}  // namespace tt::tt_metal::unit_tests::dm

#endif  // DM_COMMON_HPP
