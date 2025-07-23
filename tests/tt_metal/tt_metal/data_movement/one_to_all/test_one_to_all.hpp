// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TEST_ONE_TO_ALL_HPP
#define TEST_ONE_TO_ALL_HPP

namespace tt::tt_metal::unit_tests::dm::core_to_all {

void directed_ideal_test(
    tt::ARCH arch_,
    std::vector<IDevice*>& devices_,
    uint32_t num_devices_,
    uint32_t test_case_id,
    bool is_multicast,
    bool is_linked,
    CoreCoord mst_core_coord,
    CoreCoord sub_start_core_coord,
    CoreCoord sub_grid_size,
    bool loopback = true,
    NOC noc_id = NOC::NOC_0,
    uint32_t multicast_scheme_type = 0);
}

#endif  // TEST_ONE_TO_ALL_HPP
