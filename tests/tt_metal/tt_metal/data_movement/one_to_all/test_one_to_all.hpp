// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TEST_ONE_TO_ALL_HPP
#define TEST_ONE_TO_ALL_HPP

namespace tt::tt_metal::unit_tests::dm::core_to_all {

constexpr uint32_t START_ID = 6;
constexpr uint32_t START_ID_2_0 = 170;

void directed_ideal_test(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t test_case_id,
    bool is_multicast,
    bool is_linked,
    CoreCoord mst_core_coord,
    CoreCoord sub_start_core_coord,
    CoreCoord sub_grid_size,
    bool loopback = true,
    NOC noc_id = NOC::NOC_0,
    uint32_t multicast_scheme_type = 0,
    bool use_2_0_api = false);
}

#endif  // TEST_ONE_TO_ALL_HPP
