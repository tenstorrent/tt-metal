// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "device_fixture.hpp"
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "dm_common.hpp"
#include "test_one_to_all.hpp"

namespace tt::tt_metal {

using namespace std;
using namespace tt;
using namespace tt::test_utils;

namespace unit_tests::dm::core_to_all::multicast_schemes {

uint32_t determine_max_grid_dimension(std::vector<IDevice*>& devices_) {
    uint32_t smaller_dimension = std::min(
        devices_.at(0)->compute_with_storage_grid_size().x, devices_.at(0)->compute_with_storage_grid_size().y);
    return (smaller_dimension - 1);
}

enum class MulticastSchemeType {
    // Sender is IN the grid
    SenderInGridTopRight = 1,
    SenderInGridBottomRight,
    SenderInGridBottomLeft,
    SenderInGridTopLeft,
    // Sender is NOT in the grid
    SenderInGridStartingRowNotColumn,
    SenderInGridStartingColumnNotRow,
    SenderNotInGridStartingColumnOrRow,
    SenderInGridEndingRowNotColumn,
    SenderInGridEndingColumnNotRow,
    SenderNotInGridEndingColumnOrRow,
    End
};

// Helper function to determine coordinates based on multicast scheme type
std::pair<CoreCoord, CoreCoord> get_coordinates(uint32_t sub_grid_dimension_size, MulticastSchemeType type) {
    CoreCoord mst_core_coord;
    CoreCoord sub_start_core_coord;

    switch (type) {
        // Sender is IN the grid
        case MulticastSchemeType::SenderInGridTopRight:
            mst_core_coord = {sub_grid_dimension_size - 1, sub_grid_dimension_size - 1};
            sub_start_core_coord = {0, 0};
            break;
        case MulticastSchemeType::SenderInGridBottomRight:
            mst_core_coord = {sub_grid_dimension_size - 1, 0};
            sub_start_core_coord = {0, 0};
            break;
        case MulticastSchemeType::SenderInGridBottomLeft:
            mst_core_coord = {0, 0};
            sub_start_core_coord = {0, 0};
            break;
        case MulticastSchemeType::SenderInGridTopLeft:
            mst_core_coord = {0, sub_grid_dimension_size - 1};
            sub_start_core_coord = {0, 0};
            break;

        // Sender is NOT in the grid
        case MulticastSchemeType::SenderInGridStartingRowNotColumn:
            mst_core_coord = {0, 1};
            sub_start_core_coord = {1, 1};
            break;
        case MulticastSchemeType::SenderInGridStartingColumnNotRow:
            mst_core_coord = {1, 0};
            sub_start_core_coord = {1, 1};
            break;
        case MulticastSchemeType::SenderNotInGridStartingColumnOrRow:
            mst_core_coord = {0, 0};
            sub_start_core_coord = {1, 1};
            break;
        case MulticastSchemeType::SenderInGridEndingRowNotColumn:
            mst_core_coord = {sub_grid_dimension_size, sub_grid_dimension_size - 1};
            sub_start_core_coord = {0, 0};
            break;
        case MulticastSchemeType::SenderInGridEndingColumnNotRow:
            mst_core_coord = {sub_grid_dimension_size - 1, sub_grid_dimension_size};
            sub_start_core_coord = {0, 0};
            break;
        case MulticastSchemeType::SenderNotInGridEndingColumnOrRow:
            mst_core_coord = {sub_grid_dimension_size, sub_grid_dimension_size};
            sub_start_core_coord = {0, 0};
            break;

        default: throw std::invalid_argument("Invalid multicast scheme type");
    }

    return {mst_core_coord, sub_start_core_coord};
}

void test(
    tt::ARCH arch_,
    std::vector<IDevice*>& devices_,
    uint32_t num_devices_,
    uint32_t test_case_id,
    uint32_t sub_grid_dimension_size,
    NOC noc_id,
    MulticastSchemeType multicast_scheme_type,
    bool loopback = true,
    bool is_linked = true) {
    bool is_multicast = true;

    CoreCoord sub_grid_size = {sub_grid_dimension_size, sub_grid_dimension_size};

    auto [mst_core_coord, sub_start_core_coord] = get_coordinates(sub_grid_dimension_size, multicast_scheme_type);

    // Run the directed ideal test
    tt::tt_metal::unit_tests::dm::core_to_all::directed_ideal_test(
        arch_,
        devices_,
        num_devices_,
        test_case_id,
        is_multicast,
        is_linked,
        mst_core_coord,
        sub_start_core_coord,
        sub_grid_size,
        loopback,
        noc_id,
        static_cast<uint32_t>(multicast_scheme_type));
}

void run_all_tests(
    tt::ARCH arch_,
    std::vector<IDevice*>& devices_,
    uint32_t num_devices_,
    uint32_t test_case_id,
    bool loopback = true) {
    std::vector<NOC> noc_ids = {NOC::NOC_0, NOC::NOC_1};
    uint32_t starting_sub_grid_dimension_size = 2;  // Minimum size for sub-grid dimension
    uint32_t sub_grid_dimension_limit = determine_max_grid_dimension(devices_);

    MulticastSchemeType starting_multicast_scheme_type = MulticastSchemeType::SenderInGridTopRight;

    for (const auto& noc_id : noc_ids) {
        for (uint32_t sub_grid_dimension_size = starting_sub_grid_dimension_size;
             sub_grid_dimension_size <= sub_grid_dimension_limit;
             sub_grid_dimension_size++) {
            for (uint32_t multicast_scheme_type = static_cast<uint32_t>(starting_multicast_scheme_type);
                 multicast_scheme_type < static_cast<uint32_t>(MulticastSchemeType::End);
                 multicast_scheme_type++) {
                test(
                    arch_,
                    devices_,
                    num_devices_,
                    test_case_id,
                    sub_grid_dimension_size,
                    (noc_id),
                    static_cast<MulticastSchemeType>(multicast_scheme_type),
                    loopback);
            }
        }
    }
}

}  // namespace unit_tests::dm::core_to_all::multicast_schemes

/* ============================================================= */
/* =================== LOOP THROUGH SCHEMES ==================== */
/* ============================================================= */

TEST_F(DeviceFixture, TensixDataMovementOneToAllMulticastSchemesLoopback) {
    uint32_t test_case_id = 100;

    bool loopback = true;

    tt::tt_metal::unit_tests::dm::core_to_all::multicast_schemes::run_all_tests(
        arch_, devices_, num_devices_, test_case_id, loopback);
}

TEST_F(DeviceFixture, TensixDataMovementOneToAllMulticastSchemesNoLoopback) {
    uint32_t test_case_id = 101;

    bool loopback = false;

    tt::tt_metal::unit_tests::dm::core_to_all::multicast_schemes::run_all_tests(
        arch_, devices_, num_devices_, test_case_id, loopback);
}

/* ============================================================= */
/* ================== INDIVIDUAL TEST CASES ==================== */
/* ============================================================= */

/*
    Loopback: Loopback, No Loopback
    NOC: NOC 0, NOC 1
    Grid Size: 2, ..., 7 (WH), ... 9 (BH)
    Schemes:
        1. Sender in grid top right
        2. Sender in grid bottom right
        3. Sender in grid bottom left
        4. Sender in grid top left
        5. Sender out grid starting row not column
        6. Sender out grid starting column not row
        7. Sender out grid starting not row not column
        8. Sender out grid ending row not column
        9. Sender out grid ending column not row
        10. Sender out grid ending not row not column
*/

TEST_F(DeviceFixture, TensixDataMovementOneToAllMulticastSchemeSingle) {
    uint32_t test_case_id = 110;

    bool loopback = false;
    NOC noc_id = NOC::NOC_0;
    uint32_t sub_grid_dimension_size = arch_ == ARCH::WORMHOLE_B0 ? 7 : 9;  // Adjust based on architecture
    unit_tests::dm::core_to_all::multicast_schemes::MulticastSchemeType multicast_scheme =
        unit_tests::dm::core_to_all::multicast_schemes::MulticastSchemeType::SenderInGridTopRight;

    tt::tt_metal::unit_tests::dm::core_to_all::multicast_schemes::test(
        arch_, devices_, num_devices_, test_case_id, sub_grid_dimension_size, noc_id, multicast_scheme, loopback);
}

}  // namespace tt::tt_metal
