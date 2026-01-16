// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "multi_device_fixture.hpp"
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "dm_common.hpp"
#include "test_one_to_all.hpp"
#include <distributed/mesh_device_impl.hpp>

namespace tt::tt_metal {

using namespace std;
using namespace tt;
using namespace test_utils;

namespace unit_tests::dm::core_to_all::multicast_schemes {

uint32_t determine_max_grid_dimension(const shared_ptr<distributed::MeshDevice>& mesh_device) {
    uint32_t smaller_dimension =
        min(mesh_device->impl().get_device(0)->compute_with_storage_grid_size().x,
            mesh_device->impl().get_device(0)->compute_with_storage_grid_size().y);
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
pair<CoreCoord, CoreCoord> get_coordinates(uint32_t sub_grid_dimension_size, MulticastSchemeType type) {
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

        default: throw invalid_argument("Invalid multicast scheme type");
    }

    return {mst_core_coord, sub_start_core_coord};
}

void test(
    const shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t test_case_id,
    uint32_t sub_grid_dimension_size,
    NOC noc_id,
    MulticastSchemeType multicast_scheme_type,
    bool loopback = true,
    bool is_linked = true,
    bool use_2_0_api = false) {
    bool is_multicast = true;

    CoreCoord sub_grid_size = {sub_grid_dimension_size, sub_grid_dimension_size};

    auto [mst_core_coord, sub_start_core_coord] = get_coordinates(sub_grid_dimension_size, multicast_scheme_type);

    // Run the directed ideal test
    unit_tests::dm::core_to_all::directed_ideal_test(
        mesh_device,
        test_case_id,
        is_multicast,
        is_linked,
        mst_core_coord,
        sub_start_core_coord,
        sub_grid_size,
        loopback,
        noc_id,
        static_cast<uint32_t>(multicast_scheme_type),
        use_2_0_api);
}

void run_all_tests(
    const shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t test_case_id,
    bool loopback = true,
    bool use_2_0_api = false) {
    vector<NOC> noc_ids = {NOC::NOC_0, NOC::NOC_1};
    uint32_t starting_sub_grid_dimension_size = 2;  // Minimum size for sub-grid dimension
    uint32_t sub_grid_dimension_limit = determine_max_grid_dimension(mesh_device);

    MulticastSchemeType starting_multicast_scheme_type = MulticastSchemeType::SenderInGridTopRight;

    for (const auto& noc_id : noc_ids) {
        for (uint32_t sub_grid_dimension_size = starting_sub_grid_dimension_size;
             sub_grid_dimension_size <= sub_grid_dimension_limit;
             sub_grid_dimension_size++) {
            for (uint32_t multicast_scheme_type = static_cast<uint32_t>(starting_multicast_scheme_type);
                 multicast_scheme_type < static_cast<uint32_t>(MulticastSchemeType::End);
                 multicast_scheme_type++) {
                test(
                    mesh_device,
                    test_case_id,
                    sub_grid_dimension_size,
                    (noc_id),
                    static_cast<MulticastSchemeType>(multicast_scheme_type),
                    loopback,
                    true,
                    use_2_0_api);
            }
        }
    }
}

}  // namespace unit_tests::dm::core_to_all::multicast_schemes

/* ============================================================= */
/* =================== LOOP THROUGH SCHEMES ==================== */
/* ============================================================= */

TEST_F(GenericMeshDeviceFixture, TensixDataMovementOneToAllMulticastSchemesLoopback) {
    GTEST_SKIP() << "Skipping test";

    uint32_t test_case_id = 100;
    bool loopback = true;

    unit_tests::dm::core_to_all::multicast_schemes::run_all_tests(get_mesh_device(), test_case_id, loopback);
}

TEST_F(GenericMeshDeviceFixture, TensixDataMovementOneToAllMulticastSchemesNoLoopback) {
    GTEST_SKIP() << "Skipping test";

    uint32_t test_case_id = 101;
    bool loopback = false;

    unit_tests::dm::core_to_all::multicast_schemes::run_all_tests(get_mesh_device(), test_case_id, loopback);
}

TEST_F(GenericMeshDeviceFixture, TensixDataMovementOneToAllMulticastSchemesNoLoopback2_0) {
    uint32_t test_case_id = unit_tests::dm::core_to_all::START_ID_2_0 + 10;
    bool loopback = false;
    bool use_2_0_api = true;

    unit_tests::dm::core_to_all::multicast_schemes::run_all_tests(
        get_mesh_device(), test_case_id, loopback, use_2_0_api);
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

TEST_F(GenericMeshDeviceFixture, TensixDataMovementOneToAllMulticastSchemeSingle) {
    GTEST_SKIP() << "Skipping test";

    uint32_t test_case_id = 102;

    auto mesh_device = get_mesh_device();

    bool loopback = false;
    NOC noc_id = NOC::NOC_0;
    uint32_t sub_grid_dimension_size =
        mesh_device->impl().get_device(0)->arch() == ARCH::WORMHOLE_B0 ? 7 : 9;  // Adjust based on architecture
    unit_tests::dm::core_to_all::multicast_schemes::MulticastSchemeType multicast_scheme =
        unit_tests::dm::core_to_all::multicast_schemes::MulticastSchemeType::SenderInGridTopRight;

    unit_tests::dm::core_to_all::multicast_schemes::test(
        mesh_device, test_case_id, sub_grid_dimension_size, noc_id, multicast_scheme, loopback);
}

}  // namespace tt::tt_metal
