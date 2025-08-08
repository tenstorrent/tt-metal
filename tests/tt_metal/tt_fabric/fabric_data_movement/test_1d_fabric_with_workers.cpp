
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <boost/container/vector.hpp>
#include <boost/move/utility_core.hpp>
#include <fmt/base.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <stdint.h>
#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <unordered_map>
#include <vector>

#include <tt-metalium/assert.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/fabric_edm_packet_header.hpp>
#include <tt-metalium/fabric_edm_types.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_device_view.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/shape.hpp>
#include <tt-metalium/shape_base.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/sub_device_types.hpp>
#include "tests/tt_metal/tt_fabric/common/test_fabric_edm_common.hpp"
#include <tt-metalium/tile.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "tt_metal/test_utils/env_vars.hpp"
#include "umd/device/types/arch.h"

////////////////////////////////////////////////////////////////////
///  MESSAGE COUNT TERMINATION MODE
////////////////////////////////////////////////////////////////////

// -------------------------
// Persistent Fabric
// -------------------------

// TEST(WorkerFabricEdmDatapath, FabricEDMLoopback_With_Workers_SingleMessage_PersistentFabric) {
//     const uint32_t page_size = 2048;
//     const uint32_t num_pages_total = 1;
//     const bool src_is_dram = true;
//     const bool dest_is_dram = true;

//     auto result = TestLoopbackEntrypoint(page_size, num_pages_total, src_is_dram, dest_is_dram);
//     ASSERT_EQ(result, 0);
// }

// // Will wrapp sender but not receiver buffers
// TEST(WorkerFabricEdmDatapath, FabricEDMLoopback_With_Workers_2_messages_PersistentFabric) {
//     const uint32_t page_size = 2048;
//     const uint32_t num_pages_total = 2;
//     const bool src_is_dram = true;
//     const bool dest_is_dram = true;

//     auto result = TestLoopbackEntrypoint(page_size, num_pages_total, src_is_dram, dest_is_dram);
//     ASSERT_EQ(result, 0);
// }
// // Will wrapp sender but not receiver buffers
// TEST(WorkerFabricEdmDatapath, FabricEDMLoopback_With_Workers_10_messages_PersistentFabric) {
//     const uint32_t page_size = 2048;
//     const uint32_t num_pages_total = 10;
//     const bool src_is_dram = true;
//     const bool dest_is_dram = true;

//     auto result = TestLoopbackEntrypoint(page_size, num_pages_total, src_is_dram, dest_is_dram);
//     ASSERT_EQ(result, 0);
// }

// // Will wrapp sender and receiver buffers
// TEST(WorkerFabricEdmDatapath, FabricEDMLoopback_With_Workers_20_messages_PersistentFabric) {
//     const uint32_t page_size = 2048;
//     const uint32_t num_pages_total = 20;
//     const bool src_is_dram = true;
//     const bool dest_is_dram = true;

//     auto result = TestLoopbackEntrypoint(page_size, num_pages_total, src_is_dram, dest_is_dram);
//     ASSERT_EQ(result, 0);
// }

// TEST(WorkerFabricEdmDatapath, FabricEDMLoopback_With_Workers_PersistentFabric) {
//     const uint32_t page_size = 2048;
//     const uint32_t num_pages_total = 10000;
//     const bool src_is_dram = true;
//     const bool dest_is_dram = true;

//     auto result = TestLoopbackEntrypoint(page_size, num_pages_total, src_is_dram, dest_is_dram);
//     ASSERT_EQ(result, 0);
// }

// // Will wrapp sender but not receiver buffers
// TEST(WorkerFabricEdmDatapath, FabricEDMLoopback_With_Workers_2_messages_PersistentFabric_Scatter) {
//     const uint32_t page_size = 2048;
//     const uint32_t num_pages_total = 2;
//     const bool src_is_dram = true;
//     const bool dest_is_dram = true;

//     auto result = TestLoopbackEntrypoint(page_size, num_pages_total, src_is_dram, dest_is_dram, true);
//     ASSERT_EQ(result, 0);
// }
// TEST(WorkerFabricEdmDatapath, FabricEDMLoopback_With_Workers_3_messages_PersistentFabric_Scatter) {
//     const uint32_t page_size = 2048;
//     const uint32_t num_pages_total = 3;
//     const bool src_is_dram = true;
//     const bool dest_is_dram = true;

//     auto result = TestLoopbackEntrypoint(page_size, num_pages_total, src_is_dram, dest_is_dram, true);
//     ASSERT_EQ(result, 0);
// }
// // Will wrapp sender but not receiver buffers
// TEST(WorkerFabricEdmDatapath, FabricEDMLoopback_With_Workers_10_messages_PersistentFabric_Scatter) {
//     const uint32_t page_size = 2048;
//     const uint32_t num_pages_total = 10;
//     const bool src_is_dram = true;
//     const bool dest_is_dram = true;

//     auto result = TestLoopbackEntrypoint(page_size, num_pages_total, src_is_dram, dest_is_dram, true);
//     ASSERT_EQ(result, 0);
// }
// // Will wrapp sender and receiver buffers
// TEST(WorkerFabricEdmDatapath, FabricEDMLoopback_With_Workers_20_messages_PersistentFabric_Scatter) {
//     const uint32_t page_size = 2048;
//     const uint32_t num_pages_total = 20;
//     const bool src_is_dram = true;
//     const bool dest_is_dram = true;

//     auto result = TestLoopbackEntrypoint(page_size, num_pages_total, src_is_dram, dest_is_dram, true);
//     ASSERT_EQ(result, 0);
// }

// ////////////////////////////////

// TEST(WorkerFabricEdmDatapath, LineFabricMcast_SingleMessage_SingleSource_PersistentFabric) {
//     const uint32_t page_size = 2048;
//     const uint32_t num_pages_total = 1;
//     const bool src_is_dram = true;
//     const bool dest_is_dram = true;
//     const size_t mcast_first_chip = 1;
//     const size_t mcast_last_chip = 3;

//     auto result = TestLineFabricEntrypoint(
//         mcast_first_chip, mcast_last_chip, page_size, num_pages_total, src_is_dram, dest_is_dram);

//     ASSERT_EQ(result, 0);
// }

// // Non-functional on harvested parts. Needs testing on unharvested parts.
// TEST(WorkerFabricEdmDatapath, LineFabricMcast_ManyMessages_SingleSource_PersistentFabric) {
//     const uint32_t page_size = 2048;
//     const uint32_t num_pages_total = 10000;
//     const bool src_is_dram = true;
//     const bool dest_is_dram = true;
//     const size_t mcast_first_chip = 1;
//     const size_t mcast_last_chip = 3;

//     auto result = TestLineFabricEntrypoint(
//         mcast_first_chip, mcast_last_chip, page_size, num_pages_total, src_is_dram, dest_is_dram);

//     ASSERT_EQ(result, 0);
// }

