
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

TEST(WorkerFabricEdmDatapath, FabricEDMLoopback_With_Workers_SingleMessage_PersistentFabric) {
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 1;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;

    auto result = TestLoopbackEntrypoint(page_size, num_pages_total, src_is_dram, dest_is_dram);
    ASSERT_EQ(result, 0);
}

// Will wrapp sender but not receiver buffers
TEST(WorkerFabricEdmDatapath, FabricEDMLoopback_With_Workers_2_messages_PersistentFabric) {
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 2;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;

    auto result = TestLoopbackEntrypoint(page_size, num_pages_total, src_is_dram, dest_is_dram);
    ASSERT_EQ(result, 0);
}
// Will wrapp sender but not receiver buffers
TEST(WorkerFabricEdmDatapath, FabricEDMLoopback_With_Workers_10_messages_PersistentFabric) {
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 10;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;

    auto result = TestLoopbackEntrypoint(page_size, num_pages_total, src_is_dram, dest_is_dram);
    ASSERT_EQ(result, 0);
}

// Will wrapp sender and receiver buffers
TEST(WorkerFabricEdmDatapath, FabricEDMLoopback_With_Workers_20_messages_PersistentFabric) {
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 20;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;

    auto result = TestLoopbackEntrypoint(page_size, num_pages_total, src_is_dram, dest_is_dram);
    ASSERT_EQ(result, 0);
}

TEST(WorkerFabricEdmDatapath, FabricEDMLoopback_With_Workers_PersistentFabric) {
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 10000;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;

    auto result = TestLoopbackEntrypoint(page_size, num_pages_total, src_is_dram, dest_is_dram);
    ASSERT_EQ(result, 0);
}

// Will wrapp sender but not receiver buffers
TEST(WorkerFabricEdmDatapath, FabricEDMLoopback_With_Workers_2_messages_PersistentFabric_Scatter) {
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 2;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;

    auto result = TestLoopbackEntrypoint(page_size, num_pages_total, src_is_dram, dest_is_dram, true);
    ASSERT_EQ(result, 0);
}
TEST(WorkerFabricEdmDatapath, FabricEDMLoopback_With_Workers_3_messages_PersistentFabric_Scatter) {
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 3;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;

    auto result = TestLoopbackEntrypoint(page_size, num_pages_total, src_is_dram, dest_is_dram, true);
    ASSERT_EQ(result, 0);
}
// Will wrapp sender but not receiver buffers
TEST(WorkerFabricEdmDatapath, FabricEDMLoopback_With_Workers_10_messages_PersistentFabric_Scatter) {
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 10;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;

    auto result = TestLoopbackEntrypoint(page_size, num_pages_total, src_is_dram, dest_is_dram, true);
    ASSERT_EQ(result, 0);
}
// Will wrapp sender and receiver buffers
TEST(WorkerFabricEdmDatapath, FabricEDMLoopback_With_Workers_20_messages_PersistentFabric_Scatter) {
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 20;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;

    auto result = TestLoopbackEntrypoint(page_size, num_pages_total, src_is_dram, dest_is_dram, true);
    ASSERT_EQ(result, 0);
}

////////////////////////////////

TEST(WorkerFabricEdmDatapath, LineFabricMcast_SingleMessage_SingleSource_PersistentFabric) {
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 1;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;
    const size_t mcast_first_chip = 1;
    const size_t mcast_last_chip = 3;

    auto result = TestLineFabricEntrypoint(
        mcast_first_chip, mcast_last_chip, page_size, num_pages_total, src_is_dram, dest_is_dram);

    ASSERT_EQ(result, 0);
}

// Non-functional on harvested parts. Needs testing on unharvested parts.
TEST(WorkerFabricEdmDatapath, LineFabricMcast_ManyMessages_SingleSource_PersistentFabric) {
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 10000;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;
    const size_t mcast_first_chip = 1;
    const size_t mcast_last_chip = 3;

    auto result = TestLineFabricEntrypoint(
        mcast_first_chip, mcast_last_chip, page_size, num_pages_total, src_is_dram, dest_is_dram);

    ASSERT_EQ(result, 0);
}

TEST(EdmFabric, BasicMcastThroughputTest_SingleLink_LineSize2_SingleMcast) {
    const size_t num_mcasts = 1;
    const size_t num_unicasts = 2;
    const size_t num_links = 1;
    const size_t num_op_invocations = 1;
    const bool line_sync = true;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_sync = line_sync;
    params.line_size = 2;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}

TEST(EdmFabric, BasicMcastThroughputTest_SingleMcast) {
    const size_t num_mcasts = 1;
    const size_t num_unicasts = 2;
    const size_t num_links = 2;
    const size_t num_op_invocations = 1;
    const bool line_sync = true;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_sync = line_sync;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}
TEST(EdmFabric, DISABLED_BasicMcastThroughputTest_SenderFullNoWrap_ReceiverNoWrap_SingleWorker_2Device) {
    const size_t num_mcasts = 9;
    const size_t num_unicasts = 0;
    const size_t num_links = 1;
    const size_t num_op_invocations = 1;
    const size_t line_size = 2;
    const bool line_sync = false;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_size = line_size;
    params.line_sync = line_sync;
    params.num_devices_with_workers = 1;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}
TEST(EdmFabric, BasicMcastThroughputTest_SenderFullNoWrap_ReceiverNoWrap_2Device) {
    const size_t num_mcasts = 9;
    const size_t num_unicasts = 0;
    const size_t num_links = 1;
    const size_t num_op_invocations = 1;
    const size_t line_size = 2;
    const bool line_sync = true;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_size = line_size;
    params.line_sync = line_sync;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}
TEST(EdmFabric, DISABLED_BasicMcastThroughputTest_SenderFullNoWrap_ReceiverNoWrap_SingleWorker_4Device) {
    const size_t num_mcasts = 9;
    const size_t num_unicasts = 0;
    const size_t num_links = 1;
    const size_t num_op_invocations = 1;
    const size_t line_size = 4;
    const bool line_sync = false;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_size = line_size;
    params.line_sync = line_sync;
    params.num_devices_with_workers = 1;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}
TEST(EdmFabric, DISABLED_BasicMcastThroughputTest_SenderFullNoWrap_ReceiverNoWrap_TwoWorkers_4Device) {
    const size_t num_mcasts = 9;
    const size_t num_unicasts = 0;
    const size_t num_links = 1;
    const size_t num_op_invocations = 1;
    const size_t line_size = 4;
    const bool line_sync = false;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_size = line_size;
    params.line_sync = line_sync;
    params.num_devices_with_workers = 2;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}
TEST(EdmFabric, BasicMcastThroughputTest_SenderFullNoWrap_ReceiverNoWrap) {
    const size_t num_mcasts = 9;
    const size_t num_unicasts = 0;
    const size_t num_links = 1;
    const size_t num_op_invocations = 1;
    const bool line_sync = true;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_sync = line_sync;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}
TEST(EdmFabric, DISABLED_BasicMcastThroughputTest_SenderOneElemWrap_ReceiverNoWrap_SingleWorker_2Device) {
    const size_t num_mcasts = 10;
    const size_t num_unicasts = 0;
    const size_t num_links = 1;
    const size_t num_op_invocations = 1;
    const size_t line_size = 2;
    const bool line_sync = false;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_size = line_size;
    params.line_sync = line_sync;
    params.num_devices_with_workers = 1;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}
TEST(EdmFabric, BasicMcastThroughputTest_SenderOneElemWrap_ReceiverNoWrap_2Device) {
    const size_t num_mcasts = 10;
    const size_t num_unicasts = 0;
    const size_t num_links = 1;
    const size_t num_op_invocations = 1;
    const size_t line_size = 2;
    const bool line_sync = true;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_size = line_size;
    params.line_sync = line_sync;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}
TEST(EdmFabric, BasicMcastThroughputTest_SenderOneElemWrap_ReceiverNoWrap) {
    const size_t num_mcasts = 10;
    const size_t num_unicasts = 0;
    const size_t num_links = 1;
    const size_t num_op_invocations = 1;
    const bool line_sync = true;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_sync = line_sync;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}
TEST(EdmFabric, BasicMcastThroughputTest_SenderTwiceFilled_ReceiverOnceFilled_2Device) {
    const size_t num_mcasts = 18;
    const size_t num_unicasts = 0;
    const size_t num_links = 1;
    const size_t num_op_invocations = 1;
    const size_t line_size = 2;
    const bool line_sync = true;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_size = line_size;
    params.line_sync = line_sync;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}
TEST(EdmFabric, BasicMcastThroughputTest_SenderTwiceFilled_ReceiverOnceFilled) {
    const size_t num_mcasts = 18;
    const size_t num_unicasts = 0;
    const size_t num_links = 1;
    const size_t num_op_invocations = 1;
    const bool line_sync = true;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_sync = line_sync;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}
TEST(EdmFabric, BasicMcastThroughputTest_SenderTwoWrap_ReceiverOneWrap) {
    const size_t num_mcasts = 19;
    const size_t num_unicasts = 0;
    const size_t num_links = 1;
    const size_t num_op_invocations = 1;
    const bool line_sync = true;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_sync = line_sync;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}

TEST(EdmFabric, BasicMcastThroughputTest_SingleLink_LineSize2_SingleMcast_LineSync) {
    const size_t num_mcasts = 1;
    const size_t num_unicasts = 2;
    const size_t num_links = 1;
    const size_t num_op_invocations = 1;
    const bool line_sync = true;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_sync = line_sync;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}

TEST(EdmFabric, BasicMcastThroughputTest_SingleMcast_LineSync) {
    const size_t num_mcasts = 1;
    const size_t num_unicasts = 2;
    const size_t num_links = 2;
    const size_t num_op_invocations = 1;
    const bool line_sync = true;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_sync = line_sync;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}
TEST(EdmFabric, BasicMcastThroughputTest_SenderFullNoWrap_ReceiverNoWrap_LineSync) {
    const size_t num_mcasts = 9;
    const size_t num_unicasts = 0;
    const size_t num_links = 1;
    const size_t num_op_invocations = 1;
    const bool line_sync = true;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_sync = line_sync;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}
TEST(EdmFabric, BasicMcastThroughputTest_SenderOneElemWrap_ReceiverNoWrap_2Device_LineSync) {
    const size_t num_mcasts = 10;
    const size_t num_unicasts = 0;
    const size_t num_links = 1;
    const size_t num_op_invocations = 1;
    const size_t line_size = 2;
    const bool line_sync = true;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_size = line_size;
    params.line_sync = line_sync;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}
TEST(EdmFabric, BasicMcastThroughputTest_SenderOneElemWrap_ReceiverNoWrap_LineSync) {
    const size_t num_mcasts = 10;
    const size_t num_unicasts = 0;
    const size_t num_links = 1;
    const size_t num_op_invocations = 1;
    const bool line_sync = true;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_sync = line_sync;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}
TEST(EdmFabric, BasicMcastThroughputTest_SenderTwiceFilled_ReceiverOnceFilled_2Device_LineSync) {
    const size_t num_mcasts = 18;
    const size_t num_unicasts = 0;
    const size_t num_links = 1;
    const size_t num_op_invocations = 1;
    const size_t line_size = 2;
    const bool line_sync = true;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_size = line_size;
    params.line_sync = line_sync;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}
TEST(EdmFabric, BasicMcastThroughputTest_SenderTwiceFilled_ReceiverOnceFilled_LineSync) {
    const size_t num_mcasts = 18;
    const size_t num_unicasts = 0;
    const size_t num_links = 1;
    const size_t num_op_invocations = 1;
    const bool line_sync = true;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_sync = line_sync;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}
TEST(EdmFabric, DISABLED_BasicMcastThroughputTest_SenderFourTImesFilled_ReceiverTwiceFilled_2Device_1Worker) {
    const size_t num_mcasts = 36;
    const size_t num_unicasts = 0;
    const size_t num_links = 1;
    const size_t num_op_invocations = 1;
    const size_t line_size = 2;
    const bool line_sync = false;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_size = line_size;
    params.line_sync = line_sync;
    params.num_devices_with_workers = 1;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}
TEST(EdmFabric, BasicMcastThroughputTest_SenderFourTImesFilled_ReceiverTwiceFilled_2Device_LineSync) {
    const size_t num_mcasts = 36;
    const size_t num_unicasts = 0;
    const size_t num_links = 1;
    const size_t num_op_invocations = 1;
    const size_t line_size = 2;
    const bool line_sync = true;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_size = line_size;
    params.line_sync = line_sync;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}
TEST(EdmFabric, BasicMcastThroughputTest_SenderFourTImesFilled_ReceiverTwiceFilled_LineSync) {
    const size_t num_mcasts = 36;
    const size_t num_unicasts = 0;
    const size_t num_links = 1;
    const size_t num_op_invocations = 1;
    const bool line_sync = true;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_sync = line_sync;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}
TEST(EdmFabric, BasicMcastThroughputTest_SenderTwoWrap_ReceiverOneWrap_LineSync) {
    const size_t num_mcasts = 19;
    const size_t num_unicasts = 0;
    const size_t num_links = 1;
    const size_t num_op_invocations = 1;
    const bool line_sync = true;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_sync = line_sync;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}

TEST(EdmFabric, BasicMcastThroughputTest_SmallPerf_2Device) {
    const size_t num_mcasts = 70;
    const size_t num_unicasts = 0;
    const size_t num_links = 2;
    const size_t num_op_invocations = 1;
    const size_t line_size = 2;
    const bool report_performance = true;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_sync = report_performance;
    params.line_size = line_size;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}

TEST(EdmFabric, BasicMcastThroughputTest_SmallPerf0) {
    const size_t num_mcasts = 70;
    const size_t num_unicasts = 0;
    const size_t num_links = 2;
    const size_t num_op_invocations = 1;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_sync = true;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}
TEST(EdmFabric, BasicMcastThroughputTest_SmallPerf1) {
    const size_t num_mcasts = 70;
    const size_t num_unicasts = 0;
    const size_t num_links = 2;
    const size_t num_op_invocations = 1;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_sync = true;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}

TEST(EdmFabric, DISABLED_BasicMcastThroughputTest_0) {
    const size_t num_mcasts = 100;
    const size_t num_unicasts = 2;
    const size_t num_links = 2;
    const size_t num_op_invocations = 1;
    const bool line_sync = false;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_size = 2;
    params.line_sync = line_sync;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}
TEST(EdmFabric, DISABLED_BasicMcastThroughputTest_1) {
    const size_t num_mcasts = 1000;
    const size_t num_unicasts = 2;
    const size_t num_links = 2;
    const size_t num_op_invocations = 1;
    const bool line_sync = false;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_sync = line_sync;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}
TEST(EdmFabric, BasicMcastThroughputTest_2) {
    const size_t num_mcasts = 50000;
    const size_t num_unicasts = 2;
    const size_t num_links = 2;
    const size_t num_op_invocations = 1;

    RunWriteThroughputStabilityTestWithPersistentFabric(num_mcasts, num_unicasts, num_links, num_op_invocations);
}
TEST(EdmFabric, BasicMcastThroughputTest_3_SingleLink) {
    const size_t num_mcasts = 200000;
    const size_t num_unicasts = 0;
    const size_t num_links = 1;
    const size_t num_op_invocations = 1;
    const bool line_sync = true;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_sync = line_sync;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}
TEST(EdmFabric, BasicMcastThroughputTest_3) {
    const size_t num_mcasts = 200000;
    const size_t num_unicasts = 2;
    const size_t num_links = 2;
    const size_t num_op_invocations = 1;
    const bool line_sync = true;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_sync = line_sync;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}
TEST(EdmFabric, BasicMcastThroughputTest_3_onehop) {
    const size_t num_mcasts = 200000;
    const size_t num_unicasts = 2;
    const size_t num_links = 1;
    const size_t num_op_invocations = 1;
    const bool line_sync = true;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_sync = line_sync;
    params.line_size = 2;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}
TEST(EdmFabric, BasicMcastThroughputTest_4) {
    const size_t num_mcasts = 800000;
    const size_t num_unicasts = 2;
    const size_t num_links = 2;
    const size_t num_op_invocations = 1;
    RunWriteThroughputStabilityTestWithPersistentFabric(num_mcasts, num_unicasts, num_links, num_op_invocations);
}

TEST(EdmFabric, BasicMcastThroughputTest_5) {
    const size_t num_mcasts = 1;
    const size_t num_unicasts = 2;
    const size_t num_links = 2;
    const size_t num_op_invocations = 20000;
    RunWriteThroughputStabilityTestWithPersistentFabric(num_mcasts, num_unicasts, num_links, num_op_invocations);
}
// DISABLED due to long runtime
TEST(EdmFabric, DISABLED_BasicMcastThroughputTest_6) {
    const size_t num_mcasts = 100;
    const size_t num_unicasts = 2;
    const size_t num_links = 2;
    const size_t num_op_invocations = 8000;
    RunWriteThroughputStabilityTestWithPersistentFabric(num_mcasts, num_unicasts, num_links, num_op_invocations);
}
// DISABLED due to long runtime
TEST(EdmFabric, DISABLED_BasicMcastThroughputTest_7) {
    const size_t num_mcasts = 1000;
    const size_t num_unicasts = 2;
    const size_t num_links = 2;
    const size_t num_op_invocations = 1000;
    RunWriteThroughputStabilityTestWithPersistentFabric(num_mcasts, num_unicasts, num_links, num_op_invocations);
}
// DISABLED due to long runtime
TEST(EdmFabric, DISABLED_BasicMcastThroughputTest_8) {
    const size_t num_mcasts = 50000;
    const size_t num_unicasts = 2;
    const size_t num_links = 2;
    const size_t num_op_invocations = 200;
    RunWriteThroughputStabilityTestWithPersistentFabric(num_mcasts, num_unicasts, num_links, num_op_invocations);
}
// DISABLED due to long runtime
TEST(EdmFabric, DISABLED_BasicMcastThroughputTest_9) {
    const size_t num_mcasts = 200000;
    const size_t num_unicasts = 2;
    const size_t num_links = 2;
    const size_t num_op_invocations = 150;
    RunWriteThroughputStabilityTestWithPersistentFabric(num_mcasts, num_unicasts, num_links, num_op_invocations);
}
// DISABLED due to long runtime
TEST(EdmFabric, DISABLED_BasicMcastThroughputTest_10) {
    const size_t num_mcasts = 800000;
    const size_t num_unicasts = 2;
    const size_t num_links = 2;
    const size_t num_op_invocations = 50;
    RunWriteThroughputStabilityTestWithPersistentFabric(num_mcasts, num_unicasts, num_links, num_op_invocations);
}
TEST(EdmFabric, BasicMcastThroughputTest_6_Short) {
    const size_t num_mcasts = 100;
    const size_t num_unicasts = 2;
    const size_t num_links = 2;
    const size_t num_op_invocations = 100;
    RunWriteThroughputStabilityTestWithPersistentFabric(num_mcasts, num_unicasts, num_links, num_op_invocations);
}
TEST(EdmFabric, BasicMcastThroughputTest_7_Short) {
    const size_t num_mcasts = 1000;
    const size_t num_unicasts = 2;
    const size_t num_links = 2;
    const size_t num_op_invocations = 50;
    RunWriteThroughputStabilityTestWithPersistentFabric(num_mcasts, num_unicasts, num_links, num_op_invocations);
}
TEST(EdmFabric, BasicMcastThroughputTest_8_Short) {
    const size_t num_mcasts = 50000;
    const size_t num_unicasts = 2;
    const size_t num_links = 2;
    const size_t num_op_invocations = 20;
    RunWriteThroughputStabilityTestWithPersistentFabric(num_mcasts, num_unicasts, num_links, num_op_invocations);
}
TEST(EdmFabric, BasicMcastThroughputTest_9_Short) {
    const size_t num_mcasts = 200000;
    const size_t num_unicasts = 2;
    const size_t num_links = 2;
    const size_t num_op_invocations = 10;
    RunWriteThroughputStabilityTestWithPersistentFabric(num_mcasts, num_unicasts, num_links, num_op_invocations);
}
TEST(EdmFabric, BasicMcastThroughputTest_10_Short) {
    const size_t num_mcasts = 800000;
    const size_t num_unicasts = 2;
    const size_t num_links = 2;
    const size_t num_op_invocations = 5;
    RunWriteThroughputStabilityTestWithPersistentFabric(num_mcasts, num_unicasts, num_links, num_op_invocations);
}

TEST(EdmFabric, BasicMcastThroughputTest_0_WithLineSync) {
    const size_t num_mcasts = 100;
    const size_t num_unicasts = 2;
    const size_t num_links = 2;
    const size_t num_op_invocations = 1;
    const bool line_sync = true;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_sync = line_sync;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}
TEST(EdmFabric, BasicMcastThroughputTest_1_WithLineSync) {
    const size_t num_mcasts = 1000;
    const size_t num_unicasts = 2;
    const size_t num_links = 2;
    const size_t num_op_invocations = 1;
    const bool line_sync = true;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_sync = line_sync;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}
TEST(EdmFabric, BasicMcastThroughputTest_2_WithLineSync) {
    const size_t num_mcasts = 50000;
    const size_t num_unicasts = 2;
    const size_t num_links = 2;
    const size_t num_op_invocations = 1;
    const bool line_sync = true;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_sync = line_sync;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}
TEST(EdmFabric, BasicMcastThroughputTest_3_WithLineSync) {
    const size_t num_mcasts = 200000;
    const size_t num_unicasts = 2;
    const size_t num_links = 2;
    const size_t num_op_invocations = 1;
    const bool line_sync = true;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_sync = line_sync;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}
TEST(EdmFabric, BasicMcastThroughputTest_4_WithLineSync) {
    const size_t num_mcasts = 800000;
    const size_t num_unicasts = 2;
    const size_t num_links = 2;
    const size_t num_op_invocations = 1;
    const bool line_sync = true;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_sync = line_sync;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}

TEST(EdmFabric, RingDeadlockStabilityTest) {
    constexpr size_t num_mcasts = 200000;
    constexpr size_t num_op_invocations = 5;
    constexpr bool line_sync = true;
    size_t num_links = 1;
    std::vector<size_t> num_devices_vec;
    auto cluster_type = tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type();
    if (cluster_type == tt::ClusterType::GALAXY) {
        num_devices_vec = {4, 8};
        num_links = 4;
    } else {
        num_devices_vec = {8};
    }
    for (const auto& num_devices : num_devices_vec) {
        log_trace(
            tt::LogTest, "Running RingDeadlockStabilityTest with forward mcast only with {} devices", num_devices);
        RunRingDeadlockStabilityTestWithPersistentFabric(
            num_mcasts, num_links, num_devices, num_op_invocations, true, false);
        log_trace(
            tt::LogTest, "Running RingDeadlockStabilityTest with backward mcast only with {} devices", num_devices);
        RunRingDeadlockStabilityTestWithPersistentFabric(
            num_mcasts, num_links, num_devices, num_op_invocations, false, true);
        log_trace(
            tt::LogTest,
            "Running RingDeadlockStabilityTest with forward and backward mcast with {} devices",
            num_devices);
        RunRingDeadlockStabilityTestWithPersistentFabric(
            num_mcasts, num_links, num_devices, num_op_invocations, true, true);
    }
}

TEST(EdmFabric, RingDeadlockStabilityTest_RelaxedFabricStrictness) {
    constexpr size_t num_mcasts = 200000;
    constexpr size_t num_op_invocations = 5;
    constexpr bool line_sync = true;
    // Set to however many links are available
    std::optional<size_t> num_links = std::nullopt;
    std::vector<size_t> num_devices;
    auto cluster_type = tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type();

    if (cluster_type != tt::ClusterType::GALAXY) {
        return;
    }
    num_devices = {4, 8};
    for (size_t offset = 0; offset < num_devices[1]; offset++) {
        RunRingDeadlockStabilityTestWithPersistentFabric<Fabric1DRingRelaxedDeviceInitFixture>(
            num_mcasts,
            num_links,
            num_devices[0],
            num_op_invocations,
            true,
            false,
            tt::tt_fabric::FabricEriscDatamoverBuilder::default_packet_payload_size_bytes,
            offset);
    }
    for (size_t offset = 0; offset < num_devices[0]; offset++) {
        RunRingDeadlockStabilityTestWithPersistentFabric<Fabric1DRingRelaxedDeviceInitFixture>(
            num_mcasts,
            num_links,
            num_devices[1],
            num_op_invocations,
            true,
            false,
            tt::tt_fabric::FabricEriscDatamoverBuilder::default_packet_payload_size_bytes,
            offset);
    }
}
