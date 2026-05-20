// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <span>

#include <tt-metalium/experimental/dispatch_telemetry.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "command_queue_fixture.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/dispatch/command_queue_common.hpp"
#include "impl/dispatch/dispatch_mem_map.hpp"

namespace tt::tt_metal {
namespace {

uint32_t telemetry_addr() {
    return MetalContext::instance().dispatch_mem_map().get_device_command_queue_addr(
        CommandQueueDeviceAddrType::DISPATCH_TELEMETRY);
}

template <typename Telemetry>
std::span<const uint8_t> as_bytes(const Telemetry& telemetry) {
    return std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(&telemetry), sizeof(Telemetry));
}

class DispatchTelemetryReadApiTest : public UnitMeshCQFixture {
protected:
    IDevice* device() const { return devices_.at(0)->get_devices().front(); }

    void write_telemetry(const CoreCoord& core, const DispatchTelemetry& telemetry) {
        ASSERT_TRUE(detail::WriteToDeviceL1(device(), core, telemetry_addr(), as_bytes(telemetry), CoreType::WORKER));
    }

    void write_telemetry(const CoreCoord& core, const PrefetchTelemetry& telemetry) {
        ASSERT_TRUE(detail::WriteToDeviceL1(device(), core, telemetry_addr(), as_bytes(telemetry), CoreType::WORKER));
    }
};

}  // namespace

TEST_F(DispatchTelemetryReadApiTest, ReadDispatchTelemetryFromL1) {
    const CoreCoord core{0, 0};
    DispatchTelemetry telemetry;
    telemetry.blocked_by_host_count = 17;
    telemetry.unblocked_by_host_count = 19;
    write_telemetry(core, telemetry);

    auto actual = read_dispatch_telemetry(device(), core);

    ASSERT_TRUE(actual.has_value());
    EXPECT_EQ(actual->blocked_by_host_count, telemetry.blocked_by_host_count);
    EXPECT_EQ(actual->unblocked_by_host_count, telemetry.unblocked_by_host_count);
}

TEST_F(DispatchTelemetryReadApiTest, ReadDispatchTelemetryRejectsBadSignature) {
    const CoreCoord core{0, 0};
    DispatchTelemetry telemetry;
    telemetry.signature = INVALID_TELEMETRY_SIGNATURE;
    write_telemetry(core, telemetry);

    EXPECT_FALSE(read_dispatch_telemetry(device(), core).has_value());
}

TEST_F(DispatchTelemetryReadApiTest, ReadDispatchTelemetryRejectsBadVersion) {
    const CoreCoord core{0, 0};
    DispatchTelemetry telemetry;
    telemetry.version = DISPATCH_TELEMETRY_VERSION + 1;
    write_telemetry(core, telemetry);

    EXPECT_FALSE(read_dispatch_telemetry(device(), core).has_value());
}

TEST_F(DispatchTelemetryReadApiTest, ReadPrefetchTelemetryFromL1) {
    const CoreCoord core{0, 0};
    PrefetchTelemetry telemetry;
    telemetry.blocked_by_host_count = 23;
    telemetry.unblocked_by_host_count = 29;
    telemetry.command_count = 31;
    write_telemetry(core, telemetry);

    auto actual = read_prefetch_telemetry(device(), core);

    ASSERT_TRUE(actual.has_value());
    EXPECT_EQ(actual->blocked_by_host_count, telemetry.blocked_by_host_count);
    EXPECT_EQ(actual->unblocked_by_host_count, telemetry.unblocked_by_host_count);
    EXPECT_EQ(actual->command_count, telemetry.command_count);
}

TEST_F(DispatchTelemetryReadApiTest, ReadPrefetchTelemetryRejectsBadSignature) {
    const CoreCoord core{0, 0};
    PrefetchTelemetry telemetry;
    telemetry.signature = INVALID_TELEMETRY_SIGNATURE;
    write_telemetry(core, telemetry);

    EXPECT_FALSE(read_prefetch_telemetry(device(), core).has_value());
}

TEST_F(DispatchTelemetryReadApiTest, ReadPrefetchTelemetryRejectsBadVersion) {
    const CoreCoord core{0, 0};
    PrefetchTelemetry telemetry;
    telemetry.version = PREFETCH_TELEMETRY_VERSION + 1;
    write_telemetry(core, telemetry);

    EXPECT_FALSE(read_prefetch_telemetry(device(), core).has_value());
}

}  // namespace tt::tt_metal
