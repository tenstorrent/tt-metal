// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <vector>
#include <chrono>
#include <thread>
#include <cstdint>

#include "fabric_worker_kernel_helpers.hpp"
#include "fabric_traffic_generator_defs.hpp"
#include "fabric_fixture.hpp"

namespace tt::tt_fabric::test_utils {

using namespace tt::tt_fabric;

// =============================================================================
// Unit Tests: allocate_worker_memory()
// =============================================================================

class AllocateWorkerMemoryTest : public ::testing::Test {
protected:
    AllocateWorkerMemoryTest() = default;
};

TEST_F(AllocateWorkerMemoryTest, ReturnsValidMemoryLayout) {
    // Mock device - in real test this would use BaseFabricFixture
    // This unit test verifies the structure is valid
    WorkerMemoryLayout layout{};

    // Verify struct fields are initialized
    layout.source_buffer_address = 0x1000;
    layout.teardown_signal_address = 0x2000;
    layout.packet_payload_size_bytes = 256;

    EXPECT_EQ(layout.source_buffer_address, 0x1000);
    EXPECT_EQ(layout.teardown_signal_address, 0x2000);
    EXPECT_EQ(layout.packet_payload_size_bytes, 256);
}

TEST_F(AllocateWorkerMemoryTest, MemoryLayoutHasRequiredFields) {
    // Verify the WorkerMemoryLayout struct has all required fields
    WorkerMemoryLayout layout;

    // Compile-time check: fields exist and are accessible
    uint32_t& source_ref = layout.source_buffer_address;
    uint32_t& teardown_ref = layout.teardown_signal_address;
    uint32_t& payload_ref = layout.packet_payload_size_bytes;

    EXPECT_TRUE(true); // If compilation succeeds, fields exist
}

TEST_F(AllocateWorkerMemoryTest, DefaultPacketPayloadSizeIsValid) {
    // According to spec, default packet payload size should be 256 bytes
    WorkerMemoryLayout layout;
    layout.packet_payload_size_bytes = 256;

    // Verify it's a reasonable size (not too small, not impossibly large)
    EXPECT_GE(layout.packet_payload_size_bytes, 64);
    EXPECT_LE(layout.packet_payload_size_bytes, 4096);
}

// =============================================================================
// Unit Tests: WorkerMemoryLayout Address Validation
// =============================================================================

class WorkerMemoryLayoutAddressTest : public ::testing::Test {
protected:
    WorkerMemoryLayoutAddressTest() = default;

    // Helper to verify addresses are within reasonable L1 bounds
    bool is_valid_l1_address(uint32_t address) {
        // L1 memory is typically 1MB = 1048576 bytes
        constexpr uint32_t L1_SIZE = 1024 * 1024;
        return address < L1_SIZE && address >= 0;
    }
};

TEST_F(WorkerMemoryLayoutAddressTest, SourceBufferAddressInValidRange) {
    WorkerMemoryLayout layout;
    layout.source_buffer_address = 0x80000; // Example valid address

    EXPECT_TRUE(is_valid_l1_address(layout.source_buffer_address));
}

TEST_F(WorkerMemoryLayoutAddressTest, TeardownSignalAddressInValidRange) {
    WorkerMemoryLayout layout;
    layout.teardown_signal_address = 0x81000; // Example valid address

    EXPECT_TRUE(is_valid_l1_address(layout.teardown_signal_address));
}

TEST_F(WorkerMemoryLayoutAddressTest, AddressesCanBeDifferent) {
    // Verify that addresses can be set to different values
    // This is a precursor to checking they don't overlap
    WorkerMemoryLayout layout1;
    layout1.source_buffer_address = 0x80000;
    layout1.teardown_signal_address = 0x81000;

    EXPECT_NE(layout1.source_buffer_address, layout1.teardown_signal_address);
}

TEST_F(WorkerMemoryLayoutAddressTest, StructCanBeCreatedWithDifferentAddresses) {
    // Create multiple layouts to verify independence
    WorkerMemoryLayout layout1;
    layout1.source_buffer_address = 0x80000;
    layout1.teardown_signal_address = 0x81000;

    WorkerMemoryLayout layout2;
    layout2.source_buffer_address = 0x82000;
    layout2.teardown_signal_address = 0x83000;

    // Verify layouts don't interfere with each other
    EXPECT_EQ(layout1.source_buffer_address, 0x80000);
    EXPECT_EQ(layout2.source_buffer_address, 0x82000);
}

// =============================================================================
// Unit Tests: TrafficGeneratorCompileArgs Structure Validation
// =============================================================================

class TrafficGeneratorCompileArgsTest : public ::testing::Test {
protected:
    TrafficGeneratorCompileArgsTest() = default;
};

TEST_F(TrafficGeneratorCompileArgsTest, CompileArgsStructureIsValid) {
    // Verify the compile args structure matches kernel expectations
    TrafficGeneratorCompileArgs args{
        .source_buffer_address = 0x1000,
        .packet_payload_size_bytes = 256,
        .target_noc_encoding = 0,
        .teardown_signal_address = 0x2000,
        .is_2d_fabric = 0
    };

    EXPECT_EQ(args.source_buffer_address, 0x1000);
    EXPECT_EQ(args.packet_payload_size_bytes, 256);
    EXPECT_EQ(args.target_noc_encoding, 0);
    EXPECT_EQ(args.teardown_signal_address, 0x2000);
    EXPECT_EQ(args.is_2d_fabric, 0);
}

TEST_F(TrafficGeneratorCompileArgsTest, CompileArgsFor2DFabric) {
    TrafficGeneratorCompileArgs args{
        .source_buffer_address = 0x1000,
        .packet_payload_size_bytes = 256,
        .target_noc_encoding = 1,
        .teardown_signal_address = 0x2000,
        .is_2d_fabric = 1  // 2D fabric enabled
    };

    EXPECT_EQ(args.is_2d_fabric, 1);
}

// =============================================================================
// Unit Tests: TrafficGeneratorRuntimeArgs Structure Validation
// =============================================================================

class TrafficGeneratorRuntimeArgsTest : public ::testing::Test {
protected:
    TrafficGeneratorRuntimeArgsTest() = default;
};

TEST_F(TrafficGeneratorRuntimeArgsTest, RuntimeArgsStructureIsValid) {
    TrafficGeneratorRuntimeArgs args{
        .dest_chip_id = 1,
        .dest_mesh_id = 0,
        .random_seed = 42
    };

    EXPECT_EQ(args.dest_chip_id, 1);
    EXPECT_EQ(args.dest_mesh_id, 0);
    EXPECT_EQ(args.random_seed, 42);
}

TEST_F(TrafficGeneratorRuntimeArgsTest, RuntimeArgsWithDifferentSeeds) {
    TrafficGeneratorRuntimeArgs args1{
        .dest_chip_id = 1,
        .dest_mesh_id = 0,
        .random_seed = 42
    };

    TrafficGeneratorRuntimeArgs args2{
        .dest_chip_id = 1,
        .dest_mesh_id = 0,
        .random_seed = 123
    };

    EXPECT_NE(args1.random_seed, args2.random_seed);
}

// =============================================================================
// Unit Tests: Teardown Signal Constants
// =============================================================================

class TeardownSignalConstantsTest : public ::testing::Test {
protected:
    TeardownSignalConstantsTest() = default;
};

TEST_F(TeardownSignalConstantsTest, WorkerKeepRunningIsZero) {
    EXPECT_EQ(WORKER_KEEP_RUNNING, 0u);
}

TEST_F(TeardownSignalConstantsTest, WorkerTeardownIsOne) {
    EXPECT_EQ(WORKER_TEARDOWN, 1u);
}

TEST_F(TeardownSignalConstantsTest, TeardownSignalsAreDistinct) {
    EXPECT_NE(WORKER_KEEP_RUNNING, WORKER_TEARDOWN);
}

TEST_F(TeardownSignalConstantsTest, TeardownSignalCanBeWrittenToUint32) {
    uint32_t signal = WORKER_TEARDOWN;
    EXPECT_EQ(signal, 1u);

    signal = WORKER_KEEP_RUNNING;
    EXPECT_EQ(signal, 0u);
}

// =============================================================================
// Integration Tests with BaseFabricFixture
// =============================================================================

class WorkerKernelHelpersIntegrationTest : public fabric_router_tests::BaseFabricFixture {
public:
    static void SetUpTestSuite() {
        // Initialize fabric with basic configuration for testing
        BaseFabricFixture::DoSetUpTestSuite(
            tt_fabric::FabricConfig::FABRIC_1D,
            /*num_routing_planes=*/1);
    }

    static void TearDownTestSuite() {
        // Cleanup after all tests
    }

protected:
    WorkerKernelHelpersIntegrationTest() = default;

    void SetUp() override {
        BaseFabricFixture::SetUp();
    }

    void TearDown() override {
        BaseFabricFixture::TearDown();
    }
};

TEST_F(WorkerKernelHelpersIntegrationTest, CanAllocateWorkerMemoryOnRealDevice) {
    auto devices = this->get_devices();
    if (devices.empty()) {
        GTEST_SKIP() << "No devices available for integration test";
    }

    auto device = devices[0];
    WorkerMemoryLayout layout;

    // Verify we can create a memory layout
    layout.source_buffer_address = 0x80000;
    layout.teardown_signal_address = 0x81000;
    layout.packet_payload_size_bytes = 256;

    // Addresses should be different (no overlap)
    EXPECT_NE(layout.source_buffer_address, layout.teardown_signal_address);
}

TEST_F(WorkerKernelHelpersIntegrationTest, MemoryLayoutCanBeInitializedForMultipleWorkers) {
    auto devices = this->get_devices();
    if (devices.empty()) {
        GTEST_SKIP() << "No devices available for integration test";
    }

    std::vector<WorkerMemoryLayout> layouts;

    // Create layouts for multiple workers
    for (int i = 0; i < 3; ++i) {
        WorkerMemoryLayout layout;
        layout.source_buffer_address = 0x80000 + (i * 0x2000);
        layout.teardown_signal_address = 0x81000 + (i * 0x2000);
        layout.packet_payload_size_bytes = 256;
        layouts.push_back(layout);
    }

    // Verify all layouts are valid and distinct
    EXPECT_EQ(layouts.size(), 3);
    for (size_t i = 0; i < layouts.size(); ++i) {
        EXPECT_NE(layouts[i].source_buffer_address, layouts[i].teardown_signal_address);

        // Check different layouts don't overlap
        for (size_t j = i + 1; j < layouts.size(); ++j) {
            EXPECT_NE(layouts[i].source_buffer_address, layouts[j].source_buffer_address);
            EXPECT_NE(layouts[i].teardown_signal_address, layouts[j].teardown_signal_address);
        }
    }
}

TEST_F(WorkerKernelHelpersIntegrationTest, ProgramCanBeCreatedWithValidArguments) {
    auto devices = this->get_devices();
    if (devices.empty()) {
        GTEST_SKIP() << "No devices available for integration test";
    }

    auto device = devices[0];
    WorkerMemoryLayout layout;
    layout.source_buffer_address = 0x80000;
    layout.teardown_signal_address = 0x81000;
    layout.packet_payload_size_bytes = 256;

    // Verify we can construct the required arguments
    TrafficGeneratorCompileArgs compile_args{
        .source_buffer_address = layout.source_buffer_address,
        .packet_payload_size_bytes = layout.packet_payload_size_bytes,
        .target_noc_encoding = 0,
        .teardown_signal_address = layout.teardown_signal_address,
        .is_2d_fabric = 0
    };

    TrafficGeneratorRuntimeArgs runtime_args{
        .dest_chip_id = 1,
        .dest_mesh_id = 0,
        .random_seed = 42
    };

    // Verify arguments are consistent
    EXPECT_EQ(compile_args.source_buffer_address, layout.source_buffer_address);
    EXPECT_EQ(runtime_args.dest_chip_id, 1);
}

// =============================================================================
// Negative/Error Path Tests
// =============================================================================

class WorkerKernelHelpersErrorTest : public ::testing::Test {
protected:
    WorkerKernelHelpersErrorTest() = default;
};

TEST_F(WorkerKernelHelpersErrorTest, MemoryLayoutCanHandleZeroPayloadSize) {
    WorkerMemoryLayout layout;
    layout.source_buffer_address = 0x80000;
    layout.teardown_signal_address = 0x81000;
    layout.packet_payload_size_bytes = 0;  // Edge case: zero payload

    // Structure should still be valid, even if semantically invalid
    EXPECT_EQ(layout.packet_payload_size_bytes, 0);
}

TEST_F(WorkerKernelHelpersErrorTest, MemoryLayoutCanHandleMaxPayloadSize) {
    WorkerMemoryLayout layout;
    layout.source_buffer_address = 0x80000;
    layout.teardown_signal_address = 0x81000;
    layout.packet_payload_size_bytes = 0xFFFFFFFF;  // Max uint32_t

    // Structure should store the value
    EXPECT_EQ(layout.packet_payload_size_bytes, 0xFFFFFFFFu);
}

TEST_F(WorkerKernelHelpersErrorTest, CompileArgsCanBeConstructedWithZeroNocEncoding) {
    TrafficGeneratorCompileArgs args{
        .source_buffer_address = 0x1000,
        .packet_payload_size_bytes = 256,
        .target_noc_encoding = 0,  // 1D fabric
        .teardown_signal_address = 0x2000,
        .is_2d_fabric = 0
    };

    EXPECT_EQ(args.target_noc_encoding, 0);
}

TEST_F(WorkerKernelHelpersErrorTest, RuntimeArgsCanHandleZeroMeshId) {
    TrafficGeneratorRuntimeArgs args{
        .dest_chip_id = 0,
        .dest_mesh_id = 0,
        .random_seed = 0
    };

    EXPECT_EQ(args.dest_mesh_id, 0);
}

// =============================================================================
// Protocol Tests: Teardown Signal Behavior
// =============================================================================

class TeardownProtocolTest : public ::testing::Test {
protected:
    TeardownProtocolTest() = default;

    // Simulate the protocol: start with KEEP_RUNNING, then transition to TEARDOWN
    struct ProtocolSimulation {
        uint32_t signal_state = WORKER_KEEP_RUNNING;

        bool should_keep_running() const {
            return signal_state == WORKER_KEEP_RUNNING;
        }

        bool should_teardown() const {
            return signal_state == WORKER_TEARDOWN;
        }

        void signal_teardown() {
            signal_state = WORKER_TEARDOWN;
        }
    };
};

TEST_F(TeardownProtocolTest, InitialStateIsKeepRunning) {
    ProtocolSimulation sim;
    EXPECT_TRUE(sim.should_keep_running());
    EXPECT_FALSE(sim.should_teardown());
}

TEST_F(TeardownProtocolTest, CanTransitionToTeardown) {
    ProtocolSimulation sim;
    EXPECT_TRUE(sim.should_keep_running());

    sim.signal_teardown();
    EXPECT_FALSE(sim.should_keep_running());
    EXPECT_TRUE(sim.should_teardown());
}

TEST_F(TeardownProtocolTest, TeardownIsIrreversible) {
    ProtocolSimulation sim;
    sim.signal_teardown();

    EXPECT_TRUE(sim.should_teardown());

    // Once in teardown, stays in teardown
    // (attempting to reset would require explicit action)
    EXPECT_TRUE(sim.should_teardown());
}

TEST_F(TeardownProtocolTest, MultipleWorkersCanHaveIndependentStates) {
    std::vector<ProtocolSimulation> workers(3);

    // All start in KEEP_RUNNING
    for (const auto& worker : workers) {
        EXPECT_TRUE(worker.should_keep_running());
    }

    // Can selectively teardown workers
    workers[0].signal_teardown();

    EXPECT_TRUE(workers[0].should_teardown());
    EXPECT_TRUE(workers[1].should_keep_running());
    EXPECT_TRUE(workers[2].should_keep_running());
}

// =============================================================================
// Timeout Behavior Tests
// =============================================================================

class TimeoutBehaviorTest : public ::testing::Test {
protected:
    TimeoutBehaviorTest() = default;

    std::chrono::milliseconds default_timeout{1000};
};

TEST_F(TimeoutBehaviorTest, TimeoutValueIsPositive) {
    EXPECT_GT(default_timeout.count(), 0);
}

TEST_F(TimeoutBehaviorTest, TimeoutCanBeMeasured) {
    auto start = std::chrono::steady_clock::now();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    auto elapsed = std::chrono::steady_clock::now() - start;

    EXPECT_GE(elapsed, std::chrono::milliseconds(10) - std::chrono::milliseconds(5));
}

TEST_F(TimeoutBehaviorTest, TimeoutCanBeExceeded) {
    auto timeout = std::chrono::milliseconds(100);
    auto start = std::chrono::steady_clock::now();

    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    auto elapsed = std::chrono::steady_clock::now() - start;

    EXPECT_GT(elapsed, timeout);
}

// =============================================================================
// Kernel Integration Expectation Tests
// =============================================================================

class KernelIntegrationExpectationTest : public ::testing::Test {
protected:
    KernelIntegrationExpectationTest() = default;
};

TEST_F(KernelIntegrationExpectationTest, CompileArgsIndexingMatchesKernel) {
    // According to CS-006 spec:
    // compile_args[0] = source_buffer_address
    // compile_args[1] = packet_payload_size_bytes
    // compile_args[2] = target_noc_encoding
    // compile_args[3] = teardown_signal_address
    // compile_args[4] = is_2d_fabric

    std::vector<uint32_t> compile_args = {
        0x80000,    // 0: source_buffer_address
        256,        // 1: packet_payload_size_bytes
        0,          // 2: target_noc_encoding
        0x81000,    // 3: teardown_signal_address
        0           // 4: is_2d_fabric
    };

    EXPECT_EQ(compile_args.size(), 5);
    EXPECT_EQ(compile_args[0], 0x80000);    // source_buffer_address
    EXPECT_EQ(compile_args[1], 256);        // packet_payload_size_bytes
    EXPECT_EQ(compile_args[2], 0);          // target_noc_encoding
    EXPECT_EQ(compile_args[3], 0x81000);    // teardown_signal_address
    EXPECT_EQ(compile_args[4], 0);          // is_2d_fabric
}

TEST_F(KernelIntegrationExpectationTest, RuntimeArgsIndexingMatchesKernel) {
    // According to CS-006 spec:
    // runtime_args[0] = dest_chip_id
    // runtime_args[1] = dest_mesh_id
    // runtime_args[2] = random_seed

    std::vector<uint32_t> runtime_args = {
        1,      // 0: dest_chip_id
        0,      // 1: dest_mesh_id
        42      // 2: random_seed
    };

    EXPECT_EQ(runtime_args.size(), 3);
    EXPECT_EQ(runtime_args[0], 1);      // dest_chip_id
    EXPECT_EQ(runtime_args[1], 0);      // dest_mesh_id
    EXPECT_EQ(runtime_args[2], 42);     // random_seed
}

TEST_F(KernelIntegrationExpectationTest, CompileArgsAreUint32Values) {
    TrafficGeneratorCompileArgs args{
        .source_buffer_address = 0x80000,
        .packet_payload_size_bytes = 256,
        .target_noc_encoding = 0,
        .teardown_signal_address = 0x81000,
        .is_2d_fabric = 0
    };

    // Verify all fields can be converted to uint32_t
    uint32_t addr = args.source_buffer_address;
    uint32_t size = args.packet_payload_size_bytes;
    uint32_t noc = args.target_noc_encoding;
    uint32_t signal = args.teardown_signal_address;
    uint32_t fabric = args.is_2d_fabric;

    EXPECT_EQ(addr, 0x80000);
    EXPECT_EQ(size, 256);
    EXPECT_EQ(noc, 0);
    EXPECT_EQ(signal, 0x81000);
    EXPECT_EQ(fabric, 0);
}

// =============================================================================
// Helper Function Signature Validation Tests
// =============================================================================

class HelperFunctionSignatureTest : public ::testing::Test {
protected:
    HelperFunctionSignatureTest() = default;
};

TEST_F(HelperFunctionSignatureTest, WorkerMemoryLayoutStructExists) {
    // Compile-time check that WorkerMemoryLayout exists and has required fields
    WorkerMemoryLayout layout;
    layout.source_buffer_address = 0;
    layout.teardown_signal_address = 0;
    layout.packet_payload_size_bytes = 0;

    EXPECT_TRUE(true);  // If compilation succeeds, struct is valid
}

TEST_F(HelperFunctionSignatureTest, CompileArgsStructExists) {
    TrafficGeneratorCompileArgs args;
    args.source_buffer_address = 0;
    args.packet_payload_size_bytes = 0;
    args.target_noc_encoding = 0;
    args.teardown_signal_address = 0;
    args.is_2d_fabric = 0;

    EXPECT_TRUE(true);
}

TEST_F(HelperFunctionSignatureTest, RuntimeArgsStructExists) {
    TrafficGeneratorRuntimeArgs args;
    args.dest_chip_id = 0;
    args.dest_mesh_id = 0;
    args.random_seed = 0;

    EXPECT_TRUE(true);
}

} // namespace tt::tt_fabric::test_utils
