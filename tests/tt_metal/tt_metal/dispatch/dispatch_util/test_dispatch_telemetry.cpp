// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <array>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include <tt-metalium/experimental/dispatch_telemetry.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <umd/device/pcie/pci_device.hpp>
#include <umd/device/tt_device/tt_device.hpp>
#include <umd/device/types/core_coordinates.hpp>

#include "command_queue_fixture.hpp"
#include "multi_command_queue_fixture.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/dispatch/command_queue_common.hpp"
#include "impl/dispatch/dispatch_telemetry.hpp"
#include "impl/dispatch/dispatch_mem_map.hpp"
#include "impl/dispatch/dispatch_query_manager.hpp"
#include "distributed/mesh_trace.hpp"
#include "llrt/core_descriptor.hpp"

namespace tt::tt_metal {
namespace {

constexpr const char* kObserverChildEnv = "TT_METAL_DISPATCH_TELEMETRY_OBSERVER_PCI_DEVICE_ID";

template <typename TCoreType>
Program create_blank_program(const TCoreType& core) {
    Program program = CreateProgram();
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/blank.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    return program;
}

template <typename Func>
void for_each_worker_core(const CoreRangeSet& worker_cores, Func func) {
    for (const CoreRange& core_range : worker_cores.ranges()) {
        for (uint32_t y = core_range.start_coord.y; y <= core_range.end_coord.y; ++y) {
            for (uint32_t x = core_range.start_coord.x; x <= core_range.end_coord.x; ++x) {
                func(CoreCoord{x, y});
            }
        }
    }
}

std::optional<std::string> smc_runtime_telemetry_unavailable_reason(tt::umd::TTDevice& tt_device) {
    try {
        // Throws when the device has no firmware info provider (ex: simulators)
        auto* firmware_info_provider = tt_device.get_firmware_info_provider();
        if (!firmware_info_provider->get_runtime_telemetry_buffer_size().has_value()) {
            return "SMC runtime telemetry buffer is unavailable";
        }
        if (!firmware_info_provider->get_runtime_telemetry_buffer_address().has_value()) {
            return "SMC runtime telemetry buffer address is unavailable or invalid";
        }
        return std::nullopt;
    } catch (const std::exception& e) {
        return std::string("Firmware info provider is unavailable: ") + e.what();
    }
}

}  // namespace

class DispatchTelemetryReadApiTest : public UnitMeshCQFixture {
protected:
    void SetUp() override {
        // Run setup early to instantiate mesh device
        UnitMeshCQFixture::SetUp();
        if (IsSkipped()) {
            return;
        }

        if (MetalContext::instance().rtoptions().get_dispatch_telemetry_disabled()) {
            GTEST_SKIP() << "Dispatch telemetry is disabled";
        }

        if (auto skip_reason = smc_runtime_telemetry_unavailable_reason(tt_device()); skip_reason.has_value()) {
            GTEST_SKIP() << *skip_reason;
        }
    }

    IDevice* device() const { return devices_.at(0)->get_devices().front(); }

    tt::umd::TTDevice& tt_device() const {
        return *MetalContext::instance().get_cluster().get_driver()->get_tt_device(device()->id());
    }

    bool worker_dispatch_enabled() const {
        return MetalContext::instance().get_dispatch_core_manager().get_dispatch_core_type() == CoreType::WORKER;
    }

    std::optional<CoreCoord> available_ethernet_core_for_l1_test() const {
        const auto& inactive_ethernet_cores = device()->get_inactive_ethernet_cores();
        if (!inactive_ethernet_cores.empty()) {
            return *inactive_ethernet_cores.begin();
        }

        const auto& active_ethernet_cores = device()->get_active_ethernet_cores(true);
        if (!active_ethernet_cores.empty()) {
            return *active_ethernet_cores.begin();
        }

        return std::nullopt;
    }

    template <typename TelemetryType>
    void write_telemetry(CoreType core_type, const CoreCoord& core, const TelemetryType& telemetry) {
        // write to cq_id 0's slot to match the cq_id=0 default that read_dispatch_core_telemetry() uses at every call
        // site in this file.
        auto telemetry_addr = MetalContext::instance().dispatch_mem_map().get_device_command_queue_addr(
            CommandQueueDeviceAddrType::DISPATCH_TELEMETRY, /*cq_id=*/0);
        auto bytes = std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(&telemetry), sizeof(TelemetryType));
        ASSERT_TRUE(detail::WriteToDeviceL1(device(), core, telemetry_addr, bytes, core_type));
    }

    std::optional<CoreCoord> dispatch_s_virtual_core(uint8_t cq_id = 0) const {
        if (!MetalContext::instance().get_dispatch_query_manager().dispatch_s_enabled()) {
            return std::nullopt;
        }

        auto& metal_context = MetalContext::instance();
        auto& dcm = metal_context.get_dispatch_core_manager();
        const ChipId chip = device()->id();
        const uint16_t channel = metal_context.get_cluster().get_assigned_channel_for_device(chip);
        if (!dcm.is_dispatcher_s_core_allocated(chip, channel, cq_id)) {
            return std::nullopt;
        }

        const auto& logical_cxy = dcm.dispatcher_s_core(chip, channel, cq_id);
        const CoreType core_type = dcm.get_dispatch_core_type();
        return device()->virtual_core_from_logical_core(CoreCoord{logical_cxy.x, logical_cxy.y}, core_type);
    }

    std::optional<CoreCoord> dispatch_virtual_core(uint8_t cq_id = 0) const {
        auto& metal_context = MetalContext::instance();
        auto& dcm = metal_context.get_dispatch_core_manager();
        const ChipId chip = device()->id();
        const uint16_t channel = metal_context.get_cluster().get_assigned_channel_for_device(chip);
        if (!dcm.is_dispatcher_core_allocated(chip, channel, cq_id)) {
            return std::nullopt;
        }

        const auto& logical_cxy = dcm.dispatcher_core(chip, channel, cq_id);
        const CoreType core_type = dcm.get_dispatch_core_type();
        return device()->virtual_core_from_logical_core(CoreCoord{logical_cxy.x, logical_cxy.y}, core_type);
    }

    std::optional<CoreCoord> dispatch_telemetry_virtual_core(uint8_t cq_id = 0) const {
        if (auto dispatch_s_core = dispatch_s_virtual_core(cq_id); dispatch_s_core.has_value()) {
            return dispatch_s_core;
        }
        return dispatch_virtual_core(cq_id);
    }
};

class DispatchTelemetryMultiCQReadApiTest : public UnitMeshMultiCQSingleDeviceFixture {
protected:
    void SetUp() override {
        // Run setup early to instantiate mesh device
        UnitMeshMultiCQSingleDeviceFixture::SetUp();
        if (IsSkipped()) {
            return;
        }

        if (MetalContext::instance().rtoptions().get_dispatch_telemetry_disabled()) {
            GTEST_SKIP() << "Dispatch telemetry is disabled";
        }

        if (auto skip_reason = smc_runtime_telemetry_unavailable_reason(tt_device()); skip_reason.has_value()) {
            GTEST_SKIP() << *skip_reason;
        }
    }

    IDevice* device() const { return device_->get_devices().front(); }

    tt::umd::TTDevice& tt_device() const {
        return *MetalContext::instance().get_cluster().get_driver()->get_tt_device(device()->id());
    }

    std::optional<CoreCoord> dispatch_s_virtual_core(uint8_t cq_id = 0) const {
        if (!MetalContext::instance().get_dispatch_query_manager().dispatch_s_enabled()) {
            return std::nullopt;
        }

        auto& metal_context = MetalContext::instance();
        auto& dcm = metal_context.get_dispatch_core_manager();
        const ChipId chip = device()->id();
        const uint16_t channel = metal_context.get_cluster().get_assigned_channel_for_device(chip);
        if (!dcm.is_dispatcher_s_core_allocated(chip, channel, cq_id)) {
            return std::nullopt;
        }

        const auto& logical_cxy = dcm.dispatcher_s_core(chip, channel, cq_id);
        const CoreType core_type = dcm.get_dispatch_core_type();
        return device()->virtual_core_from_logical_core(CoreCoord{logical_cxy.x, logical_cxy.y}, core_type);
    }

    std::optional<CoreCoord> dispatch_virtual_core(uint8_t cq_id = 0) const {
        auto& metal_context = MetalContext::instance();
        auto& dcm = metal_context.get_dispatch_core_manager();
        const ChipId chip = device()->id();
        const uint16_t channel = metal_context.get_cluster().get_assigned_channel_for_device(chip);
        if (!dcm.is_dispatcher_core_allocated(chip, channel, cq_id)) {
            return std::nullopt;
        }

        const auto& logical_cxy = dcm.dispatcher_core(chip, channel, cq_id);
        const CoreType core_type = dcm.get_dispatch_core_type();
        return device()->virtual_core_from_logical_core(CoreCoord{logical_cxy.x, logical_cxy.y}, core_type);
    }

    std::optional<CoreCoord> dispatch_telemetry_virtual_core(uint8_t cq_id = 0) const {
        if (auto dispatch_s_core = dispatch_s_virtual_core(cq_id); dispatch_s_core.has_value()) {
            return dispatch_s_core;
        }
        return dispatch_virtual_core(cq_id);
    }
};

class DispatchTelemetrySlowDispatchTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (MetalContext::instance().rtoptions().get_fast_dispatch()) {
            GTEST_SKIP() << "Requires TT_METAL_SLOW_DISPATCH_MODE=1";
        }
        if (MetalContext::instance().rtoptions().get_dispatch_telemetry_disabled()) {
            GTEST_SKIP() << "Dispatch telemetry is disabled";
        }

        mesh_device_ = distributed::MeshDevice::create(distributed::MeshDeviceConfig(distributed::MeshShape{1, 1}));

        if (auto skip_reason = smc_runtime_telemetry_unavailable_reason(tt_device()); skip_reason.has_value()) {
            GTEST_SKIP() << *skip_reason;
        }
    }

    void TearDown() override {
        if (mesh_device_) {
            mesh_device_->close();
            mesh_device_.reset();
        }
    }

    IDevice* device() const { return mesh_device_->get_devices().front(); }

    tt::umd::TTDevice& tt_device() const {
        return *MetalContext::instance().get_cluster().get_driver()->get_tt_device(device()->id());
    }

    std::shared_ptr<distributed::MeshDevice> mesh_device_;
};

class DispatchTelemetryHostL1WaitTest : public DispatchTelemetryReadApiTest {
protected:
    void SetUp() override {
        DispatchTelemetryReadApiTest::SetUp();
        if (IsSkipped()) {
            return;
        }

        release_addr_ = devices_.at(0)->allocator()->get_base_allocator_addr(HalMemType::L1);
        started_addr_ = release_addr_ + sizeof(uint32_t);
    }

    void TearDown() override {
        // Defensively release every compute core before tearing down so close() can always drain,
        // regardless of how a test exited.
        if (release_addr_ != 0 && !devices_.empty() && devices_.at(0) != nullptr) {
            const CoreCoord grid_size = device()->compute_with_storage_grid_size();
            const CoreRangeSet all_cores{CoreRange(CoreCoord{0, 0}, CoreCoord{grid_size.x - 1, grid_size.y - 1})};
            release_worker(all_cores);
        }
        DispatchTelemetryReadApiTest::TearDown();
    }

    template <typename TCoreType>
    Program create_l1_wait_program(const TCoreType& core) {
        Program program = CreateProgram();
        auto kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/wait_for_host_l1_write.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        SetRuntimeArgs(program, kernel, core, {release_addr_, release_value_, started_addr_, started_value_});
        return program;
    }

    template <typename TCoreType>
    void release_core_and_finish(const TCoreType& core) {
        std::vector<uint32_t> release_word{release_value_};
        EXPECT_TRUE(detail::WriteToDeviceL1(device(), core, release_addr_, release_word));
        Finish(devices_.at(0)->mesh_command_queue());
    }

    void reset_worker_l1_state(const CoreCoord& core) {
        std::vector<uint32_t> zero_word{0};
        ASSERT_TRUE(detail::WriteToDeviceL1(device(), core, release_addr_, zero_word));
        ASSERT_TRUE(detail::WriteToDeviceL1(device(), core, started_addr_, zero_word));
    }

    void reset_worker_l1_state(const CoreRangeSet& worker_cores) {
        for_each_worker_core(worker_cores, [&](const CoreCoord& core) { reset_worker_l1_state(core); });
    }

    void release_worker(const CoreCoord& core) {
        std::vector<uint32_t> release_word{release_value_};
        EXPECT_TRUE(detail::WriteToDeviceL1(device(), core, release_addr_, release_word));
    }

    void release_worker(const CoreRangeSet& worker_cores) {
        for_each_worker_core(worker_cores, [&](const CoreCoord& core) { release_worker(core); });
    }

    bool worker_reached_l1_wait(
        IDevice* device,
        const CoreCoord& core,
        std::chrono::milliseconds timeout = std::chrono::milliseconds(5000)) const {
        auto deadline = std::chrono::steady_clock::now() + timeout;
        std::vector<uint32_t> readback(1);

        do {
            detail::ReadFromDeviceL1(device, core, started_addr_, sizeof(uint32_t), readback);
            if (readback[0] == started_value_) {
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        } while (std::chrono::steady_clock::now() < deadline);

        return false;
    }

    bool all_workers_reached_l1_wait(
        IDevice* device,
        const CoreRangeSet& worker_cores,
        std::chrono::milliseconds timeout = std::chrono::milliseconds(5000)) const {
        auto deadline = std::chrono::steady_clock::now() + timeout;
        std::vector<uint32_t> readback(1);

        do {
            bool all_started = true;
            for_each_worker_core(worker_cores, [&](const CoreCoord& core) {
                if (!all_started) {
                    return;
                }
                detail::ReadFromDeviceL1(device, core, started_addr_, sizeof(uint32_t), readback);
                all_started = readback[0] == started_value_;
            });
            if (all_started) {
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        } while (std::chrono::steady_clock::now() < deadline);

        return false;
    }

    static constexpr uint32_t release_value_ = 0x67216721;
    static constexpr uint32_t started_value_ = 0x5A5A5A5A;

    uint32_t release_addr_ = 0;
    uint32_t started_addr_ = 0;
};

TEST_F(DispatchTelemetryReadApiTest, SMCControlIsInitialized) {
    // Initialization of the mesh device should have written the default SMC control block to the device already
    auto control = read_smc_dispatch_telemetry_control(tt_device());
    ASSERT_TRUE(control.has_value());

    // TODO: When dispatch telemetry is supported on Quasar, we'll need to pass in the command queue id(s) here.
    auto expected_addr = MetalContext::instance().dispatch_mem_map().get_device_command_queue_addr(
        CommandQueueDeviceAddrType::DISPATCH_TELEMETRY, /*cq_id=*/0);
    const uint32_t control_version = control->version;
    const uint32_t control_signature = control->signature;
    const uint8_t control_flags = control->flags;
    const uint32_t dispatch_telemetry_addr = control->dispatch_telemetry_addr;
    const uint8_t num_hw_cqs = control->num_hw_cqs;
    EXPECT_EQ(control_version, dispatch_telemetry_types::DISPATCH_TELEMETRY_VERSION);
    EXPECT_EQ(control_signature, dispatch_telemetry_types::SMC_TELEMETRY_SIGNATURE);
    EXPECT_EQ(control_flags, 0);
    EXPECT_EQ(dispatch_telemetry_addr, expected_addr);
    EXPECT_EQ(num_hw_cqs, device()->num_hw_cqs());
    ASSERT_LE(num_hw_cqs, dispatch_telemetry_types::RESERVED_CQ_SPACE);

    {
        auto& metal_context = MetalContext::instance();
        auto& dcm = metal_context.get_dispatch_core_manager();
        const ChipId chip = device()->id();
        const uint16_t channel = metal_context.get_cluster().get_assigned_channel_for_device(chip);
        const CoreType core_type = dcm.get_dispatch_core_type();

        const auto dispatch_core_to_virtual_core = [&](const tt_cxy_pair& dispatch_core) {
            return device()->virtual_core_from_logical_core(CoreCoord{dispatch_core.x, dispatch_core.y}, core_type);
        };

        const auto expect_smc_coord_matches_virtual_core = [&](uint32_t smc_xy, const CoreCoord& virtual_core) {
            EXPECT_EQ(dispatch_telemetry_types::smc_dispatch_core_x(smc_xy), virtual_core.x);
            EXPECT_EQ(dispatch_telemetry_types::smc_dispatch_core_y(smc_xy), virtual_core.y);
        };

        const auto expect_prefetch_coord = [&](uint32_t smc_xy, uint8_t cq_id) {
            if (dcm.is_prefetcher_core_allocated(chip, channel, cq_id)) {
                expect_smc_coord_matches_virtual_core(
                    smc_xy, dispatch_core_to_virtual_core(dcm.prefetcher_core(chip, channel, cq_id)));
            } else if (dcm.is_prefetcher_d_core_allocated(chip, channel, cq_id)) {
                expect_smc_coord_matches_virtual_core(
                    smc_xy, dispatch_core_to_virtual_core(dcm.prefetcher_d_core(chip, channel, cq_id)));
            } else {
                FAIL() << "Expected CQ " << static_cast<uint32_t>(cq_id) << " to have an allocated prefetch core";
            }
        };

        const auto expect_dispatch_coord = [&](uint32_t smc_xy, uint8_t cq_id) {
            if (dcm.is_dispatcher_core_allocated(chip, channel, cq_id)) {
                expect_smc_coord_matches_virtual_core(
                    smc_xy, dispatch_core_to_virtual_core(dcm.dispatcher_core(chip, channel, cq_id)));
            } else if (dcm.is_dispatcher_d_core_allocated(chip, channel, cq_id)) {
                expect_smc_coord_matches_virtual_core(
                    smc_xy, dispatch_core_to_virtual_core(dcm.dispatcher_d_core(chip, channel, cq_id)));
            } else {
                FAIL() << "Expected CQ " << static_cast<uint32_t>(cq_id) << " to have an allocated dispatch core";
            }
        };

        for (uint32_t cq = 0; cq < num_hw_cqs; ++cq) {
            SCOPED_TRACE(cq);
            const auto cq_id = static_cast<uint8_t>(cq);
            const auto smc_coords = control->cq_dispatch_core_coords[cq];

            expect_prefetch_coord(smc_coords.prefetch_xy, cq_id);
            expect_dispatch_coord(smc_coords.dispatch_xy, cq_id);

            if (dcm.is_dispatcher_s_core_allocated(chip, channel, cq_id)) {
                expect_smc_coord_matches_virtual_core(
                    smc_coords.dispatch_s_xy,
                    dispatch_core_to_virtual_core(dcm.dispatcher_s_core(chip, channel, cq_id)));
            } else {
                const uint32_t dispatch_s_xy = smc_coords.dispatch_s_xy;
                EXPECT_EQ(dispatch_s_xy, dispatch_telemetry_types::INVALID_SMC_DISPATCH_CORE_COORDS);
            }
        }
    }

    // Validate cq_dispatch_core_coords for inactive CQs are invalid.
    for (uint32_t cq = num_hw_cqs; cq < dispatch_telemetry_types::RESERVED_CQ_SPACE; ++cq) {
        const auto smc_coords = control->cq_dispatch_core_coords[cq];
        const uint32_t prefetch_xy = smc_coords.prefetch_xy;
        const uint32_t dispatch_xy = smc_coords.dispatch_xy;
        const uint32_t dispatch_s_xy = smc_coords.dispatch_s_xy;
        EXPECT_EQ(prefetch_xy, dispatch_telemetry_types::INVALID_SMC_DISPATCH_CORE_COORDS);
        EXPECT_EQ(dispatch_xy, dispatch_telemetry_types::INVALID_SMC_DISPATCH_CORE_COORDS);
        EXPECT_EQ(dispatch_s_xy, dispatch_telemetry_types::INVALID_SMC_DISPATCH_CORE_COORDS);
    }
}

TEST_F(DispatchTelemetrySlowDispatchTest, SMCControlIsInitialized) {
    auto control = read_smc_dispatch_telemetry_control(tt_device());
    ASSERT_TRUE(control.has_value());

    // TODO: When dispatch telemetry is supported on Quasar, we'll need to pass in the command queue id(s) here.
    auto expected_addr = MetalContext::instance().dispatch_mem_map().get_device_command_queue_addr(
        CommandQueueDeviceAddrType::DISPATCH_TELEMETRY, /*cq_id=*/0);
    const uint32_t control_version = control->version;
    const uint32_t control_signature = control->signature;
    const uint8_t control_flags = control->flags;
    const uint32_t dispatch_telemetry_addr = control->dispatch_telemetry_addr;
    const uint8_t num_hw_cqs = control->num_hw_cqs;
    EXPECT_EQ(control_version, dispatch_telemetry_types::DISPATCH_TELEMETRY_VERSION);
    EXPECT_EQ(control_signature, dispatch_telemetry_types::SMC_TELEMETRY_SIGNATURE);
    EXPECT_TRUE(
        control_flags &
        static_cast<uint32_t>(dispatch_telemetry_types::SMCDispatchTelemetryFlags::SLOW_DISPATCH_ENABLED));
    EXPECT_EQ(dispatch_telemetry_addr, expected_addr);
    EXPECT_EQ(num_hw_cqs, device()->num_hw_cqs());
    ASSERT_LE(num_hw_cqs, dispatch_telemetry_types::RESERVED_CQ_SPACE);

    for (auto smc_coords : control->cq_dispatch_core_coords) {
        const uint32_t prefetch_xy = smc_coords.prefetch_xy;
        const uint32_t dispatch_xy = smc_coords.dispatch_xy;
        const uint32_t dispatch_s_xy = smc_coords.dispatch_s_xy;
        EXPECT_EQ(prefetch_xy, dispatch_telemetry_types::INVALID_SMC_DISPATCH_CORE_COORDS);
        EXPECT_EQ(dispatch_xy, dispatch_telemetry_types::INVALID_SMC_DISPATCH_CORE_COORDS);
        EXPECT_EQ(dispatch_s_xy, dispatch_telemetry_types::INVALID_SMC_DISPATCH_CORE_COORDS);
    }
}

// Only meant to be run as a child process within ReadInfoFromSeparateProcessWhileDeviceInUse
TEST(DispatchTelemetryObserverChild, ReadInfoFromDeviceOwnedByAnotherProcess) {
    const char* pci_device_id_env = std::getenv(kObserverChildEnv);
    if (pci_device_id_env == nullptr) {
        GTEST_SKIP() << "Observer child test only runs when " << kObserverChildEnv << " is set";
    }

    char* parse_end = nullptr;
    const long pci_device_id = std::strtol(pci_device_id_env, &parse_end, 10);
    ASSERT_NE(parse_end, pci_device_id_env);
    ASSERT_EQ(*parse_end, '\0');
    ASSERT_GE(pci_device_id, 0);

    std::unique_ptr<tt::umd::TTDevice> tt_device = tt::umd::TTDevice::create(static_cast<int>(pci_device_id));
    tt_device->init_tt_device();
    if (auto skip_reason = smc_runtime_telemetry_unavailable_reason(*tt_device); skip_reason.has_value()) {
        GTEST_SKIP() << *skip_reason;
    }
    DispatchTelemetry telemetry(*tt_device);

    EXPECT_EQ(telemetry.version(), dispatch_telemetry_types::DISPATCH_TELEMETRY_VERSION);
    auto info = telemetry.read_info();
    ASSERT_TRUE(info.has_value());
    EXPECT_FALSE(info->info_cqs.empty());
}

TEST_F(DispatchTelemetryHostL1WaitTest, ReadInfoFromSeparateProcessWhileDeviceInUse) {
    constexpr const char* observerChildFilter =
        "DispatchTelemetryObserverChild.ReadInfoFromDeviceOwnedByAnotherProcess";

    auto* pci_device = tt_device().get_pci_device();
    if (pci_device == nullptr) {
        GTEST_SKIP() << "Requires a PCIe-backed TTDevice";
    }
    const int pci_device_num = pci_device->get_device_num();

    auto& cq = devices_.at(0)->mesh_command_queue();
    const CoreCoord worker_core{0, 0};
    reset_worker_l1_state(worker_core);

    distributed::MeshWorkload waiting_workload;
    waiting_workload.add_program(device_range_, create_l1_wait_program(worker_core));
    distributed::EnqueueMeshWorkload(cq, waiting_workload, false);

    const bool worker_started = worker_reached_l1_wait(device(), worker_core);
    if (!worker_started) {
        release_core_and_finish(worker_core);
    }
    ASSERT_TRUE(worker_started);

    pid_t pid = fork();
    if (pid == -1) {
        const int fork_errno = errno;
        release_core_and_finish(worker_core);
        FAIL() << "fork() failed: " << std::strerror(fork_errno);
    }

    if (pid == 0) {
        const std::string pci_device_id = std::to_string(pci_device_num);
        if (setenv(kObserverChildEnv, pci_device_id.c_str(), 1) != 0) {
            _exit(2);
        }

        const std::string filter_arg = std::string("--gtest_filter=") + observerChildFilter;
        char* const args[] = {const_cast<char*>("/proc/self/exe"), const_cast<char*>(filter_arg.c_str()), nullptr};
        execv("/proc/self/exe", args);
        _exit(3);
    }

    int status = 0;
    const pid_t wait_result = waitpid(pid, &status, 0);
    release_core_and_finish(worker_core);

    ASSERT_EQ(wait_result, pid) << "waitpid failed";
    ASSERT_TRUE(WIFEXITED(status)) << "Child terminated abnormally";
    EXPECT_EQ(WEXITSTATUS(status), 0) << "Child exited with code " << WEXITSTATUS(status);
}

TEST_F(DispatchTelemetryReadApiTest, ReadDispatchCoreTelemetryFromWorkerL1) {
    const CoreCoord core{0, 0};
    dispatch_telemetry_types::DispatchCoreTelemetry telemetry;
    telemetry.upstream_blocked_count = 17;
    telemetry.upstream_unblocked_count = 19;
    telemetry.program_count = 21;
    write_telemetry(CoreType::WORKER, core, telemetry);

    const CoreCoord virtual_core = device()->virtual_core_from_logical_core(core, CoreType::WORKER);
    auto actual = read_dispatch_core_telemetry(tt_device(), virtual_core);

    ASSERT_TRUE(actual.has_value());
    EXPECT_EQ(actual->upstream_blocked_count, telemetry.upstream_blocked_count);
    EXPECT_EQ(actual->upstream_unblocked_count, telemetry.upstream_unblocked_count);
    EXPECT_EQ(actual->program_count, telemetry.program_count);
}

TEST_F(DispatchTelemetryReadApiTest, ReadDispatchCoreTelemetryFromEthL1) {
    const auto core = available_ethernet_core_for_l1_test();
    if (!core.has_value()) {
        GTEST_SKIP() << "No ethernet cores available for L1 telemetry test";
    }

    dispatch_telemetry_types::DispatchCoreTelemetry telemetry;
    telemetry.upstream_blocked_count = 17;
    telemetry.upstream_unblocked_count = 19;
    telemetry.program_count = 21;
    write_telemetry(CoreType::ETH, *core, telemetry);

    const CoreCoord virtual_core = device()->virtual_core_from_logical_core(*core, CoreType::ETH);
    auto actual = read_dispatch_core_telemetry(tt_device(), virtual_core);

    ASSERT_TRUE(actual.has_value());
    EXPECT_EQ(actual->upstream_blocked_count, telemetry.upstream_blocked_count);
    EXPECT_EQ(actual->upstream_unblocked_count, telemetry.upstream_unblocked_count);
    EXPECT_EQ(actual->program_count, telemetry.program_count);
}

TEST_F(DispatchTelemetryReadApiTest, ReadDispatchCoreTelemetryRejectsBadSignature) {
    const CoreCoord core{0, 0};
    dispatch_telemetry_types::DispatchCoreTelemetry telemetry;
    telemetry.signature = dispatch_telemetry_types::INVALID_TELEMETRY_SIGNATURE;
    write_telemetry(CoreType::WORKER, core, telemetry);

    const CoreCoord virtual_core = device()->virtual_core_from_logical_core(core, CoreType::WORKER);
    auto actual = read_dispatch_core_telemetry(tt_device(), virtual_core);

    EXPECT_FALSE(actual.has_value());
}

TEST_F(DispatchTelemetryReadApiTest, ReadDispatchCoreTelemetryRejectsBadVersion) {
    const CoreCoord core{0, 0};
    dispatch_telemetry_types::DispatchCoreTelemetry telemetry;
    telemetry.version = dispatch_telemetry_types::DISPATCH_TELEMETRY_VERSION + 1;
    write_telemetry(CoreType::WORKER, core, telemetry);

    const CoreCoord virtual_core = device()->virtual_core_from_logical_core(core, CoreType::WORKER);
    auto actual = read_dispatch_core_telemetry(tt_device(), virtual_core);

    EXPECT_FALSE(actual.has_value());
}

TEST_F(DispatchTelemetryReadApiTest, ReadPrefetchTelemetryFromWorkerL1) {
    const CoreCoord core{0, 0};
    dispatch_telemetry_types::PrefetchCoreTelemetry telemetry;
    telemetry.upstream_blocked_count = 23;
    telemetry.upstream_unblocked_count = 29;
    telemetry.command_count = 31;
    write_telemetry(CoreType::WORKER, core, telemetry);

    const CoreCoord virtual_core = device()->virtual_core_from_logical_core(core, CoreType::WORKER);
    auto actual = read_prefetch_core_telemetry(tt_device(), virtual_core);

    ASSERT_TRUE(actual.has_value());
    EXPECT_EQ(actual->upstream_blocked_count, telemetry.upstream_blocked_count);
    EXPECT_EQ(actual->upstream_unblocked_count, telemetry.upstream_unblocked_count);
    EXPECT_EQ(actual->command_count, telemetry.command_count);
}

TEST_F(DispatchTelemetryReadApiTest, ReadPrefetchTelemetryFromEthL1) {
    const auto core = available_ethernet_core_for_l1_test();
    if (!core.has_value()) {
        GTEST_SKIP() << "No ethernet cores available for L1 telemetry test";
    }

    dispatch_telemetry_types::PrefetchCoreTelemetry telemetry;
    telemetry.upstream_blocked_count = 17;
    telemetry.upstream_unblocked_count = 19;
    telemetry.command_count = 21;
    write_telemetry(CoreType::ETH, *core, telemetry);

    const CoreCoord virtual_core = device()->virtual_core_from_logical_core(*core, CoreType::ETH);
    auto actual = read_prefetch_core_telemetry(tt_device(), virtual_core);

    ASSERT_TRUE(actual.has_value());
    EXPECT_EQ(actual->upstream_blocked_count, telemetry.upstream_blocked_count);
    EXPECT_EQ(actual->upstream_unblocked_count, telemetry.upstream_unblocked_count);
    EXPECT_EQ(actual->command_count, telemetry.command_count);
}

TEST_F(DispatchTelemetryReadApiTest, ReadPrefetchTelemetryRejectsBadSignature) {
    const CoreCoord core{0, 0};
    dispatch_telemetry_types::PrefetchCoreTelemetry telemetry;
    telemetry.signature = dispatch_telemetry_types::INVALID_TELEMETRY_SIGNATURE;
    write_telemetry(CoreType::WORKER, core, telemetry);

    const CoreCoord virtual_core = device()->virtual_core_from_logical_core(core, CoreType::WORKER);
    auto actual = read_prefetch_core_telemetry(tt_device(), virtual_core);

    EXPECT_FALSE(actual.has_value());
}

TEST_F(DispatchTelemetryReadApiTest, ReadPrefetchTelemetryRejectsBadVersion) {
    const CoreCoord core{0, 0};
    dispatch_telemetry_types::PrefetchCoreTelemetry telemetry;
    telemetry.version = dispatch_telemetry_types::DISPATCH_TELEMETRY_VERSION + 1;
    write_telemetry(CoreType::WORKER, core, telemetry);

    const CoreCoord virtual_core = device()->virtual_core_from_logical_core(core, CoreType::WORKER);
    auto actual = read_prefetch_core_telemetry(tt_device(), virtual_core);

    EXPECT_FALSE(actual.has_value());
}

TEST_F(DispatchTelemetrySlowDispatchTest, ReadInfoReturnsNulloptButVersionIsValid) {
    DispatchTelemetry telemetry(tt_device());

    auto info = telemetry.read_info();
    EXPECT_FALSE(info.has_value());

    EXPECT_EQ(telemetry.version(), dispatch_telemetry_types::DISPATCH_TELEMETRY_VERSION);
}

TEST_F(DispatchTelemetryReadApiTest, DispatchCoreProgramCount) {
    auto& cq = devices_.at(0)->mesh_command_queue();
    constexpr size_t total_runs = 10;
    constexpr size_t num_blank_programs = 16;

    // Verify it increments per program regardless of core count
    const CoreRangeSet worker_cores{CoreRange(CoreCoord{0, 0}, CoreCoord{1, 1})};

    DispatchTelemetry telemetry(tt_device());
    auto initial = telemetry.read_info();
    ASSERT_TRUE(initial.has_value());
    ASSERT_FALSE(initial->info_cqs.empty());

    // Check a single program run
    distributed::MeshWorkload workload;
    workload.add_program(device_range_, create_blank_program(worker_cores));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    Finish(cq);

    auto after_one = telemetry.read_info();
    ASSERT_TRUE(after_one.has_value());
    ASSERT_FALSE(after_one->info_cqs.empty());
    EXPECT_EQ(after_one->info_cqs.front().program_count_since_last_read, 1);

    // Check multiple runs to ensure the delta is calculated correctly
    for (size_t run = 0; run < total_runs; ++run) {
        for (size_t i = 0; i < num_blank_programs; ++i) {
            distributed::MeshWorkload workload;
            workload.add_program(device_range_, create_blank_program(worker_cores));
            distributed::EnqueueMeshWorkload(cq, workload, false);
        }
        Finish(cq);

        auto after_many = telemetry.read_info();
        ASSERT_TRUE(after_many.has_value());
        ASSERT_FALSE(after_many->info_cqs.empty());
        EXPECT_EQ(after_many->info_cqs.front().program_count_since_last_read, num_blank_programs);
    }
}

TEST_F(DispatchTelemetryReadApiTest, DispatchCoreProgramCountForTraceReplay) {
    if (!MetalContext::instance().get_dispatch_query_manager().dispatch_s_enabled()) {
        GTEST_SKIP() << "Requires dispatch_s to be enabled";
    }

    auto mesh_device = devices_.at(0);
    auto& cq = mesh_device->mesh_command_queue();
    int num_programs = 0;
    const CoreRangeSet worker_cores{CoreRange(CoreCoord{0, 0}, CoreCoord{1, 1})};

    DispatchTelemetry telemetry(tt_device());
    auto initial = telemetry.read_info();
    ASSERT_TRUE(initial.has_value());
    ASSERT_FALSE(initial->info_cqs.empty());

    distributed::MeshWorkload workload;
    workload.add_program(device_range_, create_blank_program(worker_cores));

    // Load binaries and generate dispatch commands before trace capture; capture does not allow writes.
    distributed::EnqueueMeshWorkload(cq, workload, true);
    num_programs++;

    constexpr uint32_t num_traced_programs = 16;
    auto trace_id = distributed::BeginTraceCapture(mesh_device.get(), cq.id());
    for (uint32_t i = 0; i < num_traced_programs; ++i) {
        distributed::EnqueueMeshWorkload(cq, workload, false);
        num_programs++;
    }
    mesh_device->end_mesh_trace(cq.id(), trace_id);

    auto trace = mesh_device->get_mesh_trace(trace_id);
    ASSERT_NE(trace, nullptr);
    ASSERT_NE(trace->desc, nullptr);
    ASSERT_EQ(trace->desc->descriptors.size(), 1);
    const auto& trace_descriptor = trace->desc->descriptors.begin()->second;
    ASSERT_EQ(trace_descriptor.num_traced_programs_needing_go_signal_multicast, num_traced_programs);

    mesh_device->replay_mesh_trace(cq.id(), trace_id, true);

    auto info = telemetry.read_info();
    ASSERT_TRUE(info.has_value());
    ASSERT_FALSE(info->info_cqs.empty());
    uint32_t program_count = info->info_cqs.front().program_count_since_last_read;

    mesh_device->release_mesh_trace(trace_id);

    // Trace replay contributes one dispatch_s notification for the trace replay wrapper plus one
    // notification for each traced program command in the replayed exec buffer.
    EXPECT_EQ(program_count, num_programs);
}

TEST_F(DispatchTelemetryReadApiTest, PrefetchCommandCountIncrementsAfterProgramRuns) {
    auto& cq = devices_.at(0)->mesh_command_queue();
    constexpr size_t num_blank_programs = 4;
    const CoreRangeSet worker_cores{CoreRange(CoreCoord{0, 0}, CoreCoord{1, 1})};

    DispatchTelemetry telemetry(tt_device());
    auto initial = telemetry.read_info();
    ASSERT_TRUE(initial.has_value());
    ASSERT_FALSE(initial->info_cqs.empty());

    for (size_t i = 0; i < num_blank_programs; ++i) {
        distributed::MeshWorkload workload;
        workload.add_program(device_range_, create_blank_program(worker_cores));
        distributed::EnqueueMeshWorkload(cq, workload, false);
    }
    Finish(cq);

    auto after = telemetry.read_info();
    ASSERT_TRUE(after.has_value());
    ASSERT_FALSE(after->info_cqs.empty());
    EXPECT_GE(after->info_cqs.front().prefetch_command_count_since_last_read, num_blank_programs);
}

TEST_F(DispatchTelemetryReadApiTest, DispatchCoreEfficiencyIsNulloptWhenWorkerDispatchDisabled) {
    if (worker_dispatch_enabled()) {
        GTEST_SKIP() << "Requires worker dispatch to be disabled";
    }

    // Create the device, single-core stream, blank program, and telemetry object.
    auto& cq = devices_.at(0)->mesh_command_queue();
    const CoreRangeSet worker_core{CoreRange(CoreCoord{0, 0})};
    Program program = create_blank_program(worker_core);
    DispatchTelemetry telemetry(tt_device());

    // Run the blank program.
    distributed::MeshWorkload workload;
    workload.add_program(device_range_, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    Finish(cq);

    // Read telemetry info and verify core efficiency does not have a value.
    auto info = telemetry.read_info();
    ASSERT_TRUE(info.has_value());
    EXPECT_FALSE(info->device_core_efficiency_since_last_read.has_value());
}

TEST_F(DispatchTelemetryReadApiTest, DispatchUtilizationIsEmptyWhenWorkerDispatchDisabled) {
    if (worker_dispatch_enabled()) {
        GTEST_SKIP() << "Requires worker dispatch to be disabled";
    }

    // Create the device, single-core stream, blank program, and telemetry object.
    auto& cq = devices_.at(0)->mesh_command_queue();
    const CoreRangeSet worker_core{CoreRange(CoreCoord{0, 0})};
    Program program = create_blank_program(worker_core);
    DispatchTelemetry telemetry(tt_device());

    // Run the blank program.
    distributed::MeshWorkload workload;
    workload.add_program(device_range_, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    Finish(cq);

    // Read telemetry info and verify utilization is not available.
    auto info = telemetry.read_info();
    ASSERT_TRUE(info.has_value());
    for (const auto& cq_info : info->info_cqs) {
        EXPECT_FALSE(cq_info.utilization_since_last_read.has_value());
    }
}

TEST_F(DispatchTelemetryHostL1WaitTest, WorkerWaitReportsUpstreamBlockedState) {
    IDevice* device = this->device();
    auto& cq = devices_.at(0)->mesh_command_queue();
    const CoreCoord worker_core{0, 0};
    reset_worker_l1_state(worker_core);

    DispatchTelemetry telemetry(tt_device());
    auto initial = telemetry.read_info();
    ASSERT_TRUE(initial.has_value());
    ASSERT_FALSE(initial->info_cqs.empty());

    // Enqueue kernel that waits on host write to complete
    distributed::MeshWorkload waiting_workload;
    waiting_workload.add_program(device_range_, create_l1_wait_program(worker_core));
    distributed::EnqueueMeshWorkload(cq, waiting_workload, false);
    EXPECT_TRUE(worker_reached_l1_wait(device, worker_core));

    // Worker has the host blocked kernel in progress
    // Dispatch queue is empty
    // Prefetch queue is empty
    auto while_waiting = telemetry.read_info();
    EXPECT_TRUE(while_waiting.has_value());
    EXPECT_FALSE(while_waiting.has_value() && while_waiting->info_cqs.empty());
    if (while_waiting.has_value() && !while_waiting->info_cqs.empty()) {
        // Prefetch is waiting on upstream host for the next workload
        EXPECT_TRUE(while_waiting->info_cqs.front().prefetch_waiting_on_upstream);
        // Dispatch is waiting on upstream prefetch for the next workload
        EXPECT_TRUE(while_waiting->info_cqs.front().dispatch_waiting_on_upstream);
    }

    constexpr size_t num_blank_programs = 4;
    for (size_t i = 0; i < num_blank_programs; ++i) {
        distributed::MeshWorkload workload;
        workload.add_program(device_range_, create_blank_program(worker_core));
        distributed::EnqueueMeshWorkload(cq, workload, false);
    }

    // Worker still has the host blocked kernel in progress
    // Prefetch is stalled waiting on downstream sync
    // Dispatch is waiting for worker progress
    auto after_enqueue = telemetry.read_info();
    EXPECT_TRUE(after_enqueue.has_value());
    EXPECT_FALSE(after_enqueue.has_value() && after_enqueue->info_cqs.empty());
    if (after_enqueue.has_value() && !after_enqueue->info_cqs.empty()) {
        // Prefetch is no longer waiting on upstream host
        EXPECT_FALSE(after_enqueue->info_cqs.front().prefetch_waiting_on_upstream);
        // Dispatch is no longer waiting on upstream prefetch
        EXPECT_FALSE(after_enqueue->info_cqs.front().dispatch_waiting_on_upstream);
    }

    release_core_and_finish(worker_core);

    // Worker is no longer blocked, it has finished processing all enqueued work loads
    // Dispatch queue is empty
    // Prefetch queue is empty
    auto after_finish = telemetry.read_info();
    EXPECT_TRUE(after_finish.has_value());
    EXPECT_FALSE(after_finish.has_value() && after_finish->info_cqs.empty());
    if (after_finish.has_value() && !after_finish->info_cqs.empty()) {
        // Prefetch is waiting on upstream host for the next workload
        EXPECT_TRUE(after_finish->info_cqs.front().prefetch_waiting_on_upstream);
        // Dispatch is waiting on upstream prefetch for the next workload
        EXPECT_TRUE(after_finish->info_cqs.front().dispatch_waiting_on_upstream);
    }
}

TEST_F(DispatchTelemetryReadApiTest, DispatchSTelemetryCurrentTimestampAdvances) {
    const auto dispatch_s_core = dispatch_s_virtual_core();
    if (!worker_dispatch_enabled() || !dispatch_s_core.has_value()) {
        GTEST_SKIP() << "Requires worker dispatch and dispatch_s to be enabled";
    }

    auto first = read_dispatch_core_telemetry(tt_device(), *dispatch_s_core);
    ASSERT_TRUE(first.has_value());

    auto second = read_dispatch_core_telemetry(tt_device(), *dispatch_s_core);
    ASSERT_TRUE(second.has_value());
    EXPECT_GT(second->current_timestamp, first->current_timestamp);
}

TEST_F(DispatchTelemetryHostL1WaitTest, DispatchSTelemetryTracksWorkerRuntime) {
    const auto dispatch_s_core = dispatch_s_virtual_core();
    if (!worker_dispatch_enabled() || !dispatch_s_core.has_value()) {
        GTEST_SKIP() << "Requires worker dispatch and dispatch_s to be enabled";
    }

    auto& cq = devices_.at(0)->mesh_command_queue();
    const CoreCoord worker_core{0, 0};
    reset_worker_l1_state(worker_core);

    distributed::MeshWorkload waiting_workload;
    waiting_workload.add_program(device_range_, create_l1_wait_program(worker_core));
    distributed::EnqueueMeshWorkload(cq, waiting_workload, false);
    const bool worker_started = worker_reached_l1_wait(device(), worker_core);
    if (!worker_started) {
        release_core_and_finish(worker_core);
    }
    ASSERT_TRUE(worker_started);

    EXPECT_TRUE(worker_reached_l1_wait(device(), worker_core));
    {
        auto info = read_dispatch_core_telemetry(tt_device(), *dispatch_s_core);
        ASSERT_TRUE(info.has_value());
        EXPECT_GT(info->workers_per_sub_device[0], 0);
        EXPECT_GT(info->current_timestamp, 0);
        EXPECT_GT(info->last_work_launch_timestamp[0], 0);
    }

    release_core_and_finish(worker_core);

    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    {
        auto info = read_dispatch_core_telemetry(tt_device(), *dispatch_s_core);
        ASSERT_TRUE(info.has_value());
        EXPECT_GT(info->workers_per_sub_device[0], 0);
        EXPECT_GT(info->current_timestamp, 0);
        EXPECT_GT(info->last_work_launch_timestamp[0], 0);
        EXPECT_GE(info->current_timestamp, info->last_work_launch_timestamp[0]);
    }
}

TEST_F(DispatchTelemetryReadApiTest, DispatchTelemetryTracksWorkersPerSubDeviceCount) {
    constexpr uint8_t cq_id = 0;
    auto mesh_device = devices_.at(0);
    const auto dispatch_telemetry_core = dispatch_telemetry_virtual_core(cq_id);
    ASSERT_TRUE(dispatch_telemetry_core.has_value());

    const CoreRangeSet all_worker_cores = [&] {
        const CoreCoord grid_size = device()->compute_with_storage_grid_size();
        return CoreRangeSet(CoreRange(CoreCoord{0, 0}, CoreCoord{grid_size.x - 1, grid_size.y - 1}));
    }();

    constexpr size_t num_sub_devices = 2;
    constexpr size_t first_sub_device_core_count = 2;
    constexpr size_t second_sub_device_core_count = 3;
    constexpr std::array<size_t, num_sub_devices> sub_device_core_counts = {
        first_sub_device_core_count, second_sub_device_core_count};

    constexpr size_t required_worker_core_count = first_sub_device_core_count + second_sub_device_core_count;
    if (all_worker_cores.num_cores() < required_worker_core_count) {
        GTEST_SKIP() << "Not enough worker cores";
    }

    const std::array<CoreRangeSet, num_sub_devices> sub_device_cores = {
        select_from_corerangeset(all_worker_cores, 0, 1, true), select_from_corerangeset(all_worker_cores, 2, 4, true)};

    std::array<SubDevice, num_sub_devices> sub_devices = {
        SubDevice(std::array{sub_device_cores[0]}), SubDevice(std::array{sub_device_cores[1]})};
    auto sub_device_manager = mesh_device->create_sub_device_manager({sub_devices[0], sub_devices[1]}, 3200);
    mesh_device->load_sub_device_manager(sub_device_manager);

    auto info = read_dispatch_core_telemetry(tt_device(), *dispatch_telemetry_core);
    EXPECT_TRUE(info.has_value());
    if (info.has_value()) {
        for (size_t sub_device_index = 0; sub_device_index < num_sub_devices; ++sub_device_index) {
            EXPECT_EQ(info->workers_per_sub_device[sub_device_index], sub_device_core_counts[sub_device_index])
                << "sub_device_index=" << sub_device_index;
        }
    }

    mesh_device->clear_loaded_sub_device_manager();
    mesh_device->remove_sub_device_manager(sub_device_manager);
}

TEST_F(DispatchTelemetryMultiCQReadApiTest, DispatchTelemetryTracksWorkersPerSubDeviceCount) {
    constexpr uint8_t first_cq_id = 0;
    constexpr uint8_t second_cq_id = 1;
    auto mesh_device = device_;
    if (mesh_device->num_hw_cqs() < 2) {
        GTEST_SKIP() << "Requires at least two hardware command queues";
    }
    const auto first_dispatch_telemetry_core = dispatch_telemetry_virtual_core(first_cq_id);
    const auto second_dispatch_telemetry_core = dispatch_telemetry_virtual_core(second_cq_id);
    ASSERT_TRUE(first_dispatch_telemetry_core.has_value() && second_dispatch_telemetry_core.has_value());

    const CoreRangeSet all_worker_cores = [&] {
        const CoreCoord grid_size = device()->compute_with_storage_grid_size();
        return CoreRangeSet(CoreRange(CoreCoord{0, 0}, CoreCoord{grid_size.x - 1, grid_size.y - 1}));
    }();

    constexpr size_t num_sub_devices = 4;
    constexpr size_t cq_1_first_sub_device_index = 0;
    constexpr size_t cq_1_second_sub_device_index = 1;
    constexpr size_t cq_2_first_sub_device_index = 2;
    constexpr size_t cq_2_second_sub_device_index = 3;
    constexpr size_t cq_1_first_sub_device_core_count = 2;
    constexpr size_t cq_1_second_sub_device_core_count = 3;
    constexpr size_t cq_2_first_sub_device_core_count = 3;
    constexpr size_t cq_2_second_sub_device_core_count = 4;
    constexpr std::array<size_t, num_sub_devices> sub_device_core_counts = {
        cq_1_first_sub_device_core_count,
        cq_1_second_sub_device_core_count,
        cq_2_first_sub_device_core_count,
        cq_2_second_sub_device_core_count};

    constexpr size_t required_worker_core_count = cq_1_first_sub_device_core_count + cq_1_second_sub_device_core_count +
                                                  cq_2_first_sub_device_core_count + cq_2_second_sub_device_core_count;
    if (all_worker_cores.num_cores() < required_worker_core_count) {
        GTEST_SKIP() << "Not enough worker cores";
    }

    const std::array<CoreRangeSet, num_sub_devices> sub_device_cores = {
        select_from_corerangeset(all_worker_cores, 0, 1, true),
        select_from_corerangeset(all_worker_cores, 2, 4, true),
        select_from_corerangeset(all_worker_cores, 5, 7, true),
        select_from_corerangeset(all_worker_cores, 8, 11, true)};

    std::array<SubDevice, num_sub_devices> sub_devices = {
        SubDevice(std::array{sub_device_cores[cq_1_first_sub_device_index]}),
        SubDevice(std::array{sub_device_cores[cq_1_second_sub_device_index]}),
        SubDevice(std::array{sub_device_cores[cq_2_first_sub_device_index]}),
        SubDevice(std::array{sub_device_cores[cq_2_second_sub_device_index]})};
    auto sub_device_manager = mesh_device->create_sub_device_manager(
        {sub_devices[cq_1_first_sub_device_index],
         sub_devices[cq_1_second_sub_device_index],
         sub_devices[cq_2_first_sub_device_index],
         sub_devices[cq_2_second_sub_device_index]},
        3200);
    mesh_device->load_sub_device_manager(sub_device_manager);

    auto first_info = read_dispatch_core_telemetry(tt_device(), *first_dispatch_telemetry_core);
    EXPECT_TRUE(first_info.has_value());
    if (first_info.has_value()) {
        EXPECT_EQ(
            first_info->workers_per_sub_device[cq_1_first_sub_device_index],
            sub_device_core_counts[cq_1_first_sub_device_index]);
        EXPECT_EQ(
            first_info->workers_per_sub_device[cq_1_second_sub_device_index],
            sub_device_core_counts[cq_1_second_sub_device_index]);
    }

    auto second_info = read_dispatch_core_telemetry(tt_device(), *second_dispatch_telemetry_core);
    EXPECT_TRUE(second_info.has_value());
    if (second_info.has_value()) {
        EXPECT_EQ(
            second_info->workers_per_sub_device[cq_2_first_sub_device_index],
            sub_device_core_counts[cq_2_first_sub_device_index]);
        EXPECT_EQ(
            second_info->workers_per_sub_device[cq_2_second_sub_device_index],
            sub_device_core_counts[cq_2_second_sub_device_index]);
    }

    mesh_device->clear_loaded_sub_device_manager();
    mesh_device->remove_sub_device_manager(sub_device_manager);
}

TEST_F(DispatchTelemetryHostL1WaitTest, DispatchSTelemetryDoesNotOvercountCompletionsOnStreamReset) {
    const auto dispatch_s_core = dispatch_s_virtual_core();
    if (!worker_dispatch_enabled() || !dispatch_s_core.has_value()) {
        GTEST_SKIP() << "Requires dispatch_s to be enabled";
    }

    auto& cq = devices_.at(0)->mesh_command_queue();
    auto mesh_device = devices_.at(0);
    const CoreCoord worker_core{0, 0};

    const auto completion_counts_are_bounded = [](const dispatch_telemetry_types::DispatchCoreTelemetry& telemetry) {
        bool has_active_sub_device = false;
        for (size_t i = 0; i < dispatch_telemetry_types::MAX_SUB_DEVICES; ++i) {
            if (telemetry.workers_per_sub_device[i] == 0) {
                continue;
            }
            has_active_sub_device = true;
            if (telemetry.completion_count[i] > telemetry.workers_per_sub_device[i]) {
                return false;
            }
        }
        return has_active_sub_device;
    };

    const auto start_waiting_workload = [&]() -> std::optional<uint32_t> {
        distributed::MeshWorkload waiting_workload;
        waiting_workload.add_program(device_range_, create_l1_wait_program(worker_core));
        distributed::EnqueueMeshWorkload(cq, waiting_workload, false);
        const bool worker_started = worker_reached_l1_wait(device(), worker_core);
        if (!worker_started) {
            release_core_and_finish(worker_core);
        }
        if (!worker_started) {
            ADD_FAILURE() << "Worker did not start waiting on host L1 write";
            return std::nullopt;
        }

        auto while_working = read_dispatch_core_telemetry(tt_device(), dispatch_s_core.value());
        if (!while_working.has_value() || while_working->workers_per_sub_device[0] == 0 ||
            while_working->completion_count[0] >= while_working->workers_per_sub_device[0] ||
            while_working->last_work_launch_timestamp[0] == 0) {
            ADD_FAILURE() << "Telemetry did not report in-progress worker runtime";
            release_core_and_finish(worker_core);
            return std::nullopt;
        }

        const uint32_t worker_count = while_working->workers_per_sub_device[0];
        return worker_count;
    };

    reset_worker_l1_state(worker_core);
    const auto first_worker_count = start_waiting_workload();
    ASSERT_TRUE(first_worker_count.has_value());
    release_worker(worker_core);

    const CoreCoord grid_size = device()->compute_with_storage_grid_size();
    SubDevice full_grid_sub_device(
        std::array{CoreRangeSet(CoreRange(CoreCoord{0, 0}, CoreCoord{grid_size.x - 1, grid_size.y - 1}))});
    auto sub_device_manager = mesh_device->create_sub_device_manager({full_grid_sub_device}, 3200);
    mesh_device->load_sub_device_manager(sub_device_manager);
    Finish(cq);

    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    auto after_reset = read_dispatch_core_telemetry(tt_device(), dispatch_s_core.value());
    ASSERT_TRUE(after_reset.has_value()) << "Telemetry was not readable after reset";
    ASSERT_TRUE(completion_counts_are_bounded(*after_reset))
        << "Telemetry completion count exceeded worker semaphore count after reset";

    reset_worker_l1_state(worker_core);
    const auto second_worker_count = start_waiting_workload();
    ASSERT_TRUE(second_worker_count.has_value());

    release_worker(worker_core);
    Finish(cq);

    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    auto after_second_batch = read_dispatch_core_telemetry(tt_device(), dispatch_s_core.value());
    ASSERT_TRUE(after_second_batch.has_value()) << "Telemetry was not readable after the second batch of work";
    ASSERT_TRUE(completion_counts_are_bounded(*after_second_batch))
        << "Telemetry completion count exceeded worker semaphore count after the second batch of work";
    mesh_device->clear_loaded_sub_device_manager();
    mesh_device->remove_sub_device_manager(sub_device_manager);
}

TEST_F(DispatchTelemetryHostL1WaitTest, DispatchSTelemetryTracksMultipleSubDevices) {
    const auto dispatch_s_core = dispatch_s_virtual_core();
    if (!worker_dispatch_enabled() || !dispatch_s_core.has_value()) {
        GTEST_SKIP() << "Requires worker dispatch and dispatch_s to be enabled";
    }

    auto mesh_device = devices_.at(0);
    auto& cq = mesh_device->mesh_command_queue();
    const CoreCoord grid_size = device()->compute_with_storage_grid_size();
    if (grid_size.x * grid_size.y < 2) {
        GTEST_SKIP() << "Requires at least two worker cores";
    }

    const CoreCoord first_worker{0, 0};
    const CoreCoord second_worker = (grid_size.x > 1) ? CoreCoord{1, 0} : CoreCoord{0, 1};
    SubDevice first_sub_device(std::array{CoreRangeSet(CoreRange(first_worker, first_worker))});
    SubDevice second_sub_device(std::array{CoreRangeSet(CoreRange(second_worker, second_worker))});
    auto sub_device_manager = mesh_device->create_sub_device_manager({first_sub_device, second_sub_device}, 3200);
    mesh_device->load_sub_device_manager(sub_device_manager);

    reset_worker_l1_state(first_worker);
    reset_worker_l1_state(second_worker);

    distributed::MeshWorkload first_workload;
    first_workload.add_program(device_range_, create_l1_wait_program(first_worker));
    distributed::EnqueueMeshWorkload(cq, first_workload, false);

    const bool first_started = worker_reached_l1_wait(device(), first_worker);
    if (!first_started) {
        release_worker(first_worker);
        Finish(cq);
    }
    ASSERT_TRUE(first_started);

    auto first_while_working = read_dispatch_core_telemetry(tt_device(), dispatch_s_core.value());

    release_worker(first_worker);
    Finish(cq);

    ASSERT_TRUE(first_while_working.has_value());
    EXPECT_GT(first_while_working->workers_per_sub_device[0], 0);
    EXPECT_LT(first_while_working->completion_count[0], first_while_working->workers_per_sub_device[0]);
    EXPECT_GT(first_while_working->last_work_launch_timestamp[0], 0);

    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    auto after_first_finish = read_dispatch_core_telemetry(tt_device(), dispatch_s_core.value());
    ASSERT_TRUE(after_first_finish.has_value());
    EXPECT_GT(after_first_finish->workers_per_sub_device[0], 0);
    EXPECT_EQ(after_first_finish->completion_count[0], after_first_finish->workers_per_sub_device[0]);
    EXPECT_GT(after_first_finish->current_sub_device_work_runtime[0], 0);

    distributed::MeshWorkload second_workload;
    second_workload.add_program(device_range_, create_l1_wait_program(second_worker));
    distributed::EnqueueMeshWorkload(cq, second_workload, false);

    const bool second_started = worker_reached_l1_wait(device(), second_worker);
    if (!second_started) {
        release_worker(second_worker);
        Finish(cq);
    }
    ASSERT_TRUE(second_started);

    auto second_while_working = read_dispatch_core_telemetry(tt_device(), dispatch_s_core.value());

    release_worker(second_worker);
    Finish(cq);

    ASSERT_TRUE(second_while_working.has_value());
    EXPECT_GT(second_while_working->workers_per_sub_device[1], 0);
    EXPECT_LT(second_while_working->completion_count[1], second_while_working->workers_per_sub_device[1]);
    EXPECT_GT(second_while_working->last_work_launch_timestamp[1], 0);

    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    auto after_second_finish = read_dispatch_core_telemetry(tt_device(), dispatch_s_core.value());
    ASSERT_TRUE(after_second_finish.has_value());
    EXPECT_GT(after_second_finish->workers_per_sub_device[0], 0);
    EXPECT_GT(after_second_finish->workers_per_sub_device[1], 0);
    EXPECT_EQ(after_second_finish->completion_count[0], after_second_finish->workers_per_sub_device[0]);
    EXPECT_EQ(after_second_finish->completion_count[1], after_second_finish->workers_per_sub_device[1]);
    EXPECT_GT(after_second_finish->current_sub_device_work_runtime[0], 0);
    EXPECT_GT(after_second_finish->current_sub_device_work_runtime[1], 0);
}

TEST_F(DispatchTelemetryReadApiTest, LastWorkLaunchTimestampIncrementsPerSubDevice) {
    const auto dispatch_s_core = dispatch_s_virtual_core();
    if (!dispatch_s_core.has_value()) {
        GTEST_SKIP() << "Requires dispatch_s to be enabled";
    }

    auto mesh_device = devices_.at(0);
    auto& cq = mesh_device->mesh_command_queue();
    const CoreRangeSet all_worker_cores = [&] {
        const CoreCoord grid_size = device()->compute_with_storage_grid_size();
        return CoreRangeSet(CoreRange(CoreCoord{0, 0}, CoreCoord{grid_size.x - 1, grid_size.y - 1}));
    }();
    constexpr size_t num_sub_devices = 3;
    constexpr size_t sub_device_core_count = 2;
    if (all_worker_cores.num_cores() < num_sub_devices * sub_device_core_count) {
        GTEST_SKIP() << "Requires at least six worker cores";
    }

    // Init 3 sub devices of 2 cores each.
    const std::array<CoreRangeSet, num_sub_devices> sub_device_cores = {
        select_from_corerangeset(all_worker_cores, 0, 1, true),
        select_from_corerangeset(all_worker_cores, 2, 3, true),
        select_from_corerangeset(all_worker_cores, 4, 5, true)};
    std::array<SubDevice, num_sub_devices> sub_devices = {
        SubDevice(std::array{sub_device_cores[0]}),
        SubDevice(std::array{sub_device_cores[1]}),
        SubDevice(std::array{sub_device_cores[2]})};
    auto sub_device_manager =
        mesh_device->create_sub_device_manager({sub_devices[0], sub_devices[1], sub_devices[2]}, 3200);
    mesh_device->load_sub_device_manager(sub_device_manager);
    Finish(cq);

    const std::array<CoreRangeSet, num_sub_devices> loaded_sub_device_cores = {
        mesh_device->worker_cores(HalProgrammableCoreType::TENSIX, SubDeviceId{0}),
        mesh_device->worker_cores(HalProgrammableCoreType::TENSIX, SubDeviceId{1}),
        mesh_device->worker_cores(HalProgrammableCoreType::TENSIX, SubDeviceId{2})};
    for (size_t sub_device_index = 0; sub_device_index < num_sub_devices; ++sub_device_index) {
        ASSERT_EQ(loaded_sub_device_cores[sub_device_index].num_cores(), sub_device_core_count);
    }

    const auto read_timestamps = [&]() {
        auto info = read_dispatch_core_telemetry(tt_device(), *dispatch_s_core);
        EXPECT_TRUE(info.has_value());
        std::array<uint64_t, num_sub_devices> timestamps = {};
        if (info.has_value()) {
            for (size_t sub_device_index = 0; sub_device_index < num_sub_devices; ++sub_device_index) {
                timestamps[sub_device_index] = info->last_work_launch_timestamp[sub_device_index];
            }
        }
        return timestamps;
    };

    const auto run_blank_program = [&](const CoreRangeSet& program_cores) {
        distributed::MeshWorkload workload;
        workload.add_program(device_range_, create_blank_program(program_cores));
        distributed::EnqueueMeshWorkload(cq, workload, false);
        Finish(cq);
    };

    // Create a blank program.
    std::array<CoreRangeSet, num_sub_devices> one_core_per_sub_device = {
        select_from_corerangeset(loaded_sub_device_cores[0], 0, 0, true),
        select_from_corerangeset(loaded_sub_device_cores[1], 0, 0, true),
        select_from_corerangeset(loaded_sub_device_cores[2], 0, 0, true)};
    const CoreRangeSet one_core_on_first_sub_device = select_from_corerangeset(loaded_sub_device_cores[0], 0, 0, true);

    // Check telemetry to cache the initial last_work_launch_timestamp for all.
    auto timestamps = read_timestamps();

    // Run the blank program on all sub devices and cores, let it finish.
    for (const CoreRangeSet& sub_device_core : loaded_sub_device_cores) {
        run_blank_program(sub_device_core);
    }

    // Verify last_work_launch_timestamp incremented on all.
    auto after_all_cores = read_timestamps();
    for (size_t sub_device_index = 0; sub_device_index < num_sub_devices; ++sub_device_index) {
        EXPECT_GT(after_all_cores[sub_device_index], timestamps[sub_device_index])
            << "sub_device_index=" << sub_device_index;
    }

    // Run the blank program on 1 core for each sub device, let it finish.
    for (const CoreRangeSet& sub_device_core : one_core_per_sub_device) {
        run_blank_program(sub_device_core);
    }

    // Verify last_work_launch_timestamp incremented on all.
    auto after_one_core_each = read_timestamps();
    for (size_t sub_device_index = 0; sub_device_index < num_sub_devices; ++sub_device_index) {
        EXPECT_GT(after_one_core_each[sub_device_index], after_all_cores[sub_device_index])
            << "sub_device_index=" << sub_device_index;
    }

    // Run the blank program on only one sub devices.
    run_blank_program(one_core_on_first_sub_device);

    // Verify last_work_launch_timestamp only incremented on the single sub device that ran.
    auto after_one_sub_device = read_timestamps();
    EXPECT_GT(after_one_sub_device[0], after_one_core_each[0]);
    EXPECT_EQ(after_one_sub_device[1], after_one_core_each[1]);
    EXPECT_EQ(after_one_sub_device[2], after_one_core_each[2]);

    mesh_device->clear_loaded_sub_device_manager();
    mesh_device->remove_sub_device_manager(sub_device_manager);
}

// Verifies the assumption dispatch telemetry makes when counting worker completions
TEST_F(DispatchTelemetryReadApiTest, InactiveWorkersIncrementCompleteSem) {
    const auto dispatch_s_core = dispatch_s_virtual_core();
    if (!worker_dispatch_enabled() || !dispatch_s_core.has_value()) {
        GTEST_SKIP() << "Requires worker dispatch and dispatch_s to be enabled";
    }

    IDevice* device = this->device();
    auto mesh_device = devices_.at(0);
    auto& cq = mesh_device->mesh_command_queue();
    const CoreRangeSet all_worker_cores = [&] {
        const CoreCoord grid_size = device->compute_with_storage_grid_size();
        return CoreRangeSet(CoreRange(CoreCoord{0, 0}, CoreCoord{grid_size.x - 1, grid_size.y - 1}));
    }();
    const size_t total_worker_core_count = all_worker_cores.num_cores();
    if (total_worker_core_count < 2 || total_worker_core_count % 2 != 0) {
        GTEST_SKIP() << "Requires an even worker core count of at least two";
    }

    // Initialize 2 sub devices, each with a core count of half the total available workers.
    const size_t sub_device_core_count = total_worker_core_count / 2;
    const CoreRangeSet first_sub_device_cores =
        select_from_corerangeset(all_worker_cores, 0, static_cast<uint32_t>(sub_device_core_count - 1), true);
    const CoreRangeSet second_sub_device_cores = select_from_corerangeset(
        all_worker_cores,
        static_cast<uint32_t>(sub_device_core_count),
        static_cast<uint32_t>(total_worker_core_count - 1),
        true);
    SubDevice first_sub_device(std::array{first_sub_device_cores});
    SubDevice second_sub_device(std::array{second_sub_device_cores});
    auto sub_device_manager = mesh_device->create_sub_device_manager({first_sub_device, second_sub_device}, 3200);
    mesh_device->load_sub_device_manager(sub_device_manager);

    const CoreRangeSet loaded_first_sub_device_cores =
        mesh_device->worker_cores(HalProgrammableCoreType::TENSIX, SubDeviceId{0});
    const CoreRangeSet loaded_second_sub_device_cores =
        mesh_device->worker_cores(HalProgrammableCoreType::TENSIX, SubDeviceId{1});
    ASSERT_EQ(loaded_first_sub_device_cores.num_cores(), sub_device_core_count);
    ASSERT_EQ(loaded_second_sub_device_cores.num_cores(), sub_device_core_count);

    const auto expect_completion_count_for_sub_device = [&](size_t sub_device_index, const char* case_name) {
        auto info = read_dispatch_core_telemetry(tt_device(), *dispatch_s_core);
        ASSERT_TRUE(info.has_value());
        EXPECT_EQ(info->workers_per_sub_device[0], sub_device_core_count);
        EXPECT_EQ(info->workers_per_sub_device[1], sub_device_core_count);
        EXPECT_EQ(info->completion_count[sub_device_index], info->workers_per_sub_device[sub_device_index])
            << case_name << " sub_device_index=" << sub_device_index;
    };

    // Create a blank program.
    const auto run_blank_program_and_expect_complete =
        [&](const CoreRangeSet& first_program_cores, const CoreRangeSet& second_program_cores, const char* case_name) {
            distributed::MeshWorkload first_workload;
            first_workload.add_program(device_range_, create_blank_program(first_program_cores));
            distributed::EnqueueMeshWorkload(cq, first_workload, false);
            Finish(cq);
            expect_completion_count_for_sub_device(0, case_name);

            distributed::MeshWorkload second_workload;
            second_workload.add_program(device_range_, create_blank_program(second_program_cores));
            distributed::EnqueueMeshWorkload(cq, second_workload, false);
            Finish(cq);
            expect_completion_count_for_sub_device(1, case_name);
        };

    const size_t half_sub_device_core_count = sub_device_core_count / 2;
    ASSERT_GT(half_sub_device_core_count, 0);
    const CoreRangeSet half_first_sub_device_cores = select_from_corerangeset(
        loaded_first_sub_device_cores, 0, static_cast<uint32_t>(half_sub_device_core_count - 1), true);
    const CoreRangeSet half_second_sub_device_cores = select_from_corerangeset(
        loaded_second_sub_device_cores, 0, static_cast<uint32_t>(half_sub_device_core_count - 1), true);
    const CoreRangeSet one_first_sub_device_core = select_from_corerangeset(loaded_first_sub_device_cores, 0, 0, true);
    const CoreRangeSet one_second_sub_device_core =
        select_from_corerangeset(loaded_second_sub_device_cores, 0, 0, true);

    // Run the program on all cores on each sub device, and verify completion count matches each sub device.
    run_blank_program_and_expect_complete(loaded_first_sub_device_cores, loaded_second_sub_device_cores, "all cores");

    // Run the program on half the cores on each sub device, and verify inactive workers still ack.
    run_blank_program_and_expect_complete(half_first_sub_device_cores, half_second_sub_device_cores, "half cores");

    // Run the program on just one core on each sub device, and verify inactive workers still ack.
    run_blank_program_and_expect_complete(one_first_sub_device_core, one_second_sub_device_core, "one core");

    mesh_device->clear_loaded_sub_device_manager();
    mesh_device->remove_sub_device_manager(sub_device_manager);
}

TEST_F(DispatchTelemetryHostL1WaitTest, DispatchCoreEfficiencyAndUtilization) {
    if (!worker_dispatch_enabled() || !dispatch_s_virtual_core().has_value()) {
        GTEST_SKIP() << "Requires worker dispatch and dispatch_s to be enabled";
    }

    IDevice* device = this->device();
    auto mesh_device = devices_.at(0);
    auto& cq = mesh_device->mesh_command_queue();
    const CoreRangeSet all_worker_cores = [&] {
        const CoreCoord grid_size = device->compute_with_storage_grid_size();
        return CoreRangeSet(CoreRange(CoreCoord{0, 0}, CoreCoord{grid_size.x - 1, grid_size.y - 1}));
    }();
    const size_t total_worker_core_count = all_worker_cores.num_cores();

    // Allocate two sub-devices and verify together they cover every worker core.
    const size_t first_sub_device_core_count = total_worker_core_count / 2;
    ASSERT_GT(first_sub_device_core_count, 0);
    const CoreRangeSet first_sub_device_cores =
        select_from_corerangeset(all_worker_cores, 0, static_cast<uint32_t>(first_sub_device_core_count - 1), true);
    const CoreRangeSet second_sub_device_cores = select_from_corerangeset(
        all_worker_cores,
        static_cast<uint32_t>(first_sub_device_core_count),
        static_cast<uint32_t>(total_worker_core_count - 1),
        true);
    SubDevice first_sub_device(std::array{first_sub_device_cores});
    SubDevice second_sub_device(std::array{second_sub_device_cores});
    auto sub_device_manager = mesh_device->create_sub_device_manager({first_sub_device, second_sub_device}, 3200);
    mesh_device->load_sub_device_manager(sub_device_manager);

    const CoreRangeSet loaded_first_sub_device_cores =
        mesh_device->worker_cores(HalProgrammableCoreType::TENSIX, SubDeviceId{0});
    const CoreRangeSet loaded_second_sub_device_cores =
        mesh_device->worker_cores(HalProgrammableCoreType::TENSIX, SubDeviceId{1});

    DispatchTelemetry telemetry(tt_device());

    EXPECT_EQ(
        loaded_first_sub_device_cores.num_cores() + loaded_second_sub_device_cores.num_cores(),
        total_worker_core_count);

    // With no workload running, core efficiency and utilization since the last read should be idle.
    {
        auto info = telemetry.read_info();
        ASSERT_TRUE(info.has_value());
        ASSERT_TRUE(info->device_core_efficiency_since_last_read.has_value());
        EXPECT_FLOAT_EQ(info->device_core_efficiency_since_last_read.value(), 0.0f);
        ASSERT_TRUE(info->info_cqs.front().utilization_since_last_read.has_value());
        EXPECT_FLOAT_EQ(info->info_cqs.front().utilization_since_last_read.value(), 0.0f);
    }

    // Run L1 blocking workloads across all cores in both sub-devices.
    reset_worker_l1_state(loaded_first_sub_device_cores);
    reset_worker_l1_state(loaded_second_sub_device_cores);

    distributed::MeshWorkload first_sub_device_workload;
    first_sub_device_workload.add_program(device_range_, create_l1_wait_program(loaded_first_sub_device_cores));
    distributed::EnqueueMeshWorkload(cq, first_sub_device_workload, false);
    mesh_device->set_sub_device_stall_group({{SubDeviceId{1}}});

    distributed::MeshWorkload second_sub_device_workload;
    second_sub_device_workload.add_program(device_range_, create_l1_wait_program(loaded_second_sub_device_cores));
    distributed::EnqueueMeshWorkload(cq, second_sub_device_workload, false);

    const bool first_sub_device_started = all_workers_reached_l1_wait(device, loaded_first_sub_device_cores);
    const bool second_sub_device_started = all_workers_reached_l1_wait(device, loaded_second_sub_device_cores);
    const bool all_cores_started = first_sub_device_started && second_sub_device_started;
    if (!all_cores_started) {
        release_worker(loaded_first_sub_device_cores);
        release_worker(loaded_second_sub_device_cores);
        mesh_device->reset_sub_device_stall_group();
        Finish(cq);
    }
    ASSERT_TRUE(all_cores_started);
    mesh_device->reset_sub_device_stall_group();

    // The transition into blocked work should report nonzero core efficiency since the last read.
    {
        auto info = telemetry.read_info();
        ASSERT_TRUE(info.has_value());
        ASSERT_TRUE(info->device_core_efficiency_since_last_read.has_value());
        EXPECT_GT(info->device_core_efficiency_since_last_read.value(), 0.0f);
        ASSERT_TRUE(info->info_cqs.front().utilization_since_last_read.has_value());
        EXPECT_GT(info->info_cqs.front().utilization_since_last_read.value(), 0.0f);
    }

    // Once all workers are blocked, core efficiency since the last read should be fully occupied.
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    {
        auto info = telemetry.read_info();
        ASSERT_TRUE(info.has_value());
        ASSERT_TRUE(info->device_core_efficiency_since_last_read.has_value());
        // Since we're only putting work on the tensix cores and not the eth cores, core efficiency will not be
        // exactly 1.0
        EXPECT_GT(info->device_core_efficiency_since_last_read.value(), 0.9f);
        ASSERT_TRUE(info->info_cqs.front().utilization_since_last_read.has_value());
        EXPECT_FLOAT_EQ(info->info_cqs.front().utilization_since_last_read.value(), 1.0f);
    }

    // Release the L1 blocked workers for sub device 1
    release_worker(loaded_first_sub_device_cores);
    Finish(cq, {{SubDeviceId{0}}});

    // The Utilization should not be affected while the second sub device is still working.
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    {
        auto info = telemetry.read_info();
        ASSERT_TRUE(info.has_value());
        ASSERT_TRUE(info->device_core_efficiency_since_last_read.has_value());
        EXPECT_GT(info->device_core_efficiency_since_last_read.value(), 0.0f);
        EXPECT_LT(info->device_core_efficiency_since_last_read.value(), 1.0f);
        ASSERT_TRUE(info->info_cqs.front().utilization_since_last_read.has_value());
        EXPECT_FLOAT_EQ(info->info_cqs.front().utilization_since_last_read.value(), 1.0f);
    }

    release_worker(loaded_second_sub_device_cores);
    Finish(cq);

    // The release transition should still report nonzero core efficiency and utilization since the last read.
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    {
        auto info = telemetry.read_info();
        ASSERT_TRUE(info.has_value());
        ASSERT_TRUE(info->device_core_efficiency_since_last_read.has_value());
        EXPECT_GT(info->device_core_efficiency_since_last_read.value(), 0.0f);
        EXPECT_LT(info->device_core_efficiency_since_last_read.value(), 1.0f);
        ASSERT_TRUE(info->info_cqs.front().utilization_since_last_read.has_value());
        EXPECT_GT(info->info_cqs.front().utilization_since_last_read.value(), 0.0f);
        EXPECT_LT(info->info_cqs.front().utilization_since_last_read.value(), 1.0f);
    }

    // After the released workload drains, core efficiency and utilization since the last read should return to idle.
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    {
        auto info = telemetry.read_info();
        ASSERT_TRUE(info.has_value());
        ASSERT_TRUE(info->device_core_efficiency_since_last_read.has_value());
        EXPECT_FLOAT_EQ(info->device_core_efficiency_since_last_read.value(), 0.0f);
        ASSERT_TRUE(info->info_cqs.front().utilization_since_last_read.has_value());
        EXPECT_FLOAT_EQ(info->info_cqs.front().utilization_since_last_read.value(), 0.0f);
    }

    mesh_device->clear_loaded_sub_device_manager();
    mesh_device->remove_sub_device_manager(sub_device_manager);
}

}  // namespace tt::tt_metal
