// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <gtest/gtest.h>
#include "tt_metal/tt_metal/common/mesh_dispatch_fixture.hpp"

#include "debug_tools_test_utils.hpp"
#include "hal_types.hpp"
#include <impl/debug/dprint_server.hpp>
#include <impl/debug/watcher_server.hpp>
#include <sys/mman.h>
#include <unistd.h>
#include <cstring>
#include <cerrno>
#include "tt_stl/assert.hpp"
#include "fmt/format.h"

// Access to internal API: BuildEnvManager, CompileProgram, get_kernel
#include "jit_build/build_env_manager.hpp"
#include "impl/program/program_impl.hpp"
#include "impl/kernels/kernel.hpp"

namespace tt::tt_metal {

class DebugToolsMeshFixture : public MeshDispatchFixture {
   protected:
       bool watcher_previous_enabled{};

       void TearDown() override { MeshDispatchFixture::TearDown(); }

       template <typename T>
       void RunTestOnDevice(
           const std::function<void(T*, std::shared_ptr<distributed::MeshDevice>)>& run_function,
           const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
           auto run_function_no_args = [this, run_function, mesh_device]() { run_function(static_cast<T*>(this), mesh_device); };
           MeshDispatchFixture::RunTestOnDevice(run_function_no_args, mesh_device);
       }
};

// A version of MeshDispatchFixture with DPrint enabled on all cores.
class DPrintMeshFixture : public DebugToolsMeshFixture {
public:
    std::string dprint_file_name;

    // A function to run a program, according to which dispatch mode is set.
    void RunProgram(const std::shared_ptr<distributed::MeshDevice>& mesh_device, distributed::MeshWorkload& workload) {
        // Only difference is that we need to wait for the print server to catch
        // up after running a test.
        DebugToolsMeshFixture::RunProgram(mesh_device, workload);
        MetalContext::instance().dprint_server()->await();
    }

    // Destructor ensures file descriptor is closed even if SetUp() throws
    ~DPrintMeshFixture() override {
        if (memfd_ >= 0) {
            close(memfd_);
        }
    }

protected:
    int memfd_ = -1;  // File descriptor for memory-backed file
    // Running with dprint + watcher enabled can make the code size blow up, so let's force watcher
    // disabled for DPRINT tests.
    void SetUp() override {
        // Create a unique memory-backed file for this test to avoid parallel test conflicts
        const testing::TestInfo* test_info = testing::UnitTest::GetInstance()->current_test_info();
        std::string test_desc = fmt::format("dprint_{}_{}_{}",
            getpid(),
            test_info->test_suite_name(),
            test_info->name());

        memfd_ = memfd_create(test_desc.c_str(), 0);
        if (memfd_ < 0) {
            TT_THROW("Failed to create memory file descriptor: {}", strerror(errno));
        }

        // Use /proc/self/fd path which works transparently with ofstream/ifstream
        dprint_file_name = fmt::format("/proc/self/fd/{}", memfd_);

        // The core range (virtual) needs to be set >= the set of all cores
        // used by all tests using this fixture, so set dprint enabled for
        // all cores and all devices
        tt::tt_metal::MetalContext::instance().rtoptions().set_feature_enabled(tt::llrt::RunTimeDebugFeatureDprint, true);
        tt::tt_metal::MetalContext::instance().rtoptions().set_feature_prepend_device_core_risc(
            tt::llrt::RunTimeDebugFeatureDprint, false);
        tt::tt_metal::MetalContext::instance().rtoptions().set_feature_all_cores(
            tt::llrt::RunTimeDebugFeatureDprint, CoreType::WORKER, tt::llrt::RunTimeDebugClassWorker);
        tt::tt_metal::MetalContext::instance().rtoptions().set_feature_all_cores(
            tt::llrt::RunTimeDebugFeatureDprint, CoreType::ETH, tt::llrt::RunTimeDebugClassWorker);
        tt::tt_metal::MetalContext::instance().rtoptions().set_feature_all_cores(
            tt::llrt::RunTimeDebugFeatureDprint, CoreType::DRAM, tt::llrt::RunTimeDebugClassWorker);
        tt::tt_metal::MetalContext::instance().rtoptions().set_feature_all_chips(tt::llrt::RunTimeDebugFeatureDprint, true);
        tt::tt_metal::MetalContext::instance().rtoptions().set_feature_mesh_coords(
            tt::llrt::RunTimeDebugFeatureDprint, {});
        tt::tt_metal::MetalContext::instance().rtoptions().set_feature_chip_ids(
            tt::llrt::RunTimeDebugFeatureDprint, {});
        // Send output to a file so the test can check after program is run.
        tt::tt_metal::MetalContext::instance().rtoptions().set_feature_file_name(tt::llrt::RunTimeDebugFeatureDprint, dprint_file_name);
        tt::tt_metal::MetalContext::instance().rtoptions().set_test_mode_enabled(true);
        watcher_previous_enabled = tt::tt_metal::MetalContext::instance().rtoptions().get_watcher_enabled();
        tt::tt_metal::MetalContext::instance().rtoptions().set_watcher_enabled(false);

        ExtraSetUp();

        // Parent class initializes devices and any necessary flags
        DebugToolsMeshFixture::SetUp();
    }

    void TearDown() override {
        // Parent class tears down devices
        DebugToolsMeshFixture::TearDown();
        ExtraTearDown();

        // Close the memory-backed file descriptor
        if (memfd_ >= 0) {
            close(memfd_);
            memfd_ = -1;
        }

        // Reset DPrint settings
        tt::tt_metal::MetalContext::instance().rtoptions().set_feature_cores(tt::llrt::RunTimeDebugFeatureDprint, {});
        tt::tt_metal::MetalContext::instance().rtoptions().set_feature_enabled(tt::llrt::RunTimeDebugFeatureDprint, false);
        tt::tt_metal::MetalContext::instance().rtoptions().set_feature_all_cores(
            tt::llrt::RunTimeDebugFeatureDprint, CoreType::WORKER, tt::llrt::RunTimeDebugClassNoneSpecified);
        tt::tt_metal::MetalContext::instance().rtoptions().set_feature_all_cores(
            tt::llrt::RunTimeDebugFeatureDprint, CoreType::ETH, tt::llrt::RunTimeDebugClassNoneSpecified);
        tt::tt_metal::MetalContext::instance().rtoptions().set_feature_all_cores(
            tt::llrt::RunTimeDebugFeatureDprint, CoreType::DRAM, tt::llrt::RunTimeDebugClassNoneSpecified);
        tt::tt_metal::MetalContext::instance().rtoptions().set_feature_all_chips(tt::llrt::RunTimeDebugFeatureDprint, false);
        tt::tt_metal::MetalContext::instance().rtoptions().set_feature_mesh_coords(
            tt::llrt::RunTimeDebugFeatureDprint, {});
        tt::tt_metal::MetalContext::instance().rtoptions().set_feature_chip_ids(
            tt::llrt::RunTimeDebugFeatureDprint, {});
        tt::tt_metal::MetalContext::instance().rtoptions().set_feature_file_name(tt::llrt::RunTimeDebugFeatureDprint, "");
        tt::tt_metal::MetalContext::instance().rtoptions().set_feature_prepend_device_core_risc(
            tt::llrt::RunTimeDebugFeatureDprint, true);
        tt::tt_metal::MetalContext::instance().rtoptions().set_test_mode_enabled(false);
        tt::tt_metal::MetalContext::instance().rtoptions().set_watcher_enabled(watcher_previous_enabled);
    }

    void RunTestOnDevice(
        const std::function<void(DPrintMeshFixture*, std::shared_ptr<distributed::MeshDevice>)>& run_function,
        const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
        DebugToolsMeshFixture::RunTestOnDevice(run_function, mesh_device);
        MetalContext::instance().dprint_server()->clear_log_file();
    }

    // Override this function in child classes for additional setup commands between DPRINT setup
    // and device creation.
    virtual void ExtraSetUp() {}
    virtual void ExtraTearDown() {}
};

// For usage by tests that need the dprint server devices disabled.
class DPrintDisableMeshDevicesFixture : public DPrintMeshFixture {
protected:
    void ExtraSetUp() override {
        // For this test, mute each devices using the environment variable
        tt::tt_metal::MetalContext::instance().rtoptions().set_feature_all_chips(tt::llrt::RunTimeDebugFeatureDprint, false);
        tt::tt_metal::MetalContext::instance().rtoptions().set_feature_chip_ids(tt::llrt::RunTimeDebugFeatureDprint, {});
    }
    void ExtraTearDown() override {
        MetalContext::instance().teardown(); // Teardown dprint server so we can re-init later with all devices enabled again
    }
};

class DPrintSeparateFilesFixture : public DPrintMeshFixture {
public:
    static constexpr std::array<std::string_view, 5> suffixes = {"BRISC", "NCRISC", "TRISC0", "TRISC1", "TRISC2"};
    static void check_output(std::span<const std::string> expected) {
        const auto& enabled_processors =
            tt::tt_metal::MetalContext::instance().rtoptions().get_feature_processors(tt::llrt::RunTimeDebugFeatureDprint);
        ASSERT_EQ(expected.size(), suffixes.size());
        for (size_t i = 0; i < suffixes.size(); i++) {
            if (!enabled_processors.contains(HalProgrammableCoreType::TENSIX, i)) {
                continue;
            }
            auto filename = fmt::format("{}generated/dprint/device-0_worker-core-0-0_{}.txt",
                tt::tt_metal::MetalContext::instance().rtoptions().get_logs_dir(), suffixes[i]);
            EXPECT_TRUE(FilesMatchesString(filename, expected[i]));
        }
    }
protected:
    bool original_one_file_per_risc_{};
    void ExtraSetUp() override {
        // For this test, enable one file per risc
        original_one_file_per_risc_ = tt::tt_metal::MetalContext::instance().rtoptions().get_feature_one_file_per_risc(
            tt::llrt::RunTimeDebugFeatureDprint);
        tt::tt_metal::MetalContext::instance().rtoptions().set_feature_one_file_per_risc(
            tt::llrt::RunTimeDebugFeatureDprint, true);
    }
    void ExtraTearDown() override {
        tt::tt_metal::MetalContext::instance().rtoptions().set_feature_one_file_per_risc(
            tt::llrt::RunTimeDebugFeatureDprint, original_one_file_per_risc_);
    }
};

// A version of MeshDispatchFixture with watcher enabled
class MeshWatcherFixture : public DebugToolsMeshFixture {
public:
    std::string log_file_name;
    inline static const int interval_ms = 250;

    // A function to run a program, according to which dispatch mode is set.
    void RunProgram(
        const std::shared_ptr<distributed::MeshDevice>& mesh_device,
        distributed::MeshWorkload& workload,
        bool wait_for_dump = false) {
        // Only difference is that we need to wait for the print server to catch
        // up after running a test.
        DebugToolsMeshFixture::RunProgram(mesh_device, workload);

        // Wait for watcher to run a full dump before finishing, need to wait for dump count to
        // increase because we'll likely check in the middle of a dump.
        if (wait_for_dump) {
            int curr_count = MetalContext::instance().watcher_server()->dump_count();
            while (MetalContext::instance().watcher_server()->dump_count() < curr_count + 2) {;}
        }
    }

protected:
    int watcher_previous_interval{};
    bool watcher_previous_dump_all{};
    bool watcher_previous_append{};
    bool watcher_previous_auto_unpause{};
    bool watcher_previous_noinline{};
    bool watcher_previous_noc_sanitize_linked_transaction{};
    bool test_mode_previous{};
    void SetUp() override {
        // Initialize log file name once during setup
        log_file_name = tt::tt_metal::MetalContext::instance().rtoptions().get_logs_dir() + "generated/watcher/watcher.log";

        // Enable watcher for this test, save the previous state so we can restore it later.
        watcher_previous_enabled = tt::tt_metal::MetalContext::instance().rtoptions().get_watcher_enabled();
        watcher_previous_interval = tt::tt_metal::MetalContext::instance().rtoptions().get_watcher_interval();
        watcher_previous_dump_all = tt::tt_metal::MetalContext::instance().rtoptions().get_watcher_dump_all();
        watcher_previous_append = tt::tt_metal::MetalContext::instance().rtoptions().get_watcher_append();
        watcher_previous_auto_unpause = tt::tt_metal::MetalContext::instance().rtoptions().get_watcher_auto_unpause();
        watcher_previous_noinline = tt::tt_metal::MetalContext::instance().rtoptions().get_watcher_noinline();
        watcher_previous_noc_sanitize_linked_transaction =
            tt::tt_metal::MetalContext::instance().rtoptions().get_watcher_noc_sanitize_linked_transaction();
        test_mode_previous = tt::tt_metal::MetalContext::instance().rtoptions().get_test_mode_enabled();
        tt::tt_metal::MetalContext::instance().rtoptions().set_watcher_enabled(true);
        tt::tt_metal::MetalContext::instance().rtoptions().set_watcher_interval(interval_ms);
        tt::tt_metal::MetalContext::instance().rtoptions().set_watcher_dump_all(false);
        tt::tt_metal::MetalContext::instance().rtoptions().set_watcher_append(false);
        tt::tt_metal::MetalContext::instance().rtoptions().set_watcher_auto_unpause(true);
        tt::tt_metal::MetalContext::instance().rtoptions().set_watcher_noinline(true);
        tt::tt_metal::MetalContext::instance().rtoptions().set_test_mode_enabled(true);

        const auto detected_arch = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        tt::tt_metal::MetalContext::instance().rtoptions().set_watcher_noc_sanitize_linked_transaction(
            detected_arch == tt::ARCH::BLACKHOLE || detected_arch == tt::ARCH::QUASAR);

        // Parent class initializes devices and any necessary flags
        DebugToolsMeshFixture::SetUp();

        MetalContext::instance().watcher_server()->clear_log();
    }

    void TearDown() override {
        // Parent class tears down devices
        DebugToolsMeshFixture::TearDown();

        // Reset watcher settings to their previous values
        tt::tt_metal::MetalContext::instance().rtoptions().set_watcher_interval(watcher_previous_interval);
        tt::tt_metal::MetalContext::instance().rtoptions().set_watcher_dump_all(watcher_previous_dump_all);
        tt::tt_metal::MetalContext::instance().rtoptions().set_watcher_append(watcher_previous_append);
        tt::tt_metal::MetalContext::instance().rtoptions().set_watcher_auto_unpause(watcher_previous_auto_unpause);
        tt::tt_metal::MetalContext::instance().rtoptions().set_watcher_noinline(watcher_previous_noinline);
        tt::tt_metal::MetalContext::instance().rtoptions().set_watcher_noc_sanitize_linked_transaction(
            watcher_previous_noc_sanitize_linked_transaction);
        tt::tt_metal::MetalContext::instance().rtoptions().set_test_mode_enabled(test_mode_previous);
        tt::tt_metal::MetalContext::instance().rtoptions().set_watcher_enabled(watcher_previous_enabled);
    }

    void RunTestOnDevice(
        const std::function<void(MeshWatcherFixture*, std::shared_ptr<distributed::MeshDevice>)>& run_function,
        const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
        DebugToolsMeshFixture::RunTestOnDevice(run_function, mesh_device);
        // Wait for a final watcher poll and then clear the log.
        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
        MetalContext::instance().watcher_server()->clear_log();
    }
};

// A version of WatcherFixture with read and write debug delays enabled
class MeshWatcherDelayFixture : public MeshWatcherFixture {
public:
    tt::llrt::TargetSelection saved_target_selection[tt::llrt::RunTimeDebugFeatureCount];

    std::map<CoreType, std::vector<CoreCoord>> delayed_cores;

    void SetUp() override {
        tt::tt_metal::MetalContext::instance().rtoptions().set_watcher_debug_delay(5000000);
        delayed_cores[CoreType::WORKER] = {{0, 0}, {1, 1}};

        // Store the previous state of the watcher features
        saved_target_selection[tt::llrt::RunTimeDebugFeatureReadDebugDelay] = tt::tt_metal::MetalContext::instance().rtoptions().get_feature_targets(tt::llrt::RunTimeDebugFeatureReadDebugDelay);
        saved_target_selection[tt::llrt::RunTimeDebugFeatureWriteDebugDelay] = tt::tt_metal::MetalContext::instance().rtoptions().get_feature_targets(tt::llrt::RunTimeDebugFeatureWriteDebugDelay);
        saved_target_selection[tt::llrt::RunTimeDebugFeatureAtomicDebugDelay] = tt::tt_metal::MetalContext::instance().rtoptions().get_feature_targets(tt::llrt::RunTimeDebugFeatureAtomicDebugDelay);

        // Enable read and write debug delay for the test core
        tt::tt_metal::MetalContext::instance().rtoptions().set_feature_enabled(tt::llrt::RunTimeDebugFeatureReadDebugDelay, true);
        tt::tt_metal::MetalContext::instance().rtoptions().set_feature_cores(tt::llrt::RunTimeDebugFeatureReadDebugDelay, delayed_cores);
        tt::tt_metal::MetalContext::instance().rtoptions().set_feature_enabled(tt::llrt::RunTimeDebugFeatureWriteDebugDelay, true);
        tt::tt_metal::MetalContext::instance().rtoptions().set_feature_cores(tt::llrt::RunTimeDebugFeatureWriteDebugDelay, delayed_cores);

        // Call parent
        MeshWatcherFixture::SetUp();
    }

    void TearDown() override {
        // Call parent
        MeshWatcherFixture::TearDown();

        // Restore
        tt::tt_metal::MetalContext::instance().rtoptions().set_feature_targets(tt::llrt::RunTimeDebugFeatureReadDebugDelay, saved_target_selection[tt::llrt::RunTimeDebugFeatureReadDebugDelay]);
        tt::tt_metal::MetalContext::instance().rtoptions().set_feature_targets(tt::llrt::RunTimeDebugFeatureWriteDebugDelay, saved_target_selection[tt::llrt::RunTimeDebugFeatureWriteDebugDelay]);
        tt::tt_metal::MetalContext::instance().rtoptions().set_feature_targets(tt::llrt::RunTimeDebugFeatureAtomicDebugDelay, saved_target_selection[tt::llrt::RunTimeDebugFeatureAtomicDebugDelay]);
    }
};

// A version of MeshWatcherFixture with dump_all enabled for tile counter visibility
class MeshWatcherDumpAllFixture : public MeshWatcherFixture {
protected:
    void SetUp() override {
        MeshWatcherFixture::SetUp();
        tt::tt_metal::MetalContext::instance().rtoptions().set_watcher_dump_all(true);
    }
};

// Lightweight fixture for tests that only compile kernels (not run them) and inspect the
// resulting ELF. Devices are initialized once per test suite via SetUpTestSuite, avoiding the
// ~1s per-test device-init cost.
class DevicePrintCompileFixture : public ::testing::Test {
public:
    static void SetUpTestSuite() {
        auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
        watcher_previous_enabled_ = rtoptions.get_watcher_enabled();

        // Use a single memfd for the suite. The dprint server opens it at device init; these
        // tests never read it back since they don't run kernels.
        suite_memfd_ = memfd_create("dprint_compile_only_suite", 0);
        if (suite_memfd_ < 0) {
            TT_THROW("Failed to create memory file descriptor: {}", strerror(errno));
        }
        suite_dprint_file_name_ = fmt::format("/proc/self/fd/{}", suite_memfd_);

        rtoptions.set_feature_enabled(tt::llrt::RunTimeDebugFeatureDprint, true);
        rtoptions.set_feature_prepend_device_core_risc(tt::llrt::RunTimeDebugFeatureDprint, false);
        rtoptions.set_feature_all_cores(
            tt::llrt::RunTimeDebugFeatureDprint, CoreType::WORKER, tt::llrt::RunTimeDebugClassWorker);
        rtoptions.set_feature_all_cores(
            tt::llrt::RunTimeDebugFeatureDprint, CoreType::ETH, tt::llrt::RunTimeDebugClassWorker);
        rtoptions.set_feature_all_cores(
            tt::llrt::RunTimeDebugFeatureDprint, CoreType::DRAM, tt::llrt::RunTimeDebugClassWorker);
        rtoptions.set_feature_all_chips(tt::llrt::RunTimeDebugFeatureDprint, true);
        rtoptions.set_feature_mesh_coords(tt::llrt::RunTimeDebugFeatureDprint, {});
        rtoptions.set_feature_chip_ids(tt::llrt::RunTimeDebugFeatureDprint, {});
        rtoptions.set_feature_file_name(tt::llrt::RunTimeDebugFeatureDprint, suite_dprint_file_name_);
        rtoptions.set_test_mode_enabled(true);
        rtoptions.set_watcher_enabled(false);
        rtoptions.set_use_device_print(true);

        std::vector<ChipId> ids;
        for (ChipId id : tt::tt_metal::MetalContext::instance().get_cluster().user_exposed_chip_ids()) {
            ids.push_back(id);
        }
        const auto& dispatch_core_config = rtoptions.get_dispatch_core_config();
        id_to_device_ = distributed::MeshDevice::create_unit_meshes(
            ids, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, dispatch_core_config);
        for (const auto& [device_id, device] : id_to_device_) {
            devices_.push_back(device);
        }
    }

    static void TearDownTestSuite() {
        for (auto& [device_id, device] : id_to_device_) {
            device->close();
            device.reset();
        }
        id_to_device_.clear();
        devices_.clear();

        if (suite_memfd_ >= 0) {
            close(suite_memfd_);
            suite_memfd_ = -1;
        }

        auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
        rtoptions.set_use_device_print(false);
        rtoptions.set_watcher_enabled(watcher_previous_enabled_);
        rtoptions.set_test_mode_enabled(false);
        rtoptions.set_feature_enabled(tt::llrt::RunTimeDebugFeatureDprint, false);
        rtoptions.set_feature_prepend_device_core_risc(tt::llrt::RunTimeDebugFeatureDprint, true);
        rtoptions.set_feature_file_name(tt::llrt::RunTimeDebugFeatureDprint, "");
    }

    std::string CompileKernel(const std::string& kernel_path, stl::Span<const uint32_t> runtime_args = {}) {
        auto mesh_device = devices_.at(0);

        distributed::MeshWorkload workload;
        auto zero_coord = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
        Program program = Program();
        workload.add_program(device_range, std::move(program));
        auto& program_ = workload.get_programs().at(device_range);

        constexpr CoreCoord core = {0, 0};
        DataMovementConfig config{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default};
        KernelHandle kernel_handle = CreateKernel(program_, kernel_path, core, config);

        SetRuntimeArgs(program_, kernel_handle, core, runtime_args);
        auto* device = mesh_device->get_devices()[0];
        detail::CompileProgram(device, program_);

        const auto& hal = tt::tt_metal::MetalContext::instance().hal();
        uint32_t tensix_core_type = hal.get_programmable_core_type_index(tt::tt_metal::HalProgrammableCoreType::TENSIX);
        uint32_t dm_class_idx = enchantum::to_underlying(tt::tt_metal::HalProcessorClassType::DM);
        int riscv_id = static_cast<std::underlying_type_t<tt::tt_metal::DataMovementProcessor>>(config.processor);

        const auto& build_state = tt::tt_metal::BuildEnvManager::get_instance().get_kernel_build_state(
            device->build_id(), tensix_core_type, dm_class_idx, riscv_id);

        const auto& kernel = program_.impl().get_kernel(kernel_handle);
        const std::string full_kernel_name = kernel->get_full_kernel_name();
        return build_state.get_target_out_path(full_kernel_name);
    }

protected:
    inline static std::map<ChipId, std::shared_ptr<distributed::MeshDevice>> id_to_device_;
    inline static std::vector<std::shared_ptr<distributed::MeshDevice>> devices_;
    inline static int suite_memfd_ = -1;
    inline static std::string suite_dprint_file_name_;
    inline static bool watcher_previous_enabled_{};
};

class DevicePrintFixture : public DebugToolsMeshFixture {
public:
    // Devices are created lazily by SetUp on the first test of the suite, reused across subsequent
    // tests, and torn down in TearDownTestSuite. Tests / ExtraSetUp / ExtraTearDown that need to
    // destroy device state (e.g. via MetalContext::teardown()) must call MarkSharedPoolInvalid()
    // *before* the teardown so the shared_ptrs are released cleanly first; the next SetUp will
    // then re-initialize the pool.
    static void MarkSharedPoolInvalid() {
        // Loop with auto-copy so refcount stays >0 during the explicit close; the actual
        // destruction (and second close_impl via ~MeshDevice) happens at vector/map clear below,
        // which mirrors the long-standing pattern in MeshDispatchFixture::TearDown.
        for (auto [device_id, device] : shared_id_to_device_) {
            device->close();
            device.reset();
        }
        shared_devices_.clear();
        shared_id_to_device_.clear();
    }

    static void TearDownTestSuite() {
        MarkSharedPoolInvalid();
        auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
        rtoptions.set_use_device_print(false);
    }

protected:
    int memfd_ = -1;
    inline static std::map<ChipId, std::shared_ptr<distributed::MeshDevice>> shared_id_to_device_;
    inline static std::vector<std::shared_ptr<distributed::MeshDevice>> shared_devices_;

    void SetUp() override {
        const testing::TestInfo* test_info = testing::UnitTest::GetInstance()->current_test_info();
        std::string test_desc =
            fmt::format("dprint_{}_{}_{}", getpid(), test_info->test_suite_name(), test_info->name());

        memfd_ = memfd_create(test_desc.c_str(), 0);
        if (memfd_ < 0) {
            TT_THROW("Failed to create memory file descriptor: {}", strerror(errno));
        }
        // Use /proc/self/fd path which works transparently with ofstream/ifstream
        dprint_file_name = fmt::format("/proc/self/fd/{}", memfd_);

        auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
        rtoptions.set_feature_enabled(tt::llrt::RunTimeDebugFeatureDprint, true);
        rtoptions.set_feature_prepend_device_core_risc(tt::llrt::RunTimeDebugFeatureDprint, false);
        rtoptions.set_feature_all_cores(
            tt::llrt::RunTimeDebugFeatureDprint, CoreType::WORKER, tt::llrt::RunTimeDebugClassWorker);
        rtoptions.set_feature_all_cores(
            tt::llrt::RunTimeDebugFeatureDprint, CoreType::ETH, tt::llrt::RunTimeDebugClassWorker);
        rtoptions.set_feature_all_cores(
            tt::llrt::RunTimeDebugFeatureDprint, CoreType::DRAM, tt::llrt::RunTimeDebugClassWorker);
        rtoptions.set_feature_all_chips(tt::llrt::RunTimeDebugFeatureDprint, true);
        rtoptions.set_feature_mesh_coords(tt::llrt::RunTimeDebugFeatureDprint, {});
        rtoptions.set_feature_chip_ids(tt::llrt::RunTimeDebugFeatureDprint, {});
        rtoptions.set_feature_file_name(tt::llrt::RunTimeDebugFeatureDprint, dprint_file_name);
        rtoptions.set_test_mode_enabled(true);
        watcher_previous_enabled = rtoptions.get_watcher_enabled();
        rtoptions.set_watcher_enabled(false);
        rtoptions.set_use_device_print(true);

        ExtraSetUp();

        // Per-test instance members from MeshDispatchFixture must be initialized every test —
        // tests reference this->arch_ etc. directly (see test_print_config_register.cpp num_of_registers).
        this->DetectDispatchMode();
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        this->init_max_cbs();

        if (shared_id_to_device_.empty()) {
            // First test in the suite, or a prior test invalidated the pool — create devices.
            std::vector<ChipId> ids;
            for (ChipId id : tt::tt_metal::MetalContext::instance().get_cluster().user_exposed_chip_ids()) {
                ids.push_back(id);
            }
            const auto& dispatch_core_config = rtoptions.get_dispatch_core_config();
            shared_id_to_device_ = distributed::MeshDevice::create_unit_meshes(
                ids, l1_small_size_, trace_region_size_, 1, dispatch_core_config);
            for (const auto& [device_id, device] : shared_id_to_device_) {
                shared_devices_.push_back(device);
            }
        } else {
            // Reusing devices from earlier tests in this suite. Swap dprint output to this test's
            // memfd so output isn't shared with the prior test.
            tt::tt_metal::MetalContext::instance().dprint_server()->clear_log_file();
        }

        this->devices_ = shared_devices_;
    }

    void TearDown() override {
        // Drop the per-test instance copy of the device pointers BEFORE ExtraTearDown so that
        // fixtures whose ExtraTearDown calls MetalContext::teardown() can do so without leaving
        // a live shared_ptr<MeshDevice> on this instance whose destructor would later run after
        // backing state has been freed. The static pool keeps the devices alive until either the
        // next SetUp reuses them or TearDownTestSuite (or MarkSharedPoolInvalid) closes them.
        this->devices_.clear();

        ExtraTearDown();

        if (memfd_ >= 0) {
            close(memfd_);
            memfd_ = -1;
        }

        // If a hang was detected (e.g. await() / flush() timed out — the dprint server's polling
        // thread sets server_killed_due_to_hang_ and returns), the server is wedged for any
        // future await/flush. Force a full teardown so the next test in this suite reinitializes
        // a fresh dprint server alongside fresh devices. Without this, a single failing test
        // cascades into TT_THROW timeouts on every subsequent test in the suite.
        auto& server = tt::tt_metal::MetalContext::instance().dprint_server();
        if (server && server->hang_detected()) {
            MarkSharedPoolInvalid();
            tt::tt_metal::MetalContext::instance().teardown();
        }

        auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
        rtoptions.set_watcher_enabled(watcher_previous_enabled);
    }

    // Override this function in child classes for additional setup commands between DPRINT setup
    // and device creation.
    virtual void ExtraSetUp() {}
    virtual void ExtraTearDown() {}

public:
    std::string dprint_file_name;

    std::string CompileKernel(const std::string& kernel_path, stl::Span<const uint32_t> runtime_args = {}) {
        // Get the first available mesh device
        auto mesh_device = this->devices_.at(0);

        // Set up program
        distributed::MeshWorkload workload;
        auto zero_coord = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
        Program program = Program();
        workload.add_program(device_range, std::move(program));
        auto& program_ = workload.get_programs().at(device_range);

        // This tests prints only on a single core
        constexpr CoreCoord core = {0, 0};  // Print on first core only
        DataMovementConfig config{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default};
        KernelHandle kernel_handle = CreateKernel(program_, kernel_path, core, config);

        SetRuntimeArgs(program_, kernel_handle, core, runtime_args);
        auto* device = mesh_device->get_devices()[0];
        detail::CompileProgram(device, program_);

        // Find compiled kernel and extract format string from it to compare with expected_format_message
        const auto& hal = tt::tt_metal::MetalContext::instance().hal();
        uint32_t tensix_core_type = hal.get_programmable_core_type_index(tt::tt_metal::HalProgrammableCoreType::TENSIX);
        uint32_t dm_class_idx = enchantum::to_underlying(tt::tt_metal::HalProcessorClassType::DM);

        int riscv_id = static_cast<std::underlying_type_t<tt::tt_metal::DataMovementProcessor>>(config.processor);

        const auto& build_state = tt::tt_metal::BuildEnvManager::get_instance().get_kernel_build_state(
            device->build_id(), tensix_core_type, dm_class_idx, riscv_id);

        const auto& kernel = program_.impl().get_kernel(kernel_handle);
        const std::string full_kernel_name = kernel->get_full_kernel_name();
        return build_state.get_target_out_path(full_kernel_name);
    }

    // A function to run a program, according to which dispatch mode is set.
    void RunProgram(
        const std::shared_ptr<distributed::MeshDevice>& mesh_device,
        const std::string& kernel_path,
        stl::Span<const uint32_t> runtime_args = {}) {
        // Set up program
        distributed::MeshWorkload workload;
        auto zero_coord = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
        Program program = Program();
        workload.add_program(device_range, std::move(program));
        auto& program_ = workload.get_programs().at(device_range);

        // This tests prints only on a single core
        constexpr CoreCoord core = {0, 0};  // Print on first core only
        DataMovementConfig config{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default};
        KernelHandle kernel_handle = CreateKernel(program_, kernel_path, core, config);

        SetRuntimeArgs(program_, kernel_handle, core, runtime_args);

        // Wait for the print server, then drain device buffers and force-flush intermediate
        // streams. flush() (vs await()) is required when devices are shared across the suite
        // because clear_log_file/file-read no longer race against device-detach.
        DebugToolsMeshFixture::RunProgram(mesh_device, workload);
        MetalContext::instance().dprint_server()->flush();
    }

    // A function to run a program, according to which dispatch mode is set.
    void RunProgram(const std::shared_ptr<distributed::MeshDevice>& mesh_device, distributed::MeshWorkload& workload) {
        DebugToolsMeshFixture::RunProgram(mesh_device, workload);
        MetalContext::instance().dprint_server()->flush();
    }

    void RunTestOnDevice(
        const std::function<void(DevicePrintFixture*, std::shared_ptr<distributed::MeshDevice>)>& run_function,
        const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
        DebugToolsMeshFixture::RunTestOnDevice(run_function, mesh_device);
        MetalContext::instance().dprint_server()->clear_log_file();
    }
};

class DevicePrintSeparateFilesFixture : public DevicePrintFixture {
public:
    static constexpr std::array<std::string_view, 5> suffixes = {"BRISC", "NCRISC", "TRISC0", "TRISC1", "TRISC2"};
    static void check_output(std::span<const std::string> expected) {
        const auto& enabled_processors = tt::tt_metal::MetalContext::instance().rtoptions().get_feature_processors(
            tt::llrt::RunTimeDebugFeatureDprint);
        ASSERT_EQ(expected.size(), suffixes.size());
        for (size_t i = 0; i < suffixes.size(); i++) {
            if (!enabled_processors.contains(HalProgrammableCoreType::TENSIX, i)) {
                continue;
            }
            auto filename = fmt::format(
                "{}generated/dprint/device-0_worker-core-0-0_{}.txt",
                tt::tt_metal::MetalContext::instance().rtoptions().get_logs_dir(),
                suffixes[i]);
            EXPECT_TRUE(FilesMatchesString(filename, expected[i]));
        }
    }

    // A function to run a program, according to which dispatch mode is set.
    void RunProgram(const std::shared_ptr<distributed::MeshDevice>& mesh_device, distributed::MeshWorkload& workload) {
        DebugToolsMeshFixture::RunProgram(mesh_device, workload);
        MetalContext::instance().dprint_server()->flush();
    }

protected:
    bool original_one_file_per_risc_{};
    void ExtraSetUp() override {
        // For this test, enable one file per risc
        original_one_file_per_risc_ = tt::tt_metal::MetalContext::instance().rtoptions().get_feature_one_file_per_risc(
            tt::llrt::RunTimeDebugFeatureDprint);
        tt::tt_metal::MetalContext::instance().rtoptions().set_feature_one_file_per_risc(
            tt::llrt::RunTimeDebugFeatureDprint, true);
    }
    void ExtraTearDown() override {
        tt::tt_metal::MetalContext::instance().rtoptions().set_feature_one_file_per_risc(
            tt::llrt::RunTimeDebugFeatureDprint, original_one_file_per_risc_);
        // Per-risc dprint files at fixed paths accumulate across tests because
        // dprint_server::clear_log_file does not handle one_file_per_risc mode. Force device
        // teardown so the next SetUp re-initializes the dprint server with fresh files.
        DevicePrintFixture::MarkSharedPoolInvalid();
        MetalContext::instance().teardown();
    }

    void RunTestOnDevice(
        const std::function<void(DevicePrintSeparateFilesFixture*, std::shared_ptr<distributed::MeshDevice>)>&
            run_function,
        const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
        DebugToolsMeshFixture::RunTestOnDevice(run_function, mesh_device);
        MetalContext::instance().dprint_server()->clear_log_file();
    }
};

} // namespace tt::tt_metal
