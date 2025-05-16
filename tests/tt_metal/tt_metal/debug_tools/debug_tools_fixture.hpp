// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <gtest/gtest.h>
#include "debug/watcher_server.hpp"
#include "dispatch_fixture.hpp"
#include "tt_metal/tt_metal/common/dispatch_fixture.hpp"

#include "dprint_server.hpp"

namespace tt::tt_metal {

class DebugToolsFixture : public DispatchFixture {
   protected:
    bool watcher_previous_enabled;

    void TearDown() override {
        DispatchFixture::TearDown();
        tt::tt_metal::MetalContext::instance().rtoptions().set_watcher_enabled(watcher_previous_enabled);
    }

    template <typename T>
    void RunTestOnDevice(const std::function<void(T*, IDevice*)>& run_function, IDevice* device) {
        auto run_function_no_args = [=,this]() { run_function(static_cast<T*>(this), device); };
        DispatchFixture::RunTestOnDevice(run_function_no_args, device);
    }
};

// A version of DispatchFixture with DPrint enabled on all cores.
class DPrintFixture : public DebugToolsFixture {
public:
    inline static const string dprint_file_name = "gtest_dprint_log.txt";

    // A function to run a program, according to which dispatch mode is set.
    void RunProgram(IDevice* device, Program& program) {
        // Only difference is that we need to wait for the print server to catch
        // up after running a test.
        DebugToolsFixture::RunProgram(device, program);
        DprintServerAwait();
    }

protected:
    // Running with dprint + watcher enabled can make the code size blow up, so let's force watcher
    // disabled for DPRINT tests.
    void SetUp() override {
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
        tt::tt_metal::MetalContext::instance().rtoptions().set_feature_all_chips(tt::llrt::RunTimeDebugFeatureDprint, true);
        // Send output to a file so the test can check after program is run.
        tt::tt_metal::MetalContext::instance().rtoptions().set_feature_file_name(tt::llrt::RunTimeDebugFeatureDprint, dprint_file_name);
        tt::tt_metal::MetalContext::instance().rtoptions().set_test_mode_enabled(true);
        watcher_previous_enabled = tt::tt_metal::MetalContext::instance().rtoptions().get_watcher_enabled();
        tt::tt_metal::MetalContext::instance().rtoptions().set_watcher_enabled(false);

        ExtraSetUp();

        // Parent class initializes devices and any necessary flags
        DebugToolsFixture::SetUp();
    }

    void TearDown() override {
        // Parent class tears down devices
        DebugToolsFixture::TearDown();

        // Remove the DPrint output file after the test is finished.
        std::remove(dprint_file_name.c_str());

        // Reset DPrint settings
        tt::tt_metal::MetalContext::instance().rtoptions().set_feature_cores(tt::llrt::RunTimeDebugFeatureDprint, {});
        tt::tt_metal::MetalContext::instance().rtoptions().set_feature_enabled(tt::llrt::RunTimeDebugFeatureDprint, false);
        tt::tt_metal::MetalContext::instance().rtoptions().set_feature_all_cores(
            tt::llrt::RunTimeDebugFeatureDprint, CoreType::WORKER, tt::llrt::RunTimeDebugClassNoneSpecified);
        tt::tt_metal::MetalContext::instance().rtoptions().set_feature_all_cores(
            tt::llrt::RunTimeDebugFeatureDprint, CoreType::ETH, tt::llrt::RunTimeDebugClassNoneSpecified);
        tt::tt_metal::MetalContext::instance().rtoptions().set_feature_all_chips(tt::llrt::RunTimeDebugFeatureDprint, false);
        tt::tt_metal::MetalContext::instance().rtoptions().set_feature_file_name(tt::llrt::RunTimeDebugFeatureDprint, "");
        tt::tt_metal::MetalContext::instance().rtoptions().set_feature_prepend_device_core_risc(
            tt::llrt::RunTimeDebugFeatureDprint, true);
        tt::tt_metal::MetalContext::instance().rtoptions().set_test_mode_enabled(false);
    }

    void RunTestOnDevice(
        const std::function<void(DPrintFixture*, IDevice*)>& run_function,
        IDevice* device
    ) {
        DebugToolsFixture::RunTestOnDevice(run_function, device);
        DPrintServerClearLogFile();
        DPrintServerClearSignals();
    }

    // Override this function in child classes for additional setup commands between DPRINT setup
    // and device creation.
    virtual void ExtraSetUp() {}
};

// For usage by tests that need the dprint server devices disabled.
class DPrintDisableDevicesFixture : public DPrintFixture {
protected:
    void ExtraSetUp() override {
        // For this test, mute each devices using the environment variable
        tt::tt_metal::MetalContext::instance().rtoptions().set_feature_all_chips(tt::llrt::RunTimeDebugFeatureDprint, false);
        tt::tt_metal::MetalContext::instance().rtoptions().set_feature_chip_ids(tt::llrt::RunTimeDebugFeatureDprint, {});
    }
};

// A version of DispatchFixture with watcher enabled
class WatcherFixture : public DebugToolsFixture {
public:
    inline static const string log_file_name = "generated/watcher/watcher.log";
    inline static const int interval_ms = 250;

    // A function to run a program, according to which dispatch mode is set.
    void RunProgram(IDevice* device, Program& program, bool wait_for_dump = false) {
        // Only difference is that we need to wait for the print server to catch
        // up after running a test.
        DebugToolsFixture::RunProgram(device, program);

        // Wait for watcher to run a full dump before finishing, need to wait for dump count to
        // increase because we'll likely check in the middle of a dump.
        if (wait_for_dump) {
            int curr_count = tt::watcher_get_dump_count();
            while (tt::watcher_get_dump_count() < curr_count + 2) {;}
        }
    }

protected:
    int  watcher_previous_interval;
    bool watcher_previous_dump_all;
    bool watcher_previous_append;
    bool watcher_previous_auto_unpause;
    bool watcher_previous_noinline;
    bool test_mode_previous;
    void SetUp() override {
        // Enable watcher for this test, save the previous state so we can restore it later.
        watcher_previous_enabled = tt::tt_metal::MetalContext::instance().rtoptions().get_watcher_enabled();
        watcher_previous_interval = tt::tt_metal::MetalContext::instance().rtoptions().get_watcher_interval();
        watcher_previous_dump_all = tt::tt_metal::MetalContext::instance().rtoptions().get_watcher_dump_all();
        watcher_previous_append = tt::tt_metal::MetalContext::instance().rtoptions().get_watcher_append();
        watcher_previous_auto_unpause = tt::tt_metal::MetalContext::instance().rtoptions().get_watcher_auto_unpause();
        watcher_previous_noinline = tt::tt_metal::MetalContext::instance().rtoptions().get_watcher_noinline();
        test_mode_previous = tt::tt_metal::MetalContext::instance().rtoptions().get_test_mode_enabled();
        tt::tt_metal::MetalContext::instance().rtoptions().set_watcher_enabled(true);
        tt::tt_metal::MetalContext::instance().rtoptions().set_watcher_interval(interval_ms);
        tt::tt_metal::MetalContext::instance().rtoptions().set_watcher_dump_all(false);
        tt::tt_metal::MetalContext::instance().rtoptions().set_watcher_append(false);
        tt::tt_metal::MetalContext::instance().rtoptions().set_watcher_auto_unpause(true);
        tt::tt_metal::MetalContext::instance().rtoptions().set_watcher_noinline(true);
        tt::tt_metal::MetalContext::instance().rtoptions().set_test_mode_enabled(true);
        tt::watcher_clear_log();

        // Parent class initializes devices and any necessary flags
        DebugToolsFixture::SetUp();
    }

    void TearDown() override {
        // Parent class tears down devices
        DebugToolsFixture::TearDown();

        // Reset watcher settings to their previous values
        tt::tt_metal::MetalContext::instance().rtoptions().set_watcher_interval(watcher_previous_interval);
        tt::tt_metal::MetalContext::instance().rtoptions().set_watcher_dump_all(watcher_previous_dump_all);
        tt::tt_metal::MetalContext::instance().rtoptions().set_watcher_append(watcher_previous_append);
        tt::tt_metal::MetalContext::instance().rtoptions().set_watcher_auto_unpause(watcher_previous_auto_unpause);
        tt::tt_metal::MetalContext::instance().rtoptions().set_watcher_noinline(watcher_previous_noinline);
        tt::tt_metal::MetalContext::instance().rtoptions().set_test_mode_enabled(test_mode_previous);
        tt::watcher_server_set_error_flag(false);
    }

    void RunTestOnDevice(
        const std::function<void(WatcherFixture*, IDevice*)>& run_function,
        IDevice* device
    ) {
        DebugToolsFixture::RunTestOnDevice(run_function, device);
        // Wait for a final watcher poll and then clear the log.
        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
        tt::watcher_clear_log();
    }
};

// A version of WatcherFixture with read and write debug delays enabled
class WatcherDelayFixture : public WatcherFixture {
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
        WatcherFixture::SetUp();
    }

    void TearDown() override {
        // Call parent
        WatcherFixture::TearDown();

        // Restore
        tt::tt_metal::MetalContext::instance().rtoptions().set_feature_targets(tt::llrt::RunTimeDebugFeatureReadDebugDelay, saved_target_selection[tt::llrt::RunTimeDebugFeatureReadDebugDelay]);
        tt::tt_metal::MetalContext::instance().rtoptions().set_feature_targets(tt::llrt::RunTimeDebugFeatureWriteDebugDelay, saved_target_selection[tt::llrt::RunTimeDebugFeatureWriteDebugDelay]);
        tt::tt_metal::MetalContext::instance().rtoptions().set_feature_targets(tt::llrt::RunTimeDebugFeatureAtomicDebugDelay, saved_target_selection[tt::llrt::RunTimeDebugFeatureAtomicDebugDelay]);
    }
};

} // namespace tt::tt_metal
