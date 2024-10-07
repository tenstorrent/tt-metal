// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <thread>
#include "common_fixture.hpp"
#include "impl/debug/watcher_server.hpp"
#include "llrt/rtoptions.hpp"

// A version of CommonFixture with watcher enabled
class WatcherFixture: public CommonFixture {
public:
    inline static const string log_file_name = "generated/watcher/watcher.log";
    inline static const int interval_ms = 250;

    // A function to run a program, according to which dispatch mode is set.
    void RunProgram(Device* device, ProgramHandle program, bool wait_for_dump = false) {
        // Only difference is that we need to wait for the print server to catch
        // up after running a test.
        CommonFixture::RunProgram(device, program);

        // Wait for watcher to run a full dump before finishing, need to wait for dump count to
        // increase because we'll likely check in the middle of a dump.
        if (wait_for_dump) {
            int curr_count = tt::watcher_get_dump_count();
            while (tt::watcher_get_dump_count() < curr_count + 2) {;}
        }
    }

protected:
    bool watcher_previous_enabled;
    int  watcher_previous_interval;
    bool watcher_previous_dump_all;
    bool watcher_previous_append;
    bool watcher_previous_auto_unpause;
    bool watcher_previous_noinline;
    bool test_mode_previous;
    void SetUp() override {
        // Enable watcher for this test, save the previous state so we can restore it later.
        watcher_previous_enabled = tt::llrt::OptionsG.get_watcher_enabled();
        watcher_previous_interval = tt::llrt::OptionsG.get_watcher_interval();
        watcher_previous_dump_all = tt::llrt::OptionsG.get_watcher_dump_all();
        watcher_previous_append = tt::llrt::OptionsG.get_watcher_append();
        watcher_previous_auto_unpause = tt::llrt::OptionsG.get_watcher_auto_unpause();
        watcher_previous_noinline = tt::llrt::OptionsG.get_watcher_noinline();
        test_mode_previous = tt::llrt::OptionsG.get_test_mode_enabled();
        tt::llrt::OptionsG.set_watcher_enabled(true);
        tt::llrt::OptionsG.set_watcher_interval(interval_ms);
        tt::llrt::OptionsG.set_watcher_dump_all(false);
        tt::llrt::OptionsG.set_watcher_append(false);
        tt::llrt::OptionsG.set_watcher_auto_unpause(true);
        tt::llrt::OptionsG.set_watcher_noinline(true);
        tt::llrt::OptionsG.set_test_mode_enabled(true);
        tt::watcher_clear_log();

        // Parent class initializes devices and any necessary flags
        CommonFixture::SetUp();
    }

    void TearDown() override {
        // Parent class tears down devices
        CommonFixture::TearDown();

        // Reset watcher settings to their previous values
        tt::llrt::OptionsG.set_watcher_enabled(watcher_previous_enabled);
        tt::llrt::OptionsG.set_watcher_interval(watcher_previous_interval);
        tt::llrt::OptionsG.set_watcher_dump_all(watcher_previous_dump_all);
        tt::llrt::OptionsG.set_watcher_append(watcher_previous_append);
        tt::llrt::OptionsG.set_watcher_auto_unpause(watcher_previous_auto_unpause);
        tt::llrt::OptionsG.set_watcher_noinline(watcher_previous_noinline);
        tt::llrt::OptionsG.set_test_mode_enabled(test_mode_previous);
        tt::watcher_server_set_error_flag(false);
    }

    void RunTestOnDevice(
        const std::function<void(WatcherFixture*, Device*)>& run_function,
        Device* device
    ) {
        auto run_function_no_args = [=]() {
            run_function(this, device);
        };
        CommonFixture::RunTestOnDevice(run_function_no_args, device);
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
        tt::llrt::OptionsG.set_watcher_debug_delay(5000000);
        delayed_cores[CoreType::WORKER] = {{0, 0}, {1, 1}};

        // Store the previous state of the watcher features
        saved_target_selection[tt::llrt::RunTimeDebugFeatureReadDebugDelay] = tt::llrt::OptionsG.get_feature_targets(tt::llrt::RunTimeDebugFeatureReadDebugDelay);
        saved_target_selection[tt::llrt::RunTimeDebugFeatureWriteDebugDelay] = tt::llrt::OptionsG.get_feature_targets(tt::llrt::RunTimeDebugFeatureWriteDebugDelay);
        saved_target_selection[tt::llrt::RunTimeDebugFeatureAtomicDebugDelay] = tt::llrt::OptionsG.get_feature_targets(tt::llrt::RunTimeDebugFeatureAtomicDebugDelay);

        // Enable read and write debug delay for the test core
        tt::llrt::OptionsG.set_feature_enabled(tt::llrt::RunTimeDebugFeatureReadDebugDelay, true);
        tt::llrt::OptionsG.set_feature_cores(tt::llrt::RunTimeDebugFeatureReadDebugDelay, delayed_cores);
        tt::llrt::OptionsG.set_feature_enabled(tt::llrt::RunTimeDebugFeatureWriteDebugDelay, true);
        tt::llrt::OptionsG.set_feature_cores(tt::llrt::RunTimeDebugFeatureWriteDebugDelay, delayed_cores);

        // Call parent
        WatcherFixture::SetUp();
    }

    void TearDown() override {
        // Call parent
        WatcherFixture::TearDown();

        // Restore
        tt::llrt::OptionsG.set_feature_targets(tt::llrt::RunTimeDebugFeatureReadDebugDelay, saved_target_selection[tt::llrt::RunTimeDebugFeatureReadDebugDelay]);
        tt::llrt::OptionsG.set_feature_targets(tt::llrt::RunTimeDebugFeatureWriteDebugDelay, saved_target_selection[tt::llrt::RunTimeDebugFeatureWriteDebugDelay]);
        tt::llrt::OptionsG.set_feature_targets(tt::llrt::RunTimeDebugFeatureAtomicDebugDelay, saved_target_selection[tt::llrt::RunTimeDebugFeatureAtomicDebugDelay]);
    }
};
