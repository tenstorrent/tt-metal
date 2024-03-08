// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <thread>
#include "common_fixture.hpp"
#include "impl/debug/watcher_server.hpp"

// A version of CommonFixture with watcher enabled
class WatcherFixture: public CommonFixture {
public:
    inline static const string log_file_name = "built/watcher.log";
    inline static const int interval_ms = 250;

    // A function to run a program, according to which dispatch mode is set.
    void RunProgram(Device* device, Program& program) {
        // Only difference is that we need to wait for the print server to catch
        // up after running a test.
        CommonFixture::RunProgram(device, program);
        // Wait for a watcher interval to make sure that we get a reading after program finish.
        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
    }

protected:
    bool watcher_previous_enabled;
    int  watcher_previous_interval;
    bool watcher_previous_dump_all;
    bool watcher_previous_append;
    bool test_mode_previous;
    void SetUp() override {
        // Enable watcher for this test, save the previous state so we can restore it later.
        watcher_previous_enabled = tt::llrt::OptionsG.get_watcher_enabled();
        watcher_previous_interval = tt::llrt::OptionsG.get_watcher_interval();
        watcher_previous_dump_all = tt::llrt::OptionsG.get_watcher_dump_all();
        watcher_previous_append = tt::llrt::OptionsG.get_watcher_append();
        test_mode_previous = tt::llrt::OptionsG.get_test_mode_enabled();
        tt::llrt::OptionsG.set_watcher_enabled(true);
        tt::llrt::OptionsG.set_watcher_interval(interval_ms);
        tt::llrt::OptionsG.set_watcher_dump_all(false);
        tt::llrt::OptionsG.set_watcher_append(false);
        tt::llrt::OptionsG.set_test_mode_enabled(true);
        tt::watcher_clear_log();

        // Parent class initializes devices and any necessary flags
        CommonFixture::SetUp();
    }

    void TearDown() override {
        // Parent class tears down devices
        CommonFixture::TearDown();

        // Remove the watcher output file after the test is finished.
        std::remove(log_file_name.c_str());

        // Reset watcher settings to their previous values
        tt::llrt::OptionsG.set_watcher_enabled(watcher_previous_enabled);
        tt::llrt::OptionsG.set_watcher_interval(watcher_previous_interval);
        tt::llrt::OptionsG.set_watcher_dump_all(watcher_previous_dump_all);
        tt::llrt::OptionsG.set_watcher_append(watcher_previous_append);
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
