// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <thread>
#include "common_fixture.hpp"
#include "llrt/watcher.hpp"

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
    void SetUp() override {

        // Enable watcher for this test
        tt::llrt::OptionsG.set_watcher_enabled(true);
        tt::llrt::OptionsG.set_watcher_interval(interval_ms);
        tt::llrt::OptionsG.set_watcher_dump_all(false);
        tt::llrt::OptionsG.set_watcher_append(false);

        // Parent class initializes devices and any necessary flags
        CommonFixture::SetUp();
    }

    void TearDown() override {
        // Parent class tears down devices
        CommonFixture::TearDown();

        // Remove the watcher output file after the test is finished.
        std::remove(log_file_name.c_str());

        // Reset watcher settings
        tt::llrt::OptionsG.set_watcher_enabled(false);
        tt::llrt::OptionsG.set_watcher_interval(0);
        tt::llrt::OptionsG.set_watcher_dump_all(false);
        tt::llrt::OptionsG.set_watcher_append(false);
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
        tt::llrt::watcher_clear_log();
    }
};
