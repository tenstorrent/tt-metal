// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common_fixture.hpp"
#include "impl/debug/dprint_server.hpp"
#include "tt_metal/common/core_descriptor.hpp"

// A version of CommonFixture with DPrint enabled on all cores.
class DPrintFixture: public CommonFixture {
public:
    inline static const string dprint_file_name = "gtest_dprint_log.txt";

    // A function to run a program, according to which dispatch mode is set.
    void RunProgram(Device* device, ProgramHandle program) {
        // Only difference is that we need to wait for the print server to catch
        // up after running a test.
        CommonFixture::RunProgram(device, program);
        tt::DprintServerAwait();
    }

protected:
    // Running with dprint + watcher enabled can make the code size blow up, so let's force watcher
    // disabled for DPRINT tests.
    bool watcher_previous_enabled;
    void SetUp() override {
        // The core range (physical) needs to be set >= the set of all cores
        // used by all tests using this fixture, so set dprint enabled for
        // all cores and all devices
        tt::llrt::OptionsG.set_feature_enabled(tt::llrt::RunTimeDebugFeatureDprint, true);
        tt::llrt::OptionsG.set_feature_all_cores(
            tt::llrt::RunTimeDebugFeatureDprint, CoreType::WORKER, tt::llrt::RunTimeDebugClassWorker);
        tt::llrt::OptionsG.set_feature_all_cores(
            tt::llrt::RunTimeDebugFeatureDprint, CoreType::ETH, tt::llrt::RunTimeDebugClassWorker);
        tt::llrt::OptionsG.set_feature_all_chips(tt::llrt::RunTimeDebugFeatureDprint, true);
        // Send output to a file so the test can check after program is run.
        tt::llrt::OptionsG.set_feature_file_name(tt::llrt::RunTimeDebugFeatureDprint, dprint_file_name);
        tt::llrt::OptionsG.set_test_mode_enabled(true);
        watcher_previous_enabled = tt::llrt::OptionsG.get_watcher_enabled();
        tt::llrt::OptionsG.set_watcher_enabled(false);

        ExtraSetUp();

        // Parent class initializes devices and any necessary flags
        CommonFixture::SetUp();
    }

    void TearDown() override {
        // Parent class tears down devices
        CommonFixture::TearDown();

        // Remove the DPrint output file after the test is finished.
        std::remove(dprint_file_name.c_str());

        // Reset DPrint settings
        tt::llrt::OptionsG.set_feature_cores(tt::llrt::RunTimeDebugFeatureDprint, {});
        tt::llrt::OptionsG.set_feature_enabled(tt::llrt::RunTimeDebugFeatureDprint, false);
        tt::llrt::OptionsG.set_feature_all_cores(
            tt::llrt::RunTimeDebugFeatureDprint, CoreType::WORKER, tt::llrt::RunTimeDebugClassNoneSpecified);
        tt::llrt::OptionsG.set_feature_all_cores(
            tt::llrt::RunTimeDebugFeatureDprint, CoreType::ETH, tt::llrt::RunTimeDebugClassNoneSpecified);
        tt::llrt::OptionsG.set_feature_all_chips(tt::llrt::RunTimeDebugFeatureDprint, false);
        tt::llrt::OptionsG.set_feature_file_name(tt::llrt::RunTimeDebugFeatureDprint, "");
        tt::llrt::OptionsG.set_test_mode_enabled(false);
        tt::llrt::OptionsG.set_watcher_enabled(watcher_previous_enabled);
    }

    void RunTestOnDevice(
        const std::function<void(DPrintFixture*, Device*)>& run_function,
        Device* device
    ) {
        auto run_function_no_args = [=]() {
            run_function(this, device);
        };
        CommonFixture::RunTestOnDevice(run_function_no_args, device);
        tt::DPrintServerClearLogFile();
        tt::DPrintServerClearSignals();
    }

    // Override this function in child classes for additional setup commands between DPRINT setup
    // and device creation.
    virtual void ExtraSetUp() {}
};

// For usage by tests that need the dprint server devices disabled.
class DPrintFixtureDisableDevices: public DPrintFixture {
protected:
    void ExtraSetUp() override {
        // For this test, mute each devices using the environment variable
        tt::llrt::OptionsG.set_feature_all_chips(tt::llrt::RunTimeDebugFeatureDprint, false);
        tt::llrt::OptionsG.set_feature_chip_ids(tt::llrt::RunTimeDebugFeatureDprint, {});
    }
};
