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
    void RunProgram(Device* device, Program& program) {
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
        tt::llrt::OptionsG.set_dprint_enabled(true);
        tt::llrt::OptionsG.set_dprint_all_cores(CoreType::WORKER, true);
        tt::llrt::OptionsG.set_dprint_all_cores(CoreType::ETH, true);
        tt::llrt::OptionsG.set_dprint_all_chips(true);
        // Send output to a file so the test can check after program is run.
        tt::llrt::OptionsG.set_dprint_file_name(dprint_file_name);
        tt::llrt::OptionsG.set_test_mode_enabled(true);
        watcher_previous_enabled = tt::llrt::OptionsG.get_watcher_enabled();
        tt::llrt::OptionsG.set_watcher_enabled(false);

        // By default, exclude dispatch cores from printing
        auto num_cqs_str = getenv("TT_METAL_NUM_HW_CQS");
        int num_cqs = (num_cqs_str != nullptr)? std::stoi(num_cqs_str) : 1;
        std::map<CoreType, std::unordered_set<CoreCoord>> disabled;
        for (unsigned int id = 0; id < tt::tt_metal::GetNumAvailableDevices(); id++) {
            // TODO: Better way to get this info once we've solidified how it will be set.
            const tt::core_descriptor_t &core_desc = tt::get_core_descriptor_config(id, num_cqs);
            for (auto core : tt::get_logical_dispatch_cores(id, num_cqs)) {
                log_info(tt::LogTest, "Disable dprint on Device {}: {}", id, core);
                disabled[core_desc.dispatch_core_type].insert(core);
            }
        }
        tt::llrt::OptionsG.set_dprint_disabled_cores(disabled);

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
        tt::llrt::OptionsG.set_dprint_cores({});
        tt::llrt::OptionsG.set_dprint_enabled(false);
        tt::llrt::OptionsG.set_dprint_all_cores(CoreType::WORKER, false);
        tt::llrt::OptionsG.set_dprint_all_cores(CoreType::ETH, false);
        tt::llrt::OptionsG.set_dprint_all_chips(false);
        tt::llrt::OptionsG.set_dprint_file_name("");
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
        tt::llrt::OptionsG.set_dprint_all_chips(false);
        tt::llrt::OptionsG.set_dprint_chip_ids({});
    }
};
