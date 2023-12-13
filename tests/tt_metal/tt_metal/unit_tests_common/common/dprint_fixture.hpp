// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"
#include "tt_metal/host_api.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/llrt/rtoptions.hpp"

// A version of CommandQueueFixture with DPrint enabled on the first core.
class DPrintFixture: public ::testing::Test {
public:
    inline static const string dprint_file_name = "gtest_dprint_log.txt";
protected:
    tt::ARCH arch_;
    Device* device_;

    void SetUp() override {
        // Skip for slow dispatch for now
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch) {
            //TT_THROW("This suite can only be run with fast dispatch or TT_METAL_SLOW_DISPATCH_MODE unset");
            test_skipped = true;
            GTEST_SKIP();
        }
        // The core range (physical) needs to be set >= the set of all cores
        // used by all tests using this fixture. TODO: update with a way to
        // just set all physical cores to have printing enabled.
        tt::llrt::OptionsG.set_dprint_all_cores(true);
        // Send output to a file so the test can check after program is run.
        tt::llrt::OptionsG.set_dprint_file_name(dprint_file_name);

        // DPrint currently not supported for N300, so skip these tests for
        // now. Seeing some hanging in device setup with DPrint enabled on N300
        // so skip before creating the device.
        // TODO: remove this once N300 DPrint is working.
        auto arch = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());
        auto num_devices = tt::tt_metal::GetNumAvailableDevices();
        auto num_pci_devices = tt::tt_metal::GetNumPCIeDevices();
        if (arch == tt::ARCH::WORMHOLE_B0 and
            num_devices == 2 and
            num_pci_devices == 1) {
            tt::log_info(tt::LogTest, "DPrint tests skipped on N300 for now.");
            test_skipped = true;
            GTEST_SKIP();
        }

        // Parent call, sets up the device
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());

        const int device_id = 0;
        this->device_ = tt::tt_metal::CreateDevice(device_id);

    }

    void TearDown() override {
        if (!test_skipped) {
            tt::tt_metal::CloseDevice(this->device_);
            // Remove the DPrint output file after the test is finished.
            std::remove(dprint_file_name.c_str());
        }

        // Reset DPrint settings
        tt::llrt::OptionsG.set_dprint_cores({});
        tt::llrt::OptionsG.set_dprint_all_cores(false);
        tt::llrt::OptionsG.set_dprint_file_name("");
    }

    // A flag to mark if the test is skipped or not. Since we skip before
    // device setup, we need to skip device teardown if the test is skipped.
    bool test_skipped = false;
};
