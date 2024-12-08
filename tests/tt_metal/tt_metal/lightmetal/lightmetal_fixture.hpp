// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// #include "command_queue_fixture.hpp"
#include "dispatch_fixture.hpp"
#include "impl/device/device.hpp"
#include "llrt/hal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/hw/inc/circular_buffer_constants.h"
#include "tt_metal/impl/kernels/kernel.hpp"
#include "tt_metal/common/tt_backend_api_types.hpp"
// #include "dispatch_test_utils.hpp"
#include "tt_metal/impl/lightmetal/lightmetal_replay.hpp"
#include "command_queue_fixture.hpp"

class SingleDeviceLightMetalFixture : public CommandQueueFixture {
protected:
    bool replay_binary_;
    std::string trace_bin_path_;
    bool write_bin_to_disk_;

    void SetUp() override {
        this->validate_dispatch_mode();
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
    }

    void CreateDevice(
        const size_t trace_region_size, const bool replay_binary = false, const std::string trace_bin_path = "") {
        // Skip writing to disk by default, unless user sets env var for local testing
        write_bin_to_disk_ = std::getenv("LIGHTMETAL_SAVE_BINARY") or false;

        // If user didn't provide a specific trace bin path, set a default here based on test name
        if (trace_bin_path == "") {
            const auto test_info = ::testing::UnitTest::GetInstance()->current_test_info();
            auto trace_filename = test_info ? std::string(test_info->name()) + ".bin" : "lightmetal_trace.bin";
            this->trace_bin_path_ = "/tmp/" + trace_filename;
        }

        log_info(
            tt::LogTest,
            "Starting CreateDevice w/ replay_binary: {} trace_bin_path: {}",
            replay_binary,
            this->trace_bin_path_);
        this->create_device(trace_region_size);
        this->replay_binary_ = replay_binary || ((bool) std::getenv("LIGHTMETAL_RUN") == true);
        LightMetalBeginCapture(this->device_);
    }

    // End light metal tracing, write to optional filename (for standalone replay testing), and optionally run from binary blob
    void TearDown() override {
        log_info(tt::LogTest, "Starting TearDown w/ replay_binary: {}", replay_binary_);

        auto blob = LightMetalEndCapture(this->device_);
        if (write_bin_to_disk_ && !this->trace_bin_path_.empty()) {
            writeBlobToFile(this->trace_bin_path_, blob);
        }

        if (!this->IsSlowDispatch()) {
            tt::tt_metal::CloseDevice(this->device_);
        }

        if (replay_binary_) {
            runLightMetalBinary(blob);
        }
    }

    // Just write, limited error checking.
    bool writeBlobToFile(const std::string& filename, const std::vector<uint8_t>& blob) {
        log_info(tt::LogTest, "Writing blob of {} bytes to file: {}", blob.size(), filename);
        std::ofstream outFile(filename, std::ios::binary);
        outFile.write(reinterpret_cast<const char*>(blob.data()), blob.size());
        return outFile.good();
    }

    // Mimic the light-metal standalone run replay tool by executing the binary.
    void runLightMetalBinary(const std::vector<uint8_t>& blob) {
        tt::tt_metal::LightMetalReplay lm_replay(blob);
        if (!lm_replay.executeLightMetalBinary()) {
            tt::log_fatal("Light Metal Binary failed to execute or encountered errors.");
        } else {
            log_info(tt::LogMetalTrace, "Light Metal Binary executed successfully!");
        }
    }
};
