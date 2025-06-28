// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dispatch_fixture.hpp"
#include "env_lib.hpp"
#include <tt-metalium/device.hpp>
#include "impl/context/metal_context.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <circular_buffer_constants.h>
#include <tt-metalium/kernel.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/lightmetal_replay.hpp>
#include "command_queue_fixture.hpp"
#include <lightmetal_binary.hpp>

namespace tt::tt_metal {

// This test suite manually manages the device creation and destruction
class SingleDeviceLightMetalFixture : public DispatchFixture<SingleDeviceLightMetalFixture> {
public:
    static bool WillSkip() {
        if (IsSlowDispatch()) {
            return true;
        }
        return false;
    }

    static std::string_view GetSkipMessage() { return "Requires fast dispatch"; }

    bool replay_binary_;
    std::string trace_bin_path_;
    bool write_bin_to_disk_;
    bool replay_manages_device_;
    size_t trace_region_size_;
    IDevice* device_;

    void create_device(const size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE) {
        const chip_id_t device_id = *tt::tt_metal::MetalContext::instance().get_cluster().all_chip_ids().begin();
        const auto dispatch_core_config = tt::tt_metal::MetalContext::instance().rtoptions().get_dispatch_core_config();
        this->device_ =
            tt::tt_metal::CreateDevice(device_id, 1, DEFAULT_L1_SMALL_SIZE, trace_region_size, dispatch_core_config);
    }

    static void SetUpTestSuite() {}
    static void TearDownTestSuite() {}

    void CreateDeviceAndBeginCapture(
        const size_t trace_region_size,
        const bool replay_manages_device = false,
        const bool replay_binary = true,
        const std::string trace_bin_path = "") {
        // Skip writing to disk by default, unless user sets env var for local testing
        write_bin_to_disk_ = tt::parse_env("LIGHTMETAL_SAVE_BINARY", false);
        replay_manages_device_ = replay_manages_device;
        trace_region_size_ = trace_region_size;

        // If user didn't provide a specific trace bin path, set a default here based on test name
        if (trace_bin_path == "") {
            const auto test_info = ::testing::UnitTest::GetInstance()->current_test_info();
            auto trace_filename = test_info ? std::string(test_info->name()) + ".bin" : "lightmetal_trace.bin";
            this->trace_bin_path_ = "/tmp/" + trace_filename;
        }

        this->create_device(trace_region_size_);
        this->replay_binary_ = replay_binary && !tt::parse_env("LIGHTMETAL_DISABLE_RUN", false);
        // TODO (kmabee) - revisit placement. CreateDevice() path calls CreateKernel() on programs not
        // created with CreateProgram() traced API which leads to "program not in global_id map"
        LightMetalBeginCapture();
    }

    // End light metal tracing, write to optional filename and optionally run from binary blob
    void TearDown() override {
        LightMetalBinary binary = LightMetalEndCapture();

        if (binary.is_empty()) {
            FAIL() << "Light Metal Binary is empty for test, unexpected.";
        }
        if (write_bin_to_disk_ && !this->trace_bin_path_.empty() && !binary.is_empty()) {
            log_info(tt::LogTest, "Writing light metal binary {} bytes to {}", binary.size(), this->trace_bin_path_);
            binary.save_to_file(this->trace_bin_path_);
        }

        tt::tt_metal::CloseDevice(this->device_);

        // We could gaurd this to not attempt to replay empty binary, and still allow test to pass
        // but, would rather catch the case if the feature gets disabled at compile time.
        if (replay_binary_) {
            if (replay_manages_device_) {
                RunLightMetalBinary(std::move(binary), nullptr);
            } else {
                this->create_device(trace_region_size_);
                RunLightMetalBinary(std::move(binary), this->device_);
                tt::tt_metal::CloseDevice(this->device_);
            }
        }
    }

    // Mimic the light-metal standalone run replay tool by running the binary.
    // Note: Replay tool will open device if nullptr is provided.
    void RunLightMetalBinary(LightMetalBinary&& binary, IDevice* device) {
        tt::tt_metal::LightMetalReplay lm_replay(std::move(binary), device);
        bool success = lm_replay.run();
        if (!success) {
            FAIL() << "Light Metal Binary failed to execute or encountered errors.";
        } else {
            log_info(tt::LogMetalTrace, "Light Metal Binary executed successfully!");
        }
    }
};

}  // namespace tt::tt_metal
