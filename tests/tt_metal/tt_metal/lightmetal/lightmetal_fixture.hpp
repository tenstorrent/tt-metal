// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dispatch_fixture.hpp"
#include <tt-metalium/device_impl.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <circular_buffer_constants.h>
#include <tt-metalium/kernel.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/lightmetal_replay.hpp>
#include "command_queue_fixture.hpp"
#include <lightmetal_binary.hpp>

// Flatbuffer roundtrip tests
#include "flatbuffer/program_types_to_flatbuffer.hpp"
#include "flatbuffer/program_types_from_flatbuffer.hpp"

class SingleDeviceLightMetalFixture : public CommandQueueFixture {
protected:
    bool replay_binary_;
    std::string trace_bin_path_;
    bool write_bin_to_disk_;
    bool replay_manages_device_;
    size_t trace_region_size_;

    void SetUp() override {
        this->validate_dispatch_mode();
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        TT_FATAL(!this->IsSlowDispatch(), "Test is running in slow dispatch mode, which is not supported.");
    }

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

    // Mimic the light-metal standalone run replay tool by executing the binary.
    // Note: Replay tool will open device if nullptr is provided.
    void RunLightMetalBinary(LightMetalBinary&& binary, IDevice* device) {
        tt::tt_metal::LightMetalReplay lm_replay(std::move(binary), device);
        bool success = lm_replay.execute_binary();
        if (!success) {
            FAIL() << "Light Metal Binary failed to execute or encountered errors.";
        } else {
            log_info(tt::LogMetalTrace, "Light Metal Binary executed successfully!");
        }
    }
};

class LightMetalFlatbufferRoundtripTest : public ::testing::Test {
protected:
    flatbuffers::FlatBufferBuilder fbb;

    template <typename FBType, typename T>
    T fb_roundtrip(const T& obj) {
        fbb.Clear();
        auto offset = to_flatbuffer(fbb, obj);
        fbb.Finish(offset);
        // Get the correct pointer to the actual FlatBuffer object
        const FBType* fb_ptr = flatbuffers::GetRoot<FBType>(fbb.GetBufferPointer());
        return from_flatbuffer(fb_ptr);
    }

    template <typename FBType, typename T>
    void VerifyFlatbufferRoundtrip(const T& obj, bool compare_to_orig = false) {
        fbb.Clear();
        auto offset = to_flatbuffer(fbb, obj);
        fbb.Finish(offset);

        // Store the first serialization result
        std::vector<uint8_t> original_buffer(fbb.GetBufferPointer(), fbb.GetBufferPointer() + fbb.GetSize());

        // Deserialize and reserialize
        const FBType* fb_ptr = flatbuffers::GetRoot<FBType>(fbb.GetBufferPointer());
        T obj_roundtrip = from_flatbuffer(fb_ptr);

        fbb.Clear();
        auto new_offset = to_flatbuffer(fbb, obj_roundtrip);
        fbb.Finish(new_offset);

        // Store the second serialization result
        std::vector<uint8_t> roundtrip_buffer(fbb.GetBufferPointer(), fbb.GetBufferPointer() + fbb.GetSize());

        log_info(
            tt::LogTest,
            "Original buffer size: {}, Roundtrip buffer size: {}",
            original_buffer.size(),
            roundtrip_buffer.size());
        // Ensure the serialized buffers match exactly
        ASSERT_EQ(original_buffer.size(), roundtrip_buffer.size());
        ASSERT_EQ(memcmp(original_buffer.data(), roundtrip_buffer.data(), original_buffer.size()), 0);

        // Optional, only supported by some object types.
        if (compare_to_orig) {
            // Ensure the roundtrip object matches the original object
            ASSERT_EQ(obj, obj_roundtrip);
        }
    }
};
