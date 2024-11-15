// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <gtest/gtest.h>

#include "host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/impl/device/device_pool.hpp"
#include "command_queue_fixture.hpp"

class SingleDeviceTraceFixture : public ::testing::Test {
   protected:
    Device* device_;
    tt::ARCH arch_;

    void Setup(const size_t buffer_size, const uint8_t num_hw_cqs = 1) {
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch) {
            tt::log_info(
                tt::LogTest, "This suite can only be run with fast dispatch or TT_METAL_SLOW_DISPATCH_MODE unset");
            GTEST_SKIP();
        }
        if (num_hw_cqs > 1) {
            // Running multi-CQ test. User must set this explicitly.
            auto num_cqs = getenv("TT_METAL_GTEST_NUM_HW_CQS");
            if (num_cqs == nullptr or strcmp(num_cqs, "2")) {
                TT_THROW("This suite must be run with TT_METAL_GTEST_NUM_HW_CQS=2");
                GTEST_SKIP();
            }
        }
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        const int device_id = 0;
        this->device_ = tt::tt_metal::CreateDevice(device_id, num_hw_cqs, 0, buffer_size);
    }

    void TearDown() override {
        if (!getenv("TT_METAL_SLOW_DISPATCH_MODE")) {
            tt::tt_metal::CloseDevice(this->device_);
        }
    }
};

class RandomProgramTraceFixture : virtual public RandomProgramFixture, virtual public CommandQueueSingleCardTraceFixture {
   protected:
    static const uint32_t NUM_TRACE_ITERATIONS = 50;
    Program programs[NUM_PROGRAMS];

    void SetUp() override {
        CommandQueueSingleCardTraceFixture::SetUp();
        this->device_ = this->devices_[0];
        this->initialize_seed();
    }

    uint32_t trace_programs() {
        const uint32_t trace_id = this->capture_trace();
        this->run_trace(trace_id);
        return trace_id;
    }

   private:
    uint32_t capture_trace() {
        const uint32_t trace_id = BeginTraceCapture(this->device_, this->device_->command_queue().id());
        for (Program &program : this->programs) {
            EnqueueProgram(this->device_->command_queue(), program, false);
        }
        EndTraceCapture(this->device_, this->device_->command_queue().id(), trace_id);
        return trace_id;
    }

    void run_trace(const uint32_t trace_id) {
        for (uint32_t i = 0; i < NUM_TRACE_ITERATIONS; i++) {
            EnqueueTrace(this->device_->command_queue(), trace_id, false);
        }
    }
};
