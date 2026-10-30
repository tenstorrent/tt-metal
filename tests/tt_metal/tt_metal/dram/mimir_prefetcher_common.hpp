// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Shared by the MM-7 Tensor Prefetcher tests on Mimir CCEs.

#pragma once

#include <gtest/gtest.h>

#include <fmt/format.h>

#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "impl/program/slow_dispatch.hpp"
#include "tt_metal/tt_metal/dram/mimir_access.hpp"
#include "tt_metal/tt_metal/test_kernels/cce/cce_gcb_receiver_abi.h"

namespace tt::tt_metal::mimir_test::prefetcher {

// Receiver kernels (cce_gcb_receiver_abi.h) running as their own Program while the test drives a sender. Declare it
// after the GlobalCircularBuffer it serves, so it stops before the GCB is freed. Launch zeroes each receiver's command
// line, so a STOP left by an earlier Program cannot end this one. stop_and_wait() writes kCommandStop to each command
// line, which ends a receiver's page wait and releases a stalling one, then waits for the Program. A throw from either
// step is reported, not propagated.
class CceReceiverProgram {
public:
    CceReceiverProgram(IDevice* device, std::vector<CceKernelLaunch> receivers) :
        device_(device), receivers_(std::move(receivers)) {
        for (const CceKernelLaunch& receiver : receivers_) {
            write_cce_tl1(device_, receiver.cce_index, command_address(receiver), {cce_gcb_receiver_abi::kCommandRun});
        }
        program_ = launch_cce_program(device_, receivers_);
    }
    CceReceiverProgram(const CceReceiverProgram&) = delete;
    CceReceiverProgram& operator=(const CceReceiverProgram&) = delete;
    ~CceReceiverProgram() { stop_and_wait(); }

    void stop_and_wait() {
        if (program_ == nullptr) {
            return;
        }
        for (const CceKernelLaunch& receiver : receivers_) {
            try {
                write_cce_tl1(
                    device_, receiver.cce_index, command_address(receiver), {cce_gcb_receiver_abi::kCommandStop});
            } catch (const std::exception& error) {
                ADD_FAILURE() << fmt::format("CCE {}: stop write threw: {}", receiver.cce_index, error.what());
            }
        }
        wait_for_exit();
    }

    // Waits for the receivers to exit on their own, without a STOP.
    void wait_for_exit() {
        std::unique_ptr<Program> program = std::move(program_);
        if (program == nullptr) {
            return;
        }
        try {
            slow_dispatch::WaitProgramDone(*device_, *program);
        } catch (const std::exception& error) {
            ADD_FAILURE() << "WaitProgramDone on the receiver Program threw: " << error.what();
        }
    }

private:
    static uint64_t command_address(const CceKernelLaunch& receiver) {
        return receiver.runtime_args.at(cce_gcb_receiver_abi::kArgBlockAddr) + cce_gcb_receiver_abi::kCommandLineOffset;
    }

    IDevice* device_;
    std::vector<CceKernelLaunch> receivers_;
    std::unique_ptr<Program> program_;
};

}  // namespace tt::tt_metal::mimir_test::prefetcher
