// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Shared by the Tensor Prefetcher tests (test_mimir_prefetcher.cpp) and test_mimir_trid.cpp.

#pragma once

#include <gtest/gtest.h>

#include <fmt/format.h>

#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "distributed/mesh_device_impl.hpp"
#include "impl/buffers/tensor_prefetcher_manager.hpp"
#include "impl/program/slow_dispatch.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/tensor_prefetcher.hpp>
#include "tt_metal/tt_metal/dram/mimir_access.hpp"
#include "tt_metal/tt_metal/test_kernels/cce/cce_gcb_receiver_abi.h"

namespace tt::tt_metal::mimir_test::prefetcher {

inline constexpr const char* kGcbDiscardReceiverKernel =
    "tests/tt_metal/tt_metal/test_kernels/cce/cce_gcb_discard_receiver.cpp";
inline constexpr uint32_t kUint32TileWords = tt::constants::TILE_HW;
inline constexpr uint32_t kUint32TilePageBytes = kUint32TileWords * sizeof(uint32_t);

// rdcycle budgets per wait. On the Mimir emulator the chip clock was measured at ~85,000 cycles/s while the host polls
// it and ~90 cycles/s while the host is idle, so these count host activity rather than wall time. The wait budget
// bounds waits that should succeed (~200 s of polling, under TT_METAL_OPERATION_TIMEOUT_SECONDS=600); the stall budget
// is the sender's in the stall case (~12 s of polling), which must expire while the receiver holds its credits.
inline constexpr uint64_t kGcbSmokeWaitBudgetCycles = 1ull << 24;
inline constexpr uint64_t kGcbSmokeStallBudgetCycles = 1ull << 20;

// The DRAM bank whose prefetcher sender is CCE `cce_index`, if any.
inline std::optional<uint32_t> bank_with_sender_cce(distributed::MeshDevice& mesh, uint32_t cce_index) {
    IDevice* device = mesh.get_devices().front();
    const metal_SocDescriptor& soc = soc_desc_of(device);
    for (uint32_t bank = 0; bank < soc.get_num_dram_views(); ++bank) {
        const CoreCoord sender_logical = mesh.impl().dram_sender_logical_cores(device, bank).front();
        if (soc.get_physical_dram_core_from_logical(sender_logical) == translated_dram_core(soc, cce_index)) {
            return bank;
        }
    }
    return std::nullopt;
}

// A Tensor prefetcher session on the CCE senders of `sender_banks`. The default stage ring's third holds the widest
// K-row of the one-tile-wide tensors these tests write (one 4 KiB UINT32 tile). Declare it after any CceReceiverProgram
// it serves, so it stops first and Stop's DRAIN sees live receivers. The destructor stops through the no-throw path,
// which only logs a failure; call stop() to check the outcome.
class PrefetcherSession {
public:
    PrefetcherSession(
        distributed::MeshDevice& mesh,
        std::vector<uint32_t> sender_banks,
        uint64_t wait_budget_cycles = kGcbSmokeWaitBudgetCycles,
        uint32_t stage_ring_bytes = distributed::QuasarTensorPrefetcherConfig{}.stage_ring_bytes) :
        mesh_(mesh) {
        distributed::start_tensor_prefetcher(
            mesh_,
            {},
            distributed::QuasarTensorPrefetcherConfig{
                .dram_sender_banks = std::move(sender_banks),
                .wait_budget_cycles = wait_budget_cycles,
                .stage_ring_bytes = stage_ring_bytes});
        active_ = true;
    }
    PrefetcherSession(const PrefetcherSession&) = delete;
    PrefetcherSession& operator=(const PrefetcherSession&) = delete;

    ~PrefetcherSession() {
        if (!active_) {
            return;
        }
        mesh_.impl().tensor_prefetcher(&mesh_).stop_without_throwing();
    }

    void stop() {
        active_ = false;
        experimental::StopTensorPrefetcher(mesh_);
    }

private:
    distributed::MeshDevice& mesh_;
    bool active_ = false;
};

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
