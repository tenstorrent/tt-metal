// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Shared by the MM-7 Tensor Prefetcher tests on Mimir CCEs.

#pragma once

#include <gtest/gtest.h>

#include <fmt/format.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "distributed/mesh_device_impl.hpp"
#include "impl/buffers/dram_receiver_config_page.hpp"
#include "impl/buffers/drisc_l1_arena.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/dataflow_buffer/prefetcher_pipe.hpp"
#include "llrt/hal.hpp"
#include "llrt/tt_cluster.hpp"
#include <tt-metalium/experimental/prefetcher_pipe.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "tt_metal/tt_metal/dram/mimir_access.hpp"
#include "tt_metal/tt_metal/dram/mimir_hw_state.hpp"
#include "tt_metal/tt_metal/dram/mimir_test_common.hpp"
#include "tt_metal/tt_metal/test_kernels/cce/cce_gcb_receiver_abi.h"

namespace tt::tt_metal::mimir_test::prefetcher {

// Owns one open MeshDevice. The destructor closes it on every path; a throw from close() is reported, not propagated.
class ScopedMesh {
public:
    explicit ScopedMesh(std::shared_ptr<distributed::MeshDevice> mesh) : mesh_(std::move(mesh)) {}
    ScopedMesh(const ScopedMesh&) = delete;
    ScopedMesh& operator=(const ScopedMesh&) = delete;
    ~ScopedMesh() { close(); }

    void close() {
        std::shared_ptr<distributed::MeshDevice> mesh = std::move(mesh_);
        mesh_.reset();
        if (mesh != nullptr) {
            try {
                EXPECT_TRUE(mesh->close());
            } catch (const std::exception& error) {
                ADD_FAILURE() << "MeshDevice::close threw: " << error.what();
            }
        }
    }

private:
    std::shared_ptr<distributed::MeshDevice> mesh_;
};

using BankToDramReceivers = std::vector<std::pair<uint32_t, std::vector<CoreCoord>>>;

// One DRAM-receiver GCB: a PrefetcherPipeL1Source::DramReceiverZone space of `size` bytes per receiver and the
// pipes CreatePrefetcherPipesForTensorPrefetcher carves from it, one per bank. The space is sized for exactly
// these pipes, so a second DramReceiverGcb over the same CCEs takes its own ranges in both DRISC L1 zones.
class DramReceiverGcb {
public:
    DramReceiverGcb(distributed::MeshDevice& mesh, const BankToDramReceivers& bank_to_dram_receivers, uint32_t size) :
        space_(create_space(mesh, bank_to_dram_receivers, size)),
        pipes_(experimental::CreatePrefetcherPipesForTensorPrefetcher(space_, bank_to_dram_receivers)) {}

    uint64_t config_address() const { return space_.config_address(); }
    uint64_t buffer_address() const { return space_.buffer_address(); }
    uint32_t size() const { return space_.ring_size(); }
    // The pipes, in bank order, as QueueTensorPrefetcherRequest takes them.
    std::vector<std::reference_wrapper<const experimental::PrefetcherPipe>> pipes() const {
        return {pipes_.begin(), pipes_.end()};
    }

    // Every sender holds its counters and state block at these two addresses.
    uint64_t pages_sent_drisc_l1_base() const { return pipes_.front().impl().pages_sent_drisc_l1_base(); }
    uint64_t sender_state_drisc_l1_base() const { return pipes_.front().impl().sender_state_drisc_l1_base(); }
    std::shared_ptr<DriscL1Allocation> sender_state_drisc_l1_allocation() const {
        return pipes_.front().impl().sender_state_drisc_l1_allocation();
    }
    // The address every receiver holds its pages_sent credit line at.
    uint64_t pages_sent_worker_l1_base() const { return config_address() + kDramReceiverCreditLineOffset; }
    std::vector<std::vector<CoreCoord>> receiver_logical_cores_per_sender() const {
        std::vector<std::vector<CoreCoord>> receivers;
        for (const experimental::PrefetcherPipe& pipe : pipes_) {
            receivers.push_back(pipe.impl().dram_sender_receiver_table());
        }
        return receivers;
    }

private:
    static experimental::PrefetcherPipeSpace create_space(
        distributed::MeshDevice& mesh, const BankToDramReceivers& bank_to_dram_receivers, uint32_t size) {
        // A set, so a list the factory will reject for naming one coord twice still makes a valid domain.
        std::set<CoreCoord> domain;
        for (const auto& [bank, receivers] : bank_to_dram_receivers) {
            domain.insert(receivers.begin(), receivers.end());
        }
        const std::vector<CoreCoord> domain_cores(domain.begin(), domain.end());
        uint32_t max_receivers = 1;
        for (const auto& [bank, receivers] : bank_to_dram_receivers) {
            max_receivers = std::max(max_receivers, static_cast<uint32_t>(receivers.size()));
        }
        return experimental::CreatePrefetcherPipeSpace(
            mesh.mesh_command_queue(),
            experimental::PrefetcherPipeSpaceConfig{
                .num_dram_senders = static_cast<uint32_t>(bank_to_dram_receivers.size()),
                .receiver_domain = CoreRangeSet(ttsl::Span<const CoreCoord>(domain_cores)),
                .ring_size = size,
                .max_receivers_per_pipe = std::min(max_receivers, static_cast<uint32_t>(domain_cores.size())),
                .l1_source = experimental::PrefetcherPipeL1Source::DramReceiverZone});
    }

    experimental::PrefetcherPipeSpace space_;
    std::vector<experimental::PrefetcherPipe> pipes_;
};

inline uint64_t go_message_tl1_offset() {
    return MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::GO_MSG);
}

// The whole GO_MSG word of one CCE.
inline uint32_t read_go_word(IDevice* device, uint32_t cce_index) {
    return read_cce_tl1(
               MetalContext::instance().get_cluster(),
               device->id(),
               cce_index,
               go_message_tl1_offset(),
               sizeof(uint32_t),
               Tl1Range::WholeTl1)
        .front();
}

inline uint8_t go_signal_of(uint32_t go_word) {
    dev_msgs::go_msg_t go_message =
        MetalContext::instance().hal().get_dev_msgs_factory(HalProgrammableCoreType::DRAM).create<dev_msgs::go_msg_t>();
    TT_FATAL(go_message.size() == sizeof(uint32_t), "go_msg_t is {} bytes, not one word", go_message.size());
    std::memcpy(go_message.data(), &go_word, sizeof(go_word));
    return go_message.view().signal();
}

inline uint8_t read_go_signal(IDevice* device, uint32_t cce_index) {
    return go_signal_of(read_go_word(device, cce_index));
}

// The hart an MM-7 test kernel runs on unless its launch names another.
inline constexpr uint32_t kCceKernelHart = 0;

// One kernel of a CCE Program. Kernels on one CCE run on different harts.
struct CceKernelLaunch {
    uint32_t cce_index = 0;
    std::string kernel_path;
    std::vector<uint32_t> runtime_args;
    uint32_t hart = kCceKernelHart;
};

// A launch onto a CCE that is still running a kernel is lost: when that kernel finishes, firmware overwrites the new GO
// with DONE. So every CCE must read RUN_MSG_DONE first.
inline void require_idle_cces(IDevice* device, const std::vector<CceKernelLaunch>& launches) {
    for (const CceKernelLaunch& launch : launches) {
        const uint8_t signal = read_go_signal(device, launch.cce_index);
        TT_FATAL(
            signal == dev_msgs::RUN_MSG_DONE,
            "CCE {} GO_MSG signal is 0x{:02x}, not RUN_MSG_DONE (0x{:02x}); it is still running a kernel or lost "
            "its firmware",
            launch.cce_index,
            signal,
            dev_msgs::RUN_MSG_DONE);
    }
}

// Checks that every CCE is idle, then launches one Program holding every kernel and returns without waiting. The
// caller ends it with detail::WaitProgramDone; a blocking slow-dispatch mesh workload would stall every later host
// access instead.
inline std::unique_ptr<Program> launch_cce_program(IDevice* device, const std::vector<CceKernelLaunch>& launches) {
    require_idle_cces(device, launches);
    const metal_SocDescriptor& soc_desc = soc_desc_of(device);
    std::unique_ptr<Program> program = std::make_unique<Program>(CreateProgram());
    for (const CceKernelLaunch& launch : launches) {
        const CoreCoord logical_core = logical_cce_core(soc_desc, launch.cce_index);
        const KernelHandle kernel = CreateKernel(
            *program,
            launch.kernel_path,
            logical_core,
            DramConfig{.processor = static_cast<DataMovementProcessor>(launch.hart), .noc = NOC::NOC_0});
        SetRuntimeArgs(*program, kernel, logical_core, launch.runtime_args);
    }
    detail::LaunchProgram(device, *program, /*wait_until_cores_done=*/false, /*force_slow_dispatch=*/true);
    return program;
}

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
            detail::WaitProgramDone(device_, *program, /*read_device_profiler_results=*/false);
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
