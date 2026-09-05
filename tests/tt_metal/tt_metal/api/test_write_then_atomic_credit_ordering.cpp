// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Data-before-credit ordering when the credit is a remote ATOMIC.
//
// A sender writes a payload into a peer core's L1, drains with
// noc_async_writes_flushed() (which polls NIU_MST_NONPOSTED_WR_REQ_SENT -- the
// request has LEFT this NIU) and then credits the peer with noc_semaphore_inc.
//
// WormholeB0/NoC/Ordering.md enumerates the recipient-NIU ordering guarantees for
// two packets arriving on the same virtual channel: write->read, linked-read->any,
// atomic->atomic, write->write, and two MMIO cases. write->atomic is NOT among
// them, so the credit may be applied before the payload commits. The payload write
// and the atomic also use different command buffers (write_cmd_buf vs
// write_at_cmd_buf). The barrier form (noc_async_write_barrier, which polls
// NIU_MST_WR_ACK_RECEIVED) has no such gap.
//
// The kernels tag every payload word with the iteration number and the receiver
// checks the packet tail -- the last thing to commit -- immediately after the
// credit lands. A stale tag is a credit observed ahead of its data.

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "device_fixture.hpp"
#include "impl/context/metal_context.hpp"
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"

namespace tt::tt_metal {

namespace {

constexpr uint32_t kPayloadBytes = 262144;
constexpr uint32_t kIters = 500;
// Mirrors of the device-side VC ids (dataflow_api_common.h is a kernel header).
constexpr uint32_t kUnicastWriteVc = 1;
constexpr uint32_t kOtherVc = 4;

struct ProbeResult {
    uint32_t stale_publishes;
    uint32_t stale_words;
    uint32_t first_bad_iter;
    uint32_t iters;
};

// Runs the sender/receiver pair once. `use_barrier` selects the ack-wait
// (noc_async_write_barrier) instead of the send-wait (noc_async_writes_flushed).
ProbeResult run_probe(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    bool use_barrier,
    uint32_t credit_vc = kUnicastWriteVc,
    bool credit_first = false) {
    const CoreCoord sender_core{0, 0};
    const CoreCoord receiver_core{0, 1};

    auto& cq = mesh_device->mesh_command_queue();
    const auto zero_coord = distributed::MeshCoordinate(0, 0);
    const auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

    const uint32_t l1_base = mesh_device->allocator()->get_base_allocator_addr(HalMemType::L1);
    const uint32_t alignment = MetalContext::instance().hal().get_alignment(HalMemType::L1);
    const auto align_up = [alignment](uint32_t v) { return ((v + alignment - 1) / alignment) * alignment; };

    // Receiver L1 layout: payload | result. Sender L1 layout: payload source.
    const uint32_t payload_addr = align_up(l1_base);
    const uint32_t result_addr = align_up(payload_addr + kPayloadBytes);
    const uint32_t src_addr = payload_addr;

    // Zero the receiver's payload tail and result so a missing write is never
    // mistaken for a correct tag (tags start at 1).
    std::vector<uint32_t> zeros(kPayloadBytes / sizeof(uint32_t), 0);
    slow_dispatch::WriteToL1(*mesh_device, receiver_core, payload_addr, zeros);
    std::vector<uint32_t> zero_result(alignment / sizeof(uint32_t), 0);
    slow_dispatch::WriteToL1(*mesh_device, receiver_core, result_addr, zero_result);

    Program program = CreateProgram();
    // data_sem lives on the receiver, ack_sem on the sender; CreateSemaphore over
    // both cores keeps a single id valid on each.
    const CoreRangeSet both_cores(std::vector<CoreRange>{CoreRange(sender_core), CoreRange(receiver_core)});
    const uint32_t data_sem_id = CreateSemaphore(program, both_cores, 0);
    const uint32_t ack_sem_id = CreateSemaphore(program, both_cores, 0);

    distributed::MeshWorkload workload;
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);

    const CoreCoord virtual_sender = mesh_device->worker_core_from_logical_core(sender_core);
    const CoreCoord virtual_receiver = mesh_device->worker_core_from_logical_core(receiver_core);

    const KernelHandle sender = CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/write_then_atomic_credit_sender.cpp",
        sender_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0});
    SetRuntimeArgs(
        program_,
        sender,
        sender_core,
        {virtual_receiver.x,
         virtual_receiver.y,
         payload_addr,
         kPayloadBytes,
         data_sem_id,
         ack_sem_id,
         kIters,
         src_addr,
         static_cast<uint32_t>(use_barrier),
         credit_vc,
         static_cast<uint32_t>(credit_first)});

    const KernelHandle receiver = CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/write_then_atomic_credit_receiver.cpp",
        receiver_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0});
    SetRuntimeArgs(
        program_,
        receiver,
        receiver_core,
        {virtual_sender.x,
         virtual_sender.y,
         payload_addr,
         kPayloadBytes,
         data_sem_id,
         ack_sem_id,
         kIters,
         result_addr});

    distributed::EnqueueMeshWorkload(cq, workload, true);

    std::vector<uint32_t> readback;
    slow_dispatch::ReadFromL1(*mesh_device, receiver_core, result_addr, alignment, readback);
    return ProbeResult{readback[0], readback[1], readback[2], readback[3]};
}

}  // namespace

// The ack-wait form must never publish a credit ahead of its payload. This is the
// half that has to stay green: it is the contract the fix relies on.
TEST_F(MeshDeviceFixture, TensixWriteThenAtomicCreditBarrierIsOrdered) {
    for (const std::shared_ptr<distributed::MeshDevice>& mesh_device : this->devices_) {
        const ProbeResult r = run_probe(mesh_device, /*use_barrier=*/true);
        EXPECT_EQ(r.iters, kIters);
        EXPECT_EQ(r.stale_publishes, 0u) << "noc_async_write_barrier() must commit the payload before the atomic "
                                            "credit, but "
                                         << r.stale_publishes << " of " << r.iters
                                         << " publishes exposed a stale tail (first at iteration " << r.first_bad_iter
                                         << ", " << r.stale_words << " stale words)";
    }
}

// Companion measurement for the flush-only form. It is reported, not asserted:
// the ISA gives no write->atomic ordering guarantee, so a zero here means the race
// was not won on this part, never that the ordering is safe.
TEST_F(MeshDeviceFixture, TensixWriteThenAtomicCreditFlushOnlyIsUnordered) {
    for (const std::shared_ptr<distributed::MeshDevice>& mesh_device : this->devices_) {
        const ProbeResult r = run_probe(mesh_device, /*use_barrier=*/false);
        EXPECT_EQ(r.iters, kIters);
        log_info(
            tt::LogTest,
            "flush-only write->atomic credit: {} / {} publishes had a stale tail ({} stale words, first at iter {})",
            r.stale_publishes,
            r.iters,
            r.stale_words,
            r.first_bad_iter);
    }
}

// Mechanism probe: the ISA says two requests arriving on DIFFERENT virtual channels
// can be reordered arbitrarily at the recipient NIU, so this is the configuration
// most likely to expose the write->atomic gap if it is exposable at all.
TEST_F(MeshDeviceFixture, TensixWriteThenAtomicCreditFlushOnlyCrossVc) {
    for (const std::shared_ptr<distributed::MeshDevice>& mesh_device : this->devices_) {
        const ProbeResult r = run_probe(mesh_device, /*use_barrier=*/false, /*credit_vc=*/kOtherVc);
        log_info(
            tt::LogTest,
            "flush-only cross-VC credit: {} / {} publishes had a stale tail ({} stale words, first at iter {})",
            r.stale_publishes,
            r.iters,
            r.stale_words,
            r.first_bad_iter);
    }
}

// Detector self-test. Sends the credit before the payload is even issued, so the
// receiver MUST observe a stale tail. Without this, a zero from the probes above
// would be indistinguishable from a probe that cannot detect anything.
TEST_F(MeshDeviceFixture, TensixWriteThenAtomicCreditDetectorFires) {
    for (const std::shared_ptr<distributed::MeshDevice>& mesh_device : this->devices_) {
        const ProbeResult r =
            run_probe(mesh_device, /*use_barrier=*/false, /*credit_vc=*/kUnicastWriteVc, /*credit_first=*/true);
        EXPECT_GT(r.stale_publishes, 0u) << "detector never fired even with the credit sent before the payload -- "
                                            "the probe is not measuring ordering";
        log_info(
            tt::LogTest, "detector self-test: {} / {} publishes stale (expected ~all)", r.stale_publishes, r.iters);
    }
}

}  // namespace tt::tt_metal
