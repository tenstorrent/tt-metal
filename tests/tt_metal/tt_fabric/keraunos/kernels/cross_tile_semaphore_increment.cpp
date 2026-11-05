// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Increments a semaphore in another Keraunos core's SRAM, and reports progress in this core's own
// semaphores. Semaphore ids (CreateSemaphore on both cores, CoreType::ETH):
//   0: the counter. This core increments the peer's; the peer increments this core's.
//   1: stage word, written as this kernel advances. A hang leaves the last stage reached.
//   2: probe, host-initialised. 3: receives the value this core read from 2.
// Own semaphores go through Semaphore. On a CCE get_semaphore() gives the uncached alias: host writes are
// not snooped into the cache, and the host must see this core's stores.
//
// Runtime args:
//   0: mode. 0 = CPU load-add-store at the peer's global HSIO SRAM address (lo/hi in 2, 3).
//            1 = noc_semaphore_inc to get_noc_addr(peer x, peer y, peer counter offset) (4, 5, 6).
//   1: number of increments.
//   2, 3: peer counter, global HSIO SRAM address, low and high word.
//   4, 5: peer virtual NOC x, y.  6: peer counter, local offset.
//   7: this core's stage-word offset as the host computed it; checked against get_semaphore(1).

#include "api/dataflow/dataflow_api.h"
#include "api/semaphore.h"

namespace {

constexpr ProgrammableCoreType kCoreType = static_cast<ProgrammableCoreType>(PROGRAMMABLE_CORE_TYPE);

constexpr uint32_t kStageEntered = 0x5E000001;
constexpr uint32_t kStageProbeCopied = 0x5E000002;
constexpr uint32_t kStageIncrementBase = 0x5E100000;  // | increments issued
constexpr uint32_t kStageIncrementsDone = 0x5E000005;  // mode 1: after the atomic barrier
constexpr uint32_t kStageDone = 0x5E00000F;
constexpr uint32_t kStageBadSemaphoreAddress = 0x5E0000BA;
constexpr uint32_t kStageBadMode = 0x5E0000BB;

}  // namespace

void kernel_main() {
    const uint32_t mode = get_arg_val<uint32_t>(0);
    const uint32_t count = get_arg_val<uint32_t>(1);
    const uint64_t peer_global =
        (static_cast<uint64_t>(get_arg_val<uint32_t>(3)) << 32) | get_arg_val<uint32_t>(2);
    const uint32_t peer_noc_x = get_arg_val<uint32_t>(4);
    const uint32_t peer_noc_y = get_arg_val<uint32_t>(5);
    const uint32_t peer_counter_offset = get_arg_val<uint32_t>(6);
    const uint32_t expected_stage_offset = get_arg_val<uint32_t>(7);

    if (get_semaphore<kCoreType>(1) != expected_stage_offset + MEM_L1_UNCACHED_BASE) {
        // The host reads the stage word at the address it computed, so write that one.
        *reinterpret_cast<volatile uint32_t*>(expected_stage_offset + MEM_L1_UNCACHED_BASE) =
            kStageBadSemaphoreAddress;
        return;
    }
    Semaphore<kCoreType> stage(1);
    stage.set(kStageEntered);

    Semaphore<kCoreType>(3).set(Semaphore<kCoreType>(2).value());
    stage.set(kStageProbeCopied);

    if (mode > 1) {
        stage.set(kStageBadMode);
        return;
    }
    // Mode 0: the peer window is outside this core's cacheable SRAM; the store travels over SMN.
    volatile uint32_t* peer = reinterpret_cast<volatile uint32_t*>(static_cast<uintptr_t>(peer_global));
    const uint64_t peer_noc = get_noc_addr(peer_noc_x, peer_noc_y, peer_counter_offset);
    for (uint32_t i = 0; i < count; i++) {
        if (mode == 0) {
            *peer = *peer + 1;
            asm volatile("fence" ::: "memory");
        } else {
            noc_semaphore_inc(peer_noc, 1);
        }
        stage.set(kStageIncrementBase | (i + 1));
    }
    if (mode == 1) {
        noc_async_atomic_barrier();
        stage.set(kStageIncrementsDone);
    }
    stage.set(kStageDone);
}
