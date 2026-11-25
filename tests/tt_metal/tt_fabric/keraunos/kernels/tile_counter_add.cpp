// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Owns or adds to an LLK tile counter of a Keraunos CCE (internal/tt-2xx/quasar/cce_tile_counter.h). Semaphore ids
// (CreateSemaphore, CoreType::ETH), shared by the core's harts:
//   0: the owner's stage. 1: an adder's stage; the high half is kStageAdding, the low half the adds issued so far.
//   2: go, host-written: an adder starts once it is 1. 3: finish, host-written: the owner reads its counter once it is 1.
//   4 onward: the owner's readings, in Reading order.
//
// Runtime args: 0: role. 3: the owner's capacity, which is the occupancy expected once every adder is done; an adder's
// number of adds, each of 1.
//   0, own: 1: counter id of this CCE.
//   1, add by system address: 1, 2: the counter's POSTED register, low and high word.
//   2, add to this CCE's counter: 1: counter id.

#include "api/dataflow/dataflow_api.h"
#include "api/semaphore.h"
#include "internal/tt-2xx/quasar/cce_tile_counter.h"

namespace {

namespace tc = internal_::cce_tile_counter;

constexpr ProgrammableCoreType kCoreType = static_cast<ProgrammableCoreType>(PROGRAMMABLE_CORE_TYPE);

constexpr uint32_t kOwnerStageSem = 0;
constexpr uint32_t kAdderStageSem = 1;
constexpr uint32_t kGoSem = 2;
constexpr uint32_t kFinishSem = 3;
constexpr uint32_t kFirstReadingSem = 4;

enum Reading : uint32_t {
    kOccupancy,           // after every adder is done
    kPolls,               // polls of the occupancy until it reached the expected value, or kPollLimit
    kTotal,               // READ_POSTED: adds to POSTED since RESET, mod 2^16
    kError,               // ERROR_STATUS
    kCapacity,            // BUFFER_CAPACITY
    kOccupancyPlusOne,    // after one more add, past the capacity
    kErrorPlusOne,
    kOccupancyAfterReset,  // after RESET alone
    kErrorAfterReset,
    kCapacityAfterReset,
    kNumReadings
};

constexpr uint32_t kStageArmed = 0x7C000001;  // owner: counter reset, adders may start
constexpr uint32_t kStageDone = 0x7C00000F;
constexpr uint32_t kStageBadArgs = 0x7C0000BA;
constexpr uint32_t kStageAdding = 0x7CA00000;  // | adds issued, low 16 bits
constexpr uint32_t kPollLimit = 1u << 20;

// The CCE's READ_POSTED and ERROR_STATUS offsets from the POSTED register: 0x34 and 0x3C in a counter's block, POSTED
// at 0x08. Quasar's header, and so llk_intf_get_posted() / llk_intf_get_error(), has
// 0x14 and 0x1C, which on the CCE are other registers.
constexpr uint32_t kReadPostedFromPosted = 0x2C;
constexpr uint32_t kErrorStatusFromPosted = 0x34;

uint32_t read_from_posted(uint32_t id, uint32_t offset) {
    return *reinterpret_cast<volatile uint32_t*>(tc::posted_addr(id) + offset);
}
uint32_t occupancy(uint32_t id) { return overlay::llk_intf_get_occupancy(overlay::LLK_INTERFACE_0, id); }
uint32_t capacity(uint32_t id) { return overlay::llk_intf_get_capacity(overlay::LLK_INTERFACE_0, id); }
uint32_t error(uint32_t id) { return read_from_posted(id, kErrorStatusFromPosted); }

void own(uint32_t id, uint32_t expected) {
    Semaphore<kCoreType> stage(kOwnerStageSem);
    tc::reset(id, expected);
    stage.set(kStageArmed);
    Semaphore<kCoreType> finish(kFinishSem);
    while (finish.value() != 1) {
    }
    uint32_t r[kNumReadings];
    uint32_t polls = 0;
    // The last adds may land after the host saw the adders done.
    while (occupancy(id) != expected && polls < kPollLimit) {
        polls++;
    }
    r[kOccupancy] = occupancy(id);
    r[kPolls] = polls;
    r[kTotal] = read_from_posted(id, kReadPostedFromPosted);
    r[kError] = error(id);
    r[kCapacity] = capacity(id);
    overlay::llk_intf_inc_posted(overlay::LLK_INTERFACE_0, id, 1);
    asm volatile("fence iorw, iorw" ::: "memory");
    r[kOccupancyPlusOne] = occupancy(id);
    r[kErrorPlusOne] = error(id);
    overlay::llk_intf_reset(overlay::LLK_INTERFACE_0, id);
    asm volatile("fence iorw, iorw" ::: "memory");
    r[kOccupancyAfterReset] = occupancy(id);
    r[kErrorAfterReset] = error(id);
    r[kCapacityAfterReset] = capacity(id);
    for (uint32_t i = 0; i < kNumReadings; i++) {
        Semaphore<kCoreType>(kFirstReadingSem + i).set(r[i]);
    }
    stage.set(kStageDone);
}

void add(uint64_t posted, uint32_t count) {
    Semaphore<kCoreType> stage(kAdderStageSem);
    Semaphore<kCoreType> go(kGoSem);
    while (go.value() != 1) {
    }
    for (uint32_t i = 0; i < count; i++) {
        tc::add(posted, 1);
        if ((i & 63) == 63) {
            stage.set(kStageAdding | (i + 1));
        }
    }
    stage.set(kStageDone);
}

}  // namespace

void kernel_main() {
    const uint32_t role = get_arg_val<uint32_t>(0);
    const uint32_t arg1 = get_arg_val<uint32_t>(1);
    const uint32_t count = get_arg_val<uint32_t>(3);
    if (role == 1) {
        add((static_cast<uint64_t>(get_arg_val<uint32_t>(2)) << 32) | arg1, count);
    } else if (role > 2 || arg1 >= tc::kNumCounters) {
        Semaphore<kCoreType>(role == 0 ? kOwnerStageSem : kAdderStageSem).set(kStageBadArgs);
    } else if (role == 0) {
        own(arg1, count);
    } else {
        add(tc::posted_addr(arg1), count);
    }
}
