// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Cross-process behaviour of the SHM memory-tracking region.
//
// The region is shared state whose whole purpose is to be read by another process, so the
// cases that matter most are the ones a single-process test cannot reach: what a second
// process sees while the first is running, and what is left behind when a process dies
// without unwinding. Those were previously untested.
//
// No device is needed: SharedMemoryStatsProvider is constructed from an asic_id, so these
// tests attach to a synthetic region of their own (keyed by the test process's pid, so
// concurrent CI jobs cannot collide) and never touch a real chip's region.

#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>

#include "gtest/gtest.h"
#include "impl/memory_tracking/memory_stats_shm.hpp"

namespace tt::tt_metal {
namespace {

// Synthetic asic_id, well outside the UMD chip_unique_id space, unique per test process.
uint64_t test_asic_id() { return 0x7700000000000000ULL + static_cast<uint64_t>(getpid()); }

std::string shm_object_name(uint64_t asic_id) { return "/tt_device_" + std::to_string(asic_id) + "_memory"; }

std::unique_ptr<SharedMemoryStatsProvider> attach(uint64_t asic_id) {
    return std::make_unique<SharedMemoryStatsProvider>(
        asic_id, /*device_id=*/0, /*tracking_disabled=*/false, /*verbose=*/false);
}

uint64_t dram_for_pid(const SharedMemoryStatsProvider& provider, pid_t pid) {
    for (const auto& p : provider.get_process_stats()) {
        if (p.pid == pid) {
            return p.dram_allocated;
        }
    }
    return 0;
}

bool has_slot_for_pid(const SharedMemoryStatsProvider& provider, pid_t pid) {
    for (const auto& p : provider.get_process_stats()) {
        if (p.pid == pid) {
            return true;
        }
    }
    return false;
}

// Fixture owns the lifetime of the synthetic region so a failing test cannot leak it into
// /dev/shm and poison the next run.
class ShmMemoryTrackingMultiProcess : public ::testing::Test {
protected:
    void SetUp() override {
        asic_id_ = test_asic_id();
        shm_unlink(shm_object_name(asic_id_).c_str());  // in case a previous run died hard
        auto probe = attach(asic_id_);
        if (!probe->is_initialized()) {
            GTEST_SKIP() << "cannot map shared memory in this environment";
        }
    }

    void TearDown() override { shm_unlink(shm_object_name(asic_id_).c_str()); }

    // Fork a child that attaches, records `dram_bytes`, tells us it is ready, and then
    // either waits for permission to exit cleanly or kills itself outright.
    // Returns the child pid; `ready_fd` becomes readable once the child has recorded.
    pid_t spawn_child(uint64_t dram_bytes, bool die_by_signal, int& ready_fd, int& release_fd) {
        int ready[2], release[2];
        EXPECT_EQ(pipe(ready), 0);
        EXPECT_EQ(pipe(release), 0);
        const pid_t pid = fork();
        EXPECT_NE(pid, -1);
        if (pid == 0) {
            // Child. Never run gtest reporting or static destructors from here.
            close(ready[0]);
            close(release[1]);
            {
                auto provider = attach(asic_id_);
                if (!provider->is_initialized()) {
                    _exit(3);
                }
                provider->record_allocation(getpid(), dram_bytes, ShmBufferType::DRAM, /*chip_id=*/0);
                char token = 'r';
                ssize_t written = write(ready[1], &token, 1);
                (void)written;
                if (die_by_signal) {
                    // No destructor, no slot release: what a SIGKILLed or crashed run leaves.
                    kill(getpid(), SIGKILL);
                }
                char go = 0;
                ssize_t got = read(release[0], &go, 1);  // hold the slot until released
                (void)got;
            }  // provider destructor runs only on the clean path
            _exit(0);
        }
        close(ready[1]);
        close(release[0]);
        ready_fd = ready[0];
        release_fd = release[1];
        return pid;
    }

    static void wait_ready(int ready_fd) {
        char token = 0;
        ASSERT_EQ(read(ready_fd, &token, 1), 1) << "child never reported ready";
    }

    uint64_t asic_id_ = 0;
};

// A process that is SIGKILLed never runs its destructor, so its allocations stay in the
// region. Nothing else was ever going to remove them: the old design could only subtract a
// process's contribution from that process's own destructor, so a killed run left its bytes
// in the device totals permanently and kept reference_count above zero, which stopped the
// region from ever resetting. Reclamation has to be done by whoever attaches next.
TEST_F(ShmMemoryTrackingMultiProcess, DeadProcessSlotIsReclaimedByNextAttacher) {
    constexpr uint64_t kChildBytes = 64 * 1024 * 1024;

    auto observer = attach(asic_id_);
    ASSERT_TRUE(observer->is_initialized());
    const uint64_t before = observer->get_device_stats().dram_allocated;

    int ready_fd = -1, release_fd = -1;
    const pid_t child = spawn_child(kChildBytes, /*die_by_signal=*/true, ready_fd, release_fd);
    wait_ready(ready_fd);

    // While the child is alive its allocation is part of the device-wide total.
    EXPECT_EQ(observer->get_device_stats().dram_allocated, before + kChildBytes)
        << "a live second process's allocation is missing from the device-wide total";
    EXPECT_EQ(dram_for_pid(*observer, child), kChildBytes);

    int status = 0;
    ASSERT_EQ(waitpid(child, &status, 0), child);
    ASSERT_TRUE(WIFSIGNALED(status)) << "child was supposed to die by signal, not exit cleanly";
    close(ready_fd);
    close(release_fd);

    // The next attach is what performs the cleanup.
    auto next_attacher = attach(asic_id_);
    ASSERT_TRUE(next_attacher->is_initialized());

    EXPECT_FALSE(has_slot_for_pid(*next_attacher, child))
        << "the dead process still owns a slot; a SIGKILLed run permanently occupies one of " << MAX_PROCESSES
        << " slots";
    EXPECT_EQ(next_attacher->get_device_stats().dram_allocated, before)
        << "the dead process's bytes are still in the device-wide total; every killed run "
           "inflates what an external monitor reports, with no way to recover";
}

// The device-wide totals must describe the device, not whichever process wrote last.
TEST_F(ShmMemoryTrackingMultiProcess, DeviceTotalsSumOverLiveProcesses) {
    constexpr uint64_t kOwnBytes = 16 * 1024 * 1024;
    constexpr uint64_t kChildBytes = 48 * 1024 * 1024;

    auto own = attach(asic_id_);
    ASSERT_TRUE(own->is_initialized());
    const uint64_t before = own->get_device_stats().dram_allocated;
    own->record_allocation(getpid(), kOwnBytes, ShmBufferType::DRAM, /*chip_id=*/0);

    int ready_fd = -1, release_fd = -1;
    const pid_t child = spawn_child(kChildBytes, /*die_by_signal=*/false, ready_fd, release_fd);
    wait_ready(ready_fd);

    EXPECT_EQ(own->get_device_stats().dram_allocated, before + kOwnBytes + kChildBytes)
        << "device total is not the sum over live processes";
    EXPECT_EQ(dram_for_pid(*own, getpid()), kOwnBytes);
    EXPECT_EQ(dram_for_pid(*own, child), kChildBytes);

    // Release the child and let it unwind cleanly; its contribution must disappear.
    char go = 'g';
    ASSERT_EQ(write(release_fd, &go, 1), 1);
    int status = 0;
    ASSERT_EQ(waitpid(child, &status, 0), child);
    EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0) << "child did not exit cleanly";
    close(ready_fd);
    close(release_fd);

    // Our own attach re-derives the aggregates, which is what drops the departed process.
    auto after = attach(asic_id_);
    ASSERT_TRUE(after->is_initialized());
    EXPECT_FALSE(has_slot_for_pid(*after, child)) << "a cleanly exited process left its slot behind";
    EXPECT_EQ(after->get_device_stats().dram_allocated, before + kOwnBytes)
        << "a cleanly exited process's bytes are still counted";

    own->record_deallocation(getpid(), kOwnBytes, ShmBufferType::DRAM, /*chip_id=*/0);
    EXPECT_EQ(own->get_device_stats().dram_allocated, before) << "our own deallocation was not reflected";
}

// Repeated attach/detach must leave the region exactly as it found it. The old destructor
// tested reference_count and then subtracted from it, which is not atomic as a pair; the
// count is now derived from the live slots, so releasing the slot is the whole operation.
//
// Note the providers here do not overlap. A ProcessStats slot belongs to a pid, not to a
// provider instance, which is the right model -- a process gets one slot per region -- so
// two coexisting providers for one region in one process would share (and fight over) a
// single slot. Nothing does that: a process holds one provider per chip.
TEST_F(ShmMemoryTrackingMultiProcess, RepeatedAttachDetachLeavesNoResidue) {
    constexpr uint64_t kBytes = 1024 * 1024;

    for (int i = 0; i < 16; i++) {
        auto transient = attach(asic_id_);
        ASSERT_TRUE(transient->is_initialized()) << "attach failed on cycle " << i;
        transient->record_allocation(getpid(), kBytes, ShmBufferType::DRAM, /*chip_id=*/0);
        EXPECT_EQ(transient->get_device_stats().dram_allocated, kBytes) << "on cycle " << i;
        transient->record_deallocation(getpid(), kBytes, ShmBufferType::DRAM, /*chip_id=*/0);
    }

    // A fresh attach sees a region with exactly one slot -- its own, empty. More than one
    // means a cycle leaked a slot, and there are only MAX_PROCESSES of them; none means
    // attaching is not discoverable by other processes until this one happens to allocate.
    auto fresh = attach(asic_id_);
    ASSERT_TRUE(fresh->is_initialized());
    const auto slots = fresh->get_process_stats();
    EXPECT_EQ(slots.size(), 1u) << "a fresh attach should own exactly one slot (its own); got " << slots.size();
    EXPECT_EQ(fresh->get_device_stats().dram_allocated, 0u) << "attach/detach cycles drifted the device-wide total";
    EXPECT_TRUE(has_slot_for_pid(*fresh, getpid()));
}

}  // namespace
}  // namespace tt::tt_metal
