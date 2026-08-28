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
//
// These are processes rather than threads for a reason beyond fidelity: ThreadSanitizer cannot
// see this region's races. Its shadow state is keyed by virtual address, and every provider
// mmap()s its own view, so two parties writing the same physical word through different
// mappings are invisible to it -- verified with a deliberate 2M-iteration race across two
// mappings of one page, which TSan reported only when both threads shared a single mapping.
// A clean TSan run therefore says nothing about the invariants below; these tests are the
// coverage.

#include <fcntl.h>
#include <poll.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

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

// The attached-process counts are header fields with no public accessor, and they are what the
// concurrent-reclamation race corrupts, so read them straight out of the mapping.
struct HeaderCounts {
    uint32_t reference_count = 0;
    uint32_t num_active_processes = 0;
};

// Bounded read: a child that wedges must fail this test, not hang the suite. fork() in this
// binary happens after other fixtures have opened devices and started threads, so a child can
// inherit a mutex held by a thread that does not exist in it -- see the note on the disabled
// test below.
bool read_with_timeout(int fd, char& out, int timeout_ms) {
    struct pollfd pfd {};
    pfd.fd = fd;
    pfd.events = POLLIN;
    if (poll(&pfd, 1, timeout_ms) != 1) {
        return false;
    }
    return read(fd, &out, 1) == 1;
}

// Every raw pid in the table, negatives included, for diagnosing a mismatch.
std::string dump_raw_pids(uint64_t asic_id) {
    std::string out;
    const int fd = shm_open(shm_object_name(asic_id).c_str(), O_RDONLY, 0600);
    if (fd < 0) {
        return out;
    }
    auto* region =
        static_cast<DeviceMemoryRegion*>(mmap(nullptr, sizeof(DeviceMemoryRegion), PROT_READ, MAP_SHARED, fd, 0));
    close(fd);
    if (region == MAP_FAILED) {
        return out;
    }
    for (size_t i = 0; i < MAX_PROCESSES; i++) {
        const pid_t pid = region->processes[i].pid.load(std::memory_order_acquire);
        if (pid != 0) {
            out += " [" + std::to_string(i) + "]=" + std::to_string(pid);
        }
    }
    munmap(region, sizeof(DeviceMemoryRegion));
    return out;
}

// Build the region a pre-v4 writer leaves behind, as a v4 process finds it.
//
// The important property is that init_state reads UNINITIALIZED: it is appended last in v4, so
// an older layout never wrote it, and whether it lands in that layout's tail padding or in
// bytes ftruncate appends here, it was zero-filled at creation and stayed that way. Everything
// else -- version, reference_count, the process table -- is written as that older layout would
// have left it. Creating the object at the current size and only setting the earlier fields
// reproduces exactly that, without needing the old struct definition.
void write_legacy_region(uint64_t asic_id, uint32_t version, pid_t owner_pid, uint64_t dram_bytes) {
    const int fd = shm_open(shm_object_name(asic_id).c_str(), O_CREAT | O_RDWR, 0600);
    ASSERT_NE(fd, -1) << "could not create synthetic region";
    ASSERT_EQ(ftruncate(fd, sizeof(DeviceMemoryRegion)), 0);
    auto* region = static_cast<DeviceMemoryRegion*>(
        mmap(nullptr, sizeof(DeviceMemoryRegion), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
    close(fd);
    ASSERT_NE(region, MAP_FAILED);

    region->version.store(version, std::memory_order_relaxed);
    region->asic_id.store(asic_id, std::memory_order_relaxed);
    region->device_id.store(0, std::memory_order_relaxed);
    region->reference_count.store(1, std::memory_order_relaxed);
    region->num_active_processes.store(1, std::memory_order_relaxed);
    region->total_dram_allocated.store(dram_bytes, std::memory_order_relaxed);
    region->processes[0].pid.store(owner_pid, std::memory_order_relaxed);
    region->processes[0].dram_allocated.store(dram_bytes, std::memory_order_relaxed);
    // Left exactly as an older layout left it: never written, so still zero.
    ASSERT_EQ(region->init_state.load(std::memory_order_relaxed), SHM_INIT_UNINITIALIZED);

    munmap(region, sizeof(DeviceMemoryRegion));
}

struct LegacyRegionState {
    uint32_t version = 0;
    uint32_t init_state = 0;
    pid_t slot0_pid = 0;
    uint64_t total_dram = 0;
};

LegacyRegionState read_legacy_region(uint64_t asic_id) {
    LegacyRegionState state;
    const int fd = shm_open(shm_object_name(asic_id).c_str(), O_RDONLY, 0600);
    if (fd < 0) {
        return state;
    }
    auto* region =
        static_cast<DeviceMemoryRegion*>(mmap(nullptr, sizeof(DeviceMemoryRegion), PROT_READ, MAP_SHARED, fd, 0));
    close(fd);
    if (region == MAP_FAILED) {
        return state;
    }
    state.version = region->version.load(std::memory_order_acquire);
    state.init_state = region->init_state.load(std::memory_order_acquire);
    state.slot0_pid = region->processes[0].pid.load(std::memory_order_acquire);
    state.total_dram = region->total_dram_allocated.load(std::memory_order_acquire);
    munmap(region, sizeof(DeviceMemoryRegion));
    return state;
}

// A pid that is definitely not running: fork a child that exits immediately and reap it.
pid_t make_dead_pid() {
    const pid_t pid = fork();
    if (pid == 0) {
        _exit(0);
    }
    EXPECT_NE(pid, -1);
    int status = 0;
    EXPECT_EQ(waitpid(pid, &status, 0), pid);
    return pid;
}

HeaderCounts read_header_counts(uint64_t asic_id) {
    HeaderCounts counts;
    const int fd = shm_open(shm_object_name(asic_id).c_str(), O_RDONLY, 0600);
    if (fd < 0) {
        return counts;
    }
    auto* region =
        static_cast<DeviceMemoryRegion*>(mmap(nullptr, sizeof(DeviceMemoryRegion), PROT_READ, MAP_SHARED, fd, 0));
    close(fd);
    if (region == MAP_FAILED) {
        return counts;
    }
    counts.reference_count = region->reference_count.load(std::memory_order_acquire);
    counts.num_active_processes = region->num_active_processes.load(std::memory_order_acquire);
    munmap(region, sizeof(DeviceMemoryRegion));
    return counts;
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
    pid_t spawn_child(uint64_t dram_bytes, bool die_by_signal, int& ready_fd, int& release_fd) const {
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

// A process attaching to the region must not disturb anybody else's accounting.
//
// The device-wide totals used to be re-derived on attach -- sum the live slots, then store
// the result -- while allocations were applied to the same words with fetch_add. The two
// protocols do not compose: an allocation recorded between another process's sum and its
// store is silently dropped from the total and stays dropped, leaving the device total
// short while the owning process's own slot still reads correctly. Every aggregate
// mutation is now a delta, which commutes with concurrent writers.
//
// This exercises the race rather than proving its absence, so it runs several rounds and
// keeps the child attaching for the whole of the parent's allocation loop. Against the
// derive-and-store version it failed ~75% of the time per round even with only partial
// overlap; three fully-overlapped rounds make a surviving regression very unlikely to pass.
TEST_F(ShmMemoryTrackingMultiProcess, ConcurrentAttachDoesNotLoseAllocations) {
    constexpr uint64_t kChunk = 4096;
    constexpr int kChurnChildren = 3;
    constexpr int kAllocsPerRound = 40000;
    constexpr int kRounds = 3;
    constexpr long kMaxChildIterations = 5'000'000;  // safety valve if the parent dies

    for (int round = 0; round < kRounds; round++) {
        auto owner = attach(asic_id_);
        ASSERT_TRUE(owner->is_initialized());
        const uint64_t before = owner->get_device_stats().dram_allocated;

        int stop[2];
        int warm[2];
        ASSERT_EQ(pipe(stop), 0);
        ASSERT_EQ(pipe(warm), 0);

        std::vector<pid_t> children;
        for (int c = 0; c < kChurnChildren; c++) {
            const pid_t child = fork();
            ASSERT_NE(child, -1);
            if (child == 0) {
                // Attach and detach continuously until told to stop. Each attach and each
                // detach writes the aggregates; those writes must not disturb the parent.
                close(stop[1]);
                close(warm[0]);
                const int flags = fcntl(stop[0], F_GETFL, 0);
                fcntl(stop[0], F_SETFL, flags | O_NONBLOCK);
                {
                    SharedMemoryStatsProvider warmup(asic_id_, 0, false, false);
                }
                const char ready = 'r';
                ssize_t ignored = write(warm[1], &ready, 1);
                (void)ignored;
                char token = 0;
                for (long i = 0; i < kMaxChildIterations; i++) {
                    // Poll rarely: a read() per cycle throttles the churn by an order of
                    // magnitude, and the churn rate is what makes the window observable.
                    if ((i % 64) == 0 && read(stop[0], &token, 1) == 1) {
                        break;
                    }
                    SharedMemoryStatsProvider visitor(asic_id_, 0, false, false);
                }
                _exit(0);
            }
            children.push_back(child);
        }
        close(stop[0]);
        close(warm[1]);
        for (int c = 0; c < kChurnChildren; c++) {
            char ready = 0;
            ASSERT_EQ(read(warm[0], &ready, 1), 1) << "a churn child never finished its first attach";
        }
        close(warm[0]);

        uint64_t recorded = 0;
        for (int i = 0; i < kAllocsPerRound; i++) {
            owner->record_allocation(getpid(), kChunk, ShmBufferType::DRAM, /*chip_id=*/0);
            recorded += kChunk;
        }

        // Read the total while the churn is still running. Measuring after the children have
        // exited would hide the bug: the last detach recomputes the aggregates from the slots
        // and heals the lost update just before the assertion looks at it.
        const uint64_t expected = before + recorded;
        const uint64_t total = owner->get_device_stats().dram_allocated;
        const uint64_t own_slot = dram_for_pid(*owner, getpid());

        const char go = 'x';
        for (int c = 0; c < kChurnChildren; c++) {
            EXPECT_EQ(write(stop[1], &go, 1), 1);
        }
        close(stop[1]);
        for (const pid_t child : children) {
            int status = 0;
            ASSERT_EQ(waitpid(child, &status, 0), child);
            EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0) << "churn child did not exit cleanly";
        }
        EXPECT_EQ(total, expected) << "round " << round << ": device total lost "
                                   << (static_cast<long long>(expected) - static_cast<long long>(total)) << " of "
                                   << recorded
                                   << " bytes while other processes were attaching; "
                                      "an aggregate write clobbered a concurrent allocation";
        EXPECT_EQ(own_slot, recorded) << "round " << round << ": own slot is wrong";

        // Leaving scope releases our slot, which must subtract exactly what we added.
        owner.reset();
        auto after = attach(asic_id_);
        ASSERT_TRUE(after->is_initialized());
        EXPECT_EQ(after->get_device_stats().dram_allocated, 0u)
            << "round " << round << ": releasing the slot did not subtract its bytes";
    }
}

// Two attachers can decide to reap the same dead slot at the same moment. The byte totals
// survive that -- only one exchange returns the value -- but the attached-process counts are
// plain decrements, so both reapers used to subtract and reference_count fell below the number
// of processes actually attached. That is not cosmetic: the version-mismatch path treats a zero
// count as "nobody is using this" and reinitializes the region underneath them.
//
// The race needs real concurrency, so the children are held at a barrier and released together;
// the assertion is the invariant rather than the bug, so this can never fail spuriously.
// DISABLED: not CI-safe, for a reason unrelated to what it checks. It forks in a binary where
// earlier fixtures have opened devices and started threads, and a forked child inherits their
// mutexes without the threads that hold them -- the provider's own logging is enough to wedge
// one. Run alone it passes (6/6, and 10/10 over eight rounds each); run after the device tests
// it can hang, so it stays off until it lives in a binary that has not touched a device. The
// reads are bounded so that enabling it fails rather than hangs.
//
// It earned its place regardless: it is what found the bug below, and the ablation is in the
// commit -- release-from-load 0/3 pass, unguarded 1/3, release-from-decided-pid 3/3.
//
// Two attachers can decide to reap the same dead slot at once. Guarding that needs two things,
// and this test failed until both were in place: only the winner of a CAS may clear the slot and
// move the counts, and the CAS must be from the pid the caller decided about rather than from
// whatever the slot holds when the release runs. Without the second, the loser of the first race
// cleared whichever live process had since claimed the freed slot, stranding it -- attached and
// counted, owning nothing -- which showed up as reference_count reading above the number of
// occupied slots.
TEST_F(ShmMemoryTrackingMultiProcess, DISABLED_ConcurrentAttachersDoNotMiscountAfterReapingADeadSlot) {
    constexpr int kAttachers = 6;
    constexpr int kRounds = 8;

    for (int round = 0; round < kRounds; round++) {
        // Leave a dead slot for them to fight over.
        int dead_ready = -1, dead_release = -1;
        const pid_t dead = spawn_child(4 * 1024 * 1024, /*die_by_signal=*/true, dead_ready, dead_release);
        wait_ready(dead_ready);
        int status = 0;
        ASSERT_EQ(waitpid(dead, &status, 0), dead);
        close(dead_ready);
        close(dead_release);

        // Advance the pid allocator well past the dead pid before forking the attachers.
        // Linux hands out pids sequentially, so an attacher would otherwise be liable to be
        // handed the dead process's pid, adopt its slot (claim_own_pid_entry matches on pid
        // alone) and make this test about pid reuse instead of about the reclamation race.
        for (int burn = 0; burn < 256; burn++) {
            const pid_t throwaway = fork();
            ASSERT_NE(throwaway, -1);
            if (throwaway == 0) {
                _exit(0);
            }
            ASSERT_EQ(waitpid(throwaway, &status, 0), throwaway);
        }

        // Fork attachers that block until released, then all attach at once and hold.
        // Two pipes on purpose: `gate` carries exactly one barrier token per child, and `hold`
        // carries none -- children block on it until the parent closes its end. Sharing one
        // pipe lets a fast child consume a second token and starve a sibling.
        int gate[2], hold[2], done[2];
        ASSERT_EQ(pipe(gate), 0);
        ASSERT_EQ(pipe(hold), 0);
        ASSERT_EQ(pipe(done), 0);
        std::vector<pid_t> kids;
        for (int i = 0; i < kAttachers; i++) {
            const pid_t pid = fork();
            ASSERT_NE(pid, -1);
            if (pid == 0) {
                close(gate[1]);
                close(hold[1]);
                close(done[0]);
                char go = 0;
                ssize_t got = read(gate[0], &go, 1);  // barrier: exactly one token each
                (void)got;
                {
                    auto provider = attach(asic_id_);
                    // 'y' only if this child actually owns a slot after attaching.
                    const char token =
                        (provider->is_initialized() && has_slot_for_pid(*provider, getpid())) ? 'y' : 'n';
                    ssize_t w = write(done[1], &token, 1);
                    (void)w;
                    // Hold the slot while the parent counts. Returns 0 at EOF.
                    char sink = 0;
                    ssize_t h = read(hold[0], &sink, 1);
                    (void)h;
                }  // destructor releases the slot -- _exit() would not run it
                _exit(0);
            }
            kids.push_back(pid);
        }
        close(gate[0]);
        close(hold[0]);
        close(done[1]);

        // Release them together, then wait for all to report attached.
        for (int i = 0; i < kAttachers; i++) {
            ASSERT_EQ(write(gate[1], "g", 1), 1);
        }
        for (int i = 0; i < kAttachers; i++) {
            char token = 0;
            ASSERT_TRUE(read_with_timeout(done[0], token, 10000))
                << "round " << round << ": an attacher never reported in (see the fork-safety note)";
            EXPECT_EQ(token, 'y') << "round " << round << ": an attacher attached without owning a slot";
        }

        // Every child holds a slot; this process adds one of its own.
        auto observer = attach(asic_id_);
        ASSERT_TRUE(observer->is_initialized());
        const auto slots = observer->get_process_stats();
        EXPECT_EQ(slots.size(), static_cast<size_t>(kAttachers) + 1)
            << "round " << round << ": expected one slot per attacher plus this observer; a dead slot was "
            << "either not reclaimed or reclaimed twice";
        for (const auto& slot : slots) {
            EXPECT_NE(slot.pid, dead) << "round " << round << ": the dead process still owns a slot";
            EXPECT_GT(slot.pid, 0) << "round " << round << ": a slot is stuck mid-release";
        }

        // The counts are what the race corrupts: two reapers of one dead slot each subtracting
        // leaves reference_count below the number of attached processes, and the
        // version-mismatch path reads zero as "nobody is using this".
        const HeaderCounts counts = read_header_counts(asic_id_);
        EXPECT_EQ(counts.reference_count, slots.size())
            << "round " << round
            << ": reference_count disagrees with the number of occupied slots; raw pids:" << dump_raw_pids(asic_id_)
            << " (this pid " << getpid() << ", dead was " << dead << ")";
        EXPECT_EQ(counts.num_active_processes, slots.size())
            << "round " << round << ": num_active_processes disagrees with the number of occupied slots";

        close(hold[1]);  // EOF: every child falls through and exits
        for (const pid_t pid : kids) {
            ASSERT_EQ(waitpid(pid, &status, 0), pid);
        }
        close(gate[1]);
        close(done[0]);
    }
}

// A region written by a layout older than v4 publishes no init_state -- the field is appended
// last, so an older writer never set it and it reads UNINITIALIZED. That is the same value a
// brand-new zero-filled region has, so init_state alone cannot tell them apart, and attaching
// on the strength of it would claim the right to initialize and run initialize_region() over a
// region an older process is still using: totals zeroed, process table wiped, that process
// left attached to a region that no longer knows about it.
//
// `version` is what separates the two, so a nonzero version has to be put through the same
// live-owner test as a wrong-version region that *did* publish READY.
TEST_F(ShmMemoryTrackingMultiProcess, LegacyRegionWithLiveOwnerIsNotReinitialized) {
    constexpr uint64_t kOwnerBytes = 32 * 1024 * 1024;

    // A child that does nothing but stay alive, so its pid in the table is a live owner.
    int gate[2];
    ASSERT_EQ(pipe(gate), 0);
    const pid_t owner = fork();
    ASSERT_NE(owner, -1);
    if (owner == 0) {
        close(gate[1]);
        char go = 0;
        ssize_t got = read(gate[0], &go, 1);  // blocks until the parent closes its end
        (void)got;
        _exit(0);
    }
    close(gate[0]);

    shm_unlink(shm_object_name(asic_id_).c_str());  // drop the fixture's v4 probe region
    write_legacy_region(asic_id_, /*version=*/3, owner, kOwnerBytes);

    auto provider = attach(asic_id_);
    EXPECT_FALSE(provider->is_initialized())
        << "attached to a v3 region whose owner is still running; it should have disabled itself";

    const LegacyRegionState state = read_legacy_region(asic_id_);
    EXPECT_EQ(state.version, 3u) << "the older layout's region was reinitialized underneath its owner";
    EXPECT_EQ(state.init_state, SHM_INIT_UNINITIALIZED) << "init_state was published into a foreign layout";
    EXPECT_EQ(state.slot0_pid, owner) << "the live owner's slot was cleared";
    EXPECT_EQ(state.total_dram, kOwnerBytes) << "the live owner's bytes were erased";

    close(gate[1]);
    int status = 0;
    ASSERT_EQ(waitpid(owner, &status, 0), owner);
}

// The other half: the guard must not be one-way. A pre-v4 region whose owner is gone -- the
// ordinary case after an upgrade, and the one a killed older process leaves -- is still taken
// over, because the process table says nobody is there. reference_count cannot be what decides
// it: an older writer leaks it permanently when killed, which would wedge the region forever.
TEST_F(ShmMemoryTrackingMultiProcess, LegacyRegionWithNoLiveOwnerIsReclaimed) {
    const pid_t dead = make_dead_pid();

    shm_unlink(shm_object_name(asic_id_).c_str());
    // reference_count is left at 1, as a killed owner would have left it.
    write_legacy_region(asic_id_, /*version=*/3, dead, /*dram_bytes=*/64 * 1024 * 1024);

    auto provider = attach(asic_id_);
    ASSERT_TRUE(provider->is_initialized()) << "a v3 region with no live owner must be reclaimed, not refused";

    const LegacyRegionState state = read_legacy_region(asic_id_);
    EXPECT_EQ(state.version, DEVICE_MEMORY_REGION_VERSION) << "region was not upgraded to the current layout";
    EXPECT_EQ(state.init_state, SHM_INIT_READY) << "reinitialized region never published readiness";
    EXPECT_EQ(state.slot0_pid, getpid()) << "expected this process to take the freed first slot";
    EXPECT_EQ(state.total_dram, 0u) << "the dead owner's bytes survived the reinitialization";
}

}  // namespace
}  // namespace tt::tt_metal
