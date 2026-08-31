// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// The bank sweep, divided and conquered.
//
// Where an on-chip coprocessor is the mover, one core walked every bank looking for armed status words, and
// recovery.md's model made the cost of that explicit:
//
//     interval = S + F/D      S ~= cycles of per-transfer work
//                             F ~= cycles of BANK-SWEEP PERIOD
//                             D  = banks holding a ready request when the sweep passes
//
// F is not a latency that depth hides -- it is one serial walk over every bank, and it is
// there because a single core is the only thing doing the walking. Move the register
// file to host memory and that term stops being structural: 32 host cores can walk 110
// banks concurrently, and the time to notice an armed word stops growing with the number
// of banks at all.
//
// Scanning is cheap, cache-local, and embarrassingly divisible: each worker owns a
// contiguous shard of bank indices, which keeps workers off each other's cache lines and
// off each other's pages. Sharding it statically is right -- stealing scan duty would
// mean two workers touching the same 64 B line, which is precisely the false sharing the
// bank stride was chosen to prevent.
//
// SERVICING is the expensive half -- decode, route, a transport post, a completion --
// and it is bursty: a message arriving on core 3 costs the worker that owns core 3 the
// full transport latency, during which its remaining banks go unwatched while other
// workers spin on idle shards. THAT is the imbalance worth fixing, so servicing is what
// gets queued and stolen:
//
//   1. scan my shard; every armed bank becomes a Job pushed to my own deque
//   2. drain my own deque, LIFO -- the job I just pushed is the one still in cache
//   3. when my deque is empty, steal from another worker's TAIL, FIFO
//
// LIFO for self, FIFO for thieves is the standard Chase-Lev discipline and it is not
// arbitrary: the owner gets locality, the thief gets the OLDEST job, which is the one
// most likely to still have work behind it and least likely to be about to be claimed.
//
// ON THE LOCK. Each deque is a std::deque behind its own mutex rather than a lock-free
// Chase-Lev deque. Deliberate, and worth revisiting only with a profile: a job costs
// microseconds of transport time, jobs are rare relative to scans, and the lock is
// per-worker so uncontended in the common case. A lock-free deque here would be a
// correctness risk taken to optimise something that is not yet known to be hot. The
// counters this pool reports (stolen, donated, idle_spins) are what would show it.
#pragma once

#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "host_region.hpp"
#include "host_stats.hpp"
#include "host_uva_layout.hpp"

namespace tt::tt_metal::experimental {

// One armed control word, snapshotted. The SNAPSHOT is the important part: the worker
// reads the control word once, and everything downstream works from that copy rather
// than re-reading the register. Re-reading would race the next arming of the same bank,
// and the failure mode is servicing a message with one iteration's opcode and the next
// iteration's operands.
// Which control register armed this job. The two directions are serviced by completely
// different code -- a TX job routes and sends, an RX job delivers into L1 -- so the
// direction travels with the job rather than being re-derived downstream.
enum class Dir : uint32_t { Tx = 0, Rx = 1 };

struct Job {
    Dir dir = Dir::Tx;
    uint32_t core = 0;
    // WHICH RECEIVE SLOT this notice came from. Always 0 on the TX side and at payloads big
    // enough that only one fits. A receiver needs it because with a shared pool the arena
    // offset is no longer implied by the destination core -- see rx_slot_offset().
    uint32_t slot = 0;
    uint64_t ctrl = 0;
    uint32_t sequence = 0;
    uint64_t operand[kDataRegisters] = {};
    uint32_t operand_count = 0;
    uint64_t t_notice = 0;  // when the armed word became visible
    uint64_t t_queued = 0;  // when it landed in a deque -- kHopSteal is the gap to t_start
};

// What a worker does with a decoded job. Returns bytes transferred, or 0.
//
// Injected rather than called directly so the scanner has no dependency on the transport:
// the pool can be exercised, benchmarked and its imbalance measured with a service
// function that does nothing at all, which is the only way to separate "the sweep is
// slow" from "the transport is slow".
using ServiceFn = std::function<uint64_t(const Job&, WorkerStats&)>;

// Called when a control word is found but refused. Optional; exists so a stalled run can
// report rejections while it is still stalled, instead of only in the summary it may never
// reach.
using RejectFn = std::function<void(uint32_t verdict)>;

struct ScanConfig {
    // THE VOLUME LADDER, or null. Carried on the pool's config because the pool is what
    // creates the WorkerStats, and each worker needs the pointer before it services its
    // first message. Must outlive the pool -- it points into the caller's RunStats.
    const VolumeLadder* ladder = nullptr;
    // Non-null only when the ladder is quiesced. Same lifetime rule as `ladder`.
    LadderSync* ladder_sync = nullptr;

    uint32_t workers = 0;             // 0 => one per online CPU
    std::vector<int> cpus;            // explicit affinity list; empty => 0..workers-1
    bool pin_threads = true;
    uint64_t stop_after_messages = 0; // 0 => run until stop() is called
    bool scan_rx = false;             // also watch kCtrlRx (delivery / return path)

    uint32_t steal_attempts = 2;      // victims tried per idle pass before backing off

    // The scanner records two timing samples of its own -- kHopDecode and kHopStealWait --
    // and until this existed it had no way to know a warmup was in progress: nothing in this
    // header mentioned one, and the driver's add_sample() choke point is on the other side of
    // the ServiceFn boundary. So those two rows spanned the whole run while every stage row
    // spanned the timed window, and the CSV said `warmup_applied=1` on all of them alike.
    //
    // A POINTER, not a copy or a callback. The flag is flipped by whoever owns the gate --
    // the producer thread by iteration number, or recording_now() by message count -- and the
    // scanner only ever loads it. A std::function would put an indirect call on a path taken
    // once per scan-find; a copy would freeze the answer at construction.
    //
    // nullptr means "record everything", which is the right default for a caller that has no
    // warmup and for the pool benchmarks that run with no driver at all.
    //
    // LIFETIME: must outlive join(). Both callers hold it in an object that owns the scanner.
    const std::atomic<bool>* recording = nullptr;

};

class BankScanner {
public:
    BankScanner(HostRegion& region, ScanConfig cfg, ServiceFn service, RejectFn on_reject = {});
    ~BankScanner();

    BankScanner(const BankScanner&) = delete;
    BankScanner& operator=(const BankScanner&) = delete;

    void start();
    void stop();                       // asks workers to finish; does not block
    void join();                       // blocks until every worker has exited
    bool running() const { return running_.load(std::memory_order_acquire); }
    uint64_t serviced() const { return serviced_.load(std::memory_order_relaxed); }

    RunStats collect() const;

    // The shard a worker owns.
    struct Shard {
        uint32_t first = 0;
        uint32_t last = 0;  // inclusive; first > last means an empty shard
    };
    Shard shard_of(uint32_t worker) const;

    static uint32_t default_worker_count();

private:
    struct alignas(64) Deque {
        std::mutex m;
        std::deque<Job> q;
        char pad[64];
    };

    void worker_loop(uint32_t id);
    bool try_pop_local(uint32_t id, Job& out);
    bool try_steal(uint32_t id, Job& out);
    void run_job(uint32_t id, Job& job);

    // Are this scanner's own timing samples being kept? See ScanConfig::recording.
    // Relaxed: the gate opens once and never closes, so the only cost of racing it is one
    // sample either side of the boundary -- the same tolerance the driver's own gate has.
    bool recording() const {
        return cfg_.recording == nullptr || cfg_.recording->load(std::memory_order_relaxed);
    }

    HostRegion& region_;
    ScanConfig cfg_;
    ServiceFn service_;
    RejectFn on_reject_;

    std::vector<std::thread> threads_;
    std::vector<std::unique_ptr<Deque>> deques_;
    mutable std::vector<WorkerStats> stats_;

    // Last sequence serviced per bank. The reason the control
    // word carries a sequence at all: a control word stays armed in the register after
    // the host services it -- nothing clears it but the next arming -- so without this
    // every scan pass would re-service the same message forever. Two arrays because a
    // bank's TX and RX words advance independently.
    //
    // "NEVER SERVICED" IS A SENTINEL VALUE, NOT A SECOND ARRAY, AND THAT IS THE FIX FOR A
    // REAL DUPLICATE DELIVERY. This used to be `last_seq_` plus a parallel
    // `std::vector<bool> seen_`, tested as `seen_[d][core] && last_seq_[d][core] == seq`.
    // std::vector<bool> is a PACKED BITSET: `seen_[d][core] = true` is a non-atomic
    // read-modify-write of a whole word of storage, so although workers own DISJOINT CORE
    // RANGES they do not own disjoint memory locations -- at 8 cores all eight bits sit in
    // one 64-bit word. Two workers setting their own bit concurrently lose one update, the
    // loser reads `seen_ == false` on its next pass, the && short-circuits, and the
    // still-armed word is serviced a SECOND time. Measured as `delivered 161` where 160
    // was expected, in roughly one run of four at `--selftest --cores 8 --iters 20`.
    //
    // With the sentinel there is one array, one store per find, and every element is
    // touched by exactly one thread -- the shard's owner -- so the race has no state left
    // to corrupt. Distinct uint32_t elements ARE distinct memory locations, which is
    // precisely the guarantee vector<bool> does not give.
    //
    // The sentinel must be a value ctrl_sequence() cannot return; the static_assert is
    // what stops a future widening of the sequence field from silently colliding with it
    // and wedging every bank's first message.
    static constexpr uint32_t kSeqNever = 0xFFFFFFFFu;
    static_assert(kSeqNever > kCtrlSeqMask, "the sentinel must be outside the sequence range");
    std::vector<uint32_t> last_seq_[2];

    std::atomic<bool> running_{false};
    std::atomic<bool> stopping_{false};
    std::atomic<uint64_t> serviced_{0};
    uint64_t t_start_ = 0;
    uint64_t t_end_ = 0;
    uint64_t clock_overhead_ = 0;
};

// Pins the calling thread to `cpu`. Returns an error string, empty on success. Reported
// rather than ignored: an unpinned worker in a pinned pool is a thread the scheduler can
// migrate mid-run, and it shows up as one worker with inexplicably bad numbers.
std::string pin_this_thread(int cpu);

}  // namespace tt::tt_metal::experimental
