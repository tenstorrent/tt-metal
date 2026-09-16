// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

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

#include <tt-metalium/experimental/sockets/internal/host_region.hpp>
#include <tt-metalium/experimental/sockets/internal/host_stats.hpp>
#include <tt-metalium/experimental/sockets/internal/host_uva_layout.hpp>

namespace tt::tt_metal::experimental {

// used to snapshot an armed control word. the worker the control word once, and
// everything downstream works from that copy rather re-reading the register.
// Re-reading would race the next arming of the same bank, and the failure mode is
// servicing a message with one iteration's opcode and the next iteration's operands.
// Which control register armed this job. The two directions are serviced by completely
// different code -- a TX job routes and sends, an RX job delivers into L1 -- so the
// direction travels with the job rather than being re-derived downstream.
enum class Dir : uint32_t { Tx = 0, Rx = 1 };

struct Job {
    Dir dir = Dir::Tx;
    uint32_t core = 0;
    // which receive slot this notice came from. Always 0 on the TX side and at payloads big
    // enough that only one fits. A receiver needs it because with a shared pool the arena
    // offset is no longer implied by the destination core -- see rx_slot_offset().
    uint32_t slot = 0;
    uint64_t ctrl = 0;
    uint32_t sequence = 0;
    uint64_t operand[kDataRegisters] = {};
    uint32_t operand_count = 0;
    uint64_t t_notice = 0;  // when the armed word became visible
    uint64_t t_queued = 0;  // when it landed in a deque -- kHopStealWait is the gap to t_start
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
    // volume ladder or null. Carried on the pool's config because the pool is what
    // creates the WorkerStats, and each worker needs the pointer before it services its
    // first message. Must outlive the pool -- it points into the caller's RunStats.
    // Non-null only when the ladder is quiesced. Same lifetime rule as `ladder`.

    uint32_t workers = 0;   // 0 => one per online CPU
    std::vector<int> cpus;  // explicit affinity list; empty => 0..workers-1
    bool pin_threads = true;
    uint64_t stop_after_messages = 0;  // 0 => run until stop() is called
    bool scan_rx = false;              // also watch kCtrlRx (delivery / return path)

    uint32_t rx_slots = 1;

    uint32_t steal_attempts = 2;  // victims tried per idle pass before backing off

    // The scanner records two timing samples of its own -- kHopDecode and kHopStealWait
    const std::atomic<bool>* recording = nullptr;
};

class BankScanner {
public:
    BankScanner(HostRegion& region, ScanConfig cfg, ServiceFn service, RejectFn on_reject = {});
    ~BankScanner();

    BankScanner(const BankScanner&) = delete;
    BankScanner& operator=(const BankScanner&) = delete;

    void start();
    void stop();  // asks workers to finish; does not block
    void join();  // blocks until every worker has exited
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

    // See ScanConfig::recording.
    // Relaxed: the gate opens once and never closes, so the only cost of racing it is one
    // sample either side of the boundary -- the same tolerance the driver's own gate has.
    bool recording() const { return cfg_.recording == nullptr || cfg_.recording->load(std::memory_order_relaxed); }

    HostRegion& region_;
    ScanConfig cfg_;
    ServiceFn service_;
    RejectFn on_reject_;

    std::vector<std::thread> threads_;
    std::vector<std::unique_ptr<Deque>> deques_;
    mutable std::vector<WorkerStats> stats_;

    // last sequence processed; one element per (direction, core, slot). A control word stays
    // armed in the register after the host services it -- nothing clears it but the next
    // arming -- so without this every scan pass would re-service the same message forever.
    // Two arrays because a bank's TX and RX words advance independently; the slot dimension
    // because pool slots are independent streams, so slot 2's sequence says nothing about
    // slot 5's. Indexed core * kRxNoticeSlots + slot -- see the BankScanner constructor.
    //
    // kSeqNever must be a value ctrl_sequence() cannot return, or a bank's FIRST message reads
    // as already-serviced; the static_assert catches a future widening of the sequence field
    // colliding with it.
    static constexpr uint32_t kSeqNever = 0xFFFFFFFFu;
    static_assert(kSeqNever > kCtrlSeqMask, "the sentinel must be outside the sequence range");
    std::vector<uint32_t> last_seq_[2];

    // next expected RX slot per core. The H2DSocket ring's read pointer advances one payload
    // per delivery, while the peer writes whichever slot its own send counter names -- and MPI
    // RMA does not order separate Rputs, so slot 3 can land before slot 2. A worker therefore
    // dispatches only from the slot it expects, advancing on dispatch, which keeps the device
    // reading the slot the pointer is actually on. Payloads still overlap on the wire.
    std::vector<uint32_t> next_rx_slot_;

    // one RX job per core in flight anywhere. next_rx_slot_ orders what gets queued, which is
    // not enough: the sweep queues every armed slot of a core in one lap, and try_pop_local()
    // is LIFO by design -- the owner takes q.back() for locality -- so an ordered queue came
    // back reversed and a high slot reached delivery while the ring pointer sat on a low one.
    // Popping FIFO would not fix it either: a thief and the owner can hold consecutive slots of
    // one core and race into write_payload in either order. So a core has at most one rx job
    // outstanding -- set when the sweep queues one, cleared after service_() returns, by
    // whichever worker ran it. This costs no pipelining: per-core delivery was already
    // serialised by D2H2H2DSocket::deliver_m_, and the hand-off is small beside the wire leg
    // (compare the diag:l1-write and h2h:payload-at-peer rows, both config-dependent). The
    // overlap the send window buys is on the WIRE and is untouched.
    std::vector<std::atomic<uint32_t>> rx_busy_;

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
