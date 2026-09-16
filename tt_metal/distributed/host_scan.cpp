// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/sockets/internal/host_scan.hpp>

#include <pthread.h>
#include <sched.h>

#include <cerrno>
#include <cstring>
#include <sstream>

namespace tt::tt_metal::experimental {

std::string pin_this_thread(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    const int rc = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    if (rc != 0) {
        std::ostringstream m;
        m << "pthread_setaffinity_np(cpu " << cpu << ") failed: " << std::strerror(rc);
        return m.str();
    }
    return {};
}

uint32_t BankScanner::default_worker_count() {
    const unsigned hc = std::thread::hardware_concurrency();
    return hc == 0 ? 1u : hc;
}

BankScanner::BankScanner(HostRegion& region, ScanConfig cfg, ServiceFn service, RejectFn on_reject) :
    region_(region), cfg_(std::move(cfg)), service_(std::move(service)), on_reject_(std::move(on_reject)) {
    if (cfg_.workers == 0) {
        cfg_.workers = default_worker_count();
    }
    // More workers than banks is not an error but it is worth capping: the extra threads
    // would own empty shards and do nothing but steal, which is measurable overhead for
    // no scan coverage. mmio_bench found the knee at exactly the CPU count (32 on that
    // box) and per-thread bandwidth degrading past it, so oversubscription here is a
    // known-bad regime rather than a hypothetical one.
    const uint32_t cores = region_.cores_in_use();
    cfg_.workers = std::min(cfg_.workers, cores);

    deques_.reserve(cfg_.workers);
    for (uint32_t i = 0; i < cfg_.workers; ++i) {
        deques_.push_back(std::make_unique<Deque>());
    }
    stats_.resize(cfg_.workers);
    // kSeqNever, not 0: sequence 0 is a legitimate value a bank's FIRST message can
    // carry, and initialising to it would make that message look already-serviced.
    // ONE SEQUENCE PER (DIRECTION, CORE, SLOT). It was per (direction, core), which was
    // right while a core had one receive slot with one lifetime sender. Pool slots are
    // independent streams -- slot 2's sequence says nothing about slot 5's -- and sharing
    // one counter would drop a legitimate message as a duplicate whenever two slots
    // happened to carry the same sequence number.
    next_rx_slot_.assign(cores, 0u);
    rx_busy_ = std::vector<std::atomic<uint32_t>>(cores);
    for (auto& b : rx_busy_) {
        b.store(0u, std::memory_order_relaxed);
    }
    for (auto& per_direction : last_seq_) {
        per_direction.assign(static_cast<size_t>(cores) * kRxNoticeSlots, kSeqNever);
    }
}

BankScanner::~BankScanner() {
    stop();
    join();
}

BankScanner::Shard BankScanner::shard_of(uint32_t worker) const {
    const uint32_t cores = region_.cores_in_use();
    const uint32_t w = cfg_.workers;
    // Remainder spread over the low-numbered workers rather than dumped on the last one.
    const uint32_t base = cores / w;
    const uint32_t extra = cores % w;
    const uint32_t first = worker * base + (worker < extra ? worker : extra);
    const uint32_t count = base + (worker < extra ? 1u : 0u);
    if (count == 0) {
        return Shard{1, 0};
    }
    return Shard{first, first + count - 1};
}

void BankScanner::start() {
    if (running_.exchange(true)) {
        return;
    }
    stopping_.store(false, std::memory_order_release);
    clock_overhead_ = measure_clock_overhead_ns();
    t_start_ = now_ns();
    threads_.reserve(cfg_.workers);
    for (uint32_t i = 0; i < cfg_.workers; ++i) {
        threads_.emplace_back([this, i] { worker_loop(i); });
    }
}

void BankScanner::stop() { stopping_.store(true, std::memory_order_release); }

void BankScanner::join() {
    for (auto& t : threads_) {
        if (t.joinable()) {
            t.join();
        }
    }
    threads_.clear();
    if (running_.exchange(false)) {
        t_end_ = now_ns();
    }
}

bool BankScanner::try_pop_local(uint32_t id, Job& out) {
    Deque& d = *deques_[id];
    std::lock_guard<std::mutex> g(d.m);
    if (d.q.empty()) {
        return false;
    }
    // LIFO for the owner: the job just pushed is the one whose operands are still in this
    // core's cache.
    out = d.q.back();
    d.q.pop_back();
    return true;
}

bool BankScanner::try_steal(uint32_t id, Job& out) {
    const uint32_t w = cfg_.workers;
    if (w < 2) {
        return false;
    }
    // Start at a different victim per worker so an idle pool does not have every thread
    // hammering worker 0's mutex in lockstep. Rotating by id costs nothing and turns a
    // convoy into a spread.
    for (uint32_t attempt = 0; attempt < cfg_.steal_attempts; ++attempt) {
        const uint32_t victim = (id + 1 + attempt) % w;
        if (victim == id) {
            continue;
        }
        Deque& d = *deques_[victim];
        // try_lock, not lock: a thief must never block the owner. If the victim is busy
        // with its own deque, moving on to another victim is strictly better than waiting
        // for a job that the owner is probably about to take anyway.
        std::unique_lock<std::mutex> g(d.m, std::try_to_lock);
        if (!g.owns_lock() || d.q.empty()) {
            continue;
        }
        // FIFO from the tail: the oldest job, farthest from what the owner will take next.
        out = d.q.front();
        d.q.pop_front();
        stats_[id].stolen++;
        stats_[victim].donated++;
        return true;
    }
    return false;
}

void BankScanner::run_job(uint32_t id, Job& job) {
    WorkerStats& ws = stats_[id];
    const uint64_t t_start = now_ns();
    // Warmup-gated like every other sample in the tree. It was not, until now: this is one of
    // the two sites where the scanner records its own timing, and neither could see the gate.
    if (recording() && job.t_queued != 0 && t_start > job.t_queued) {
        ws.hop[kHopStealWait].add(t_start - job.t_queued);
    }
    const uint64_t bytes = service_ ? service_(job, ws) : 0;
    // RELEASED HERE, not at queue time and not by the owner specifically: a stolen job is
    // serviced by a thief, and the core must stay claimed until the delivery this job carries
    // has actually reached the device. Unconditional store rather than a compare: only one rx
    // job per core exists, so this is the one that claimed it.
    if (job.dir == Dir::Rx && job.core < rx_busy_.size()) {
        rx_busy_[job.core].store(0u, std::memory_order_release);
    }
    ws.found++;
    ws.bytes += bytes;
    serviced_.fetch_add(1, std::memory_order_relaxed);
}

void BankScanner::worker_loop(uint32_t id) {
    // Bound before the pin attempt so a failure has somewhere to land. Safe: stats_ is
    // sized in the constructor, long before start() spawns any thread.
    WorkerStats& ws = stats_[id];

    if (cfg_.pin_threads) {
        const int cpu = id < cfg_.cpus.size() ? cfg_.cpus[id] : static_cast<int>(id);
        // A failure here is recorded, not fatal: an unpinned worker still does correct
        // work, it just makes its own timing rows untrustworthy. Better to run and say so
        // than to refuse to start on a machine with a restrictive cpuset.
        const std::string err = pin_this_thread(cpu);
        if (!err.empty()) {
            ws.pinned = false;
            ws.pin_error = err;
        }
    }

    const Shard shard = shard_of(id);
    const bool has_shard = shard.first <= shard.last;

    Job job;
    while (!stopping_.load(std::memory_order_acquire)) {
        if (cfg_.stop_after_messages != 0 &&
            serviced_.load(std::memory_order_relaxed) >= cfg_.stop_after_messages) {
            break;
        }

        bool did_work = false;

        // 1. scan my own shard, both TX & RX in the same pass over the same bank. Two passes would
        // double the cache traffic over the bank array for no benefit the line holding one is
        // already being pulled in when the other is read.
        if (has_shard) {
            const uint32_t dirs = cfg_.scan_rx ? 2u : 1u;
            for (uint32_t core = shard.first; core <= shard.last; ++core) {   // core
                // THE RX SIDE IS A POOL, swept slot by slot. TX stays single: a core has
                // one message outbound at a time whatever the topology, so there is nothing
                // to sweep there.
                //
                // This MUST equal kNumAliasRingSlots -- the value the aliased ring was sized with
                // and the sender's window. Sweeping fewer than the sender uses loses messages
                // outright: they arm a slot nobody reads. cfg_.rx_slots defaults to 1, so a
                // caller that does not set it keeps the old behaviour exactly.
                const uint32_t slots = cfg_.rx_slots ? cfg_.rx_slots : 1u;
                for (uint32_t d = 0; d < dirs; ++d) {
                  const Dir dir = (d == 0) ? Dir::Tx : Dir::Rx;
                  const uint32_t slot_count = (dir == Dir::Rx) ? slots : 1u;
                  for (uint32_t slot = 0; slot < slot_count; ++slot) {
                    // IN SLOT ORDER, ONE AT A TIME -- see next_rx_slot_ and rx_busy_ in the
                    // header. An armed slot that is not the expected one is left alone, and no
                    // new slot is taken while this core still has one in flight; the next lap
                    // picks it up once its predecessor has reached the device.
                    if (dir == Dir::Rx && slots > 1u) {
                        if (slot != next_rx_slot_[core]) {
                            continue;
                        }
                        if (rx_busy_[core].load(std::memory_order_acquire) != 0u) {
                            continue;
                        }
                    }
                    const uint64_t ctrl = load_acquire(dir == Dir::Tx
                                                           ? region_.ctrl_tx(core)
                                                           : region_.rx_notice(core, slot));
                    ws.scanned++;

                    const CtrlVerdict v = ctrl_validate(ctrl);
                    if (v == kCtrlIdle) {
                        continue;
                    }
                    if (v != kCtrlOk) {
                        // Counted by class and skipped. NOT serviced and NOT fatal: a bank
                        // holding a word this build cannot parse is a fact to report, and
                        // stopping the pool for it would lose every other core's data.
                        ws.rejected[v]++;
                        if (on_reject_) {
                            on_reject_(v);
                        }
                        continue;
                    }

                    const uint32_t seq = ctrl_sequence(ctrl);
                    // The parallel `seen_` bitset this replaced was shared storage between
                    // workers that owned disjoint cores, and losing one of its
                    // read-modify-writes re-serviced an already-delivered message. See
                    // kSeqNever in host_scan.hpp.
                    const size_t seq_idx = static_cast<size_t>(core) * kRxNoticeSlots + slot;
                    if (last_seq_[d][seq_idx] == seq) {
                        continue;
                    }
                    last_seq_[d][seq_idx] = seq;
                    // accepted, so slot's turn is over. Advancing here rather than after
                    // delivery is correct because the job is now queued in order; the deque
                    // preserves that order and the device sees the payloads in ring order.
                    if (dir == Dir::Rx && slots > 1u) {
                        next_rx_slot_[core] = (slot + 1u) % slots;
                    }

                    const uint64_t t_notice = now_ns();

                    job = Job{};
                    job.dir = dir;
                    job.core = core;
                    job.slot = slot;
                    job.ctrl = ctrl;
                    job.sequence = seq;
                    job.t_notice = t_notice;

                    // Snapshot the operands here, in the scan -- as close as possible to
                    // the control word that vouched for them. A job that sits in a deque
                    // before being stolen could otherwise read operands the sender has
                    // already overwritten with its next message.
                    //
                    // the TX and RX carry operands differently. A t6 writes TX operands into
		    // real registers over PCIe, where the bus orders them ahead of the trigger
		    // for free. A remote host has no such guarantee, so an inbound notice
		    // packs its operands into the same line as its control word and arrives as
		    // one RMA -- the receiver cannot see the trigger without them.
		    //
                    if (dir == Dir::Rx) {
                        // THIS SLOT'S notice line, not the core's. With a pool the operands
                        // live beside the control word that vouched for them, and reading
                        // ctrl_rx unconditionally would snapshot slot 0's operands under any
                        // slot's trigger -- a plausible wrong length and a plausible wrong
                        // address, with nothing reporting either.
                        const volatile uint8_t* line =
                            reinterpret_cast<const volatile uint8_t*>(region_.rx_notice(core, slot));
                        job.operand[0] = load_acquire(
                            reinterpret_cast<const volatile uint64_t*>(line + kNoticeLengthOffset));
                        job.operand[1] = load_acquire(
                            reinterpret_cast<const volatile uint64_t*>(line + kNoticeElapsedOffset));
                        job.operand[2] = load_acquire(
                            reinterpret_cast<const volatile uint64_t*>(line + kNoticeOriginOffset));
                        job.operand_count = 3;
                        // the 4th word exists exclusively for the store pseudo instruction, and the
			// opcode in word 0 is what says so. It arrived in the same indivisible
			// transfer as the control word, so there is nothing to order and nothing to
                        // wait for -- but reading it unconditionally would pick up whatever
                        // the previous message left in that word on a kOpSendUva notice, and
                        // a stale UVA is a store to a plausible wrong address.
                        if (ctrl_op_is_store(ctrl_opcode(ctrl))) {
                            job.operand[3] = load_acquire(
                                reinterpret_cast<const volatile uint64_t*>(line + kNoticeUvaOffset));
                            job.operand_count = 4;
                        }
                    } else {
                        // the immediate pseudo instruction has no base/count -- those bits are its length --
                        // so its operand layout is fixed by the opcode instead: register 0 is
                        // the destination UVA, 2 the elapsed accumulator, 3 the origin core.
                        // Register 1 is not written by the kernel and is not read here.
                        // Decoding it with ctrl_base()/ctrl_count() would read a register
                        // range derived from a byte count.
                        if (ctrl_op_has_imm(ctrl_opcode(ctrl))) {
                            job.operand[0] = load_acquire(region_.reg(core, 0));
                            job.operand[1] = 0;
                            job.operand[2] = load_acquire(region_.reg(core, 2));
                            job.operand[3] = load_acquire(region_.reg(core, 3));
                            job.operand_count = 4;
                        } else {
                            const uint32_t base = ctrl_base(ctrl);
                            const uint32_t count = ctrl_count(ctrl);
                            for (uint32_t k = 0; k < count; ++k) {
                                job.operand[k] = load_acquire(region_.reg(core, base + k));
                            }
                            job.operand_count = count;
                        }
                    }
                    // Zero TX control register. The instruction has been decoded into
                    // a job; the register is now idle and says so.
                    //
                    // after the operand snapshot, not before. Zeroing first would publish
                    // "this slot is free" while we still had the operands to read, and a
                    // producer that re-armed in that window would have its new operands read
                    // against the old trigger -- the exact tearing the single indivisible
                    // trigger write exists to make impossible.
                    //
                    // No CAS necessary. Workers scan disjoint core shards -- the same property
                    // last_seq_ above documents -- so exactly one thread ever reaches this
                    // line for a given core. Stealing happens on the job deque, after this.
                    //
                    // RELEASE, so the operand reads above cannot sink past it.
                    //
                    // bi-directional and the RX case needs it's own argument.
                    //
                    // An RX notice is armed by the peer's RMA (or by arm_rx_local on the
                    // one-host path). The peer never READS this register -- it only writes it
                    // -- so a zero here tells it nothing, and the credit stays the sole
                    // authority on when the slot may be reused. Zeroing is therefore not a
                    // second, competing flow-control mechanism; it is just the word saying
                    // what is already true.
                    //
                    // The ordering enforces safety, and getting it wrong loses
                    // messages silently. deliver_to_l1() returns the credit after the bytes
                    // are in L1, and the peer will not re-arm until it sees that credit. So a
                    // zero issued at scan time, before delivery, before the credit
                    // and cannot land on top of a fresh notice. Zeroing after the credit
                    // would be a real race: the peer's next RMA could arrive between the
                    // credit and this store, and we would erase a live message with nothing
                    // anywhere reporting it.
                    //
                    // additive on TX: the kernel still paces on
                    // rdma_completion rather than polling its own register, so nothing
                    // depends on the zero yet. It is what a kernel that watches its own
                    // register would need, and it makes a serviced bank read idle in a stall
                    // dump instead of holding a word that was already handled.
                    store_release(
                        dir == Dir::Tx ? region_.ctrl_tx(core) : region_.rx_notice(core, slot), 0);

                    // Warmup-gated: the other of the scanner's two own samples. Before the
                    // gate reached this class, this row alone carried the startup transient
                    // while every stage row beside it had it removed.
                    if (recording()) {
                        ws.hop[kHopDecode].add(now_ns() - t_notice);
                    }

                    // do not service inline - enqueue
                    //
                    // Queuing unconditionally is what creates stealable work: a pass that
                    // finds five armed banks leaves four for an idle worker to take.
                    // CLAIM BEFORE QUEUING, so no lap can queue a second slot for this core
                    // between the push and the service. Released in run_job() after service_().
                    if (dir == Dir::Rx && slots > 1u) {
                        rx_busy_[core].store(1u, std::memory_order_release);
                    }
                    {
                        Deque& dq = *deques_[id];
                        std::lock_guard<std::mutex> g(dq.m);
                        job.t_queued = now_ns();
                        dq.q.push_back(job);
                    }
                    did_work = true;
                  }   // slot
                }     // dir
              }       // core
            }

        // --- 2. drain my own deque -----------------------------------------
        while (try_pop_local(id, job)) {
            run_job(id, job);
            did_work = true;
        }

        // --- 3. steal ------------------------------------------------------
        if (!did_work) {
            if (try_steal(id, job)) {
                run_job(id, job);
                did_work = true;
            }
        }

        if (!did_work) {
            ws.idle_spins++;
            // A pause, not a yield or a sleep. This loop's entire job is to notice an
            // armed word quickly, and both alternatives cost far more than they save:
            // sched_yield() can hand the core to another runnable thread for a full
            // timeslice, and any sleep puts a floor under the notice latency that is
            // orders of magnitude above the thing being measured. The pause hint just
            // stops the spin from saturating the memory pipeline and the SMT sibling.
#if defined(__x86_64__) || defined(__i386__)
            __builtin_ia32_pause();
#else
            // Not a portability afterthought: Blackhole hosts are x86, but a bare
            // compiler builtin would make this file fail to compile anywhere else
            // rather than merely spin a little hotter.
            std::this_thread::yield();
#endif
        }
    }
}

RunStats BankScanner::collect() const {
    RunStats s;
    s.per_worker = stats_;
    s.clock_overhead_ns = clock_overhead_;
    s.wall_ns = 0;
    if(t_end_ > t_start_) {
       s.wall_ns = t_end_ - t_start_;
    }
    else if(t_start_) {
       s.wall_ns = now_ns() - t_start_;
    }
    return s;
}

}  // namespace tt::tt_metal::experimental
