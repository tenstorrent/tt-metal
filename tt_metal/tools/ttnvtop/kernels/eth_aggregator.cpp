// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// ttnvtop idle-eth aggregator -- Phase 2.2.
//
// Runs forever on ONE idle ethernet core. Sweeps every Tensix core's util_sampler
// ring over the LOCAL NOC and accumulates the samples into a journal in its own L1.
// The host reads that journal where it lies.
//
// IT USES NO FABRIC. That is a deliberate reversal of the earlier push design
// (PLAN_ETH_AGGREGATOR.md 3.3), and the reason is that a monitor must not degrade the
// thing it monitors:
//
//   - EDMChannelWorkerLocationInfo holds exactly ONE worker, and
//     WorkerToFabricEdmSender::open_start() overwrites it with no test-and-set and no
//     arbitration. A persistent telemetry client on sender channel 0 does not "share"
//     it -- it clobbers whatever CCL op connects next, and gets clobbered in turn.
//     Measured at 5 sweeps vs 73,284 for a contended slot (5f).
//   - Reserving a dedicated sender channel, or taking a link_idx, removes interconnect
//     resource from the workload. Not acceptable for monitoring.
//
// So the aggregator consumes no fabric resource at all. What leaves this chip is one
// host-initiated read of a compact journal, instead of the 64 per-core reads the host
// does today -- which attacks the mechanism 5b identified, starvation by transaction
// VOLUME (~770 NON_MMIO acquire/release cycles per sweep).
//
// PERSISTENCE: this kernel never returns, and that is the mechanism, not an oversight.
// Idle eth cores are always DISPATCH_MODE_HOST, and the idle-erisc firmware regains
// control only when the kernel returns. IDLE_ETH appears nowhere in
// impl/program/dispatch.cpp, so program dispatch cannot disturb us. What does end us is
// device init: assert_inactive_ethernet_cores() resets RiscType::ALL. The host detects
// that as a stalled sweep_count and re-attaches (3.5).
//
// Runtime args:
//    0: num_cores               nx*ny. NEVER assume 64 -- WH ships 64- and 80-core
//                               parts, BH up to 140.
//    1: tensix_nx               count of live TRANSLATED x coordinates
//    2: tensix_ny               count of live TRANSLATED y coordinates
//  3..3+nx-1                    the live translated x values
//  3+nx..3+nx+ny-1              the live translated y values
//    then:
//       last_head_scratch       L1 addr of num_cores u32s, our per-core cursor
//       head_scratch            L1 addr of num_cores * 16 B. Sixteen, not four: a NOC
//                               read into L1 must be 16 B aligned at BOTH ends, and
//                               `head` sits at offset 8 of the ring header -- so we
//                               pull the aligned 16 B chunk that contains it.
//       seq_scratch             L1 addr of num_cores u32s. In L1 and not on the stack:
//                               MEM_IERISC_STACK_MIN_SIZE is 128 BYTES, so a local
//                               seq[] array silently smashes the stack.
//       last_wall_scratch       L1 addr of num_cores u32s: previous wall_clock_l per core
//       last_fpu_scratch        L1 addr of num_cores u32s: previous fpu_count per core
//       journal_base            L1 addr of the header + per-core state table in THIS L1
//       capacity                entries in states[]; equals num_cores
//       src_chip                PHYSICAL chip id, stamped into the header so the host
//                               can attribute entries without a mesh map
//       sweep_interval_cyc      idle cycles between sweeps
//       publish_every           republish the header every N sweeps. The header must be
//                               STABLE for longer than a host read takes, or the read
//                               tears -- see the publish note below.
//       sample_pad_addr         L1 addr of a 16 B landing pad for one raw sample
//       dbg_addr                L1 addr of 4 u32 liveness markers
//       heartbeat_addr          ETH firmware heartbeat word, or 0 to disable. See below

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "core_config.h"
#include "util_aggregator.h"
#include "util_sampler.h"

// The ring layout we sweep. Mirrors util_sampler.h, included above so its
// static_asserts bind against the real firmware definition.
static constexpr uint32_t kRingSize = UTIL_SAMPLER_RING_SIZE;  // 62
static constexpr uint32_t kRingHeaderBytes = 32u;              // util_sampler_msg_t header
static constexpr uint32_t kRingHeadOffset = 8u;                // offsetof(util_sampler_msg_t, head)
static constexpr uint32_t kSampleBytes = 16u;                  // util_sampler_entry_t

// The NOC we read Tensix rings on. Pinned rather than inherited from noc_index: the
// coordinates below are TRANSLATED, which is NOC-independent, and pinning keeps the
// behaviour identical whatever brisc_noc_id the launch message carried.
static constexpr uint8_t kSweepNoc = 0;

// An interval longer than this means we lost the thread of a core's timeline (a device
// reset, a very long stall) rather than the core genuinely idling. Accumulating it would
// swamp the wall-cycle denominator and drive the reported utilization to zero. ~1 s at
// 1 GHz.
static constexpr uint32_t kMaxPlausibleWallDelta = 1000000000u;

// Signature UMD's eth_heartbeat_running() requires in the upper 16 bits of the
// heartbeat word (umd erisc_firmware.hpp: BASE_FW_HEARTBEAT_SIGNATURE).
static constexpr uint32_t kEthFwHeartbeatSignature = 0xABCDu;

// COOPERATIVE STOP. The host writes this to dbg[3] and the kernel RETURNS.
//
// Returning is a materially better exit than the reset `stop_aggregator()` asserts. When
// this kernel returns, idle_erisc.cc regains its wait loop -- which sets RUN_MSG_DONE and
// then posts its OWN heartbeat (0xAABB) while waiting for the next RUN_MSG_GO. So the
// core is left discoverable and relaunchable.
//
// A reset leaves it neither. Worse, it leaves our 0xABCD heartbeat word FROZEN with a
// valid signature, which tt-metal turns into a hard error on the next device open
// ("Timed out waiting for ETH heartbeat ... Stuck at 0xabcd....") -- a board reset to
// clear. That is the standing open risk in
// [[tt-eth-idle-core-firmware-contracts]], and this is the fix for it.
//
// It rides in dbg[3] rather than a new runtime argument because the argument budget is
// genuinely full: the args must fit between rta_offset and kernel_text_offset, and on a
// 120-core Blackhole that is 39 words against a 160 B gap -- 4 bytes spare. dbg[3]
// otherwise holds num_cores, which the journal header already carries.
static constexpr uint32_t kStopRequest = 0x504F5453u;  // 'STOP' little-endian

void kernel_main() {
    size_t arg_idx = 0;
    const uint32_t num_cores = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t tensix_nx = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t tensix_ny = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t x_arg_base = arg_idx;
    const uint32_t y_arg_base = x_arg_base + tensix_nx;
    arg_idx = y_arg_base + tensix_ny;

    const uint32_t last_head_scratch = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t head_scratch = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t seq_scratch = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t last_wall_scratch = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t last_fpu_scratch = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t journal_base = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t capacity = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t src_chip = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t sweep_interval_cyc = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t publish_every = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t sample_pad_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t dbg_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t heartbeat_addr = get_arg_val<uint32_t>(arg_idx++);

    volatile tt_l1_ptr uint32_t* last_head = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(last_head_scratch);
    volatile tt_l1_ptr uint32_t* seq = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(seq_scratch);
    volatile tt_l1_ptr util_agg_msg_t* hdr = reinterpret_cast<volatile tt_l1_ptr util_agg_msg_t*>(journal_base);
    volatile tt_l1_ptr util_agg_core_state_t* states =
        reinterpret_cast<volatile tt_l1_ptr util_agg_core_state_t*>(journal_base + UTIL_AGG_STATES_OFFSET);
    volatile tt_l1_ptr uint32_t* last_wall = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(last_wall_scratch);
    volatile tt_l1_ptr uint32_t* last_fpu = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(last_fpu_scratch);

    // One 16 B landing pad per sweep for the sample we are folding in. A NOC read into
    // L1 must be 16 B aligned at both ends, and util_agg_core_state_t no longer has a
    // 16 B sample field to read straight into -- we need the raw sample to compute
    // deltas from, then we discard it.
    volatile tt_l1_ptr util_agg_sample_t* pad =
        reinterpret_cast<volatile tt_l1_ptr util_agg_sample_t*>(sample_pad_addr);

    volatile tt_l1_ptr uint32_t* dbg = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dbg_addr);
    dbg[0] = 0xA66E0000u;  // reached kernel_main
    dbg[1] = 0;            // sweeps completed
    dbg[2] = 0;            // entries written
    dbg[3] = num_cores;

    // `head` lives at offset 8 of the 16 B chunk we fetched for core i.
    auto ring_head = [&](uint32_t i) -> uint32_t {
        return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(head_scratch + i * 16u + kRingHeadOffset);
    };

    // Core i's sampler ring, addressed directly in TRANSLATED space.
    //
    // No host-supplied address table: the live cores are the CROSS PRODUCT of a live-x
    // list and a live-y list, because harvesting removes whole rows (WH) or whole
    // columns (BH). So nx + ny coordinates describe all nx*ny cores. NOT a contiguous
    // rectangle -- BH takes its translated coords from the NOC0 core list, which skips
    // the non-Tensix columns, so its live x values have gaps (2.1b).
    //
    // Deriving the address here is safe ONLY because these are TRANSLATED coordinates,
    // which are NOC-independent by construction. NOC_XY_ADDR is
    // (y << 42) | (x << 36) | addr, byte-identical on WH and BH, with no
    // NOC_X_PHYS_COORD flip. Do not substitute get_noc_addr(), which resolves against
    // this kernel's own noc_index and would silently address the mirrored core.
    auto ring_base_of = [&](uint32_t i) -> uint64_t {
        const uint32_t tx = get_arg_val<uint32_t>(x_arg_base + (i % tensix_nx));
        const uint32_t ty = get_arg_val<uint32_t>(y_arg_base + (i / tensix_nx));
        return NOC_XY_ADDR(tx, ty, (uint64_t)MEM_UTIL_SAMPLER_BASE);
    };

    for (uint32_t i = 0; i < num_cores; i++) {
        last_head[i] = 0;
        seq[i] = 0;
        last_wall[i] = 0;
        last_fpu[i] = 0;
        states[i].busy_cycles = 0;
        states[i].wall_cycles = 0;
        states[i].samples = 0;
        states[i].kernel_id = 0;
        states[i].resets = 0;
        states[i].seq = 0;
        states[i].counter_sel = 0;
        states[i].flags = 0;
    }

    // Static fields and their checksum: written once, never touched again.
    hdr->capacity = capacity;
    hdr->num_cores = num_cores;
    hdr->src_chip = src_chip;
    hdr->hdr_checksum = util_agg_hdr_checksum(UTIL_AGG_MAGIC, UTIL_AGG_VERSION, capacity, num_cores, src_chip);
    hdr->sweep_count = 0;
    hdr->lost = 0;
    hdr->head = 0;
    hdr->head_xor = 0u ^ UTIL_AGG_HEAD_SALT;
    hdr->version = UTIL_AGG_VERSION;
    hdr->magic = UTIL_AGG_MAGIC;

    uint32_t head = 0;
    uint32_t lost = 0;
    uint32_t sweep_count = 0;

    // First sweep only latches the rings' current heads. Everything already in the
    // rings predates us and has no reliable ordering against our journal.
    bool primed = false;

    while (true) {
        // Phase 1: fetch every ring's head in one burst. num_cores reads issued back to
        // back, then a single barrier -- the round trips overlap instead of
        // serializing. This is intra-chip NOC traffic and never touches ethernet.
        for (uint32_t i = 0; i < num_cores; i++) {
            noc_async_read(ring_base_of(i), head_scratch + i * 16u, 16u, kSweepNoc);
        }
        noc_async_read_barrier(kSweepNoc);

        if (!primed) {
            for (uint32_t i = 0; i < num_cores; i++) {
                last_head[i] = ring_head(i);
            }
            primed = true;
        }

        // Phase 2: fold what advanced into each core's accumulated state.
        //
        // This is the arithmetic the host used to do: wrap-aware wall_clock_l deltas,
        // FPU counter deltas, and counter-reset detection. Doing it here is what keeps
        // the published payload fixed-size.
        for (uint32_t i = 0; i < num_cores; i++) {
            const uint32_t h = ring_head(i);
            uint32_t lh = last_head[i];
            if (h == lh) {
                continue;
            }
            // Unsigned subtraction is wrap-correct: the ring head is monotonic and only
            // wraps at 2^32, which at 1 kHz per core is ~50 days.
            uint32_t behind = h - lh;
            if (behind > kRingSize) {
                // The producer lapped us. Everything but the newest kRingSize entries is
                // already overwritten; count it and resync. NOTE this counts what the
                // ON-CHIP sweep missed. The host cannot lose anything -- the
                // accumulators below are monotonic.
                lost += behind - kRingSize;
                lh = h - kRingSize;
                behind = kRingSize;
            }

            const uint64_t ring_base = ring_base_of(i);
            uint32_t prev_wall = last_wall[i];
            uint32_t prev_fpu = last_fpu[i];
            uint32_t busy_acc = states[i].busy_cycles;
            uint32_t wall_acc = states[i].wall_cycles;
            uint32_t nsamples = states[i].samples;
            uint32_t nresets = states[i].resets;
            uint32_t kid = states[i].kernel_id;
            uint32_t csel = states[i].counter_sel;

            for (uint32_t k = 0; k < behind; k++) {
                const uint32_t ring_slot = (lh + k) % kRingSize;
                noc_async_read(
                    ring_base + kRingHeaderBytes + (uint64_t)ring_slot * kSampleBytes,
                    sample_pad_addr,
                    kSampleBytes,
                    kSweepNoc);
                noc_async_read_barrier(kSweepNoc);

                const uint32_t wall_now = pad->wall_clock_l;
                const uint32_t fpu_now = pad->fpu_count;
                kid = pad->kernel_id;
                csel = pad->counter_sel;

                if (nsamples != 0 || prev_wall != 0) {
                    // Counter reset: the Tensix perf counter went backwards, which
                    // happens when a kernel re-arms it. A post-reset absolute is not a
                    // delta -- drop the interval rather than accumulate garbage.
                    if (fpu_now < prev_fpu) {
                        nresets++;
                    } else {
                        const uint32_t wall_d = wall_now - prev_wall;  // wrap-correct
                        const uint32_t fpu_d = fpu_now - prev_fpu;
                        // Guard against an implausible interval, which means we lost the
                        // thread of this core's timeline rather than genuinely idled.
                        //
                        // fpu_d <= wall_d is the one that was missing, and it is not a
                        // nicety: a core cannot be busy for more cycles than elapsed. When
                        // it fails, the timeline is broken -- `wall_clock_l` is 32-bit and
                        // wraps every ~4.3 s at 1 GHz, so a core whose ring the sweep has
                        // not reached for that long yields a SMALL wrapped wall_d against a
                        // LARGE fpu_d. Accumulating that pushes busy_acc past wall_acc, the
                        // host divides, gets >1, and clamps -- which is why whole remote
                        // chips displayed a flat, impossible 100.0% on every core while the
                        // host's own register path (which has always had this guard, see
                        // the sampler's `fpu_d <= wall_d`) read ~22% for the same load.
                        //
                        // Dropping the interval is right: it is unmeasurable, not saturated.
                        // It lands in `resets`, which the host already surfaces.
                        if (wall_d != 0 && wall_d < kMaxPlausibleWallDelta && fpu_d <= wall_d) {
                            busy_acc += fpu_d;
                            wall_acc += wall_d;
                            nsamples++;
                        } else if (wall_d != 0) {
                            nresets++;
                        }
                    }
                }
                prev_wall = wall_now;
                prev_fpu = fpu_now;
            }

            last_wall[i] = prev_wall;
            last_fpu[i] = prev_fpu;
            states[i].busy_cycles = busy_acc;
            states[i].wall_cycles = wall_acc;
            states[i].kernel_id = kid;
            states[i].resets = nresets;
            states[i].counter_sel = (uint8_t)csel;
            // samples LAST of the value fields, then seq: a host that sees seq advance
            // knows this core's block is fully written.
            states[i].samples = nsamples;
            states[i].seq = states[i].seq + 1;
            head += behind;
            last_head[i] = h;
        }

        // Publish. `head` then `head_xor`, adjacent in the first 16 B chunk, so a
        // host that reads that chunk can tell whether it caught us mid-write --
        // whatever the rest of the header did. See the field-order note in
        // util_aggregator.h: a remote read is served in 16 B chunks from different
        // moments, and an earlier layout with head at offset 8 and its checksum at
        // offset 32 tore on every single remote read.
        //
        // `publish_every` throttles how often we touch the header at all. Correctness
        // no longer depends on it -- chunk 0 is self-validating -- but writing less
        // often keeps the host's view stable for longer and costs nothing.
        sweep_count++;
        if (publish_every == 0 || (sweep_count % publish_every) == 0) {
            // Republish the magic, not just set it once at startup.
            //
            // The launcher zeroes this header before writing the go word so a stale
            // journal cannot be read as live. On a REMOTE chip those two writes are not
            // ordered against each other: measured 2026-08-30 on T3K chip 7, the zeroing
            // landed AFTER the kernel had already stamped the magic, leaving a live
            // aggregator publishing an advancing sweep_count under magic 0x0. That is
            // invisible to `probe_landings` (which gates on the magic), so the launcher
            // called it NOT RUNNING and `--stop-aggregator` could not reach it either --
            // an unstoppable aggregator on a core nobody could see.
            //
            // Writing it here costs one store per publish_every sweeps and makes the
            // journal self-healing against any late or reordered header write.
            hdr->magic = UTIL_AGG_MAGIC;
            hdr->head = head;
            hdr->head_xor = head ^ UTIL_AGG_HEAD_SALT;
            hdr->sweep_count = sweep_count;
            hdr->lost = lost;
        }

        dbg[1] = sweep_count;
        dbg[2] = head;

        // Cooperative stop, checked once per sweep.
        //
        // The journal's magic is cleared on the way out. A journal left behind keeps a
        // valid magic and a valid header checksum forever with its sweep_count simply
        // frozen, and every reader that trusted the magic read a dead aggregator as a
        // live one -- twice. Clearing it means a later probe finds nothing, which is the
        // truth.
        if (dbg[3] == kStopRequest) {
            hdr->magic = 0u;
            dbg[0] = 0xA66E0001u;  // exited on request
            return;
        }

        // Keep the ethernet firmware's heartbeat alive.
        //
        // We occupy ERISC0, so the idle-erisc firmware that normally increments this is
        // not running. UMD's topology discovery polls it on EVERY eth core
        // (eth_heartbeat_running) and waits out its timeouts when it does not change --
        // which stalls every subsequent device open, for any process, not just ours.
        // Measured before this: discovery hanging for 5-8 minutes and needing a board
        // reset. Maintaining the word is what makes this kernel a good citizen rather
        // than a platform-wide hazard.
        //
        // Upper 16 bits are the signature UMD checks; the low 16 must simply change.
        if (heartbeat_addr) {
            volatile tt_l1_ptr uint32_t* hb = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(heartbeat_addr);
            *hb = (kEthFwHeartbeatSignature << 16) | (sweep_count & 0xFFFFu);
        }

        // IDLE, do not spin. A telemetry loop that keeps the core hot raises AICLK,
        // which changes both the power envelope and the thing being measured.
        if (sweep_interval_cyc) {
            riscv_wait(sweep_interval_cyc);
        }
    }
}
