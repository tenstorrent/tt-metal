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
//       journal_base            L1 addr of the journal (header + ring) in THIS core's L1
//       capacity                entries in the journal ring
//       src_chip                PHYSICAL chip id, stamped into the header so the host
//                               can attribute entries without a mesh map
//       sweep_interval_cyc      idle cycles between sweeps
//       publish_every           republish the header every N sweeps. The header must be
//                               STABLE for longer than a host read takes, or the read
//                               tears -- see the publish note below.
//       dbg_addr                L1 addr of 4 u32 liveness markers

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
    const uint32_t journal_base = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t capacity = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t src_chip = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t sweep_interval_cyc = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t publish_every = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t dbg_addr = get_arg_val<uint32_t>(arg_idx++);

    volatile tt_l1_ptr uint32_t* last_head = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(last_head_scratch);
    volatile tt_l1_ptr uint32_t* seq = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(seq_scratch);
    volatile tt_l1_ptr util_agg_msg_t* hdr = reinterpret_cast<volatile tt_l1_ptr util_agg_msg_t*>(journal_base);
    volatile tt_l1_ptr util_agg_entry_t* journal =
        reinterpret_cast<volatile tt_l1_ptr util_agg_entry_t*>(journal_base + UTIL_AGG_JOURNAL_OFFSET);

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

        // Phase 2: copy what advanced straight into the journal ring. No staging
        // buffer and no wrap-split: the journal is local memory, so each entry is
        // written where it belongs and the modulo handles the wrap.
        const uint32_t head_at_sweep_start = head;
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
                // The producer lapped us. Everything but the newest kRingSize entries
                // is already overwritten; count it and resync.
                lost += behind - kRingSize;
                lh = h - kRingSize;
                behind = kRingSize;
            }

            const uint64_t ring_base = ring_base_of(i);
            for (uint32_t k = 0; k < behind; k++) {
                const uint32_t ring_slot = (lh + k) % kRingSize;
                const uint32_t slot = head % capacity;
                // Read the 16 B sample DIRECTLY into the journal entry's first field.
                // That is why util_agg_entry_t puts `sample` at offset 0: a NOC read
                // into L1 must be 16 B aligned, and a 32 B entry always is.
                noc_async_read(
                    ring_base + kRingHeaderBytes + (uint64_t)ring_slot * kSampleBytes,
                    (uint32_t)&journal[slot].sample,
                    kSampleBytes,
                    kSweepNoc);
                journal[slot].core_id = i;
                journal[slot].seq = seq[i]++;
                head++;
            }
            last_head[i] = h;
        }

        // Phase 3: make sure every sample has landed BEFORE publishing the head that
        // claims it has. The host reads entries below `head` and trusts them.
        if (head != head_at_sweep_start) {
            noc_async_read_barrier(kSweepNoc);
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
            hdr->head = head;
            hdr->head_xor = head ^ UTIL_AGG_HEAD_SALT;
            hdr->sweep_count = sweep_count;
            hdr->lost = lost;
        }

        dbg[1] = sweep_count;
        dbg[2] = head;

        // IDLE, do not spin. A telemetry loop that keeps the core hot raises AICLK,
        // which changes both the power envelope and the thing being measured.
        if (sweep_interval_cyc) {
            riscv_wait(sweep_interval_cyc);
        }
    }
}
