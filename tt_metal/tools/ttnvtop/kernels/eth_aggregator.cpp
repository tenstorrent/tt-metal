// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// ttnvtop idle-eth aggregator -- Phase 2.2, milestone M1.
//
// Runs forever on ONE idle ethernet core of a REMOTE chip. Sweeps every
// Tensix core's util_sampler ring over the local NOC, packs new samples into a
// journal, and PUSHES the journal over fabric into an idle-eth L1 landing spot
// on the MMIO chip. The host then reads that landing copy over plain PCIe and
// never touches the ethernet tunnel.
//
// WHY THIS EXISTS: a host read of a remote chip is read_non_mmio, which writes
// a command into an ethernet core's firmware queue and polls while holding
// UMD's NON_MMIO mutex. Under Llama-3.3-70B those cores do not service the
// command promptly and the host blocks for tens of seconds -- it stalls the
// workload it is supposed to be observing. See PLAN_ETH_AGGREGATOR.md 5c.
//
// PERSISTENCE: this kernel never returns, and that is the mechanism, not an
// oversight. Idle eth cores are always DISPATCH_MODE_HOST, and the idle-erisc
// firmware regains control only when the kernel returns. IDLE_ETH appears
// nowhere in impl/program/dispatch.cpp, so program dispatch cannot disturb us.
// What does end us is device init: assert_inactive_ethernet_cores() resets
// RiscType::ALL. The host detects that as a stalled sweep_count and re-attaches.
// See PLAN_ETH_AGGREGATOR.md 3.5.
//
// DO NOT DERIVE NOC COORDINATES HERE. Every destination and every ring base is
// computed host-side and passed in whole. NOC_X_PHYS_COORD() resolves against
// this kernel's own noc_index, which is not necessarily the NOC a given write
// travels on; an earlier attempt to rebuild an address in-kernel silently
// landed on the wrong tile, and an attempt to decompose a driver-supplied
// address hung the fabric router.
//
// Runtime args (before the appended fabric-connection args):
//    0: num_cores               Tensix cores to sweep. NEVER assume 64 -- WH ships
//                               both 64-core (2 rows harvested) and 80-core parts.
//    1: ring_addr_table         L1 addr of num_cores * 2 u32s: the host-computed
//                               64-bit NOC address of each Tensix core, lo/hi, with
//                               a ZERO local-address field. We OR MEM_UTIL_SAMPLER_BASE
//                               in below. The host cannot supply the full address
//                               because MEM_UTIL_SAMPLER_BASE is a firmware-only
//                               macro; OR-ing into a field the host left zeroed is
//                               safe, and is what cq_prefetch.cpp does. It is NOT
//                               the same as decomposing an address the driver gave
//                               us -- that hung the fabric router once already.
//    2: last_head_scratch       L1 addr of num_cores u32s, our per-core cursor
//    3: head_scratch            L1 addr of num_cores * 16 B. Sixteen, not four:
//                               a NOC read into L1 must be 16 B aligned at BOTH
//                               ends, and `head` sits at offset 8 of the ring
//                               header -- so we pull the aligned 16 B chunk that
//                               contains it and index into it locally.
//    4: stage_addr              L1 staging for outgoing journal entries
//    5: stage_entries_max       entries that fit in one fabric packet AND in stage
//    6: hdr_stage_addr          L1 staging for the 64 B journal header
//    7: dest_base_lo            landing journal base on the MMIO chip, lo
//    8: dest_base_hi            landing journal base on the MMIO chip, hi
//    9: capacity                entries in the landing journal ring
//   10: src_chip                fabric node id of this chip, stamped into the header
//   11: sweep_interval_cyc      idle cycles between sweeps
//   12: unicast_hops            fabric distance to the MMIO chip
//   13: seq_scratch              L1 addr of num_cores u32s. In L1 and not on the
//                               stack: MEM_IERISC_STACK_MIN_SIZE is 128 BYTES, so
//                               a local seq[] array silently smashes the stack.
//   14: dbg_addr                 L1 addr of 4 u32 liveness markers, written locally.
//                               A local write cannot fail, so the host can tell
//                               "kernel never started" from "started but the fabric
//                               connection never opened" from "running, sweeping".

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_stream_regs.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "core_config.h"
#include "util_aggregator.h"
#include "util_sampler.h"

using namespace tt::tt_fabric;

// The ring layout we sweep. Mirrors util_sampler.h, which is included above so
// the static_asserts there bind against the real firmware definition.
static constexpr uint32_t kRingSize = UTIL_SAMPLER_RING_SIZE;  // 62
static constexpr uint32_t kRingHeaderBytes = 32u;              // util_sampler_msg_t header
static constexpr uint32_t kRingHeadOffset = 8u;                // offsetof(util_sampler_msg_t, head)
static constexpr uint32_t kSampleBytes = 16u;                  // util_sampler_entry_t

// The NOC we read Tensix rings on. Pinned rather than inherited from
// noc_index: the ring addresses in the table are NOC0-encoded host-side, and a
// kernel launched with brisc_noc_id == 1 would read them on the wrong NOC and
// silently return whatever lives at the mirrored coordinates.
static constexpr uint8_t kSweepNoc = 0;

void kernel_main() {
    size_t arg_idx = 0;
    const uint32_t num_cores = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t ring_addr_table = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t last_head_scratch = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t head_scratch = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t stage_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t stage_entries_max = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t hdr_stage_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t dest_base_lo = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t dest_base_hi = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t capacity = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t src_chip = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t sweep_interval_cyc = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t unicast_hops = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t seq_scratch = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t dbg_addr = get_arg_val<uint32_t>(arg_idx++);

    const uint64_t dest_base = ((uint64_t)dest_base_hi << 32) | (uint64_t)dest_base_lo;

    volatile tt_l1_ptr uint32_t* dbg = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dbg_addr);
    dbg[0] = 0xA66E0000u;  // reached kernel_main
    dbg[1] = 0;            // sweeps completed
    dbg[2] = 0;            // entries pushed
    dbg[3] = num_cores;

    volatile tt_l1_ptr uint32_t* ring_tbl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ring_addr_table);
    volatile tt_l1_ptr uint32_t* last_head = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(last_head_scratch);
    volatile tt_l1_ptr uint32_t* seq = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(seq_scratch);
    volatile tt_l1_ptr util_agg_entry_t* stage = reinterpret_cast<volatile tt_l1_ptr util_agg_entry_t*>(stage_addr);
    volatile tt_l1_ptr util_agg_msg_t* hdr = reinterpret_cast<volatile tt_l1_ptr util_agg_msg_t*>(hdr_stage_addr);

    // `head` lives at offset 8 of the 16 B chunk we fetched for core i.
    auto ring_head = [&](uint32_t i) -> uint32_t {
        return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(head_scratch + i * 16u + kRingHeadOffset);
    };

    for (uint32_t i = 0; i < num_cores; i++) {
        last_head[i] = 0;
        seq[i] = 0;
    }

    hdr->magic = UTIL_AGG_MAGIC;
    hdr->version = UTIL_AGG_VERSION;
    hdr->head = 0;
    hdr->capacity = capacity;
    hdr->num_cores = num_cores;
    hdr->sweep_count = 0;
    hdr->lost = 0;
    hdr->src_chip = src_chip;

    auto sender = tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::IDLE_ETH>(arg_idx);
    sender.open();
    dbg[0] = 0x09E00000u;  // fabric connection opened

    auto* packet_header = PacketHeaderPool::allocate_header();

    // One unicast fabric write of `bytes` from `src_l1` to `dst`.
    auto push = [&](uint32_t src_l1, uint64_t dst, uint32_t bytes) {
        sender.wait_for_empty_write_slot();
        fabric_set_unicast_route<false>((LowLatencyPacketHeader*)packet_header, (uint8_t)unicast_hops);
        packet_header->to_noc_unicast_write(tt::tt_fabric::NocUnicastCommandHeader{dst}, bytes);
        sender.send_payload_without_header_non_blocking_from_address(src_l1, bytes);
        sender.send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
        noc_async_writes_flushed();
    };

    uint32_t head = 0;
    uint32_t lost = 0;
    uint32_t sweep_count = 0;
    uint32_t staged = 0;

    // Ship `staged` entries into the landing ring at `head`, splitting the write
    // when the ring wraps -- a single write across the end would run off the
    // journal and into whatever follows it in the receiver's L1.
    auto flush_stage = [&]() {
        if (staged == 0) {
            return;
        }
        noc_async_read_barrier(kSweepNoc);
        const uint32_t slot = head % capacity;
        const uint32_t fit = capacity - slot;
        const uint64_t at = dest_base + UTIL_AGG_JOURNAL_OFFSET + (uint64_t)slot * sizeof(util_agg_entry_t);
        if (staged <= fit) {
            push(stage_addr, at, staged * sizeof(util_agg_entry_t));
        } else {
            push(stage_addr, at, fit * sizeof(util_agg_entry_t));
            push(
                stage_addr + fit * sizeof(util_agg_entry_t),
                dest_base + UTIL_AGG_JOURNAL_OFFSET,
                (staged - fit) * sizeof(util_agg_entry_t));
        }
        head += staged;
        staged = 0;
    };

    // Publish the initial header so the host can tell "aggregator is up, no data
    // yet" from "nothing was ever launched here".
    hdr->hdr_checksum =
        util_agg_hdr_checksum(UTIL_AGG_MAGIC, UTIL_AGG_VERSION, head, capacity, num_cores, sweep_count, lost, src_chip);
    push(hdr_stage_addr, dest_base, sizeof(util_agg_msg_t));

    // First sweep only latches the rings' current heads. Everything already in
    // the rings predates us and has no reliable ordering against our journal.
    bool primed = false;

    while (true) {
        // Phase 1: fetch every ring's head in one burst. num_cores reads of 4 B
        // issued back to back, then a single barrier -- the round trips overlap
        // instead of serializing. At 64 cores this is ~64 * 4 B of NOC traffic
        // per sweep and never touches ethernet.
        for (uint32_t i = 0; i < num_cores; i++) {
            const uint64_t ring_base =
                (((uint64_t)ring_tbl[2 * i + 1] << 32) | (uint64_t)ring_tbl[2 * i]) | (uint64_t)MEM_UTIL_SAMPLER_BASE;
            // 16 B from the ring base, NOT 4 B from base+8. Both ends of a NOC
            // L1 read must be 16 B aligned (NOC_L1_READ_ALIGNMENT_BYTES == 16 on
            // WH and BH alike); base+8 is not, and neither is a 4 B-strided
            // destination. The chunk we pull covers magic/version/head/period.
            noc_async_read(ring_base, head_scratch + i * 16u, 16u, kSweepNoc);
        }
        noc_async_read_barrier(kSweepNoc);

        if (!primed) {
            for (uint32_t i = 0; i < num_cores; i++) {
                last_head[i] = ring_head(i);
            }
            primed = true;
        }

        // Phase 2: drain what advanced, one packet's worth at a time.
        for (uint32_t i = 0; i < num_cores; i++) {
            const uint32_t h = ring_head(i);
            uint32_t lh = last_head[i];
            if (h == lh) {
                continue;
            }
            // Unsigned subtraction is wrap-correct: the ring head is monotonic
            // and only wraps at 2^32, which at 1 kHz per core is ~50 days.
            uint32_t behind = h - lh;
            if (behind > kRingSize) {
                // The producer lapped us. Everything but the newest kRingSize
                // entries is already overwritten; count it and resync.
                lost += behind - kRingSize;
                lh = h - kRingSize;
                behind = kRingSize;
            }

            const uint64_t ring_base =
                (((uint64_t)ring_tbl[2 * i + 1] << 32) | (uint64_t)ring_tbl[2 * i]) | (uint64_t)MEM_UTIL_SAMPLER_BASE;
            for (uint32_t k = 0; k < behind; k++) {
                if (staged == stage_entries_max) {
                    // Staging (or the fabric packet) is full. Ship it and keep
                    // going; `lh` records exactly how far we got, so nothing is
                    // dropped and the next iteration resumes mid-core.
                    flush_stage();
                }
                const uint32_t slot = (lh + k) % kRingSize;
                // Read the 16 B sample DIRECTLY into the entry's first field.
                // That is why util_agg_entry_t puts `sample` at offset 0: a NOC
                // read into L1 must be 16 B aligned, and entry_base is 32 B
                // aligned by construction.
                noc_async_read(
                    ring_base + kRingHeaderBytes + (uint64_t)slot * kSampleBytes,
                    (uint32_t)&stage[staged].sample,
                    kSampleBytes,
                    kSweepNoc);
                stage[staged].core_id = i;
                stage[staged].seq = seq[i]++;
                staged++;
            }
            last_head[i] = h;
        }

        // Phase 3: ship the tail, then the header. Header LAST and on the same
        // connection, so a host that sees an advanced `head` is guaranteed the
        // entries behind it already landed.
        flush_stage();

        sweep_count++;
        dbg[1] = sweep_count;
        dbg[2] = head;
        hdr->head = head;
        hdr->sweep_count = sweep_count;
        hdr->lost = lost;
        hdr->hdr_checksum = util_agg_hdr_checksum(
            UTIL_AGG_MAGIC, UTIL_AGG_VERSION, head, capacity, num_cores, sweep_count, lost, src_chip);
        push(hdr_stage_addr, dest_base, sizeof(util_agg_msg_t));

        // IDLE, do not spin. A telemetry loop that keeps the core hot raises
        // AICLK, which changes both the power envelope and the thing being
        // measured. Duty cycle here is microseconds of work per sweep.
        if (sweep_interval_cyc) {
            riscv_wait(sweep_interval_cyc);
        }
    }
}
