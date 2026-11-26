// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Prefill -> decode state handoff (BRISC, one core per chip). Walks this chip's job list: a job copies
// `count` pages of an interleaved DRAM tensor (the prefill's GDN states, KV caches) to chip dst_chip,
// either to consecutive pages of an interleaved tensor (kind 0) or to consecutive bytes of one DRAM bank
// (kind 1, the decode's bank-sharded KV cache). Pages for this chip are written over the NoC, the others
// over fabric (1D line, unicast; a fused increment of the destination's handoff semaphore per packet).
// The kernel ends once this chip has sent its pages and received `expected` packets (taken off the semaphore).
//
// Job (16 words, 64 B: DRAM reads land 64 B aligned): dst_chip, src_base, src_page, page_bytes, kind, dst_base,
// dst_page (kind 0) / bank (kind 1), count, transcode, 7 unused.
// Kind 2 (the reverse of kind 1, a decode slot's KV pages back into the prefill's scratch, Strata S1): the source is
// `count` x page_bytes consecutive bytes of DRAM bank src_page at src_base, the destination consecutive pages of an
// interleaved tensor (as kind 0). Job word 8, transcode (S0-9, V in Bfp2_b): 0 none; 1 (kind 1) the source pages are
// bfp4_b tiles (576 B), sent as Bfp2_b tiles (320 B) rounded to nearest: magnitude m (of m 2^(e - 129)) >= 2 becomes
// bfp2's 1 (2^(e - 127), the same exponent byte: a bfp4 row's max has m >= 4), smaller 0 — exact for values already on
// the Bfp2_b grid (m 0 or 4); 2 (kind 2) the source bytes are Bfp2_b tiles, written as bfp4_b pages (exact).
// Runtime args: 0 jobs address (DRAM interleaved, 64 B pages), 1
// jobs, 2 chip, 3 semaphore address (global semaphore), 4 expected packets, then FabricConnectionManager args (fwd,
// bwd).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"

constexpr uint32_t cb_pages = 0;
constexpr uint32_t cb_jobs = 1;
constexpr uint32_t kSlots = 8;      // pages in flight (the page buffer holds kSlots x 4 KB)
constexpr uint32_t kPacket = 4096;  // fabric payload per packet
constexpr uint32_t kJobBatch = 64;
constexpr uint32_t kBfp4Tile = 576, kBfp2Tile = 320;

// one tile: bfp4_b (64 exponent bytes, then per face row two words of 4-bit sign | magnitude, element j at nibble j)
// <-> Bfp2_b (64 exponent bytes, then per face row one word, element c at bits 2c: magnitude, 2c + 1: sign)
FORCE_INLINE void bfp4_to_bfp2(uint32_t in, uint32_t out) {
    volatile tt_l1_ptr uint32_t* a = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in);
    volatile tt_l1_ptr uint32_t* b = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out);
    for (uint32_t i = 0; i < 16; i++) {
        b[i] = a[i];
    }
    for (uint32_t fr = 0; fr < 64; fr++) {
        uint32_t o = 0;
        for (uint32_t h = 0; h < 2; h++) {
            const uint32_t w = a[16 + fr * 2 + h];
            for (uint32_t j = 0; j < 8; j++) {
                const uint32_t nib = (w >> (4 * j)) & 0xF;
                o |= ((nib & 7) >= 2 ? (1u | ((nib >> 3) << 1)) : 0u) << (2 * (8 * h + j));
            }
        }
        b[16 + fr] = o;
    }
}
FORCE_INLINE void bfp2_to_bfp4(uint32_t in, uint32_t out) {
    volatile tt_l1_ptr uint32_t* a = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in);
    volatile tt_l1_ptr uint32_t* b = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out);
    for (uint32_t i = 0; i < 16; i++) {
        b[i] = a[i];
    }
    for (uint32_t fr = 0; fr < 64; fr++) {
        const uint32_t w = a[16 + fr];
        for (uint32_t h = 0; h < 2; h++) {
            uint32_t o = 0;
            for (uint32_t j = 0; j < 8; j++) {
                const uint32_t two = (w >> (2 * (8 * h + j))) & 3;
                o |= ((two & 1) ? (4u | ((two >> 1) << 3)) : 0u) << (4 * j);
            }
            b[16 + fr * 2 + h] = o;
        }
    }
}

void kernel_main() {
    size_t arg_idx = 0;
    const uint32_t jobs_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t n_jobs = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t chip = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t sem_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t expected = get_arg_val<uint32_t>(arg_idx++);
    auto fabric =
        FabricConnectionManager::build_from_args<FabricConnectionManager::BUILD_AND_OPEN_CONNECTION_START_ONLY>(
            arg_idx);
    PacketHeaderPool::reset();
    volatile PACKET_HEADER_TYPE* hdr_fwd = PacketHeaderPool::allocate_header();
    volatile PACKET_HEADER_TYPE* hdr_bwd = PacketHeaderPool::allocate_header();
    if (fabric.is_logically_connected()) {
        fabric.open_finish();
    }
    const uint64_t sem_noc = get_noc_addr(my_x[0], my_y[0], sem_addr);

    const uint32_t buf = get_write_ptr(cb_pages);
    const uint32_t jobs_l1 = get_write_ptr(cb_jobs);
    const InterleavedAddrGen<true> jobs{.bank_base_address = jobs_addr, .page_size = 64};

    // send `bytes` at l1 to dst (a NoC address on chip dst_chip)
    auto send = [&](uint32_t dst_chip, uint32_t l1, uint64_t dst, uint32_t bytes) {
        if (dst_chip == chip) {
            noc_async_write(l1, dst, bytes);
            return;
        }
        const bool fwd = dst_chip > chip;
        volatile PACKET_HEADER_TYPE* h = fwd ? hdr_fwd : hdr_bwd;
        h->to_chip_unicast(static_cast<uint8_t>(fwd ? dst_chip - chip : chip - dst_chip));
        h->to_noc_fused_unicast_write_atomic_inc(
            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst, sem_noc, 1, true}, bytes);
        // blocking: the header is rewritten for the next packet
        perform_payload_send<true, true>(
            fwd ? fabric.get_forward_connection() : fabric.get_backward_connection(), l1, bytes, h);
    };

    for (uint32_t j0 = 0; j0 < n_jobs; j0 += kJobBatch) {
        const uint32_t nb = n_jobs - j0 < kJobBatch ? n_jobs - j0 : kJobBatch;
        for (uint32_t j = 0; j < nb; j++) {
            noc_async_read(jobs.get_noc_addr(j0 + j), jobs_l1 + j * 64, 64);
        }
        noc_async_read_barrier();
        for (uint32_t j = 0; j < nb; j++) {
            volatile tt_l1_ptr uint32_t* job = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(jobs_l1 + j * 64);
            const uint32_t dst_chip = job[0], page_bytes = job[3], kind = job[4], dst_base = job[5];
            const uint32_t dst_idx = job[6], count = job[7], xc = job[8];
            // (transcode: the source tiles, then the converted ones after them in the page buffer)
            const uint32_t out_bytes = xc == 1 ? kBfp2Tile : (xc == 2 ? kBfp4Tile : page_bytes);
            const InterleavedAddrGen<true> src{.bank_base_address = job[1], .page_size = page_bytes};
            const InterleavedAddrGen<true> dst{.bank_base_address = dst_base, .page_size = out_bytes};
            const uint32_t src_page = job[2];
            const uint32_t slots = xc ? (kSlots * 4096) / (page_bytes + out_bytes)
                                      : (page_bytes <= 1024 * 4 ? (kSlots * 4096) / page_bytes : 1);
            const uint32_t out = xc ? buf + slots * page_bytes : buf;
            for (uint32_t i0 = 0; i0 < count; i0 += slots) {
                const uint32_t n = count - i0 < slots ? count - i0 : slots;
                if (kind == 2) {
                    noc_async_read(
                        get_noc_addr_from_bank_id<true>(src_page, job[1] + i0 * page_bytes), buf, n * page_bytes);
                } else {
                    for (uint32_t i = 0; i < n; i++) {
                        noc_async_read(src.get_noc_addr(src_page + i0 + i), buf + i * page_bytes, page_bytes);
                    }
                }
                noc_async_read_barrier();
                if (xc) {
                    // (NOC-written: Blackhole's RISC-V L1 cache invalidated before the reads)
                    invalidate_l1_cache();
                    for (uint32_t i = 0; i < n; i++) {
                        if (xc == 1) {
                            bfp4_to_bfp2(buf + i * page_bytes, out + i * out_bytes);
                        } else {
                            bfp2_to_bfp4(buf + i * page_bytes, out + i * out_bytes);
                        }
                    }
                }
                if (kind == 0 || kind == 2) {
                    for (uint32_t i = 0; i < n; i++) {
                        send(dst_chip, out + i * out_bytes, dst.get_noc_addr(dst_idx + i0 + i), out_bytes);
                    }
                } else {
                    // consecutive bytes of one bank: whole packets
                    const uint32_t bytes = n * out_bytes;
                    for (uint32_t o = 0; o < bytes; o += kPacket) {
                        const uint32_t len = bytes - o < kPacket ? bytes - o : kPacket;
                        send(
                            dst_chip,
                            out + o,
                            get_noc_addr_from_bank_id<true>(dst_idx, dst_base + i0 * out_bytes + o),
                            len);
                    }
                }
                noc_async_write_barrier();
            }
        }
    }
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
    noc_semaphore_wait_min(sem, expected);
    // take this call's packets off the count atomically (not a reset to 0): in a pipelined pass (run_batch) another
    // chip may already run the next boundary's handoff, whose packets can land before this call ends
    if (expected) {
        noc_semaphore_inc(sem_noc, 0u - expected);
        noc_async_atomic_barrier();
    }
    if (fabric.is_logically_connected()) {
        fabric.close();
    }
}
