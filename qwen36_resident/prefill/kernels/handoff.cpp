// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Prefill -> decode state handoff (BRISC, one core per chip). Walks this chip's job list: a job copies
// `count` pages of an interleaved DRAM tensor (the prefill's GDN states, KV caches) to chip dst_chip,
// either to consecutive pages of an interleaved tensor (kind 0) or to consecutive bytes of one DRAM bank
// (kind 1, the decode's bank-sharded KV cache). Pages for this chip are written over the NoC, the others
// over fabric (1D line, unicast; a fused increment of the destination's handoff semaphore per packet).
// The kernel ends once this chip has sent its pages and received `expected` packets.
//
// Job (16 words, 64 B: DRAM reads land 64 B aligned): dst_chip, src_base, src_page, page_bytes, kind, dst_base,
// dst_page (kind 0) / bank (kind 1), count, 8 unused. Runtime args: 0 jobs address (DRAM interleaved, 64 B pages), 1
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
            const uint32_t dst_idx = job[6], count = job[7];
            const InterleavedAddrGen<true> src{.bank_base_address = job[1], .page_size = page_bytes};
            const InterleavedAddrGen<true> dst{.bank_base_address = dst_base, .page_size = page_bytes};
            const uint32_t src_page = job[2];
            const uint32_t slots = page_bytes <= 1024 * 4 ? (kSlots * 4096) / page_bytes : 1;
            for (uint32_t i0 = 0; i0 < count; i0 += slots) {
                const uint32_t n = count - i0 < slots ? count - i0 : slots;
                for (uint32_t i = 0; i < n; i++) {
                    noc_async_read(src.get_noc_addr(src_page + i0 + i), buf + i * page_bytes, page_bytes);
                }
                noc_async_read_barrier();
                if (kind == 0) {
                    for (uint32_t i = 0; i < n; i++) {
                        send(dst_chip, buf + i * page_bytes, dst.get_noc_addr(dst_idx + i0 + i), page_bytes);
                    }
                } else {
                    // consecutive bytes of one bank: whole packets
                    const uint32_t bytes = n * page_bytes;
                    for (uint32_t o = 0; o < bytes; o += kPacket) {
                        const uint32_t len = bytes - o < kPacket ? bytes - o : kPacket;
                        send(
                            dst_chip,
                            buf + o,
                            get_noc_addr_from_bank_id<true>(dst_idx, dst_base + i0 * page_bytes + o),
                            len);
                    }
                }
                noc_async_write_barrier();
            }
        }
    }
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
    noc_semaphore_wait_min(sem, expected);
    noc_semaphore_set(sem, 0);
    if (fabric.is_logically_connected()) {
        fabric.close();
    }
}
