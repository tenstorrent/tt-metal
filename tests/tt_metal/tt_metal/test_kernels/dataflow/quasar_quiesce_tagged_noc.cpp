// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"
#include "internal/tt-2xx/quasar/overlay/overlay_addresses.h"

void kernel_main() {
    const uint32_t handshake_addr = get_arg(args::handshake_addr);
    const uint32_t src_stage_addr = get_arg(args::src_stage_addr);
    const uint32_t dst_noc_x = get_arg(args::dst_noc_x);
    const uint32_t dst_noc_y = get_arg(args::dst_noc_y);
    const uint32_t dst_l1_addr = get_arg(args::dst_l1_addr);

    constexpr uint32_t KERNEL_STARTED_MAGIC = 0xB007ED01U;
    constexpr uint32_t HOST_GATE_OPEN_MAGIC = 0x6A7E0001U;
    constexpr uint32_t PAYLOAD_BASE = 0x721D0000U;

    volatile tt_l1_ptr uint32_t* handshake =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(MEM_L1_UNCACHED_BASE + handshake_addr);

    const uint32_t tid = get_my_thread_id();
    if (tid == 0U) {
        handshake[0] = KERNEL_STARTED_MAGIC;
        asm volatile("fence" ::: "memory");
    }

    // Wait for host to observe KERNEL_STARTED_MAGIC, arm quiesce.control mid-kernel, and open the gate.
    while (handshake[1] != HOST_GATE_OPEN_MAGIC) {
        asm volatile("nop");
    }

    // Thread 0 issues tagged writes on TRIDs 1..4; Thread 1 issues tagged writes on TRIDs 5..7.
    // Each slot is 16B-aligned for NOC write alignment.
    const uint32_t first_trid = (tid == 0U) ? 1U : 5U;
    const uint32_t last_trid = (tid == 0U) ? 4U : 7U;

    for (uint32_t trid = first_trid; trid <= last_trid; ++trid) {
        const uint32_t slot_offset = (trid - 1U) * L1_ALIGNMENT;
        const uint32_t local_slot = src_stage_addr + slot_offset;
        volatile tt_l1_ptr uint32_t* stage_uncached =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(MEM_L1_UNCACHED_BASE + local_slot);
        stage_uncached[0] = PAYLOAD_BASE + trid;
        stage_uncached[1] = 0U;
        stage_uncached[2] = 0U;
        stage_uncached[3] = 0U;
        asm volatile("fence" ::: "memory");

        const uint64_t dst_noc_addr = get_noc_addr(dst_noc_x, dst_noc_y, dst_l1_addr + slot_offset);
        noc_async_write_one_packet_with_trid(local_slot, dst_noc_addr, L1_ALIGNMENT, trid);
    }

    // Deliberately omit a kernel-side NOC barrier here: firmware's subordinate user-TRID drain
    // (noc_drain_trid_range(0, 7)) and DM0's quiesce_and_park_worker_tile() must drain TRIDs 1..7.
    noc_async_write_set_trid(0U);
}
