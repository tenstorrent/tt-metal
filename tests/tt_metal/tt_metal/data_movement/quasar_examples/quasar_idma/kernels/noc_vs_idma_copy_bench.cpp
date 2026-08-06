// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Simple local TL1 copy benchmark for Quasar.
//
// The kernel measures two paths over the same source/destination buffers:
//   config_id=0: iDMA copy split across the 8 local iDMA engines.
//   config_id=1: NoC loopback copy through cmdbuf 0.
//
// Each result row reports the transfer size plus average issue and wait cycles.
// Results are written to the host-provided L1 buffer.

#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
#include "internal/tt-2xx/quasar/overlay/cmdbuff_api.hpp"
#include <cstdint>

using namespace overlay;

constexpr uint32_t kSweepSizes[] = {16384, 32768, 49152, 65536, 98304, 131072};
constexpr uint32_t kNumSweepSizes = sizeof(kSweepSizes) / sizeof(kSweepSizes[0]);
constexpr uint32_t kRepeats = 5;
constexpr uint32_t kNumIdmaEngines = 8;
constexpr uint32_t kResultWordsPerSample = 4;

FORCE_INLINE uint32_t rdcycle() {
    uint32_t cycles;
    asm volatile("rdcycle %0" : "=r"(cycles));
    return cycles;
}

// -------------------------------------------------------------------------------------------------
// iDMA local TL1 copy
// -------------------------------------------------------------------------------------------------

FORCE_INLINE void setup_idma() {
    reset_cmdbuf_0();
    idma_setup_as_copy_cmdbuf_0(false);
    setup_ongoing_cmdbuf_0(
        /*src_addr_inc_en=*/false,
        /*dest_addr_inc_en=*/false,
        /*trid_inc_en=*/false,
        /*req_vc_inc_en=*/true,
        /*resp_vc_inc_en=*/false);
    setup_wrapping_vcs_cmdbuf_0(
        /*wr=*/true,
        /*req_start_vc=*/0,
        /*req_end_vc=*/kNumIdmaEngines - 1);
    setup_trids_cmdbuf_0(CMDBUF_DEF_TRID);
}

FORCE_INLINE uint32_t issue_idma(uint32_t src_addr, uint32_t dst_addr, uint32_t total_bytes) {
    const uint32_t t0 = rdcycle();
    const uint32_t chunk_bytes = total_bytes / kNumIdmaEngines;
    uint32_t offset = 0;

    for (uint32_t engine = 0; engine < kNumIdmaEngines; ++engine) {
        set_src_cmdbuf_0(src_addr + offset);
        set_dest_cmdbuf_0(dst_addr + offset);
        set_len_cmdbuf_0(chunk_bytes);
        issue_cmdbuf_0();
        offset += chunk_bytes;
    }

    return rdcycle() - t0;
}

FORCE_INLINE uint32_t wait_idma() {
    const uint32_t t0 = rdcycle();

    while (!idma_acked_cmdbuf_0()) {
    }

    return rdcycle() - t0;
}

// -------------------------------------------------------------------------------------------------
// NoC loopback local TL1 copy
// -------------------------------------------------------------------------------------------------

FORCE_INLINE void setup_noc(uint32_t local_xy, uint64_t dst_noc_base) {
    // iDMA setup resets cmdbuf 0. Reinitialize the NoC write buffer so write acks route back here.
    init_wr_cmd_buf(local_xy);
    ncrisc_noc_write_set_state<false /*posted*/, false /*one_packet*/>(
        noc_index, write_cmd_buf, dst_noc_base, 0 /*len set per issue*/, NOC_UNICAST_WRITE_VC);
}

FORCE_INLINE uint32_t issue_noc(uint32_t src_addr, uint32_t dst_addr, uint32_t total_bytes) {
    const uint32_t t0 = rdcycle();
    ncrisc_noc_write_with_state<DM_DEDICATED_NOC, false /*posted*/, true /*update_counter*/, false /*one_packet*/>(
        noc_index, write_cmd_buf, src_addr, dst_addr, total_bytes);
    return rdcycle() - t0;
}

FORCE_INLINE uint32_t wait_noc() {
    const uint32_t t0 = rdcycle();

    while (!ncrisc_noc_nonposted_writes_flushed(noc_index)) {
    }

    return rdcycle() - t0;
}

// -------------------------------------------------------------------------------------------------

FORCE_INLINE void write_sample(
    volatile uint32_t* results,
    uint32_t sample_idx,
    uint32_t config_id,
    uint32_t total_bytes,
    uint32_t avg_issue_cycles,
    uint32_t avg_wait_cycles) {
    const uint32_t base = sample_idx * kResultWordsPerSample;

    results[base + 0] = config_id;
    results[base + 1] = total_bytes;
    results[base + 2] = avg_issue_cycles;
    results[base + 3] = avg_wait_cycles;
}

FORCE_INLINE uint32_t run_idma_samples(
    uint32_t src_addr, uint32_t dst_addr, uint32_t total_bytes, volatile uint32_t* results, uint32_t sample_idx) {
    setup_idma();
    uint32_t issue_cycles_sum = 0;
    uint32_t wait_cycles_sum = 0;

    for (uint32_t repeat = 0; repeat < kRepeats; ++repeat) {
        issue_cycles_sum += issue_idma(src_addr, dst_addr, total_bytes);
        wait_cycles_sum += wait_idma();
    }

    write_sample(results, sample_idx, 0, total_bytes, issue_cycles_sum / kRepeats, wait_cycles_sum / kRepeats);
    return sample_idx + 1;
}

FORCE_INLINE uint32_t run_noc_samples(
    uint32_t src_addr,
    uint32_t dst_addr,
    uint32_t total_bytes,
    uint32_t local_xy,
    volatile uint32_t* results,
    uint32_t sample_idx) {
    const uint64_t dst_noc_base = get_noc_addr_helper(local_xy, dst_addr);
    setup_noc(local_xy, dst_noc_base);
    uint32_t issue_cycles_sum = 0;
    uint32_t wait_cycles_sum = 0;

    for (uint32_t repeat = 0; repeat < kRepeats; ++repeat) {
        issue_cycles_sum += issue_noc(src_addr, dst_addr, total_bytes);
        wait_cycles_sum += wait_noc();
    }

    write_sample(results, sample_idx, 1, total_bytes, issue_cycles_sum / kRepeats, wait_cycles_sum / kRepeats);
    return sample_idx + 1;
}

void kernel_main() {
    constexpr uint32_t src_addr = get_arg(args::src_addr);
    constexpr uint32_t dst_addr = get_arg(args::dst_addr);
    constexpr uint32_t result_addr = get_arg(args::result_addr);
    constexpr uint32_t noc_x = get_arg(args::noc_x);
    constexpr uint32_t noc_y = get_arg(args::noc_y);
    constexpr uint32_t local_xy = NOC_XY_COORD(noc_x, noc_y);
    volatile uint32_t* results = reinterpret_cast<volatile uint32_t*>(result_addr + MEM_L1_UNCACHED_BASE);
    uint32_t sample_idx = 0;

    for (uint32_t i = 0; i < kNumSweepSizes; ++i) {
        const uint32_t total_bytes = kSweepSizes[i];

        sample_idx = run_idma_samples(src_addr, dst_addr, total_bytes, results, sample_idx);
        sample_idx = run_noc_samples(src_addr, dst_addr, total_bytes, local_xy, results, sample_idx);
    }

    asm volatile("fence rw, rw" ::: "memory");
}
