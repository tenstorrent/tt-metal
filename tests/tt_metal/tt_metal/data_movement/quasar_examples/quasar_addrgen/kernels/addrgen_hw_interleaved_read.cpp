// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Phase B, step B1 proof: walks an interleaved TensorAccessor with the real Quasar hardware address
// generator, cross-checks every generated address against the software TensorAccessor::get_noc_addr()
// (the path Phase A already proved correct), and, if requested, reads the page through the ordinary NoC
// V3 API using the *hardware* address. A page whose addresses disagree is never issued, so a bad walk
// reports mismatches instead of hanging the core on a transaction that can't complete.
//
// Compile-time args:
//   is_dram          - 1 if tensor::src is a DRAM interleaved tensor, 0 for L1
//   issue_real_reads - 1: also noc_async_read each page (at the HW address) into dest_addr
// Runtime args:
//   num_pages   - number of pages to walk, starting at page 0
//   dest_addr   - L1 address the pages land at, back to back (only with issue_real_reads)
//   report_addr - L1 address for the result summary (see ReportWord)

#include "api/dataflow/dataflow_api.h"
#include "api/debug/device_print.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"
#include "internal/tt-2xx/quasar/tensor/tensor_accessor_addrgen.h"

// Layout of the report region. Must match the host test.
enum ReportWord : uint32_t {
    kMismatches = 0,
    kFirstBadPage,
    kFirstBadHwLo,
    kFirstBadHwHi,
    kFirstBadSwLo,
    kFirstBadSwHi,
    kPagesIssued,
    kWalkable,  // echo of the host's walkable CTA: 0 means the recipe doesn't apply; nothing was popped or issued
    kNumBanks,
    kBankSelector0,                     // kMaxReportedBanks words: ATT selector of bank i
    kBankOffset0 = kBankSelector0 + 8,  // kMaxReportedBanks words: bank_to_{dram,l1}_offset[i]
    kNumReportWords = kBankOffset0 + 8,
};

void kernel_main() {
    // `report` below is written through the uncached alias: Quasar DM stores go through the data cache and
    // the host reads L1 directly, so a cached store may never be seen (same as runtime_args_kernel_2_0.cpp).
    constexpr bool is_dram = get_arg(args::is_dram) != 0;
    constexpr bool issue_real_reads = get_arg(args::issue_real_reads) != 0;
    // Host-checked (interleaved_banks_walkable in the test): whether the interleaved recipe applies on this device.
    constexpr bool walkable = get_arg(args::walkable) != 0;
    const uint32_t num_pages = get_arg(args::num_pages);
    const uint32_t dest_addr = get_arg(args::dest_addr);
    const uint32_t report_addr = get_arg(args::report_addr);

    const auto src = TensorAccessor(tensor::src);
    const uint32_t page_size = src.get_aligned_page_size();

    volatile tt_l1_ptr uint32_t* report =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(report_addr + MEM_L1_UNCACHED_BASE);

    constexpr uint32_t kMaxReportedBanks = 8;
    constexpr uint32_t num_banks = is_dram ? NUM_DRAM_BANKS : NUM_L1_BANKS;
    report[kNumBanks] = num_banks;
    for (uint32_t bank = 0; bank < num_banks; ++bank) {
        const uint32_t selector = tt_addrgen::interleaved_bank_selector<is_dram>(bank);
        const uint32_t offset = interleaved_addr_gen::get_bank_offset<is_dram>(bank);
        DEVICE_PRINT("addrgen bank {} -> ATT selector {} bank offset {}\n", bank, selector, offset);
        if (bank < kMaxReportedBanks) {
            report[kBankSelector0 + bank] = selector;
            report[kBankOffset0 + bank] = offset;
        }
    }
    report[kWalkable] = walkable ? 1u : 0u;
    if constexpr (!walkable) {
        DEVICE_PRINT("addrgen: bank selectors are not an ascending stride-1 run; recipe does not apply\n");
        return;
    }

    tt_addrgen::configure_addrgen_src_interleaved<is_dram>(src);

    uint32_t mismatches = 0;
    uint32_t pages_issued = 0;
    uint32_t first_bad_page = 0xFFFFFFFF;
    uint64_t first_bad_hw = 0;
    uint64_t first_bad_sw = 0;
    for (uint32_t page_id = 0; page_id < num_pages; ++page_id) {
        const uint64_t hw_addr = tt_addrgen::pop_src_noc_addr_interleaved<is_dram>();
        const uint64_t sw_addr = src.get_noc_addr(page_id);
        DEVICE_PRINT(
            "addrgen page {} hw 0x{:x} sw 0x{:x} match {}\n",
            page_id,
            hw_addr,
            sw_addr,
            static_cast<uint32_t>(hw_addr == sw_addr));
        if (hw_addr != sw_addr) {
            if (mismatches == 0) {
                first_bad_page = page_id;
                first_bad_hw = hw_addr;
                first_bad_sw = sw_addr;
            }
            ++mismatches;
            continue;
        }
        if constexpr (issue_real_reads) {
            noc_async_read(hw_addr, dest_addr + page_id * page_size, page_size);
            ++pages_issued;
        }
    }
    if constexpr (issue_real_reads) {
        noc_async_read_barrier();
    }

    report[kMismatches] = mismatches;
    report[kFirstBadPage] = first_bad_page;
    report[kFirstBadHwLo] = static_cast<uint32_t>(first_bad_hw);
    report[kFirstBadHwHi] = static_cast<uint32_t>(first_bad_hw >> 32);
    report[kFirstBadSwLo] = static_cast<uint32_t>(first_bad_sw);
    report[kFirstBadSwHi] = static_cast<uint32_t>(first_bad_sw >> 32);
    report[kPagesIssued] = pages_issued;
}
