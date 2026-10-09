// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Raw address-generator cost breakdown (AddrgenRawPerf): where the TensorAccessor sequencer's per-transfer cycles go,
// and what pushing straight into the command buffer would save. One interleaved tensor, num_pages pages, read in
// page order. Address generator 1's source side walks it with the ATT window bits folded into the outer loop's start
// (outer_start = window.compare), so every generated address is the complete NoC address and can be pushed as is.
// Timed with rdcycle, each section over num_pages transfers:
//   0 loop        - empty loop (subtracted on the host)
//   1 pop         - pop only
//   2 pop_issue   - pop, then noc_async_read() of that address (the software issue path), one barrier at the end
//   3 sw_issue    - TensorAccessor::get_noc_addr(), then noc_async_read(), one barrier at the end
//   4 push        - push only (count-less push builtin), nothing issued
//   5 push_issue  - DEST_ADDR + LEN write, push, issue, one barrier at the end
//   6 push_issue_barrier - section 5 with a barrier after every page
//   7 sw_issue_barrier   - section 3 with a barrier after every page
//   8 sequencer      - tensor_accessor::generated_noc_addr() (the sequencer as shipped), address only
// The push breakdown, a ladder from the bare push to the NoC API (one barrier at the end of each):
//   9 sequencer_push    - the sequencer with push allowed (generated_noc_addr<Read, MayPush>), nothing issued: on a hit
//   the
//                      address goes into the command buffer instead of back to the RISC-V (vs 8: the same with pop)
//  10 push_v3        - raw push, then the NoC V3 issue of a pushed address (ncrisc_noc_fast_read<src_in_cmd_buf>:
//                      VCs, DEST_ADDR, LEN, issue, counter every transfer) (vs 5: DEST_ADDR + LEN only)
//  11 sequencer_push_v3 - section 9's sequencer, then section 10's issue: the shipped push path below the Noc API
//  12 noc_api        - Noc::async_read(tensor, scratchpad, ...), the API kernels call (vs 11: the Noc / traits layer)
//  13 sequencer_pop_v3  - the sequencer without push, then noc_async_read() of the popped address (the path before
//  push) 14 push_x         - raw push with a skip count in a register, push_src_pop_x(generator, 0) -- the form the
//  sequencer
//                      uses (its stride is a run-time value) -- nothing issued (vs 4: the count-less push)
//  15 push_x_v3      - section 14's push, then section 10's issue
// The sequencer sections (8, 9, 11, 12, 13) are `flatten`: the sequencer is inlined into each loop, as in a kernel that
// calls it from one place. (Without it GCC outlined the push variant, which this kernel calls from four places, and
// those sections measured a function call.) The noinline slow paths stay calls.
// Checks (untimed): the folded walk's pops equal get_noc_addr() for every page; pages read through push (pop_x 0 and
// the count-less builtin) and through section 12 carry their own page id; how many of a sequential walk's requests
// the sequencer pushed.
//
// Runtime args: report_addr. Report: 16 section cycle counts (64-bit, low word first), then pop mismatches, push_x data
// mismatches, push (count-less) data mismatches, num_pages, the sink, noc_api data mismatches, sequencer pushes.

#include <cstdint>
#include <type_traits>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "internal/tt-2xx/quasar/overlay/addrgen_api.hpp"
#include "internal/tt-2xx/quasar/overlay/rocc_instructions.hpp"
#include "internal/tt-2xx/quasar/tensor/addrgen_sequencer.h"

namespace {

constexpr uint32_t kNumSections = 16;
#define RAW_PERF_KEEP(v) asm volatile("" : "+r"(v))

inline uint64_t cycles() {
    uint64_t c;
    asm volatile("rdcycle %0" : "=r"(c));
    return c;
}

constexpr uint32_t kRdCmdBuf = static_cast<uint32_t>(overlay::paired_cmdbuf(overlay::ADDRGEN_1));
static_assert(kRdCmdBuf == 1, "ADDRGEN_PUSH_SRC_POP_X in the checks hard-codes command buffer 1");

inline void cmdbuf_wr(uint32_t reg_offset, uint64_t value) {
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(kRdCmdBuf, reg_offset / 8, value);
}

// Program addrgen_1's source side to walk the tensor from page 0, emitting complete NoC addresses.
template <bool IsDram>
void program_walk(uint32_t bank_base_address, uint32_t page_size) {
    constexpr uint32_t num_banks = IsDram ? NUM_DRAM_BANKS : NUM_L1_BANKS;
    const noc_att::Window& window = tt_addrgen::interleaved_window<IsDram>();
    overlay::reset_addrgen<overlay::ADDRGEN_1>();
    overlay::setup_src_banking_addrgen<overlay::ADDRGEN_1>(overlay::BankingConfig{
        .endpoint_id_shift = window.endpoint_shift,
        .size = num_banks,
        .skip = 1,
        .base = tt_addrgen::interleaved_bank_selector<IsDram>(0),
        .current = 0,
        .bank_order = overlay::BANK_INNER,
    });
    overlay::setup_src_inner_loop_addrgen<overlay::ADDRGEN_1>(
        page_size, tt_addrgen::kInnerEndSentinel, bank_base_address);
    // Window bits folded into the outer loop: stride 0, never wraps (compare < kOuterEndSentinel, static_asserted).
    overlay::setup_src_outer_loop_addrgen<overlay::ADDRGEN_1>(0, tt_addrgen::kOuterEndSentinel, window.compare);
}

}  // namespace

void kernel_main() {
    const uint32_t report_addr = get_arg(args::report_addr);
    const auto ta = TensorAccessor(tensor::src0);
    constexpr bool kIsDram = std::decay_t<decltype(ta)>::DSpec::is_dram;
    Scratchpad<uint32_t> pad(scratch::pad);
    const uint32_t page_size = ta.get_aligned_page_size();
    const uint32_t num_pages = pad.size_in_bytes() / page_size;  // the scratchpad holds one slot per page
    const uint32_t pad_base = pad.get_base_address();
    const uint32_t bank_base = ta.get_bank_base_address();

    uint64_t elapsed[kNumSections];
    uint64_t sink = 0;
    auto time = [&](uint32_t s, auto&& body) {
        const uint64_t t0 = cycles();
        body();
        elapsed[s] = cycles() - t0;
    };

    // The push path issues on the read command buffer directly. SRC_BASE must be 0 (the window bits are in the walk),
    // and the VCs are set once; nothing else uses this command buffer during the push sections.
    overlay::setup_src_base_start_addrgen<overlay::ADDRGEN_1>(0);
    auto push_setup = [&] {
        cmdbuf_wr(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_REQ_VC_REG_OFFSET, NOC_OVERLAY_RD_REQ_VC);
        cmdbuf_wr(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_RESP_VC_REG_OFFSET, NOC_OVERLAY_RD_RESP_VC);
        cmdbuf_wr(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_LEN_BYTES_REG_OFFSET, page_size);
    };
    auto push_issue = [&](uint32_t i) __attribute__((always_inline)) {
        cmdbuf_wr(
            TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET, noc_v3_local_operand(pad_base + i * page_size));
        cmdbuf_wr(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_LEN_BYTES_REG_OFFSET, page_size);
        overlay::push_src_addrgen<overlay::ADDRGEN_1>();  // count-less push: advances by 1 (checked below)
        __builtin_riscv_ttrocc_cmdbuf_issue_trans(kRdCmdBuf);
        noc_reads_num_issued[noc_index] += 1;
    };

    time(0, [&] {
        for (uint32_t i = 0; i < num_pages; ++i) {
            sink += i;
            RAW_PERF_KEEP(sink);
        }
    });
    program_walk<kIsDram>(bank_base, page_size);
    time(1, [&] {
        for (uint32_t i = 0; i < num_pages; ++i) {
            sink += overlay::pop_src_addrgen<overlay::ADDRGEN_1>(1);
            RAW_PERF_KEEP(sink);
        }
    });
    program_walk<kIsDram>(bank_base, page_size);
    time(2, [&] {
        for (uint32_t i = 0; i < num_pages; ++i) {
            noc_async_read(overlay::pop_src_addrgen<overlay::ADDRGEN_1>(1), pad_base + i * page_size, page_size);
        }
        noc_async_read_barrier();
    });
    time(3, [&] {
        for (uint32_t i = 0; i < num_pages; ++i) {
            noc_async_read(ta.get_noc_addr(i), pad_base + i * page_size, page_size);
        }
        noc_async_read_barrier();
    });
    program_walk<kIsDram>(bank_base, page_size);
    time(4, [&] {
        for (uint32_t i = 0; i < num_pages; ++i) {
            overlay::push_src_addrgen<overlay::ADDRGEN_1>();
        }
    });
    program_walk<kIsDram>(bank_base, page_size);
    push_setup();
    time(5, [&] {
        for (uint32_t i = 0; i < num_pages; ++i) {
            push_issue(i);
        }
        noc_async_read_barrier();
    });
    program_walk<kIsDram>(bank_base, page_size);
    push_setup();
    time(6, [&] {
        for (uint32_t i = 0; i < num_pages; ++i) {
            push_issue(i);
            noc_async_read_barrier();
        }
    });
    time(7, [&] {
        for (uint32_t i = 0; i < num_pages; ++i) {
            noc_async_read(ta.get_noc_addr(i), pad_base + i * page_size, page_size);
            noc_async_read_barrier();
        }
    });
    // 10: a raw push, then the NoC V3 issue of a pushed address, which rewrites the VCs, DEST_ADDR and LEN every time.
    constexpr uint32_t kRdVc = NOC_UNICAST_WRITE_VC;  // the request VC Noc::async_read uses by default
    auto issue_pushed = [&](uint32_t i) __attribute__((always_inline)) {
        ncrisc_noc_fast_read<noc_mode, /*src_in_cmd_buf=*/true>(
            noc_index, read_cmd_buf, 0, pad_base + i * page_size, page_size, kRdVc);
    };
    program_walk<kIsDram>(bank_base, page_size);
    time(10, [&] {
        for (uint32_t i = 0; i < num_pages; ++i) {
            overlay::push_src_addrgen<overlay::ADDRGEN_1>();
            issue_pushed(i);
        }
        noc_async_read_barrier();
    });
    // 14, 15: the sequencer's push form. The skip count is a run-time value, as the sequencer's stride is.
    uint32_t skip = 0;
    RAW_PERF_KEEP(skip);
    program_walk<kIsDram>(bank_base, page_size);
    time(14, [&] {
        for (uint32_t i = 0; i < num_pages; ++i) {
            __builtin_riscv_ttrocc_addrgen_push_src_pop_x(overlay::ADDRGEN_1, skip);
        }
    });
    program_walk<kIsDram>(bank_base, page_size);
    time(15, [&] {
        for (uint32_t i = 0; i < num_pages; ++i) {
            __builtin_riscv_ttrocc_addrgen_push_src_pop_x(overlay::ADDRGEN_1, skip);
            issue_pushed(i);
        }
        noc_async_read_barrier();
    });

    // Checks. Each page's first word is its page id (written by the host).
    volatile tt_l1_ptr uint32_t* pad_words =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pad_base + MEM_L1_UNCACHED_BASE);
    auto clear_pad = [&] {
        for (uint32_t i = 0; i < num_pages; ++i) {
            pad_words[i * page_size / sizeof(uint32_t)] = 0xFFFFFFFFu;
        }
    };
    auto data_mismatches = [&] {
        uint32_t bad = 0;
        for (uint32_t i = 0; i < num_pages; ++i) {
            bad += pad_words[i * page_size / sizeof(uint32_t)] != i;
        }
        return bad;
    };
    uint32_t pop_mismatches = 0;
    program_walk<kIsDram>(bank_base, page_size);
    for (uint32_t i = 0; i < num_pages; ++i) {
        pop_mismatches += overlay::pop_src_addrgen<overlay::ADDRGEN_1>(1) != ta.get_noc_addr(i);
    }
    // Push with a skip count of 0: the skip counts addresses *beyond* the push's own advance (a skip of 1 read every
    // other page), unlike pop_x, whose count is the total advance.
    clear_pad();
    program_walk<kIsDram>(bank_base, page_size);
    push_setup();
    for (uint32_t i = 0; i < num_pages; ++i) {
        cmdbuf_wr(
            TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET, noc_v3_local_operand(pad_base + i * page_size));
        ADDRGEN_PUSH_SRC_POP_X(1, 0);  // command buffer 1 as an immediate (the macro encodes it in the opcode)
        __builtin_riscv_ttrocc_cmdbuf_issue_trans(kRdCmdBuf);
        noc_reads_num_issued[noc_index] += 1;
    }
    noc_async_read_barrier();
    const uint32_t push_x_mismatches = data_mismatches();
    clear_pad();
    program_walk<kIsDram>(bank_base, page_size);
    push_setup();
    for (uint32_t i = 0; i < num_pages; ++i) {
        cmdbuf_wr(
            TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET, noc_v3_local_operand(pad_base + i * page_size));
        overlay::push_src_addrgen<overlay::ADDRGEN_1>();
        __builtin_riscv_ttrocc_cmdbuf_issue_trans(kRdCmdBuf);
        noc_reads_num_issued[noc_index] += 1;
    }
    noc_async_read_barrier();
    const uint32_t push_mismatches = data_mismatches();
    overlay::reset_addrgen<overlay::ADDRGEN_1>();

    // 8: the sequencer as shipped (it resets and programs the generators it uses itself).
    time(8, [&]() __attribute__((flatten)) {
        for (uint32_t i = 0; i < num_pages; ++i) {
            sink += tensor_accessor::generated_noc_addr<tensor_accessor::TransferDir::Read>(ta, i, 0, noc_index);
            RAW_PERF_KEEP(sink);
        }
    });

    // The rest of the push ladder (10 ran above, before the sequencer owned generator 1). Each sequencer section starts
    // over at page 0, so the sequencer re-seeks once per section (as in 8).
    time(9, [&]() __attribute__((flatten)) {
        for (uint32_t i = 0; i < num_pages; ++i) {
            sink += tensor_accessor::generated_noc_addr<tensor_accessor::TransferDir::Read, true>(ta, i, 0, noc_index);
            RAW_PERF_KEEP(sink);
        }
    });
    time(11, [&]() __attribute__((flatten)) {
        for (uint32_t i = 0; i < num_pages; ++i) {
            const uint64_t addr =
                tensor_accessor::generated_noc_addr<tensor_accessor::TransferDir::Read, true>(ta, i, 0, noc_index);
            if (addr == tt_addrgen::kAddrPushed) {
                issue_pushed(i);
            } else {
                noc_async_read(addr, pad_base + i * page_size, page_size, noc_index, kRdVc);
            }
        }
        noc_async_read_barrier();
    });
    clear_pad();
    Noc noc;
    time(12, [&]() __attribute__((flatten)) {
        for (uint32_t i = 0; i < num_pages; ++i) {
            noc.async_read(ta, pad, page_size, {.page_id = i}, {.offset_bytes = i * page_size});
        }
        noc.async_read_barrier();
    });
    const uint32_t noc_api_mismatches = data_mismatches();
    time(13, [&]() __attribute__((flatten)) {
        for (uint32_t i = 0; i < num_pages; ++i) {
            noc_async_read(
                tensor_accessor::generated_noc_addr<tensor_accessor::TransferDir::Read>(ta, i, 0, noc_index),
                pad_base + i * page_size,
                page_size,
                noc_index,
                kRdVc);
        }
        noc_async_read_barrier();
    });
    // Untimed: how many of a sequential walk's requests the sequencer pushed (all of them, its seek included).
    uint32_t sequencer_pushes = 0;
    for (uint32_t i = 0; i < num_pages; ++i) {
        sequencer_pushes += tensor_accessor::generated_noc_addr<tensor_accessor::TransferDir::Read, true>(
                                ta, i, 0, noc_index) == tt_addrgen::kAddrPushed;
    }

    volatile tt_l1_ptr uint32_t* report =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(report_addr + MEM_L1_UNCACHED_BASE);
    for (uint32_t s = 0; s < kNumSections; ++s) {
        report[2 * s] = static_cast<uint32_t>(elapsed[s]);
        report[2 * s + 1] = static_cast<uint32_t>(elapsed[s] >> 32);
    }
    report[2 * kNumSections + 0] = pop_mismatches;
    report[2 * kNumSections + 1] = push_x_mismatches;
    report[2 * kNumSections + 2] = push_mismatches;
    report[2 * kNumSections + 3] = num_pages;
    report[2 * kNumSections + 4] = static_cast<uint32_t>(sink);
    report[2 * kNumSections + 5] = noc_api_mismatches;
    report[2 * kNumSections + 6] = sequencer_pushes;
}
