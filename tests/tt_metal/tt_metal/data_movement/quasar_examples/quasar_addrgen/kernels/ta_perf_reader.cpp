// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// TensorAccessor transfer-address microbenchmark (TensorAccessorAddrgenPerf). Walks num_tensors bound tensors in one
// page-id order, round-robin per page, and times four sections with rdcycle (this DM hart's own cycle counter):
//   0 ids     - generate the page ids only (the pattern's own cost, subtracted from the others on the host)
//   1 addr    - tensor_accessor::transfer_noc_addr() per page, no NoC transfer
//   2 read    - Noc::async_read() of each page into a scratchpad, with a barrier after every read (the usual ttnn
//   shape) 3 batched - the same reads, one barrier at the end
// Built with and without TT_TA_ADDRGEN_DISABLE, the same kernel measures the address-generator path against the
// software path. Page ids follow access patterns taken from ttnn kernels:
//   SEQ         - 0, 1, 2, ...                               (unary readers, most writers)
//   BLOCKED     - runs of `run` pages along a row, jumping a row of `width` pages between them, block by block
//                 (matmul in0 / in1 blocked readers)
//   STRIDE_BACK - down each column of a `width`-wide grid, then back up to the next column (transpose WH reader)
//   RANDOM      - an LCG over the pages (embedding, grid sample)
//
// Compile-time args: pattern, num_tensors (1..5), num_pages (per tensor, power of 2), width, run, num_transfers
// (page ids per tensor per section).
// Runtime args: report_addr -- kReportWords 32-bit words: 4 section cycle counts (64-bit, low word first), transfers
// per section, the sink, then the TransferStats counters when TT_TA_ADDRGEN_STATS is defined.

#include <cstdint>

#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

namespace {

enum Pattern : uint32_t { SEQ = 0, BLOCKED = 1, STRIDE_BACK = 2, RANDOM = 3 };

constexpr uint32_t kPattern = get_arg(args::pattern);
constexpr uint32_t kNumTensors = get_arg(args::num_tensors);
constexpr uint32_t kNumPages = get_arg(args::num_pages);
constexpr uint32_t kWidth = get_arg(args::width);
constexpr uint32_t kRun = get_arg(args::run);
constexpr uint32_t kNumTransfers = get_arg(args::num_transfers);
constexpr uint32_t kRows = kNumPages / kWidth;
static_assert(kNumTensors >= 1 && kNumTensors <= 5);
static_assert((kNumPages & (kNumPages - 1)) == 0, "num_pages must be a power of 2");
static_assert(kRows * kWidth == kNumPages && kWidth % kRun == 0);

// Keeps the compiler from folding a section's loop (the page ids are compile-time computable): `v` must be live in
// a register every iteration.
#define TA_PERF_KEEP(v) asm volatile("" : "+r"(v))

inline uint64_t cycles() {
    uint64_t c;
    asm volatile("rdcycle %0" : "=r"(c));
    return c;
}

// Page id of step i. Stateless except RANDOM, whose LCG state the caller threads through.
inline uint32_t page_id(uint32_t i, uint32_t& lcg) {
    if constexpr (kPattern == SEQ) {
        return i & (kNumPages - 1);
    } else if constexpr (kPattern == BLOCKED) {
        const uint32_t c = i % kRun;
        const uint32_t r = (i / kRun) % kRows;
        const uint32_t block = (i / (kRun * kRows)) % (kWidth / kRun);
        return r * kWidth + block * kRun + c;
    } else if constexpr (kPattern == STRIDE_BACK) {
        const uint32_t h = i % kRows;
        const uint32_t w = (i / kRows) % kWidth;
        return h * kWidth + w;
    } else {
        lcg = lcg * 1664525u + 1013904223u;
        return (lcg >> 8) & (kNumPages - 1);
    }
}

}  // namespace

void kernel_main() {
    const uint32_t report_addr = get_arg(args::report_addr);
    Noc noc;
    Scratchpad<uint32_t> pad(scratch::pad);
    const uint32_t page_bytes = pad.size_in_bytes();
    const auto src0 = TensorAccessor(tensor::src0);
    const auto src1 = TensorAccessor(tensor::src1);
    const auto src2 = TensorAccessor(tensor::src2);
    const auto src3 = TensorAccessor(tensor::src3);
    const auto src4 = TensorAccessor(tensor::src4);
    // Run f(accessor) for tensor t. Every tensor is its own accessor type (its binding id is part of the type).
    auto with_tensor = [&](uint32_t t, auto&& f) __attribute__((always_inline)) {
        switch (t) {
            case 0: f(src0); break;
            case 1: f(src1); break;
            case 2: f(src2); break;
            case 3: f(src3); break;
            default: f(src4); break;
        }
    };

    uint64_t elapsed[4];
    uint64_t sink = 0;

    // 0: page ids only.
    {
        uint32_t lcg = 1;
        const uint64_t t0 = cycles();
        for (uint32_t i = 0; i < kNumTransfers; ++i) {
            const uint32_t id = page_id(i, lcg);
            for (uint32_t t = 0; t < kNumTensors; ++t) {
                sink += id + t;
                TA_PERF_KEEP(sink);
            }
        }
        elapsed[0] = cycles() - t0;
    }
    // 1: transfer address only.
    {
        uint32_t lcg = 1;
        const uint64_t t0 = cycles();
        for (uint32_t i = 0; i < kNumTransfers; ++i) {
            const uint32_t id = page_id(i, lcg);
            for (uint32_t t = 0; t < kNumTensors; ++t) {
                with_tensor(t, [&](const auto& ta) {
                    sink +=
                        tensor_accessor::transfer_noc_addr<tensor_accessor::TransferDir::Read>(ta, id, 0, noc_index);
                    TA_PERF_KEEP(sink);
                });
            }
        }
        elapsed[1] = cycles() - t0;
    }
    // 2: read with a barrier after every page.
    {
        uint32_t lcg = 1;
        const uint64_t t0 = cycles();
        for (uint32_t i = 0; i < kNumTransfers; ++i) {
            const uint32_t id = page_id(i, lcg);
            for (uint32_t t = 0; t < kNumTensors; ++t) {
                with_tensor(t, [&](const auto& ta) {
                    noc.async_read(ta, pad, page_bytes, {.page_id = id}, {.offset_bytes = 0});
                    noc.async_read_barrier();
                });
            }
        }
        elapsed[2] = cycles() - t0;
    }
    // 3: the same reads, one barrier at the end.
    {
        uint32_t lcg = 1;
        const uint64_t t0 = cycles();
        for (uint32_t i = 0; i < kNumTransfers; ++i) {
            const uint32_t id = page_id(i, lcg);
            for (uint32_t t = 0; t < kNumTensors; ++t) {
                with_tensor(t, [&](const auto& ta) {
                    noc.async_read(ta, pad, page_bytes, {.page_id = id}, {.offset_bytes = 0});
                });
            }
        }
        noc.async_read_barrier();
        elapsed[3] = cycles() - t0;
    }

    // Quasar DM stores go through the data cache; report through the uncached alias so the host sees it.
    volatile tt_l1_ptr uint32_t* report =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(report_addr + MEM_L1_UNCACHED_BASE);
    for (uint32_t s = 0; s < 4; ++s) {
        report[2 * s] = static_cast<uint32_t>(elapsed[s]);
        report[2 * s + 1] = static_cast<uint32_t>(elapsed[s] >> 32);
    }
    report[8] = kNumTransfers * kNumTensors;
    report[9] = static_cast<uint32_t>(sink ^ (sink >> 32));
#if defined(TT_TA_ADDRGEN_STATS)
    report[10] = tensor_accessor::detail::transfer_stats.hw;
    report[11] = tensor_accessor::detail::transfer_stats.sw_ineligible;
    report[12] = tensor_accessor::detail::transfer_stats.sw_unsupported;
    report[13] = tensor_accessor::detail::transfer_stats.seeks;
    report[14] = tensor_accessor::detail::transfer_stats.skips;
    report[15] = tensor_accessor::detail::transfer_stats.restores;
    report[16] = tensor_accessor::detail::transfer_stats.fallbacks;
    report[17] = tensor_accessor::detail::transfer_stats.reload_save_cycles;
    report[18] = tensor_accessor::detail::transfer_stats.reload_swap_cycles;
    report[19] = tensor_accessor::detail::transfer_stats.reload_restore_cycles;
    report[20] = tensor_accessor::detail::transfer_stats.reload_serve_cycles;
    report[21] = tensor_accessor::detail::transfer_stats.pushes;
#endif
}
