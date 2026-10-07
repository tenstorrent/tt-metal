// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Sharded transfer-address microbenchmark (TensorAccessorAddrgenShardedPerf): reads every page of one sharded tensor
// into a one-page scratchpad, addressed one of four ways, timed with rdcycle. Built with the address-generator path,
// with the software path (TT_TA_ADDRGEN_DISABLE), or with TT_TA_ADDRGEN_STATS for the walker's counts.
//   mode 0  page-id loop: noc.async_read(ta, ..., {.page_id})
//   mode 1  pages(): the page iterator (it computes each page's software address as it goes)
//   mode 2  shard_pages(): shard by shard, in storage order
//   mode 3  ShardView: every page slot of every shard as the shard's base + an offset
// Sections, each over all the mode's transfers:
//   0 loop      - the iteration alone (subtracted on the host)
//   1 addr      - the transfer address only (tensor_accessor::transfer_*noc_addr), no NoC
//   2 read+bar  - Noc::async_read of each page, a barrier after every read
//   3 batched   - the same reads, one barrier at the end
//
// Compile-time args: mode.
// Runtime args: report_addr -- 18 words: 4 section cycle counts (64-bit, low word first), transfers per section, the
// sink, then {hw, sw_ineligible, sw_unsupported, seeks, skips, restores, fallbacks, pushes} (TT_TA_ADDRGEN_STATS).

#include <cstdint>
#include <type_traits>

#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

namespace {

#define TA_PERF_KEEP(v) asm volatile("" : "+r"(v))

inline uint64_t cycles() {
    uint64_t c;
    asm volatile("rdcycle %0" : "=r"(c));
    return c;
}

}  // namespace

void kernel_main() {
    constexpr uint32_t mode = get_arg(args::mode);
    const uint32_t report_addr = get_arg(args::report_addr);
    Noc noc;
    Scratchpad<uint32_t> pad(scratch::pad);
    const auto ta = TensorAccessor(tensor::src);
    using TA = std::decay_t<decltype(ta)>;
    static_assert(!TA::DSpec::is_interleaved, "the sharded benchmark needs a sharded tensor");
    const uint32_t page_bytes = ta.get_aligned_page_size();
    const uint32_t num_pages = ta.dspec().tensor_volume();
    const uint32_t num_shards = ta.dspec().num_shards();
    const uint32_t shard_volume = ta.dspec().shard_volume();
    using PageArgs = noc_traits_t<tensor_accessor::Page>::src_args_type;

    // Calls f(endpoint, src_args) for every transfer of the mode, in order.
    auto for_each = [&](auto&& f) __attribute__((always_inline)) {
        if constexpr (mode == 0) {
            for (uint32_t id = 0; id < num_pages; ++id) {
                f(ta, typename noc_traits_t<TA>::src_args_type{.page_id = id});
            }
        } else if constexpr (mode == 1) {
            for (const auto& page : ta.pages()) {
                f(page, PageArgs{});
            }
        } else if constexpr (mode == 2) {
            for (uint32_t shard = 0; shard < num_shards; ++shard) {
                for (const auto& page : ta.shard_pages(shard)) {
                    f(page, PageArgs{});
                }
            }
        } else {
            const ShardView view(ta);
            for (uint32_t shard = 0; shard < num_shards; ++shard) {
                for (uint32_t p = 0; p < shard_volume; ++p) {
                    f(view,
                      typename noc_traits_t<ShardView<TA>>::src_args_type{
                          .shard_id = shard, .offset_bytes = p * page_bytes});
                }
            }
        }
    };
    // The transfer address alone, as the NoC traits ask for it.
    auto address = [&](const auto& endpoint, const auto& args) __attribute__((always_inline)) -> uint64_t {
        using E = std::decay_t<decltype(endpoint)>;
        using tensor_accessor::TransferDir;
        if constexpr (std::is_same_v<E, TA>) {
            return tensor_accessor::transfer_noc_addr<TransferDir::Read>(ta, args.page_id, 0, noc_index);
        } else if constexpr (std::is_same_v<E, ShardView<TA>>) {
            return tensor_accessor::transfer_shard_noc_addr<TransferDir::Read>(
                ta, args.shard_id, args.offset_bytes, noc_index);
        } else {
            return tensor_accessor::transfer_noc_addr<TransferDir::Read>(endpoint, 0, noc_index);
        }
    };

    uint64_t elapsed[4];
    uint64_t sink = 0;
    uint32_t transfers = 0;
    {
        const uint64_t t0 = cycles();
        for_each([&](const auto&, const auto&) {
            ++transfers;
            sink += transfers;
            TA_PERF_KEEP(sink);
        });
        elapsed[0] = cycles() - t0;
    }
    {
        const uint64_t t0 = cycles();
        for_each([&](const auto& endpoint, const auto& args) {
            sink += address(endpoint, args);
            TA_PERF_KEEP(sink);
        });
        elapsed[1] = cycles() - t0;
    }
    {
        const uint64_t t0 = cycles();
        for_each([&](const auto& endpoint, const auto& args) {
            noc.async_read(endpoint, pad, page_bytes, args, {.offset_bytes = 0});
            noc.async_read_barrier();
        });
        elapsed[2] = cycles() - t0;
    }
    {
        const uint64_t t0 = cycles();
        for_each([&](const auto& endpoint, const auto& args) {
            noc.async_read(endpoint, pad, page_bytes, args, {.offset_bytes = 0});
        });
        noc.async_read_barrier();
        elapsed[3] = cycles() - t0;
    }

    volatile tt_l1_ptr uint32_t* report =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(report_addr + MEM_L1_UNCACHED_BASE);
    for (uint32_t s = 0; s < 4; ++s) {
        report[2 * s] = static_cast<uint32_t>(elapsed[s]);
        report[2 * s + 1] = static_cast<uint32_t>(elapsed[s] >> 32);
    }
    report[8] = transfers;
    report[9] = static_cast<uint32_t>(sink ^ (sink >> 32));
#if defined(TT_TA_ADDRGEN_STATS)
    const auto& stats = tensor_accessor::detail::transfer_stats;
    report[10] = stats.hw;
    report[11] = stats.sw_ineligible;
    report[12] = stats.sw_unsupported;
    report[13] = stats.seeks;
    report[14] = stats.skips;
    report[15] = stats.restores;
    report[16] = stats.fallbacks;
    report[17] = stats.pushes;
#endif
}
