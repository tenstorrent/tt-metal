// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// TensorAccessor API coverage (TensorAccessorAddrgenApi): copies tensor src to tensor dst through a scratchpad that
// holds one slot per page, using one NoC/TensorAccessor API per mode, so each API's transfer address goes through
// tensor_accessor::transfer_noc_addr (on Quasar, the address-generator walker). With several threads, each thread
// copies the pages it owns.
//   mode 0  pages() with no range (sharded: the whole tensor) / pages(0, num_pages) (interleaved)
//   mode 1  shard_pages(shard, start, end) in two halves per shard (sharded only)
//   mode 2  async_read / async_write<NocOptions::TXN_ID> to and from L1 (the generic paths, not the DFB overloads)
//   mode 3  inline_dw_write of one word per dst page (no read; the host checks the first word of every page)
//   mode 4  strided_pages(): thread t of T copies pages t, t+T, ... (multi-threaded)
//   mode 5  strided_shard_pages(): thread t of T copies shards t, t+T, ... (multi-threaded, sharded only)
// Not covered here: the stateful set_*_state / *_with_state APIs (not supported on the address-generator path), and
// async_write_zeros (DRAM pages are addressed in software; local L1 is zeroed by the iDMA zero device).
//
// Compile-time args: mode, num_pages.
// Runtime args: report_addr -- per thread (at + thread * 64 bytes), 16 words: {hw, sw_ineligible, sw_unsupported,
// seeks, address requests, skips, restores, write seeks, write restores, 0, fallbacks, write fallbacks, pushes,
// done marker, 0, 0} (TT_TA_ADDRGEN_STATS builds; see api/tensor/transfer_noc_addr.h).

#include <type_traits>

#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

namespace {
constexpr uint32_t kReportStride = 64;
constexpr uint32_t kDoneMarker = 0x600DD00Du;
}  // namespace

void kernel_main() {
    constexpr uint32_t mode = get_arg(args::mode);
    constexpr uint32_t num_pages = get_arg(args::num_pages);
    const uint32_t report_addr = get_arg(args::report_addr);

    Noc noc;
    Scratchpad<uint32_t> pad(scratch::pad);
    const auto src_ta = TensorAccessor(tensor::src);
    const auto dst_ta = TensorAccessor(tensor::dst);
    const uint32_t page_size = src_ta.get_aligned_page_size();

    uint32_t requests = 0;  // address requests that went through transfer_noc_addr
    // An iterator page: its own args types (empty offset), its page id picks the scratchpad slot.
    auto read = [&](const auto& page) {
        using Traits = noc_traits_t<std::decay_t<decltype(page)>>;
        noc.async_read(
            page, pad, page_size, typename Traits::src_args_type{}, {.offset_bytes = page.page_id() * page_size});
        ++requests;
    };
    auto write = [&](const auto& page) {
        using Traits = noc_traits_t<std::decay_t<decltype(page)>>;
        noc.async_write(
            pad, page, page_size, {.offset_bytes = page.page_id() * page_size}, typename Traits::dst_args_type{});
        ++requests;
    };

    // Generic lambda: the sharded-only branches compile only for a sharded accessor (a discarded `if constexpr`
    // branch is only skipped inside a template).
    auto run = [&](const auto& src, const auto& dst) {
        using TA = std::decay_t<decltype(src)>;
        constexpr bool kSharded = !TA::DSpec::is_interleaved;
        if constexpr (mode == 0) {
            if constexpr (kSharded) {
                for (const auto& page : src.pages()) {
                    read(page);
                }
                noc.async_read_barrier();
                for (const auto& page : dst.pages()) {
                    write(page);
                }
            } else {
                for (const auto& page : src.pages(0, num_pages)) {
                    read(page);
                }
                noc.async_read_barrier();
                for (const auto& page : dst.pages(0, num_pages)) {
                    write(page);
                }
            }
        } else if constexpr (mode == 1) {
            static_assert(kSharded, "shard_pages() needs a sharded tensor");
            const uint32_t volume = src.dspec().shard_volume();
            const uint32_t half = volume / 2;
            for (uint32_t shard = 0; shard < src.dspec().num_shards(); ++shard) {
                for (const auto& [start, end] : {std::pair{0u, half}, std::pair{half, volume}}) {
                    if (start == end) {
                        continue;
                    }
                    for (const auto& page : src.shard_pages(shard, start, end)) {
                        read(page);
                    }
                }
            }
            noc.async_read_barrier();
            for (uint32_t shard = 0; shard < dst.dspec().num_shards(); ++shard) {
                for (const auto& [start, end] : {std::pair{0u, half}, std::pair{half, volume}}) {
                    if (start == end) {
                        continue;
                    }
                    for (const auto& page : dst.shard_pages(shard, start, end)) {
                        write(page);
                    }
                }
            }
        } else if constexpr (mode == 2) {
            for (uint32_t p = 0; p < num_pages; ++p) {
                noc.async_read<NocOptions::TXN_ID>(
                    src, pad, page_size, {.page_id = p}, {.offset_bytes = p * page_size}, {.trid = 1});
                ++requests;
            }
            noc.async_read_barrier();
            for (uint32_t p = 0; p < num_pages; ++p) {
                noc.async_write<NocOptions::TXN_ID>(
                    pad, dst, page_size, {.offset_bytes = p * page_size}, {.page_id = p}, {.trid = 1});
                ++requests;
            }
        } else if constexpr (mode == 3) {
            for (uint32_t p = 0; p < num_pages; ++p) {
                noc.inline_dw_write<NocOptions::INLINE_L1>(dst, 0xD0D00000u | p, {.page_id = p});
                ++requests;
            }
        } else if constexpr (mode == 4) {
            auto src_pages = [&] {
                if constexpr (kSharded) {
                    return src.strided_pages();
                } else {
                    return src.strided_pages(num_pages);
                }
            };
            auto dst_pages = [&] {
                if constexpr (kSharded) {
                    return dst.strided_pages();
                } else {
                    return dst.strided_pages(num_pages);
                }
            };
            for (const auto& page : src_pages()) {
                read(page);
            }
            noc.async_read_barrier();
            for (const auto& page : dst_pages()) {
                write(page);
            }
        } else {
            static_assert(mode == 5 && kSharded, "strided_shard_pages() needs a sharded tensor");
            for (const auto& shard : src.strided_shard_pages()) {
                for (const auto& page : shard) {
                    read(page);
                }
            }
            noc.async_read_barrier();
            for (const auto& shard : dst.strided_shard_pages()) {
                for (const auto& page : shard) {
                    write(page);
                }
            }
        }
    };
    run(src_ta, dst_ta);
    noc.async_write_barrier();

    volatile tt_l1_ptr uint32_t* report = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        report_addr + get_my_thread_id() * kReportStride + MEM_L1_UNCACHED_BASE);
#if defined(TT_TA_ADDRGEN_STATS)
    const auto& stats = tensor_accessor::detail::transfer_stats;
    report[0] = stats.hw;
    report[1] = stats.sw_ineligible;
    report[2] = stats.sw_unsupported;
    report[3] = stats.seeks;
    report[5] = stats.skips;
    report[6] = stats.restores;
    report[7] = stats.write_seeks;
    report[8] = stats.write_restores;
    report[10] = stats.fallbacks;
    report[11] = stats.write_fallbacks;
    report[12] = stats.pushes;
#endif
    report[4] = requests;
    report[9] = 0;
    report[13] = kDoneMarker;
}
