// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// One DFB -> three tensors, round-robin: page p of dst0, dst1, dst2, then page p + 1 of each, ... A direction has two
// address-generator sides and the first walk on a side keeps it, so two tensors use the hardware and the third uses
// software (TensorAccessorAddrgenContention rows).
//
// Named RTAs: num_pages (per tensor), report_addr (12 stats words, see ta_reader_to_dfb.cpp)

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_pages = get_arg(args::num_pages);
    const uint32_t report_addr = get_arg(args::report_addr);

    Noc noc;
    DataflowBuffer dfb(dfb::in);
    const auto dst0 = TensorAccessor(tensor::dst0);
    const auto dst1 = TensorAccessor(tensor::dst1);
    const auto dst2 = TensorAccessor(tensor::dst2);

    uint32_t transfers = 0;
    auto write_page = [&](const auto& ta, uint32_t page_id) {
        noc.async_write<NocOptions::TXN_ID>(
            dfb, ta, {}, typename noc_traits_t<std::decay_t<decltype(ta)>>::dst_args_type{.page_id = page_id});
        ++transfers;
    };
    for (uint32_t page_id = 0; page_id < num_pages; ++page_id) {
        write_page(dst0, page_id);
        write_page(dst1, page_id);
        write_page(dst2, page_id);
    }
    dfb.finish();
    dfb.write_barrier(noc);

#if defined(TT_TA_ADDRGEN_STATS)
    volatile tt_l1_ptr uint32_t* report =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(report_addr + MEM_L1_UNCACHED_BASE);
    report[0] = tensor_accessor::detail::transfer_stats.hw;
    report[1] = tensor_accessor::detail::transfer_stats.sw_ineligible;
    report[2] = tensor_accessor::detail::transfer_stats.sw_unsupported;
    report[3] = tensor_accessor::detail::transfer_stats.seeks;
    report[4] = transfers;
    report[5] = tensor_accessor::detail::transfer_stats.skips;
    report[6] = tensor_accessor::detail::transfer_stats.restores;
    report[7] = tensor_accessor::detail::transfer_stats.write_seeks;
    report[8] = tensor_accessor::detail::transfer_stats.write_restores;
    report[10] = tensor_accessor::detail::transfer_stats.fallbacks;
    report[11] = tensor_accessor::detail::transfer_stats.write_fallbacks;
    report[12] = tensor_accessor::detail::transfer_stats.pushes;
#else
    (void)report_addr;
    (void)transfers;
#endif
}
