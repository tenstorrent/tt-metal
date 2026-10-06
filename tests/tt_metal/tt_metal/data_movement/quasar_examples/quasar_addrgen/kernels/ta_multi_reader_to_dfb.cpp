// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Three tensors -> one DFB, round-robin: page p of src0, src1, src2, then page p + 1 of each, ... A direction has two
// address-generator sides and the first walk on a side keeps it, so two tensors use the hardware and the third uses
// software (TensorAccessorAddrgenContention rows).
//
// Named RTAs: num_pages (per tensor), report_addr (12 stats words, see ta_reader_to_dfb.cpp)

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

#if defined(TT_TA_ADDRGEN_STATS) && defined(ARCH_QUASAR)
// Stack high-water mark, for the walker-state budget (DM cores share 8 KB between thread-local storage and stack):
// paint the free stack with a pattern at entry, count the untouched words at exit. Same scheme as
// internal/debug/stack_usage.h, which only exists with the watcher on. Reported as bytes never used.
extern thread_local uint32_t __stack_base_lwm[];
extern uint32_t __stack_base_offset[];
static inline void paint_stack() {
    uint32_t* base = __stack_base_lwm + reinterpret_cast<uintptr_t>(__stack_base_offset);
    uint32_t* sp;
    asm volatile("mv %0,sp" : "=r"(sp));
    for (uint32_t* p = sp - 8; p != base;) {  // leave a few words for this function's own frame
        *--p = 0xBABABABAu;
    }
}
static inline uint32_t unused_stack_bytes() {
    uint32_t* base = __stack_base_lwm + reinterpret_cast<uintptr_t>(__stack_base_offset);
    uint32_t* p = base;
    while (*p == 0xBABABABAu) {
        ++p;
    }
    return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(p) - reinterpret_cast<uintptr_t>(base));
}
#endif

void kernel_main() {
#if defined(TT_TA_ADDRGEN_STATS) && defined(ARCH_QUASAR)
    paint_stack();
#endif
    const uint32_t num_pages = get_arg(args::num_pages);
    const uint32_t report_addr = get_arg(args::report_addr);

    Noc noc;
    DataflowBuffer dfb(dfb::out);
    const auto src0 = TensorAccessor(tensor::src0);
    const auto src1 = TensorAccessor(tensor::src1);
    const auto src2 = TensorAccessor(tensor::src2);

    uint32_t transfers = 0;
    auto read_page = [&](const auto& ta, uint32_t page_id) {
        noc.async_read<NocOptions::TXN_ID>(
            ta, dfb, typename noc_traits_t<std::decay_t<decltype(ta)>>::src_args_type{.page_id = page_id}, {});
        ++transfers;
    };
    for (uint32_t page_id = 0; page_id < num_pages; ++page_id) {
        read_page(src0, page_id);
        read_page(src1, page_id);
        read_page(src2, page_id);
    }
    dfb.finish();

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
    report[9] = unused_stack_bytes();
    report[10] = tensor_accessor::detail::transfer_stats.fallbacks;
    report[11] = tensor_accessor::detail::transfer_stats.write_fallbacks;
    report[12] = tensor_accessor::detail::transfer_stats.pushes;
#else
    (void)report_addr;
    (void)transfers;
#endif
}
