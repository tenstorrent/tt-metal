// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Companion to report_page_size.cpp, exercising the RuntimePageSize (dynamic page-size)
// relaxation on its home turf -- the device TensorAccessor -- in isolation from Metal 2.0.
//
// The INPUT TensorAccessorArgs has the RuntimePageSize bit set in its args_config CTA word
// and its aligned-page-size CTA slot OMITTED ("A-collapse"), so the page size is sourced
// from a common runtime arg instead of a compile-time arg. The host forges this layout by
// hand (the legacy TensorAccessorArgs(buffer) path never sets the bit).
//
// We build the input accessor with the explicit 3-arg ctor (args, addr, page_size), mirroring
// what the Metal 2.0 binding token does internally -- the 2-arg ctor would default page_size to
// the *static* AlignedPageSize, which is deliberately 0 when RuntimePageSize is set.
//
// Reports two words for host verification:
//   ptr[0] = input accessor's aligned page size  -> must equal the common-runtime-arg value
//            (proves get_aligned_page_size() reads the CRTA, not a stale/zero static slot)
//   ptr[1] = output accessor's aligned page size  -> must equal its static CTA value
//            (proves the input's A-collapse left the *next* accessor's CTA offset intact)

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t input_addr = get_arg_val<uint32_t>(0);
    uint32_t output_addr = get_arg_val<uint32_t>(1);

    // The input's dynamic page size is supplied as common runtime arg 0.
    constexpr uint32_t input_page_size_crta_index = 0;

    constexpr uint32_t input_ta_cta_offset = 0;
    constexpr auto input_ta_args = TensorAccessorArgs<input_ta_cta_offset>();

    // A-collapse: the input accessor consumes one fewer CTA word (no page-size slot), so the
    // output accessor's CTA offset must come from next_compile_time_args_offset(), not a fixed +2.
    constexpr uint32_t output_ta_cta_offset = input_ta_args.next_compile_time_args_offset();
    constexpr auto output_ta_args = TensorAccessorArgs<output_ta_cta_offset>();

    // 3-arg ctor: feed the page size from the args getter (which reads the CRTA), mirroring the
    // Metal 2.0 binding token. The default-constructed args reads common arg index 0.
    auto input_accessor = TensorAccessor(input_ta_args, input_addr, input_ta_args.get_aligned_page_size());
    auto output_accessor = TensorAccessor(output_ta_args, output_addr);

    constexpr uint32_t output_cb = 0;
    cb_reserve_back(output_cb, 1);
    uint32_t l1_addr = get_write_ptr(output_cb);
    auto* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_addr);
    ptr[0] = input_accessor.get_aligned_page_size();
    ptr[1] = output_accessor.get_aligned_page_size();

    uint64_t dram_noc_addr = output_accessor.get_noc_addr(0);
    noc_async_write(l1_addr, dram_noc_addr, output_accessor.get_aligned_page_size());
    noc_async_write_barrier();
}
