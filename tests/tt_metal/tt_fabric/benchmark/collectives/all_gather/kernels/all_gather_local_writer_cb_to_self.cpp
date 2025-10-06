// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "dataflow_api.h"
#include "accessor/tensor_accessor.h"
#include "accessor/tensor_accessor_args.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"

using namespace tt;

void kernel_main() {
    // CT args: 0: TOTAL_PAGES, 1: PAGE_SIZE
    constexpr auto ta_args = TensorAccessorArgs<0>();
    constexpr uint32_t CTA_BASE = ta_args.next_compile_time_args_offset();
    constexpr uint32_t TOTAL_PAGES = get_compile_time_arg_val(CTA_BASE + 0);
    constexpr uint32_t PAGE_SIZE = get_compile_time_arg_val(CTA_BASE + 1);
    constexpr uint32_t CB_ID = tt::CBIndex::c_0;

    // RT args: 0: dst_base (u32)
    size_t idx = 0;
    const uint32_t dst_base = get_arg_val<uint32_t>(idx++);

    const auto dst_acc = TensorAccessor(ta_args, /*bank_base=*/dst_base, /*page_size=*/PAGE_SIZE);

    for (uint32_t i = 0; i < TOTAL_PAGES; ++i) {
        cb_wait_front(CB_ID, 1);
        const uint32_t src_l1_addr = get_read_ptr(CB_ID);

        const uint64_t self_noc_addr = dst_acc.get_noc_addr(/*page_id=*/i, /*offset=*/0, /*noc=*/0);
        noc_async_write(src_l1_addr, self_noc_addr, PAGE_SIZE);

        cb_pop_front(CB_ID, 1);
    }
    // Finish the local NoC writes before returning
    noc_async_write_barrier();
}
