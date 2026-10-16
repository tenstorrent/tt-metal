// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Reads WORDS L1 words (the global semaphores of the line transport) on this core and writes them, as uint32, into
// row 0 of page `slot` of a uint32 TILE tensor: the host reads every device's counters back after a run.
// Compile-time args: 0 WORDS, 1.. TensorAccessorArgs of the tensor.  Runtime args: 0 tensor address, 1 slot,
// 2.. the WORDS addresses.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "../../kernels/zones.h"

constexpr uint32_t WORDS = get_compile_time_arg_val(0);
constexpr uint32_t CB = 0;

void kernel_main() {
    FUSED_ZONE("fz_gf_probe");
    constexpr auto args = TensorAccessorArgs<1>();
    const auto out = TensorAccessor(args, get_arg_val<uint32_t>(0));
    const uint32_t slot = get_arg_val<uint32_t>(1);
    Noc noc;
    DataflowBuffer page(CB);
    page.reserve_back(1);
    volatile tt_l1_ptr uint32_t* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(page.get_write_ptr());
    for (uint32_t i = 0; i < WORDS; ++i) {
        words[i] = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(2 + i));
    }
    for (uint32_t i = WORDS; i < 16; ++i) {
        words[i] = 0;
    }
    page.push_back(1);
    page.wait_front(1);
    noc.async_write(page, out, get_tile_size(CB), {}, {.page_id = slot});
    noc.async_write_barrier();
    page.pop_front(1);
}
