// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

// Op-to-op R/W inference for the sharded pages() iterator: each page it yields is the NoC source endpoint, so src must
// resolve as READ only, and dst, written through its accessor, as WRITE only. The data is not checked, only the
// .tt.BUF_RW notes.
void kernel_main() {
    Scratchpad<uint32_t> pad(scratch::pad);
    const uint32_t size = pad.size_in_bytes();
    Noc noc;

    TensorAccessor src(tensor::src);
    TensorAccessor dst(tensor::dst);
    for (const auto& page : src.pages()) {
        noc.async_read(page, pad, size, {}, {});
        noc.async_read_barrier();
        noc.async_write(pad, dst, size, {}, {.page_id = page.page_id()});
        noc.async_write_barrier();
    }
}
