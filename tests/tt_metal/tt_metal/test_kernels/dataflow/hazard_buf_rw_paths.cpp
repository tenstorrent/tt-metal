// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/local_tensor_accessor.h"
#include "experimental/kernel_args.h"

// Op-to-op R/W inference coverage: each tensor binding is touched through exactly one NoC transfer path or endpoint
// kind, so the kernel's resolved R/W sets show, per binding, whether that path is attributed (and as what). One page
// per access; the data is not checked, only the .tt.BUF_RW notes.
void kernel_main() {
    Scratchpad<uint32_t> pad(scratch::pad);
    const uint32_t l1 = pad.get_base_address();
    const uint32_t size = pad.size_in_bytes();
    Noc noc;

    // Typed Noc API, endpoints that stand for an accessor: PageView, pages() / shard_pages() iterator pages,
    // ShardView -> READ.
    TensorAccessor view(tensor::view);
    noc.async_read(PageView(view), pad, size, {.page_id = 0}, {});
    TensorAccessor iter(tensor::iter);
    for (const auto& page : iter.pages(0, 1)) {
        noc.async_read(page, pad, size, {}, {});
    }
    TensorAccessor shard_iter(tensor::shard_iter);
    for (const auto& page : shard_iter.shard_pages(0)) {
        noc.async_read(page, pad, size, {}, {});
    }
    TensorAccessor shard(tensor::shard);
    noc.async_read(ShardView(shard), pad, size, {.shard_id = 0}, {});
    noc.async_read_barrier();

    // Typed Noc API, transfer paths other than plain async_write: stateful write, DRAM zero-fill -> WRITE.
    TensorAccessor state(tensor::state);
    noc.set_async_write_state(state, size, {.page_id = 0});
    noc.async_write_with_state(pad, state, size, {}, {.page_id = 0});
    noc.async_write_barrier();
    TensorAccessor zero(tensor::zero);
    noc.async_write_zeros(zero, size, {.page_id = 0}, pad);
    noc.write_zeros_dram_barrier();

    // Legacy accessor free functions -> READ / WRITE.
    TensorAccessor legacy_r(tensor::legacy_r);
    noc_async_read_page(0, legacy_r, l1);
    noc_async_read_barrier();
    TensorAccessor legacy_w(tensor::legacy_w);
    noc_async_write_page(0, legacy_w, l1);
    noc_async_write_barrier();

    // The binding is erased or the address leaves it: type-erased wrapper, raw NoC on an escaped address, direct CPU
    // access through LocalTensorAccessor -> READ and WRITE (the kernel could do either).
    TensorAccessor wrapped(tensor::wrap);
    AbstractTensorAccessorWrapper wrapper(wrapped);
    noc.async_read(wrapper, pad, size, {.page_id = 0}, {});
    noc.async_read_barrier();
    TensorAccessor escape(tensor::escape);
    noc_async_read(escape.get_noc_addr(0), l1, size);
    noc_async_read_barrier();
    LocalTensorAccessor<uint32_t> local(tensor::local);
    volatile uint32_t sink = local[0];
    (void)sink;

    TensorAccessor unused(tensor::unused);  // bound but never accessed -> in neither set
    (void)unused;
}
