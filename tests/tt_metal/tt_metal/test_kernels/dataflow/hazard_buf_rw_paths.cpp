// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/local_tensor_accessor.h"
#include "experimental/kernel_args.h"
#include "internal/local_mem/address_access.h"

// Op-to-op R/W inference coverage: one tensor binding per NoC access path, so the kernel's resolved R/W sets show,
// per binding, whether that path is attributed (and as what). A path that fails to note at all would show up as its
// binding MISSING from both sets -- the dangerous failure (a detector sees no access and could wrongly relax a real
// hazard). Over-attributing (both READ+WRITE, or OPAQUE) is merely imprecise, not dangerous, so it's expected for the
// bindings below where the specific op can't be told apart. One page per access; the data is not checked, only the
// .tt.BUF_RW notes. Expected outcome per binding (see test_op_to_op_hazards.cpp's
// BufRwCoversEveryTransferPathAndEndpoint for the actual assertions):
//   view, iter, shard_iter, shard   -- noc.async_read via PageView/iterator pages/ShardView      -> READ only
//   bank_r, bank_state              -- BankView: noc.async_read; set_async_read_state + async_read_with_state
//                                       -> READ only
//   bank_w                          -- BankView: noc.async_write                                  -> WRITE only
//   state, zero                     -- noc.async_write_with_state, noc.async_write_zeros         -> WRITE only
//   legacy_r, legacy_w              -- pre-Noc-class free functions noc_async_read/write_page    -> READ/WRITE only
//   wrap                            -- AbstractTensorAccessorWrapper (type-erased)                -> READ and WRITE
//   escape                          -- get_noc_addr() fed to a raw noc_async_read                 -> READ and WRITE,
//                                       plus trips the raw call's own OPAQUE tag (it can't tell an escaped-but-known
//                                       address apart from a truly untracked one)
//   local                           -- LocalTensorAccessor direct CPU-side access, no NoC call     -> READ and WRITE
//   unused                          -- bound but never touched                                    -> neither set
void kernel_main() {
    Scratchpad<uint32_t> pad(scratch::pad);
    // Raw address: the legacy free functions and the raw noc_async_read below take a uint32_t L1 address.
    const uint32_t l1 = static_cast<uint32_t>(local_mem::detail::AddressAccess::address(pad.local_mem()));
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

    // BankView, plain and stateful -> READ; a write through it -> WRITE. Each tensor is one page in bank 0, the
    // same size as the transfer, so these stay inside the tensor.
    TensorAccessor bank_r(tensor::bank_r);
    noc.async_read(BankView(bank_r), pad, size, {.bank_id = 0}, {});
    TensorAccessor bank_state(tensor::bank_state);
    noc.set_async_read_state(BankView(bank_state), size, {.bank_id = 0});
    noc.async_read_with_state(BankView(bank_state), pad, size, {.bank_id = 0}, {});
    noc.async_read_barrier();
    TensorAccessor bank_w(tensor::bank_w);
    noc.async_write(pad, BankView(bank_w), size, {}, {.bank_id = 0});
    noc.async_write_barrier();

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

    // The binding is erased: a type-erased wrapper -> READ and WRITE (the kernel could do either).
    TensorAccessor wrapped(tensor::wrap);
    AbstractTensorAccessorWrapper wrapper(wrapped);
    noc.async_read(wrapper, pad, size, {.page_id = 0}, {});
    noc.async_read_barrier();
    // The address leaves the accessor and is fed to a raw NoC call: READ and WRITE from the escape, but the
    // raw dataflow_api.h entry point also conservatively tags the whole kernel OPAQUE (it cannot tell this
    // uint64_t apart from a truly untracked one).
    TensorAccessor escape(tensor::escape);
    noc_async_read(escape.get_noc_addr(0), l1, size);
    noc_async_read_barrier();
    LocalTensorAccessor<uint32_t> local(tensor::local);
    volatile uint32_t sink = local[0];
    (void)sink;

    TensorAccessor unused(tensor::unused);  // bound but never accessed -> in neither set
    (void)unused;
}
