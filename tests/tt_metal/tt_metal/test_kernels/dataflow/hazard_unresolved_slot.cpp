// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

// Reads one bound tensor through the Noc API (a resolvable READ), plus a hand-emitted READ whose slot names no binding
// of this kernel -- stand-in for stale or foreign metadata. The host must not invent an access for it.
void kernel_main() {
    Scratchpad<uint32_t> pad(scratch::pad);
    TensorAccessor in(tensor::in);
    Noc noc;
    noc.async_read(in, pad, pad.size_in_bytes(), {.page_id = 0}, {});
    noc.async_read_barrier();
    tt_buf_rw::note<0x4000, tt_buf_rw::READ>();  // no binding at this CRTA offset
}
