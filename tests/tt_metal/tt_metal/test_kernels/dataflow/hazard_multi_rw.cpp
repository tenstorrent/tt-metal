// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Multi-tensor hazard-test kernel for op-to-op R/W inference. It binds FOUR tensors and deliberately
// touches a proper subset: it READS in0 and in2, WRITES out, and leaves in1 bound-but-untouched. This
// exercises that resolve_buf_rw tracks the RIGHT objects -- the correct read set {in0, in2}, the correct
// write set {out}, and NOT the bound-but-unaccessed in1 (a single-tensor test cannot catch a mix-up).
//
// The `Scratchpad`, `scratch::`, `tensor::`, and `args::` tokens are emitted by genfiles from the
// kernel's scratchpad/tensor/runtime-arg bindings; no manual include for those.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "c_tensix_core.h"

void kernel_main() {
    Scratchpad<uint32_t> pad(scratch::pad);
    TensorAccessor in0(tensor::in0);
    TensorAccessor in1(tensor::in1);  // bound but NEVER accessed -> must not appear in the R/W sets
    TensorAccessor in2(tensor::in2);
    TensorAccessor out(tensor::out);
    (void)in1;

    Noc noc;
    // READ in0, then READ in2 (skip in1) -- each emits a READ note for its own binding.
    noc.async_read(in0, pad, pad.size_in_bytes(), {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    noc.async_read(in2, pad, pad.size_in_bytes(), {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();

    // WRITE out -- emits a WRITE note for out's binding.
    noc.async_write(pad, out, pad.size_in_bytes(), {.offset_bytes = 0}, {.page_id = 0});
    noc.async_write_barrier();
}
