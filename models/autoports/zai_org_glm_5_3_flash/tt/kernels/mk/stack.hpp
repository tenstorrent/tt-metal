// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 14b (PRODUCTION_PATH.md M5): the layer loop of the persistent stack program (tt/mk_stack.py MkStack). Every
// RISC of every core runs its phase sequence MK_LAYERS times inside ONE launch.
//
// Per-layer runtime args: a data-movement kernel whose runtime args differ between layers (weight / state / parameter
// addresses) gets three extra args at MK_LARGS_RT: the DRAM address of the stack's argument table (uint32 rows of
// MK_LARGS_PAGE bytes), this (kernel, core)'s row for layer 0, and the row stride per layer. A row is the kernel's
// COMPLETE runtime-arg vector for that layer (incl. these three), so after `load_args(layer)` points rta_l1_base at the
// row (in this RISC's slot of the MK_ARG_CB scratch, offset MK_LARGS_OFF) every phase reads that layer's args
// unchanged. Compute kernels' args are identical for every layer (checked on the host) and never reloaded.
// Without MK_LAYERS the macros compile to the single-layer program.
#pragma once

#include <cstdint>

#if !defined(MK_LAYERS)
#define MK_LAYERS 1
#endif

#if !defined(COMPILE_FOR_TRISC)
#include "api/dataflow/dataflow_api.h"

namespace mkstack {
// rows of the argument table -> this RISC's runtime-arg pointer (no-op for kernels without per-layer args)
FORCE_INLINE void load_args(uint32_t layer) {
#if defined(MK_LARGS_RT)
    const uint32_t tbl = get_arg_val<uint32_t>(MK_LARGS_RT);
    const uint32_t row = get_arg_val<uint32_t>(MK_LARGS_RT + 1) + layer * get_arg_val<uint32_t>(MK_LARGS_RT + 2);
    const uint32_t dst = ((get_write_ptr(MK_ARG_CB) + 63) & ~63u) + MK_LARGS_OFF;  // DRAM reads: 64-B aligned L1
    const InterleavedAddrGen<true> t{.bank_base_address = tbl, .page_size = MK_LARGS_PAGE};
    noc_async_read(t.get_noc_addr(row), dst, MK_LARGS_PAGE);
    noc_async_read_barrier();
    rta_l1_base = reinterpret_cast<uint32_t tt_l1_ptr*>(dst);
#else
    (void)layer;
#endif
}
}  // namespace mkstack
#endif
