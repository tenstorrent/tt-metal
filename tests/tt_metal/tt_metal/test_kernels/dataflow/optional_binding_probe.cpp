// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Kernel used by optional-binding host tests: one always-present DFB and one optional
// scratchpad / DFB that may be Null or NonNull.
// Compiles for both ProgramSpecs — with a real optional binding and with NullBinding —
// using if constexpr on the token instead of #ifdef.
//
// Construction from an optional token must go through a NullState-templated helper so the
// DataflowBuffer / Scratchpad ctor call is dependent. A direct
//   if constexpr (dfb::optional_dfb) { DataflowBuffer(dfb::optional_dfb); }
// is still ill-formed when the token is Null: the discarded branch is non-dependent, and
// the deleted Null ctor is diagnosed anyway.

#include "api/dataflow/dataflow_buffer.h"
#include "api/scratchpad.h"

template <NullState S>
void maybe_use_dfb(DFBBindingToken<S> tok) {
    if constexpr (tok) {
        DataflowBuffer opt(tok);
        (void)opt;
    }
}

template <NullState S>
void maybe_use_scratch(ScratchpadBindingToken<S> tok) {
    if constexpr (tok) {
        Scratchpad<uint32_t> pad(tok);
        (void)pad;
    }
}

void kernel_main() {
    // Always-present DFB (host always binds this).
    DataflowBuffer always(dfb::always);

    // Optional DFB / scratchpad: symbols always exist; construction only when NonNull.
    maybe_use_dfb(dfb::optional_dfb);
    maybe_use_scratch(scratch::optional_pad);

    // Naming a Null token is always fine.
    auto tok = dfb::optional_dfb;
    (void)tok;
    if constexpr (tok.is_null) {
        // this KernelSpec did not attach dfb::optional_dfb
    }
}
