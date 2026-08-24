// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Kernel used by optional-binding host tests: one always-present DFB and one optional
// scratchpad / DFB that may be a real token or the matching null token.
// Compiles for both ProgramSpecs — with the optional resource declared and with it omitted —
// using overloads / a function template instead of #ifdef.
//
// Construction from a possibly-null name must go through an overload or a function template
// so the DataflowBuffer / Scratchpad ctor call is on a parameter. A direct
//   if constexpr (!dfb::optional_dfb.is_null) { DataflowBuffer(dfb::optional_dfb); }
// is still ill-formed when the symbol is a null type: the discarded branch is non-dependent,
// and the deleted null ctor is diagnosed anyway.

#include "api/dataflow/dataflow_buffer.h"
#include "api/scratchpad.h"

void maybe_use_dfb(DFBBindingToken tok) {
    DataflowBuffer opt(tok);
    (void)opt;
}
void maybe_use_dfb(NullDFBBindingToken) {}

template <typename Tok>
void maybe_use_scratch(Tok tok) {
    if constexpr (!tok.is_null) {
        Scratchpad<uint32_t> pad(tok);
        (void)pad;
    }
}

void kernel_main() {
    // Always-present DFB (host always binds this).
    DataflowBuffer always(dfb::always);

    // Optional DFB / scratchpad: symbols always exist; construction only from a real token.
    maybe_use_dfb(dfb::optional_dfb);
    maybe_use_scratch(scratch::optional_pad);

    // Naming a null token is always fine.
    auto tok = dfb::optional_dfb;
    (void)tok;
    if constexpr (tok.is_null) {
        // this KernelSpec did not attach dfb::optional_dfb
    }
}
