// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Kernel used by optional-binding host tests: one always-present DFB and one optional
// scratchpad / DFB / (conceptually) token that may be Null or NonNull.
// Compiles for both ProgramSpecs — with a real optional binding and with NullBinding —
// using if constexpr on the token instead of #ifdef.

#include "api/dataflow/dataflow_buffer.h"
#include "api/scratchpad.h"

void kernel_main() {
    // Always-present DFB (host always binds this).
    DataflowBuffer always(dfb::always);

    // Optional DFB: symbol always exists; constructing is legal only when NonNull.
    if constexpr (dfb::optional_dfb) {
        DataflowBuffer opt(dfb::optional_dfb);
        (void)opt;
    }

    // Optional scratchpad: same contract.
    if constexpr (scratch::optional_pad) {
        Scratchpad<uint32_t> pad(scratch::optional_pad);
        (void)pad;
    }

    // Naming a Null token is always fine; constructing from it must not compile
    // (covered by the deleted ctor — not exercised on the Null path here).
    auto tok = dfb::optional_dfb;
    (void)tok;
    if constexpr (tok.is_null) {
        // this KernelSpec did not attach dfb::optional_dfb
    }
}
