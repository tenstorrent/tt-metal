// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Function optimization attribute experiment, NOT descriptor/linker opt_level.
// Default build/link flags (including FP reassociation prohibitions) are unchanged.
#pragma GCC push_options
#if SDPA_CODEGEN_OPT == 2
#pragma GCC optimize("O2")
#elif SDPA_CODEGEN_OPT == 3
#pragma GCC optimize("O3")
#elif SDPA_CODEGEN_OPT == 4
#pragma GCC optimize("Os")
#else
#error "Select an explicit code-generation experiment"
#endif
#include "experiments/sdpa-l2/compute-sprint-v1/fp32/compute.cpp"
#pragma GCC pop_options
