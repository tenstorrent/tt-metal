// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Function attributes only: numerical recipes and global build flags unchanged.
#pragma GCC push_options
#if SDPA_CODEGEN_OPT == 2
#pragma GCC optimize("O2")
#elif SDPA_CODEGEN_OPT == 3
#pragma GCC optimize("O3")
#else
#error "Select explicit O2 or O3 wrapper"
#endif
#include "experiments/sdpa-l2/compute-sprint-v1/bf16/compute.cpp"
#pragma GCC pop_options
