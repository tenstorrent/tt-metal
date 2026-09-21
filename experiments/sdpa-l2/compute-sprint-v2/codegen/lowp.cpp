// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#ifdef SDPA_CODEGEN_OPT
#pragma GCC push_options
#pragma GCC optimize("O2")
#endif
#include "experiments/sdpa-l2/compute-sprint-v2/compensated/compute.cpp"
#ifdef SDPA_CODEGEN_OPT
#pragma GCC pop_options
#endif
