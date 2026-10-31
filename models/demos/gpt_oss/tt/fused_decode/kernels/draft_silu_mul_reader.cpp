// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Batch-32 DFlash drafter MLP (dflash/batched.py BatchedDraft, B32_DRAFT_GU): act = silu(gate) * up from one joint
// gate|up matmul's block shards (core (j, m): [32 rows, 2 NH tiles] = gate column tiles j NH .. j NH + NH - 1, then the
// up ones of the same columns), reader (NCRISC): the shard is the CB (in place).
// compile-time args: [cb_gu, NH]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_gu = get_compile_time_arg_val(0);
    constexpr uint32_t NH = get_compile_time_arg_val(1);
#ifndef SM_MPC
#define SM_MPC 1
#endif
    cb_reserve_back(cb_gu, 2 * NH * SM_MPC);  // (SM_MPC row tiles per core: the whole shard)
    cb_push_back(cb_gu, 2 * NH * SM_MPC);
}
