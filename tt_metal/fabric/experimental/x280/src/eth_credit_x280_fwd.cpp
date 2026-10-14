// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// src/eth_credit_x280_fwd.cpp -- X280 active firmware entry point for the
// eth_credit forwarders. Every hart enters here; each reads its own record of
// the args block and returns when it has none.

#include <stddef.h>
#include <stdint.h>

#include "rt/x280_idle.h"
#include "eth_credit/eth_credit_abi.h"
#include "eth_credit/x280_fwd.hpp"

int main(uint64_t hartid) {
    const uint64_t run_value = x280_hart_control(hartid)->run_value;
    const uintptr_t block = (uintptr_t)x280_hart_args(hartid);
    const uint64_t block_bytes = x280_hart_args_bytes();

    if (block == 0u || (block % EC_LIM_ALIGN) != 0u || block_bytes < EC_X280_HARTS * sizeof(ec_x280_fwd_args_t)) {
        x280_hart_set_result(hartid, (uint64_t)EC_ERROR_BAD_ARGS);
        return 0;
    }

    const ec_x280_fwd_args_t* args = reinterpret_cast<const ec_x280_fwd_args_t*>(block);
    const ec_error_t err = eth_credit::run_forwarder(hartid, run_value, args);
    x280_hart_set_result(hartid, (uint64_t)err);
    return 0;
}
