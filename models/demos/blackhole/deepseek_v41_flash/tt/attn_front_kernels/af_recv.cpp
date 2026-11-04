// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// attention-front POC, NCRISC of a helper core (split configuration): receives the weight units of its columns that the
// bank core (af_reader2.cpp) forwards into its ring cb_w.  Two units are kept reserved (credits sent) ahead of the unit
// being waited for; a unit is pushed to the compute kernel when the data semaphore shows it arrived.  24 units of 40
// tiles: wqkv columns 4..7 (4 units per column) then the odd wq_b columns (1 unit per column). CT args: 0 cb_w, 1 sem
// credit id, 2 sem data id.  RT args: bank core noc x, y.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_w = get_compile_time_arg_val(0);
    constexpr uint32_t sem_credit_id = get_compile_time_arg_val(1);
    constexpr uint32_t sem_data_id = get_compile_time_arg_val(2);
    constexpr uint32_t FU = 40;
    constexpr uint32_t TOTAL = 24;
    const uint32_t bank_x = get_arg_val<uint32_t>(0), bank_y = get_arg_val<uint32_t>(1);
    volatile tt_l1_ptr uint32_t* data = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_data_id));
    const uint32_t credit_addr = get_semaphore(sem_credit_id);

    uint32_t credits_sent = 0;
    for (uint32_t i = 0; i < TOTAL; ++i) {
        const uint32_t ahead = (TOTAL - i) >= 2 ? 2 : 1;
        cb_reserve_back(cb_w, ahead * FU);
        while (credits_sent < i + ahead) {
            noc_semaphore_inc(get_noc_addr(bank_x, bank_y, credit_addr), 1);
            credits_sent++;
        }
        noc_semaphore_wait_min(data, i + 1);
        cb_push_back(cb_w, FU);
    }
    noc_async_atomic_barrier();
    *data = 0;
}
