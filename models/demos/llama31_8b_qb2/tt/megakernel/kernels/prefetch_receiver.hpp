// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "api/dataflow/dataflow_api.h"

uint64_t head_prefetch_peer(uint32_t bank, uint32_t address) {
    return get_noc_addr(get_arg_val<uint32_t>(HEAD_PREFETCH_CONSUMER_RT + 3 + 2 * bank),
                        get_arg_val<uint32_t>(HEAD_PREFETCH_CONSUMER_RT + 4 + 2 * bank), address);
}
uint64_t wait_head_prefetch(uint32_t bank) {
    const uint32_t ready = get_arg_val<uint32_t>(HEAD_PREFETCH_CONSUMER_RT);
    const uint32_t pointer = get_arg_val<uint32_t>(HEAD_PREFETCH_CONSUMER_RT + 1);
    const uint32_t consumed = get_arg_val<uint32_t>(HEAD_PREFETCH_CONSUMER_RT + 2);
    const uint32_t next = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(consumed) ^ 1u;
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready), next);
    return head_prefetch_peer(bank, *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pointer));
}
void finish_head_prefetch(uint32_t bank) {
    // Called only after all helper-to-local CB reads have completed. Compute
    // can continue using its local copies while the helper refills next layer.
    const uint32_t consumed = get_arg_val<uint32_t>(HEAD_PREFETCH_CONSUMER_RT + 2);
    auto* value = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(consumed);
    *value = *value ^ 1u;
    noc_async_write(consumed, head_prefetch_peer(bank, consumed), 4);
    noc_async_write_barrier();
}
