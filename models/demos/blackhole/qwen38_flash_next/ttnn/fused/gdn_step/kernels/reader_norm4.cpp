// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Norm core of the GDN step split 4 (QWEN38_GDN_CONV_SPLIT=4), reader: reads the fp32 reduce scaler tile 0 (1.0) of the
// scalers tensor into CB_SCALER, waits until the 4 conv tiles (q or k of one key head) are in CB_QKV slots 0-3 (SEM_QK
// counts 4) and publishes them.  Compile-time args: TensorAccessorArgs of scalers.  Runtime args: 0 scalers address.

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"
#include "../../kernels/zones.h"

namespace {
constexpr uint32_t CB_SCALER = 9, CB_QKV = 11;
constexpr uint32_t SEM_QK = 0, HT = 4;
constexpr uint32_t FP32_TILE = 4096;
}  // namespace

void kernel_main() {
    const uint32_t scalers_addr = get_arg_val<uint32_t>(0);
    constexpr auto scalers_args = TensorAccessorArgs<0>();
    const auto scalers = TensorAccessor(scalers_args, scalers_addr);
    volatile tt_l1_ptr uint32_t* sem_qk = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_QK));

    FUSED_ZONE("fz_gv_n_r");
    cb_reserve_back(CB_QKV, HT);  // the conv cores write the 4 slots
    cb_reserve_back(CB_SCALER, 1);
    noc_async_read_page(0, scalers, get_write_ptr(CB_SCALER));
    noc_async_read_barrier();
    cb_push_back(CB_SCALER, 1);
    noc_semaphore_wait_min(sem_qk, HT);
    cb_push_back(CB_QKV, HT);
    noc_semaphore_set(sem_qk, 0);
}
