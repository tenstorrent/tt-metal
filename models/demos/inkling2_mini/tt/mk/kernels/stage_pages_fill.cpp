// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Staging of a message-gate combine core's F or G slice (tt/mk/gate_stage.py): the kz x nt fp32 tiles at pages
// kt * nct + col0 + j (4 KB, interleaved DRAM) into this core's L1 staging shard at `stage`, in the order the
// combine reads them ([kt][j]). One read in flight (the hosting program waits on small DRAM reads of its own).
// Runtime args: tensor address, col0, staging L1 address.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t kz = get_named_compile_time_arg_val("kz");
constexpr uint32_t nt = get_named_compile_time_arg_val("nt");
constexpr uint32_t nct = get_named_compile_time_arg_val("nct");

void kernel_main() {
    const uint32_t addr = get_arg_val<uint32_t>(0);
    const uint32_t col0 = get_arg_val<uint32_t>(1);
    const uint32_t stage = get_arg_val<uint32_t>(2);
    const InterleavedAddrGen<true> src = {.bank_base_address = addr, .page_size = 4096};
    for (uint32_t kt = 0; kt < kz; ++kt) {
        for (uint32_t j = 0; j < nt; ++j) {
            noc_async_read(src.get_noc_addr(kt * nct + col0 + j), stage + (kt * nt + j) * 4096, 4096);
            noc_async_read_barrier();
        }
    }
}
