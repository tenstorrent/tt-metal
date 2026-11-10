// Probe for engine/cbplan.py (spec §20, MK_V12_CBCHK): run with the engine program's CB list; every core writes
// [magic "MKBC", get_write_ptr(id) for each planned id] to row (arg 1) of a DRAM uint32 [ncores, 64] output. The host
// compares with the planned addresses. Reads the CB interfaces before touching L1; the scratch is this core's first CB.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr auto out_args = TensorAccessorArgs<0>();
    const uint32_t out_addr = get_arg_val<uint32_t>(0), row = get_arg_val<uint32_t>(1), n = get_arg_val<uint32_t>(2);
    uint32_t w[64];
    w[0] = 0x43424B4Du;
    for (uint32_t i = 0; i < n; ++i) w[1 + i] = get_write_ptr(get_arg_val<uint32_t>(3 + i));
    for (uint32_t i = 1 + n; i < 64; ++i) w[i] = 0;
    const auto out = TensorAccessor(out_args, out_addr, 64 * 4);
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(get_arg_val<uint32_t>(3)));
    for (uint32_t i = 0; i < 64; ++i) s[i] = w[i];
    noc_async_write(reinterpret_cast<uint32_t>(s), out.get_noc_addr(row), 64 * 4);
    noc_async_write_barrier();
}
