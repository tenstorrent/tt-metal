// Test kernel for engine/argblock.py: every core of the kind's grid reads its block via the common runtime arg and
// writes [magic ok, kind, n, version, payload[0..n)] to row (payload[0]) of a DRAM uint32 [ncores, 32] output.
#include "api/dataflow/dataflow_api.h"
#include "../../attn/kernels/mk_argblock.hpp"

void kernel_main() {
    constexpr uint32_t KIND = get_compile_time_arg_val(0);
    constexpr auto out_args = TensorAccessorArgs<1>();
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t scratch = get_write_ptr(24);
    const auto out = TensorAccessor(out_args, out_addr, 32 * 4);
    const mk::ArgBlock b = mk::argblock(KIND);
    volatile tt_l1_ptr uint32_t* r = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    for (uint32_t i = 0; i < 32; ++i) r[i] = 0;
    r[0] = b.ok(KIND) ? 1u : 0u;
    r[1] = b.p[1];
    r[2] = b.size();
    r[3] = b.version();
    for (uint32_t i = 0; i < b.size() && i < 28; ++i) r[4 + i] = b[i];
    noc_async_write(scratch, out.get_noc_addr(b[0]), 32 * 4);
    noc_async_write_barrier();
}
