// L2-norm root reader: the group's N fp32 conv tiles (bench: from DRAM; megakernel: pushed by the chunk cores), the
// reduce scaler (fp32 1.0, all faces' column 0 as the reduce expects), eps_l2 (bf16 col-scalar) and mask_T (bf16).
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "l2r_common.hpp"

static inline void zero_tile(uint32_t addr, uint32_t bytes) {
    volatile uint32_t* p = reinterpret_cast<volatile uint32_t*>(addr);
    for (uint32_t i = 0; i < bytes / 4; ++i) p[i] = 0;
}

void kernel_main() {
    constexpr uint32_t N = get_compile_time_arg_val(0), T = get_compile_time_arg_val(1);
    constexpr auto src_args = TensorAccessorArgs<2>();
    const uint32_t src_addr = get_arg_val<uint32_t>(0), tile0 = get_arg_val<uint32_t>(1);
    const uint32_t eps_bits = get_arg_val<uint32_t>(2);
    const auto src = TensorAccessor(src_args, src_addr, 4096);
    DeviceZoneScopedN("L2R-R");
    cb_reserve_back(L_SRC, N);
    for (uint32_t i = 0; i < N; ++i) noc_async_read(src.get_noc_addr(tile0 + i), get_write_ptr(L_SRC) + i * 4096, 4096);
    // constants (megakernel: resident)
    cb_reserve_back(L_SCALER, 1);
    {
        const uint32_t a = get_write_ptr(L_SCALER);
        zero_tile(a, 4096);
        volatile uint32_t* p = reinterpret_cast<volatile uint32_t*>(a);
        for (uint32_t f = 0; f < 4; ++f)
            for (uint32_t c = 0; c < 16; ++c) p[f * 256 + c] = 0x3F800000u;  // row 0 of every face = 1.0 (fp32)
    }
    cb_push_back(L_SCALER, 1);
    cb_reserve_back(L_EPS, 1);
    {
        const uint32_t a = get_write_ptr(L_EPS);
        zero_tile(a, 2048);
        volatile uint16_t* p = reinterpret_cast<volatile uint16_t*>(a);
        const uint16_t v = eps_bits >> 16;
        for (uint32_t k = 0; k < 4; k += 2)
            for (uint32_t j = 0; j < 256; j += 16) p[(k << 8) + j] = v;  // column 0 of faces 0 / 2 (generate_bcast_col_scalar)
    }
    cb_push_back(L_EPS, 1);
    cb_reserve_back(L_MASK, 1);
    {
        const uint32_t a = get_write_ptr(L_MASK);
        zero_tile(a, 2048);
        volatile uint16_t* p = reinterpret_cast<volatile uint16_t*>(a);
        for (uint32_t t = 0; t < T; ++t) p[(t >> 4) * 512 + (t & 15) * 16] = 0x3F80;  // (row t, col 0): face 0 / 2
    }
    cb_push_back(L_MASK, 1);
    {DeviceZoneScopedN("L2R-wait"); noc_async_read_barrier();}
    cb_push_back(L_SRC, N);
}
