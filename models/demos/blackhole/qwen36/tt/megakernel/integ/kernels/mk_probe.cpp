// Megakernel integration stage0 probe (QWEN36_R16_MEGAKERNEL=stage0): one core per device reads the bucket's phase
// table from DRAM (uint32 ROW_MAJOR [n_phases, 16]) and writes (n_phases, sum(word0), FNV-1a-32 of all words, marker)
// to out[0..3]. It proves the per-bucket table tensor, the mesh generic_op inside the captured verify trace, and replay,
// without touching the residual. Must match integ/phase_table.py:checksum().
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t WORDS = 16;
    constexpr auto tbl_args = TensorAccessorArgs<0>();
    constexpr auto out_args = TensorAccessorArgs<tbl_args.next_compile_time_args_offset()>();
    uint32_t a = 0;
    const uint32_t tbl_addr = get_arg_val<uint32_t>(a++);
    const uint32_t out_addr = get_arg_val<uint32_t>(a++);
    const uint32_t n_phases = get_arg_val<uint32_t>(a++);
    const uint32_t scratch = get_write_ptr(24);  // CB 24 (scratch, 4 KB) base: 64 B aligned L1
    const uint32_t row_bytes = WORDS * 4;
    const auto tbl = TensorAccessor(tbl_args, tbl_addr, row_bytes);
    const auto out = TensorAccessor(out_args, out_addr, 32 * 4);

    volatile tt_l1_ptr uint32_t* row = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    uint32_t sum = 0, x = 0x811C9DC5u;  // FNV-1a-32 over all words (phase_table.checksum)
    for (uint32_t p = 0; p < n_phases; ++p) {
        noc_async_read(tbl.get_noc_addr(p), scratch, row_bytes);
        noc_async_read_barrier();
        sum += row[0];
        for (uint32_t w = 0; w < WORDS; ++w) {
            x = (x ^ row[w]) * 0x01000193u;
        }
    }
    volatile tt_l1_ptr uint32_t* res = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + 64 * 2);
    for (uint32_t i = 0; i < 32; ++i) res[i] = 0;
    res[0] = n_phases;
    res[1] = sum;
    res[2] = x;
    res[3] = 0x4D4B5030;  // "MKP0" marker
    noc_async_write(scratch + 64 * 2, out.get_noc_addr(0), 32 * 4);
    noc_async_write_barrier();
}
