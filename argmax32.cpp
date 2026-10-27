// Per-row argmax: one core per batch row, no inter-core communication.
// Each core reads its full row of bf16 values in chunks, finds the max index,
// and writes the uint32 result replicated 8x to its own 32-byte-aligned page.
// Output tensor is [batch, 8] uint32 ROW_MAJOR (32-byte pages avoid DRAM cache-line race).
//
// compile_time_args:
//   0: cb_scr (CB index)
//   1: row_elems (vocab, e.g. 154880)
//   2: chunk_elems (elements per read, e.g. 16384)
//   3+: TensorAccessorArgs for src
//   next+: TensorAccessorArgs for dst
//
// runtime_args (per core):
//   0: src buffer address
//   1: dst buffer address
//   2: row_id (0..batch-1)

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

namespace {
inline uint32_t key_of(uint16_t b) { return (b & 0x8000u) ? ((~b) & 0xFFFFu) : (b | 0x8000u); }
}  // namespace

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);
    const uint32_t row_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_scr = get_compile_time_arg_val(0);
    constexpr uint32_t row_elems = get_compile_time_arg_val(1);
    constexpr uint32_t chunk_elems = get_compile_time_arg_val(2);
    constexpr auto src_ta = TensorAccessorArgs<3>();
    constexpr auto dst_ta = TensorAccessorArgs<src_ta.next_compile_time_args_offset()>();

    constexpr uint32_t chunk_bytes = ((chunk_elems * 2u) + 31u) & ~31u;
    constexpr uint32_t out_page_bytes = 32u;

    const auto s_src = TensorAccessor(src_ta, src_addr, row_elems * 2u);
    const auto s_dst = TensorAccessor(dst_ta, dst_addr, out_page_bytes);

    const uint32_t base = get_write_ptr(cb_scr);

    uint32_t best_key = 0;
    uint32_t best_idx = 0;

    for (uint32_t start = 0; start < row_elems; start += chunk_elems) {
        uint32_t n = (start + chunk_elems <= row_elems) ? chunk_elems : (row_elems - start);
        uint32_t read_bytes = ((n * 2u) + 31u) & ~31u;

        uint64_t noc_addr = s_src.get_noc_addr(row_id);
        noc_async_read(noc_addr + (uint64_t)(start * 2u), base, read_bytes);
        noc_async_read_barrier();

        volatile tt_l1_ptr uint16_t* vals = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base);
        for (uint32_t i = 0; i < n; i++) {
            uint32_t k = key_of(vals[i]);
            if (k > best_key) {
                best_key = k;
                best_idx = start + i;
            }
        }
    }

    const uint32_t out_l1 = base + chunk_bytes;
    volatile tt_l1_ptr uint32_t* out_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_l1);
    for (uint32_t i = 0; i < 8; i++) {
        out_ptr[i] = best_idx;
    }

    uint64_t dst_noc = s_dst.get_noc_addr(row_id);
    noc_async_write(out_l1, dst_noc, out_page_bytes);
    noc_async_write_barrier();
}
