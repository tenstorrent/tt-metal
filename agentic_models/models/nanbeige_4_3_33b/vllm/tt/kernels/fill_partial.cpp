#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr uint32_t B = get_compile_time_arg_val(0), H = get_compile_time_arg_val(1),
                       T = get_compile_time_arg_val(2), W = get_compile_time_arg_val(3);
    constexpr uint32_t PT = get_compile_time_arg_val(4), META = get_compile_time_arg_val(5),
                       CORES = get_compile_time_arg_val(6), TILE = get_compile_time_arg_val(7);
    constexpr auto sa = TensorAccessorArgs<8>();
    constexpr auto ca = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    constexpr auto pa = TensorAccessorArgs<ca.next_compile_time_args_offset()>();
    constexpr auto ma = TensorAccessorArgs<pa.next_compile_time_args_offset()>();
    auto src = TensorAccessor(sa, get_arg_val<uint32_t>(0), TILE);
    auto dst = TensorAccessor(ca, get_arg_val<uint32_t>(1), TILE);
    auto pt = TensorAccessor(pa, get_arg_val<uint32_t>(2), PT);
    auto meta = TensorAccessor(ma, get_arg_val<uint32_t>(3), META);
    uint32_t me = get_arg_val<uint32_t>(4);
    uint32_t s = get_write_ptr(0), d = s + TILE, p = get_write_ptr(1), m = get_write_ptr(2);
    noc_async_read(meta.get_noc_addr(0), m, META);
    noc_async_read_barrier();
    auto md = reinterpret_cast<volatile tt_l1_ptr int32_t*>(m);
    // Metadata triples: aligned source absolute start, first valid source row, valid row count.
    for (uint32_t job = me; job < B * H * (T / 32) * W; job += CORES) {
        uint32_t w = job % W, q = job / W % (T / 32), h = job / (W * (T / 32)) % H, b = job / (W * (T / 32) * H);
        int32_t offset = get_arg_val<uint32_t>(5);
        int32_t start = md[3 * b] + offset;
        int32_t first = md[3 * b + 1] - offset;
        int32_t end = first + md[3 * b + 2];
        if (first < 0) {
            first = 0;
        }
        if (end > int(T)) {
            end = T;
        }
        int32_t n = end - first;
        if (n <= 0 || int(q * 32 + 32) <= first || int(q * 32) >= first + n) {
            continue;
        }
        noc_async_read(pt.get_noc_addr(b), p, PT);
        noc_async_read_barrier();
        auto pages = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(p);
        uint32_t block = pages[(start + q * 32) / 32];
        uint32_t di = (block * H + h) * W + w;
        noc_async_read(src.get_noc_addr(((b * H + h) * (T / 32) + q) * W + w), s, TILE);
        noc_async_read(dst.get_noc_addr(di), d, TILE);
        noc_async_read_barrier();
        auto a = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(s);
        auto z = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(d);
        for (uint32_t row = 0; row < 32; row++) {
            if (int(q * 32 + row) >= first && int(q * 32 + row) < first + n) {
                if constexpr (TILE == 2048) {
                    for (uint32_t col = 0; col < 32; col++) {
                        uint32_t k = (row / 16) * 512 + (col / 16) * 256 + (row % 16) * 16 + col % 16;
                        z[k] = a[k];
                    }
                } else {
                    // BFP8 has one exponent per face row (16 values). A logical
                    // row owns two complete blocks, so no neighboring row is requantized.
                    auto ab = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(s);
                    auto zb = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(d);
                    for (uint32_t half = 0; half < 2; half++) {
                        uint32_t face = (row / 16) * 2 + half;
                        uint32_t exponent = face * 16 + row % 16;
                        zb[exponent] = ab[exponent];
                        for (uint32_t col = 0; col < 16; col++) {
                            uint32_t k = 64 + face * 256 + (row % 16) * 16 + col;
                            zb[k] = ab[k];
                        }
                    }
                }
            }
        }
        noc_async_write(d, dst.get_noc_addr(di), TILE);
        noc_async_write_barrier();
    }
}
