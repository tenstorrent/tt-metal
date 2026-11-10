// Megakernel integration stage1 stand-in engine (QWEN36_R16_MK_ENGINE_KIND=identity): copies the residual x_in
// (DRAM interleaved, bf16 TILE, replicated on every device) into the persistent per-bucket x_out page by page.
// Proves the stage1 x_in -> generic_op -> x_out path inside the verify trace. The op-by-op layer loop then runs on
// x_out, so the outputs are bit-identical to the flag off ONLY if this copy ran on every replay.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr auto in_args = TensorAccessorArgs<0>();
    constexpr auto out_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    uint32_t a = 0;
    const uint32_t in_addr = get_arg_val<uint32_t>(a++);
    const uint32_t out_addr = get_arg_val<uint32_t>(a++);
    const uint32_t page0 = get_arg_val<uint32_t>(a++);
    const uint32_t n_pages = get_arg_val<uint32_t>(a++);
    const uint32_t page_bytes = get_arg_val<uint32_t>(a++);
    const uint32_t batch = get_arg_val<uint32_t>(a++);  // pages per read burst (fits CB 24)
    const uint32_t buf = get_write_ptr(24);
    const auto src = TensorAccessor(in_args, in_addr, page_bytes);
    const auto dst = TensorAccessor(out_args, out_addr, page_bytes);
    for (uint32_t p = page0; p < page0 + n_pages; p += batch) {
        const uint32_t n = (page0 + n_pages - p) < batch ? (page0 + n_pages - p) : batch;
        for (uint32_t i = 0; i < n; ++i) {
            noc_async_read(src.get_noc_addr(p + i), buf + i * page_bytes, page_bytes);
        }
        noc_async_read_barrier();
        for (uint32_t i = 0; i < n; ++i) {
            noc_async_write(buf + i * page_bytes, dst.get_noc_addr(p + i), page_bytes);
        }
        noc_async_write_barrier();
    }
}
