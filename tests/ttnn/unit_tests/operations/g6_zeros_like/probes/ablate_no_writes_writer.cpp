// Ablation of g6_zeros_like_writer.cpp: same CT/RT args, accessor setup and final barrier; no NoC payload.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t page_size = get_compile_time_arg_val(1);
    constexpr auto output_args = TensorAccessorArgs<5>();
    Noc noc;
    volatile uint32_t start_page = get_arg_val<uint32_t>(0);
    volatile uint32_t num_pages = get_arg_val<uint32_t>(1);
    const auto output = TensorAccessor(output_args, get_common_arg_val<uint32_t>(0), page_size);
    volatile uint64_t sink = output.get_noc_addr(start_page + num_pages);
    (void)sink;
    {
        DeviceZoneScopedN("g6zl-write-barrier");
        noc.async_write_barrier();
    }
}
