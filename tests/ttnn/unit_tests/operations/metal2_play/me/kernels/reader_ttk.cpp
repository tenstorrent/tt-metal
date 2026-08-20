// Reader authored in the TT_KERNEL "1st world args" syntax: CTAs are template
// (non-type) parameters, RTAs/CRTAs are function parameters. The JIT generates
// kernel_main() from this signature.
//
// TTK_UNBOUND_TOKEN: names dfb::out2 inside an `if constexpr` branch that is
// discarded for this instantiation, WITHOUT the host binding it. If NTTP CTAs
// fixed the conditional-binding problem, this would compile. It does not.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

template <uint32_t tiles_per_iter, uint32_t touch_optional>         // CTAs (uint32_t ONLY -- bool is rejected)
TT_KERNEL void read_pairs(uint32_t num_tiles, uint32_t start_id) {  // RTAs
    DataflowBuffer dfb_a(dfb::in_a);
    DataflowBuffer dfb_b(dfb::in_b);
    Noc noc;
    const auto acc_a = TensorAccessor(tensor::a);
    const auto acc_b = TensorAccessor(tensor::b);
    const uint32_t tile_bytes = dfb_a.get_tile_size();

#ifdef TTK_UNBOUND_TOKEN
    if constexpr (touch_optional != 0) {
        // Non-dependent name lookup happens at template DEFINITION time, so this
        // is an error even though `touch_optional` is false for this instantiation.
        DataflowBuffer dfb_opt(dfb::out2);
        (void)dfb_opt.get_tile_size();
    }
#endif

    // Real compile-time branching on an NTTP: this part DOES work.
    for (uint32_t i = start_id; i < start_id + num_tiles; i += tiles_per_iter) {
        if constexpr (tiles_per_iter == 1) {
            dfb_a.reserve_back(1);
            dfb_b.reserve_back(1);
            noc.async_read(acc_a, dfb_a, tile_bytes, {.page_id = i}, {.offset_bytes = 0});
            noc.async_read(acc_b, dfb_b, tile_bytes, {.page_id = i}, {.offset_bytes = 0});
            noc.async_read_barrier();
            dfb_a.push_back(1);
            dfb_b.push_back(1);
        } else {
            // unrolled path -- only instantiated when tiles_per_iter != 1
            for (uint32_t k = 0; k < tiles_per_iter && (i + k) < start_id + num_tiles; ++k) {
                dfb_a.reserve_back(1);
                dfb_b.reserve_back(1);
                noc.async_read(acc_a, dfb_a, tile_bytes, {.page_id = i + k}, {.offset_bytes = 0});
                noc.async_read(acc_b, dfb_b, tile_bytes, {.page_id = i + k}, {.offset_bytes = 0});
                noc.async_read_barrier();
                dfb_a.push_back(1);
                dfb_b.push_back(1);
            }
        }
    }
}
