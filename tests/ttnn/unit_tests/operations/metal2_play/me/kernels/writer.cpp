// Writer: dfb::out -> out tensor.  Optionally also dfb::out2 -> out2 tensor.
//
// STAGE_LOCAL additionally exercises a purely LOCAL L1->L1 copy through a self-looped
// staging DFB: dfb::out -> dfb::stage (local) -> DRAM.  The writer is bound as BOTH
// producer and consumer of dfb::stage.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/core_local_mem.h"
#include "api/dataflow/endpoints.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_tiles = get_arg(args::num_tiles);
    const uint32_t start_id = get_arg(args::start_id);

    DataflowBuffer dfb_out(dfb::out);
    Noc noc;
    const auto acc_out = TensorAccessor(tensor::out);
    const uint32_t tile_bytes = dfb_out.get_tile_size();

#if defined(GATE_IFDEF) || defined(GATE_ALWAYS)
    DataflowBuffer dfb_out2(dfb::out2);
    const auto acc_out2 = TensorAccessor(tensor::out2);
#endif
#ifdef GATE_ALWAYS
    constexpr bool emit_second = get_arg(args::emit_second);
#endif
#ifdef STAGE_LOCAL
    DataflowBuffer dfb_stage(dfb::stage);
#endif

    for (uint32_t i = start_id; i < start_id + num_tiles; ++i) {
        dfb_out.wait_front(1);

#ifdef STAGE_LOCAL
        // Local L1 -> L1 copy into a self-looped staging DFB.
        dfb_stage.reserve_back(1);
#if defined(LOCAL_VIA_SELF_READ)
        // Route D (the correct one): local L1 -> L1 through the NoC READ path.
        // The DFB stays a first-class typed DESTINATION (dst is resolved LOCAL_L1);
        // only the source is an address, which is inherent to a copy.
        // Coords MUST come from this Noc's id -- NOC 0 and NOC 1 have different spaces.
        UnicastEndpoint self_src;
        const uint8_t nid = noc.get_noc_id();
        noc.async_read(
            self_src,
            dfb_stage,
            tile_bytes,
            {.noc_x = my_x[nid], .noc_y = my_y[nid], .addr = dfb_out.get_read_ptr()},
            {.offset_bytes = 0});
        noc.async_read_barrier();
#elif defined(LOCAL_VIA_UNICAST_SELF)
        // Route B: NoC loopback -- unicast to my own (x,y). Requires extracting the
        // destination DFB's raw write pointer to build the endpoint.
        UnicastEndpoint self;
        noc.async_write(
            dfb_out,
            self,
            tile_bytes,
            {.offset_bytes = 0},
            {.noc_x = my_x[0], .noc_y = my_y[0], .addr = dfb_stage.get_write_ptr()});
        noc.async_write_barrier();
#else
        // Route C: no NoC at all -- typed local L1 views over both DFBs' pointers.
        CoreLocalMem<uint32_t> src(dfb_out.get_read_ptr());
        CoreLocalMem<uint32_t> dst(dfb_stage.get_write_ptr());
        for (uint32_t w = 0; w < tile_bytes / sizeof(uint32_t); ++w) {
            dst[w] = src[w];
        }
#endif
        dfb_stage.push_back(1);

        dfb_stage.wait_front(1);
        noc.async_write(dfb_stage, acc_out, tile_bytes, {.offset_bytes = 0}, {.page_id = i});
        noc.async_write_barrier();
        dfb_stage.pop_front(1);
#else
        noc.async_write(dfb_out, acc_out, tile_bytes, {.offset_bytes = 0}, {.page_id = i});
        noc.async_write_barrier();
#endif
        dfb_out.pop_front(1);

#ifdef GATE_IFDEF
        dfb_out2.wait_front(1);
        noc.async_write(dfb_out2, acc_out2, tile_bytes, {.offset_bytes = 0}, {.page_id = i});
        noc.async_write_barrier();
        dfb_out2.pop_front(1);
#endif
#ifdef GATE_ALWAYS
        if constexpr (emit_second) {
            dfb_out2.wait_front(1);
            noc.async_write(dfb_out2, acc_out2, tile_bytes, {.offset_bytes = 0}, {.page_id = i});
            noc.async_write_barrier();
            dfb_out2.pop_front(1);
        }
#endif
    }
}
