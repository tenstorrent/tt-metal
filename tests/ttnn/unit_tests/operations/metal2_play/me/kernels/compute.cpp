// Compute: out = a*b.  Optionally ALSO publishes the same tile to a second, conditionally
// bound DFB (dfb::out2) -- the "optional CB" experiment.
//
// Three gating styles are selected by host-emitted defines:
//   GATE_IFDEF   : #ifdef around every dfb::out2 reference          (upstream-sanctioned)
//   GATE_ALWAYS  : dfb::out2 always bound; runtime bool decides     (pays L1 unconditionally)
//   (neither)    : unfused
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_tiles = get_arg(args::num_tiles);

    DataflowBuffer dfb_a(dfb::in_a);
    DataflowBuffer dfb_b(dfb::in_b);
    DataflowBuffer dfb_out(dfb::out);
#if defined(GATE_IFDEF) || defined(GATE_ALWAYS)
    DataflowBuffer dfb_out2(dfb::out2);
#endif
#ifdef GATE_ALWAYS
    constexpr bool emit_second = get_arg(args::emit_second);  // CTA
#endif

    binary_op_init_common(dfb::in_a, dfb::in_b, dfb::out);
    mul_tiles_init(dfb::in_a, dfb::in_b);

    for (uint32_t i = 0; i < num_tiles; ++i) {
        dfb_a.wait_front(1);
        dfb_b.wait_front(1);

        tile_regs_acquire();
        mul_tiles(dfb::in_a, dfb::in_b, 0, 0, 0);
        tile_regs_commit();

        dfb_a.pop_front(1);
        dfb_b.pop_front(1);

        dfb_out.reserve_back(1);
        tile_regs_wait();
        pack_tile(0, dfb::out, 0);

#ifdef GATE_IFDEF
        dfb_out2.reserve_back(1);
        pack_tile(0, dfb::out2, 0);
#endif
#ifdef GATE_ALWAYS
        if constexpr (emit_second) {
            dfb_out2.reserve_back(1);
            pack_tile(0, dfb::out2, 0);
        }
#endif
        tile_regs_release();
        dfb_out.push_back(1);
#ifdef GATE_IFDEF
        dfb_out2.push_back(1);
#endif
#ifdef GATE_ALWAYS
        if constexpr (emit_second) {
            dfb_out2.push_back(1);
        }
#endif
    }
}
