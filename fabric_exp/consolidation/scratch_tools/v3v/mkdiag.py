#!/usr/bin/env python3
"""mkdiag.py SRC_DIR DST_DIR CYCLES: diagnostic variant (never for shipping): E1 waits CYCLES core cycles after each
main-loop pass in which no receiver channel made progress."""
import shutil, sys, os
src, dst, cyc = sys.argv[1], sys.argv[2], int(sys.argv[3])
H = "tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_v3_receiver_erisc.hpp"
if os.path.exists(dst):
    shutil.rmtree(dst)
shutil.copytree(src, dst)
p = os.path.join(dst, H)
s = open(p).read()
name = os.path.basename(dst.rstrip("/"))
helper = f"""// DIAGNOSTIC ONLY ({name}; V3E1_NOTES §12.15). Never for shipping. After a main-loop pass in which no receiver
// channel made progress, E1 waits diag_idle_wait_cycles core cycles before its next pass. It reads only its own cycle
// counter (the mcycle CSR) while it waits: no stream register, L1, NIU or TX queue access. This tests whether E1's idle
// polling slows E0 through the tile's shared resources.
constexpr uint32_t diag_idle_wait_cycles = {cyc};

FORCE_INLINE uint32_t diag_core_cycles() {{
    uint32_t cycles;
    asm volatile("csrr %0, mcycle" : "=r"(cycles));
    return cycles;
}}

FORCE_INLINE void diag_idle_wait() {{
    const uint32_t start = diag_core_cycles();
    while (diag_core_cycles() - start < diag_idle_wait_cycles) {{
    }}
}}

"""
anchor = "// The main loop: every iterations_between_ctx_switch_and_teardown_checks passes, check termination, drain a deferred"
assert s.count(anchor) == 1
s = s.replace(anchor, helper + anchor)
a = "            rx_progress |= rx_channel_step<0>("
assert s.count(a) == 1
s = s.replace(a, "            uint32_t pass_progress = rx_channel_step<0>(")
b = "                rx_progress |= rx_channel_step<1>("
assert s.count(b) == 1
s = s.replace(b, "                pass_progress |= rx_channel_step<1>(")
# after the VC1 block closes (the #endif before the end of the for body), add the wait
loop_end = """#endif
        }
        if constexpr (RxChannel<0>::defers_flushed_atomics) {"""
assert s.count(loop_end) == 1, s.count(loop_end)
s = s.replace(loop_end, """#endif
            rx_progress |= pass_progress;
            if (!pass_progress) {
                diag_idle_wait();  // DIAGNOSTIC ONLY
            }
        }
        if constexpr (RxChannel<0>::defers_flushed_atomics) {""")
open(p, "w").write(s)
print("ok", dst)
