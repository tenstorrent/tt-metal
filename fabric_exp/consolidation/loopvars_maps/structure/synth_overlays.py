"""synth_overlays.py VARIANT...: headroom-probe copies of each variant (study only), in fabric_exp/variants:
  <v>_x1  one extra value carried from pass to pass (live across the channel steps, opaque to GCC)
  <v>_x2  two such values
  <v>_fd  a flush-defer-sized record per channel in memory, tested at the step's start, settled out of line"""
import os, re, shutil, sys
V = "/home/aagarwal/tt-metal/fabric_exp/variants"
R = "tt_metal/fabric/hw/inc/edm_fabric/router/router_receiver.hpp"
HELPERS = """// ---------------------------------------------------------------------------------------------------------------------
// Headroom probe (study only, not a design): synthetic state that the guard's headroom test adds to measure what moves.
// ---------------------------------------------------------------------------------------------------------------------

// A value carried from pass to pass: in a register at each pass's end, opaque to GCC (it cannot fold it).
FORCE_INLINE void synthetic_carried(uint32_t& value) { asm volatile("" : "+r"(value)); }

// A flush-defer-sized record per channel (FLUSH_DEFER_DESIGN.md: the atomic's NoC address and value, and whether it
// waits), in memory, tested at the step's start; the settle is out of line.
struct SyntheticPendingAtomic {
    uint64_t noc_addr;
    uint32_t value;
    uint32_t waiting;
};
inline SyntheticPendingAtomic synthetic_pending[MAX_NUM_RECEIVER_CHANNELS];
__attribute__((noinline)) bool synthetic_settle(SyntheticPendingAtomic& atomic) {
    asm volatile("" : "+m"(atomic));
    return atomic.waiting == 0;
}
template <size_t CH>
FORCE_INLINE bool synthetic_atomic_waits() {
    keep_in_memory(synthetic_pending[CH]);
    return synthetic_pending[CH].waiting != 0 && !synthetic_settle(synthetic_pending[CH]);
}

"""
kinds = [k for k in ("x1", "x2", "fd") if os.environ.get("KINDS", "x1,x2,fd").count(k)]
for v in sys.argv[1:]:
    for kind in kinds:
        dst = f"{V}/{v}_{kind}"
        if os.path.exists(dst):
            sys.exit(f"{dst} exists")
        shutil.copytree(f"{V}/{v}", dst, symlinks=True)
        p = f"{dst}/{R}"
        s = open(p).read()
        anchor = "// One step of receiver channel CH (`arrived_raw`"
        if s.count(anchor) != 1:  # keyed by VC (v4rxf1 on)
            anchor = "// One step of VC's receiver channel (`arrived_raw`"
        assert s.count(anchor) == 1, (v, "anchor")
        s = s.replace(anchor, HELPERS + anchor)
        if kind in ("x1", "x2"):
            decl = "    bool credit_frame_due = false;  // the credit counts changed since the last frame\n"
            i = s.index(decl, s.index("void run(DownstreamConnections& downstreams)"))
            n = 1 if kind == "x1" else 2
            s = s[:i + len(decl)] + "".join(f"    uint32_t synthetic{k} = {k + 1};\n" for k in range(n)) + s[i + len(decl):]
            m = re.search(r"\n(            pass\([^\n]*\);\n)", s[s.index("void run(DownstreamConnections& downstreams)"):])
            assert m, (v, "pass call")
            j = s.index("void run(DownstreamConnections& downstreams)") + m.end()
            s = s[:j] + "".join(f"            synthetic_carried(synthetic{k});\n" for k in range(n)) + s[j:]
        else:
            forms = [("CH", "    const bool progress = receive<CH>(cursor, sources, downstreams, arrived, me);",
                      "    const bool progress = !atomic_waits && receive<CH>(cursor, sources, downstreams, arrived, me);"),
                     ("VC", "    const bool handled_packet = receive<VC>(progress, sources, downstreams, arrived, my_chip);",
                      "    const bool handled_packet = !atomic_waits && receive<VC>(progress, sources, downstreams, arrived, my_chip);"),
                     ("VC", "    const bool handled_packet = receive<VC>(progress, sources, downstreams, arrived, route_loaded, my_chip);",
                      "    const bool handled_packet = !atomic_waits && receive<VC>(progress, sources, downstreams, arrived, route_loaded, my_chip);")]
            key, call, new_call = next(f for f in forms if s.count(f[1]) == 1)
            s = s.replace(call, new_call)
            a = "    const uint32_t arrived = arrived_raw & stream_count_mask;\n"
            k = s.index(a, s.index("FORCE_INLINE void step("))
            s = s[:k + len(a)] + f"    const bool atomic_waits = synthetic_atomic_waits<{key}>();\n" + s[k + len(a):]
        open(p, "w").write(s)
        print("made", dst)
