"""synth_overlays_tx.py VARIANT...: sender headroom-probe copies of each variant (study only; V4_SENDER_OWNERSHIP.md §7.4),
in fabric_exp/variants:
  <v>_tx1  one extra value carried from pass to pass in sender::run (live across the pass, opaque to GCC)
  <v>_tx2  two such values
  <v>_twk  a worker-sized state: five values updated once per pass (a line VC's worker channel state)
  <v>_trr  a remote receiver's progress: two values read and updated together once per pass (a line VC's far channel;
           per pass rather than per send: on a busy row the first VC sends about once per pass)
The values are declared at the top of sender::run and updated right after its pass() call, so they are live across
every pass. KINDS=tx1,tx2,twk,trr picks the kinds."""
import os, re, shutil, sys

V = "/home/aagarwal/tt-metal/fabric_exp/variants"
R = "tt_metal/fabric/hw/inc/edm_fabric/router/router_sender.hpp"
HELPERS = """// ---------------------------------------------------------------------------------------------------------------------
// Headroom probe (study only, not a design): synthetic state that the guard's headroom test adds to measure what moves.
// ---------------------------------------------------------------------------------------------------------------------

// Values carried from pass to pass: in registers at each pass's end, opaque to GCC (it cannot fold them).
FORCE_INLINE void synthetic_carried(uint32_t& value) { asm volatile("" : "+r"(value)); }
FORCE_INLINE void synthetic_carried_pair(uint32_t& a, uint32_t& b) { asm volatile("" : "+r"(a), "+r"(b)); }
FORCE_INLINE void synthetic_carried_five(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d, uint32_t& e) {
    asm volatile("" : "+r"(a), "+r"(b), "+r"(c), "+r"(d), "+r"(e));
}

"""
DECLS = {
    "tx1": ["uint32_t synthetic0 = 1;"],
    "tx2": ["uint32_t synthetic0 = 1;", "uint32_t synthetic1 = 2;"],
    "twk": [f"uint32_t synthetic{k} = {k + 1};" for k in range(5)],
    "trr": ["uint32_t synthetic0 = 1;", "uint32_t synthetic1 = 2;"],
}
UPDATES = {
    "tx1": ["synthetic_carried(synthetic0);"],
    "tx2": ["synthetic_carried(synthetic0);", "synthetic_carried(synthetic1);"],
    "twk": ["synthetic_carried_five(synthetic0, synthetic1, synthetic2, synthetic3, synthetic4);"],
    "trr": ["synthetic_carried_pair(synthetic0, synthetic1);"],
}
kinds = [k for k in ("tx1", "tx2", "twk", "trr") if k in os.environ.get("KINDS", "tx1,tx2,twk,trr").split(",")]
for v in sys.argv[1:]:
    for kind in kinds:
        dst = f"{V}/{v}_{kind}"
        if os.path.exists(dst):
            sys.exit(f"{dst} exists")
        shutil.copytree(f"{V}/{v}", dst, symlinks=True)
        p = f"{dst}/{R}"
        s = open(p).read()
        m = re.search(r"\n(__attribute__\(\(noinline\)\) void run\([^\n]*\{\n)", s)
        assert m, (v, "run")
        start = m.start(1)
        body = m.end(1)
        t = s.rfind("\ntemplate <", 0, start)  # the run's template header, if it has one, comes first
        if t >= 0 and "\n" not in s[t + 1:start].rstrip("\n").split("\n", 1)[-1] and s.count("\n", t + 1, start) == 1:
            start = t + 1
        s = s[:start] + HELPERS + s[start:body] + "".join(f"    {d}\n" for d in DECLS[kind]) + s[body:]
        r = s.index("__attribute__((noinline)) void run(")
        c = re.search(r"\n( +)pass\([^\n]*\);\n", s[r:])
        assert c, (v, "pass call")
        j = r + c.end()
        s = s[:j] + "".join(f"{c.group(1)}{u}\n" for u in UPDATES[kind]) + s[j:]
        open(p, "w").write(s)
        print("made", dst)
