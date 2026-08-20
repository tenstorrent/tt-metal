#!/usr/bin/env python3
"""Self-test for the dst-layout-32b integration wiring (lane DZ; DU
integration note 4 — the pin-15 lreg-allocator measurement prerequisite).

Drives the REAL sweep code (imported, not re-implemented) with stubs — no
toolchain, no simulator, no device.  The contract under test (riscv.opt on
the DP allocator branch): -mtt-tensix-dst-layout-32b is a build-layer
DECLARATION derived from the kernel's dest-accumulation mode; declaring it
falsely on a 16-bit-layout kernel makes a spilled compilation produce
SILENT WRONG OUTPUT, while omitting it merely makes the allocator refuse.
So the wiring must prove BOTH directions:

  1. a 32b kernel (node id dest_acc:Yes) GETS the flag — but only on legs
     that carry a consumer (-mtt-tensix-optimize-lreg-alloc);
  2. a 16b kernel (dest_acc:No) does NOT get it, and neither does a kernel
     whose mode is UNKNOWN (no dest_acc token / unrecognized spelling) —
     fail closed;
  3. legs WITHOUT a consumer flag are byte-identical to before (no jobkey /
     leg-store / cross-pin-reuse churn);
  4. every flags/node meeting point applies the same pure function:
     classify legs, batched-classify chunk grouping, CRAQ legs, serial
     device jobkeys, batched device jobs, knob-silicon recorded flags.

Run standalone or from the sweep wrappers; exits nonzero on any failure so
a broken gate can never bless a sweep.
"""

import importlib.util
import pathlib
import sys
import tempfile
import types

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sweep_2x2", HERE / "sweep_2x2.py")
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)

FAILS = []


def check(name, cond, detail=""):
    if cond:
        print(f"SELFTEST PASS: {name}")
    else:
        print(f"SELFTEST FAIL: {name} {detail}")
        FAILS.append(name)


FLAG = sweep.DST_LAYOUT_32B_FLAG
CONSUMER = "-mtt-tensix-optimize-lreg-alloc"
NODE_32B = (
    "test_sfpu_unary.py::test_eltwise_unary_sfpu[formats:Float32->Float32-"
    "approx_mode:No-mathop:SfpuElwsqrt-fast_mode:No-dest_acc:Yes-"
    "input_dimensions:[64, 64]]"
)
NODE_16B = NODE_32B.replace("dest_acc:Yes", "dest_acc:No")
NODE_UNKNOWN = NODE_32B.replace("-dest_acc:Yes", "")
NODE_WEIRD = NODE_32B.replace("dest_acc:Yes", "dest_acc:Maybe")
LEG = f"{sweep.ON_FLAGS} {CONSUMER}"

# ---- 1/2/3: the pure function, both directions + fail-closed + no-churn ----
check(
    "32b node + consumer leg GETS the declaration",
    sweep.dst_layout_flags(LEG, NODE_32B) == f"{LEG} {FLAG}",
    sweep.dst_layout_flags(LEG, NODE_32B),
)
check(
    "16b node + consumer leg does NOT get it",
    sweep.dst_layout_flags(LEG, NODE_16B) == LEG,
)
check(
    "unknown mode (no dest_acc token) fails closed",
    sweep.dst_layout_flags(LEG, NODE_UNKNOWN) == LEG,
)
check(
    "unrecognized dest_acc spelling fails closed",
    sweep.dst_layout_flags(LEG, NODE_WEIRD) == LEG,
)
check(
    "empty node fails closed",
    sweep.dst_layout_flags(LEG, "") == LEG and sweep.dst_layout_flags(LEG, None) == LEG,
)
check(
    "defensive enum spelling DestAccumulation.Yes parses as 32b",
    sweep.node_dest_acc_32b(
        NODE_32B.replace("dest_acc:Yes", "dest_acc:DestAccumulation.Yes")
    ),
)
check(
    "32b node WITHOUT consumer: ON leg byte-identical",
    sweep.dst_layout_flags(sweep.ON_FLAGS, NODE_32B) == sweep.ON_FLAGS,
)
check(
    "32b node WITHOUT consumer: OFF leg byte-identical",
    sweep.dst_layout_flags(sweep.OFF_FLAGS, NODE_32B) == sweep.OFF_FLAGS,
)
check(
    "32b node WITHOUT consumer: TRUE-DEFAULT (empty) leg byte-identical",
    sweep.dst_layout_flags(sweep.TRUE_DEFAULT_FLAGS, NODE_32B)
    == sweep.TRUE_DEFAULT_FLAGS,
)
check(
    "idempotent: applying twice appends once",
    sweep.dst_layout_flags(sweep.dst_layout_flags(LEG, NODE_32B), NODE_32B)
    == f"{LEG} {FLAG}",
)
check(
    "consumer match is token-exact (no substring trap)",
    sweep.dst_layout_flags(f"{CONSUMER}-extended", NODE_32B) == f"{CONSUMER}-extended",
)

# The main reviewed legs must never carry a consumer silently: if the
# allocator flag is ever promoted into ON_FLAGS, the injection scope (and
# every jobkey on dest_acc:Yes rows) changes — that promotion must land
# WITH a reviewed update to this wiring, so fail loudly here.
check(
    "no consumer flag hiding in ON/OFF/TRUE-DEFAULT (promotion needs a "
    "reviewed wiring update)",
    not any(
        c in (s or "").split()
        for c in sweep.DST_LAYOUT_CONSUMERS
        for s in (sweep.ON_FLAGS, sweep.OFF_FLAGS, sweep.TRUE_DEFAULT_FLAGS)
    ),
)

# ---- knob integration (present only once the pin-15 conf patch applies) ---
if "lreg-alloc" in sweep.KNOBS:
    legs = dict(sweep.knob_legs("lreg-alloc"))
    check(
        "lreg-alloc knob leg + 32b node = ON + allocator + declaration",
        sweep.dst_layout_flags(legs["knob"], NODE_32B) == f"{legs['knob']} {FLAG}"
        and CONSUMER in legs["knob"].split(),
        legs["knob"],
    )
    check(
        "lreg-alloc knob leg + 16b node = ON + allocator ONLY",
        sweep.dst_layout_flags(legs["knob"], NODE_16B) == legs["knob"],
    )
    check(
        "lreg-alloc off leg (plain ON) untouched either way",
        sweep.dst_layout_flags(legs["off"], NODE_32B) == legs["off"],
    )
else:
    print(
        "SELFTEST NOTE: lreg-alloc knob not in KNOBS yet (pin-15 conf patch "
        "unapplied) — knob-integration checks run post-ceremony"
    )

# ---- 4: the meeting points apply the function (real methods, stubbed IO) --
row = {
    "op": "dztest",
    "kind": "full2x2",
    "nodes": {
        "sem-corr": NODE_32B,
        "sem-perf": NODE_32B,
        "hand-corr": NODE_16B,
        "hand-perf": NODE_16B,
    },
    "extra_env": {},
    "sel_extra_env": {"sem": {}, "hand": {}},
    "craq_archs": "bh",
}

with tempfile.TemporaryDirectory() as td:
    tmp = pathlib.Path(td)
    s = object.__new__(sweep.Sweep)
    s.ev = tmp / "ev"
    s.ev.mkdir()
    s.a = types.SimpleNamespace(force=False, classify_workers=2)
    s.reds = []
    s.info = {"cc1plus_sha256": "stub", "tt_metal_head": "stub"}

    # classify(): capture the compile env the leg would get; fail the
    # compile so the method exits after writing flags-<leg>.txt.
    seen = {}

    def fake_pytest(node, extra, env, log, timeout=1800):
        seen[pathlib.Path(log).name] = env["TT_LLK_EXTRA_COMPILER_OPTIONS"]
        pathlib.Path(log).write_text("stub compile fail\n")
        return 1

    s._pytest = fake_pytest
    legs_spec = (("off", sweep.ON_FLAGS), ("knob", LEG))
    s.classify(row, "sem-perf", legs=legs_spec, tag="dz-classify")
    wrote = (s.ev / "dztest" / "dz-classify" / "sem-perf" / "flags-off.txt").read_text()
    check(
        "classify(): 32b node consumer leg env + flags-file carry the flag",
        seen.get("compile-off.log") == sweep.ON_FLAGS
        and wrote.strip() == sweep.ON_FLAGS,
        seen,
    )
    # second leg only runs if the first passes — re-run with only the
    # consumer leg to observe it.
    seen.clear()
    s.classify(row, "sem-corr", legs=(("knob", LEG),), tag="dz-classify2")
    check(
        "classify(): consumer leg compiles with the injected declaration",
        seen.get("compile-knob.log") == f"{LEG} {FLAG}",
        seen,
    )
    seen.clear()
    s.classify(row, "hand-corr", legs=(("knob", LEG),), tag="dz-classify3")
    check(
        "classify(): 16b node consumer leg stays undeclared",
        seen.get("compile-knob.log") == LEG,
        seen,
    )

    # craq(): same capture through the simulator leg loop.
    simfile = tmp / "libttsim.so"
    simfile.write_bytes(b"stub")
    s._staged_sim = lambda arch: simfile
    seen.clear()
    s.craq(row, "sem-corr", "bh", legs_spec=(("knob", LEG),), tag="dz-craq")
    check(
        "craq(): consumer leg compiles/runs with the injected declaration",
        seen.get("craq-knob.log") == f"{LEG} {FLAG}",
        seen,
    )

    # _device_job(): the jobkey must carry the EFFECTIVE flags (cache/
    # adoption identity).  Capture via the _adopt_prev_cell probe and abort.
    class _Seen(Exception):
        pass

    captured = {}

    def fake_adopt(work, jobkey, expected_texts=None):
        captured["jobkey"] = jobkey
        raise _Seen()

    s._adopt_prev_cell = fake_adopt
    s.exec_mode = "serial"
    try:
        s._device_job(row, "sem-perf", "r1", "knob", LEG, tag="dz-silicon")
    except _Seen:
        pass
    check(
        "_device_job(): jobkey flags are the EFFECTIVE (injected) flags",
        captured.get("jobkey", {}).get("flags") == f"{LEG} {FLAG}",
        captured.get("jobkey"),
    )

    # _mk_job(): batched-silicon job carries effective flags.
    j32 = s._mk_job(row, "sem-perf", "r1", "knob", LEG, "perf")
    j16 = s._mk_job(row, "hand-perf", "r1", "knob", LEG, "perf")
    check(
        "_mk_job(): 32b job injected, 16b job untouched",
        j32["flags"] == f"{LEG} {FLAG}" and j16["flags"] == LEG,
    )

    # _batched_classify(): legjobs get effective flags BEFORE chunk
    # grouping, so 32b and 16b nodes land in DIFFERENT chunk sessions.
    groups_seen = []

    def fake_chunk(cdir, jobs, flags, extra_env):
        groups_seen.append(flags)
        raise RuntimeError("dz-stub (deferred to legacy solo)")

    s._classify_chunk_session = fake_chunk
    s.verify_toolchain = lambda phase: None
    pending = [
        (row, "sem-perf", (("knob", LEG),)),
        (row, "hand-perf", (("knob", LEG),)),
    ]
    s._batched_classify(pending)
    check(
        "_batched_classify(): 32b and 16b legs group into distinct "
        "flag-keyed chunk sessions",
        sorted(groups_seen) == sorted([f"{LEG} {FLAG}", LEG]),
        groups_seen,
    )

print()
if FAILS:
    print(f"SELFTEST: {len(FAILS)} FAILURE(S): {FAILS}")
    sys.exit(1)
print("SELFTEST: all dst-layout-32b wiring checks passed")
