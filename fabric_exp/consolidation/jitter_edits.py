#!/usr/bin/env python3
"""Jitter-band edits for the v3 router (RESULTS §96): jitter_edits.py BASE_VARIANT OUT_DIR [EDIT...]

Writes OUT_DIR/<edit>/ for each edit below, each a copy of BASE with one change a contributor might make without
expecting a performance effect. None changes what the router does per packet. Every anchor must match exactly once,
so an edit never silently becomes a no-op. With EDIT names, only those are written. Hardware runs of these builds give the spread of cycles/packet that
ordinary edits cause (the "jitter band"), at -O3 and -Os.

  reorder      perturb.py: swap two members of E0's sender channel state
  statefield   perturb.py: one unused member in each `// STABILITY_PROBE_STATE` struct (E0 and E1 state)
  hb           both heartbeats every 128 rounds instead of 64 (E0's main loop, E1's out-of-line end_round)
  coldmul      an init-time helper with constant multiplies and a division, called once on each ERISC (cold code)
  rxreorder    swap two members of E1's receiver channel state (RxChannelState)
  cursorfield  one more member in E1's DownstreamCursor (32 -> 40 bytes with alignment; the static_assert updated as a contributor
               would), which makes the runtime index into the downstream cursors a multiply by 40
  trid17       the deferred atomic's one-trid-left value written out as 0x100 | (t << 4) | t (the same value), which GCC
               turns into t * 17 inside E1's loop. Not one of the jitter runs: a probe of the multiply gotcha.
  init         v3r2's init cleanup (plain pointers in init, one shared handshake), via derive_edits.py
  templates    v3r2's templates cleanup (plain functions instead of dummy templates), via derive_edits.py
  os, heartbeat, param, bittest: v3r2's other cleanups, by name only (not in the default set)
"""
import os, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
FX = os.path.dirname(HERE)
RX = "tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_v3_receiver_erisc.hpp"
TX = "tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_v3_sender_erisc.hpp"
RT = "fabric_erisc_router_v3.cpp"

COLDMUL_FN = """
// An init-time value a contributor might compute (cold code): constant multiplies and a division, run once.
__attribute__((noinline)) uint32_t init_layout_checksum(uint32_t x) {
    return x * 17 + x * 40 + x * 3 + x * 144 + x / 7;
}
"""
COLDMUL_CALL = """    {
        volatile uint32_t layout_seed = MY_ETH_CHANNEL;
        volatile uint32_t layout_checksum = init_layout_checksum(layout_seed);
        (void)layout_checksum;
    }
    set_eth_txq_seq_update_timeouts();
"""

EDITS = {
    "hb": [
        (TX, "if ((++fabric_heartbeat_counter & 0x3F) == 0) {", "if ((++fabric_heartbeat_counter & 0x7F) == 0) {", 1),
        (RX, "if ((++fabric_heartbeat_counter & 0x3F) == 0) {", "if ((++fabric_heartbeat_counter & 0x7F) == 0) {", 1),
    ],
    "coldmul": [
        (RT, "__attribute__((optimize(\"Os\"))) void set_eth_txq_seq_update_timeouts() {",
         COLDMUL_FN + "\n__attribute__((optimize(\"Os\"))) void set_eth_txq_seq_update_timeouts() {", 1),
        (RT, "    set_eth_txq_seq_update_timeouts();\n", COLDMUL_CALL, 2),
    ],
    "rxreorder": [
        (RX, "    uint32_t avail;           // packets known to be in the ring and not yet retired (lower bound of the register)\n"
             "    uint32_t slot_addr;       // header address of the next slot to retire\n",
         "    uint32_t slot_addr;       // header address of the next slot to retire\n"
         "    uint32_t avail;           // packets known to be in the ring and not yet retired (lower bound of the register)\n", 1),
    ],
    "trid17": [
        (RX, "pending_atomic_flush = (pending_atomic_flush & ~0xFu) | trid_b;",
         "pending_atomic_flush = 0x100u | (trid_b << 4) | trid_b;", 1),
    ],
    "cursorfield": [
        (RX, "    uint64_t credit_noc_addr;\n};\n", "    uint64_t credit_noc_addr;\n    uint32_t connection_epoch;\n};\n", 1),
        (RX, "static_assert(sizeof(DownstreamCursor) == 32);", "static_assert(sizeof(DownstreamCursor) == 40);", 1),
    ],
}


def apply(base, out, name, subs):
    d = os.path.join(out, name)
    shutil.copytree(base, d)
    for f, a, b, n in subs:
        p = os.path.join(d, f)
        s = open(p).read()
        if s.count(a) != n:
            sys.exit(f"{name}: anchor found {s.count(a)} times (want {n}) in {f}: {a[:70]!r}")
        open(p, "w").write(s.replace(a, b))


def main():
    base, out = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    want = set(sys.argv[3:]) or (set(EDITS) - {"trid17"}) | {"reorder", "statefield", "init", "templates"}
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    for name, subs in EDITS.items():
        if name in want:
            apply(base, out, name, subs)
    for e in [e for e in ("reorder", "statefield") if e in want]:
        subprocess.run([sys.executable, os.path.join(HERE, "perturb.py"), base, out, e], check=True,
                       stdout=subprocess.DEVNULL)
    tmp = out + "/.tmp"
    for e in [e for e in ("init", "templates", "os", "heartbeat", "param", "bittest") if e in want]:
        shutil.rmtree(tmp, ignore_errors=True)
        subprocess.run([sys.executable, os.path.join(HERE, "v3r_renames.py"), "join", base, tmp + "/j"], check=True,
                       stdout=subprocess.DEVNULL)
        subprocess.run([sys.executable, os.path.join(FX, "variants/v3t/derive_edits.py"), tmp + "/j", tmp + "/k", e],
                       check=True, stdout=subprocess.DEVNULL)
        subprocess.run([sys.executable, os.path.join(HERE, "v3r_renames.py"), "split", tmp + "/k", out + "/" + e],
                       check=True, stdout=subprocess.DEVNULL)
    shutil.rmtree(tmp, ignore_errors=True)
    for d in sorted(os.listdir(out)):
        for f in ("derive_edits.py",):
            p = os.path.join(out, d, f)
            if os.path.exists(p):
                os.remove(p)
        print(d)


if __name__ == "__main__":
    main()
