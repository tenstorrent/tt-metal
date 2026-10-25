#!/usr/bin/env python3
"""The 2x2 test of what keeps E1's main loop stable under cold edits (RESULTS §97): twobytwo.py [make|build|report]

Two factors, on v3tk:
  P  per-round work (termination check, heartbeat, context-switch handshake): P0 out of line (v3t's noinline
     termination_requested and end_round), P1 inlined into the loop (FORCE_INLINE)
  M  a non-power-of-two constant multiply in the loop: M0 none, M1 the deferred atomic's t * 17 (jitter_edits trid17)
Inlining the per-round work also brings the termination check's mailbox index (launch[rd] * 144) into the loop, so the
P1 cells index the 8-entry launch mailbox through an offset table instead (tt_fabric_utils.h override): no multiply.
Two more cells keep the natural code: p1nat (P1 with its * 144) and p1nat_m1 (p1nat + t * 17; close to draft vA).
Each cell gets four cold edits (jitter_edits.py: coldmul, init, templates, os), and every cell and edit is built
offline for the hot E1 configs (W, N/S, 1D, VC1) at -O3, -O2 and -Os. A cell's loop is insulated when no cold edit
changes run_main_loop's code (fncmp.py: same, or moved = same code at another address).
"""
import concurrent.futures, glob, os, re, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = "/home/aagarwal/tt-metal"
VAR = TREE + "/fabric_exp/variants"
CELLS = VAR + "/v3tk_2x2"
OUT = "/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/twobytwo"
RX = "tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_v3_receiver_erisc.hpp"
UT = "tt_metal/fabric/hw/inc/tt_fabric_utils.h"
HOT = {"W": "10856749301170935304", "NS": "1539970155337268356", "1D": "10388088869406661095",
       "VC1": "9530214107180203457"}
EDITS = ("coldmul", "init", "templates", "os")
FNCMP = HERE + "/scratch_tools/v3t/fncmp.py"

INLINE = [
    (RX, "__attribute__((noinline)) bool termination_requested() {", "FORCE_INLINE bool termination_requested() {"),
    (RX, "__attribute__((noinline)) void end_round(uint16_t& fabric_heartbeat_counter) {",
     "FORCE_INLINE void end_round(uint16_t& fabric_heartbeat_counter) {"),
]
TABLE = [
    (UT, "    tt_l1_ptr launch_msg_t* const launch_msg = GET_MAILBOX_ADDRESS_DEV(launch[launch_msg_rd_ptr]);\n",
     "    // 2x2 probe (RESULTS §97): the entry's offset from a table, not launch_msg_rd_ptr * sizeof(launch_msg_t).\n"
     "    constexpr uint32_t s = sizeof(launch_msg_t);\n"
     "    static_assert(launch_msg_buffer_num_entries == 8);\n"
     "    static constexpr uint32_t launch_offsets[8] = {0, s, 2 * s, 3 * s, 4 * s, 5 * s, 6 * s, 7 * s};\n"
     "    tt_l1_ptr launch_msg_t* const launch_msg = reinterpret_cast<tt_l1_ptr launch_msg_t*>(\n"
     "        reinterpret_cast<uintptr_t>(GET_MAILBOX_ADDRESS_DEV(launch[0])) + launch_offsets[launch_msg_rd_ptr]);\n"),
]
TRID17 = [
    (RX, "pending_atomic_flush = (pending_atomic_flush & ~0xFu) | trid_b;",
     "pending_atomic_flush = 0x100u | (trid_b << 4) | trid_b;"),
]
CELL_EDITS = {
    "p0m0": [], "p0m1": TRID17, "p1m0": INLINE + TABLE, "p1m1": INLINE + TABLE + TRID17,
    "p1nat": INLINE, "p1nat_m1": INLINE + TRID17,
}


def make():
    if os.path.exists(CELLS):
        shutil.rmtree(CELLS)
    for cell, subs in CELL_EDITS.items():
        d = os.path.join(CELLS, cell)
        shutil.copytree(VAR + "/v3tk", d)
        for f, a, b in subs:
            p = os.path.join(d, f)
            if not os.path.exists(p):
                os.makedirs(os.path.dirname(p), exist_ok=True)
                shutil.copy(os.path.join(TREE, f), p)
            s = open(p).read()
            if s.count(a) != 1:
                sys.exit(f"{cell}: anchor found {s.count(a)} times in {f}: {a[:70]!r}")
            open(p, "w").write(s.replace(a, b))
        subprocess.run([sys.executable, HERE + "/jitter_edits.py", d, OUT + "/edits/" + cell, *EDITS], check=True,
                       stdout=subprocess.DEVNULL)
        print(cell, "+", ", ".join(sorted(os.listdir(OUT + "/edits/" + cell))))


def one(job):
    variant, out, opt = job
    hs = [h for k, h in HOT.items() if not (opt == "Os" and k == "VC1")]
    r = [subprocess.run([sys.executable, HERE + "/offline_build.py", "build", "--variant", variant, "--out", out,
                         "--opt", opt, "--hashes", ",".join(hs), "--jobs", "4"], capture_output=True, text=True)]
    if opt == "Os":  # VC1's -Os commands are not replayable (MPI-interleaved logs): its -O2 command at -Os
        r.append(subprocess.run([sys.executable, HERE + "/offline_build.py", "build", "--variant", variant, "--out",
                                 out + "_vc1", "--opt", "O2", "--set-opt", "Os", "--hashes", HOT["VC1"], "--jobs", "1"],
                                capture_output=True, text=True))
    return out, [x.stdout.strip().splitlines()[-1] if x.stdout.strip() else x.stderr[-300:] for x in r]


def build():
    jobs = []
    for cell in CELL_EDITS:
        for opt in ("O3", "O2", "Os"):
            jobs.append((os.path.join(CELLS, cell), f"{OUT}/b/{cell}_{opt}", opt))
            for e in EDITS:
                jobs.append((f"{OUT}/edits/{cell}/{e}", f"{OUT}/b/{cell}+{e}_{opt}", opt))
    with concurrent.futures.ThreadPoolExecutor(12) as ex:
        for out, res in ex.map(one, jobs):
            if any("built" not in x or re.search(r"built (\d+)/\1 ", x) is None for x in res):
                print("BUILD PROBLEM", out, res)
    print(f"{len(jobs)} builds done")


def elf(base, k, opt):
    d = base + ("_vc1" if (k == "VC1" and opt == "Os") else "")
    return f"{d}/{HOT[k]}/subordinate_active_erisc/subordinate_active_erisc.elf"


def loop_status(a, b):
    r = subprocess.run([sys.executable, FNCMP, a, b], capture_output=True, text=True).stdout
    for line in r.splitlines():
        m = re.match(r"(\S+)\s+_ZN14receiver_erisc13run_main_loop", line)
        if m:
            st = m.group(1)
            n = re.search(r"(\d+) differing", line)
            return st + (f"({n.group(1)})" if n and st.startswith("DIFF") else "")
    return "?"


def mults(cell):
    """Non-power-of-two constant multiplies in run_main_loop, from an -O3 GIMPLE dump of the N/S config."""
    out = f"{OUT}/dump/{cell}"
    subprocess.run([sys.executable, HERE + "/offline_build.py", "build", "--variant", os.path.join(CELLS, cell),
                    "--out", out, "--opt", "O3", "--hashes", HOT["NS"], "--flags=-fdump-tree-optimized", "--jobs", "1"],
                   capture_output=True)
    f = glob.glob(f"{out}/{HOT['NS']}/subordinate_active_erisc/*optimized")
    if not f:
        return "?"
    s = open(f[0]).read()
    for m in re.finditer(r";; Function (\S+) \(([^,]+),", s):
        st = m.end(); nx = s.find("\n;; Function", st); body = s[st:nx if nx > 0 else None]
        if "run_main_loop" in m.group(2):
            return sorted({int(c) for c in re.findall(r"= \S+ \* (\d+);", body) if int(c) & (int(c) - 1)})
    return "?"


def report():
    print("cell        loop's non-pow2 multiplies (N/S, -O3)")
    for cell in CELL_EDITS:
        print(f"  {cell:10s}{mults(cell)}")
    print("\nrun_main_loop under each cold edit vs the cell's own build (W NS 1D VC1); same/moved = unchanged code")
    for cell in CELL_EDITS:
        for opt in ("O3", "O2", "Os"):
            parts = []
            for e in EDITS:
                st = [loop_status(elf(f"{OUT}/b/{cell}_{opt}", k, opt), elf(f"{OUT}/b/{cell}+{e}_{opt}", k, opt))
                      for k in HOT]
                parts.append(f"{e}: " + " ".join(st))
            print(f"{cell:9s} {opt}  " + " | ".join(parts))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("make", "all"):
        make()
    if cmd in ("build", "all"):
        build()
    if cmd in ("report", "all"):
        report()
