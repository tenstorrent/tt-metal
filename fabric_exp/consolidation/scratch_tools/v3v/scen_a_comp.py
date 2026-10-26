#!/usr/bin/env python3
"""scen_a_comp.py DIS ELF [SRC_VARIANT_DIR]: v3a-layout E1 (generic receiver step) write and write + completion
paths (compu: one credit of n; compm: 4 credit stores), as rxscen.py's compu/compm."""
import re, sys, subprocess, os
dis, elf = sys.argv[1:3]
vd = sys.argv[3] if len(sys.argv) > 3 else "/home/aagarwal/tt-metal/fabric_exp/variants/v3a"
src = open(vd + "/fabric_erisc_router_v3.cpp").read().split("\n")
hdr = open(vd + "/fabric_edm_packet_transmission.hpp").read().split("\n")
def find(lines, pat, start=0, nth=0):
    hits = [i + 1 for i, l in enumerate(lines) if pat in l and i + 1 > start]
    return hits[nth]
R = "fabric_erisc_router_v2.cpp"; T = "fabric_edm_packet_transmission.hpp"; FC = "fabric_router_flow_control.hpp"
def rng(f, a, b): return f"{f}:{a}-{b}"
impl = find(hdr, "execute_chip_unicast_to_local_chip_impl(")
h_write = (find(hdr, "case tt::tt_fabric::NocSendType::NOC_UNICAST_WRITE: {", impl), find(hdr, "case tt::tt_fabric::NocSendType::NOC_UNICAST_ATOMIC_INC: {", impl) - 1)
els = find(hdr, "FORCE_INLINE void execute_local_semaphore_inc(")
loop = find(src, "for (size_t i = 0; i < iterations_between_ctx_switch_and_teardown_checks; i++) {")
step = find(src, "receiver_forward_timer.set_should_dump(unwritten_packets);")
pend = find(src, "if (pending_atomic_flush) {", step)
pend_end = find(src, "if constexpr (is_2d_fabric) {", pend) - 1
fd = find(src, "uint8_t forward_2d_dispatch(")
fwd = [rng(R, c, c + 6) for c in [find(src, "forward_payload_to_downstream_edm<enable_deadlock_avoidance, stateful_api>(", fd, k) for k in range(4)]]
comp = find(src, "const uint32_t batch = uncompleted > batch_target ? batch_target : uncompleted;")
ifprog = find(src, "if (!progress && batch > 1) {", comp)
av = [rng(R, pend + 1, pend_end), "fabric_router_adapter.hpp:1-2000", rng(T, els, els + 34)] + fwd
need = [rng(T, *h_write)]
L = open(dis).read().split("\n")
ins = []; cur = None
for l in L:
    m = re.match(r"^(/\S+):(\d+)", l)
    if m: cur = (m.group(1).split("/")[-1], int(m.group(2))); continue
    m = re.match(r"^\s+([0-9a-f]+):\t[0-9a-f]+\s+(\S+)\s*(.*)$", l)
    if m: ins.append((int(m.group(1), 16), m.group(2), m.group(3), cur))
cands = []
for ad, op, arg, loc in ins:
    if loc and loc[0] == R and loc[1] == loop and op in ("bnez", "bne", "bgeu", "bltu", "beqz"):
        t = int(re.sub(r" <[^>]*>", "", arg).split(",")[-1], 16)
        if t < ad and ad - t > 16: cands.append((ad, t))
WALK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "walk.py")
for sc in sys.argv[4:] or ["write", "compu", "compm"]:
    xa = ["--avoid", ",".join(av), "--need", ",".join(need), "--elf", elf, "--eq", rng(R, pend, pend), "--issues", "1"]
    if sc == "write":
        xa[1] += "," + rng(R, comp, comp + 40)
    elif sc == "compu":
        xa += ["--cnt", f"{FC}:40", "--cntmin", "1"]
    else:
        xa += ["--cnt", f"{FC}:40", "--cntmin", "4"]
    best = None
    for end, head in cands:
        r = subprocess.run(["python3", WALK, dis, f"{head:x}", f"{end:x}"] + xa + (["-v"] if "-v" in sys.argv else []), capture_output=True, text=True).stdout
        last = r.strip().splitlines()[-1] if r.strip() else "ERR"
        m = re.match(r"instr (\d+)", last)
        if m and (best is None or int(m.group(1)) < best[0]): best = (int(m.group(1)), r)
    if "-v" in sys.argv and best: print(best[1])
    print(f"{sc:6s} {best[1].strip().splitlines()[-1] if best else 'no path'}")
