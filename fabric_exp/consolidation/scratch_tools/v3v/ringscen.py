#!/usr/bin/env python3
"""ringscen.py VARIANT_DIR DIS ELF [scen...] [-v]: per-packet paths of the generic 1D receiver step with first-level ack
(ring E1). Scenarios:
  wf      WRITE_AND_FORWARD write: first-level ack + forward (call) + local write + one completion, in one pass
  wfnoack WRITE_AND_FORWARD write without the ack and completion (just the packet)
  w       WRITE_ONLY write + ack + completion
  fwd     FORWARD_ONLY + ack + completion
Paths go from the head of the main loop's inner loop to its back edge."""
import os, re, subprocess, sys
vdir, dis, elf = sys.argv[1:4]
scens = [x for x in sys.argv[4:] if not x.startswith("-")] or ["wf", "wfnoack", "w", "fwd"]
src = open(os.path.join(vdir, "fabric_erisc_router_v3.cpp")).read().split("\n")
hdr = open(os.path.join(vdir, "fabric_edm_packet_transmission.hpp")).read().split("\n")
R = "fabric_erisc_router_v2.cpp"; T = "fabric_edm_packet_transmission.hpp"; FC = "fabric_router_flow_control.hpp"
def find(lines, pat, start=0, nth=0):
    hits = [i + 1 for i, l in enumerate(lines) if pat in l and i + 1 > start]
    if not hits: raise SystemExit(f"anchor not found: {pat!r}")
    return hits[nth]
def rng(f, a, b): return f"{f}:{a}-{b}"
ml = find(src, "FORCE_INLINE void run_fabric_edm_main_loop(")
loop = find(src, "for (size_t i = 0; i < iterations_between_ctx_switch_and_teardown_checks; i++) {", ml)
rfp = find(src, "FORCE_INLINE void receiver_forward_packet(")
wo = find(src, "case LowLatencyFields::WRITE_ONLY:", rfp)
fo = find(src, "case LowLatencyFields::FORWARD_ONLY: {", rfp)
wf = find(src, "case LowLatencyFields::WRITE_AND_FORWARD: {", rfp)
sp0 = find(src, "if (packed.noc_send_type == tt::tt_fabric::NocSendType::NOC_SPARSE_MCAST_WRITE) [[unlikely]] {", wf)
sp1 = find(src, "} else {", sp0)
wf_fwd = find(src, "forward_payload_to_downstream_edm<", sp1)
rfp_end = find(src, "#endif  // !FABRIC_2D", rfp)
impl = find(hdr, "execute_chip_unicast_to_local_chip_impl(")
h_write = (find(hdr, "case tt::tt_fabric::NocSendType::NOC_UNICAST_WRITE: {", impl), find(hdr, "case tt::tt_fabric::NocSendType::NOC_UNICAST_ATOMIC_INC: {", impl) - 1)
fpd = find(hdr, "forward_payload_to_downstream_edm(")
ack = 48; comp = 33
L = open(dis).read().split("\n")
ins = []; cur = None
for l in L:
    m = re.match(r"^(/\S+):(\d+)", l)
    if m: cur = (m.group(1).split("/")[-1], int(m.group(2))); continue
    m = re.match(r"^\s+([0-9a-f]+):\t[0-9a-f]+\s+(\S+)\s*(.*)$", l)
    if m: ins.append((int(m.group(1), 16), m.group(2), m.group(3), cur))
cands = []
for ad, op, arg, loc in ins:
    if loc and loc[0] == R and loc[1] == loop and op in ("bnez", "bne", "bgeu", "bltu", "beqz", "beq", "bltz", "bgez"):
        t = int(re.sub(r" <[^>]*>", "", arg).split(",")[-1], 16)
        if t < ad and ad - t > 16: cands.append((ad, t))
WALK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "walk.py")
for sc in scens:
    av = [rng(R, sp0, sp1 - 1)]
    need = []
    if sc in ("wf", "wfnoack"):
        av += [rng(R, wo, fo - 1 + 6)]
        need += [rng(T, *h_write)]
        need += ["fabric_router_adapter.hpp:1-3000,edm_fabric_worker_adapters.hpp:1-3000"]
    elif sc == "w":
        av += [rng(R, fo, rfp_end)]
        need += [rng(T, *h_write)]
    elif sc == "fwd":
        av += [rng(R, wo, fo - 1), rng(R, wf, rfp_end), rng(T, impl - 5, impl + 200)]
        need += ["fabric_router_adapter.hpp:1-3000,edm_fabric_worker_adapters.hpp:1-3000"]
    xa = ["--avoid", ",".join(av), "--need", ";".join(need), "--elf", elf]
    if sc != "wfnoack":
        xa += ["--cnt", f"{FC}:{ack},{FC}:{comp}", "--cntmin", "2"]
    else:
        xa[1] += f",{FC}:1-200"
    best = None
    for end, head in cands:
        r = subprocess.run(["python3", WALK, dis, f"{head:x}", f"{end:x}"] + xa + (["-v"] if "-v" in sys.argv else []), capture_output=True, text=True).stdout
        last = r.strip().splitlines()[-1] if r.strip() else "ERR"
        m = re.match(r"instr (\d+)", last)
        if m and (best is None or int(m.group(1)) < best[0]): best = (int(m.group(1)), r)
    if "-v" in sys.argv and best: print(best[1])
    print(f"{sc:8s} {best[1].strip().splitlines()[-1] if best else 'no path'}")
