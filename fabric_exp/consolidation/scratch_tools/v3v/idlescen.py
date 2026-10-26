#!/usr/bin/env python3
"""rxscen.py VARIANT_DIR DIS ELF [scenario...] [-v] [-p]: per-packet path counts of receiver_erisc (v3sb layout:
the E1 program in tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_v3_receiver_erisc.hpp).

Scenarios: write, fwd, scatter, atomic, fused, fusedfl, compu (write + a completion of 4 from one sender, i.e. the
shortest path with one credit store), compm (write + a completion of 4 slots credited one by one: 4 credit stores).
Paths go from the head of the main loop's inner loop to its back edge (shortest over all back edges of that line);
see walk.py for the constraint semantics."""
import os, re, subprocess, sys

vdir, dis, elf = sys.argv[1:4]
scens = [x for x in sys.argv[4:] if not x.startswith("-")] or ["write", "fwd", "atomic", "fused", "fusedfl", "scatter"]
RXP = os.path.join(vdir, "tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_v3_receiver_erisc.hpp")
rx = open(RXP).read().split("\n")
hdr = open(os.path.join(vdir, "fabric_edm_packet_transmission.hpp")).read().split("\n")
RX = "fabric_erisc_router_v3_receiver_erisc.hpp"
T = "fabric_edm_packet_transmission.hpp"
FC = "fabric_router_flow_control.hpp"
WALK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "walk.py")


def find(lines, pat, start=0, nth=0):
    hits = [i + 1 for i, l in enumerate(lines) if pat in l and i + 1 > start]
    if not hits:
        raise SystemExit(f"anchor not found: {pat!r} after {start}")
    return hits[nth]


def rng(f, a, b):
    return f"{f}:{a}-{b}"


# transmission header
impl = find(hdr, "execute_chip_unicast_to_local_chip_impl(")
s0 = find(hdr, "case tt::tt_fabric::NocSendType::NOC_UNICAST_SCATTER_WRITE: {", impl)
h_sc_sem_end = find(hdr, "case tt::tt_fabric::NocSendType::NOC_MULTICAST_WRITE:", s0) - 1
els = find(hdr, "FORCE_INLINE void execute_local_semaphore_inc(")
fl_if = find(hdr, "if (flush != tt::tt_fabric::NocFlushType::FLUSH_NONE) {", els)
fl = find(hdr, "if (!ncrisc_noc_nonposted_writes_flushed(tt::tt_fabric::edm_to_local_chip_noc)) {", els)

# receiver header
loop_pat = os.environ.get("LOOPPAT", "for (size_t i = 0; i < iterations_between_ctx_switch_and_teardown_checks; i++) {")
loop = find(rx, loop_pat, find(rx, "run_main_loop("))
av_line = find(rx, "st.avail = read_stream_count<C::pkts_stream>();")
if any("FORCE_INLINE void resolve_deferred_atomic(" in l for l in rx):
    # v3v layout: the pending test in rx_packet, the resolution in resolve_deferred_atomic
    pend = find(rx, "if (st.pending_atomic) {", find(rx, "FORCE_INLINE bool rx_packet("))
    pend_av_ranges = [(pend + 1, pend + 6), (find(rx, "FORCE_INLINE void resolve_deferred_atomic("), find(rx, "// The packet at the head of receiver channel CH:") - 1)]
else:
    stp = find(rx, "FORCE_INLINE bool rx_channel_step(")
    pend = find(rx, "if (pending_atomic_flush) {", stp)
    pend_end = find(rx, "const uint32_t src_addr = reinterpret_cast<uint32_t>(packet_header);", pend) - 1
    pend_av_ranges = [(pend + 1, pend_end)]
comp = (find(rx, "constexpr bool word_batches") if any("constexpr bool word_batches" in l for l in rx)
        else find(rx, "batch = uncompleted > T ? T : uncompleted;"))
comp_end = find(rx, "st.uncompleted = uncompleted - batch;", comp)
if any("if (reads != 0) {" in l for l in rx):
    c_reads = find(rx, "if (reads != 0) {", comp)
    c_other = find(rx, "if (other == 0) {", c_reads)
elif any("one_read_class" in l for l in rx):
    c_reads = find(rx, "if (((recs >> 24) & SLOT_READS) != 0) {", comp)
    c_other = None
else:
    c_reads = find(rx, "if ((local_reads | fwd_reads) != 0) {", comp)
    c_other = None
c_out = find(rx, "if (outgoing[newest_trid] != 0) {", c_reads)
# retire: next slot index wraps to 0 once per ring (forced not to wrap)
wrap = find(rx, "st.slot_addr = next_idx == 0 ? C::first_slot : st.slot_addr + C::slot_bytes;")
fp = find(rx, "FORCE_INLINE void forward_payload(")
fwd1 = (fp - 1, find(rx, "c.free_slots--;", fp))
ft = find(rx, "FORCE_INLINE void forward_to(")
fwd2 = (ft, ft + 6)
ld = find(rx, "FORCE_INLINE void local_delivery(")
def first_of(*pats):
    for p in pats:
        if any(p in l for l in rx):
            return p
    raise SystemExit(f"none of {pats}")
lw0 = find(rx, first_of("constexpr bool local_write_state_preset =", "constexpr bool local_write_ctrl_preset =",
                        "FORCE_INLINE void local_write_uncounted("))
ld_end = find(rx, "#if defined(FABRIC_2D)", ld) - 1
w = find(rx, "if (noc_send_type == NocSendType::NOC_UNICAST_WRITE) {", ld)
wc = find(rx, first_of("local_write_with_state</*COUNTED=*/true>(", "local_write</*COUNTED=*/true>(",
                       "noc_async_write_one_packet_with_trid</*update_counter=*/false, /*posted=*/false>("), w)
sc = find(rx, "if (noc_send_type == NocSendType::NOC_UNICAST_SCATTER_WRITE) {", ld)
sc_more = find(rx, "if (chunk_count > 2) {", sc) + 1
sc_more_end = find(rx, "constexpr uint8_t CHUNK_ENCODING_MASK = 0b11;", sc_more) - 1
sc_final_if = find(rx, "if (chunk_encoding == ChunkEncoding::CHUNK_ENCODING_UNICAST_WRITE) {", sc)
sc_end = find(rx, "if (noc_send_type == NocSendType::NOC_UNICAST_ATOMIC_INC) {", sc) - 1
at = sc_end + 1
fu = find(rx, "if (noc_send_type == NocSendType::NOC_FUSED_UNICAST_ATOMIC_INC) {", at)
fu_end = find(rx, "} else {", fu) - 1

base_av = [rng(RX, av_line, av_line)] + [rng(RX, a, b) for a, b in pend_av_ranges] + [rng(RX, comp, comp_end)]
fwd_av = [rng(RX, *fwd1), rng(RX, *fwd2)]
local_av = [rng(T, impl - 5, h_sc_sem_end + 40), rng(RX, lw0, ld_end)]
sem_av = [rng(T, els, els + 34)]
arms = {
    "write": [rng(RX, wc, wc + 7)],
    "fwd": [rng(RX, *fwd1)],
    "scatter": [rng(RX, sc, sc_end)],
    "atomic": [rng(RX, at, fu - 1)],
    "fused": [rng(RX, fu, fu_end)],
    "fusedfl": [rng(RX, fu, fu_end)],
    "compu": [rng(RX, wc, wc + 7)],
    "compm": [rng(RX, wc, wc + 7)],
}
issues = {"write": 1, "fwd": 1, "scatter": 2, "atomic": 1, "fused": 2, "fusedfl": 1, "compu": 1, "compm": 1}
dis_text = open(dis).read()
has_fl = f"{T}:{fl}\n" in dis_text or f"{T}:{fl} " in dis_text

ins = []
cur = None
for l in dis_text.split("\n"):
    m = re.match(r"^(/\S+):(\d+)", l)
    if m:
        cur = (m.group(1).split("/")[-1], int(m.group(2)))
        continue
    m = re.match(r"^\s+([0-9a-f]+):\t[0-9a-f]+\s+(\S+)\s*(.*)$", l)
    if m:
        ins.append((int(m.group(1), 16), m.group(2), m.group(3), cur))
cands = []
for ad, op, arg, loc in ins:
    if loc and loc[0] == RX and loc[1] == loop and op in ("bnez", "bne", "bgeu", "bltu", "beqz", "beq", "bltz", "bgez", "j"):
        t = int(re.sub(r" <[^>]*>", "", arg).split(",")[-1], 16)
        if t < ad and ad - t > 16:
            cands.append((ad, t))
if not cands:
    raise SystemExit("no loop back edge found")


def run_walk(extra, verbose):
    best = None
    for end, head in cands:
        r = subprocess.run(
            ["python3", WALK, dis, f"{head:x}", f"{end:x}", "--elf", elf] + extra + (["-v"] if verbose else []),
            capture_output=True,
            text=True,
        ).stdout
        last = r.strip().splitlines()[-1] if r.strip() else "ERR"
        m = re.match(r"instr (\d+)", last)
        if m and (best is None or int(m.group(1)) < best[0]):
            best = (int(m.group(1)), r, head, end)
    return best



for s in scens:
    av = [rng(RX, comp, comp_end)] + local_av + fwd_av + sem_av
    if "-np" in sys.argv: av = [rng(RX, comp, comp_end)]
    rsc = find(rx, "FORCE_INLINE uint32_t read_stream_count() {") + 1
    xa = ["--avoid", ",".join(av), "--need", rng(RX, rsc, rsc)]
    b = run_walk(xa, "-v" in sys.argv)
    print(b[1] if b else "none")
