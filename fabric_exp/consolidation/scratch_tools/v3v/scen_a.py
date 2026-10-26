#!/usr/bin/env python3
"""scen.py VARIANT_NAME DIS ELF [scenario...]: per-packet path counts for local-delivery scenarios on an E1 build.
VARIANT_NAME selects the router source (v3a | v3e1b | v3e1c | v3e1d)."""
import re, sys, subprocess, os
VAR = {"v3a": "v3a", "v3e1b": "v3l2be1b", "v3e1c": "v3e1c", "v3e1d": "v3e1d"}
name, dis, elf = sys.argv[1:4]
scens = [x for x in sys.argv[4:] if not x.startswith("-")] or ["write", "scatter", "atomic", "fused"]
src = open(os.environ.get("SRCDIR", f"/home/aagarwal/tt-metal/fabric_exp/variants/{VAR[name]}") + "/fabric_erisc_router_v3.cpp").read().split("\n")
hdr = open(os.environ.get("SRCDIR", f"/home/aagarwal/tt-metal/fabric_exp/variants/{VAR[name]}") + "/fabric_edm_packet_transmission.hpp").read().split("\n")
def find(lines, pat, start=0, nth=0):
    hits = [i + 1 for i, l in enumerate(lines) if pat in l and i + 1 > start]
    return hits[nth]
R = "fabric_erisc_router_v2.cpp"; T = "fabric_edm_packet_transmission.hpp"
# header arm ranges (ERISC path of execute_chip_unicast_to_local_chip_impl)
impl = find(hdr, "execute_chip_unicast_to_local_chip_impl(")
h_sparse = (find(hdr, "if (noc_send_type == tt::tt_fabric::NocSendType::NOC_SPARSE_MCAST_WRITE) [[unlikely]]", impl), find(hdr, "return;", impl))
h_write = (find(hdr, "case tt::tt_fabric::NocSendType::NOC_UNICAST_WRITE: {", impl), find(hdr, "case tt::tt_fabric::NocSendType::NOC_UNICAST_ATOMIC_INC: {", impl) - 1)
h_atomic = (h_write[1] + 1, find(hdr, "case tt::tt_fabric::NocSendType::NOC_UNICAST_INLINE_WRITE: {", impl) - 1)
h_inline = (h_atomic[1] + 1, find(hdr, "case tt::tt_fabric::NocSendType::NOC_FUSED_UNICAST_ATOMIC_INC: {", impl) - 1)
h_fused = (h_inline[1] + 1, find(hdr, "case tt::tt_fabric::NocSendType::NOC_UNICAST_SCATTER_WRITE: {", impl) - 1)
s0 = h_fused[1] + 1
h_sc_more = (find(hdr, "if (chunk_count > 2) {", find(hdr, "#else", s0)) + 1, find(hdr, "// Last chunk is either a unicast write", s0) - 1)
h_sc_final = find(hdr, "const uint16_t final_chunk_size", h_sc_more[1])
h_sc_sem = (find(hdr, "} else if (chunk_encoding != ChunkEncoding::CHUNK_ENCODING_NOP) {", h_sc_final), find(hdr, "case tt::tt_fabric::NocSendType::NOC_MULTICAST_WRITE:", s0) - 1)
h_scatter = (s0, h_sc_sem[1])
els = find(hdr, "FORCE_INLINE void execute_local_semaphore_inc(")
h_defer = (find(hdr, "*pending_atomic_flush =", els), find(hdr, "*pending_atomic_flush =", els) + 2)
h_now = find(hdr, "noc_semaphore_inc_with_state<true, /*skip_cmdbuf_chk=*/true>(tt::tt_fabric::edm_to_local_chip_noc);", els)
h_nopend = find(hdr, "complete_pending_atomic_flush(rx_channel_id);", els)
h_unfl = find(hdr, "noc_semaphore_inc<true>(", els)
def rng(f, a, b): return f"{f}:{a}-{b}"
avoid = []; need = {}
if name == "v3a":
    loop = find(src, "for (size_t i = 0; i < iterations_between_ctx_switch_and_teardown_checks; i++) {")
    step = find(src, "receiver_forward_timer.set_should_dump(unwritten_packets);")
    pend = find(src, "if (pending_atomic_flush) {", step)
    pend_end = find(src, "if constexpr (is_2d_fabric) {", pend) - 1
    comp = find(src, "const uint32_t batch = uncompleted > batch_target ? batch_target : uncompleted;")
    comp_end = find(src, "completion_counter.increment_n(batch);", comp)
    fd = find(src, "uint8_t forward_2d_dispatch(")
    fwd = [rng(R, c, c + 6) for c in [find(src, "forward_payload_to_downstream_edm<enable_deadlock_avoidance, stateful_api>(", fd, k) for k in range(4)]]
    base_av = [rng(R, pend + 1, pend_end), rng(R, comp, comp + 14), rng(R, comp + 18, comp_end)]
    fwd_av = ["fabric_router_adapter.hpp:1-2000"] + fwd
    avoid += base_av + fwd_av
    local_av = [rng(T, impl - 5, find(hdr, "} break;", h_sc_sem[1] - 3) + 40)]
    fwd_need = ["fabric_router_adapter.hpp:1-2000"]
    nopkt = find(src, "if (unwritten_packets) {", step)
    arms = {"write": [rng(T, *h_write)], "scatter": [rng(T, *h_scatter)], "atomic": [rng(T, *h_atomic)], "fused": [rng(T, *h_fused)]}
else:
    loop = find(src, "for (size_t i = 0; i < iterations_between_ctx_switch_and_teardown_checks; i++) {", find(src, "FORCE_INLINE void run_e1_receive_program("))
    av = find(src, "st.avail = e1_read_stream_count<C::pkts_stream>();")
    stp = find(src, "FORCE_INLINE bool e1_rx_step(")
    pend = find(src, "if (pending_atomic_flush) {", stp)
    pend_end = find(src, "const uint32_t src_addr = reinterpret_cast<uint32_t>(packet_header);", pend) - 1
    comp = find(src, "    const uint32_t batch = uncompleted > T ? T : uncompleted;")
    comp_end = find(src, "    st.uncompleted = uncompleted - batch;", comp)
    fwd1 = (find(src, "template <bool STATEFUL, uint8_t DATA_CMD_BUF, uint8_t SYNC_CMD_BUF>"), find(src, "FORCE_INLINE void e1_forward_payload") + 20)
    fwd2 = (find(src, "FORCE_INLINE uint8_t e1_forward_if_used("), find(src, "FORCE_INLINE uint8_t e1_forward_2d(") + 20)
    rel = find(src, "c.free_slots = e1_read_stream_count<LOCAL_CREDIT_STREAM>();")
    base_av = [rng(R, av, av), rng(R, pend + 1, pend_end), rng(R, comp, comp_end)]
    fwd_av = [rng(R, *fwd1), rng(R, *fwd2)]
    avoid += base_av + fwd_av
    ld = find(src, "FORCE_INLINE void e1_local_delivery(")
    lw0 = find(src, "FORCE_INLINE void e1_local_write_uncounted(") if "e1_local_write_uncounted(" in "\n".join(src) else ld
    local_av = [rng(T, impl - 5, find(hdr, "} break;", h_sc_sem[1] - 3) + 40), rng(R, min(lw0, ld), find(src, "#if defined(FABRIC_2D)", ld) - 1)]
    fwd_need = [rng(R, *fwd1)]
    if name == "v3e1b":
        wc = find(src, "noc_async_write_one_packet_with_trid</*update_counter=*/false, /*posted=*/false>(", ld)
        arms = {"write": [rng(R, wc, wc + 7)], "scatter": [rng(T, *h_scatter)], "atomic": [rng(T, *h_atomic)], "fused": [rng(T, *h_fused)]}
    else:
        sc = find(src, "else if (noc_send_type == NocSendType::NOC_UNICAST_SCATTER_WRITE) {", ld) if name != "x" else 0
        sc_more = find(src, "if (chunk_count > 2) {", sc) + 1
        sc_more_end = find(src, "constexpr uint8_t CHUNK_ENCODING_MASK = 0b11;", sc_more) - 1
        sc_final = find(src, "if (chunk_encoding == ChunkEncoding::CHUNK_ENCODING_UNICAST_WRITE) {", sc) + 1
        sc_sem = find(src, "if (chunk_encoding != ChunkEncoding::CHUNK_ENCODING_UNICAST_WRITE) {", sc)
        w = find(src, "if (noc_send_type == NocSendType::NOC_UNICAST_WRITE) {", ld)
        wc = find(src, "noc_async_write_one_packet_with_trid</*update_counter=*/false, /*posted=*/false>(", w)
        arms = {"write": [rng(R, wc, wc + 7)], "scatter": [rng(R, sc, sc_sem + 14)]}
        armavoid_sc = [rng(R, sc_more, sc_more_end)]
        if name == "v3e1c":
            at = find(src, "else if (noc_send_type == NocSendType::NOC_UNICAST_ATOMIC_INC) {", ld)
            fu = find(src, "} else if (noc_send_type == NocSendType::NOC_FUSED_UNICAST_ATOMIC_INC) {", ld)
            arms["atomic"] = [rng(R, at + 1, at + 8)]; arms["fused"] = [rng(R, fu + 1, fu + 20)]
        else:
            arms["atomic"] = [rng(T, *h_atomic)]; arms["fused"] = [rng(T, *h_fused)]
# per scenario avoid of semaphore paths and scatter shapes
common_sc_avoid = [rng(T, *h_sc_more), rng(T, *h_sc_sem)]
sem = {"atomic": [], "fused": []}
fl_if = find(hdr, "if (flush != tt::tt_fabric::NocFlushType::FLUSH_NONE) {", els)
fl = find(hdr, "if (!ncrisc_noc_nonposted_writes_flushed(tt::tt_fabric::edm_to_local_chip_noc)) {", els)
need_sem = {"atomic": [], "fused": []}
dis_text = open(dis).read()
has_fl = f"{T}:{fl}\n" in dis_text or f"{T}:{fl} " in dis_text
eqne = {"atomic": (([rng(T, fl, fl)] if has_fl else []), [rng(T, fl_if, fl_if)]), "fused": ([rng(T, fl_if, fl_if)], []), "fusedfl": ([], [rng(T, fl_if, fl_if)] + ([rng(T, fl, fl)] if has_fl else []))}
# loop head/end
L = open(dis).read().split("\n")
ins = []
cur = None
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
WALK = "/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/v3v/walk.py"
def run_walk(extra, verbose):
    best = None
    for end, head in cands:
        r = subprocess.run(["python3", WALK, dis, f"{head:x}", f"{end:x}"] + extra + (["-v"] if verbose else []), capture_output=True, text=True).stdout
        last = r.strip().splitlines()[-1] if r.strip() else "ERR"
        m = re.match(r"instr (\d+)", last)
        if m and (best is None or int(m.group(1)) < best[0]): best = (int(m.group(1)), r, head, end)
    return best
arms["fwd"] = fwd_need
for sc in scens:
    av = list(avoid) if sc != "fwd" else base_av + local_av
    nd = list(arms["fused" if sc == "fusedfl" else sc])
    if sc in ("write", "scatter"): av += [rng(T, els, els + 34)]
    if sc == "scatter":
        av += common_sc_avoid
        if name in ("v3e1c", "v3e1d"): av += armavoid_sc
    if sc in sem:
        av += sem[sc]; nd2 = need_sem[sc]
    else:
        nd2 = []
    needs = ";".join([",".join(nd)] + nd2)
    xa = []
    if sc in eqne:
        if eqne[sc][0]: xa += ["--eq", ",".join(eqne[sc][0])]
        if eqne[sc][1]: xa += ["--ne", ",".join(eqne[sc][1])]
    eqs = [rng(R, pend, pend)]
    if sc == "scatter": eqs.append(rng(R, sc_final - 1, sc_final - 1) if name in ("v3e1c", "v3e1d") else rng(T, h_sc_final - 1, h_sc_final - 1))
    if "--eq" in xa:
        k = xa.index("--eq"); xa[k + 1] += "," + ",".join(eqs)
    else:
        xa += ["--eq", ",".join(eqs)]
    xa += ["--issues", str({"write": 1, "scatter": 2, "atomic": 1, "fused": 2, "fusedfl": 1, "fwd": 1}[sc])]
    if "-p" in sys.argv: print(av, needs, [(hex(h), hex(e)) for e, h in cands])
    if "-d" in sys.argv:
        import re as _re
        b = run_walk(["--need", needs, "--elf", elf] + xa, True)
        out = b[1] if b else ""
        def ins_(f, ln):
            for t in av:
                ff, r = t.rsplit(":", 1); lo, _, hi = r.partition("-")
                if f.startswith(ff) and int(lo) <= ln <= int(hi or lo): return t
        for l in out.splitlines():
            m = _re.search(r"(\S+):(\d+)$", l)
            if m and l.split()[1] in ("lw", "lh", "lhu", "lb", "lbu", "sw", "sh", "sb", "jal") and ins_(m.group(1), int(m.group(2))): print("   HIT", ins_(m.group(1), int(m.group(2))), l)
        print("   ", out.strip().splitlines()[-1] if out.strip() else "no path")
    b = run_walk(["--avoid", ",".join(av), "--need", needs, "--elf", elf] + xa, "-v" in sys.argv)
    r = b[1] if b else ""
    if "-v" in sys.argv: print(r, hex(b[2]) if b else "", hex(b[3]) if b else "")
    print(f"{name:6s} {sc:8s} {r.strip().splitlines()[-1] if r.strip() else 'no path'}")
