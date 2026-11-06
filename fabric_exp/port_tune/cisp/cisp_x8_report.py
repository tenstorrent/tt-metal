#!/usr/bin/env python3
"""cisp_x8_report.py RESULTS_DIR: the X8 Ethernet link-state polls (cisp_ethpoll.sh, profiler on), per order
(i: stock L1/L2 alternation, ii: L1-only, gap: alternation with an idle pause after each L2 block).

  1. perturbation: per row type, the CSV bandwidth median of the polled arm against the same yaml unpolled, and the
     slow-iteration counts (an NX write iteration with a router whose telemetry cycles are more than 2% above the off
     arm's median for that row type and router: cisp_router_tally.py's rule);
  2. link state: every value PCS_STATUS, ERR_STAT, port / train status and rx_link_up took on any router core;
  3. error and recovery counters: their change over each polled run, per router core (any nonzero listed);
  4. every polled word that changed during the run, by router class (link-0: used by the num_links 1 rows; link-1:
     the second link) and iteration class (L1 write rows of slow iterations, of normal iterations, L2 rows, gap rows,
     between iterations): changes per second and mean rate of change, so a counter that moves only where the slow
     state builds stands out;
  5. base FW link status checks (the metal mailbox's next-check time moving): count per run and how many fell in
     slow / normal L1 iterations, per router class.
Observation only: a counter moving with the slow state is not its cause (CLAUDE.md)."""
import collections, csv, json, os, statistics, sys
import numpy as np

R = sys.argv[1]
sys.argv = ["x", R]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/aagarwal/tt-metal/fabric_exp/consolidation")
import cisp_pcs2_report as P  # noqa: E402
import cisp_router_tally as T  # noqa: E402

# ---- names for the polled words (L1 offsets from 0x7C400: debug_buf_t at 0, boot_results_t at 0x800)
NAMES = {}
for k, n in enumerate(["txq0_resend_last", "txq1_resend_last", "txq2_resend_last", "rxq0_drop_last", "rxq1_drop_last",
                       "rxq2_drop_last"]):
    NAMES[8 + 4 * k] = "dbg." + n
for k, n in enumerate(["retrain_count", "rx_link_up", "link_flap_count", "link_poll_alive_count"]):
    NAMES[32 + 4 * k] = "dbg.last_live." + n
for k, n in enumerate(["serdes_ctrl_lock_counter", "lead_serdes_status", "follow_serdes_status", "partner",
                       "eth_ctrl_intp_stat_raw", "eth_ctrl_intp_stat", "mac_rx_int", "mac_tx_int", "pcs_tx_int",
                       "pcs_rx_int", "pma_int", "user_command", "link_down_cnt", "link_recovery_cnt",
                       "link_recovery_retry_cnt", "link_recovery_give_up_cnt", "runtime_serdes_retrain_cnt",
                       "runtime_serdes_reinit_cnt", "runtime_macpcs_reinit_cnt", "runtime_macpcs_only_reinit_cnt"]):
    NAMES[320 + 4 * k] = "dbg." + n
for k in range(64):
    NAMES[1600 + 4 * k] = f"dbg.eth_api_metrics[{k}]"
for off, n in ((1856, "link_recovery_backoff_ts"), (1864, "link_status_check_ts"), (1872, "dynamic_state_check_ts")):
    NAMES[off], NAMES[off + 4] = f"dbg.{n}.lo", f"dbg.{n}.hi"
for k, n in enumerate(["magic", "run_count", "entry_lo", "entry_hi", "exit_lo", "exit_hi", "exit_valid"]):
    NAMES[1984 + 4 * k] = "dbg.ptp_trace." + n
B = 0x800
for k, n in enumerate(["postcode", "port_status", "train_status", "train_speed", "fw_stage0", "fw_stage1"]):
    NAMES[B + 4 * k] = "eth_status." + n
for k in range(4):
    NAMES[B + 0x70 + 4 * k] = f"eth_status.heartbeat[{k}]"
for idx, n in ((8, "anlt_retry_cnt"), (9, "manual_eq_retry_cnt"), (10, "lcpll_lock_fail_cnt"), (42, "cdr_unlocked_cnt"),
               (43, "cdr_unlock_transitions")):
    NAMES[B + 0x80 + 4 * idx] = "serdes." + n
NAMES[B + 0x180 + 4], NAMES[B + 0x180 + 8] = "macpcs.retry_cnt", "macpcs.eth_cntrl_int"
L = B + 0x200
for k, n in enumerate(["retrain_count", "rx_link_up", "link_flap_count", "link_poll_alive_count"]):
    NAMES[L + 4 * k] = "live." + n
for k, n in enumerate(["frames_txd", "frames_txd_ok", "frames_txd_badfcs", "bytes_txd", "bytes_txd_ok",
                       "bytes_txd_badfcs", "frames_rxd", "frames_rxd_ok", "frames_rxd_badfcs", "frames_rxd_dropped",
                       "bytes_rxd", "bytes_rxd_ok", "bytes_rxd_badfcs", "bytes_rxd_dropped", "corr_cw", "uncorr_cw",
                       "txq0_resend_cnt", "txq1_resend_cnt", "txq2_resend_cnt", "rxq0_pkt_drop", "rxq1_pkt_drop",
                       "rxq2_pkt_drop"]):
    NAMES[L + 32 + 8 * k], NAMES[L + 36 + 8 * k] = f"live.{n}", f"live.{n}.hi"
for k, n in enumerate(["host", "risc1", "cmfw", "other"]):
    NAMES[0xC00 + 16 * k] = f"fw_mailbox.{n}.msg"
ERRORS = ["live.retrain_count", "live.link_flap_count", "live.frames_txd_badfcs", "live.frames_rxd_badfcs",
          "live.frames_rxd_dropped", "live.uncorr_cw", "live.corr_cw", "live.txq0_resend_cnt", "live.txq1_resend_cnt",
          "live.txq2_resend_cnt", "live.rxq0_pkt_drop", "live.rxq1_pkt_drop", "live.rxq2_pkt_drop",
          "dbg.link_down_cnt", "dbg.link_recovery_cnt", "dbg.link_recovery_retry_cnt", "dbg.link_recovery_give_up_cnt",
          "dbg.runtime_serdes_retrain_cnt", "dbg.runtime_serdes_reinit_cnt", "dbg.runtime_macpcs_reinit_cnt",
          "dbg.runtime_macpcs_only_reinit_cnt", "serdes.cdr_unlocked_cnt", "serdes.cdr_unlock_transitions",
          "macpcs.retry_cnt", "txq0_RESEND_CNT", "txq1_RESEND_CNT", "txq2_RESEND_CNT"]
STATE = ["PCS_STATUS", "ERR_STAT", "eth_status.port_status", "eth_status.train_status", "live.rx_link_up"]


def words(v):
    """(names, matrix [records x words]) of every polled 32-bit word of one run."""
    meta = json.load(open(f"{R}/x8_{v}_on_poll_eth.json"))
    rec = np.load(f"{R}/x8_{v}_on_poll_eth.npz")["recs"]
    l1 = rec["l1"].view("<u4").reshape(len(rec), -1)
    mb = rec["mbx"].view("<u4").reshape(len(rec), -1)
    names = (list(meta["reg_names"]) + [NAMES.get(4 * k, f"l1+0x{0x7C400 + 4 * k:05x}") for k in range(l1.shape[1])]
             + [f"mbx+0x{0x590 + 4 * k:03x}" for k in range(mb.shape[1])])
    names[len(meta["reg_names"]) + l1.shape[1] + 10] = "mbx.next_link_check.lo"
    names[len(meta["reg_names"]) + l1.shape[1] + 11] = "mbx.next_link_check.hi"
    return meta, rec, names, np.concatenate([rec["regs"], l1, mb], axis=1).astype(np.int64)


def baselines(v):
    off = T.parse(f"{R}/x8_{v}_off.log")
    offr = list(csv.DictReader(open(f"{R}/x8_{v}_off_bandwidth_results.csv")))
    base = collections.defaultdict(list)
    for (n, tel), r in zip(off, offr):
        for rt, c in tel.items():
            base[(T.rowkey(n, r), rt)].append(c)
    return {k: statistics.median(x) for k, x in base.items()}


def iterations(v, tag):
    """[(rowkey, slow?, t_launch, t_finished)] of one on-arm log."""
    base = baselines(v)
    on = T.parse(f"{R}/x8_{v}_on_{tag}.log")
    onr = list(csv.DictReader(open(f"{R}/x8_{v}_on_{tag}_bandwidth_results.csv")))
    win = P.windows(f"{R}/x8_{v}_on_{tag}.log")
    assert len(win) == len(on) == len(onr), (v, tag, len(win), len(on), len(onr))
    out = []
    for (n, tel), r, w in zip(on, onr, win):
        k = T.rowkey(n, r)
        low = {rt for rt, c in tel.items() if (k, rt) in base and c / base[(k, rt)] - 1 > 0.02}
        out.append((k, low, float(r["bandwidth_GB_s"]), w[2], w[3]))
    return out


# link-0 routers: those the num_links 1 write rows use (off logs of every order)
used_l1 = set()
for v in ("i", "ii", "gap"):
    for (n, tel), r in zip(T.parse(f"{R}/x8_{v}_off.log"), csv.DictReader(open(f"{R}/x8_{v}_off_bandwidth_results.csv"))):
        if T.rowkey(n, r)[2] == "1" and r["num_packets"] != "1":
            used_l1 |= set(tel)

print("== 1. perturbation: polled vs unpolled, same yaml, profiler on (CSV bandwidth_GB_s per row type: mean over\n"
      "   iterations, median over the iterations with no low router; slow iterations poll/nopoll)")
its = {}
for v in ("i", "ii", "gap"):
    its[v] = {tag: iterations(v, tag) for tag in ("poll", "nopoll")}
    for tag in ("poll", "nopoll"):
        w = [x for x in its[v][tag] if x[0][3] != "1"]
        print(f"   {v:3s} {tag:6s}: write iterations {len(w)}, slow (a router >2% low) {sum(1 for x in w if x[1])}")
    keys = sorted({x[0] for x in its[v]["poll"]})
    for k in keys:
        row = f"      size {k[0]} links {k[2]} pkts {k[3]:>6s}:"
        for nm, f in (("mean", lambda x: True), ("median of normal its", lambda x: not x[1])):
            a = [x[2] for x in its[v]["poll"] if x[0] == k and f(x)]
            b = [x[2] for x in its[v]["nopoll"] if x[0] == k and f(x)]
            agg = statistics.mean if nm == "mean" else statistics.median
            if a and b:
                row += f"  {nm} poll {agg(a):7.3f} nopoll {agg(b):7.3f} ({agg(a) / agg(b) - 1:+.2%}, n {len(a)}/{len(b)})"
        print(row + f"  slow {sum(1 for x in its[v]['poll'] if x[0] == k and x[1])}/"
              f"{sum(1 for x in its[v]['nopoll'] if x[0] == k and x[1])}")

print("\n== 2. link state values seen (all polls, all router cores)")
data = {}
for v in ("i", "ii", "gap"):
    meta, rec, names, M = words(v)
    data[v] = (meta, rec, names, M)
    tg = [(t["chip"], t["chan"]) for t in meta["targets"]]
    print(f"   {v}: {meta['records']} polls, {len(tg)} cores, {meta['sweeps']} sweeps, {meta['sweep_ms']} ms/sweep, "
          f"read errors {sum(meta['errors'].values())}")
    for s in STATE:
        c = names.index(s)
        vals = collections.Counter(M[:, c].tolist())
        print(f"      {s:24s} " + ", ".join(f"0x{x:x} x{n}" for x, n in vals.most_common(6)))

print("\n== 3. error / recovery counters: change over the run (cores that changed, total, per-core rate /s)")
for v in ("i", "ii", "gap"):
    meta, rec, names, M = data[v]
    tg = [(t["chip"], t["chan"]) for t in meta["targets"]]
    print(f"   {v}:")
    any_ = False
    for e in ERRORS:
        c = names.index(e)
        per = {"link-0": [], "link-1": []}
        for i, rt in enumerate(tg):
            sel = rec["i"] == i
            col, tt = M[sel, c], rec["t"][sel]
            if len(col) > 1 and col.max() != col.min():
                per["link-0" if rt in used_l1 else "link-1"].append((rt, int(col[-1] - col[0]), tt[-1] - tt[0]))
        if not per["link-0"] and not per["link-1"]:
            continue
        any_ = True
        print(f"      {e:34s} " + "; ".join(
            f"{rc}: {len(x)} cores, total {sum(d for _, d, _ in x)}, median {statistics.median(d / dt for _, d, dt in x):.4g}/s"
            for rc, x in per.items() if x))
        if e != "live.corr_cw":
            for rc, x in per.items():
                for rt, d, dt in x:
                    print(f"         {rc} {T.rname(rt) if rt in T.peer else rt}: +{d} in {dt:.2f} s")
    if not any_:
        print("      none changed")

print("\n== 4. polled words that change, by router class and iteration class (changes/s, mean d/s)")
CLS = ["pre", "L1 slow it", "L1 normal it", "L2", "gap row", "between", "post"]
tab = collections.defaultdict(lambda: collections.defaultdict(lambda: [0.0, 0, 0.0]))  # (word, rclass) -> cls -> [s, n, sum d]
movers = set()
link_checks = collections.Counter()
for v in ("i", "ii", "gap"):
    meta, rec, names, M = data[v]
    tg = [(t["chip"], t["chan"]) for t in meta["targets"]]
    t_abs = rec["t"] + meta["t_start_epoch"]
    it = its[v]["poll"]
    starts = np.array([x[3] for x in it])
    ends = np.array([x[4] for x in it])
    nlc = names.index("mbx.next_link_check.lo")
    for i, rt in enumerate(tg):
        rc = "link-0" if rt in used_l1 else "link-1"
        sel = np.nonzero(rec["i"] == i)[0]
        if len(sel) < 2:
            continue
        X, t = M[sel], t_abs[sel]
        dX = np.diff(X, axis=0)
        dt = np.diff(t)
        mid = (t[1:] + t[:-1]) / 2
        j = np.searchsorted(starts, mid, side="right") - 1
        for q in range(len(mid)):
            jj = j[q]
            if mid[q] < starts[0]:
                cl = "pre"
            elif mid[q] > ends[-1]:
                cl = "post"
            elif mid[q] > ends[jj]:
                cl = "between"
            else:
                k, low = it[jj][0], it[jj][1]
                cl = "gap row" if k[3] == "1" else "L2" if k[2] == "2" else "L1 slow it" if low else "L1 normal it"
            ch = np.nonzero(dX[q])[0]
            for c in ch:
                movers.add(c)
                a = tab[(c, rc)][(v, cl)]
                a[1] += 1
                a[2] += dX[q, c]
            tab[("dt", rc)][(v, cl)][0] += dt[q]
            if dX[q, nlc]:
                link_checks[(v, rc, cl)] += 1
names = data["i"][2]
during = {c for c in movers if any(tab[(c, rc)][(v, cl)][1] for rc in ("link-0", "link-1") for v in ("i", "ii", "gap")
                                   for cl in CLS if cl not in ("pre", "post"))}
print("   words that change only before the first / after the last iteration (process start / teardown): " +
      ", ".join(names[c] for c in sorted(movers - during)))
movers = during
print("   core-seconds of poll intervals per class: " + "; ".join(
    f"{rc} " + ", ".join(f"{v}/{cl} {tab[('dt', rc)][(v, cl)][0]:.2f}" for v in ("i", "ii", "gap") for cl in CLS
                         if tab[('dt', rc)][(v, cl)][0]) for rc in ("link-0", "link-1")))
hdr = "".join(f"{v + '/' + cl:>17s}" for v in ("i", "ii", "gap") for cl in ("L1 slow it", "L1 normal it", "between"))
for rc in ("link-1", "link-0"):
    print(f"\n   -- {rc} cores: changes per second (count) in L1 write windows / between, per order")
    print(f"   {'word':34s}{hdr}")
    for c in sorted(movers):
        if (c, rc) not in tab:
            continue
        row = ""
        for v in ("i", "ii", "gap"):
            for cl in ("L1 slow it", "L1 normal it", "between"):
                s = tab[("dt", rc)][(v, cl)][0]
                n = tab[(c, rc)][(v, cl)][1]
                row += f"{n / s:17.1f}" if s else f"{'-':>17s}"
        print(f"   {names[c][:34]:34s}{row}")
    print(f"   -- {rc} cores: mean change per second in the same classes")
    for c in sorted(movers):
        if (c, rc) not in tab:
            continue
        row = ""
        for v in ("i", "ii", "gap"):
            for cl in ("L1 slow it", "L1 normal it", "between"):
                s = tab[("dt", rc)][(v, cl)][0]
                d = tab[(c, rc)][(v, cl)][2]
                row += f"{d / s:17.4g}" if s else f"{'-':>17s}"
        print(f"   {names[c][:34]:34s}{row}")

print("\n== 5. base FW link status checks (next-check time moved between two polls), by router and iteration class")
for v in ("i", "ii", "gap"):
    print(f"   {v}: " + "; ".join(f"{rc}: " + ", ".join(f"{cl} {link_checks[(v, rc, cl)]}" for cl in CLS)
                                 for rc in ("link-0", "link-1")))

print("\n== 6. rates between consecutive polls of one core, in L1 write windows (median [p10, p90] per second)")
RATE = [("txq0_PKT_START_CNT", 32), ("txq0_WORD_CNT", 32), ("txq0_TRANSFER_CNT", 32), ("txq1_PKT_START_CNT", 32),
        ("txq1_TRANSFER_CNT", 32), ("txq2_PKT_START_CNT", 32), ("txq2_TRANSFER_CNT", 32), ("txq0_RESEND_CNT", 32),
        ("txq1_RESEND_CNT", 32), ("txq2_RESEND_CNT", 32)]
# (eth_status.heartbeat[0] is left out: both router ERISCs write their own round count to it, so the difference
# between two polls mixes two counters; it moves on every core in every class, which is all it shows here)
rates = collections.defaultdict(list)
for v in ("i", "ii", "gap"):
    meta, rec, names, M = data[v]
    tg = [(t["chip"], t["chan"]) for t in meta["targets"]]
    t_abs = rec["t"] + meta["t_start_epoch"]
    it = its[v]["poll"]
    starts = np.array([x[3] for x in it])
    ends = np.array([x[4] for x in it])
    for i, rt in enumerate(tg):
        rc = "link-0" if rt in used_l1 else "link-1"
        sel = np.nonzero(rec["i"] == i)[0]
        X, t = M[sel], t_abs[sel]
        # intervals that lie wholly inside one L1 write window
        j0 = np.searchsorted(starts, t[:-1], side="right") - 1
        j1 = np.searchsorted(starts, t[1:], side="right") - 1
        for q in np.nonzero((j0 == j1) & (j0 >= 0))[0]:
            jj = j0[q]
            if t[q + 1] > ends[jj]:
                continue
            k, low = it[jj][0], it[jj][1]
            if k[2] != "1" or k[3] == "1":
                continue
            cl = "slow" if low else "normal"
            dt = t[q + 1] - t[q]
            for nm, bits in RATE:
                c = names.index(nm)
                d = (X[q + 1, c] - X[q, c]) % (1 << bits)
                rates[(nm, rc, v, cl)].append(d / dt)
for rc in ("link-1", "link-0"):
    print(f"   -- {rc} cores")
    cols = [(v, cl) for v in ("i", "ii", "gap") for cl in ("slow", "normal") if rates.get(("txq0_PKT_START_CNT", rc, v, cl))]
    print(f"   {'counter':26s}" + "".join(f"{v + '/' + cl + ' (n ' + str(len(rates[('txq0_PKT_START_CNT', rc, v, cl)])) + ')':>30s}"
                                     for v, cl in cols))
    for nm, _ in RATE:
        row = ""
        for v, cl in cols:
            x = np.array(rates[(nm, rc, v, cl)])
            row += f"{np.median(x):10.4g} [{np.percentile(x, 10):7.3g},{np.percentile(x, 90):8.3g}]"
        print(f"   {nm:26s}{row}")

print("\n== 7. FEC corrected codewords (live.corr_cw, a snapshot base FW refreshes at ERISC0's link check, ~1/s; the")
print("   teardown snapshot included): per core, the rate over the run in each order, and the per-core ratio to (ii)")
fec, per_int = {}, collections.defaultdict(list)
for v in ("i", "ii", "gap"):
    meta, rec, names, M = data[v]
    tg = [(t["chip"], t["chan"]) for t in meta["targets"]]
    t_abs = rec["t"] + meta["t_start_epoch"]
    it = its[v]["poll"]
    slow_w = [(x[3], x[4]) for x in it if x[1] and x[0][2] == "1" and x[0][3] != "1"]
    c, cu = names.index("live.corr_cw"), names.index("live.uncorr_cw")
    for i, rt in enumerate(tg):
        sel = np.nonzero(rec["i"] == i)[0]
        col, t = M[sel, c], t_abs[sel]
        ch = np.nonzero(np.diff(col))[0] + 1  # the first poll showing each new snapshot
        if len(ch) < 2:
            continue
        a, b = ch[0], ch[-1]
        fec[(v, rt)] = ((col[b] - col[a]) / (t[b] - t[a]), int(M[sel[b], cu] - M[sel[a], cu]))
        for x0, x1 in zip(ch[:-1], ch[1:]):
            ta, tb = t[x0], t[x1]
            frac = sum(max(0.0, min(tb, e) - max(ta, s_)) for s_, e in slow_w) / (tb - ta)
            per_int[(v, "link-0" if rt in used_l1 else "link-1")].append(((col[x1] - col[x0]) / (tb - ta), frac))
cores = sorted({rt for _, rt in fec})
for rc in ("link-1", "link-0"):
    cs = [rt for rt in cores if (rt in used_l1) == (rc == "link-0")]
    print(f"   {rc}: " + "; ".join(
        f"{v}: {sum(1 for rt in cs if (v, rt) in fec)} cores, median {np.median([fec[(v, rt)][0] for rt in cs if (v, rt) in fec]):.0f}/s, "
        f"uncorrected {sum(fec[(v, rt)][1] for rt in cs if (v, rt) in fec)}" for v in ("i", "ii", "gap")))
    for v in ("i", "gap"):
        r = [fec[(v, rt)][0] / fec[("ii", rt)][0] for rt in cs if (v, rt) in fec and ("ii", rt) in fec and fec[("ii", rt)][0]]
        if r:
            print(f"      per-core ratio {v}/ii: median {np.median(r):.3f}, range {min(r):.3f} .. {max(r):.3f} (n {len(r)})")
for (v, rc), x in sorted(per_int.items()):
    if v == "ii":
        continue
    r, f = np.array([a for a, _ in x]), np.array([b for _, b in x])
    hi = f > np.median(f)
    print(f"   {v:3s} {rc}: {len(x)} snapshot intervals, slow share median {np.median(f):.1%} (max {f.max():.1%}); corr/s "
          f"median in the intervals above the median share {np.median(r[hi]):.0f}, at or below {np.median(r[~hi]):.0f}")
