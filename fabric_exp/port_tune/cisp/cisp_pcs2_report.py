#!/usr/bin/env python3
"""cisp_pcs2_report.py RESULTS_DIR: where the router ERISCs' PC samples go in a cisp_pcs2.sh run (profiler on and off,
whole-run sampling), split by inline chain, and how the samples of the routers that took a stall event in an iteration
differ from the routers that did not. A within-ERISC split, never a bound (CLAUDE.md): PC samples don't name the cause.

Mapping as cisp_pcs_split.py: a target's build is the cache image (<proc>.elf.xip.elf) whose text starts with the image
head read at its launch message's load address; kernel PCs map to pc - load + text start; addr2line -f -i -C on the
cache's -g <proc>.elf gives the inline chain and the innermost source line. PCs outside the kernel image are "firmware"
(the base firmware, or a kernel image other than the one at launch).
Buckets by the whole chain (first match wins): link_sync (link_sync::), eth_ptp, ring_ensure_room_slow,
kernel_profiler (other kernel_profiler::), ctx_switch (with_other_erisc_parked / run_routing_without_noc_sync /
context-switch frames), end_round (the rest of end_round), telemetry (telemetry:: outside end_round), passes (the rest).
Windows: per test iteration, from its "Launching programs" to its "Finished." log line (wall clock; the sampler's
t_start_epoch lines samples up). Stall events: per router and iteration, telemetry cycles more than 300k above the off
arm's median for that test and router (cisp_router_map.py's rule)."""
import collections, datetime, glob, json, os, re, statistics, subprocess, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cisp_router_map as rmap  # noqa: E402

A2L = "/home/aagarwal/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-addr2line"
D = sys.argv[1]
CACHE = {"on": "/home/aagarwal/.tt-metal-cache-cisp-on-ci", "off": "/home/aagarwal/.tt-metal-cache-cisp-off-ci"}
TS = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)")
ORDER = ["passes", "end_round", "ctx_switch", "telemetry", "link_sync", "eth_ptp", "ring_ensure_room_slow",
         "kernel_profiler", "firmware"]


def text_segment(path):
    import struct
    d = open(path, "rb").read()
    phoff, = struct.unpack_from("<I", d, 28)
    phentsize, phnum = struct.unpack_from("<HH", d, 42)
    for i in range(phnum):
        p_type, p_off, p_vaddr, _, p_filesz, _, p_flags, _ = struct.unpack_from("<IIIIIIII", d, phoff + i * phentsize)
        if p_type == 1 and p_flags & 1:
            return p_vaddr, d[p_off:p_off + p_filesz]
    raise ValueError(path)


_chains = {}


def chains(elf, addrs):
    """addr -> [(function, file:line), ...] innermost first."""
    todo = [a for a in addrs if (elf, a) not in _chains]
    for k0 in range(0, len(todo), 2000):
        part = todo[k0:k0 + 2000]
        out = subprocess.run([A2L, "-a", "-f", "-i", "-C", "-e", elf] + [hex(a) for a in part],
                             capture_output=True, text=True).stdout.splitlines()
        cur, k = None, 0
        while k < len(out):
            if re.fullmatch(r"0x[0-9a-f]+", out[k]):
                cur = int(out[k], 16)
                _chains[(elf, cur)] = []
                k += 1
                continue
            fn = out[k]
            while re.search(r"<[^<>]*>", fn):
                fn = re.sub(r"<[^<>]*>", "", fn)
            fn = re.sub(r"\(.*", "", fn).split()[-1] if fn.strip() else "?"
            line = out[k + 1] if k + 1 < len(out) else "?"
            line = re.sub(r" \(discriminator \d+\)", "", line)
            line = "/".join(line.split("/")[-2:])
            _chains[(elf, cur)].append((fn, line))
            k += 2
    return {a: _chains[(elf, a)] for a in addrs}


def bucket(ch):
    fns = [f for f, _ in ch]
    s = " ".join(fns)
    if "link_sync::" in s:
        return "link_sync"
    if "eth_ptp::" in s:
        return "eth_ptp"
    if "ring_ensure_room_slow" in s:
        return "ring_ensure_room_slow"
    if "kernel_profiler::" in s:
        return "kernel_profiler"
    if re.search(r"with_other_erisc_parked|run_routing_without_noc_sync|context_switch|ctx_switch", s):
        return "ctx_switch"
    if any(f.endswith("end_round") for f in fns):
        return "end_round"
    if "telemetry::" in s:
        return "telemetry"
    return "passes"


def windows(log):
    """[(test, iteration index, t_launch, t_finished)] in epoch seconds."""
    out, cur, k, t_l = [], None, collections.Counter(), None
    for line in open(log, errors="replace"):
        m = TS.match(line)
        if not m:
            continue
        t = datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f").replace(
            tzinfo=datetime.timezone.utc).timestamp()
        if (r := rmap.RUN.search(line)):
            cur = r.group(1)
        elif "Launching programs" in line:
            t_l = t
        elif cur and t_l is not None and f"Test {cur} Finished." in line:
            out.append((cur, k[cur], t_l, t))
            k[cur] += 1
            t_l = None
    return out


def load(arm):
    pre = os.path.join(D, f"pcs2_{arm}_samples")
    meta = json.load(open(pre + ".json"))
    rec = np.fromfile(pre + ".bin", dtype=np.dtype([("t", "<f8"), ("i", "<u2"), ("pc", "<u4")]))
    t_abs = rec["t"] + meta["t_start_epoch"]
    proc_imgs = {}
    for proc in ("active_erisc", "subordinate_active_erisc"):
        proc_imgs[proc] = [(e, *text_segment(e)) for e in sorted(glob.glob(
            f"{CACHE[arm]}/**/fabric_erisc_router*/*/{proc}/{proc}.elf.xip.elf", recursive=True))]
    tg_info = {}
    bucket_of = np.full(len(rec), -1, dtype=np.int8)
    line_of = {}
    for idx, tg in enumerate(meta["targets"]):
        e = tg["erisc"]
        proc = "active_erisc" if e == 0 else "subordinate_active_erisc"
        lay = meta["layouts"][f"{tg['chip']}/{tg['chan']}"]["erisc"][str(e)]
        lay_end = meta["layouts_end"].get(f"{tg['chip']}/{tg['chan']}", {}).get("erisc", {}).get(str(e), {})
        head, ld, size = bytes.fromhex(lay["head"]), lay["load"], lay["text_size"]
        m = next(((x, base) for x, base, text in proc_imgs[proc] if head and text[:len(head)] == head), None)
        sel = np.nonzero(rec["i"] == idx)[0]
        tg_info[idx] = dict(tg, matched=m is not None, n=len(sel), relaunched=lay_end.get("head") != lay["head"])
        if m is None:
            continue
        elf, base = m[0][: -len(".xip.elf")], m[1]
        pcs = rec["pc"][sel]
        inside = (pcs >= ld) & (pcs < ld + size)
        bucket_of[sel[~inside]] = ORDER.index("firmware")
        link = pcs[inside].astype(np.int64) - ld + base
        uniq = sorted(set(link.tolist()))
        ch = chains(elf, uniq)
        bmap = {a: ORDER.index(bucket(ch[a])) for a in uniq}
        bucket_of[sel[inside]] = np.array([bmap[a] for a in link.tolist()], dtype=np.int8)
        for j, a in zip(sel[inside].tolist(), link.tolist()):
            line_of[j] = (elf, a)
        for j, pc in zip(sel[~inside].tolist(), pcs[~inside].tolist()):
            line_of[j] = ("fw", pc)
    return meta, rec, t_abs, tg_info, bucket_of, line_of


def shares(mask, bucket_of, rec, tg_info, erisc):
    sel = mask & np.isin(rec["i"], [i for i, t in tg_info.items() if t["erisc"] == erisc and t["matched"]])
    c = collections.Counter(bucket_of[sel].tolist())
    n = sum(v for k, v in c.items() if k >= 0)
    return n, {b: c.get(ORDER.index(b), 0) / max(1, n) for b in ORDER}


def main():
    peer, c2chip = rmap.topology()
    arms = {}
    for arm in ("on", "off"):
        if not os.path.exists(os.path.join(D, f"pcs2_{arm}_samples.bin")):
            continue
        meta, rec, t_abs, tg_info, bucket_of, line_of = load(arm)
        win = windows(os.path.join(D, f"pcs2_{arm}.log"))
        in_win = np.zeros(len(rec), dtype=bool)
        for _, _, a, b in win:
            in_win |= (t_abs >= a) & (t_abs <= b)
        arms[arm] = (meta, rec, t_abs, tg_info, bucket_of, line_of, win, in_win)
        print(f"== {arm}: {meta['samples']} samples, {meta['duration_s']} s, {meta['rate_per_s']}/s, "
              f"{len(meta['targets'])} targets ({sum(1 for t in tg_info.values() if not t['matched'])} unmatched, "
              f"{sum(1 for t in tg_info.values() if t['relaunched'])} relaunched), read errors "
              f"{sum(meta['errors'].values())}; {len(win)} iteration windows, {in_win.mean():.1%} of samples inside")
    for erisc in (0, 1):
        print(f"\n== ERISC{erisc}: share of kernel+firmware samples by bucket (all / inside iteration windows)")
        print("   bucket                  " + "   ".join(f"{a + ' all':>10s} {a + ' win':>10s}" for a in arms) +
              ("   on-off all  on-off win" if len(arms) == 2 else ""))
        res = {}
        for arm, (meta, rec, t_abs, tg_info, bucket_of, line_of, win, in_win) in arms.items():
            res[arm] = (shares(np.ones(len(rec), bool), bucket_of, rec, tg_info, erisc),
                        shares(in_win, bucket_of, rec, tg_info, erisc))
        for b in ORDER:
            row = "   ".join(f"{100 * res[a][0][1][b]:9.3f}% {100 * res[a][1][1][b]:9.3f}%" for a in arms)
            if len(arms) == 2:
                row += f"   {100 * (res['on'][0][1][b] - res['off'][0][1][b]):+9.3f}  {100 * (res['on'][1][1][b] - res['off'][1][1][b]):+9.3f}"
            print(f"   {b:22s}  {row}")
        print("   samples: " + ", ".join(f"{a} {res[a][0][0]} / {res[a][1][0]}" for a in arms))
        # the non-passes buckets by innermost function and source line
        for arm, (meta, rec, t_abs, tg_info, bucket_of, line_of, win, in_win) in arms.items():
            sel = np.nonzero(np.isin(rec["i"], [i for i, t in tg_info.items() if t["erisc"] == erisc and t["matched"]]))[0]
            tot = len(sel)
            c = collections.Counter()
            for j in sel.tolist():
                b = bucket_of[j]
                if b < 0 or ORDER[b] in ("passes",):
                    continue
                key = line_of.get(j)
                if key is None:
                    continue
                if key[0] == "fw":
                    c[(ORDER[b], f"pc 0x{key[1] & ~0xff:x}xx")] += 1
                else:
                    ch = _chains[key]
                    c[(ORDER[b], f"{ch[0][0]} {ch[0][1]}")] += 1
            print(f"   {arm}: top non-passes lines (share of all ERISC{erisc} samples):")
            for (b, l), n in c.most_common(12):
                print(f"      {100 * n / max(1, tot):7.3f}%  {b:22s} {l}")
    # stall events: routers with an event in an iteration vs the rest, same test rows
    if "on" in arms and "off" in arms:
        on_it = rmap.iterations(os.path.join(D, "pcs2_on.log"))
        off_it = rmap.iterations(os.path.join(D, "pcs2_off.log"))
        meta, rec, t_abs, tg_info, bucket_of, line_of, win, in_win = arms["on"]
        tidx = {(t["chip"], t["chan"], t["erisc"]): i for i, t in tg_info.items() if t["matched"]}
        hit_c = {0: collections.Counter(), 1: collections.Counter()}
        cold_c = {0: collections.Counter(), 1: collections.Counter()}
        nhit = ncold = 0
        hit_rows = collections.Counter()
        early = {0: collections.Counter(), 1: collections.Counter()}
        for test, k, a, b in win:
            if test not in off_it or test not in on_it or k >= len(on_it[test]) or test.startswith("Latency"):
                continue
            for rt, cyc in on_it[test][k].items():
                base = statistics.median(it[rt] for it in off_it[test] if rt in it)
                (r, c), ch = rt
                chip = c2chip[(r, c)]
                hit = cyc - base > 300e3
                if hit:
                    nhit += 1
                    hit_rows[re.sub(r"R\d\d_", "_", test) + f" it{k + 1}"] += 1
                else:
                    ncold += 1
                for e in (0, 1):
                    i = tidx.get((chip, ch, e))
                    if i is None:
                        continue
                    m = (rec["i"] == i) & (t_abs >= a) & (t_abs <= b)
                    cnt = collections.Counter(bucket_of[m].tolist())
                    (hit_c if hit else cold_c)[e].update(cnt)
                    if hit:  # the first 2 ms of the window, where the events cluster (hypothesis)
                        early[e].update(bucket_of[m & (t_abs <= a + 0.002)].tolist())
        print(f"\n== stall events (on arm): {nhit} router-iterations over 300k above the off median, {ncold} without")
        print("   rows: " + ", ".join(f"{k} x{v}" for k, v in hit_rows.most_common(10)))
        for e in (0, 1):
            nh, nc = sum(hit_c[e].values()), sum(cold_c[e].values())
            print(f"   ERISC{e}: samples in event router-iterations {nh}, in the others {nc}")
            for bi, b in enumerate(ORDER):
                h, c = hit_c[e].get(bi, 0), cold_c[e].get(bi, 0)
                print(f"      {b:22s} event {h:7d} ({100 * h / max(1, nh):6.2f}%)   others {c:8d} ({100 * c / max(1, nc):6.2f}%)")


if __name__ == "__main__":
    main()
