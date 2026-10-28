#!/usr/bin/env python3
"""pcs_lines.py CACHE GDIR PREFIX ROW [ROW ...] [--erisc 0|1] [--target CHIP/CHAN] [--top N]: line-level PC-sampling
profile of the busiest router ERISC of one kind (default E1), or of the given router's, for finding where it waits. The
busiest is chosen by v4's work sections (pcs_profile.WORK); for other variants name the router with --target.

Same inputs as pcs_profile.py (PREFIX = pcsv_<tag>; GDIR holds the -g twins from `pcs_sections.py twins`). Each sampled
address maps through its inline chain to its innermost source line; lines are ranked by their share of the ERISC's
kernel samples times the row's cycles/packet. Each line shows its inline chain (innermost first, up to the loop
function) and its hottest instruction. On this core a waiting instruction holds the PC, so a load's wait shows on the
first instruction that uses its value (RESULTS §17): read a hot line together with the instruction shown."""
import argparse, collections, csv, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pcs_profile as pp  # noqa: E402
import pcs_sections as ps  # noqa: E402

OD = os.path.join(os.path.dirname(os.path.dirname(HERE)), "runtime/sfpi/compiler/bin/riscv-tt-elf-objdump")


def instructions(elf, addrs):
    """{addr: 'mnemonic operands'} for the given addresses."""
    lo, hi = min(addrs), max(addrs) + 4
    out = subprocess.run([OD, "-d", "--no-show-raw-insn", f"--start-address={lo:#x}", f"--stop-address={hi:#x}", elf],
                         capture_output=True, text=True).stdout
    res = {}
    for line in out.splitlines():
        m = re.match(r"\s+([0-9a-f]+):\s+(.*)", line)
        if m:
            res[int(m.group(1), 16)] = re.sub(r"\s+", " ", m.group(2).split("#")[0]).strip()
    return res


def short(fn):
    fn = re.sub(r"\(.*", "", fn)
    fn = re.sub(r"<.*>", "<>", fn)
    return fn.split("::")[-1] if fn.count("::") > 1 else fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache"); ap.add_argument("gdir"); ap.add_argument("prefix"); ap.add_argument("rows", nargs="+")
    ap.add_argument("--erisc", type=int, default=1)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--target", default="")
    a = ap.parse_args()
    for row in a.rows:
        vals = [float(r["avg_packets_per_s"]) for r in csv.DictReader(open(f"{ps.RES}/{a.prefix}_{row}_13129088_trial.csv"))]
        cpp = 1.35e9 / (sum(vals) / len(vals))
        best = None
        a2l = {}
        for i, (tgt, h, addrs) in pp.target_samples(a.cache, a.prefix, row).items():
            if tgt["erisc"] != a.erisc or not addrs:
                continue
            if a.target and f"{tgt['chip']}/{tgt['chan']}" != a.target:
                continue
            proc = "active_erisc" if a.erisc == 0 else "subordinate_active_erisc"
            twin = f"{a.gdir}/{h}/{proc}/{proc}.elf"
            if not os.path.exists(twin):
                continue
            cache = a2l.setdefault(twin, {})
            need = sorted(set(addrs) - set(cache))
            if need:
                cache.update(ps.a2l(twin, need))
            secs = collections.Counter(pp.classify(cache.get(x, [])) for x in addrs)
            work = sum(secs[s] for s in pp.WORK[a.erisc]) / len(addrs)
            if best is None or work > best[0]:
                best = (work, tgt, twin, addrs)
        if best is None:
            print(f"\n### {row}: no E{a.erisc} target with a -g twin")
            continue
        _, tgt, twin, addrs = best
        frames = a2l[twin]
        n = len(addrs)
        by_addr = collections.Counter(addrs)
        insn = instructions(twin, list(by_addr))
        lines = collections.defaultdict(collections.Counter)
        chains = {}
        for addr, c in by_addr.items():
            fr = frames.get(addr, [])
            key = f"{fr[0][1]}:{fr[0][2]}" if fr else "?"
            lines[key][addr] += c
            chains.setdefault(key, " < ".join(short(f) for f, _, _ in fr[:4]))
        print(f"\n### {row}: {cpp:.1f} cycles/packet; E{a.erisc} chip {tgt['chip']} chan {tgt['chan']} ({n} samples)")
        ranked = sorted(lines.items(), key=lambda kv: -sum(kv[1].values()))
        for key, cnt in ranked[:a.top]:
            tot = sum(cnt.values())
            hot, hc = cnt.most_common(1)[0]
            print(f"  {tot / n * cpp:6.1f}  {tot / n * 100:5.1f}%  {key:32} {chains[key][:70]:70}  "
                  f"[{hot:#x} {insn.get(hot, '?')}: {hc / n * cpp:.1f}]")


if __name__ == "__main__":
    main()
