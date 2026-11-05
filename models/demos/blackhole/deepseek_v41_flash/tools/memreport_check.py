"""memreport_check.py <server.log> [--min-free 40] [--min-largest 40] [--budget B=16 --levers embed_shard,trace_1e9 --copies attn_front,...]

Boot-time MEMREPORT check: parses the four 'MEMREPORT ...' lines of a server log (after the chunk trace / model warm-up / vLLM warm-up / vision traces = end of the build),
prints allocated / free / largest block and the trace-region use per stage, the delta between stages, and (with --budget) the prediction of tt/dram_levers.budget next to the
measurement. Exit code 0 when the END-OF-BUILD free MiB/bank >= --min-free (default 40) and the largest free block >= --min-largest, 1 otherwise, 2 when no MEMREPORT line.
No ttnn / torch import: runs anywhere.
"""
import re
import sys

PAT = re.compile(
    r"MEMREPORT (?P<tag>.*?): DRAM allocated (?P<alloc>[\d.]+) MiB/bank, free (?P<free>[\d.]+), largest free block (?P<largest>[\d.]+)"
    r"(?:; trace region allocated (?P<talloc>[\d.]+) MiB/bank, free (?P<tfree>[\d.]+))?"
)


def parse(lines):
    out = []
    for ln in lines:
        m = PAT.search(ln)
        if m:
            d = m.groupdict()
            out.append(
                {
                    "tag": d["tag"],
                    "alloc": float(d["alloc"]),
                    "free": float(d["free"]),
                    "largest": float(d["largest"]),
                    "trace_used": float(d["talloc"]) if d["talloc"] else None,
                    "trace_free": float(d["tfree"]) if d["tfree"] else None,
                }
            )
    return out


def check(rows, min_free=40.0, min_largest=40.0):
    """(ok, message) on the LAST report (end of the build)."""
    if not rows:
        return None, "no MEMREPORT line"
    last = rows[-1]
    ok = last["free"] >= min_free and last["largest"] >= min_largest
    return ok, (
        f"end of build '{last['tag']}': free {last['free']:.1f} MiB/bank (need >= {min_free}), largest block {last['largest']:.1f} (need >= {min_largest})"
        f" -> {'PASS' if ok else 'FAIL'}"
    )


def main(argv):
    path = argv[1]
    opt = dict(zip(argv[2::2], argv[3::2]))
    rows = parse(open(path, errors="replace").read().splitlines())
    prev = None
    for r in rows:
        d = "" if prev is None else f" (d alloc {r['alloc'] - prev['alloc']:+.1f})"
        tr = "" if r["trace_used"] is None else f" | trace used {r['trace_used']:.1f} free {r['trace_free']:.1f}"
        print(f"{r['tag'][:60]:60s} alloc {r['alloc']:8.1f} free {r['free']:7.1f} largest {r['largest']:7.1f}{d}{tr}")
        prev = r
    if "--budget" in opt:
        from models.demos.blackhole.deepseek_v41_flash.tt import dram_levers as DL

        B = int(opt["--budget"].split("=")[-1])
        levers = [x for x in opt.get("--levers", "").split(",") if x]
        copies = [x for x in opt.get("--copies", ",".join(DL.COPIES)).split(",") if x]
        a, f = DL.budget(B, copies, levers)
        print(f"predicted B={B} copies={copies} levers={levers}: alloc {a:.1f} free {f:.1f}")
    ok, msg = check(rows, float(opt.get("--min-free", 40)), float(opt.get("--min-largest", 40)))
    print(msg)
    return 2 if ok is None else (0 if ok else 1)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
