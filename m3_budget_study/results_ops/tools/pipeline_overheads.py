#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""P1-D readout: per-stage compute, blocking send, hop and host gap of the common-runner pipeline sessions.

  pipeline_overheads.py results_ops/pipeline [--out pipeline_overheads.txt]

Reads <session>/runner.log (PP_TIMING lines from PREFILL_PP_TIMING=1, CHUNK_START / CHUNK_COMPUTE), the
producer logs <stream>.log (push order -> which chunk belongs to which stream; DONE wall + layer-ack drain ->
tok/s) and session.env (W, sync, share_fabric_links). Warm streams are left out.

Per stage, medians over the chunks of a stream (one PP_TIMING line per chunk and rank):
  compute   CHUNK_COMPUTE (synchronize before and after prefill_chunk; sync sessions only)
  enqueue   host time of the prefill_chunk call (enqueue ~ compute -> the stage is host-bound)
  push      outbound_socket_service_sync + release_fabric_links (send_ms)
  lease_out wait_for_fabric_links on the outbound service at the next loop top: the rest of the send. In a sync
            session this is pure transfer; un-synced it also holds the device tail of the compute.
  block     push + lease_out = the blocking send (stage busy after compute until the send completes). Pavlo's
            fit: ~7.5 ms at W=2048, 18 ms at 5120 (linear: 0.5 + 3.42e-3 * W ms).
  pre_sync  synchronize_device before prefill_chunk (sync sessions): device work still queued from the last chunk
  recv      input wait: H2D socket (rank 0) or inbound D2D (the upstream compute + hop when this stage is idle)
  cycle     t_top(c+1) - t_top(c): the stage's loop period
  host_gap  cycle - (lease_in + lease_out + recv + pre_sync + compute|enqueue + push): host time outside every
            timed piece (logging, metadata decode, python)
  hop       (sync sessions) t_recv(s+1, c) - t_sent(s, c), only when stage s+1 was already waiting (its recv began before s sent):
            forward + receiver drain. hop_start = CHUNK_START(s+1, c) - t_sent(s, c) on the same chunks.
  hop_sp2   the SP2-study definition: CHUNK_START(s+1) - (CHUNK_START(s) + compute(s)), next stage idle.
Model check (sync sessions): tok/s = W / max_s(compute) and W / max_s(compute + block), vs the measured tok/s of
the un-synced session's open stream at the same W when there is one.
"""

import argparse
import glob
import os
import re
import statistics
from collections import defaultdict

TIMING = re.compile(r"\[pp rank (\d+)\] PP_TIMING c=(\d+) (.*)$")
KV = re.compile(r"(\w+)=([-\d.]+)")
START = re.compile(r"\[pp rank (\d+)\] CHUNK_START c=(\d+) compute_start=([\d.]+)")
COMPUTE = re.compile(r"\[pp rank (\d+)\] CHUNK_COMPUTE c=(\d+) compute_ms=([\d.]+)")
PUSH = re.compile(r"\[producer\] push slot=\d+ cidx=\d+ start=(\d+)")
DONE = re.compile(r"DONE wall=([\d.]+)s pushes=(\d+) requests=(\d+) tokens=(\d+)")
PCC = re.compile(r"kv_cache_pcc_complete slots_checked=\d+ min_pcc=[-\d.]+(?: \w+_pcc=[-\d.]+)*")
LAYER_PCC = re.compile(r"layer +(\d+): (K=[\d.]+ V=[\d.]+(?: index_k=[\d.]+)?)")
DRAIN = re.compile(r"drained (\d+)/(\d+) layer acks in ([\d.]+)s")


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def fmt(x, w=7, p=2):
    return f"{'-':>{w}}" if x is None else f"{x:{w}.{p}f}"


def pavlo_block(W):
    return 0.5 + (18.0 - 7.5) / (5120 - 2048) * W


def read_env(d):
    env = {}
    p = os.path.join(d, "session.env")
    if os.path.exists(p):
        for line in open(p):
            if "=" in line:
                k, v = line.rstrip("\n").split("=", 1)
                env[k] = v
    return env


def producer_logs(d):
    logs = [p for p in glob.glob(os.path.join(d, "*.log")) if os.path.basename(p) not in ("runner.log", "reset.log")]
    return sorted(logs, key=os.path.getmtime)


def streams(d):
    out = {}
    for log in producer_logs(d):
        name = os.path.basename(log)[:-4]
        txt = open(log, errors="replace").read()
        m, dr = DONE.search(txt), DRAIN.findall(txt)
        if m and dr:
            wall, toks, drain = float(m[1]), int(m[4]), float(dr[-1][2])
            out[name] = (int(m[2]), toks, wall + drain, toks / (wall + drain))
        else:
            out[name] = None
    return out


def chunk_labels(d):
    order = []
    for log in producer_logs(d):
        name = os.path.basename(log)[:-4]
        order += [name] * len(PUSH.findall(open(log, errors="replace").read()))
    return order


def parse_runner(d):
    t, st, cp = {}, {}, {}
    for line in open(os.path.join(d, "runner.log"), errors="replace"):
        if m := TIMING.search(line):
            t[(int(m[1]), int(m[2]))] = {k: float(v) for k, v in KV.findall(m[3])}
        elif m := START.search(line):
            st[(int(m[1]), int(m[2]))] = float(m[3])
        elif m := COMPUTE.search(line):
            cp[(int(m[1]), int(m[2]))] = float(m[3])
    return t, st, cp


def analyse(d):
    env = read_env(d)
    W = int(env.get("PREFILL_CHUNK_SIZE", "0") or 0)
    sync = env.get("PREFILL_SYNC_PER_CHUNK", "?")
    share = env.get("PREFILL_D2D_SHARE_FABRIC_LINKS", "1")
    t, st, cp = parse_runner(d)
    ranks = sorted({r for r, _ in t} | {r for r, _ in st})
    order = chunk_labels(d)
    res = {}
    for label in dict.fromkeys(order):
        if "warm" in label:
            continue
        cs = [c for c, lab in enumerate(order) if lab == label]
        per = {}
        for r in ranks:
            v = defaultdict(list)
            for c in cs:
                x = t.get((r, c))
                if x is None:
                    continue
                comp = cp.get((r, c)) or x.get("compute_ms")
                work = comp if comp is not None else x.get("enqueue_ms")
                v["compute"].append(comp)
                v["enqueue"].append(x.get("enqueue_ms"))
                v["push"].append(x.get("send_ms") if r < ranks[-1] else None)
                v["lease_in"].append(x.get("lease_in_ms"))
                v["lease_out"].append(x.get("lease_out_ms") if r < ranks[-1] else None)
                nxt = t.get((r, c + 1))
                # lease_out of THIS chunk's send is measured at the top of the next iteration
                lo = nxt.get("lease_out_ms") if nxt and r < ranks[-1] else None
                v["block"].append((x.get("send_ms") or 0.0) + lo if lo is not None else None)
                v["recv"].append(x.get("recv_ms"))
                v["pre_sync"].append(x.get("pre_sync_ms"))
                if nxt and "t_top" in nxt and "t_top" in x:
                    cyc = (nxt["t_top"] - x["t_top"]) * 1000.0
                    parts = [x.get(k) or 0.0 for k in ("lease_in_ms", "lease_out_ms", "recv_ms", "pre_sync_ms")]
                    v["cycle"].append(cyc)
                    v["host_gap"].append(cyc - sum(parts) - (work or 0.0) - (x.get("send_ms") or 0.0))
                if r < ranks[-1]:
                    y = t.get((r + 1, c))
                    if sync == "1" and y and "t_sent" in x and "t_recv" in y:
                        recv_began = y["t_recv"] - y.get("recv_ms", 0.0) / 1000.0
                        if recv_began <= x["t_sent"]:
                            v["hop"].append((y["t_recv"] - x["t_sent"]) * 1000.0)
                            if "t_start" in y:
                                v["hop_start"].append((y["t_start"] - x["t_sent"]) * 1000.0)
                    if (r, c) in st and (r, c) in cp and (r + 1, c) in st:
                        end_r = st[(r, c)] + cp[(r, c)] / 1000.0
                        prev = st.get((r + 1, c - 1), 0) + cp.get((r + 1, c - 1), 0) / 1000.0
                        if prev <= end_r:
                            v["hop_sp2"].append((st[(r + 1, c)] - end_r) * 1000.0)
            per[r] = {k: (med(x), len([y for y in x if y is not None])) for k, x in v.items()}
        # runner-side span: first chunk's t_start on the first stage -> last chunk's t_sent on the last stage
        # (host stamps; in a sync session t_sent follows the device sync, so it is the end of the compute)
        t0 = [t[(ranks[0], c)]["t_start"] for c in cs if "t_start" in t.get((ranks[0], c), {})]
        t1 = [t[(ranks[-1], c)]["t_sent"] for c in cs if "t_sent" in t.get((ranks[-1], c), {})]
        per["span"] = (max(t1) - min(t0), len(cs)) if t0 and t1 else None
        res[label] = per
    steady = {}
    for label, per in res.items():
        cyc = max((row.get("cycle", (None, 0))[0] or 0.0) for r, row in per.items() if r != "span")
        if cyc:
            steady[label] = W / cyc * 1000.0
    return dict(
        W=W,
        sync=sync,
        share=share,
        layers=env.get("PREFILL_NUM_LAYERS"),
        ranks=ranks,
        per=res,
        streams=streams(d),
        n_timing=len(t),
        steady=steady,
    )


COLS = [
    "compute",
    "enqueue",
    "push",
    "lease_out",
    "block",
    "lease_in",
    "recv",
    "pre_sync",
    "cycle",
    "host_gap",
    "hop",
    "hop_start",
    "hop_sp2",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="results_ops/pipeline (one sub-directory per session)")
    ap.add_argument("--out", help="write the report here as well as to stdout")
    args = ap.parse_args()

    lines = [
        "P1-D pipeline overheads, 4 x (2,4) common runner (medians in ms per stage; n = chunks in the median).",
        "block = push + lease_out (blocking send); hop only on chunks where the next stage was already waiting.",
        "",
    ]
    sessions = {}
    for d in sorted(glob.glob(os.path.join(args.root, "*"))):
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "runner.log")):
            sessions[os.path.basename(d)] = analyse(d)
    for name, a in sessions.items():
        lines.append(
            f"## {name}  W={a['W']} sync={a['sync']} share_fabric_links={a['share']}  "
            f"(PP_TIMING lines: {a['n_timing']}; Pavlo block fit at this W: {pavlo_block(a['W']):.1f} ms)"
        )
        for s, v in a["streams"].items():
            if "warm" in s:
                continue
            lines.append(
                f"  {s:12} "
                + ("(no DONE / drain line)" if v is None else f"chunks {v[0]:3d} time {v[2]:6.2f} s  tok/s {v[3]:7.0f}")
            )
        for label, per in a["per"].items():
            span = per.pop("span", None)
            if not any(n for row in per.values() for _, n in row.values()):
                lines.append(f"  [{label}]  no PP_TIMING / CHUNK_COMPUTE data")
                continue
            lines.append(f"  [{label}]  " + " ".join(f"{c:>9}" for c in COLS))
            for r in a["ranks"]:
                row = per.get(r, {})
                lines.append(
                    f"    s{r}{'':8}"
                    + " ".join(fmt(row.get(c, (None, 0))[0], 9) for c in COLS)
                    + f"   n={max((x[1] for x in row.values()), default=0)}"
                )
            cyc = max((per[r].get("cycle", (None, 0))[0] or 0.0) for r in per)
            if cyc:
                lines.append(
                    f"    runner: steady {a['W'] / cyc * 1000:.0f} tok/s (W / slowest median cycle {cyc:.1f} ms)"
                    + (
                        f"; span {span[0]:.2f} s for {span[1]} chunks -> {span[1] * a['W'] / span[0]:.0f} tok/s"
                        if span
                        else ""
                    )
                )
            comp = {r: per[r].get("compute", (None, 0))[0] for r in per}
            blk = {r: per[r].get("block", (None, 0))[0] or 0.0 for r in per}
            if all(x for x in comp.values()) and comp:
                b = max(comp, key=comp.get)
                bb = max(comp, key=lambda r: comp[r] + blk[r])
                lines.append(
                    f"    model: bottleneck s{b} compute {comp[b]:.1f} ms -> {a['W'] / comp[b] * 1000:.0f} tok/s; "
                    f"with the blocking send s{bb} {comp[bb] + blk[bb]:.1f} ms -> "
                    f"{a['W'] / (comp[bb] + blk[bb]) * 1000:.0f} tok/s"
                )
        lines.append("")

    # sync model vs un-synced measurement, per W and stream kind
    lines.append("## Model (sync sessions) vs un-synced tok/s, same W and stream kind")
    for name, a in sessions.items():
        if a["sync"] != "1":
            continue
        for label, per in a["per"].items():
            per = {r: v for r, v in per.items() if r != "span"}
            comp = {r: per[r].get("compute", (None, 0))[0] for r in per}
            if not comp or not all(comp.values()):
                continue
            blk = {r: per[r].get("block", (None, 0))[0] or 0.0 for r in per}
            m0 = a["W"] / max(comp.values()) * 1000
            m1 = a["W"] / max(comp[r] + blk[r] for r in comp) * 1000
            for other, b in sessions.items():
                if b["sync"] == "1" or b["W"] != a["W"] or b["layers"] != a["layers"]:
                    continue
                v = b["streams"].get(label)
                sb = b["steady"].get(label)
                if v:
                    lines.append(
                        f"  W={a['W']} {label}: model {m0:.0f} tok/s (compute) / {m1:.0f} (compute + block); "
                        f"{other} producer {v[3]:.0f} ({v[3] / m0 - 1:+.1%} vs compute-only, {v[3] / m1 - 1:+.1%} vs +block)"
                        + (f", runner steady {sb:.0f} ({sb / m0 - 1:+.1%} / {sb / m1 - 1:+.1%})" if sb else "")
                    )
    lines.append("")
    lines.append("## Async handoff A/B (lease vs OWN, un-synced, same W)")
    for name, a in sessions.items():
        if a["sync"] == "1" or a["share"] != "0":
            continue
        for other, b in sessions.items():
            if b["sync"] == "1" or b["share"] == "0" or b["W"] != a["W"] or b["layers"] != a["layers"]:
                continue
            for s, v in a["streams"].items():
                w = b["streams"].get(s)
                if "warm" in s or not v or not w:
                    continue
                ls, os_ = b["steady"].get(s), a["steady"].get(s)
                lines.append(
                    f"  W={a['W']} {s:12} producer: lease {w[3]:7.0f} tok/s  own {v[3]:7.0f} tok/s  ({v[3] / w[3] - 1:+.1%})"
                    + (
                        f";  runner steady: lease {ls:7.0f}  own {os_:7.0f}  ({os_ / ls - 1:+.1%})"
                        if ls and os_
                        else ""
                    )
                    + f"  [{other} vs {name}]"
                )
    for name, a in sessions.items():
        wd = os.path.join(args.root, name, "watchdog")
        if a["share"] == "0" and os.path.exists(wd):
            last = {}
            for r, c in parse_runner(os.path.join(args.root, name))[0]:
                last[r] = max(last.get(r, -1), c)
            lines.append(
                f"  {name}: {open(wd).read().strip()} (killed by the watchdog); last PP_TIMING chunk per rank: "
                + (", ".join(f"s{r} c={c}" for r, c in sorted(last.items())) or "none")
                + f"; streams {', '.join(k for k, v in a['streams'].items() if v is None) or '-'} without a DONE"
            )
    lines.append("")
    lines.append("## KV read-back PCC vs the golden (PREFILL_PRODUCER_CHECK_PCC=1 streams; last request per slot)")
    for name, a in sessions.items():
        for log in producer_logs(os.path.join(args.root, name)):
            txt = open(log, errors="replace").read()
            m = PCC.search(txt)
            if m or "verify=True" in txt:
                verdict = "PASSED" if "KV cache PCC PASSED" in txt else "below threshold" if m else "no result"
                lines.append(
                    f"  {name} share_fabric_links={a['share']} {os.path.basename(log)[:-4]}: {verdict}"
                    + (f"  {m[0]}" if m else "")
                )
    per_layer = {}  # session -> {(slot pass, layer, cache): pcc}
    for name in sessions:
        for log in producer_logs(os.path.join(args.root, name)):
            vals = LAYER_PCC.findall(open(log, errors="replace").read())
            if vals:
                seen = defaultdict(int)
                d = {}
                for layer, rest in vals:
                    n = seen[layer]
                    seen[layer] += 1
                    for cache, v in re.findall(r"(\w+)=([\d.]+)", rest):
                        d[(n, int(layer), cache)] = float(v)
                per_layer[name] = d
    for name, d in per_layer.items():
        for other, e in per_layer.items():
            if other == name or sessions[name]["share"] != "0" or sessions[other]["share"] == "0":
                continue
            common = set(d) & set(e)
            if common:
                k = max(common, key=lambda k: abs(d[k] - e[k]))
                lines.append(
                    f"  {name} vs {other}: {len(common)} per-layer PCCs, max |diff| {abs(d[k] - e[k]):.5f} "
                    f"(slot pass {k[0]}, layer {k[1]} {k[2]}: {d[k]:.5f} vs {e[k]:.5f})"
                )
    txt = "\n".join(lines) + "\n"
    print(txt, end="")
    if args.out:
        with open(args.out, "w") as f:
            f.write(txt)


if __name__ == "__main__":
    main()
