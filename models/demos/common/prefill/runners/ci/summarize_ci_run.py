# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
import argparse
import json
import os
import re

_KV = re.compile(r"slot\s+(\d+)\s+layer\s+(\d+)\s+KV PCC:\s+nope=([-\d.]+)\s+pe=([-\d.]+)")
_GQA = re.compile(r"layer\s+(\d+):\s+K=([-\d.]+)\s+V=([-\d.]+)")
_INDEX = re.compile(r"slot\s+(\d+)\s+layer\s+(\d+)\s+\(index rank\s+(\d+)\)\s+index PCC:\s+([-\d.]+)")
_CHUNK_START = re.compile(r"\[pp rank (\d+)\] CHUNK_START c=(\d+) compute_start=([\d.]+)")
_CHUNK_COMPUTE = re.compile(r"\[pp rank (\d+)\] CHUNK_COMPUTE c=(\d+) compute_ms=([\d.]+)")
# prefill_runner's configuration block, one per rank: what the runner actually ran, whatever the manifest says.
_RUNNER_CONFIG = re.compile(
    r"^\s*(?:\[[\d,]+\]<\w+>:)?\s*(PREFILL_USE_TRACE|PREFILL_NUM_USERS|PREFILL_NUM_LAYERS)\s+= (\S+)"
)

# Chunks spanned by a throughput probe. A single chunk can run half again as long as its neighbours,
# so one interval is too noisy to be a rate; eight is still under 5% of the shortest measured request.
_TPUT_WINDOW = 8


def _iter_lines(root):
    for dirpath, _dirs, files in os.walk(root):
        for name in sorted(files):
            path = os.path.join(dirpath, name)
            try:
                with open(path, errors="replace") as fh:
                    for line in fh:
                        yield line
            except OSError:
                continue


def _pcc_matrix(root):
    layers = {}
    have_index = False
    have_mla = False
    have_gqa = False
    for line in _iter_lines(root):
        m = _KV.search(line)
        if m:
            layer, nope, pe = int(m.group(2)), float(m.group(3)), float(m.group(4))
            cell = layers.setdefault(layer, {})
            cell["nope"] = min(nope, cell.get("nope", nope))
            cell["pe"] = min(pe, cell.get("pe", pe))
            have_mla = True
            continue
        m = _GQA.search(line)
        if m:
            layer, key, value = int(m.group(1)), float(m.group(2)), float(m.group(3))
            cell = layers.setdefault(layer, {})
            cell["k"] = min(key, cell.get("k", key))
            cell["v"] = min(value, cell.get("v", value))
            have_gqa = True
            continue
        m = _INDEX.search(line)
        if m:
            layer, idx = int(m.group(2)), float(m.group(4))
            cell = layers.setdefault(layer, {})
            cell["index"] = min(idx, cell.get("index", idx))
            have_index = True

    print("==================== per-layer x per-cache KV PCC vs golden ====================")
    if not layers:
        print("no per-layer PCC lines found in ranklogs (producer verify may not have run)")
        return
    columns = [("nope", "kvpe.nope"), ("pe", "kvpe.pe")] if have_mla or not have_gqa else []
    if have_gqa:
        columns += [("k", "K"), ("v", "V")]
    header = f"{'layer':>5}" + "".join(f"  {label:>10}" for _, label in columns)
    print(header + (f"  {'index':>10}" if have_index else ""))
    worst = 1.0
    for layer in sorted(layers):
        cell = layers[layer]
        row = f"{layer:>5}" + "".join(f"  {cell.get(key, float('nan')):>10.5f}" for key, _ in columns)
        if have_index:
            row += f"  {cell.get('index', float('nan')):>10.5f}" if "index" in cell else f"  {'-':>10}"
        print(row)
        worst = min([worst] + [v for v in cell.values()])
    print(f"{'min':>5}  (worst cell across all layers/caches) -> {worst:.6f}")


def _timing_from_csvs(timing_dir):
    if not timing_dir or not os.path.isdir(timing_dir):
        return None
    ranks = {}
    for name in sorted(os.listdir(timing_dir)):
        if not name.endswith(".csv"):
            continue
        try:
            with open(os.path.join(timing_dir, name), errors="replace") as fh:
                for line in fh:
                    parts = line.strip().split(",")
                    if len(parts) != 4:
                        continue
                    try:
                        rank, c, start, ms = int(parts[0]), int(parts[1]), float(parts[2]), float(parts[3])
                    except ValueError:
                        continue
                    ranks.setdefault(rank, {})[c] = [start, ms]
        except OSError:
            continue
    return ranks or None


def _timing_from_ranklogs(root):
    ranks = {}
    for line in _iter_lines(root):
        m = _CHUNK_START.search(line)
        if m:
            ranks.setdefault(int(m.group(1)), {}).setdefault(int(m.group(2)), [None, None])[0] = float(m.group(3))
            continue
        m = _CHUNK_COMPUTE.search(line)
        if m:
            ranks.setdefault(int(m.group(1)), {}).setdefault(int(m.group(2)), [None, None])[1] = float(m.group(3))
    return ranks


def _end(cell):
    start, ms = cell
    return None if (start is None or ms is None) else start + ms / 1000.0


def _select_measured(ranks, real_chunks, warmup_chunks=0):
    kept = {}
    cs_union = set()
    for rank, cells in ranks.items():
        cs = sorted(cells)
        # A run that died mid-request has fewer than real_chunks after its warmup; keeping the last real_chunks
        # would slide the warmup onto the shallow probes, so skip the warmup from the front instead.
        if real_chunks > 0 and len(cs) < warmup_chunks + real_chunks:
            sel = cs[warmup_chunks:]
        else:
            sel = cs[-real_chunks:] if real_chunks > 0 else cs
        kept[rank] = {c: cells[c] for c in sel}
        cs_union.update(sel)
    cs_sorted = sorted(cs_union)
    disp = {c: i for i, c in enumerate(cs_sorted)}
    return kept, cs_sorted, disp


def _cell_metrics(kept, disp):
    inv = {i: c for c, i in disp.items()}
    ct, end = {}, {}
    for d, c in inv.items():
        starts = [kept[r][c][0] for r in kept if c in kept[r] and kept[r][c][0] is not None]
        ends = [_end(kept[r][c]) for r in kept if c in kept[r] and _end(kept[r][c]) is not None]
        if starts and ends:
            ct[d] = (max(ends) - min(starts)) * 1000.0
            end[d] = max(ends)
    t0 = None
    if 0 in inv:
        s0 = [kept[r][inv[0]][0] for r in kept if inv[0] in kept[r] and kept[r][inv[0]][0] is not None]
        t0 = min(s0) if s0 else None
    ttft = {d: end[d] - t0 for d in end} if t0 is not None else {}
    return ct, ttft, end


def _occupancy(kept):
    spans, busy = {}, {}
    for rank, cells in kept.items():
        cs = [c for c in sorted(cells) if cells[c][0] is not None and cells[c][1] is not None]
        # A single measured chunk makes the span identically the compute, so occupancy is 1 by construction.
        if len(cs) < 2:
            return [], {}
        # Span and busy are both single-host deltas, so these ratios carry no inter-host clock skew,
        # and each rank's own fill and drain fall outside its own span.
        spans[rank] = _end(cells[cs[-1]]) - cells[cs[0]][0]
        busy[rank] = sum(cells[c][1] for c in cs) / 1000.0
    if len(spans) < 2 or min(spans.values()) <= 0:
        return [], {}

    b = max(busy, key=lambda r: busy[r])
    rec = {
        "balance": (sum(busy.values()) / len(busy)) / busy[b],
        "bottleneck_saturation": busy[b] / spans[b],
        "global_occupancy": sum(busy.values()) / sum(spans.values()),
        "bottleneck_rank": b,
        "rank_span_s": {str(r): spans[r] for r in sorted(spans)},
        "rank_busy_s": {str(r): busy[r] for r in sorted(busy)},
    }
    out = [
        "occupancy: span = a rank's own first measured chunk start -> its own last measured chunk end;"
        " busy = its summed device compute",
        f"  balance               = {rec['balance']:.4f}",
        f"  bottleneck_saturation = {rec['bottleneck_saturation']:.4f}",
        f"  global_occupancy      = {rec['global_occupancy']:.4f}",
    ]
    return out, rec


def _runner_config(root):
    """{traced, num_users, num_layers} as the runner ranks printed them; a list where the ranks disagree."""
    seen = {}
    for line in _iter_lines(root):
        m = _RUNNER_CONFIG.match(line)
        if m:
            seen.setdefault(m.group(1), set()).add(m.group(2))
    parse = {"PREFILL_USE_TRACE": ("traced", lambda v: v == "True")}
    parse.update({k: (k.removeprefix("PREFILL_").lower(), int) for k in ("PREFILL_NUM_USERS", "PREFILL_NUM_LAYERS")})
    config = {}
    for key, values in seen.items():
        field, cast = parse[key]
        values = sorted(cast(v) for v in values)
        config[field] = values[0] if len(values) == 1 else values
    return config


def _run_lines(run):
    """What a perf number belongs to: commit, traced or eager, slots, and every host's power class."""
    hosts = run.get("hosts") or []
    lines = [
        "==================== run ====================",
        f"commit={run.get('commit', '?')} traced={run.get('traced', '?')} num_users={run.get('num_users', '?')} "
        f"num_layers={run.get('num_layers', '?')}",
    ]
    lines += [
        f"  host {h.get('host')}: TDP_LIMIT_MAX={h.get('tdp_limit_max')} -> {h.get('power_class')} "
        f"(hostname {h.get('hostname')})"
        for h in hosts
    ] or ["  no host power recorded"]
    return lines


def _publish(lines, name):
    title = f"disaggregated prefill perf -- {name or 'run'}"
    print(title)
    print("\n".join(lines))
    root = os.environ.get("PREFILL_SUMMARIES")
    if not root or not name:
        return
    # perf/*.md is concatenated into the job summary by the workflow's publish step.
    try:
        out = os.path.join(root, "perf")
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, f"{name}.md"), "w") as fh:
            fh.write(f"### {title}\n\n```text\n" + "\n".join(lines) + "\n```\n")
    except OSError as exc:
        print(f"perf summary not persisted ({exc})")


def _publish_json(rec, name):
    if not rec or not name:
        return
    root = os.environ.get("PREFILL_SUMMARIES")
    if not root:
        print("perf metrics not persisted (PREFILL_SUMMARIES unset)")
        return
    model, _, config = name.rpartition("_")
    rec = {"name": name, "model": model or name, "config": config or "unknown", **rec}
    try:
        out = os.path.join(root, "perf_json")
        os.makedirs(out, exist_ok=True)
        path = os.path.join(out, f"{name}.json")
        with open(path, "w") as fh:
            json.dump(rec, fh, indent=2, sort_keys=True)
            fh.write("\n")
        print(f"perf metrics -> {path}")
    except OSError as exc:
        print(f"perf metrics not persisted ({exc})")


def _perf_metrics(kept, cs_sorted, disp, chunk_size, probe_chunks=None):
    n = len(cs_sorted)
    if n == 0:
        return [], {}
    max_seq = n * chunk_size
    ranks = len(kept)
    ct, ttft, end = _cell_metrics(kept, disp)
    rec = {
        "chunk_size": chunk_size,
        "num_chunks": n,
        "max_seq": max_seq,
        "pipeline_ranks": ranks,
        "chunk_index": {},
        "chunk_time_ms": {},
        "ttft_s": {},
        "throughput_tok_s": {},
    }

    def probes(offsets, *, skip_first_chunk=False):
        if probe_chunks is None:
            picked = {max(0, min(n - 1, tok // chunk_size)) for tok in offsets}
        else:
            picked = {d for d in probe_chunks if 0 <= d < n}
        if skip_first_chunk:
            picked.discard(0)
        picked = sorted(picked)
        toks = [(d + 1) * chunk_size for d in picked]
        labels = [f"@{round(t / 1048576)}M" if t >= 1048576 else f"@{round(t / 1024)}k" for t in toks]
        if len(set(labels)) != len(labels):
            labels = [f"@{t}" for t in toks]
        return list(zip(labels, picked))

    out = ["==================== perf metrics (measured request, warmup excluded) ===================="]
    out.append(f"max_seq={max_seq} tok ({n} chunks x {chunk_size})")
    out.append("labels are tokens prefilled through that chunk, in units of 1024 (M past 1Mi); probes snap to chunks")
    out.append("chunk_time = first-rank start -> last-rank finish (cross-rank; assumes NTP-comparable clocks)")
    for lbl, d in probes((0, 50000, max_seq // 2, max_seq - chunk_size)):
        rec["chunk_index"][lbl] = d
        val = f"{ct[d]:>12.3f}" if d in ct else f"{'-':>12}"
        out.append(f"  chunk_time {lbl:>14} (chunk {d:>3}): {val} ms")
        if d in ct:
            rec["chunk_time_ms"][lbl] = ct[d]
    out.append("ttft = request start -> chunk finish")
    for lbl, d in probes((50000, max_seq // 2, max_seq), skip_first_chunk=True):
        rec["chunk_index"][lbl] = d
        val = f"{ttft[d]:>12.3f}" if d in ttft else f"{'-':>12}"
        out.append(f"  ttft       {lbl:>14} (chunk {d:>3}): {val} s")
        if d in ttft:
            rec["ttft_s"][lbl] = ttft[d]

    out.append(f"throughput = tokens / wall time across up to {_TPUT_WINDOW} chunk completions centred on the probe")
    for lbl, d in probes((0, 50000, max_seq // 2, max_seq)):
        # The rate is the completion cadence. A chunk's own latency covers every stage at once and so
        # carries no information about how many chunks are in flight; only the interval between
        # completions does. The window absorbs one slow chunk while staying short enough that the
        # attention cost, which grows with depth, barely drifts across it.
        lo, hi = max(0, d - _TPUT_WINDOW // 2), min(n - 1, d + _TPUT_WINDOW // 2)
        if hi - lo < 1 or end.get(lo) is None or end.get(hi) is None or end[hi] <= end[lo]:
            out.append(f"  throughput {lbl:>14} (chunk {d:>3}): {'-':>12}")
            continue
        span = end[hi] - end[lo]
        tput = (hi - lo) * chunk_size / span
        out.append(f"  throughput {lbl:>14} (chunk {d:>3}): {tput:>12,.1f} tok/s")
        rec["throughput_tok_s"][lbl] = tput

    # Occupancy compares a rank against its own timeline, so it is meaningless without a pipeline.
    if ranks >= 2:
        occ_lines, occ_rec = _occupancy(kept)
        out.extend(occ_lines)
        rec.update(occ_rec)
    return out, rec


def _timing_matrix(root, real_chunks, timing_dir=None, chunk_size=0, probe_chunks=None, warmup_chunks=0):
    ranks = _timing_from_csvs(timing_dir)
    if ranks is None:
        ranks = _timing_from_ranklogs(root)

    print("==================== per-rank x per-chunk timing (measured request) ====================")
    if not ranks:
        print("no timing rows found (set PREFILL_SYNC_PER_CHUNK=1 on the runner; timing CSVs / CHUNK_* logs absent)")
        return [], {}
    kept, cs_sorted, disp = _select_measured(ranks, real_chunks, warmup_chunks)
    all_starts = [c[0] for r in kept.values() for c in r.values() if c[0] is not None]
    if not all_starts:
        print("timing rows present but no compute_start timestamps parsed")
        return [], {}
    t0 = min(all_starts)
    print(f"start/end are seconds relative to the earliest chunk start ({t0:.6f} epoch); ms = device compute time")
    print(f"{'rank':>4}  {'chunk':>5}  {'start_s':>10}  {'end_s':>10}  {'ms':>9}")
    for rank in sorted(kept):
        for c in sorted(kept[rank]):
            start, ms = kept[rank][c]
            start_s = f"{start - t0:>10.3f}" if start is not None else f"{'-':>10}"
            end_s = f"{start - t0 + ms / 1000.0:>10.3f}" if (start is not None and ms is not None) else f"{'-':>10}"
            ms_s = f"{ms:>9.3f}" if ms is not None else f"{'-':>9}"
            print(f"{rank:>4}  {disp[c]:>5}  {start_s}  {end_s}  {ms_s}")

    if chunk_size > 0:
        return _perf_metrics(kept, cs_sorted, disp, chunk_size, probe_chunks)
    return [], {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranklogs", required=True, help="mpirun --output-filename root (runner + producer)")
    ap.add_argument("--timing-dir", default=None, help="per-rank timing CSV dir (preferred timing source)")
    ap.add_argument("--real-chunks", type=int, default=0, help="chunks in the measured request (0 => all)")
    ap.add_argument("--warmup-chunks", type=int, default=0, help="untimed chunks the producer sends before it")
    ap.add_argument("--chunk-size", type=int, default=0, help="tokens per chunk (0 => skip throughput)")
    ap.add_argument(
        "--probe-chunks",
        default=None,
        help="comma-separated chunk indices to probe, shared across configs (default: derived from this request)",
    )
    ap.add_argument(
        "--summary-name", default=None, help="<model>_<config>; names the metrics sidecar the scaling job reads"
    )
    ap.add_argument("--run-context", default=None, help="JSON from the harness: commit and every host's power")
    args = ap.parse_args()
    if not os.path.isdir(args.ranklogs):
        print(f"ranklogs dir {args.ranklogs} not found; nothing to summarize")
        return
    run = {}
    if args.run_context:
        try:
            with open(args.run_context) as fh:
                run = json.load(fh)
        except (OSError, ValueError) as exc:
            print(f"run context unreadable ({exc})")
    run.update(_runner_config(args.ranklogs))
    _pcc_matrix(args.ranklogs)
    probe_chunks = None
    if args.probe_chunks:
        probe_chunks = {int(tok) for tok in args.probe_chunks.split(",") if tok.strip()}
    lines, rec = _timing_matrix(
        args.ranklogs, args.real_chunks, args.timing_dir, args.chunk_size, probe_chunks, args.warmup_chunks
    )
    # Written even when nothing was timed, so a failed run still says what it was.
    _publish(_run_lines(run) + lines, args.summary_name)
    _publish_json({**rec, **run}, args.summary_name)


if __name__ == "__main__":
    main()
