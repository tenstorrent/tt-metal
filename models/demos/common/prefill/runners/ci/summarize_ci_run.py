# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
import argparse
import json
import math
import os
import re
import statistics

_KV = re.compile(r"slot\s+(\d+)\s+layer\s+(\d+)\s+KV PCC:\s+nope=([-\d.]+)\s+pe=([-\d.]+)")
_GQA = re.compile(r"layer\s+(\d+):\s+K=([-\d.]+)\s+V=([-\d.]+)")
_INDEX = re.compile(r"slot\s+(\d+)\s+layer\s+(\d+)\s+\(index rank\s+(\d+)\)\s+index PCC:\s+([-\d.]+)")
_CHUNK_RETIRE = re.compile(
    r"\[producer\] CHUNK_RETIRE slot=(\d+) start=(\d+) end=(\d+) first=([\d.]+) retire=([\d.]+) warmup=([01])"
    r"(?: acks=(\d+))?(?: layers=(\S+))?"
)
# One entry of a CHUNK_RETIRE line's layers= field: layers [a, b) acked t ms after the chunk's first ack.
_LAYER_ACK = re.compile(r"(\d+)-(\d+)@([\d.]+)")
# A runner rank's start-up line: the layers [lo, hi) its pipeline stage owns.
_RANK_LAYERS = re.compile(r"\[pp rank (\d+)/(\d+)\] .*?layers=\[(\d+), (\d+)\)")
# A stage whose acks are closer than this apart (median gap) acked as a batch, not layer by layer: the producer
# drains a batch in well under a millisecond, while a layer takes several.
_MIN_LAYER_ACK_GAP_S = 0.5e-3
# Each runner rank's start-up line (prefill_runner.py).
_RANK_SETUP = re.compile(
    r"\[pp rank (?P<rank>\d+)/(?P<ranks>\d+)\] mesh=\((?P<rows>\d+), (?P<cols>\d+)\) "
    r"layers=\[(?P<lo>\d+), (?P<hi>\d+)\) .*?chunk_size=(?P<chunk>\d+) forward_chunks=(?P<cpf>\d+) "
    r"max_seq_len=\d+ num_users=(?P<users>\d+) "
    r"sp=(?P<sp>\d+) tp=(?P<tp>\d+) sp_axis=(?P<axis>\d+) traced=(?P<traced>[01])"
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
    # The config is the scale-out suffix plus any layout (sc1, sc4, sc4_8x4), so "kimi_k3_sc4_8x4" is kimi_k3 at
    # sc4_8x4; splitting at the last underscore made it kimi_k3_sc4 at 8x4, which no baseline or summary row names.
    m = re.fullmatch(r"(.+)_(sc\d+(?:_[0-9a-z]+)?)", name)
    model, config = m.groups() if m else (name.rpartition("_")[0], name.rpartition("_")[2])
    rec = {"name": name, "model": model or name, "config": config or "unknown", **rec}
    try:
        out = os.path.join(root, "perf_json")
        os.makedirs(out, exist_ok=True)
        path = os.path.join(out, f"{name}.json")
        with open(path, "w") as fh:
            json.dump(_json_safe(rec), fh, indent=2, sort_keys=True, allow_nan=False)
            fh.write("\n")
        print(f"perf metrics -> {path}")
    except OSError as exc:
        print(f"perf metrics not persisted ({exc})")


def _json_safe(value):
    """value with NaN as None: JSON has no NaN, and its readers take null as missing."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _spans_from_logs(root, real_chunks):
    """(first ack, retire, acks, [(layer lo, layer hi, ack time)]) per measured chunk, on the producer clock."""
    spans = []
    for line in _iter_lines(root):
        m = _CHUNK_RETIRE.search(line)
        if m and m.group(6) == "0":
            first = float(m.group(4))
            layers = [(int(a), int(b), first + float(t) / 1000.0) for a, b, t in _LAYER_ACK.findall(m.group(8) or "")]
            spans.append((first, float(m.group(5)), int(m.group(7) or 0), layers))
    spans.sort()
    return spans[-real_chunks:] if real_chunks > 0 else spans


def _stage_ranges(root):
    """Each pipeline stage's layers [lo, hi), in stage order; [] unless every runner rank logged them."""
    ranks, num_ranks = {}, 0
    for line in _iter_lines(root):
        m = _RANK_LAYERS.search(line)
        if m:
            ranks[int(m[1])] = (int(m[3]), int(m[4]))
            num_ranks = int(m[2])
    return [ranks[r] for r in range(num_ranks)] if sorted(ranks) == list(range(num_ranks)) else []


def _stage_timing(acks, stages):
    """One forward's (working ms, last ack) per stage; None when its acks do not split at the stage boundaries.

    Working ms is the stage's first to last layer ack, scaled to all its layers (the first ack's layers count at the
    others' average). Both acks come from the stage's own layers, so a queue in front of it or a busy stage after it
    does not move them. Only the trunk layers count (an MTP level's acks are outside every stage). NaN when the acks
    do not show when the stage started: acked as a batch, or in one ack."""
    out = []
    for lo, hi in stages:
        own = [(a, b, t) for a, b, t in acks if lo <= a < hi]
        firsts = [(b - a, t) for a, b, t in own if a == lo]
        if not firsts or any(b > hi for _a, b, _t in own) or not any(b == hi for _a, b, _t in own):
            return None
        first_layers, first = firsts[0]
        times = sorted(t for _a, _b, t in own)
        gaps = [b - a for a, b in zip(times, times[1:])]
        spread = hi - lo > first_layers and statistics.median(gaps) >= _MIN_LAYER_ACK_GAP_S
        work = (times[-1] - first) * (hi - lo) / (hi - lo - first_layers) * 1000.0 if spread else math.nan
        out.append((work, times[-1]))
    return out


def _stage_times(spans, stages, cpf):
    """Per forward, keyed by its last chunk: ([working ms per stage, per chunk], queued ms, latency ms); {} when no
    forward splits by stage.

    A packed forward's stage time is divided by its real chunk count. queued sums, over stages k >= 1, how long stage
    k was still busy with the previous forward after stage k-1 finished this one; a queue before stage 0 is not seen.
    latency is the forward's first to last layer ack, over all its chunks. A forward that does not split has NaN stage
    times and queued,
    as has the next forward's queued."""
    ends = [i for i in range(len(spans)) if (i + 1) % cpf == 0 or i == len(spans) - 1]
    timing = {i: _stage_timing(spans[i][3], stages) for i in ends} if stages else {}
    if not any(timing.values()):
        return {}
    out, prev = {}, None
    for j, i in enumerate(ends):
        t = timing[i]
        first = min(span[0] for span in spans[(ends[j - 1] + 1 if j else 0) : i + 1])
        latency = (spans[i][1] - first) * 1000.0
        if t is None:
            out[i] = ([math.nan] * len(stages), math.nan, latency)
        else:
            if j == 0:
                queued = 0.0
            elif prev is None:
                queued = math.nan
            else:
                queued = sum(max(0.0, prev[k][1] - t[k - 1][1]) for k in range(1, len(stages))) * 1000.0
            chunks = i - (ends[j - 1] if j else -1)
            out[i] = ([work / chunks for work, _last in t], queued, latency)
        prev = t
    return out


def _finite_median(xs):
    """Median of the finite values; NaN unless they are most of xs, so one unsplit forward does not lose a probe."""
    finite = [x for x in xs if math.isfinite(x)]
    return statistics.median(finite) if 2 * len(finite) > len(xs) else math.nan


def _setup_from_logs(root):
    """What ran, from the runner ranks' start-up lines: stages x stage mesh, the SP axis, each stage's layers, slots,
    chunking and packing, tracing. Empty when no rank printed one."""
    ranks = {}
    for line in _iter_lines(root):
        m = _RANK_SETUP.search(line)
        if m:
            ranks[int(m["rank"])] = m.groupdict()
    if not ranks:
        return ""
    g = ranks[min(ranks)]
    stages = (
        f"{g['ranks']} stage{'s' if g['ranks'] != '1' else ''} x [{g['rows']},{g['cols']}]"
        f" (SP{g['sp']} on mesh axis {g['axis']}, TP{g['tp']})"
    )
    counts = [int(r["hi"]) - int(r["lo"]) for _, r in sorted(ranks.items())]
    layers = f"{sum(counts)} layers" + (f" as {'+'.join(map(str, counts))}" if len(counts) > 1 else "")
    if len(ranks) < int(g["ranks"]):
        layers += f" ({len(ranks)} of {g['ranks']} ranks reported)"
    cpf = int(g["cpf"])
    packing = f"{cpf} per forward (packed)" if cpf > 1 else "1 per forward"
    tracing = "traced" if g["traced"] == "1" else "eager"
    return ", ".join([stages, layers, f"{g['users']} slots", f"{g['chunk']}-token chunks, {packing}", tracing])


def _perf_metrics(spans, chunk_size, pipeline_ranks, probe_chunks=None, chunks_per_forward=1, setup="", stages=()):
    n = len(spans)
    if n == 0 or chunk_size <= 0:
        return [], {}
    max_seq = n * chunk_size
    t0 = spans[0][0]
    end = [span[1] for span in spans]
    whole_chunk_acks = all(span[2] == 1 for span in spans)
    rec = {
        "setup": setup,
        "chunk_size": chunk_size,
        "num_chunks": n,
        "max_seq": max_seq,
        "pipeline_ranks": pipeline_ranks,
        "chunks_per_forward": chunks_per_forward,
        "chunk_index": {},
        "chunk_time_ms": {},
        "stage_ms": {},
        "queued_ms": {},
        "latency_ms": {},
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
    if setup:
        out.append(f"run: {setup}")
    cpf = max(1, chunks_per_forward)
    packing = f"packed, {cpf} chunks per forward" if cpf > 1 else "unpacked, one chunk per forward"
    out.append(f"max_seq={max_seq} tok ({n} chunks x {chunk_size}; {packing})")
    out.append("labels are tokens prefilled through that chunk, in units of 1024 (M past 1Mi); probes snap to chunks")
    out.append("all times are the producer's clock, from the runner's layer acks")
    if whole_chunk_acks:
        out.append("chunk_time, ttft: n/a (the runner acks each chunk in one message)")
    else:
        _chunk_time(out, rec, spans, stages, cpf, probes((0, 50000, max_seq // 2, max_seq - chunk_size)))
        out.append("ttft = the request's first layer ack -> the chunk's last layer ack")
        for lbl, d in probes((50000, max_seq // 2, max_seq), skip_first_chunk=True):
            rec["chunk_index"][lbl] = d
            rec["ttft_s"][lbl] = end[d] - t0
            out.append(f"  ttft       {lbl:>14} (chunk {d:>3}): {rec['ttft_s'][lbl]:>12.3f} s")

    out.append(f"throughput = tokens / wall time across up to {_TPUT_WINDOW} chunk retires centred on the probe")
    if cpf > 1:
        out.append("  (a packed forward's chunks retire together: windows span whole forwards)")
    # A packed forward retires all its chunks at once, so only the last chunk of a forward (or the
    # request's last chunk, which closes a short final forward) marks a point where the rate is known.
    # A short final forward counts: whatever it costs beyond its share is time the request really spends.
    forward_ends = [i for i in range(n) if (i + 1) % cpf == 0 or i == n - 1]
    for lbl, d in probes((0, 50000, max_seq // 2, max_seq)):
        # The rate is the retire cadence. A chunk's own latency covers every stage at once and so
        # carries no information about how many chunks are in flight; only the interval between
        # retires does. The window absorbs one slow chunk while staying short enough that the
        # attention cost, which grows with depth, barely drifts across it.
        lo, hi = max(0, d - _TPUT_WINDOW // 2), min(n - 1, d + _TPUT_WINDOW // 2)
        # Snap to forward ends: credit exactly the chunks retired in (end[lo], end[hi]]. Otherwise a
        # window edge inside a forward counts chunks that retired at the window's start, e.g. 4 chunks
        # against the time of a 2-chunk tail forward. One chunk per forward leaves lo and hi as they are.
        lo = max((i for i in forward_ends if i <= lo), default=forward_ends[0])
        hi = min((i for i in forward_ends if i >= hi), default=forward_ends[-1])
        if hi <= lo:  # both ends on one forward end (the first probe, cpf > window): take a neighbour
            later = [i for i in forward_ends if i > lo]
            earlier = [i for i in forward_ends if i < lo]
            lo, hi = (lo, later[0]) if later else ((earlier[-1], lo) if earlier else (lo, hi))
        if hi - lo < 1 or end[hi] <= end[lo]:
            out.append(f"  throughput {lbl:>14} (chunk {d:>3}): {'-':>12}")
            continue
        tput = (hi - lo) * chunk_size / (end[hi] - end[lo])
        out.append(f"  throughput {lbl:>14} (chunk {d:>3}): {tput:>12,.1f} tok/s")
        rec["throughput_tok_s"][lbl] = tput
    return out, rec


def _chunk_time(out, rec, spans, stages, cpf, probe_list):
    """chunk_time per probe: the stages' working times summed, median over the probe's forward and 4 either side.

    A pipeline holds about one forward in front of its slowest stage. A chunk's first to last layer ack (latency)
    counts that wait only when the slowest stage is past the first, and with near-equal stages which one is slowest
    changes from run to run. Summed stage work leaves the wait out, along with hand-offs between stages."""
    per_forward = _stage_times(spans, stages, cpf)
    if not per_forward:
        out.append("chunk_time: n/a (no per-stage layer acks, or a runner rank did not log its layers)")
        rec["chunk_time_curve_ms"] = [math.nan] * len(spans)
        for lbl, d in probe_list:
            rec["chunk_index"][lbl], rec["chunk_time_ms"][lbl] = d, math.nan
        return
    fwd = sorted(per_forward)

    def windowed(at):
        window = fwd[max(0, at - _TPUT_WINDOW // 2) : at + _TPUT_WINDOW // 2 + 1]
        stage = [_finite_median([per_forward[i][0][k] for i in window]) for k in range(len(stages))]
        total = _finite_median([sum(per_forward[i][0]) for i in window])
        queued = _finite_median([per_forward[i][1] for i in window])
        return total, stage, queued, _finite_median([per_forward[i][2] for i in window])

    smoothed = [windowed(at) for at in range(len(fwd))]  # chunk c is in forward c // cpf
    rec["chunk_time_curve_ms"] = [smoothed[c // cpf][0] for c in range(len(spans))]
    rec["latency_curve_ms"] = [smoothed[c // cpf][3] for c in range(len(spans))]
    measured = any(math.isfinite(sum(stage)) for stage, _queued, _latency in per_forward.values())
    if not measured:
        out.append(
            "chunk_time: n/a (every forward has a stage whose acks do not show when it started: acked as a batch or "
            "in one ack)"
        )
    else:
        out.append(
            "chunk_time = sum over stages of each stage's first to last layer ack, scaled to all its layers (queues "
            f"and hand-offs excluded); median of the probe's forward and {_TPUT_WINDOW // 2} either side; n/a where "
            "half of them have such a stage or do not split by stage"
        )
        out.append("  stages: per stage; queued: waits between stages (not before stage 0); latency: first to last ack")
        if cpf > 1:
            out.append("  packed: chunk_time and stages per chunk; queued and latency per forward")

    def ms(x, fmt=".0f"):
        return f"{x:{fmt}}" if math.isfinite(x) else "n/a"

    for lbl, d in probe_list:
        total, stage, queued, latency = smoothed[d // cpf]
        rec["chunk_index"][lbl] = d
        rec["chunk_time_ms"][lbl], rec["stage_ms"][lbl] = total, stage
        rec["queued_ms"][lbl], rec["latency_ms"][lbl] = queued, latency
        if measured:
            timed = [k for k in range(len(stage)) if math.isfinite(stage[k])]
            out.append(
                f"  chunk_time {lbl:>14} (chunk {d:>3}): {ms(total, '.3f'):>12} ms   stages "
                f"{' '.join(ms(x) for x in stage)} (slowest {max(timed, key=stage.__getitem__) if timed else '-'}), "
                f"queued {ms(queued)}, latency {ms(latency)}"
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranklogs", required=True, help="mpirun --output-filename root (runner + producer)")
    ap.add_argument("--real-chunks", type=int, default=0, help="chunks in the measured request (0 => all)")
    ap.add_argument("--chunk-size", type=int, default=0, help="tokens per chunk (0 => skip perf)")
    ap.add_argument("--pipeline-ranks", type=int, default=1, help="runner pipeline ranks (scaling ideal)")
    ap.add_argument(
        "--chunks-per-forward",
        type=int,
        default=1,
        help="chunks packed into one forward (PREFILL_FORWARD_TOKENS_BUDGET // chunk size); they retire together",
    )
    ap.add_argument(
        "--probe-chunks",
        default=None,
        help="comma-separated chunk indices to probe, shared across configs (default: derived from this request)",
    )
    ap.add_argument(
        "--summary-name", default=None, help="<model>_<config>; names the metrics sidecar the scaling job reads"
    )
    args = ap.parse_args()
    if not os.path.isdir(args.ranklogs):
        print(f"ranklogs dir {args.ranklogs} not found; nothing to summarize")
        return
    _pcc_matrix(args.ranklogs)
    probe_chunks = None
    if args.probe_chunks:
        probe_chunks = {int(tok) for tok in args.probe_chunks.split(",") if tok.strip()}
    spans = _spans_from_logs(args.ranklogs, args.real_chunks)
    if not spans:
        print("no producer CHUNK_RETIRE lines found (the producer's completion drain did not run)")
        return
    lines, rec = _perf_metrics(
        spans,
        args.chunk_size,
        args.pipeline_ranks,
        probe_chunks,
        args.chunks_per_forward,
        _setup_from_logs(args.ranklogs),
        _stage_ranges(args.ranklogs),
    )
    if lines:
        _publish(lines, args.summary_name)
    if rec and args.real_chunks > 0:
        rec["expected_chunks"] = args.real_chunks
    _publish_json(rec, args.summary_name)


if __name__ == "__main__":
    main()
