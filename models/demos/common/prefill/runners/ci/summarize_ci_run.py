# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
import argparse
import json
import os
import re

_KV = re.compile(r"slot\s+(\d+)\s+layer\s+(\d+)\s+KV PCC:\s+nope=([-\d.]+)\s+pe=([-\d.]+)")
_GQA = re.compile(r"layer\s+(\d+):\s+K=([-\d.]+)\s+V=([-\d.]+)")
_INDEX = re.compile(r"slot\s+(\d+)\s+layer\s+(\d+)\s+\(index rank\s+(\d+)\)\s+index PCC:\s+([-\d.]+)")
_CHUNK_RETIRE = re.compile(
    r"\[producer\] CHUNK_RETIRE slot=(\d+) start=(\d+) end=(\d+) first=([\d.]+) retire=([\d.]+) warmup=([01])"
    r"(?: acks=(\d+))?(?: fwd=(\d+))?"
)
# A chunk's layer acks, "<layer_start>-<layer_end>@<ms after its first ack>", and each runner rank's layer range.
_LAYER_ACK = re.compile(r"(\d+)-(\d+)@([\d.]+)")
_RANK_LAYERS = re.compile(r"\[pp rank (\d+)/(\d+)\].*? layers=\[(\d+), (\d+)\)")

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
            json.dump(rec, fh, indent=2, sort_keys=True)
            fh.write("\n")
        print(f"perf metrics -> {path}")
    except OSError as exc:
        print(f"perf metrics not persisted ({exc})")


def _spans_from_logs(root, real_chunks):
    """(first ack, retire, acks, forward, {layer_end: ack time}) per measured chunk, in first-ack order. A log
    without fwd= (an older producer) makes every chunk its own forward."""
    spans = []
    for line in _iter_lines(root):
        m = _CHUNK_RETIRE.search(line)
        if m and m.group(6) == "0":
            first = float(m.group(4))
            fwd = int(m.group(8)) if m.group(8) is not None else -1 - len(spans)
            acked = {}
            for _a, b, ms in _LAYER_ACK.findall(line[m.end() :]):
                acked[int(b)] = max(acked.get(int(b), 0.0), first + float(ms) / 1000.0)
            spans.append((first, float(m.group(5)), int(m.group(7) or 0), fwd, acked))
    spans.sort(key=lambda span: span[:2])
    return spans[-real_chunks:] if real_chunks > 0 else spans


def _stage_ends_from_logs(root):
    """Each pipeline stage's last layer + 1, in order, from the runner ranks' "[pp rank r/N] ... layers=[a, b)"
    banner; [] when there is no banner or only one stage."""
    ranges = {}
    for line in _iter_lines(root):
        m = _RANK_LAYERS.search(line)
        if m:
            ranges[int(m.group(1))] = (int(m.group(3)), int(m.group(4)))
    return [hi for _lo, hi in sorted(ranges.values())] if len(ranges) > 1 else []


def _queue_waits(spans, stage_ends):
    """Per chunk, the time it sat finished on one stage while the next stage was still busy with an earlier
    forward: at each boundary, max(0, next stage's previous completion - this chunk's completion here).

    A chunk's first-ack -> last-ack span counts that wait, so it measures where the pipeline's queue sits (it
    builds in front of the slowest stage, and the slowest stage moves with depth) as much as what the chunk
    costs. Subtracting it leaves the chunk's own latency through every stage. Needs the per-layer acks and the
    stage layout; a chunk missing either gets no correction."""
    waits = [0.0] * len(spans)
    for prev_end, end in zip(stage_ends, stage_ends[1:]):
        done = sorted((span[4][end], span[3], i) for i, span in enumerate(spans) if end in span[4])
        busy_until, last_fwd, fwd_done = None, None, None
        for t, fwd, i in done:
            if fwd != last_fwd:
                busy_until, last_fwd = fwd_done, fwd
            fwd_done = t
            ready = spans[i][4].get(prev_end)
            if busy_until is not None and ready is not None:
                waits[i] += max(0.0, busy_until - ready)
    return waits


def _perf_metrics(spans, chunk_size, pipeline_ranks, probe_chunks=None, stage_ends=()):
    n = len(spans)
    if n == 0 or chunk_size <= 0:
        return [], {}
    max_seq = n * chunk_size
    t0 = spans[0][0]
    end = [span[1] for span in spans]
    whole_chunk_acks = all(span[2] == 1 for span in spans)
    rec = {
        "chunk_size": chunk_size,
        "num_chunks": n,
        "max_seq": max_seq,
        "pipeline_ranks": pipeline_ranks,
        "chunk_index": {},
        "chunk_time_ms": {},
        "chunk_span_ms": {},
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
    out.append("all times are the producer's clock: first layer ack -> last layer ack of a chunk")
    if whole_chunk_acks:
        out.append("chunk_time, ttft: n/a (the runner acks each chunk in one message)")
    else:
        waits = _queue_waits(spans, list(stage_ends)) if len(stage_ends) > 1 else [0.0] * n
        span_ms = [(span[1] - span[0]) * 1000.0 for span in spans]
        rec["chunk_span_curve_ms"] = span_ms
        rec["queue_wait_curve_ms"] = [w * 1000.0 for w in waits]
        rec["chunk_time_curve_ms"] = [s - w * 1000.0 for s, w in zip(span_ms, waits)]
        out.append(
            "chunk_time = the chunk's first layer ack -> its last layer ack, less the time it queued between "
            "stages (finished on one, the next still busy with an earlier forward)"
        )
        if len(stage_ends) < 2:
            out.append("  (one stage, or no '[pp rank]' layer banner: no queue to subtract)")
        for lbl, d in probes((0, 50000, max_seq // 2, max_seq - chunk_size)):
            rec["chunk_index"][lbl] = d
            rec["chunk_time_ms"][lbl] = rec["chunk_time_curve_ms"][d]
            rec["chunk_span_ms"][lbl] = span_ms[d]
            out.append(
                f"  chunk_time {lbl:>14} (chunk {d:>3}): {rec['chunk_time_ms'][lbl]:>12.3f} ms"
                f"   (span {span_ms[d]:,.1f} ms, queued {waits[d] * 1000.0:,.1f} ms)"
            )
        out.append("ttft = the request's first layer ack -> the chunk's last layer ack")
        for lbl, d in probes((50000, max_seq // 2, max_seq), skip_first_chunk=True):
            rec["chunk_index"][lbl] = d
            rec["ttft_s"][lbl] = end[d] - t0
            out.append(f"  ttft       {lbl:>14} (chunk {d:>3}): {rec['ttft_s'][lbl]:>12.3f} s")

    out.append(f"throughput = tokens / wall time across up to {_TPUT_WINDOW} chunk retires centred on the probe")
    for lbl, d in probes((0, 50000, max_seq // 2, max_seq)):
        # The rate is the retire cadence. A chunk's own latency covers every stage at once and so
        # carries no information about how many chunks are in flight; only the interval between
        # retires does. The window absorbs one slow chunk while staying short enough that the
        # attention cost, which grows with depth, barely drifts across it.
        lo, hi = max(0, d - _TPUT_WINDOW // 2), min(n - 1, d + _TPUT_WINDOW // 2)
        if hi - lo < 1 or end[hi] <= end[lo]:
            out.append(f"  throughput {lbl:>14} (chunk {d:>3}): {'-':>12}")
            continue
        tput = (hi - lo) * chunk_size / (end[hi] - end[lo])
        out.append(f"  throughput {lbl:>14} (chunk {d:>3}): {tput:>12,.1f} tok/s")
        rec["throughput_tok_s"][lbl] = tput
    return out, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranklogs", required=True, help="mpirun --output-filename root (runner + producer)")
    ap.add_argument("--real-chunks", type=int, default=0, help="chunks in the measured request (0 => all)")
    ap.add_argument("--chunk-size", type=int, default=0, help="tokens per chunk (0 => skip perf)")
    ap.add_argument("--pipeline-ranks", type=int, default=1, help="runner pipeline ranks (scaling ideal)")
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
        spans, args.chunk_size, args.pipeline_ranks, probe_chunks, _stage_ends_from_logs(args.ranklogs)
    )
    if lines:
        _publish(lines, args.summary_name)
    _publish_json(rec, args.summary_name)


if __name__ == "__main__":
    main()
