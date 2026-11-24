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
    r"(?: acks=(\d+))?"
)

# The runner's PREFILL_D2D_TIMING line (prefill_runner._D2DStageClock): rank-clock seconds of three device points per
# chunk. slot=- marks the shutdown sentinel, whose post ends the last chunk.
_D2D_TIMING = re.compile(
    r"\[pp rank (\d+)\] D2D_TIMING c=(\d+) slot=(-|\d+) start=(-|\d+) post=([\d.]+) recv=([\d.]+) free=([\d.]+)"
)
# The rank's stage, as the runner prints it at startup (a looped model's pass positions first).
_RANK_STAGE = re.compile(
    r"\[pp rank (\d+)/\d+\] mesh=.*? ((?:passes=\[\d+, \d+\) loop_passes=\([\d, ]*\) )?layers=\[\d+, \d+\))"
)

# The runner's line when a rank has built its stage and entered its request loop.
_RANK_READY = re.compile(r"\[pp rank (\d+)\] setup complete, entering request loop")

# Chunks spanned by a throughput probe. A single chunk can run half again as long as its neighbours,
# so one interval is too noisy to be a rate; eight is still under 5% of the shortest measured request.
_TPUT_WINDOW = 8


def _log_files(root):
    # Every file under the root, whatever its name: OpenMPI's --output-filename writes rank.N/{stdout,stderr} on some
    # builds and flat <prefix>.<job>@1.N.{out,err} files on others (the SC4 runners).
    for dirpath, _dirs, files in os.walk(root):
        for name in sorted(files):
            yield os.path.join(dirpath, name)


def _iter_lines(root):
    for path in _log_files(root):
        try:
            with open(path, errors="replace") as fh:
                for line in fh:
                    yield line
        except OSError:
            continue


def ready_ranks(root):
    """(rank-log files under root, distinct runner ranks that logged entering their request loop)."""
    files = sum(1 for _ in _log_files(root))
    ranks = {m.group(1) for line in _iter_lines(root) if (m := _RANK_READY.search(line))}
    return files, len(ranks)


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
    spans = []
    for line in _iter_lines(root):
        m = _CHUNK_RETIRE.search(line)
        if m and m.group(6) == "0":
            spans.append((float(m.group(4)), float(m.group(5)), int(m.group(7) or 0)))
    spans.sort()
    return spans[-real_chunks:] if real_chunks > 0 else spans


def _probes(n, chunk_size, probe_chunks, offsets, *, skip_first_chunk=False):
    """[(label, chunk)]: the launcher's probe chunks within the n measured ones, else the chunks holding `offsets`."""
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


def _stage_records(root, real_chunks, chunk_size):
    """{rank: [(post, recv, free)]} for the measured request's chunks plus the shutdown record that ends the last,
    and {rank: its stage text}. A rank's records are its last real_chunks chunks, which must be the request's chunks
    in order (start = index x chunk_size); a rank that does not show that is left out, with a note."""
    raw, stages, notes = {}, {}, []
    for line in _iter_lines(root):
        m = _D2D_TIMING.search(line)
        if m:
            rank, c = int(m.group(1)), int(m.group(2))
            start = None if m.group(4) == "-" else int(m.group(4))
            raw.setdefault(rank, {})[c] = (start, float(m.group(5)), float(m.group(6)), float(m.group(7)))
            continue
        m = _RANK_STAGE.search(line)
        if m:
            stages[int(m.group(1))] = m.group(2)
    records = {}
    for rank, by_c in sorted(raw.items()):
        rows = [by_c[c] for c in sorted(by_c)]
        if not rows or rows[-1][0] is not None:
            notes.append(f"rank {rank}: no shutdown record, so its last chunk has no end; left out")
            continue
        measured = rows[-1 - real_chunks : -1] if real_chunks > 0 else rows[:-1]
        if [r[0] for r in measured] != [i * chunk_size for i in range(len(measured))] or (
            real_chunks > 0 and len(measured) != real_chunks
        ):
            notes.append(f"rank {rank}: its last {real_chunks} chunks are not the request's in order; left out")
            continue
        records[rank] = [r[1:] for r in measured] + [rows[-1][1:]]
    return records, stages, notes


def _median(values):
    s = sorted(v for v in values if v is not None)
    if not s:
        return None
    return s[len(s) // 2] if len(s) % 2 else 0.5 * (s[len(s) // 2 - 1] + s[len(s) // 2])


def _stage_metrics(records, stages, notes, chunk_size, probe_chunks=None):
    """Per-stage chunk times and per-boundary D2D transfer times from the runners' D2D_TIMING lines.

    Every quantity is a difference of one rank's own stamps, so no cross-host clock enters. For chunk i on rank r
    (post/recv/free as prefill_runner._D2DStageClock defines them; i+1 is the next chunk, or the shutdown record):
      fwd   = post(i+1) - free(i)    the stage's forward and send hand-off: its compute per chunk
      cycle = post(i+1) - post(i)    the rank's period: fwd plus the receive and the lease fence
      recv  = recv(i) - post(i)      receive side: waiting for the upstream send, then the transfer (rank > 0)
      send  = free(i+1) - post(i+1)  send side: waiting for the downstream grant, then the transfer, then the fence;
                                     it also holds the next chunk's receive (rank < last)
    A boundary's transfer for chunk i is min(sender's send, receiver's recv): each side is the transfer plus a wait
    for the other side, and at most one side waits, so the smaller is the transfer (an upper bound if both waited).
    Each probe is the median over the chunks d-4..d+4."""
    if not records:
        return [], {}
    n = min(len(rows) - 1 for rows in records.values())
    ranks = sorted(records)
    last = max(ranks)

    def per_chunk(rank):
        rows = records[rank]
        out = []
        for i in range(n):
            (post, recv, free), (post_next, _, free_next) = rows[i], rows[i + 1]
            out.append(
                {
                    "fwd": (post_next - free) * 1000.0,
                    "cycle": (post_next - post) * 1000.0,
                    "recv": (recv - post) * 1000.0 if rank > 0 else None,
                    "send": (free_next - post_next) * 1000.0 if rank < last else None,
                }
            )
        return out

    chunks = {rank: per_chunk(rank) for rank in ranks}
    window = _TPUT_WINDOW // 2
    probes = _probes(n, chunk_size, probe_chunks, (0, 50000, n * chunk_size // 2, (n - 1) * chunk_size))
    rec = {"stage": {}, "stage_ms": {}, "boundary_ms": {}}
    out = [
        "==================== per-stage chunk time and D2D boundary transfer (runner D2D_TIMING) ===================="
    ]
    out.append(
        "each value is one rank's own clock (no cross-host clock); median over chunks d-4..d+4 of the measured request"
    )
    out.append("fwd = stage compute per chunk (lease fence passed -> its forward and send hand-off done)")
    out.append("cycle = the rank's period; recv / send = receive / send side of its inbound / outbound boundary")
    for note in notes:
        out.append(f"note: {note}")
    for rank in ranks:
        rec["stage"][str(rank)] = stages.get(rank, "")
        out.append(f"rank {rank}: {stages.get(rank, '(stage line not found)')}")
    out.append(
        f"{'rank':>4}  {'probe':>8}  {'chunk':>5}  {'fwd ms':>9}  {'cycle ms':>9}  {'recv ms':>9}  {'send ms':>9}"
    )
    fmt = lambda v: f"{v:>9.3f}" if v is not None else f"{'-':>9}"  # noqa: E731
    for rank in ranks:
        cells = rec["stage_ms"].setdefault(str(rank), {})
        for label, d in probes:
            span = chunks[rank][max(0, d - window) : d + window + 1]
            cell = {key: _median(c[key] for c in span) for key in ("fwd", "cycle", "recv", "send")}
            cells[label] = {k: v for k, v in cell.items() if v is not None}
            out.append(
                f"{rank:>4}  {label:>8}  {d:>5}  {fmt(cell['fwd'])}  {fmt(cell['cycle'])}  {fmt(cell['recv'])}  "
                f"{fmt(cell['send'])}"
            )
    out.append("boundary transfer per chunk = min(sender's send, receiver's recv); median over chunks d-4..d+4")
    out.append(f"{'boundary':>8}  {'probe':>8}  {'chunk':>5}  {'send ms':>9}  {'recv ms':>9}  {'transfer':>9}")
    for src in ranks:
        dst = src + 1
        if dst not in chunks:
            continue
        cells = rec["boundary_ms"].setdefault(f"{src}->{dst}", {})
        for label, d in probes:
            lo, hi = max(0, d - window), d + window + 1
            send = [c["send"] for c in chunks[src][lo:hi]]
            recv = [c["recv"] for c in chunks[dst][lo:hi]]
            cell = {
                "send": _median(send),
                "recv": _median(recv),
                "transfer": _median(min(s, r) for s, r in zip(send, recv)),
            }
            cells[label] = cell
            out.append(
                f"{src:>3}->{dst:<3}  {label:>8}  {d:>5}  {fmt(cell['send'])}  {fmt(cell['recv'])}  "
                f"{fmt(cell['transfer'])}"
            )
    return out, rec


def _perf_metrics(spans, chunk_size, pipeline_ranks, probe_chunks=None):
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
        "ttft_s": {},
        "throughput_tok_s": {},
    }

    def probes(offsets, *, skip_first_chunk=False):
        return _probes(n, chunk_size, probe_chunks, offsets, skip_first_chunk=skip_first_chunk)

    out = ["==================== perf metrics (measured request, warmup excluded) ===================="]
    out.append(f"max_seq={max_seq} tok ({n} chunks x {chunk_size})")
    out.append("labels are tokens prefilled through that chunk, in units of 1024 (M past 1Mi); probes snap to chunks")
    out.append("all times are the producer's clock: first layer ack -> last layer ack of a chunk")
    if whole_chunk_acks:
        out.append("chunk_time, ttft: n/a (the runner acks each chunk in one message)")
    else:
        out.append("chunk_time = the chunk's first layer ack -> its last layer ack")
        rec["chunk_time_curve_ms"] = [(span[1] - span[0]) * 1000.0 for span in spans]
        for lbl, d in probes((0, 50000, max_seq // 2, max_seq - chunk_size)):
            rec["chunk_index"][lbl] = d
            rec["chunk_time_ms"][lbl] = (spans[d][1] - spans[d][0]) * 1000.0
            out.append(f"  chunk_time {lbl:>14} (chunk {d:>3}): {rec['chunk_time_ms'][lbl]:>12.3f} ms")
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
    ap.add_argument(
        "--ready-ranks",
        action="store_true",
        help="print '<rank-log files> <ranks in their request loop>' and exit (the launcher's readiness check)",
    )
    args = ap.parse_args()
    if args.ready_ranks:
        print(*ready_ranks(args.ranklogs))
        return
    if not os.path.isdir(args.ranklogs):
        print(f"ranklogs dir {args.ranklogs} not found; nothing to summarize")
        return
    _pcc_matrix(args.ranklogs)
    probe_chunks = None
    if args.probe_chunks:
        probe_chunks = {int(tok) for tok in args.probe_chunks.split(",") if tok.strip()}
    spans = _spans_from_logs(args.ranklogs, args.real_chunks)
    lines, rec = [], {}
    if spans:
        lines, rec = _perf_metrics(spans, args.chunk_size, args.pipeline_ranks, probe_chunks)
    else:
        print("no producer CHUNK_RETIRE lines found (the producer's completion drain did not run)")
    if args.chunk_size > 0:
        stage_lines, stage_rec = _stage_metrics(
            *_stage_records(args.ranklogs, args.real_chunks, args.chunk_size), args.chunk_size, probe_chunks
        )
        lines += stage_lines
        rec.update(stage_rec)
    if lines:
        _publish(lines, args.summary_name)
    _publish_json(rec, args.summary_name)


if __name__ == "__main__":
    main()
