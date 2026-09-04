# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
import argparse
import os
import re
import sys

_KV = re.compile(r"slot\s+(\d+)\s+layer\s+(\d+)\s+KV PCC:\s+nope=([-\d.]+)\s+pe=([-\d.]+)")
_INDEX = re.compile(r"slot\s+(\d+)\s+layer\s+(\d+)\s+\(index rank\s+(\d+)\)\s+index PCC:\s+([-\d.]+)")
_CHUNK_START = re.compile(r"\[pp rank (\d+)\] CHUNK_START c=(\d+) compute_start=([\d.]+)")
_CHUNK_COMPUTE = re.compile(r"\[pp rank (\d+)\] CHUNK_COMPUTE c=(\d+) compute_ms=([\d.]+)")


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
    for line in _iter_lines(root):
        m = _KV.search(line)
        if m:
            layer, nope, pe = int(m.group(2)), float(m.group(3)), float(m.group(4))
            cell = layers.setdefault(layer, {})
            cell["nope"] = min(nope, cell.get("nope", nope))
            cell["pe"] = min(pe, cell.get("pe", pe))
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
    header = f"{'layer':>5}  {'kvpe.nope':>10}  {'kvpe.pe':>10}" + (f"  {'index':>10}" if have_index else "")
    print(header)
    worst = 1.0
    for layer in sorted(layers):
        cell = layers[layer]
        row = f"{layer:>5}  {cell.get('nope', float('nan')):>10.5f}  {cell.get('pe', float('nan')):>10.5f}"
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


def _select_measured(ranks, real_chunks):
    kept = {}
    cs_union = set()
    for rank, cells in ranks.items():
        cs = sorted(cells)
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
    return ct, ttft


def _publish(lines, name):
    title = f"disaggregated prefill perf -- {name or 'run'}"
    if name:
        home = os.environ.get("TT_METAL_HOME")
        if home and home not in sys.path:
            sys.path.insert(0, home)
        try:
            from models.demos.deepseek_v3_d_p.utils.prefill_summary_utils import emit_summary

            emit_summary("perf", name, title, lines)
            return
        except Exception as exc:
            print(f"perf summary not published ({exc})")
    print(title)
    print("\n".join(lines))


def _perf_metrics(kept, cs_sorted, disp, chunk_size, win_chunks):
    n = len(cs_sorted)
    if n == 0:
        return []
    max_seq = n * chunk_size
    ct, ttft = _cell_metrics(kept, disp)

    def idx(tok):
        return max(0, min(n - 1, tok // chunk_size))

    out = ["==================== perf metrics (measured request, warmup excluded) ===================="]
    out.append(f"max_seq={max_seq} tok ({n} chunks x {chunk_size}); offsets snapped to the containing chunk")
    out.append("chunk_time = first-rank start -> last-rank finish (cross-rank; assumes NTP-comparable clocks)")
    for lbl, tok in (
        ("5k@0", 0),
        ("5k@50k", 50000),
        ("5k@max_seq/2", max_seq // 2),
        ("5k@max_seq-5k", max_seq - chunk_size),
    ):
        d = idx(tok)
        val = f"{ct[d]:>12.3f}" if d in ct else f"{'-':>12}"
        out.append(f"  chunk_time {lbl:>14} (chunk {d:>3}): {val} ms")
    out.append("ttft = request start -> chunk finish")
    for lbl, tok in (("@50k", 50000), ("@max_seq/2", max_seq // 2), ("@max_seq", max_seq)):
        d = idx(tok)
        val = f"{ttft[d]:>12.3f}" if d in ttft else f"{'-':>12}"
        out.append(f"  ttft       {lbl:>14} (chunk {d:>3}): {val} s")

    rank0 = min(kept)
    inv = {i: c for c, i in disp.items()}
    out.append(f"throughput = rank{rank0} start->start rate over the {win_chunks} chunks ending at the offset")
    for lbl, tok in (("@50k", 50000), ("@max_seq/2", max_seq // 2), ("@max_seq", max_seq)):
        d = idx(tok)
        first = max(0, d - win_chunks + 1)
        ca, cb = inv.get(first), inv.get(d)
        usable = first < d and ca is not None and cb is not None and ca in kept[rank0] and cb in kept[rank0]
        sa = kept[rank0][ca][0] if usable else None
        sb = kept[rank0][cb][0] if usable else None
        span = f"chunks {first:>3}..{d:>3}"
        if sa is None or sb is None or sb <= sa:
            out.append(f"  throughput {lbl:>14} ({span}): {'-':>12}")
            continue
        dt = sb - sa
        tokens = (d - first) * chunk_size
        out.append(f"  throughput {lbl:>14} ({span}): {tokens / dt:>12,.1f} tok/s  ({dt:.3f} s)")
    return out


def _timing_matrix(root, real_chunks, timing_dir=None, chunk_size=0, win_chunks=4):
    ranks = _timing_from_csvs(timing_dir)
    if ranks is None:
        ranks = _timing_from_ranklogs(root)

    print("==================== per-rank x per-chunk timing (measured request) ====================")
    if not ranks:
        print("no timing rows found (set PREFILL_SYNC_PER_CHUNK=1 on the runner; timing CSVs / CHUNK_* logs absent)")
        return []
    kept, cs_sorted, disp = _select_measured(ranks, real_chunks)
    all_starts = [c[0] for r in kept.values() for c in r.values() if c[0] is not None]
    if not all_starts:
        print("timing rows present but no compute_start timestamps parsed")
        return []
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
        return _perf_metrics(kept, cs_sorted, disp, chunk_size, win_chunks)
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranklogs", required=True, help="mpirun --output-filename root (runner + producer)")
    ap.add_argument("--timing-dir", default=None, help="per-rank timing CSV dir (preferred timing source)")
    ap.add_argument("--real-chunks", type=int, default=0, help="chunks in the measured request (0 => all)")
    ap.add_argument("--chunk-size", type=int, default=0, help="tokens per chunk (0 => skip throughput)")
    ap.add_argument("--perf-window-chunks", type=int, default=4, help="chunks per throughput window at each offset")
    ap.add_argument(
        "--summary-name", default=None, help="publish the perf block under PREFILL_SUMMARIES/perf/<name>.md"
    )
    args = ap.parse_args()
    if not os.path.isdir(args.ranklogs):
        print(f"ranklogs dir {args.ranklogs} not found; nothing to summarize")
        return
    _pcc_matrix(args.ranklogs)
    lines = _timing_matrix(args.ranklogs, args.real_chunks, args.timing_dir, args.chunk_size, args.perf_window_chunks)
    if lines:
        _publish(lines, args.summary_name)


if __name__ == "__main__":
    main()
