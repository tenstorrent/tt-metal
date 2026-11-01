# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Accuracy of a Kimi-K3 ``layers.0`` KDA run against the FP64 long-context golden, from compact error sums.

A run directory holds per-token output sums and per-head recurrent-state sums against the golden (written by
``test_layer0_dev_run.py`` while the device runs, or by ``from-trace`` here for a vLLM-format trace such as
FlashKDA's or Triton's). Every metric below is derived from those sums, so analysis never reloads the golden and
every implementation is scored with the same definitions:

* NRMSE ``‖actual − golden‖ / ‖golden‖`` (FP64) over the whole run, the first call, every 1,024-token window;
* per-token NRMSE with the golden-wide floor (1% of the median golden token norm, as ``golden_floor``);
* gain ``⟨actual, golden⟩ / ⟨golden, golden⟩``: the least-squares scale of the golden in the actual output,
  below 1 when the output is too small (PCC is blind to it);
* per head and snapshot: recurrent-state NRMSE and gain; the pooled state over heads, with and without the
  slowest-decaying head 6; per-head means over snapshots;
* distributions: token-error percentiles (p50/p90/p95/p99, worst) per call and per 64k window and over the run,
  relative to each token's own norm and to the span's RMS norm; per-head state percentiles over heads per snapshot,
  at the final snapshot and over each head's worst snapshot; drift per 100k tokens; tail spread p99 / median;
* the reported metric set in zoom-in tiers (``tiers``): tier 0 output NRMSE and final pooled state NRMSE; tier 1
  p50/p95/p99/worst of output per token and per 1,024-token window, final state per head, pooled state per call;
  tier 2 output gain, final state gain per head, p50/p95/p99/worst of state per head per call.

Run directory layout::

    run.json             provenance (commit, config, mesh, golden identity) and, when measured, perf
    per_call.jsonl       one line per call, appended as the run progresses
    token_sums.npy       float64 [tokens, 4]: Σ(a−g)², Σg², Σa·g, Σa² over hidden, per token
    state_sums.npy       float64 [snapshots, heads, 4]: the same sums per head of the recurrent state
    state_tokens.json    token position of every state snapshot
    state.safetensors    float32 [snapshots, heads, k, v] recurrent state (golden orientation), optional
    DONE                 written last

Usage::

    python layer0_run_analysis.py from-trace --trace <vLLM trace dir> --golden <golden> --out <run dir>
    python layer0_run_analysis.py report <run dir> [--baseline name=<run dir> ...] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

HEADS = 96
HEAD_DIM = 128
WINDOW = 1024
LONG_MEMORY_HEADS = (6, 62, 68, 36, 75, 17)
SUM_FIELDS = ("error_sq", "golden_sq", "cross", "actual_sq")


def log(message: str) -> None:
    print(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {message}", flush=True)


def error_sums(golden: torch.Tensor, actual: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """``[..., 4]`` FP64 sums over ``dim``: squared error, squared golden, cross term, squared actual."""
    golden = golden.double()
    actual = actual.double()
    return torch.stack(
        [
            (actual - golden).square().sum(dim),
            golden.square().sum(dim),
            (actual * golden).sum(dim),
            actual.square().sum(dim),
        ],
        dim=-1,
    )


class RunWriter:
    """Preallocated run files filled call by call, readable at any point of an interrupted run."""

    def __init__(self, run_dir: Path, tokens: int, snapshots: int, save_state: bool) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.token_sums = np.lib.format.open_memmap(
            self.run_dir / "token_sums.npy", mode="w+", dtype=np.float64, shape=(tokens, 4)
        )
        self.token_sums[:] = np.nan
        self.state_sums = np.lib.format.open_memmap(
            self.run_dir / "state_sums.npy", mode="w+", dtype=np.float64, shape=(snapshots, HEADS, 4)
        )
        self.state_sums[:] = np.nan
        self.state_tokens: list[int] = []
        self.state = (
            _preallocated_safetensors(
                self.run_dir / "state.safetensors", "recurrent_state", "F32", (snapshots, HEADS, HEAD_DIM, HEAD_DIM)
            )
            if save_state
            else None
        )

    def write_call(
        self, start: int, end: int, snapshot: int, token_sums: torch.Tensor, state_sums: torch.Tensor, state
    ) -> None:
        self.token_sums[start:end] = token_sums.numpy()
        self.state_sums[snapshot] = state_sums.numpy()
        if self.state is not None:
            self.state[snapshot] = state.float().numpy()
        self.state_tokens.append(end)
        (self.run_dir / "state_tokens.json").write_text(json.dumps(self.state_tokens))

    def flush(self) -> None:
        self.token_sums.flush()
        self.state_sums.flush()
        if self.state is not None:
            self.state.flush()


def _preallocated_safetensors(path: Path, name: str, dtype: str, shape: tuple[int, ...]) -> np.memmap:
    np_dtype = {"F32": np.float32, "BF16": np.uint16}[dtype]
    nbytes = int(np.prod(shape)) * np.dtype(np_dtype).itemsize
    header = json.dumps({name: {"dtype": dtype, "shape": list(shape), "data_offsets": [0, nbytes]}}).encode()
    header += b" " * (-len(header) % 8)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(header)))
        handle.write(header)
        handle.truncate(8 + len(header) + nbytes)
    return np.memmap(path, dtype=np_dtype, mode="r+", offset=8 + len(header), shape=shape)


def call_summary(token_sums: torch.Tensor, state_sums: torch.Tensor) -> dict:
    """Headline numbers of one call, for the progress line and ``per_call.jsonl``."""
    out = token_sums.sum(0)
    head_nrmse = (state_sums[:, 0] / state_sums[:, 1]).sqrt()
    head_gain = state_sums[:, 2] / state_sums[:, 1]
    return {
        "output_nrmse": float((out[0] / out[1]).sqrt()),
        "output_gain": float(out[2] / out[1]),
        "state_nrmse": float((state_sums[:, 0].sum() / state_sums[:, 1].sum()).sqrt()),
        "h6_nrmse": float(head_nrmse[6]),
        "h6_gain": float(head_gain[6]),
        "median_head_nrmse": float(head_nrmse.median()),
        "worst_head": int(head_nrmse.argmax()),
        "worst_head_nrmse": float(head_nrmse.max()),
    }


# --------------------------------------------------------------------------------------------------- analysis


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(num / den)


def load_run(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    token_sums = np.load(run_dir / "token_sums.npy", mmap_mode="r")
    state_sums = np.load(run_dir / "state_sums.npy", mmap_mode="r")
    done_tokens = int(np.isfinite(token_sums[:, 0]).sum())
    snapshots = json.loads((run_dir / "state_tokens.json").read_text())
    meta = json.loads((run_dir / "run.json").read_text()) if (run_dir / "run.json").exists() else {}
    return {
        "dir": run_dir,
        "meta": meta,
        "token_sums": np.asarray(token_sums[:done_tokens]),
        "state_sums": np.asarray(state_sums[: len(snapshots)]),
        "state_tokens": snapshots,
        "complete": (run_dir / "DONE").exists(),
    }


def analyze(run: dict, first_call: int = 5120, windows: tuple[int, ...] = (5120, 65536)) -> dict:
    """Every metric of one run; JSON-serializable."""
    t = run["token_sums"]
    s = run["state_sums"]
    tokens = len(t)
    whole = t.sum(0)
    window_sums = np.add.reduceat(t, np.arange(0, tokens, WINDOW), axis=0)
    window_nrmse = _ratio(window_sums[:, 0], window_sums[:, 1])
    window_gain = window_sums[:, 2] / window_sums[:, 1]
    golden_norm = np.sqrt(t[:, 1])
    floor = 0.01 * float(np.median(golden_norm))
    token_nrmse = np.sqrt(t[:, 0]) / np.maximum(golden_norm, floor)
    head = _ratio(s[..., 0], s[..., 1])  # [snapshots, heads]
    head_gain = s[..., 2] / s[..., 1]
    pooled = _ratio(s[..., 0].sum(-1), s[..., 1].sum(-1))
    others = np.delete(s, 6, axis=1)
    pooled_without_h6 = _ratio(others[..., 0].sum(-1), others[..., 1].sum(-1))
    head_mean = head.mean(0)
    order = np.argsort(-token_nrmse)[:20]
    result = {
        "tokens": tokens,
        "complete": run["complete"],
        "output": {
            "nrmse": float(_ratio(whole[0], whole[1])),
            "gain": float(whole[2] / whole[1]),
            "first_call_nrmse": float(_ratio(t[:first_call, 0].sum(), t[:first_call, 1].sum())),
            "first_call_gain": float(t[:first_call, 2].sum() / t[:first_call, 1].sum()),
            "window_nrmse_max": float(window_nrmse.max()),
            "window_nrmse_last": float(window_nrmse[-1]),
            "token_floor": floor,
            "token_nrmse_p50": float(np.median(token_nrmse)),
            "token_nrmse_p99": float(np.quantile(token_nrmse, 0.99)),
            "token_nrmse_max": float(token_nrmse.max()),
            "worst_tokens": [[int(i), float(token_nrmse[i])] for i in order],
        },
        "output_windows": {
            "window": WINDOW,
            "nrmse": window_nrmse.tolist(),
            "gain": window_gain.tolist(),
        },
        "state": {
            "snapshots": len(s),
            "pooled_mean": float(pooled.mean()),
            "pooled_final": float(pooled[-1]),
            "pooled_max": float(pooled.max()),
            "without_h6_mean": float(pooled_without_h6.mean()),
            "median_head_mean": float(np.median(head_mean)),
            "worst_head": int(head_mean.argmax()),
            "heads": {
                str(h): {
                    "nrmse_mean": float(head_mean[h]),
                    "nrmse_final": float(head[-1, h]),
                    "gain_final": float(head_gain[-1, h]),
                }
                for h in LONG_MEMORY_HEADS
            },
        },
        "state_snapshots": {
            "tokens": run["state_tokens"],
            "pooled": pooled.tolist(),
            "pooled_gain": (s[..., 2].sum(-1) / s[..., 1].sum(-1)).tolist(),
            "without_h6": pooled_without_h6.tolist(),
            "median_head": np.median(head, axis=1).tolist(),
            "head_nrmse": {str(h): head[:, h].tolist() for h in LONG_MEMORY_HEADS},
            "head_gain": {str(h): head_gain[:, h].tolist() for h in LONG_MEMORY_HEADS},
            "head_nrmse_mean_all": head_mean.tolist(),
        },
        "distributions": distributions(t, s, run["state_tokens"], floor, windows),
        "tiers": tiers(t, s, run["state_tokens"], floor),
    }
    return result


PERCENTILES = (50, 90, 95, 99)
DRIFT_TOKENS = 100_000


def _spread(values: np.ndarray, with_max: bool) -> dict:
    """Percentiles of ``values`` along the last axis (and the maximum with its index), as lists or floats."""
    out = {f"p{p}": np.percentile(values, p, axis=-1) for p in PERCENTILES}
    if with_max:
        out["max"] = values.max(-1)
        out["argmax"] = values.argmax(-1)
    return {k: v.tolist() for k, v in out.items()}


def _token_level(t: np.ndarray, floor: float, start: int = 0) -> dict:
    """Token-error distribution of one span: per-token error relative to the token's own golden norm (floored)
    and to the span's RMS golden norm, with the span's pooled NRMSE and gain."""
    error = np.sqrt(t[:, 0])
    own = _spread(error / np.maximum(np.sqrt(t[:, 1]), floor), with_max=True)
    span_rms = _spread(error / np.sqrt(t[:, 1].mean()), with_max=True)
    for level in (own, span_rms):
        level["argmax"] += start
    whole = t.sum(0)
    return {
        "own_norm": own,
        "span_rms": span_rms,
        "pooled": float(_ratio(whole[0], whole[1])),
        "gain": float(whole[2] / whole[1]),
    }


def _drift(position: np.ndarray, value: np.ndarray) -> dict:
    """Growth factor per ``DRIFT_TOKENS`` of a positive series: exp of the least-squares slope of log(value)
    over position, fitted over the whole run and over its second half (a plateau shows as a factor near 1);
    None where a span has fewer than two points."""
    half = len(position) // 2
    fit = lambda x, y: float(np.exp(np.polyfit(x, np.log(y), 1)[0] * DRIFT_TOKENS)) if len(x) > 1 else None
    return {"whole": fit(position, value), "second_half": fit(position[half:], value[half:])}


def distributions(
    t: np.ndarray, s: np.ndarray, state_tokens: list[int], floor: float, windows: tuple[int, ...]
) -> dict:
    """Percentile view of token and per-head state error, per window over position and over the whole run.

    Token error per window: percentiles and the worst token, relative to each token's own golden norm and to the
    window's RMS golden norm. State per snapshot: percentiles over heads and the three worst heads; overall both
    at the final snapshot and for each head's worst snapshot. Drift and tail spread (p99 / median) are taken from
    the first window size, gain from the pooled sums.
    """
    tokens = len(t)
    per_window = {}
    for size in windows:
        spans = [_token_level(t[a : a + size], floor, a) for a in range(0, tokens, size)]
        merged = {
            "end": [min(a + size, tokens) for a in range(0, tokens, size)],
            "pooled": [w["pooled"] for w in spans],
            "gain": [w["gain"] for w in spans],
        }
        for norm in ("own_norm", "span_rms"):
            merged[norm] = {k: [w[norm][k] for w in spans] for k in spans[0][norm]}
        per_window[str(size)] = merged

    head = _ratio(s[..., 0], s[..., 1])  # [snapshots, heads]
    worst3 = np.argsort(-head, axis=1)[:, :3]
    state_over_position = _spread(head, with_max=False)
    state_over_position["worst3"] = [[[int(h), float(head[i, h])] for h in row] for i, row in enumerate(worst3)]

    def heads_level(values: np.ndarray) -> dict:
        level = {k: float(v) for k, v in _spread(values, with_max=False).items()}
        level["worst3"] = [[int(h), float(values[h])] for h in np.argsort(-values)[:3]]
        return level

    calls = per_window[str(windows[0])]
    end = np.asarray(calls["end"], dtype=np.float64)
    own = calls["own_norm"]
    drift = {f"token_{k}": _drift(end, np.asarray(own[k])) for k in (*(f"p{p}" for p in PERCENTILES), "max")}
    drift["token_pooled"] = _drift(end, np.asarray(calls["pooled"]))
    state_end = np.asarray(state_tokens, dtype=np.float64)
    drift.update(
        {f"state_{k}": _drift(state_end, np.asarray(v)) for k, v in state_over_position.items() if k[0] == "p"}
    )

    overall = _token_level(t, floor)
    for norm in ("own_norm", "span_rms"):
        overall[norm] = {k: v if isinstance(v, int) else float(v) for k, v in overall[norm].items()}
    return {
        "percentiles": list(PERCENTILES),
        "token_floor": floor,
        "token_windows": per_window,
        "state_over_position": {"tokens": state_tokens, **state_over_position},
        "overall": {
            "token": overall,
            "token_tail_spread": overall["own_norm"]["p99"] / overall["own_norm"]["p50"],
            "state_final": heads_level(head[-1]),
            "state_worst_snapshot": heads_level(head.max(0)),
        },
        "tail_spread_over_position": (np.asarray(own["p99"]) / np.asarray(own["p50"])).tolist(),
        "drift_per_100k_tokens": drift,
    }


TIER_PERCENTILES = (50, 95, 99)


def _tail(values: np.ndarray, positions: list) -> dict:
    """p50/p95/p99 (linear interpolation) and the worst value with its position."""
    worst = int(values.argmax())
    out = {f"p{p}": float(np.percentile(values, p)) for p in TIER_PERCENTILES}
    out.update(worst=float(values[worst]), worst_at=positions[worst])
    return out


def tiers(t: np.ndarray, s: np.ndarray, state_tokens: list[int], floor: float) -> dict:
    """The decided accuracy metric set, in zoom-in tiers (latency, the third tier-0 metric, is measured separately).

    Tier 0: output NRMSE over the run; pooled state NRMSE after the last call. Tier 1: output per token (own norm,
    floored) and per 1,024-token window; state per head at the final snapshot and pooled per call-end snapshot.
    Tier 2: output gain; pooled state gain after the last call, and each head's distance from gain 1 there (with the
    worst head's signed gain); state per head per snapshot. Positions are token indices (window start, snapshot end)
    or head ids.
    """
    whole = t.sum(0)
    window_starts = np.arange(0, len(t), WINDOW)
    window_sums = np.add.reduceat(t, window_starts, axis=0)
    token_nrmse = np.sqrt(t[:, 0]) / np.maximum(np.sqrt(t[:, 1]), floor)
    head = _ratio(s[..., 0], s[..., 1])  # [snapshots, heads]
    pooled = _ratio(s[..., 0].sum(-1), s[..., 1].sum(-1))
    final_gain = s[-1, :, 2] / s[-1, :, 1]
    heads = list(range(HEADS))
    head_gain_final = _tail(np.abs(final_gain - 1), heads)
    head_gain_final["worst_gain"] = float(final_gain[head_gain_final["worst_at"]])
    return {
        "tier0": {"output_nrmse": float(_ratio(whole[0], whole[1])), "state_final_nrmse": float(pooled[-1])},
        "tier1": {
            "output_token": _tail(token_nrmse, list(range(len(t)))),
            "output_window": _tail(_ratio(window_sums[:, 0], window_sums[:, 1]), window_starts.tolist()),
            "state_head_final": _tail(head[-1], heads),
            "state_window": _tail(pooled, list(state_tokens)),
        },
        "tier2": {
            "output_gain": float(whole[2] / whole[1]),
            "state_gain_final": float(s[-1, :, 2].sum() / s[-1, :, 1].sum()),
            "state_head_gain_final": head_gain_final,
            "state_head_window": _tail(
                head.ravel(), [[state_tokens[i], h] for i in range(len(state_tokens)) for h in heads]
            ),
        },
        "continuity": {"state_pooled_mean": float(pooled.mean())},
    }


def _pct(value: float) -> str:
    return f"{value * 100:.3f}%"


def _tail_rows(label: str, key: str, tier: str, where: str) -> list[tuple]:
    rows = [(f"{label} p{p}", lambda a, k=key, p=p: _pct(a["tiers"][tier][k][f"p{p}"])) for p in TIER_PERCENTILES]
    rows.append(
        (
            f"{label} worst ({where})",
            lambda a, k=key: f"{_pct(a['tiers'][tier][k]['worst'])} @ {a['tiers'][tier][k]['worst_at']}",
        )
    )
    return rows


REPORT_ROWS = {
    "tier 0": [
        ("output NRMSE", lambda a: _pct(a["tiers"]["tier0"]["output_nrmse"])),
        ("final state NRMSE", lambda a: _pct(a["tiers"]["tier0"]["state_final_nrmse"])),
    ],
    "tier 1": [
        *_tail_rows("output per token", "output_token", "tier1", "token"),
        *_tail_rows("output per 1k window", "output_window", "tier1", "window start"),
        *_tail_rows("final state per head", "state_head_final", "tier1", "head"),
        *_tail_rows("state per call", "state_window", "tier1", "token"),
    ],
    "tier 2": [
        ("output gain", lambda a: f"{a['tiers']['tier2']['output_gain']:.5f}"),
        ("final state gain", lambda a: f"{a['tiers']['tier2']['state_gain_final']:.5f}"),
        *_tail_rows("final state per head |1 - gain|", "state_head_gain_final", "tier2", "head"),
        *_tail_rows("state per head per call", "state_head_window", "tier2", "[token, head]"),
    ],
    "continuity": [("state pooled mean", lambda a: _pct(a["tiers"]["continuity"]["state_pooled_mean"]))],
}


def report(runs: dict[str, dict]) -> str:
    names = list(runs)
    lines = []
    for tier, rows in REPORT_ROWS.items():
        lines += [f"\n{tier}", "| metric | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
        lines += [f"| {label} | " + " | ".join(get(runs[n]) for n in names) + " |" for label, get in rows]
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------- vLLM-format traces


def _safetensors_array(path: Path) -> np.memmap:
    with path.open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(size))
    (name,) = [k for k in header if k != "__metadata__"]
    info = header[name]
    dtype = {"BF16": np.uint16, "F32": np.float32}[info["dtype"]]
    return np.memmap(path, dtype=dtype, mode="r", offset=8 + size + info["data_offsets"][0], shape=tuple(info["shape"]))


def _bf16(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array).astype(np.int16)).view(torch.bfloat16)


def from_trace(trace: Path, golden: Path, out: Path, layer: int = 0) -> None:
    """Write run sums for a vLLM-format KDA trace (output bf16 [T, H], state f32 [N, heads, v, k] per call)."""
    trace, golden, out = Path(trace), Path(golden), Path(out)
    manifest = json.loads((golden / "manifest.json").read_text())
    segment = manifest["segment"]
    output = _safetensors_array(trace / "kda" / f"kda_output_layer_{layer}.safetensors")
    states = _safetensors_array(trace / "kda" / f"kda_recurrent_state_layer_{layer}.safetensors")
    tokens = min(output.shape[0], manifest["tokens"])
    snapshots = states.shape[0]
    every = tokens // snapshots
    writer = RunWriter(out, tokens, snapshots, save_state=False)
    orientation = None
    began = time.perf_counter()
    for k in range(snapshots):
        start, end = k * every, (k + 1) * every
        payloads = [
            torch.load(golden / f"segment_{i:06d}.pt", weights_only=True)
            for i in range(start // segment, end // segment)
        ]
        golden_rows = torch.cat([p["output"] for p in payloads])
        golden_state = payloads[-1]["state"]["recurrent"][0]
        state = torch.from_numpy(np.ascontiguousarray(states[k]))
        if orientation is None:  # vLLM traces store V x K; check against the golden rather than assume
            direct = float(error_sums(golden_state, state, dim=(-2, -1))[:, 0].sum())
            swapped = float(error_sums(golden_state, state.transpose(-1, -2), dim=(-2, -1))[:, 0].sum())
            orientation = "transposed" if swapped < direct else "direct"
            log(f"state orientation {orientation} (error sum direct {direct:.4g}, transposed {swapped:.4g})")
        if orientation == "transposed":
            state = state.transpose(-1, -2)
        writer.write_call(
            start,
            end,
            k,
            error_sums(golden_rows, _bf16(output[start:end]).double()),
            error_sums(golden_state, state, dim=(-2, -1)),
            None,
        )
        if k % 10 == 0 or k == snapshots - 1:
            writer.flush()
            elapsed = time.perf_counter() - began
            log(f"snapshot {k + 1}/{snapshots} ETA {elapsed / (k + 1) * (snapshots - k - 1):.0f} s")
    writer.flush()
    (out / "run.json").write_text(
        json.dumps({"kind": "vllm-trace", "trace": str(trace), "golden": str(golden), "state_orientation": orientation})
    )
    (out / "DONE").write_text("from-trace\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    trace = sub.add_parser("from-trace")
    trace.add_argument("--trace", type=Path, required=True)
    trace.add_argument("--golden", type=Path, required=True)
    trace.add_argument("--out", type=Path, required=True)
    rep = sub.add_parser("report")
    rep.add_argument("run", type=Path)
    rep.add_argument("--baseline", action="append", default=[], help="name=<run dir>")
    rep.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    if args.command == "from-trace":
        torch.set_num_threads(8)
        from_trace(args.trace, args.golden, args.out)
        return
    runs = {"run": analyze(load_run(args.run))}
    for item in args.baseline:
        name, path = item.split("=", 1)
        runs[name] = analyze(load_run(Path(path)))
    print(report(runs))
    if args.json:
        args.json.write_text(json.dumps(runs))


if __name__ == "__main__":
    sys.exit(main())
