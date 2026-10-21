# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Accuracy of a Kimi-K3 ``layers.0`` KDA run against the FP64 long-context golden, from compact error sums.

A run directory holds per-token output sums and per-head recurrent-state sums against the golden (written by
``test_layer0_dev_run.py`` while the device runs, or by ``from-trace`` here for a vLLM-format trace such as
FlashKDA's or Triton's). Every metric below is derived from those sums, so every implementation is scored with the
same definitions (``tiers``):

* NRMSE ``‖actual − golden‖ / ‖golden‖`` (FP64); gain ``⟨actual, golden⟩ / ⟨golden, golden⟩``, the least-squares
  scale of the golden in the actual output, below 1 when the output is too small (PCC is blind to it);
* per-token NRMSE with the golden's token floor (``golden_floor`` over every token of the golden), so runs of any
  length share one floor;
* tier 0: output NRMSE over the run, pooled state NRMSE after the last call; tier 1: p50/p95/p99/worst of output per
  token and per 1,024-token window, final state per head, pooled state per call; tier 2: output gain, final pooled
  state gain and per-head gain, state per head per call.

Run directory layout::

    run.json             provenance (commit, host, power, fabric, golden identity, token floor) and perf
    token_sums.npy       float64 [tokens, 3]: Σ(a−g)², Σg², Σa·g over hidden, per token
    state_sums.npy       float64 [snapshots, heads, 3]: the same sums per head of the recurrent state
    state_tokens.json    token position of every state snapshot; the run's progress
    DONE                 written last

Usage::

    python layer0_run_analysis.py from-trace --trace <vLLM trace dir> --golden <golden> --out <new run dir>
    python layer0_run_analysis.py report <run dir> [--baseline name=<run dir> ...] [--json out.json]
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from models.demos.common.prefill.runners.runner_utils import load_trace_token_ids
from models.demos.deepseek_v3_d_p.tests.kda.long_context_golden import (
    MANIFEST,
    load_window,
    state_after,
    token_ids_sha256,
)
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import golden_floor

WINDOW = 1024
_FLOOR_READ_SEGMENTS = 16  # golden segments per read when computing the token floor (~1 GB at 1,024 x 7,168 FP64)


def log(message: str) -> None:
    print(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {message}", flush=True)


def error_sums(golden: torch.Tensor, actual: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """``[..., 3]`` FP64 sums over ``dim``: squared error, squared golden, cross term."""
    golden = golden.double()
    actual = actual.double()
    return torch.stack(
        [(actual - golden).square().sum(dim), golden.square().sum(dim), (actual * golden).sum(dim)], dim=-1
    )


class RunWriter:
    """Preallocated sums in an existing run directory, filled call by call, readable at any point of an
    interrupted run: ``state_tokens.json`` is written after each call's rows and marks the progress."""

    def __init__(self, run_dir: Path, tokens: int, snapshots: int, heads: int) -> None:
        self.run_dir = Path(run_dir)
        self.token_sums = np.lib.format.open_memmap(
            self.run_dir / "token_sums.npy", mode="w+", dtype=np.float64, shape=(tokens, 3)
        )
        self.token_sums[:] = np.nan
        self.state_sums = np.lib.format.open_memmap(
            self.run_dir / "state_sums.npy", mode="w+", dtype=np.float64, shape=(snapshots, heads, 3)
        )
        self.state_sums[:] = np.nan
        self.state_tokens: list[int] = []

    def write_call(self, start: int, end: int, token_sums: torch.Tensor, state_sums: torch.Tensor) -> None:
        self.token_sums[start:end] = token_sums.numpy()
        self.state_sums[len(self.state_tokens)] = state_sums.numpy()
        self.token_sums.flush()
        self.state_sums.flush()
        self.state_tokens.append(end)
        (self.run_dir / "state_tokens.json").write_text(json.dumps(self.state_tokens))


def sums_token_floor(token_sums: np.ndarray) -> float:
    """``golden_floor`` over the golden token norms in a run's sums; the golden's floor when the run covers it all."""
    return golden_floor(torch.from_numpy(np.sqrt(token_sums[:, 1])))


@functools.cache
def golden_token_floor(golden_dir: Path) -> float:
    """The golden's per-token NRMSE floor: ``manifest["token_floor"]``, or ``golden_floor`` over every golden token
    (reads the whole golden: minutes for 1M tokens, so once per process)."""
    golden_dir = Path(golden_dir)
    manifest = json.loads((golden_dir / MANIFEST).read_text(encoding="utf-8"))
    if "token_floor" in manifest:
        return manifest["token_floor"]
    end = manifest["segments"][-1]["end"]
    step = manifest["segment"] * _FLOOR_READ_SEGMENTS
    log(f"token floor: reading all {end} tokens of {golden_dir} (its manifest has no token_floor)")
    norms = [
        load_window(golden_dir, start, min(start + step, end))[0].double().square().sum(-1).sqrt()
        for start in range(0, end, step)
    ]
    return golden_floor(torch.cat(norms))


def _nrmse(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(num / den)


def load_run(run_dir: Path) -> dict:
    """A run's finished calls and its token floor (from ``run.json``, else from the golden it names)."""
    run_dir = Path(run_dir)
    state_tokens = json.loads((run_dir / "state_tokens.json").read_text())
    done = state_tokens[-1] if state_tokens else 0
    meta = json.loads((run_dir / "run.json").read_text())
    token_sums = np.asarray(np.load(run_dir / "token_sums.npy", mmap_mode="r")[:done])
    floor = meta.get("token_floor")
    if floor is None:
        golden_dir = Path(meta["golden"])
        manifest = json.loads((golden_dir / MANIFEST).read_text(encoding="utf-8"))
        if "token_floor" not in manifest and done == manifest["segments"][-1]["end"]:
            floor = sums_token_floor(token_sums)
        else:
            floor = golden_token_floor(golden_dir)
    return {
        "meta": meta,
        "token_sums": token_sums,
        "state_sums": np.asarray(np.load(run_dir / "state_sums.npy", mmap_mode="r")[: len(state_tokens)]),
        "state_tokens": state_tokens,
        "token_floor": floor,
        "complete": (run_dir / "DONE").exists(),
    }


def analyze(run: dict) -> dict:
    """The tiers of one run, its provenance, and the per-position series the plots use; JSON-serializable."""
    t = run["token_sums"]
    s = run["state_sums"]
    window_sums = np.add.reduceat(t, np.arange(0, len(t), WINDOW), axis=0)
    return {
        "tokens": len(t),
        "complete": run["complete"],
        "fabric_config": run["meta"].get("fabric_config"),
        "commit": run["meta"].get("commit"),
        "token_floor": run["token_floor"],
        "tiers": tiers(t, s, run["state_tokens"], run["token_floor"]),
        "series": {
            "window": WINDOW,
            "window_nrmse": _nrmse(window_sums[:, 0], window_sums[:, 1]).tolist(),
            "state_tokens": run["state_tokens"],
            "state_pooled": _nrmse(s[..., 0].sum(-1), s[..., 1].sum(-1)).tolist(),
            "state_pooled_gain": (s[..., 2].sum(-1) / s[..., 1].sum(-1)).tolist(),
            "state_head_nrmse": _nrmse(s[..., 0], s[..., 1]).tolist(),
        },
    }


TIER_PERCENTILES = (50, 95, 99)


def _tail(values: np.ndarray, positions: list) -> dict:
    """p50/p95/p99 (linear interpolation) and the worst value with its position."""
    worst = int(values.argmax())
    out = {f"p{p}": float(np.percentile(values, p)) for p in TIER_PERCENTILES}
    out.update(worst=float(values[worst]), worst_at=positions[worst])
    return out


def tiers(t: np.ndarray, s: np.ndarray, state_tokens: list[int], floor: float) -> dict:
    """The decided accuracy metric set, in zoom-in tiers.

    Tier 0: output NRMSE over the run; pooled state NRMSE after the last call. Tier 1: output per token (own norm,
    floored at ``floor``) and per 1,024-token window; state per head at the final snapshot and pooled per call-end
    snapshot. Tier 2: output gain; pooled state gain after the last call, and each head's distance from gain 1 there
    (with the worst head's signed gain); state per head per snapshot. Positions are token indices (window start,
    snapshot end) or head ids.
    """
    whole = t.sum(0)
    window_starts = np.arange(0, len(t), WINDOW)
    window_sums = np.add.reduceat(t, window_starts, axis=0)
    token_nrmse = np.sqrt(t[:, 0]) / np.maximum(np.sqrt(t[:, 1]), floor)
    head = _nrmse(s[..., 0], s[..., 1])  # [snapshots, heads]
    pooled = _nrmse(s[..., 0].sum(-1), s[..., 1].sum(-1))
    final_gain = s[-1, :, 2] / s[-1, :, 1]
    heads = list(range(s.shape[1]))
    head_gain_final = _tail(np.abs(final_gain - 1), heads)
    head_gain_final["worst_gain"] = float(final_gain[head_gain_final["worst_at"]])
    return {
        "tier0": {"output_nrmse": float(_nrmse(whole[0], whole[1])), "state_final_nrmse": float(pooled[-1])},
        "tier1": {
            "output_token": _tail(token_nrmse, list(range(len(t)))),
            "output_window": _tail(_nrmse(window_sums[:, 0], window_sums[:, 1]), window_starts.tolist()),
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
    "run": [
        ("tokens", lambda a: f"{a['tokens']:,}"),
        ("complete", lambda a: "yes" if a["complete"] else "no"),
        ("fabric", lambda a: a["fabric_config"] or "-"),
        ("commit", lambda a: (a["commit"] or "-")[:11]),
        ("token floor", lambda a: f"{a['token_floor']:.6g}"),
    ],
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
        *_tail_rows("final state per head distance from gain 1", "state_head_gain_final", "tier2", "head"),
        *_tail_rows("state per head per call", "state_head_window", "tier2", "[token, head]"),
    ],
}


def report(runs: dict[str, dict]) -> str:
    names = list(runs)
    lines = []
    for tier, rows in REPORT_ROWS.items():
        lines += [f"\n{tier}", "| metric | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
        lines += [f"| {label} | " + " | ".join(get(runs[n]) for n in names) + " |" for label, get in rows]
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------- vLLM-format traces


def from_trace(trace: Path, golden: Path, out: Path) -> None:
    """Write run sums for a vLLM-format KDA ``layers.0`` trace into the new directory ``out``.

    The trace must hold every token from 0 (no row window) and one state snapshot per ``kda_state_every`` tokens
    from the start (no head/tail snapshot grid), on the golden's tokens; anything else is refused, not remapped.
    """
    trace, golden, out = Path(trace), Path(golden), Path(out)
    meta = json.loads((trace / "metadata.json").read_text())
    manifest = json.loads((golden / MANIFEST).read_text(encoding="utf-8"))
    if meta.get("row_window", {}).get("enabled"):
        raise ValueError(f"{trace} keeps only a row window of the prompt; the converter needs every token from 0")
    if meta.get("kda_snapshot_grid", {}).get("enabled"):
        raise ValueError(f"{trace} keeps only head/tail state snapshots; the converter needs every snapshot from 0")
    every = meta["kda_state_every"]
    available = manifest["segments"][-1]["end"]
    if every % manifest["segment"]:
        raise ValueError(f"snapshots every {every} tokens do not fall on the golden's {manifest['segment']} segments")
    with contextlib.ExitStack() as stack:
        output_file, state_file = (
            stack.enter_context(safe_open(trace / "kda" / f"{name}_layer_0.safetensors", "pt"))
            for name in ("kda_output", "kda_recurrent_state")
        )
        output = output_file.get_slice(output_file.keys()[0])
        states = state_file.get_slice(state_file.keys()[0])
        snapshots = states.get_shape()[0]
        tokens = snapshots * every
        if output.get_shape()[0] != tokens or tokens > available:
            raise ValueError(
                f"{trace}: {output.get_shape()[0]} output rows and {snapshots} snapshots every {every} tokens "
                f"do not cover [0, {tokens}) inside the {available} golden tokens"
            )
        if token_ids_sha256(load_trace_token_ids(trace, manifest["tokens"])) != manifest["token_ids_sha256"]:
            raise ValueError(f"{trace} does not hold the golden's tokens")
        out.mkdir(parents=True, exist_ok=False)
        writer = RunWriter(out, tokens, snapshots, states.get_shape()[1])
        orientation = None
        began = time.perf_counter()
        for k in range(snapshots):
            start, end = k * every, (k + 1) * every
            golden_rows, _ = load_window(golden, start, end)
            golden_state = state_after(golden, end).recurrent[0]
            state = states[k]
            if orientation is None:  # vLLM traces store V x K; check against the golden rather than assume
                direct = float(error_sums(golden_state, state, dim=(-2, -1))[:, 0].sum())
                swapped = float(error_sums(golden_state, state.transpose(-1, -2), dim=(-2, -1))[:, 0].sum())
                orientation = "transposed" if swapped < direct else "direct"
                log(f"state orientation {orientation} (error sum direct {direct:.4g}, transposed {swapped:.4g})")
            if orientation == "transposed":
                state = state.transpose(-1, -2)
            writer.write_call(
                start, end, error_sums(golden_rows, output[start:end]), error_sums(golden_state, state, dim=(-2, -1))
            )
            if k % 10 == 0 or k == snapshots - 1:
                elapsed = time.perf_counter() - began
                log(f"snapshot {k + 1}/{snapshots} ETA {elapsed / (k + 1) * (snapshots - k - 1):.0f} s")
    run = {
        "kind": "vllm-trace",
        "trace": str(trace),
        "tracer": meta.get("tracer"),
        "vllm_version": meta.get("vllm_version"),
        "kda_state_every": every,
        "golden": str(golden),
        "token_floor": manifest.get("token_floor"),
        "state_orientation": orientation,
    }
    (out / "run.json").write_text(json.dumps(run, indent=1))
    (out / "DONE").write_text("from-trace\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    trace = sub.add_parser("from-trace")
    trace.add_argument("--trace", type=Path, required=True)
    trace.add_argument("--golden", type=Path, required=True)
    trace.add_argument("--out", type=Path, required=True, help="run directory to create")
    rep = sub.add_parser("report")
    rep.add_argument("run", type=Path)
    rep.add_argument("--baseline", action="append", default=[], help="name=<run dir>")
    rep.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    if args.command == "from-trace":
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
