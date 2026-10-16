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
  slowest-decaying head 6; per-head means over snapshots.

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


def analyze(run: dict, first_call: int = 5120) -> dict:
    """Every metric of one run; JSON-serializable."""
    t = run["token_sums"]
    s = run["state_sums"]
    tokens = len(t)
    whole = t.sum(0)
    windows = np.add.reduceat(t, np.arange(0, tokens, WINDOW), axis=0)
    window_nrmse = _ratio(windows[:, 0], windows[:, 1])
    window_gain = windows[:, 2] / windows[:, 1]
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
            "without_h6": pooled_without_h6.tolist(),
            "median_head": np.median(head, axis=1).tolist(),
            "head_nrmse": {str(h): head[:, h].tolist() for h in LONG_MEMORY_HEADS},
            "head_gain": {str(h): head_gain[:, h].tolist() for h in LONG_MEMORY_HEADS},
            "head_nrmse_mean_all": head_mean.tolist(),
        },
    }
    return result


HEADLINE = (
    ("output 1M NRMSE", lambda a: a["output"]["nrmse"], "pct"),
    ("output first 5k", lambda a: a["output"]["first_call_nrmse"], "pct"),
    ("output max 1k window", lambda a: a["output"]["window_nrmse_max"], "pct"),
    ("output gain", lambda a: a["output"]["gain"], "gain"),
    ("token p99", lambda a: a["output"]["token_nrmse_p99"], "pct"),
    ("state pooled mean", lambda a: a["state"]["pooled_mean"], "pct"),
    ("state h6 mean", lambda a: a["state"]["heads"]["6"]["nrmse_mean"], "pct"),
    ("state h6 final gain", lambda a: a["state"]["heads"]["6"]["gain_final"], "gain"),
    ("state median head", lambda a: a["state"]["median_head_mean"], "pct"),
    ("state other 95 heads", lambda a: a["state"]["without_h6_mean"], "pct"),
)


def _fmt(value: float, kind: str) -> str:
    return f"{value * 100:.3f}%" if kind == "pct" else f"{value:.5f}"


def report(runs: dict[str, dict]) -> str:
    names = list(runs)
    lines = ["| metric | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    for label, get, kind in HEADLINE:
        lines.append(f"| {label} | " + " | ".join(_fmt(get(runs[n]), kind) for n in names) + " |")
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
