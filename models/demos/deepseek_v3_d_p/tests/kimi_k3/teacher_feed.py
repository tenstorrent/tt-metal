# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""What the Kimi-K3.1 single-layer teacher-feed harnesses share (``dsa_teacher_feed.py``, ``moe_teacher_feed.py``):
a vLLM trace reader, the AttnRes state a layer's module reads rebuilt from the trace, the resumable input store, a
device run's output directory lifecycle and the selection measures. No device, no ttnn.

Each harness has three steps, and only the middle one opens a device, so CPU work never holds it:

1. ``input``: the module's input, rebuilt from the trace segment by segment (``build_input``), stored with a
   ``manifest.json`` (resumable, with ``tests/kda/long_context_golden.py``'s manifest helpers) and checked per
   segment against something the trace recorded downstream of that input (fails closed).
2. The device test runs the production module chunk by chunk over the whole prefix (``load_input``), saves each
   chunk's outputs into a run directory (``start_run`` / ``chunk_path`` / ``finish_run``).
3. ``score``: every saved row against the trace, per window of context positions, with proposed bars
   (``check_bars``).

AttnRes (block size 12): every module reads a softmax mix of the sealed block snapshots and the live in-block sum,
not the residual the trace's ``decoder_input`` alias names. A trace that records every layer's ``attn_output`` and
FFN output (``decoder_output_layer_i``: the tracer's FFN output alone) rebuilds that state by the tracer's replay rule
(``debug_trace_io.get_prefix_sum``) in vLLM's bf16 order, which reproduces its sealed snapshots bit for bit.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import torch
from safetensors import safe_open

from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.tests.kda.long_context_golden import (
    MANIFEST,
    atomic_save,
    resume_manifest,
    segment_path,
    write_manifest,
)

EPS = KimiK3Config.RMS_NORM_EPS
ATTENTION, FFN = "attention", "ffn"  # the two AttnRes read sites of a layer: before its attention, before its FFN


class Trace:
    """A ``chunked_group_a_v1`` vLLM trace: each stream a directory of row shards listed in ``index.json``."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.streams = json.loads((self.path / "index.json").read_text())["tensor_streams"]
        self.metadata = json.loads((self.path / "metadata.json").read_text())

    @property
    def prompt_len(self) -> int:
        return int(self.metadata["n_prompt_tokens"])

    def has(self, key: str) -> bool:
        return key in self.streams

    def rows(self, key: str, start: int, end: int) -> torch.Tensor:
        """Rows ``[start, end)`` of one stream in its stored dtype."""
        parts = []
        for chunk in self.streams[key]["chunks"]:
            lo, hi = chunk["row_start"], chunk["row_end"]
            if hi <= start or lo >= end:
                continue
            with safe_open(str(self.path / chunk["path"]), framework="pt") as handle:
                parts.append(handle.get_slice(key)[max(start, lo) - lo : min(end, hi) - lo])
        if not parts:
            raise ValueError(f"{self.path.name}: no {key} rows in [{start}, {end})")
        return torch.cat(parts)


def replay_missing(trace: Trace, layer: int, site: str = ATTENTION) -> list[int]:
    """The layers whose attn_output or FFN output the replay to ``layer``'s ``site`` needs and the trace lacks: every
    layer below ``layer``, and at the FFN site ``layer``'s own attention output too."""
    missing = [
        l for l in range(layer) if not (trace.has(f"attn_output_layer_{l}") and trace.has(f"decoder_output_layer_{l}"))
    ]
    if site == FFN and not trace.has(f"attn_output_layer_{layer}"):
        missing.append(layer)
    return missing


def replay_live_and_sealed(
    trace: Trace, start: int, end: int, layer: int, site: str = ATTENTION
) -> tuple[torch.Tensor, torch.Tensor]:
    """AttnRes state the read at ``layer``'s ``site`` sees for rows ``[start, end)``: (live sum ``[n, d]``, sealed
    snapshots ``[n, S, d]``), by the tracer's replay rule in bf16 (vLLM's order, which reproduces its snapshots bit
    for bit): P = emb; per layer L: P += ffn_{L-1} (L > 0); at a block start seal P and restart P = attn_L, else
    P += attn_L. The attention site of ``layer`` is P before ``layer``'s seal with ffn_{layer-1} added; its FFN site
    is P after ``layer``'s own seal and attention (``hf_walk.hf_layer``). ``layer`` >= 1."""
    assert site in (ATTENTION, FFN), f"site {site!r}: {ATTENTION!r} or {FFN!r}"
    assert layer >= 1, "layer 0's attention reads no AttnRes; its FFN site would need no replay either"
    block = KimiK3Config.ATTN_RES_BLOCK_SIZE
    last = layer if site == FFN else layer - 1  # the last layer whose attention output enters the state
    keys = ["decoder_input_layer_0"]
    for l in range(last + 1):
        keys.append(f"attn_output_layer_{l}")
        if l < layer:
            keys.append(f"decoder_output_layer_{l}")
    with ThreadPoolExecutor(max_workers=16) as pool:
        streams = dict(zip(keys, pool.map(lambda k: trace.rows(k, start, end), keys)))
    live = streams["decoder_input_layer_0"]
    sealed = []
    for l in range(last + 1):
        if l > 0:
            live = live + streams[f"decoder_output_layer_{l - 1}"]
        if l % block == 0:
            sealed.append(live)
            live = streams[f"attn_output_layer_{l}"]
        else:
            live = live + streams[f"attn_output_layer_{l}"]
    if site == ATTENTION:
        live = live + streams[f"decoder_output_layer_{layer - 1}"]
    return live, torch.stack(sealed, dim=1)


def checkpoint_tensors(checkpoint: Path, layer: int | None, names: list[str]) -> dict[str, torch.Tensor]:
    """Tensors ``layers.<layer>.<name>`` (``layer`` None: ``<name>`` under the model root, e.g. ``embed_tokens.weight``)
    of a K3 / K3.1 checkpoint (either key root), keyed by ``name``, read through its safetensors index opening each
    shard once. Imports no device code (the K3 golden's loader imports the device adapters, and so ttnn)."""
    weight_map = json.loads((Path(checkpoint) / "model.safetensors.index.json").read_text())["weight_map"]
    base = "" if layer is None else f"layers.{layer}."
    roots = [r for r in ("model.", "language_model.model.") if f"{r}{base}{names[0]}" in weight_map]
    if not roots:
        raise ValueError(f"{checkpoint} has no {base}{names[0]}")
    full = {f"{roots[0]}{base}{n}": n for n in names}
    by_shard: dict[str, list[str]] = {}
    for key in full:
        by_shard.setdefault(weight_map[key], []).append(key)
    out = {}
    for shard, keys in by_shard.items():
        with safe_open(str(Path(checkpoint) / shard), framework="pt") as handle:
            out.update({full[k]: handle.get_tensor(k) for k in keys})
    return out


def rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """The model reference's RMSNorm (`modeling_kimi_k3_mla.KimiRMSNorm`) in bf16: normalize in fp32, round to bf16,
    then scale by the bf16 weight (a second rounding)."""
    xf = x.float()
    normed = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)).to(torch.bfloat16)
    return weight.to(torch.bfloat16) * normed


def chain_identity(chain: Path) -> dict:
    """What a teacher-fed input built from an FP64 chain run (``fp64_chain.py``) records of its source, under the keys
    a trace-built input uses (``trace``, ``trace_commit``, ``checkpoint``) plus the chain's own identity; ``chain`` in an
    input's manifest also tells the scorers to score against the chain."""
    m = json.loads((Path(chain) / MANIFEST).read_text())
    return {
        "chain": str(chain),
        "chain_producer_version": m["producer_version"],
        "chain_token_ids_sha256": m["token_ids_sha256"],
        "trace": m["identity"].get("token_source"),
        "trace_commit": m["identity"].get("trace_commit"),
        "checkpoint": m["identity"].get("checkpoint"),
    }


def build_input(
    *,
    out: Path,
    expected: dict,
    count: int,
    segment: int,
    make: Callable[[int, int], tuple[torch.Tensor, dict[str, float]]],
    check_bars: dict[str, float],
) -> dict:
    """Write (or resume) a teacher-fed input of ``count`` rows in ``segment``-row pieces into ``out``.

    ``make(start, end)`` returns the rows' bf16 input ``[end - start, d]`` and its check ``{name: value}``; a segment
    whose check is below its ``check_bars[name]`` fails the build before anything of it is stored. ``expected`` (the
    input's identity: layer, trace, rebuild rule, rounding, ...) must match a resumed manifest's."""
    out.mkdir(parents=True, exist_ok=True)
    segments = -(-count // segment)
    manifest = resume_manifest(out, {**expected, "count": count, "segment": segment}, segments)
    for index in range(len(manifest["segments"]), segments):
        start, end = index * segment, min((index + 1) * segment, count)
        began = time.perf_counter()
        normed, check = make(start, end)
        failed = {k: v for k, v in check.items() if k in check_bars and not v >= check_bars[k]}
        if failed:
            raise ValueError(f"rows [{start}, {end}): rebuilt input fails its check {failed} (bars {check_bars})")
        atomic_save({"start": start, "end": end, "normed": normed}, segment_path(out, index))
        seconds = time.perf_counter() - began
        manifest["segments"].append({"start": start, "end": end, "seconds": round(seconds, 2), **check})
        write_manifest(out, manifest)
        print(
            f"{time.strftime('%H:%M:%S')} segment {index + 1}/{segments} rows [{start}, {end}) {seconds:.1f} s "
            + " ".join(f"{k}={v:.6f}" for k, v in check.items()),
            flush=True,
        )
    return manifest


def load_input(run_dir: Path, start: int, end: int) -> torch.Tensor:
    """Teacher-fed input rows ``[start, end)`` as bf16 ``[n, d]``; the input must hold them."""
    manifest = json.loads((run_dir / MANIFEST).read_text())
    segment = manifest["segment"]
    available = manifest["segments"][-1]["end"] if manifest["segments"] else 0
    if not 0 <= start < end <= available:
        raise ValueError(f"rows [{start}, {end}) are not inside the {available} input rows of {run_dir}")
    rows = []
    for index in range(start // segment, -(-end // segment)):
        payload = torch.load(segment_path(run_dir, index), weights_only=True)
        rows.append(payload["normed"][max(start, payload["start"]) - payload["start"] : end - payload["start"]])
    return torch.cat(rows)


# --- a device run's output directory ---------------------------------------------------------------------------

# What a device run writes into its output directory; a new run clears all of it first.
RUN_FILES = ("chunk_*.pt", "caches.pt", "metrics.json", "windows.tsv", "rows.pt", "host.json", MANIFEST)


def chunk_path(out: Path, index: int) -> Path:
    return out / f"chunk_{index:06d}.pt"


def start_run(out: Path, manifest: dict) -> None:
    """Begin a device run in ``out``: remove every file an earlier run left (so no stale chunk or cache can be
    scored with this run's) and write this run's manifest, not yet marked complete. Refuses ``out`` when it is the
    run's input directory or holds a manifest that is not a device run's (no ``chunk``), so a mistyped directory is
    never cleared."""
    if "input" in manifest and out.resolve() == Path(manifest["input"]).resolve():
        raise ValueError(f"{out} is the run's input directory; a device run needs its own output directory")
    if (out / MANIFEST).is_file() and "chunk" not in json.loads((out / MANIFEST).read_text()):
        raise ValueError(f"{out} holds a manifest that is not a device run's; refusing to clear it")
    out.mkdir(parents=True, exist_ok=True)
    for pattern in RUN_FILES:
        for stale in out.glob(pattern):
            stale.unlink()
    write_manifest(out, {**manifest, "complete": False})


def finish_run(out: Path, manifest: dict, seconds: float) -> None:
    """Mark the run in ``out`` complete, after everything it writes is written; ``complete_run`` requires it."""
    write_manifest(out, {**manifest, "complete": True, "seconds": round(seconds, 1)})


def complete_run(out: Path) -> dict:
    """The manifest of the complete device run in ``out``; refuses a run that is not marked complete."""
    manifest = json.loads((out / MANIFEST).read_text())
    if not manifest.get("complete"):
        raise ValueError(f"{out} holds no complete device run (its manifest is not marked complete)")
    return manifest


# --- selection measures and bars --------------------------------------------------------------------------------


def quantile(t: torch.Tensor, q: float) -> float:
    """The ``q`` quantile of all of ``t``'s values (linear interpolation, as ``torch.quantile``), NaN if empty; for any
    size (``torch.quantile`` refuses more than 2^24 values, which 1M rows' top-k misses exceed)."""
    v = t.flatten().float().sort().values
    if not v.numel():
        return float("nan")
    pos = q * (v.numel() - 1)
    lo = int(pos)
    hi = min(lo + 1, v.numel() - 1)
    return float(v[lo] + (v[hi] - v[lo]) * (pos - lo))


def topk_recall(got: torch.Tensor, want: torch.Tensor) -> torch.Tensor:
    """Per row the fraction of `want`'s valid (>= 0) keys that occur in `got`, i.e. |got ∩ want| / |want| as sets
    (a key `got` holds twice counts once). Both `[rows, k]` int64; `got`'s no-key entries (the uint32 0xFFFFFFFF
    sentinel, or -1) match nothing. Rows with no valid `want` entry read NaN."""
    rows = torch.arange(want.shape[0]).unsqueeze(1) << 32
    valid = want >= 0
    return (torch.isin(rows + want, rows + got) & valid).sum(dim=1) / valid.sum(dim=1)


def missed_keys(got: torch.Tensor, want: torch.Tensor, scores: torch.Tensor) -> dict[str, torch.Tensor]:
    """Every trace key the device missed, one entry per miss: its ``row`` (index into the chunk), its ``rank`` in the
    trace's own selection ordered by the trace's score (0 = best), that selection's size ``selected``, and its
    ``spread_share`` (score - cut) / (top - cut), the cut being the lowest selected score. A rank near ``selected``
    or a share near 0 is a near-tie at the cut."""
    rows = torch.arange(want.shape[0]).unsqueeze(1) << 32
    valid = want >= 0
    missed = ~torch.isin(rows + want, rows + got) & valid
    masked = scores.masked_fill(~valid, -float("inf"))
    rank = torch.empty_like(want)
    rank.scatter_(1, masked.argsort(dim=1, descending=True), torch.arange(want.shape[1]).expand_as(want).contiguous())
    top = masked.max(dim=1, keepdim=True).values
    cut = scores.masked_fill(~valid, float("inf")).min(dim=1, keepdim=True).values
    share = (scores - cut) / (top - cut).clamp_min(1e-30)
    row = torch.arange(want.shape[0]).unsqueeze(1).expand_as(want)
    return {
        "row": row[missed],
        "rank": rank[missed],
        "selected": valid.sum(dim=1, keepdim=True).expand_as(want)[missed],
        "spread_share": share[missed].float(),
    }


def check_bars(windows: list[dict], bars: dict, growth: tuple[str, float]) -> list[str]:
    """The windows that break a bar of ``bars`` (``{metric: ("min" | "max", value)}``; NaN, a metric a window does
    not define, passes), and growth of ``growth[0]`` (worst window over the first) beyond ``growth[1]``."""
    failures = []
    for name, (kind, bar) in bars.items():
        for w in windows:
            v = w[name]
            if v == v and (v < bar if kind == "min" else v > bar):
                failures.append(f"{name} {v:.6f} {'<' if kind == 'min' else '>'} {bar} at [{w['start']}, {w['end']})")
    metric, factor = growth
    defined = [w for w in windows if w[metric] == w[metric]]  # (a window without rows for the metric reads NaN)
    if not defined:
        return failures
    first = defined[0][metric]
    worst = max(defined, key=lambda w: w[metric])
    if worst[metric] > factor * first:
        ratio = worst[metric] / first if first > 0 else float("inf")  # (from an exact first window, any error grows)
        failures.append(f"{metric} grows {ratio:.3f}x (> {factor}) by [{worst['start']}, {worst['end']})")
    return failures
