# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared plumbing for the Qwen3.8-27B campaign benchmarks.

BENCH_JSON emission (one greppable line per result), latency statistics,
TP GDN state snapshot/restore around trace capture, and offline-safe prompt
synthesis. Consumed by bench_decode.py / bench_prefill.py; records are read
back by parse_bench.py.
"""

import json
import os
import subprocess
import time

import torch

import ttnn


def git_ref():
    """Short SHA of the tree under test; the REF env (set by run_bench.sbatch) wins."""
    ref = os.environ.get("REF")
    if ref:
        return ref
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def stats_ms(samples_s):
    """min/median/p90/max/mean over per-step wall times (seconds in, milliseconds out)."""
    xs = sorted(samples_s)
    n = len(xs)
    if n == 0:
        return {}
    median = xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])
    p90 = xs[min(n - 1, max(0, -(-9 * n // 10) - 1))]
    return {
        "n": n,
        "min_ms": round(xs[0] * 1000.0, 3),
        "median_ms": round(median * 1000.0, 3),
        "p90_ms": round(p90 * 1000.0, 3),
        "max_ms": round(xs[-1] * 1000.0, 3),
        "mean_ms": round(sum(xs) / n * 1000.0, 3),
    }


def emit_bench_json(kind, config, metrics):
    """Print one greppable BENCH_JSON line; returns the record for further logging."""
    record = {
        "kind": kind,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "ref": git_ref(),
        "mesh": os.environ.get("MESH_DEVICE", "P150x4"),
        "hf_model": os.environ.get("HF_MODEL", ""),
        "config": config,
        "metrics": metrics,
    }
    # Flush so the line survives in a partial log if the job is killed right after.
    print("BENCH_JSON " + json.dumps(record), flush=True)
    return record


def gdn_layers(model):
    return [layer.attention for layer in model.layers if not layer.is_full_attention]


def snapshot_gdn_tp(model):
    """Host snapshot of TP GDN recurrent+conv state.

    Trace capture runs the forward once as a throwaway, which advances the
    non-idempotent GDN state; the snapshot is restored afterwards so measured
    decode starts from the true post-prefill state (mirrors text_demo).
    """
    comp = ttnn.ConcatMeshToTensor(model.mesh_device, dim=0)
    return [
        (
            ttnn.to_torch(dn.rec_state, mesh_composer=comp),
            [ttnn.to_torch(c, mesh_composer=comp) for c in dn.conv_states],
        )
        for dn in gdn_layers(model)
    ]


def restore_gdn_tp(model, snap):
    """Restore a snapshot_gdn_tp result via ttnn.copy (preserves trace-baked addresses)."""
    mesh = model.mesh_device
    mapper = ttnn.ShardTensorToMesh(mesh, dim=0)

    def _back(t, dtype):
        return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=mapper)

    for dn, (rec, convs) in zip(gdn_layers(model), snap):
        r = _back(rec, dn.rec_state.dtype)
        ttnn.copy(r, dn.rec_state)
        ttnn.deallocate(r)
        for j, c in enumerate(convs):
            cc = _back(c, dn.conv_states[j].dtype)
            ttnn.copy(cc, dn.conv_states[j])
            ttnn.deallocate(cc)


def bench_prompt(isl, tokenizer):
    """~isl tokens for benchmarking.

    Default is offline-safe: tile the local shared question file to length
    (device timing is content-independent, so repeated text is fine). Set
    QWEN38_BENCH_REAL_PROMPT=1 for the corpus-backed prompts text_demo uses
    (requires network on first run to populate the context cache).
    """
    from models.demos.blackhole.qwen36.demo.text_demo import _FRANKENSTEIN_CONFIGS, _get_prompt

    if os.environ.get("QWEN38_BENCH_REAL_PROMPT") == "1" and isl > 256:
        keys = sorted(_FRANKENSTEIN_CONFIGS)
        key = next((k for k in keys if k >= isl), keys[-1])
        ids = _get_prompt(key, tokenizer, max_prompt_len=isl)
        if ids.shape[1] >= isl:
            return ids[:, :isl]
        # Corpus shorter than requested: fall through to tiling below.

    ids = _get_prompt(min(isl, 256), tokenizer)
    while ids.shape[1] < isl:
        ids = torch.cat([ids, ids], dim=1)
    return ids[:, :isl]
