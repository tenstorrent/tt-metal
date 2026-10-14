# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Batch-1 prefill TTFT harness for GPT-OSS on a 1x4 mesh (frozen measurement contract).

Loads the full model once (KV cache always sized for 128K so a subset run measures the same configuration as the
full sweep), then for each requested length L runs prefill on a prompt of exactly L tokens (PREFILL_PROMPT):
one compile/warm call, then N timed calls of Generator.prefill_forward_text (the call the demo times; the wall
clock includes the last-token logits read back to host). Untraced: GPT-OSS has no prefill trace on P150x4, and
run_prefill raises if the generator would trace a length (the harness would no longer match the demo).

    PREFILL_LENGTHS=all PREFILL_OUT=<json> \
      pytest models/demos/gpt_oss/tests/perf/prefill_ttft.py -k 1x4

Env:
    PREFILL_PROMPT   'random' (default; fixed-seed uniform ids, the iteration-0 baseline) | 'text' (first L tokens
                     of the gate's frozen real-text stream, goldens/prompt_tokens.pt). Routed-MoE timing depends
                     on the token -> expert distribution: measure MoE changes with 'text'.
    PREFILL_LENGTHS  'all' (default) | 'fast' | comma list, e.g. '128,1k,8k'
    PREFILL_N        timed calls per length after the compile pass (default 1)
    PREFILL_OUT      JSON output path (default: <runs>/ttft/ttft_<timestamp>.json)
    PREFILL_TAG      free-form label stored in the JSON
"""

import json
import os
import platform
import statistics
import time
from pathlib import Path

import pytest
from loguru import logger

from models.demos.gpt_oss.tests.perf.prefill_common import (
    PREFILL_LENGTHS,
    RUNS_DIR,
    build_prefill_generator,
    default_n,
    env_snapshot,
    git_info,
    parse_lengths,
    prompt_mode,
    prompt_tokens,
    run_prefill,
)
from models.demos.gpt_oss.tests.test_factory import parametrize_mesh_with_fabric
from models.tt_transformers.tt.common import get_padded_prefill_len

MAX_SEQ_LEN = max(PREFILL_LENGTHS)


def _write(out_path, record):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2))
    tmp.replace(out_path)


@pytest.mark.timeout(6 * 3600)
@parametrize_mesh_with_fabric([(1, 4)])
def test_prefill_ttft(mesh_device, device_params):
    lengths = parse_lengths(os.getenv("PREFILL_LENGTHS", "all"))
    mode = prompt_mode()
    n_override = os.getenv("PREFILL_N")
    out_path = Path(
        os.getenv("PREFILL_OUT", str(RUNS_DIR / "ttft" / f"ttft_{time.strftime('%Y%m%d_%H%M%S')}.json"))
    ).resolve()
    record = {
        "harness": "models/demos/gpt_oss/tests/perf/prefill_ttft.py",
        "regime": "untraced warm wall clock of Generator.prefill_forward_text (logits to host), burst",
        "tag": os.getenv("PREFILL_TAG"),
        "prompt": mode,
        "mesh": list(mesh_device.shape),
        "max_seq_len": MAX_SEQ_LEN,
        "git": git_info(),
        "env": env_snapshot(),
        "host": platform.node(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "lengths": lengths,
        "results": {},
    }

    t0 = time.perf_counter()
    ctx = build_prefill_generator(mesh_device, MAX_SEQ_LEN)
    record["model_build_s"] = time.perf_counter() - t0
    logger.info(f"[ttft] model built in {record['model_build_s']:.1f}s")

    for L in lengths:
        n = int(n_override) if n_override else default_n(L)
        tokens = prompt_tokens(L, mode)
        compile_s, _ = run_prefill(ctx, tokens)
        samples = []
        for i in range(n):
            dt, logits = run_prefill(ctx, tokens)
            samples.append(dt * 1000.0)
            logger.info(f"[ttft] L={L} call {i + 1}/{n}: {dt * 1000.0:.1f} ms")
        res = {
            "padded_len": get_padded_prefill_len(L),
            "n": n,
            "compile_ms": compile_s * 1000.0,
            "samples_ms": samples,
            "median_ms": statistics.median(samples),
            "min_ms": min(samples),
            "max_ms": max(samples),
            "spread_pct": 100.0 * (max(samples) - min(samples)) / statistics.median(samples),
            "tok_per_s": L / (statistics.median(samples) / 1000.0),
            "top1": int(logits.argmax()),
        }
        record["results"][str(L)] = res
        logger.info(
            f"[ttft] L={L} (padded {res['padded_len']}) N={n} median={res['median_ms']:.1f} ms "
            f"min={res['min_ms']:.1f} max={res['max_ms']:.1f} compile={res['compile_ms']:.0f} ms"
        )
        record["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        _write(out_path, record)

    lines = [
        "",
        f"[ttft] results ({mode} prompt) -> {out_path}",
        f"{'length':>8} {'N':>3} {'median_ms':>11} {'min_ms':>10} {'max_ms':>10}",
    ]
    for L in lengths:
        r = record["results"][str(L)]
        lines.append(f"{L:>8} {r['n']:>3} {r['median_ms']:>11.1f} {r['min_ms']:>10.1f} {r['max_ms']:>10.1f}")
    logger.info("\n".join(lines))
