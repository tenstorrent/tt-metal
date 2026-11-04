# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Batch-32 prefill TTFT harness for GPT-OSS on a 1x4 mesh (frozen measurement contract).

Builds the model for global batch 32 the way the demo's non-row-sharded path would (max_batch_size=32,
users_row_sharded=False, paged KV for 32 users; prefill_common.build_prefill_generator_b32), then for each length L
feeds 32 distinct prompts of exactly L tokens to ONE Generator.prefill_forward_text call (untraced, last-token logits
of all 32 users to host): one compile/warm call, then N timed calls. Metric: wall ms of the whole 32-user call
("batch TTFT"); per-user ms = batch ms / 32.

    PREFILL_PROMPT=text PREFILL_LENGTHS=128,512,1k,2k,4k,8k,16k,32k PREFILL_OUT=<json> \
      pytest models/demos/gpt_oss/tests/perf/prefill_ttft_b32.py -k 1x4

Env:
    PREFILL_PROMPT   'text' (default here; user u = tokens [4096 u, 4096 u + L) of the gate's frozen stream,
                     goldens/prompt_tokens.pt, wrapping) | 'random' (fixed seed per user)
    PREFILL_LENGTHS  'all' (default: 128, 512, 1K ... 128K) | 'fast' | comma list, e.g. '128,1k,8k'
    PREFILL_N        timed calls per length after the compile call (default 1)
    PREFILL_B32_KV_LEN  KV tokens per user (page-table width x 64). Default: max(B32_DEFAULT_KV_LEN, max length of
                     the run), so every run up to the default measures the same KV configuration; a longer length
                     (fit probe) raises it for that process only. The JSON records it.
    PREFILL_OUT      JSON output path (default: <runs>-b32/ttft/ttft_b32_<timestamp>.json)
    PREFILL_TAG      free-form label stored in the JSON

Fit: a model build that fails (KV allocation) or a length that fails (activations) is recorded in the JSON
(build_error / results[L].error with the exception text) and the test fails. Run the largest lengths alone in a fresh
process.
"""

import json
import os
import platform
import statistics
import time
import traceback
from pathlib import Path

import pytest
from loguru import logger

from models.demos.gpt_oss.tests.perf.prefill_common import (
    B32_BATCH,
    B32_PROMPT_STRIDE,
    b32_prompt_tokens,
    build_prefill_generator_b32,
    default_n,
    dram_view,
    env_snapshot,
    git_info,
    kv_bytes_per_device,
    padded_prefill_len,
    parse_lengths,
    run_prefill_batch,
)
from models.demos.gpt_oss.tests.test_factory import parametrize_mesh_with_fabric

# Per-user KV of the frozen configuration: the largest length at which 32 users fit (fit table, README).
B32_DEFAULT_KV_LEN = 64 * 1024
B32_RUNS_DIR = Path(
    os.getenv(
        "PREFILL_B32_RUNS_DIR", str(Path(__file__).resolve().parents[5] / "generated" / "gpt_oss_prefill" / "runs_b32")
    )
)


def _write(out_path, record):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2))
    tmp.replace(out_path)


def _err(e):
    # TT_FATAL messages lead with the cause (e.g. "Out of Memory: ... B DRAM buffer ...") and end with a native
    # backtrace, so keep the head of the message.
    msg = str(e)
    return {"type": type(e).__name__, "message": msg[:2000], "traceback": traceback.format_exc()[-6000:]}


@pytest.mark.timeout(6 * 3600)
@parametrize_mesh_with_fabric([(1, 4)])
def test_prefill_ttft_b32(mesh_device, device_params):
    lengths = parse_lengths(os.getenv("PREFILL_LENGTHS", "all"))
    mode = os.getenv("PREFILL_PROMPT", "text").strip().lower()
    assert mode in ("text", "random"), mode
    os.environ["PREFILL_PROMPT"] = mode  # env_snapshot()._prompt_mode reports the mode actually used
    assert os.getenv("PREFILL_TRACE", "0") == "0", "the batch-32 harness is untraced only"
    n_override = os.getenv("PREFILL_N")
    kv_len = int(os.getenv("PREFILL_B32_KV_LEN", "0")) or max(B32_DEFAULT_KV_LEN, max(lengths))
    out_path = Path(
        os.getenv("PREFILL_OUT", str(B32_RUNS_DIR / "ttft" / f"ttft_b32_{time.strftime('%Y%m%d_%H%M%S')}.json"))
    ).resolve()
    record = {
        "harness": "models/demos/gpt_oss/tests/perf/prefill_ttft_b32.py",
        "regime": "untraced warm wall clock of ONE Generator.prefill_forward_text call for all 32 users "
        "(last-token logits of every user to host), burst",
        "batch": B32_BATCH,
        "traced": False,
        "tag": os.getenv("PREFILL_TAG"),
        "prompt": mode,
        "prompt_stride": B32_PROMPT_STRIDE if mode == "text" else None,
        "mesh": list(mesh_device.shape),
        "kv_len_per_user": kv_len,
        "kv_num_blocks": B32_BATCH * kv_len // 64,
        "kv_bytes_per_device_expected": kv_bytes_per_device(B32_BATCH * kv_len // 64),
        "git": git_info(),
        "env": env_snapshot(),
        "host": platform.node(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "lengths": lengths,
        "results": {},
    }
    record["dram_before_build"] = dram_view(mesh_device)

    t0 = time.perf_counter()
    try:
        ctx = build_prefill_generator_b32(mesh_device, kv_len)
    except Exception as e:  # noqa: BLE001 -- fit table: record the exact allocation error
        record["build_error"] = _err(e)
        record["build_failed_after_s"] = time.perf_counter() - t0
        try:
            record["dram_at_build_error"] = dram_view(mesh_device)
        except Exception:  # noqa: BLE001
            pass
        record["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        _write(out_path, record)
        logger.error(f"[ttft_b32] model build failed (kv {B32_BATCH} x {kv_len}): {e}")
        raise
    record["model_build_s"] = time.perf_counter() - t0
    for k in ("kv_dtype", "kv_shape_per_device", "kv_bytes_per_device", "kv_num_blocks"):
        record[k] = ctx[k]
    record["dram_after_build"] = dram_view(mesh_device)
    _write(out_path, record)
    logger.info(
        f"[ttft_b32] model built in {record['model_build_s']:.1f}s; KV {ctx['kv_dtype']} "
        f"{ctx['kv_shape_per_device']} per tensor, {ctx['kv_bytes_per_device'] / 1e9:.2f} GB/device; "
        f"DRAM free {record['dram_after_build'].get('total_bytes_free_per_device_gb', float('nan')):.2f} GB/device"
    )

    failed = None
    for L in lengths:
        assert L <= kv_len, f"L={L} > KV per user {kv_len}"
        n = int(n_override) if n_override else default_n(L)
        tokens = b32_prompt_tokens(L, mode)
        padded = padded_prefill_len(ctx["model_args"], L)
        res = {"padded_len": padded, "n": n}
        try:
            compile_s, _ = run_prefill_batch(ctx, tokens)
            res["compile_ms"] = compile_s * 1000.0
            samples = []
            for i in range(n):
                dt, logits = run_prefill_batch(ctx, tokens)
                samples.append(dt * 1000.0)
                logger.info(f"[ttft_b32] L={L} call {i + 1}/{n}: {dt * 1000.0:.1f} ms")
        except Exception as e:  # noqa: BLE001 -- fit table: record the exact error and stop
            res["fits"] = False
            res["error"] = _err(e)
            try:
                res["dram_at_error"] = dram_view(mesh_device)
            except Exception:  # noqa: BLE001
                pass
            record["results"][str(L)] = res
            record["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            _write(out_path, record)
            logger.error(f"[ttft_b32] L={L} failed: {e}")
            failed = (L, e)
            break
        med = statistics.median(samples)
        res.update(
            {
                "fits": True,
                "samples_ms": samples,
                "median_ms": med,
                "min_ms": min(samples),
                "max_ms": max(samples),
                "spread_pct": 100.0 * (max(samples) - min(samples)) / med,
                "per_user_ms": med / B32_BATCH,
                "tok_per_s": B32_BATCH * L / (med / 1000.0),
                "top1": [int(x) for x in logits.argmax(-1)],
                "dram_after": dram_view(mesh_device),
            }
        )
        record["results"][str(L)] = res
        logger.info(
            f"[ttft_b32] L={L} (padded {padded}) N={n} batch median={med:.1f} ms per-user={med / B32_BATCH:.1f} ms "
            f"compile={res['compile_ms']:.0f} ms"
        )
        record["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        _write(out_path, record)

    lines = [
        "",
        f"[ttft_b32] results ({mode} prompt, batch {B32_BATCH}, KV {B32_BATCH} x {kv_len}) -> {out_path}",
        f"{'length':>8} {'N':>3} {'batch_ms':>11} {'per_user_ms':>12} {'min_ms':>10} {'max_ms':>10}",
    ]
    for L in lengths:
        r = record["results"].get(str(L))
        if r and r.get("fits"):
            lines.append(
                f"{L:>8} {r['n']:>3} {r['median_ms']:>11.1f} {r['per_user_ms']:>12.2f} {r['min_ms']:>10.1f} "
                f"{r['max_ms']:>10.1f}"
            )
        elif r:
            lines.append(f"{L:>8} FAILED: {r['error']['type']}: {r['error']['message'][:200]}")
    logger.info("\n".join(lines))
    if failed:
        raise RuntimeError(f"L={failed[0]} failed: {failed[1]}") from failed[1]
