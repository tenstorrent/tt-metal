# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Standalone bench of the device prompt enhancer (Gemma-4-E2B) on the Galaxy 4x8 ring, without LTX.

Same mesh params, fabric topology and enhancer class as the traced LTX distilled pipeline, but no DiT,
VAE or encoder: load, eager warmup, decode-trace capture, then a fixed prompt set rewritten at each
temperature. Each request records its timings, host staging counts and the generated token ids, so two
runs with different knobs can be compared token for token (BENCH_REF / BENCH_REF_TAG) as well as on latency.

    BENCH_TEMPS        comma list of temperatures (default "0.0,0.7")
    BENCH_REPEATS      passes over the prompt set per temperature (default 2; pass 0 is reported as cold)
    BENCH_MAX_NEW      token budget (default: the enhancer's own, 512)
    BENCH_IGNORE_STOP  1: decode the full budget past stop ids (fixed-length latency; warmup/capture unaffected)
    BENCH_TAG          output name (default "bench")
    BENCH_REF          path of an earlier bench JSON: report token equality per (prompt, temperature, repeat)
    BENCH_REF_TAG      the same by tag: <out dir>/<tag>.json (BENCH_REF wins when both are set)
    BENCH_ROW_CHECK    1: after each request, read element 0 of the decode token-feedback buffer on the first
                       device of every mesh row and record whether the rows agree
    LTX_ENHANCER_EXP_DIR  output root (JSON lands under <root>/enhancer_bench/)

Each request's ``calls`` counts host touch points (H2D copies, trace launches, host PLI, decode restages, eager
sampler runs, seed uploads); the summary also gives them per decode step.
"""

import json
import os
import time

import pytest
from loguru import logger

import ttnn
from models.tt_dit.pipelines.ltx.prompt_enhancer import ENHANCER_SEED, DevicePromptEnhancer, default_ltx_enhancer
from models.tt_dit.tests.models.ltx.ltx_mesh_params import LTX_DISTILLED_MESH_PARAMS_DL
from models.tt_dit.utils.test import skip_if_unsupported_num_links

OUT_DIR = os.path.join(
    os.environ.get("LTX_ENHANCER_EXP_DIR", os.path.expanduser("~/ltx_enhancer_experiments")), "enhancer_bench"
)
GALAXY_RING_TRACED = [p for p in LTX_DISTILLED_MESH_PARAMS_DL if p.id == "4x8sp1tp0nl2_ring_is_fsdp0"]

# Raw prompts as a user would type them: short ones land in the 1024 prefill bucket, the long one in 2048.
BENCH_PROMPTS = (
    "beekeeper",
    "a cat sleeping on a sunny windowsill",
    "a red paper boat drifting across a still pond at dusk",
    "drone shot over a snowy mountain ridge at sunrise, wind blowing snow off the peaks",
    "a chef in a busy restaurant kitchen flips vegetables in a flaming wok while orders are called out",
    "a woman in a grey wool coat is walking along a rain-wet harbour promenade at dusk, holding a paper cup of "
    "coffee in both hands. Fishing boats rock gently at their moorings behind her and strings of warm bulbs sway "
    "between the lamp posts. She pauses at the railing and takes a slow sip as a gull lands on the post beside her",
)

# Host PLI itself: compute_host_pli wraps it per decode token, and prefill calls it once per request.
MODEL_COUNTERS = (
    "prepare_decode_inputs_host",
    "compute_host_pli",
    "_compute_per_layer_inputs",
    "prepare_inputs_prefill",
)


def _env_flag(name):
    return os.environ.get(name, "0").strip().lower() in ("1", "true", "yes")


def _counting(obj, name, counts, key=None, restores=None):
    """Wrap ``obj.name`` so every call bumps ``counts[key or name]``; skipped when ``obj`` has no ``name``."""
    orig = getattr(obj, name, None) if obj is not None else None
    if orig is None:
        return
    key = key or name
    counts.setdefault(key, 0)

    def wrapped(*args, **kwargs):
        counts[key] += 1
        return orig(*args, **kwargs)

    setattr(obj, name, wrapped)
    if restores is not None:
        restores.append((obj, name, orig))


def _row_tokens(generator, mesh_shape):
    """Element 0 of the decode token-feedback buffer (the trace's token input, which the device sampler writes)
    on the first device of each mesh row. None when the generator keeps no such buffer (host PLI restage)."""
    feedback_buffer = getattr(generator, "_gemma4_eager_token_feedback_buffer", None)
    if feedback_buffer is None:
        return None
    fb = feedback_buffer(0)
    # Before a decode trace exists gemma4 hands out a pre-capture dummy instead; that is not the live buffer.
    if fb is None or fb is getattr(generator, "_g4_warmup_token_feedback_dummy", None):
        return None
    shards = ttnn.get_device_tensors(fb)
    rows, cols = int(mesh_shape[0]), int(mesh_shape[1])
    return [int(ttnn.to_torch(shards[r * cols]).reshape(-1)[0].item()) for r in range(rows)]


def _first_divergence(a, b):
    div = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), None)
    if div is None and len(a) != len(b):
        div = min(len(a), len(b))
    return div


def _mean(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 3) if values else None


@pytest.mark.parametrize(
    "mesh_device, sp_axis, tp_axis, num_links, device_params, topology, is_fsdp, dynamic_load",
    GALAXY_RING_TRACED,
    indirect=["mesh_device", "device_params"],
)
def test_enhancer_bench(mesh_device, device_params, sp_axis, tp_axis, num_links, dynamic_load, topology, is_fsdp):
    skip_if_unsupported_num_links(mesh_device, num_links)
    os.makedirs(OUT_DIR, exist_ok=True)
    temps = [float(t) for t in os.environ.get("BENCH_TEMPS", "0.0,0.7").split(",") if t.strip()]
    repeats = int(os.environ.get("BENCH_REPEATS", "2"))
    tag = os.environ.get("BENCH_TAG", "bench")
    row_check = _env_flag("BENCH_ROW_CHECK")

    # The pipeline runs the enhancer on its own full-mesh submesh with the pipeline's fabric topology.
    submesh = mesh_device.create_submesh(ttnn.MeshShape(*tuple(mesh_device.shape)))
    enhancer = DevicePromptEnhancer(
        default_ltx_enhancer(), mesh_device=submesh, ccl_topology=topology, enable_trace=True
    )
    if os.environ.get("BENCH_MAX_NEW"):
        enhancer.max_new_tokens = int(os.environ["BENCH_MAX_NEW"])

    knobs = {k: v for k, v in os.environ.items() if k.startswith(("LTX_ENHANCER_", "GEMMA4_", "BENCH_"))}
    results = {"tag": tag, "knobs": knobs, "max_new_tokens": enhancer.max_new_tokens, "lifecycle": {}, "requests": []}
    out_path = os.path.join(OUT_DIR, f"{tag}.json")

    def dump():
        with open(out_path, "w") as f:
            json.dump(results, f, indent=1)

    t0 = time.time()
    enhancer.ensure_loaded()
    results["lifecycle"]["load_s"] = round(time.time() - t0, 1)
    t0 = time.time()
    enhancer.warmup(capture=False)
    results["lifecycle"]["warmup_s"] = round(time.time() - t0, 1)
    t0 = time.time()
    enhancer.capture_trace()
    results["lifecycle"]["capture_s"] = round(time.time() - t0, 1)
    dump()
    # Only the measured requests run past stop ids; warmup and capture above keep their natural length.
    enhancer.ignore_stop = _env_flag("BENCH_IGNORE_STOP")

    counts: dict = {}
    restores: list = []
    generator = enhancer._generator
    model0 = generator.model[0] if isinstance(getattr(generator, "model", None), (list, tuple)) else None
    if model0 is not None:
        for name in MODEL_COUNTERS:
            _counting(model0, name, counts)
        sampling_module = getattr(model0, "sampling", None)
        _counting(sampling_module, "_run_sampling", counts)
        _counting(getattr(sampling_module, "seed_manager", None), "write_device_seed_values", counts)
        results["device_pli"] = bool(getattr(model0, "_device_pli", False))
        results["decode_token_feedback"] = bool(getattr(model0, "_tt_supports_decode_token_feedback", False))
    for name in ("decode_forward", "prefill_forward_text"):
        _counting(generator, name, counts)
    # Module attributes: every caller looks these up on ttnn at call time. Restored after the requests.
    for name in ("copy_host_to_device_tensor", "execute_trace"):
        _counting(ttnn, name, counts, key=f"ttnn.{name}", restores=restores)

    ref = None
    ref_path = os.environ.get("BENCH_REF") or (
        os.path.join(OUT_DIR, f"{os.environ['BENCH_REF_TAG']}.json") if os.environ.get("BENCH_REF_TAG") else None
    )
    if ref_path:
        if os.path.isfile(ref_path):
            with open(ref_path) as f:
                ref = {(r["prompt_idx"], r["temperature"], r["repeat"]): r for r in json.load(f)["requests"]}
            results["ref"] = ref_path
        else:
            logger.warning(f"bench: reference {ref_path} not found; no ref comparison")

    try:
        for temperature in temps:
            enhancer.temperature = temperature
            for repeat in range(repeats):
                for i, prompt in enumerate(BENCH_PROMPTS):
                    before = dict(counts)
                    t0 = time.time()
                    text = enhancer.enhance(prompt, mode="t2v", seed=ENHANCER_SEED + i)
                    wall = time.time() - t0
                    stats = dict(enhancer.last_stats or {})
                    ids = list(getattr(enhancer, "last_token_ids", []) or [])
                    steps = stats.get("decode_steps")
                    rec = {
                        "prompt_idx": i,
                        "temperature": temperature,
                        "repeat": repeat,
                        "wall_s": round(wall, 3),
                        "ms_per_token": round(1000 * stats["decode_s"] / steps, 2) if steps else None,
                        "calls": {k: counts[k] - before.get(k, 0) for k in counts},
                        **stats,
                        "token_ids": ids,
                        "text": text,
                    }
                    if row_check:
                        rows = _row_tokens(generator, tuple(submesh.shape))
                        rec["row_tokens"] = rows
                        rec["rows_equal"] = None if rows is None else len(set(rows)) == 1
                    if repeat > 0:
                        first = next(
                            r
                            for r in results["requests"]
                            if (r["prompt_idx"], r["temperature"], r["repeat"]) == (i, temperature, 0)
                        )
                        rec["self_equal"] = _first_divergence(ids, first["token_ids"]) is None
                    if ref is not None:
                        other = ref.get((i, temperature, repeat))
                        if other is not None and other.get("token_ids"):
                            div = _first_divergence(ids, other["token_ids"])
                            rec["ref_equal"] = div is None
                            rec["ref_first_divergence"] = div
                    results["requests"].append(rec)
                    logger.info(
                        f"bench t={temperature} r={repeat} p={i}: {stats.get('prompt_tokens')}+"
                        f"{stats.get('new_tokens')} tok, prefill {stats.get('prefill_s')}s decode "
                        f"{stats.get('decode_s')}s ({rec['ms_per_token']} ms/tok) wall {wall:.2f}s "
                        f"calls={rec['calls']} self_equal={rec.get('self_equal')} "
                        f"ref_equal={rec.get('ref_equal')} rows_equal={rec.get('rows_equal')}"
                    )
                    dump()
    finally:
        for obj, name, orig in restores:
            setattr(obj, name, orig)

    # Steady state: every repeat after the first.
    for temperature in temps:
        warm = [r for r in results["requests"] if r["temperature"] == temperature and r["repeat"] > 0]
        if not warm:
            continue
        summary = {
            "mean_wall_s": _mean(r["wall_s"] for r in warm),
            "mean_new_tokens": _mean(r["new_tokens"] for r in warm),
            "mean_decode_steps": _mean(r.get("decode_steps") for r in warm),
            "mean_ms_per_token": _mean(r["ms_per_token"] for r in warm),
            "mean_prefill_s": _mean(r["prefill_s"] for r in warm),
            "self_equal": sum(bool(r.get("self_equal")) for r in warm),
            "ref_equal": None if ref is None else sum(bool(r.get("ref_equal")) for r in warm),
            "n": len(warm),
            "calls_per_request": {k: _mean(r["calls"].get(k, 0) for r in warm) for k in counts},
            "calls_per_step": {
                k: _mean(r["calls"].get(k, 0) / r["decode_steps"] for r in warm if r.get("decode_steps"))
                for k in counts
            },
        }
        if any("dispatch_ms_median" in r for r in warm):
            summary["mean_dispatch_ms"] = _mean(r.get("dispatch_ms_median") for r in warm)
            summary["mean_read_ms"] = _mean(r.get("read_ms_median") for r in warm)
        if row_check:
            # Over every repeat: a cold request's rows diverge as readily as a warm one's.
            every = [r for r in results["requests"] if r["temperature"] == temperature]
            summary["rows_equal"] = sum(bool(r.get("rows_equal")) for r in every)
            summary["rows_checked"] = sum(r.get("rows_equal") is not None for r in every)
        results.setdefault("summary", {})[str(temperature)] = summary
    dump()
    logger.info(f"bench summary {tag}: {json.dumps(results.get('summary'))}")
    enhancer.unload()
