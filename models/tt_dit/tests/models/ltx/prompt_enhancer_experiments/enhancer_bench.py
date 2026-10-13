# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Standalone bench of the device prompt enhancer (Gemma-4-E2B) on the Galaxy 4x8 ring, without LTX.

Same mesh params, fabric topology and enhancer class as the traced LTX distilled pipeline, but no DiT,
VAE or encoder: load, eager warmup, decode-trace capture, then a fixed prompt set rewritten at each
temperature. Each request records its timings, host staging counts and the generated token ids, so two
runs with different knobs can be compared token for token (BENCH_REF) as well as on latency.

    BENCH_TEMPS        comma list of temperatures (default "0.0,0.7")
    BENCH_REPEATS      passes over the prompt set per temperature (default 2; pass 0 is reported as cold)
    BENCH_MAX_NEW      token budget (default: the enhancer's own, 512)
    BENCH_TAG          output name (default "bench")
    BENCH_REF          path of an earlier bench JSON: report token equality per (prompt, temperature, repeat)
    LTX_ENHANCER_EXP_DIR  output root (JSON lands under <root>/enhancer_bench/)
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


def _counting(obj, name: str, counts: dict) -> None:
    """Wrap ``obj.name`` so every call bumps ``counts[name]``."""
    orig = getattr(obj, name, None)
    if orig is None:
        return
    counts.setdefault(name, 0)

    def wrapped(*args, **kwargs):
        counts[name] += 1
        return orig(*args, **kwargs)

    setattr(obj, name, wrapped)


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

    counts: dict = {}
    generator = enhancer._generator
    model0 = generator.model[0] if isinstance(getattr(generator, "model", None), (list, tuple)) else None
    if model0 is not None:
        for name in ("prepare_decode_inputs_host", "compute_host_pli", "prepare_inputs_prefill"):
            _counting(model0, name, counts)
        results["device_pli"] = bool(getattr(model0, "_device_pli", False))
        results["decode_token_feedback"] = bool(getattr(model0, "_tt_supports_decode_token_feedback", False))
    for name in ("decode_forward", "prefill_forward_text"):
        _counting(generator, name, counts)

    ref = None
    if os.environ.get("BENCH_REF"):
        with open(os.environ["BENCH_REF"]) as f:
            ref = {(r["prompt_idx"], r["temperature"], r["repeat"]): r for r in json.load(f)["requests"]}

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
                rec = {
                    "prompt_idx": i,
                    "temperature": temperature,
                    "repeat": repeat,
                    "wall_s": round(wall, 3),
                    "ms_per_token": round(1000 * stats["decode_s"] / max(1, stats["new_tokens"] - 1), 2)
                    if stats.get("new_tokens")
                    else None,
                    "calls": {k: counts[k] - before.get(k, 0) for k in counts},
                    **stats,
                    "token_ids": ids,
                    "text": text,
                }
                if ref is not None:
                    other = ref.get((i, temperature, repeat))
                    if other is not None and other.get("token_ids"):
                        a, b = ids, other["token_ids"]
                        div = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), None)
                        if div is None and len(a) != len(b):
                            div = min(len(a), len(b))
                        rec["ref_equal"] = div is None
                        rec["ref_first_divergence"] = div
                results["requests"].append(rec)
                logger.info(
                    f"bench t={temperature} r={repeat} p={i}: {stats.get('prompt_tokens')}+{stats.get('new_tokens')} tok, "
                    f"prefill {stats.get('prefill_s')}s decode {stats.get('decode_s')}s ({rec['ms_per_token']} ms/tok) "
                    f"wall {wall:.2f}s calls={rec['calls']} ref_equal={rec.get('ref_equal')}"
                )
                dump()

    # Steady state: every repeat after the first.
    for temperature in temps:
        warm = [r for r in results["requests"] if r["temperature"] == temperature and r["repeat"] > 0]
        if warm:
            results.setdefault("summary", {})[str(temperature)] = {
                "mean_wall_s": round(sum(r["wall_s"] for r in warm) / len(warm), 3),
                "mean_new_tokens": round(sum(r["new_tokens"] for r in warm) / len(warm), 1),
                "mean_ms_per_token": round(sum(r["ms_per_token"] for r in warm) / len(warm), 2),
                "mean_prefill_s": round(sum(r["prefill_s"] for r in warm) / len(warm), 3),
                "ref_equal": None if ref is None else sum(bool(r.get("ref_equal")) for r in warm),
                "n": len(warm),
            }
    dump()
    logger.info(f"bench summary {tag}: {json.dumps(results.get('summary'))}")
    enhancer.unload()
