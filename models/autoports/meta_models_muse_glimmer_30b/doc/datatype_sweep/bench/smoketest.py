# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reduced-build smoketest for one precision artifact.

``$datatype-sweep``: *"sometimes different datatypes require small semantic
changes to the code. KV cache is a common example ... So when changing datatypes
first run a quick one-decoder smoketest to check it works correctly and get that
right before using it or rejecting it in a full model pareto sweep."*

This is that smoketest.  It builds a **two-layer** model -- one sliding and one
full-attention layer, with their real weights and the real terminal path -- from
a candidate artifact, then:

* proves the requested policy is the built one (:func:`check_propagation`);
* prefills a non-tile-aligned prompt, so ``paged_fill_cache`` runs at the
  candidate's cache dtype;
* takes traced decode steps, so ``paged_update_cache`` and the decode SDPA run
  against that cache;
* reports the greedy tokens and the PCC of the candidate's logits against the
  baseline artifact's, on the same prompt.

It is **not** accuracy evidence: two layers is not the model.  It is the "does
this dtype combination run correctly at all" gate that keeps a broken op
contract out of the 52-layer sweep.

Usage::

    python doc/datatype_sweep/bench/smoketest.py --configs c05-kv4,c08-attn4-kv4-cclbfp8
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[3]
REPO = ROOT.parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

import ttnn  # noqa: E402
from models.autoports.meta_models_muse_glimmer_30b.tt import precision_config as pc  # noqa: E402
from models.autoports.meta_models_muse_glimmer_30b.tt.generator import (  # noqa: E402
    DEFAULT_TRACE_REGION_SIZE,
    GREEDY,
    build_generator,
    clear_generator_cache,
)
from models.autoports.meta_models_muse_glimmer_30b.tt.multichip_decoder import (  # noqa: E402
    close_multichip_mesh,
    open_multichip_mesh,
)
from models.autoports.meta_models_muse_glimmer_30b.tt.optimized_decoder import (  # noqa: E402
    LAYER_KIND_FULL,
    LAYER_KIND_SLIDING,
    resolve_layer_kind,
)

CONFIG_DIR = ROOT / "doc/datatype_sweep/configs"
OUT = ROOT / "doc/datatype_sweep"

#: A prompt length that is not a multiple of the 32-row tile, the 64-token page
#: or the 8192-token prefill chunk, so the pad/slice path runs here too.  The
#: tokens are the **real** AIME24 chat prompt's first ``PROMPT_LEN``, not random
#: ids: on random ids the two-layer stack's logits are near-degenerate and every
#: perturbation lands in the same attractor, which makes the token comparison
#: say nothing.
PROMPT_LEN = 200
DECODE_STEPS = 6
REFERENCE = ROOT / "readiness_aime24_chat.refpt"


def say(*args) -> None:
    print(*args, flush=True)


def reduced_layer_indices(model_id: str) -> list[int]:
    """One sliding and one full-attention layer, by their real indices."""
    from transformers import AutoConfig

    from models.autoports.meta_models_muse_glimmer_30b.tt.functional_decoder import _text_config
    from models.autoports.meta_models_muse_glimmer_30b.tt.generator import weights_snapshot_dir

    hf_config = AutoConfig.from_pretrained(str(weights_snapshot_dir(model_id)), local_files_only=True)
    text = _text_config(hf_config)
    picked: dict[str, int] = {}
    for idx in range(text.num_hidden_layers):
        kind = resolve_layer_kind(hf_config, idx)
        picked.setdefault(kind, idx)
    return sorted(picked[kind] for kind in (LAYER_KIND_SLIDING, LAYER_KIND_FULL) if kind in picked)


def run_one(mesh, config_path: pathlib.Path, prompt: list[int], layer_indices: list[int]) -> dict:
    config = pc.load_precision_config(config_path)
    result: dict = {"config_id": config["config_id"], "config_path": str(config_path.relative_to(REPO))}
    started = time.perf_counter()
    generator = build_generator(ROOT, mesh, layer_indices=layer_indices, precision_config=config_path, reuse=False)
    result["build_seconds"] = round(time.perf_counter() - started, 1)
    try:
        report = generator.capability_report()
        realised = report["precision_policy"]
        result["precision_config_id_reported"] = realised["selected_config_id"]
        result["propagation_problems"] = pc.check_propagation(config, realised)
        result["realised"] = realised

        # Prefill at a non-aligned length -> paged_fill_cache at the cache dtype.
        generator.reset()
        logits = generator.prefill_forward(torch.tensor([prompt], dtype=torch.int32), prompt_lens=[len(prompt)])
        logits = logits.reshape(-1)[: generator.model.config.vocab_size].float()
        result["prefill_logits_finite"] = bool(torch.isfinite(logits).all())
        result["prefill_top1"] = int(torch.argmax(logits))
        result["prefill_logits"] = logits

        # Traced decode -> paged_update_cache + paged SDPA at the cache dtype.
        generator.reset()
        tokens = generator.generate(
            prompt_token_ids=prompt, max_new_tokens=DECODE_STEPS, sampling_params=GREEDY, enable_trace=True
        )
        result["decode_tokens"] = [int(t) for t in tokens]
        result["decode_trace_replays"] = int(generator.counters.get("trace_replays", 0))
    finally:
        generator.teardown()
        clear_generator_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default="", help="comma-separated config ids; default every candidate")
    parser.add_argument("--baseline", default="c00-baseline-attn8-mlp4-kv8-lofi")
    parser.add_argument("--out", default="smoketest.json")
    args = parser.parse_args()

    from models.autoports.meta_models_muse_glimmer_30b.tt.generator import HF_MODEL_ID

    ids = [c.strip() for c in args.configs.split(",") if c.strip()] or sorted(p.stem for p in CONFIG_DIR.glob("*.json"))
    if args.baseline not in ids:
        ids = [args.baseline] + ids

    layer_indices = reduced_layer_indices(HF_MODEL_ID)
    say(f"SMOKE reduced build layers={layer_indices}")
    from models.common.readiness_check.schema import load_reference

    reference = load_reference(REFERENCE)
    prompt = [int(t) for t in torch.as_tensor(reference.entries[0].prompt_tokens).reshape(-1)][:PROMPT_LEN]
    say(f"SMOKE prompt = first {len(prompt)} tokens of {REFERENCE.name} entry[0]")

    mesh = open_multichip_mesh(trace_region_size=DEFAULT_TRACE_REGION_SIZE)
    summary: dict = {"layer_indices": layer_indices, "prompt_len": PROMPT_LEN, "results": {}}
    baseline_logits = None
    baseline_tokens = None
    try:
        for config_id in ids:
            path = CONFIG_DIR / f"{config_id}.json"
            say(f"SMOKE ===== {config_id} =====")
            try:
                result = run_one(mesh, path, prompt, layer_indices)
            except Exception as exc:  # a candidate that cannot build is a recorded blocker
                say(f"SMOKE {config_id} FAILED: {type(exc).__name__}: {exc}")
                summary["results"][config_id] = {"error": f"{type(exc).__name__}: {exc}"}
                continue
            logits = result.pop("prefill_logits")
            if config_id == args.baseline:
                baseline_logits = logits
            if baseline_logits is not None:
                a, b = logits - logits.mean(), baseline_logits - baseline_logits.mean()
                result["prefill_pcc_vs_baseline"] = float((a @ b) / (a.norm() * b.norm() + 1e-12))
            if config_id == args.baseline:
                baseline_tokens = result["decode_tokens"]
            result["decode_tokens_match_baseline"] = baseline_tokens == result["decode_tokens"]
            summary["results"][config_id] = result
            say(
                f"SMOKE {config_id} propagation_problems={len(result['propagation_problems'])} "
                f"pcc={result.get('prefill_pcc_vs_baseline')} tokens={result['decode_tokens']}"
            )
            for problem in result["propagation_problems"]:
                say(f"SMOKE   propagation: {problem}")
    finally:
        path = OUT / args.out
        path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
        say(f"SMOKE summary -> {path}")
        close_multichip_mesh(mesh)
    say("SMOKE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
