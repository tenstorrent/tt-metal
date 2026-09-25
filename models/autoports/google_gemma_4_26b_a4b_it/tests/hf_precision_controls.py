# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only FP32/BF16 controls matching the batch-32 decoder harness inputs."""

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoConfig
from transformers.cache_utils import DynamicCache
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer, Gemma4TextRotaryEmbedding

MODEL = "google/gemma-4-26B-A4B-it"
REVISION = "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"
ROOT = Path(__file__).resolve().parents[1]
BATCH, LENGTH, EXTENT = 32, 33, 128
THRESHOLD = 0.995
CASES = [(0, "sliding", [5, 15, 17, 25, 27]), (5, "full", [9, 11, 21, 29])]


class BF16RoundedCache(DynamicCache):
    """Keep DynamicCache semantics while rounding every newly stored K/V value."""

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        return super().update(
            key_states.bfloat16().float(), value_states.bfloat16().float(), layer_idx, *args, **kwargs
        )


def cached_file(name):
    return hf_hub_download(MODEL, name, revision=REVISION, local_files_only=True)


def load_real_layer(config, layer_idx):
    """Match run_decoder.load_layer(real=True), without its TTNN import."""
    with torch.device("meta"):
        layer = Gemma4TextDecoderLayer(config, layer_idx)
    index = json.loads(Path(cached_file("model.safetensors.index.json")).read_text())
    prefix = f"model.language_model.layers.{layer_idx}."
    state = {}
    for shard in sorted({value for key, value in index["weight_map"].items() if key.startswith(prefix)}):
        with safe_open(cached_file(shard), framework="pt", device="cpu") as source:
            for key in source.keys():
                if key.startswith(prefix):
                    state[key[len(prefix) :]] = source.get_tensor(key).float()
    layer.load_state_dict(state, assign=True)
    return layer.eval()


def pcc(left, right):
    left, right = left.double().flatten(), right.double().flatten()
    left, right = left - left.mean(), right - right.mean()
    return float(torch.dot(left, right) / (left.norm() * right.norm()))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.no_grad()
def run(layer, x, dx, cos, sin, cache):
    phase = ["prefill"]
    routes, stages = {}, {}

    def capture_stage(name):
        def hook(_module, _inputs, output):
            if phase[0] == "decode":
                value = output[0] if isinstance(output, tuple) else output
                stages[name] = value.detach().float().clone()

        return hook

    def capture_router(_module, inputs, output):
        routes[phase[0]] = output[2].detach().clone()
        if phase[0] == "decode":
            stages["router_input"] = inputs[0].detach().float().clone()

    names = {
        "input_layernorm",
        "self_attn",
        "post_attention_layernorm",
        "router.proj",
        "mlp",
        "experts",
        "post_feedforward_layernorm_1",
        "post_feedforward_layernorm_2",
        "post_feedforward_layernorm",
    }
    handles = [
        module.register_forward_hook(capture_stage(name)) for name, module in layer.named_modules() if name in names
    ]
    handles.append(layer.router.register_forward_hook(capture_router))
    mask = torch.zeros(LENGTH, LENGTH).masked_fill(
        torch.triu(torch.ones(LENGTH, LENGTH, dtype=torch.bool), 1), float("-inf")
    )[None, None]
    try:
        prefill = layer(
            x,
            position_embeddings=(cos[:, :LENGTH], sin[:, :LENGTH]),
            attention_mask=mask,
            past_key_values=cache,
        ).float()
        phase[0] = "decode"
        decode = layer(
            dx,
            position_embeddings=(cos[:, LENGTH : LENGTH + 1], sin[:, LENGTH : LENGTH + 1]),
            attention_mask=torch.zeros(BATCH, 1, 1, LENGTH + 1),
            past_key_values=cache,
        ).float()
    finally:
        for handle in handles:
            handle.remove()
    metadata = {
        "sequence_length": int(cache.get_seq_length(layer.layer_idx)),
        "storage_dtype": str(cache.layers[layer.layer_idx].keys.dtype),
    }
    return prefill, decode, routes, stages, metadata


def compare_layer(control, layer_idx, attention, targeted):
    # Preserve batched.py's RNG order: seed, config, meta layer, real weights, x, dx.
    torch.manual_seed(42)
    config = AutoConfig.from_pretrained(MODEL, revision=REVISION, local_files_only=True).text_config
    config._attn_implementation = "eager"
    layer = load_real_layer(config, layer_idx)
    x = torch.randn(BATCH, LENGTH, config.hidden_size).bfloat16().float()
    dx = torch.randn(BATCH, 1, config.hidden_size).bfloat16().float()
    cos, sin = Gemma4TextRotaryEmbedding(config)(
        x, torch.arange(EXTENT)[None], layer_type=config.layer_types[layer_idx]
    )
    ref_pre, ref_dec, ref_routes, ref_stages, ref_cache = run(layer, x, dx, cos, sin, DynamicCache())
    if control == "bf16":
        layer = layer.to(torch.bfloat16)
        actual = run(layer, x.bfloat16(), dx.bfloat16(), cos.bfloat16(), sin.bfloat16(), DynamicCache())
    elif control == "cache":
        actual = run(layer, x, dx, cos, sin, BF16RoundedCache())
    else:
        cache_only = run(layer, x, dx, cos, sin, BF16RoundedCache())
        actual = run(layer, x, dx, cos.bfloat16().float(), sin.bfloat16().float(), BF16RoundedCache())
    alt_pre, alt_dec, alt_routes, alt_stages, alt_cache = actual

    tt_path = ROOT / f"doc/functional_decoder/batch32_{attention}_final.json"
    tt = json.loads(tt_path.read_text())
    rows = []
    for slot in range(BATCH):
        baseline_ids, actual_ids = ref_routes["decode"][slot], alt_routes["decode"][slot]
        intersection = int((baseline_ids[:, None] == actual_ids[None, :]).any(-1).sum())
        value = pcc(ref_dec[slot], alt_dec[slot])
        ranked = ref_stages["router.proj"][slot].sort(descending=True).values
        row = {
            "slot": slot,
            "pcc": value,
            "passed": value >= THRESHOLD,
            "route_intersection": intersection,
            "fp32_experts": baseline_ids.tolist(),
            {
                "bf16": "bf16_experts",
                "cache": "bf16_cache_experts",
                "rope-cache": "bf16_rope_cache_experts",
            }[control]: actual_ids.tolist(),
            "fp32_logit_rank8_rank9_gap": float(ranked[7] - ranked[8]),
            "recorded_tt_pcc": tt["decode"][slot]["pcc"],
            "targeted_tt_failure": slot in targeted,
        }
        if not row["passed"] or row["targeted_tt_failure"] or intersection != 8:
            row["stage_pcc"] = {name: pcc(ref_stages[name][slot], alt_stages[name][slot]) for name in ref_stages}
        if control == "rope-cache":
            cache_ids = cache_only[2]["decode"][slot]
            row["cache_only_pcc"] = pcc(ref_dec[slot], cache_only[1][slot])
            row["cache_only_route_intersection"] = int((baseline_ids[:, None] == cache_ids[None, :]).any(-1).sum())
            row["cache_only_experts"] = cache_ids.tolist()
            row["rope_cache_vs_cache_only_pcc"] = pcc(cache_only[1][slot], alt_dec[slot])
        rows.append(row)
    prefill_pcc = [pcc(ref_pre[slot], alt_pre[slot]) for slot in range(BATCH)]
    pre_matches = (ref_routes["prefill"][:, :, None] == alt_routes["prefill"][:, None, :]).any(-1).sum(-1)
    failed = {row["slot"] for row in rows if not row["passed"]}
    return {
        "layer": layer_idx,
        "layer_type": config.layer_types[layer_idx],
        "x_sha256": hashlib.sha256(x.numpy().tobytes()).hexdigest(),
        "dx_sha256": hashlib.sha256(dx.numpy().tobytes()).hexdigest(),
        "oracle_cache": ref_cache,
        "control_cache": alt_cache,
        "prefill_pcc_per_slot": prefill_pcc,
        "prefill_failed_slots": [i for i, value in enumerate(prefill_pcc) if value < THRESHOLD],
        "prefill_changed_route_tokens": int((pre_matches != 8).sum()),
        "prefill_tokens": BATCH * LENGTH,
        "decode_failed_slots": sorted(failed),
        "decode_changed_route_slots": [row["slot"] for row in rows if row["route_intersection"] != 8],
        "recorded_tt_artifact": str(tt_path.relative_to(ROOT)),
        "recorded_tt_artifact_sha256": digest(tt_path),
        "summary": {
            "tt_failing_slots_also_failing_control": sorted(set(targeted) & failed),
            "tt_failing_slots_passing_control": sorted(set(targeted) - failed),
            "control_failing_slots_passing_recorded_tt": sorted(failed - set(targeted)),
            "minimum_decode_pcc": min(row["pcc"] for row in rows),
        },
        "decode": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", choices=["bf16", "cache", "rope-cache", "all"], default="all")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "doc/functional_decoder")
    args = parser.parse_args()
    torch.set_num_threads(8)
    controls = ["bf16", "cache", "rope-cache"] if args.control == "all" else [args.control]
    for control in controls:
        start = time.monotonic()
        result = {
            "model": MODEL,
            "revision": REVISION,
            "seed": 42,
            "batch": BATCH,
            "prefill_length": LENGTH,
            "decode_position": LENGTH,
            "extent": EXTENT,
            "cpu_threads": torch.get_num_threads(),
            "torch_version": torch.__version__,
            "pcc_threshold": THRESHOLD,
            "harness": "Exact batched.py RNG order and real checkpoint loading; x/dx BF16-roundtripped FP32",
            "control": {
                "bf16": "HF eager FP32 versus HF eager BF16 weights, activations, cache and RoPE; built-in FP32 RMSNorm/attention-softmax reductions retained; masks FP32",
                "cache": "FP32 HF eager throughout; only DynamicCache.update K/V inputs rounded through BF16 before FP32 storage and return; applies to prefill and decode",
                "rope-cache": "FP32 HF eager throughout; round RoPE cos/sin through BF16 to FP32 and round cached K/V through BF16 to FP32 in prefill/decode; compare against FP32 oracle and separately measured cache-only control",
            }[control],
            "route_comparison": "Unordered top-8 expert sets; CPU torch.topk tie-breaking",
            "limitations": "CPU numerical controls do not reproduce TT kernels or establish unavoidable precision limits. Cache control prefill attends to rounded K/V. Threshold and implementation are unchanged.",
            "command": f"HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python_env/bin/python -m models.autoports.google_gemma_4_26b_a4b_it.tests.hf_precision_controls --control {control}",
            "control_source_sha256": digest(Path(__file__)),
            "batched_harness_sha256": digest(ROOT / "tests/batched.py"),
            "layers": [],
        }
        for case in CASES:
            comparison = compare_layer(control, *case)
            result["layers"].append(comparison)
            print(json.dumps({"control": control, "layer": case[0], **comparison["summary"]}), flush=True)
            gc.collect()
        result["elapsed_seconds"] = time.monotonic() - start
        result["ttnn_imported"] = "ttnn" in sys.modules
        assert not result["ttnn_imported"], "This control must remain CPU-only"
        filename = {
            "bf16": "batch32_hf_precision_control.json",
            "cache": "batch32_hf_cache_precision_control.json",
            "rope-cache": "batch32_hf_rope_cache_precision_control.json",
        }[control]
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / filename).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
