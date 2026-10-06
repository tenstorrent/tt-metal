# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only long-context cache/RoPE controls on exact device-harness inputs."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoConfig
from transformers.cache_utils import DynamicCache
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

from models.autoports.google_gemma_4_26b_a4b_it.tests.hf_precision_controls import (
    MODEL,
    REVISION,
    ROOT,
    THRESHOLD,
    BF16RoundedCache,
    digest,
    load_real_layer,
    pcc,
)

TARGET_POSITIONS = {4110, 4149}


def tensor_digest(value):
    return hashlib.sha256(value.contiguous().float().numpy().tobytes()).hexdigest()


def load_inputs(path, generated_x, generated_decode):
    fixture = torch.load(path, map_location="cpu", weights_only=True)
    x = fixture["x"] if "x" in fixture else fixture["prefill"]
    decode = fixture["decode_inputs"] if "decode_inputs" in fixture else fixture["decode"]
    if isinstance(decode, list):
        decode = torch.cat(decode, dim=1)
    x, decode = x.float(), decode.float()
    assert x.shape == generated_x.shape and decode.shape == generated_decode.shape
    assert torch.equal(x, generated_x), "Device prefill input differs from the CPU seed reconstruction"
    assert torch.equal(decode, generated_decode), "Device decode input stream differs from CPU seed reconstruction"
    return x, decode


@torch.no_grad()
def run(layer, x, decode_inputs, cos, sin, cache, label):
    length = x.shape[1]
    indices = torch.arange(length)
    allowed = indices[:, None] >= indices[None, :]
    if layer.self_attn.is_sliding:
        allowed &= indices[:, None] - indices[None, :] < layer.config.sliding_window
    mask = torch.zeros(length, length).masked_fill(~allowed, float("-inf"))[None, None]
    rng = torch.get_rng_state().clone()
    print(f"{label} PREFILL_START", flush=True)
    prefill = layer(
        x, position_embeddings=(cos[:, :length], sin[:, :length]), attention_mask=mask, past_key_values=cache
    )
    assert torch.equal(rng, torch.get_rng_state()), "HF prefill consumed RNG"
    stages = {}

    def capture(name):
        def hook(_module, _inputs, output):
            stages[name] = (output[0] if isinstance(output, tuple) else output).detach().clone()

        return hook

    def capture_router(_module, inputs, output):
        stages["router_input"] = inputs[0].detach().clone()
        stages["router_ids"] = output[2].detach().clone()

    names = {
        "self_attn",
        "post_attention_layernorm",
        "router.proj",
        "experts",
        "post_feedforward_layernorm_2",
        "post_feedforward_layernorm",
    }
    handles = [module.register_forward_hook(capture(name)) for name, module in layer.named_modules() if name in names]
    handles.append(layer.router.register_forward_hook(capture_router))
    outputs = []
    try:
        for step in range(decode_inputs.shape[1]):
            position = length + step
            decode_mask = torch.zeros(1, 1, 1, position + 1)
            if layer.self_attn.is_sliding:
                decode_mask[..., : max(0, position + 1 - layer.config.sliding_window)] = float("-inf")
            result = layer(
                decode_inputs[:, step : step + 1],
                position_embeddings=(cos[:, position : position + 1], sin[:, position : position + 1]),
                attention_mask=decode_mask,
                past_key_values=cache,
            )
            outputs.append({"output": result.detach().clone(), **stages})
            if (step + 1) % 32 == 0:
                print(f"{label} DECODE {step + 1}/{decode_inputs.shape[1]}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    assert torch.equal(rng, torch.get_rng_state()), "HF decode consumed RNG"
    return prefill, outputs


def compare(reference, actual, position):
    ids = reference["router_ids"].flatten()
    actual_ids = actual["router_ids"].flatten()
    intersection = int((ids[:, None] == actual_ids[None, :]).any(-1).sum())
    value = pcc(reference["output"], actual["output"])
    ref_logits = reference["router.proj"].flatten()
    actual_logits = actual["router.proj"].flatten()
    ranked = ref_logits.sort(descending=True).values
    ranked_actual = actual_logits.sort(descending=True).values
    probabilities = ranked.softmax(-1)
    row = {
        "position": position,
        "pcc": value,
        "passed": value >= THRESHOLD,
        "route_intersection": intersection,
        "fp32_experts": ids.tolist(),
        "control_experts": actual_ids.tolist(),
        "fp32_logit_rank8_rank9_gap": float(ranked[7] - ranked[8]),
        "control_logit_rank8_rank9_gap": float(ranked_actual[7] - ranked_actual[8]),
        "fp32_probability_rank8_rank9_gap": float(probabilities[7] - probabilities[8]),
        "maximum_logit_absolute_error": float((ref_logits - actual_logits).abs().max()),
    }
    if position in TARGET_POSITIONS or intersection != 8 or not row["passed"]:
        row["stage_pcc"] = {
            name: pcc(reference[name], actual[name]) for name in reference if name not in {"output", "router_ids"}
        }
        candidates = torch.unique(torch.cat((ref_logits.topk(9).indices, actual_logits.topk(9).indices))).tolist()
        row["candidate_logits"] = [
            {"expert": expert, "fp32": float(ref_logits[expert]), "control": float(actual_logits[expert])}
            for expert in candidates
        ]
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-fixture", type=Path, required=True)
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "doc/functional_decoder/long_decode_hf_precision_control.json"
    )
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(8)
    config = AutoConfig.from_pretrained(Path(__file__).parent).text_config
    config._attn_implementation = "eager"
    layer = load_real_layer(config, 0)
    generated_x = torch.randn(1, args.length, config.hidden_size).bfloat16().float()
    first_decode = torch.randn(1, 1, config.hidden_size).bfloat16().float()
    generated_decode = torch.cat(
        [first_decode] + [torch.randn_like(first_decode).bfloat16().float() for _ in range(1, args.steps)], dim=1
    )
    x, decode_inputs = load_inputs(args.input_fixture, generated_x, generated_decode)
    extent = (args.length + max(128, args.steps) + 1023) // 1024 * 1024
    cos, sin = Gemma4TextRotaryEmbedding(config)(x, torch.arange(extent)[None], layer_type="sliding_attention")
    print("DEVICE_INPUT_FIXTURE_MATCHES_CPU_RNG_RECONSTRUCTION", flush=True)
    report = {
        "model": MODEL,
        "revision": REVISION,
        "layer": 0,
        "seed": 42,
        "length": args.length,
        "steps": args.steps,
        "positions": [args.length, args.length + args.steps - 1],
        "sliding_window": config.sliding_window,
        "extent": extent,
        "cpu_threads": torch.get_num_threads(),
        "torch_version": torch.__version__,
        "pcc_threshold": THRESHOLD,
        "input_fixture": str(args.input_fixture),
        "input_fixture_sha256": digest(args.input_fixture),
        "device_inputs_match_cpu_seed_reconstruction": True,
        "x_sha256": tensor_digest(x),
        "decode_stream_sha256": tensor_digest(decode_inputs),
        "decode_inputs": [
            {"position": args.length + i, "sha256": tensor_digest(decode_inputs[:, i : i + 1])}
            for i in range(args.steps)
        ],
        "config_sha256": digest(Path(__file__).parent / "config.json"),
        "source_sha256": digest(Path(__file__)),
        "shared_helper_source_sha256": digest(Path(__file__).parent / "hf_precision_controls.py"),
        "run_decoder_source_sha256": digest(Path(__file__).parent / "run_decoder.py"),
        "command": "HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python_env/bin/python -m models.autoports.google_gemma_4_26b_a4b_it.tests.hf_long_decode_precision_controls "
        f"--input-fixture {args.input_fixture} --length {args.length} --steps {args.steps}",
        "reference": "Real weights and all HF eager computation FP32; same local config and exact saved input stream as device run",
        "limitations": "CPU numerical controls do not emulate TT kernels. Cache quantization also affects CPU prefill attention. No acceptance threshold changed.",
        "controls": {},
    }
    start = time.monotonic()
    reference_prefill, reference = run(layer, x, decode_inputs, cos, sin, DynamicCache(), "FP32_ORACLE")
    for control in ("cache", "rope-cache"):
        control_cos, control_sin = (
            (cos, sin) if control == "cache" else (cos.bfloat16().float(), sin.bfloat16().float())
        )
        prefill, outputs = run(layer, x, decode_inputs, control_cos, control_sin, BF16RoundedCache(), control)
        rows = [compare(ref, actual, args.length + i) for i, (ref, actual) in enumerate(zip(reference, outputs))]
        comparison = {
            "description": "FP32 HF plus BF16-roundtripped FP32 cached K/V"
            + (" and BF16-roundtripped FP32 RoPE cos/sin tables" if control == "rope-cache" else ""),
            "prefill_pcc": pcc(reference_prefill, prefill),
            "minimum_decode_pcc": min(row["pcc"] for row in rows),
            "failed_positions": [row["position"] for row in rows if not row["passed"]],
            "changed_route_positions": [row["position"] for row in rows if row["route_intersection"] != 8],
            "decode": rows,
        }
        report["controls"][control] = comparison
        report["elapsed_seconds"] = time.monotonic() - start
        report["ttnn_imported"] = "ttnn" in sys.modules
        assert not report["ttnn_imported"]
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"control": control, **{k: v for k, v in comparison.items() if k != "decode"}}), flush=True)
        for row in rows:
            if row["position"] in TARGET_POSITIONS:
                print("TARGET", control, json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
