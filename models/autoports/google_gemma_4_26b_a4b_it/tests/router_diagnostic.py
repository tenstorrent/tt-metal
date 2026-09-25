# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only controls for BF16 router probability storage and checkpoint statistics."""

import argparse
import hashlib
import json
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


def cached_file(filename):
    return hf_hub_download(MODEL, filename, revision=REVISION, local_files_only=True)


def synthetic_layer(config, layer_idx):
    """Preserve the original run_decoder.py state order and RNG consumption."""
    with torch.device("meta"):
        layer = Gemma4TextDecoderLayer(config, layer_idx)
    state = {}
    for name, param in layer.state_dict().items():
        if "norm" in name or "scale" in name or "scalar" in name:
            state[name] = torch.ones(param.shape)
        else:
            state[name] = torch.randn(param.shape) * 0.02
    layer.load_state_dict(state, assign=True)
    return layer.eval()


def pcc(reference, actual):
    left = reference.detach().double().flatten()
    right = actual.detach().double().flatten()
    left = left - left.mean()
    right = right - right.mean()
    return float(torch.dot(left, right) / (left.norm() * right.norm()))


def set_comparison(reference, actual):
    matches = (reference[:, :, None] == actual[:, None, :]).any(dim=-1).sum(dim=-1)
    counts = torch.bincount(matches, minlength=reference.shape[-1] + 1)
    return {
        "tokens": reference.shape[0],
        "identical_sets": int((matches == reference.shape[-1]).sum()),
        "changed_sets": int((matches != reference.shape[-1]).sum()),
        "intersection_histogram": {str(i): int(count) for i, count in enumerate(counts) if count},
        "intersection_per_token": matches.tolist(),
        "selected_ids": actual.tolist(),
    }


@torch.no_grad()
def routing_experiment(layer_idx, length):
    torch.manual_seed(42)
    config = AutoConfig.from_pretrained(MODEL, revision=REVISION, local_files_only=True).text_config
    config._attn_implementation = "eager"
    layer = synthetic_layer(config, layer_idx)
    print("SYNTHETIC_LAYER_READY", flush=True)
    hidden = torch.randn(1, length, config.hidden_size).bfloat16().float()
    rope = Gemma4TextRotaryEmbedding(config)
    layer_type = config.layer_types[layer_idx]
    extent = max(32, (length + 128 + 31) // 32 * 32)
    cos, sin = rope(hidden, torch.arange(extent)[None], layer_type=layer_type)
    positions = torch.arange(length)
    allowed = positions[:, None] >= positions[None, :]
    if layer_type == "sliding_attention":
        allowed &= positions[:, None] - positions[None, :] < config.sliding_window
    mask = torch.zeros(length, length).masked_fill(~allowed, float("-inf"))[None, None]
    captured = {}

    def capture_residual(_module, inputs):
        captured["residual"] = inputs[0].detach().clone()

    def capture(name):
        def hook(_module, _inputs, output):
            captured[name] = output.detach().clone()

        return hook

    handles = [
        layer.router.register_forward_pre_hook(capture_residual),
        layer.router.proj.register_forward_hook(capture("logits")),
        layer.experts.register_forward_hook(capture("experts_raw")),
        layer.post_feedforward_layernorm_1.register_forward_hook(capture("shared_post_norm")),
        layer.post_feedforward_layernorm_2.register_forward_hook(capture("experts_post_norm")),
    ]
    reference = layer(
        hidden,
        position_embeddings=(cos[:, :length], sin[:, :length]),
        attention_mask=mask,
        past_key_values=DynamicCache(),
    )
    for handle in handles:
        handle.remove()
    print("HF_FORWARD_READY", flush=True)

    logits = captured["logits"]
    top_k = config.top_k_experts
    fp32_probs = logits.softmax(-1)
    rounded_probs = fp32_probs.bfloat16().float()
    logit_ids = logits.topk(top_k, dim=-1).indices
    fp32_ids = fp32_probs.topk(top_k, dim=-1).indices
    rounded_weights, rounded_ids = rounded_probs.topk(top_k, dim=-1)
    rounded_weights = rounded_weights / rounded_weights.sum(-1, keepdim=True)
    rounded_weights = rounded_weights * layer.router.per_expert_scale[rounded_ids]
    selected_logits = logits.gather(-1, logit_ids)
    selected_softmax = selected_logits.softmax(-1)
    expected_selected_weights = fp32_probs.gather(-1, logit_ids)
    expected_selected_weights /= expected_selected_weights.sum(-1, keepdim=True)

    ranked_probs, ranked_ids = fp32_probs.sort(dim=-1, descending=True)
    rounded_ranked = rounded_probs.gather(-1, ranked_ids)
    cutoff_ties = rounded_ranked[:, top_k - 1] == rounded_ranked[:, top_k]
    cutoff_tie_sizes = (rounded_probs == rounded_ranked[:, top_k - 1, None]).sum(-1)

    residual = captured["residual"]
    new_experts_raw = layer.experts(layer.pre_feedforward_layernorm_2(residual), rounded_ids, rounded_weights)
    new_experts_post_norm = layer.post_feedforward_layernorm_2(new_experts_raw.reshape(hidden.shape))
    new_output = residual.reshape(hidden.shape) + layer.post_feedforward_layernorm(
        captured["shared_post_norm"] + new_experts_post_norm
    )
    new_output *= layer.layer_scalar
    original_rebuilt = residual.reshape(hidden.shape) + layer.post_feedforward_layernorm(
        captured["shared_post_norm"] + captured["experts_post_norm"]
    )
    original_rebuilt *= layer.layer_scalar
    assert not config.hidden_size_per_layer_input
    assert torch.equal(original_rebuilt, reference), "Tail reconstruction must match the original HF forward exactly"

    result = {
        "model": MODEL,
        "revision": REVISION,
        "layer": layer_idx,
        "layer_type": layer_type,
        "length": length,
        "seed": 42,
        "cpu_threads": torch.get_num_threads(),
        "torch_version": torch.__version__,
        "synthetic_initialization": "state_dict order; norm/scale/scalar=ones; all other weights=randn*0.02",
        "input": "randn(1,length,hidden_size) after all synthetic state initialization; BF16 roundtrip to FP32",
        "residual_source": "router forward-pre-hook during the exact eager HF decoder forward",
        "hidden_size": config.hidden_size,
        "experts": config.num_experts,
        "top_k": top_k,
        "logits_std": float(logits.std(correction=0)),
        "probabilities_min": float(fp32_probs.min()),
        "probabilities_max": float(fp32_probs.max()),
        "probability_roundtrip_pcc": pcc(fp32_probs, rounded_probs),
        "reference_selected_ids": logit_ids.tolist(),
        "fp32_logits_vs_fp32_probability_topk": set_comparison(logit_ids, fp32_ids),
        "fp32_logits_vs_bf16_probability_topk": set_comparison(logit_ids, rounded_ids),
        "fp32_logits_vs_bf16_logit_topk": set_comparison(
            logit_ids, logits.bfloat16().float().topk(top_k, dim=-1).indices
        ),
        "fp32_logits_vs_bf16_logit_and_probability_topk": set_comparison(
            logit_ids, logits.bfloat16().float().softmax(-1).bfloat16().float().topk(top_k, dim=-1).indices
        ),
        "bf16_probability_rank_8_9_ties": int(cutoff_ties.sum()),
        "bf16_probability_cutoff_tie_size_per_token": cutoff_tie_sizes.tolist(),
        "fp32_probability_rank_8_9_gap_per_token": (ranked_probs[:, top_k - 1] - ranked_probs[:, top_k]).tolist(),
        "selected_logit_softmax_max_abs_error_vs_renormalized_fp32_probabilities": float(
            (selected_softmax - expected_selected_weights).abs().max()
        ),
        "probability_storage_only_output_control": {
            "description": "FP32 HF residual, logits, weights, expert computation and all norms held fixed; only full softmax probability storage roundtrips through BF16 before topk; CPU torch.topk tie-breaking",
            "original_tail_reconstruction_bitwise_equal": torch.equal(original_rebuilt, reference),
            "raw_experts_pcc": pcc(captured["experts_raw"], new_experts_raw),
            "post_norm_experts_pcc": pcc(captured["experts_post_norm"], new_experts_post_norm),
            "decoder_output_pcc": pcc(reference, new_output),
            "decoder_output_max_abs_error": float((reference - new_output).abs().max()),
            "decoder_output_mean_abs_error": float((reference - new_output).abs().mean()),
        },
        "limitations": "CPU topk tie-breaking does not establish the actual TT-selected expert sets; this isolates probability rounding, not TT projection/input drift.",
    }
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("logits_std", "bf16_probability_rank_8_9_ties", "probability_storage_only_output_control")
            }
        ),
        flush=True,
    )
    return result


@torch.no_grad()
def weight_statistics(layers):
    index = json.loads(Path(cached_file("model.safetensors.index.json")).read_text())
    prefixes = tuple(f"model.language_model.layers.{layer}." for layer in layers)
    relevant = {key: shard for key, shard in index["weight_map"].items() if key.startswith(prefixes)}
    records = []
    for shard in sorted(set(relevant.values())):
        with safe_open(cached_file(shard), framework="pt", device="cpu") as checkpoint:
            for key in sorted(key for key, value in relevant.items() if value == shard):
                tensor = checkpoint.get_tensor(key)
                source_dtype = str(tensor.dtype)
                fp32 = tensor.float()
                variance, mean = torch.var_mean(fp32, correction=0)
                records.append(
                    {
                        "name": key,
                        "shape": list(tensor.shape),
                        "source_dtype": source_dtype,
                        "numel": tensor.numel(),
                        "mean": float(mean),
                        "std": float(variance.sqrt()),
                        "min": float(fp32.min()),
                        "max": float(fp32.max()),
                    }
                )
                print(f"WEIGHT_STATS {key} mean={float(mean):.8g} std={float(variance.sqrt()):.8g}", flush=True)
                del fp32, tensor
    return {
        "model": MODEL,
        "revision": REVISION,
        "layers": layers,
        "cpu_threads": torch.get_num_threads(),
        "statistics": "All tensor elements; convert source tensor to FP32; torch.var_mean(correction=0), population std",
        "tensors": sorted(records, key=lambda item: item["name"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--length", type=int, default=32)
    parser.add_argument("--output", type=Path, default=ROOT / "doc/functional_decoder/router_diagnostic.json")
    parser.add_argument("--stats-output", type=Path, default=ROOT / "doc/functional_decoder/weight_stats.json")
    args = parser.parse_args()
    torch.set_num_threads(8)
    result = routing_experiment(args.layer, args.length)
    result["diagnostic_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    args.stats_output.write_text(json.dumps(weight_statistics([0, 5]), indent=2) + "\n")


if __name__ == "__main__":
    main()
