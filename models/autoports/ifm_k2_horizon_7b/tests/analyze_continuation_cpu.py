"""CPU-only fixed-input HF layer discriminator for saved TT continuation traces.

Loads one pinned HF decoder layer, without importing TTNN or opening a device.
Complete row metrics and intermediate tensors remain in the ignored raw directory.
"""

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.masking_utils import create_causal_mask

from .hf_qualitative import MODEL, REVISION

RAW = Path("bringup/artifacts/ifm_k2_full_model_raw_20260928")
DOC = Path("models/autoports/ifm_k2_horizon_7b/doc/full_model")


def fingerprint(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def row_metrics(reference, candidate):
    a = reference.float().reshape(-1, reference.shape[-1])
    b = candidate.float().reshape_as(a)
    ac, bc = a - a.mean(-1, keepdim=True), b - b.mean(-1, keepdim=True)
    return {
        "pcc": (ac * bc).sum(-1) / (ac.norm(dim=-1) * bc.norm(dim=-1)).clamp_min(1e-30),
        "relative_l2": (a - b).norm(dim=-1) / a.norm(dim=-1).clamp_min(1e-30),
        "reference_max": a.abs().amax(-1),
        "candidate_max": b.abs().amax(-1),
    }


def summary(reference, candidate, raw_metrics, key):
    rows = row_metrics(reference, candidate)
    raw_metrics[key] = rows
    a, b = reference.float().flatten(), candidate.float().flatten()
    bad = torch.where(rows["pcc"] < 0.995)[0].tolist()
    worst = torch.argsort(rows["pcc"])[:8].tolist()

    def row(index):
        return {"position": index, **{name: float(value[index]) for name, value in rows.items()}}

    return {
        "all_pcc": float(torch.corrcoef(torch.stack([a, b]))[0, 1]),
        "all_relative_l2": float((a - b).norm() / a.norm()),
        "finite": bool(torch.isfinite(candidate).all()),
        "last": row(len(rows["pcc"]) - 1),
        "row_pcc_min": float(rows["pcc"].min()),
        "row_pcc_median": float(rows["pcc"].median()),
        "rows_below_0_995": len(bad),
        "first_positions_below_0_995": bad[:16],
        "worst_rows": [row(index) for index in worst],
    }


def load_layer(index, attention_backend):
    config = AutoConfig.from_pretrained(MODEL, revision=REVISION, trust_remote_code=True, local_files_only=True)
    cls = get_class_from_dynamic_module(
        config.auto_map["AutoModelForCausalLM"],
        MODEL,
        revision=REVISION,
        code_revision=REVISION,
        local_files_only=True,
    )
    module = importlib.import_module(cls.__module__)
    config._attn_implementation = attention_backend
    with torch.device("meta"):
        layer = module.K2HorizonDecoderLayer(config, index)
    snapshot = Path(snapshot_download(MODEL, revision=REVISION, local_files_only=True))
    weight_map = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"model.layers.{index}."
    selected = {name: shard for name, shard in weight_map.items() if name.startswith(prefix)}
    state = {}
    for shard in sorted(set(selected.values())):
        with safe_open(snapshot / shard, framework="pt", device="cpu") as handle:
            for name, filename in selected.items():
                if filename == shard:
                    state[name[len(prefix) :]] = handle.get_tensor(name)
    layer.load_state_dict(state, strict=True, assign=True)
    layer.eval()
    return config, layer, module.K2HorizonRotaryEmbedding(config).eval()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tt-artifact", type=Path, default=RAW / "continuation_l3_s31.pt")
    parser.add_argument("--hf-artifact", type=Path, default=RAW / "hf_continuation_l3.pt")
    parser.add_argument("--layer-index", "--layer", dest="layer", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--attention-backend", choices=["sdpa", "eager"], default="sdpa")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is None:
        suffix = "" if args.layer == 2 else f"_l{args.layer}"
        args.output = DOC / f"continuation_fixed_input_hf{suffix}.json"
    if not 1 <= args.threads <= 4 or args.layer < 1:
        parser.error("Use 1..4 CPU threads and a layer with a saved predecessor")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    tt = torch.load(args.tt_artifact, map_location="cpu", weights_only=True)
    hf = torch.load(args.hf_artifact, map_location="cpu", weights_only=True)
    previous = args.layer - 1
    inputs = {
        "hf": hf["layer_outputs"][previous],
        "tt_whole": tt["full"][previous].squeeze(0),
        "tt_split": torch.cat([tt["prefix"][previous], tt["split"][previous]], dim=2).squeeze(0),
    }
    expected = {
        "hf": hf["layer_outputs"][args.layer],
        "tt_whole": tt["full"][args.layer].squeeze(0),
        "tt_split": torch.cat([tt["prefix"][args.layer], tt["split"][args.layer]], dim=2).squeeze(0),
    }
    assert all(value.shape == inputs["hf"].shape for value in [*inputs.values(), *expected.values()])
    config, layer, rotary = load_layer(args.layer, args.attention_backend)
    print(f"Loaded CPU HF layer {args.layer}; dtype={next(layer.parameters()).dtype}", flush=True)
    positions = torch.arange(inputs["hf"].shape[1]).unsqueeze(0)
    raw_metrics, outputs, intermediates = {}, {}, {}
    for name, hidden in inputs.items():
        hidden = hidden.to(next(layer.parameters()).dtype)
        mask = create_causal_mask(
            config=config,
            inputs_embeds=hidden,
            attention_mask=torch.ones(hidden.shape[:2], dtype=torch.long),
            past_key_values=None,
            position_ids=positions,
        )
        outputs[name] = layer(
            hidden,
            position_embeddings=rotary(hidden, positions),
            attention_mask=mask,
            position_ids=positions,
            cache_position=positions[0],
            use_cache=False,
        )
        normed = layer.input_layernorm(hidden)
        intermediates[name] = {"group_normalized": normed}
        for role in ("q", "k", "v"):
            intermediates[name][f"{role}_projection"] = getattr(layer.self_attn, f"{role}_proj")(normed)
        print(f"Completed {name}", flush=True)
    result = {
        "model": MODEL,
        "revision": REVISION,
        "device": "cpu",
        "threads": args.threads,
        "layer_index": args.layer,
        "attention_backend": args.attention_backend,
        "weight_dtype": str(next(layer.parameters()).dtype),
        "split_position": tt["prefix"][previous].shape[2],
        "bos_positions": torch.where(hf["prompt_tokens"][0] == config.bos_token_id)[0].tolist(),
        "artifacts": {str(path): fingerprint(path) for path in (args.tt_artifact, args.hf_artifact)},
        "reference_reproduction": summary(expected["hf"], outputs["hf"], raw_metrics, "reference_reproduction"),
        "variants": {},
    }
    for name in ("tt_whole", "tt_split"):
        metrics = {}
        for key, a, b in (
            ("input_residual_vs_hf", inputs["hf"], inputs[name]),
            ("hf_on_tt_input_vs_original_hf", expected["hf"], outputs[name]),
            ("tt_output_vs_hf_on_same_input", outputs[name], expected[name]),
            ("tt_output_vs_original_hf", expected["hf"], expected[name]),
        ):
            metrics[key] = summary(a, b, raw_metrics, f"{name}/{key}")
        metrics["hf_math_input_sensitivity"] = {
            stage: summary(value, intermediates[name][stage], raw_metrics, f"{name}/{stage}")
            for stage, value in intermediates["hf"].items()
        }
        metrics["bos_rows"] = [
            {
                "position": position,
                **{
                    label: {
                        metric: float(raw_metrics[f"{name}/{key}"][metric][position])
                        for metric in ("pcc", "relative_l2", "reference_max", "candidate_max")
                    }
                    for label, key in (
                        ("input_residual", "input_residual_vs_hf"),
                        ("hf_group_normalized_input", "group_normalized"),
                        ("hf_on_tt_input_vs_original_hf", "hf_on_tt_input_vs_original_hf"),
                        ("tt_output_vs_hf_on_same_input", "tt_output_vs_hf_on_same_input"),
                        ("tt_output_vs_original_hf", "tt_output_vs_original_hf"),
                    )
                },
            }
            for position in result["bos_positions"]
        ]
        result["variants"][name] = metrics
    raw_path = args.tt_artifact.with_name(f"{args.output.stem}.pt")
    torch.save({"hf_outputs": outputs, "intermediates": intermediates, "row_metrics": raw_metrics}, raw_path)
    result["raw_results"] = str(raw_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "reference_reproduction": result["reference_reproduction"]["last"],
                "variants": {
                    name: {
                        key: value["last"]
                        for key, value in metrics.items()
                        if key not in ("hf_math_input_sensitivity", "bos_rows")
                    }
                    for name, metrics in result["variants"].items()
                },
            },
            indent=2,
        ),
        flush=True,
    )
    assert result["reference_reproduction"]["all_relative_l2"] < 1e-5, "Isolated HF layer did not reproduce saved HF"


if __name__ == "__main__":
    main()
