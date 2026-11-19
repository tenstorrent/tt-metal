# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU-only router sensitivity on retained, real TT normalized inputs.

CPU FP32/FP64 matvec is informative, not a bit-exact emulation of native TT
matmul or topk. This analysis neither changes gates nor waives kernel checks.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = "345d467b117d1fd5eda9b879a5988de9ae5f47a7"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[10, 14, 28, 30, 39])
    parser.add_argument("--position", type=int, default=2559)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()

    import torch
    from safetensors import safe_open

    torch.set_num_threads(8)
    snapshot = args.checkpoint or (
        Path(os.environ["HF_HOME"]) / "hub/models--Qwen--Qwen3.6-35B-A3B/snapshots" / REVISION
    )
    assert snapshot.name == REVISION
    index_path = snapshot / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())["weight_map"]
    meta_path, stat_path = (args.captures_dir / name for name in ("capture_metadata.json", "statistics.json"))
    meta, stats = (json.loads(path.read_text()) for path in (meta_path, stat_path))
    captures = torch.load(args.captures_dir / "captures.pt", map_location="cpu", weights_only=True)
    assert stats["status"] == "complete" and stats["gate_modified"] is False
    assert all(value["bit_equal"] for value in stats["instrumentation_vs_original_logits"].values())
    sources = [
        Path(__file__),
        meta_path,
        stat_path,
        index_path,
        *(
            ROOT / "tt" / name
            for name in ("prefill.py", "multichip_decoder.py", "fused_decoder.py", "mesh_fp32_topk.py")
        ),
    ]
    result = dict(
        command=sys.argv,
        scope=__doc__.strip(),
        checkpoint=str(snapshot),
        revision=REVISION,
        source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        captures_dir=str(args.captures_dir.resolve()),
        position=args.position,
        actual_routing=dict(
            input="Captured BF16 normalized post-attention hidden, replicated rank0",
            weights="Native checkpoint BF16 values uploaded as FP32",
            projection="ttnn.linear with FP32 output, HiFi4, FP32 destination; ordered_router=False in both prefill calls",
            selection="fresh multirow: native ttnn.topk; resumed singleton: MeshFP32TopK with lower-ID exact tie break",
            scores="softmax over selected eight FP32 values, then BF16 cast",
            cpu="torch FP32 matvec plus FP64 sensitivity control; stable descending sort with lower-ID tie break",
        ),
        gate_modified=False,
        original_gate=stats["gate"],
        detail_layers=args.layers,
        layers=[],
        all_layer_summary=[],
    )

    def input_row(phase, layer):
        matches = [
            key
            for key, value in meta.items()
            if value["phase"] == phase
            and value["layer"] == layer
            and value["event"] == "normalized.post_attention_layernorm.weight"
            and args.position in value.get("positions", [])
        ]
        assert len(matches) == 1, (phase, layer, matches)
        key = matches[0]
        value = captures[key][meta[key]["positions"].index(args.position)]
        assert value.dtype == torch.bfloat16 and value.shape == (2048,)
        return value, key

    def rank(logits):
        ids = torch.argsort(logits, descending=True, stable=True)
        chosen = ids[:8]
        return dict(
            top8_ids=chosen.tolist(),
            top8_logits=logits[chosen].tolist(),
            top8_scores_fp32=torch.softmax(logits[chosen].float(), dim=-1).tolist(),
            top8_scores_bf16=torch.softmax(logits[chosen].float(), dim=-1).bfloat16().float().tolist(),
            rank9_id=int(ids[8]),
            rank9_logit=float(logits[ids[8]]),
            margin_8_9=float(logits[ids[7]] - logits[ids[8]]),
        )

    for layer in range(40):
        fresh, fk = input_row("fresh", layer)
        suffix, sk = input_row("suffix", layer)
        weight_key = f"model.language_model.layers.{layer}.mlp.gate.weight"
        with safe_open(snapshot / index[weight_key], framework="pt", device="cpu") as handle:
            native_weight = handle.get_tensor(weight_key)
        assert native_weight.shape == (256, 2048) and native_weight.dtype == torch.bfloat16
        weight = native_weight.float()
        logits32 = [weight @ value.float() for value in (fresh, suffix)]
        logits64 = [weight.double() @ value.double() for value in (fresh, suffix)]
        routing = [rank(value) for value in logits32]
        routing64 = [rank(value) for value in logits64]
        left, right = (set(value["top8_ids"]) for value in routing)
        removed, added = sorted(left - right), sorted(right - left)
        delta = suffix.double() - fresh.double()
        hidden = [
            value
            for value in stats["hidden_comparisons"]
            if value["layer"] == layer
            and value["position"] == args.position
            and value["left"].startswith("fresh/")
            and value["right"].startswith("suffix/")
        ]
        by_event = {value["event"]: value for value in hidden}
        summary = dict(
            layer=layer,
            removed=removed,
            added=added,
            same_top8_set=left == right,
            same_top8_order=routing[0]["top8_ids"] == routing[1]["top8_ids"],
            margin_8_9_fresh=routing[0]["margin_8_9"],
            margin_8_9_suffix=routing[1]["margin_8_9"],
            normalized_input_pcc=by_event["normalized.post_attention_layernorm.weight"]["pcc"],
            normalized_input_rms=float(delta.square().mean().sqrt()),
            actual_hidden_before_moe_pcc=by_event["moe_input"]["pcc"],
            actual_hidden_after_moe_pcc=by_event["moe_output"]["pcc"],
            cpu_fp32_fp64_top8_order_agrees=all(a["top8_ids"] == b["top8_ids"] for a, b in zip(routing, routing64)),
        )
        result["all_layer_summary"].append(summary)
        if layer not in args.layers:
            continue
        dense_scores = []
        for selected in routing:
            values = torch.zeros(256)
            values[selected["top8_ids"]] = torch.tensor(selected["top8_scores_bf16"])
            dense_scores.append(values)
        crossings = []
        for removed_id in removed:
            for added_id in added:
                direction = weight[removed_id].double() - weight[added_id].double()
                gap_before = logits64[0][removed_id] - logits64[0][added_id]
                gap_after = logits64[1][removed_id] - logits64[1][added_id]
                predicted = direction @ delta
                crossings.append(
                    dict(
                        removed=removed_id,
                        added=added_id,
                        gap_fresh=float(gap_before),
                        gap_suffix=float(gap_after),
                        gap_change_from_input_delta=float(predicted),
                        linear_identity_residual=float((gap_after - gap_before) - predicted),
                        interpolated_fraction_at_crossing=float(-gap_before / predicted),
                        cauchy_bound_on_gap_change=float(direction.norm() * delta.norm()),
                    )
                )
        detail = dict(
            **summary,
            captures=dict(fresh=fk, suffix=sk),
            weight_key=weight_key,
            weight_sha256=hashlib.sha256(native_weight.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
            fresh=routing[0],
            suffix=routing[1],
            cpu_fp64_fresh=routing64[0],
            cpu_fp64_suffix=routing64[1],
            cpu_fp32_vs_fp64_max_abs=[float((a.double() - b).abs().max()) for a, b in zip(logits32, logits64)],
            normalized_input_max_abs_delta=float(delta.abs().max()),
            normalized_input_relative_l2=float(delta.norm() / fresh.double().norm()),
            router_logit_max_abs_delta=float((logits32[1] - logits32[0]).abs().max()),
            router_score_bf16_dense_l1_delta=float((dense_scores[1] - dense_scores[0]).abs().sum()),
            crossings=crossings,
            actual_hidden_metrics=hidden,
            interpretation=(
                "CPU router support changes from the captured small input perturbation; this explains a routing discontinuity but does not prove actual TT IDs or absolve expert arithmetic"
                if removed
                else "CPU router retains the same support; a support flip does not explain this layer's measured MoE divergence in this control"
            ),
        )
        result["layers"].append(detail)

    retained, cast = [], []
    for key, value in meta.items():
        if not key.startswith("prefix/") or value["event"] != "state.last_attention.after":
            continue
        layer, name = value["layer"], key.split("/")[-1]
        before = f"suffix/l{layer:02}/state.public_begin.before_bf16_cast/{name}"
        after = f"suffix/l{layer:02}/state.public_begin.after_bf16_cast/{name}"
        retained.append(
            dict(
                layer=layer,
                state=name,
                per_rank_equal=[torch.equal(a, b) for a, b in zip(captures[key], captures[before])],
            )
        )
        cast.append(
            dict(
                layer=layer,
                state=name,
                per_rank_changed_elements=[
                    int(torch.count_nonzero(a != b)) for a, b in zip(captures[before], captures[after])
                ],
            )
        )
    result["retained_state"] = dict(
        scope="All retained recurrent and convolution states in 30 linear-attention layers; full-attention KV bytes were not captured by this harness",
        layer_count=len({item["layer"] for item in retained}),
        tensors=len(retained),
        rank_comparisons=sum(len(item["per_rank_equal"]) for item in retained),
        all_bit_equal=all(all(item["per_rank_equal"]) for item in retained),
        exceptions=[item for item in retained if not all(item["per_rank_equal"])],
        public_cast_changes=cast,
    )
    result["status"] = "completed_cpu_sensitivity_control"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            dict(
                output=str(args.output),
                layers=[
                    {
                        key: item[key]
                        for key in (
                            "layer",
                            "removed",
                            "added",
                            "margin_8_9_fresh",
                            "margin_8_9_suffix",
                            "actual_hidden_before_moe_pcc",
                            "actual_hidden_after_moe_pcc",
                        )
                    }
                    for item in result["layers"]
                ],
                all_layer_router_support_flips=[
                    item["layer"] for item in result["all_layer_summary"] if not item["same_top8_set"]
                ],
                retained_state_equal=result["retained_state"]["all_bit_equal"],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
