# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only full layer-1 NF4 error isolation; no TT device acceptance claim.

Reproduces probe_ple_decoder.py's seed1731, 33-token prefix and three changed
decode inputs, including EOS. Only selected checkpoint rows are read. All
other full decoder weights and HF operations are identical across candidates.
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig, DynamicCache
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextDecoderLayer

from models.autoports.qwen_qwen3_8_flash_next.tests.probe_ple_quantization import SelectedRows, quantize


class GroupedSelectedRows(SelectedRows):
    """Diagnostic group quantization using only each real row's own values."""

    candidate = None
    codebooks = None

    def forward(self, ids):
        exact = super().forward(ids)
        if self.candidate is None:
            return exact
        method, group_size = self.candidate
        if method == "scale8":
            from models.autoports.qwen_qwen3_8_flash_next.tests.probe_ple_scale8_hypothesis import reconstruct_scale8

            assert group_size == self.scale8_spec["group_size"]
            return reconstruct_scale8(exact, self.scale8_spec)[0]
        if "mixed" in method:
            count = 40 if "mixed40" in method else 64
            head = torch.bucketize(ids, self.head_offsets[1:], right=True)
            mask = self.important_dimensions[count][head]
            result = []
            for part, selected in zip(exact.split(group_size, dim=-1), mask.split(group_size, dim=-1)):
                values = part.float()
                books = [self.codebooks[(group_size, False)], self.codebooks[(group_size, "32")]]
                boundaries = [(book[:-1] + book[1:]) / 2 for book in books]
                scale = values.abs().amax(-1, keepdim=True).clamp_min(1e-20)

                def levels():
                    candidates = [
                        book[torch.bucketize(values / scale, boundary)] for book, boundary in zip(books, boundaries)
                    ]
                    return torch.where(selected, candidates[1], candidates[0])

                for _ in range(5):
                    codes = levels()
                    scale = ((values * codes).sum(-1, keepdim=True) / codes.square().sum(-1, keepdim=True)).clamp_min(
                        1e-20
                    )
                scale = scale.bfloat16().float()
                result.append((levels() * scale).bfloat16())
            return torch.cat(result, -1)
        if method in ("absmax", "lsq"):
            return torch.cat([quantize(part, method) for part in exact.split(group_size, dim=-1)], dim=-1)
        return torch.cat(
            [self.quantize_group(part, method, group_size) for part in exact.split(group_size, dim=-1)], dim=-1
        )

    def quantize_group(self, part, method, group_size):
        from models.autoports.qwen_qwen3_8_flash_next.tests.probe_ple_quantization import NF4

        values = part.float()
        center = values.mean(-1, keepdim=True).bfloat16().float() if "affine" in method else 0.0
        values = values - center
        codebook = self.codebooks[(group_size, "affine" in method)] if "learned" in method else torch.tensor(NF4)
        thresholds = (codebook[:-1] + codebook[1:]) / 2
        scale = values.abs().amax(-1, keepdim=True).clamp_min(1e-20)
        if "lsq" in method:
            for _ in range(5):
                codes = codebook[torch.bucketize(values / scale, thresholds)]
                scale = ((values * codes).sum(-1, keepdim=True) / codes.square().sum(-1, keepdim=True)).clamp_min(1e-20)
        scale = scale.bfloat16().float()
        return (codebook[torch.bucketize(values / scale, thresholds)] * scale + center).bfloat16()


def fit_codebook(exact_rows, group_size, affine, levels=16):
    """Symmetric Lloyd MSE fit on checkpoint rows disjoint from evaluation IDs."""
    normalized = []
    for part in exact_rows.split(group_size, dim=-1):
        part = part.float()
        if affine:
            part = part - part.mean(-1, keepdim=True).bfloat16().float()
        scale = part.abs().amax(-1, keepdim=True).clamp_min(1e-20).bfloat16().float()
        normalized.append((part / scale).abs().flatten())
    values = torch.cat(normalized)
    positive_count = levels // 2
    positive = torch.linspace(1 / levels, (levels - 1) / levels, positive_count)
    for _ in range(100):
        idx = torch.bucketize(values, (positive[:-1] + positive[1:]) / 2)
        counts = torch.bincount(idx, minlength=positive_count)
        means = torch.bincount(idx, weights=values, minlength=positive_count) / counts.clamp_min(1)
        new = torch.where(counts > 0, means, positive)
        if (new - positive).abs().max() < 1e-7:
            positive = new
            break
        positive = new
    return torch.cat((-positive.flip(0), positive))


def metrics(expected, actual, residual):
    def row_pcc(left, right):
        left, right = left.float().reshape(-1, left.shape[-1]), right.float().reshape(-1, right.shape[-1])
        left, right = left - left.mean(-1, keepdim=True), right - right.mean(-1, keepdim=True)
        return (left * right).sum(-1) / (left.square().sum(-1) * right.square().sum(-1)).sqrt()

    whole = row_pcc(expected, actual)
    contribution = row_pcc(expected.float() - residual.float(), actual.float() - residual.float())
    return {
        "whole_min_row_pcc": whole.min().item(),
        "whole_per_row_pcc": whole.tolist(),
        "contribution_min_row_pcc": contribution.min().item(),
        "contribution_per_row_pcc": contribution.tolist(),
        "global_whole_pcc": torch.corrcoef(torch.stack((expected.float().flatten(), actual.float().flatten())))[
            0, 1
        ].item(),
        "global_contribution_pcc": torch.corrcoef(
            torch.stack(
                ((expected.float() - residual.float()).flatten(), (actual.float() - residual.float()).flatten())
            )
        )[0, 1].item(),
        "passed": bool(torch.isfinite(actual).all() and (whole >= 0.995).all() and (contribution >= 0.995).all()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=33)
    parser.add_argument("--seed", type=int, default=1731)
    parser.add_argument("--codebook-report", type=Path, help="Reuse an already fitted immutable codebook")
    parser.add_argument("--scale8-spec", type=Path, help="Immutable independently calibrated scale-byte hypothesis")
    parser.add_argument("--lengths", default="", help="Additional fresh-prefix lengths")
    parser.add_argument("--continuation-starts", default="", help="Additional prefix starts followed by8 resumed rows")
    parser.add_argument(
        "--methods",
        default="absmax:160,lsq:160",
        help="Comma-separated method:group-size candidates; final group may be shorter",
    )
    parser.add_argument("--routing-diagnostics", action="store_true")
    parser.add_argument(
        "--routing-counterfactual",
        action="store_true",
        help="Also rerun each candidate forcing exact-reference expert sets, diagnostic only",
    )
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    config = AutoConfig.from_pretrained(args.snapshot, local_files_only=True).text_config
    config._attn_implementation = "eager"
    index = json.loads((args.snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = "model.language_model.layers.1."
    keys = [key for key in index if key.startswith(prefix) and ".ngram_embedding.shard_" not in key]
    weights = {}
    for filename in sorted({index[key] for key in keys}):
        with safe_open(args.snapshot / filename, framework="pt") as f:
            for key in keys:
                if index[key] == filename:
                    weights[key.removeprefix(prefix)] = f.get_tensor(key)
    with torch.device("meta"):
        reference = Qwen4ExpTextDecoderLayer(config, 1)
    result = reference.load_state_dict(weights, strict=False, assign=True)
    if result.missing_keys != ["ple.ple_embedding.ngram_embedding.weight"] or result.unexpected_keys:
        raise ValueError(f"Unexpected state contract: {result}")
    rows = GroupedSelectedRows(args.snapshot, index, prefix + "ple.ple_embedding.ngram_embedding.", "absmax")
    if args.scale8_spec:
        rows.scale8_spec = json.loads(args.scale8_spec.read_text())
    reference.ple.ple_embedding.ngram_embedding = rows
    reference.eval()
    captures = None
    forced_router = None

    def ple_hook(module, inputs, output):
        if captures is not None:
            captures["ple"].append(output.clone())

    def mlp_hook(module, inputs):
        if captures is not None:
            captures["pre_moe"].append(inputs[0].clone())

    def router_hook(module, inputs, output):
        if captures is not None:
            captures["router"].append(tuple(value.clone() for value in output))
        if forced_router is not None:
            indices = forced_router[len(captures["router"]) - 1][2]
            logits = output[0]
            scores = logits.float().softmax(-1).gather(-1, indices)
            if config.norm_topk_prob:
                scores = scores / scores.sum(-1, keepdim=True)
            return logits, scores.to(logits.dtype), indices

    if args.routing_diagnostics or args.routing_counterfactual:
        reference.ple.register_forward_hook(ple_hook)
        reference.mlp.register_forward_pre_hook(mlp_hook)
        reference.mlp.gate.register_forward_hook(router_hook)
    prompt = (torch.randn(1, args.seq_len, 10240) * 0.02).bfloat16()
    decode_inputs = [(torch.randn(1, 1, 10240) * 0.02).bfloat16() for _ in range(3)]
    ids = torch.randint(0, config.vocab_size, (1, args.seq_len + 3))
    eos = config.eos_token_id[0] if isinstance(config.eos_token_id, list) else config.eos_token_id
    ids[0, max(args.seq_len - 2, 0)], ids[0, args.seq_len + 1] = eos, eos
    report = {
        "scope": __doc__,
        "snapshot": str(args.snapshot),
        "seed": args.seed,
        "seq_len": args.seq_len,
        "threads": 8,
        "threshold": 0.995,
        "moe_weight_precision": "exact checkpoint BF16; no BFP8/BFP4 expert approximation in this CPU experiment",
        "candidates": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def run():
        nonlocal captures
        captures = {"ple": [], "pre_moe": [], "router": []}
        cache = DynamicCache(config=config)
        output = [
            reference(
                prompt, position_embeddings=(None, None), past_key_values=cache, ple_input_ids=ids[:, : args.seq_len]
            )
        ]
        for step, value in enumerate(decode_inputs):
            output.append(
                reference(
                    value,
                    position_embeddings=(None, None),
                    past_key_values=cache,
                    ple_input_ids=ids[:, args.seq_len + step : args.seq_len + step + 1],
                )
            )
        return output, captures

    with torch.inference_mode():
        expected, expected_captures = run()
        print("CPU_FULL_PLE_EXACT_READY", flush=True)
        selections = [(method, int(size)) for method, size in (item.split(":") for item in args.methods.split(","))]
        if any(method == "scale8" for method, _ in selections):
            assert args.scale8_spec is not None
            overlap = len(set(rows.scale8_spec["training_rows"]) & rows.ids)
            assert overlap == 0, "Scale-byte calibration rows overlap short decoder holdout IDs"
            report["scale8_spec"] = rows.scale8_spec
            report["scale8_calibration_holdout_overlap"] = overlap
        if args.codebook_report:
            training = json.loads(args.codebook_report.read_text())["training"]
            rows.codebooks = {
                (size, "affine" in method): torch.tensor(
                    training["fixed_codebooks"][f"group{size}_affine{'affine' in method}"]
                )
                for method, size in selections
                if "learned" in method
            }
            report["training"] = training
            report["training_reused_from"] = str(args.codebook_report)
            report["new_validation_training_id_overlap"] = len(set(training["global_row_ids"]) & rows.ids)
            if report["new_validation_training_id_overlap"]:
                raise ValueError("Immutable training rows overlap these evaluation IDs")
        elif any("learned" in method for method, _ in selections):
            generator = torch.Generator().manual_seed(8197)
            validation_ids = set(rows.ids)
            training_ids = []
            # Equal coverage of all128 checkpoint shards; never fit on any
            # selected validation embedding ID or any model activation.
            for shard in range(128):
                accepted = set()
                while len(accepted) < 32:
                    proposed = shard * 2500012 + int(torch.randint(0, 2500012, (), generator=generator))
                    if proposed not in validation_ids:
                        accepted.add(proposed)
                training_ids.extend(sorted(accepted))
            training_rows = rows(torch.tensor(training_ids))
            rows.ids = validation_ids.copy()
            rows.codebooks = {}
            codebook_report = {}
            for method, size in selections:
                if "learned" not in method:
                    continue
                key = size, "affine" in method
                if key not in rows.codebooks:
                    rows.codebooks[key] = fit_codebook(training_rows, *key)
                    codebook_report[f"group{size}_affine{key[1]}"] = rows.codebooks[key].tolist()
            report["training"] = {
                "seed": 8197,
                "rows": len(training_ids),
                "all_shards": 128,
                "validation_id_overlap": len(set(training_ids) & validation_ids),
                "global_row_ids": training_ids,
                "fixed_codebooks": codebook_report,
            }
            print("CPU_FULL_PLE_CODEBOOKS_READY", json.dumps(codebook_report), flush=True)
        if any("mixed" in method for method, _ in selections):
            # Select dimensions from projection weights only. No selected token,
            # input activation, or failed output row influences the choice.
            importance = []
            for name in ("ple.key_proj.weight", "ple.value_proj.weight"):
                norms = weights[name].float().square().sum(0).reshape(16, 160)
                importance.append(norms / norms.mean(-1, keepdim=True))
            importance = sum(importance)
            rows.important_dimensions = {}
            rows.head_offsets = reference.ple.ple_embedding.ngram_heads_offsets
            report["projection_importance"] = {
                "source": "sum of separately head-normalized real key/value projection column squared norms",
                "selected_dimensions": {},
            }
            validation_ids = rows.ids.copy()
            training_rows = rows(torch.tensor(report["training"]["global_row_ids"]))
            rows.ids = validation_ids
            for method, size in selections:
                if "mixed" not in method:
                    continue
                count = 40 if "mixed40" in method else 64
                selected = importance.topk(count, dim=-1).indices.sort(-1).values
                mask = torch.zeros_like(importance, dtype=torch.bool).scatter_(-1, selected, True)
                rows.important_dimensions[count] = mask
                report["projection_importance"]["selected_dimensions"][str(count)] = selected.tolist()
                if (size, "32") not in rows.codebooks:
                    rows.codebooks[(size, "32")] = fit_codebook(training_rows, size, False, levels=32)
                    report["training"]["fixed_codebooks"][f"group{size}_32levels"] = rows.codebooks[
                        (size, "32")
                    ].tolist()
        for selection in args.methods.split(","):
            method, size = selection.split(":")
            size = int(size)
            if (
                method
                not in (
                    "absmax",
                    "lsq",
                    "learned",
                    "learned_lsq",
                    "affine",
                    "affine_lsq",
                    "affine_learned",
                    "affine_learned_lsq",
                    "mixed40_learned_lsq",
                    "mixed64_learned_lsq",
                    "scale8",
                )
                or not 1 <= size <= 160
            ):
                raise ValueError(f"Invalid quantization candidate {selection}")
            rows.candidate = method, size
            actual, actual_captures = run()
            candidate = [
                dict(phase="prefill" if step == 0 else f"decode{step - 1}", **metrics(target, observed, residual))
                for step, (target, observed, residual) in enumerate(zip(expected, actual, [prompt, *decode_inputs]))
            ]
            if args.routing_diagnostics or args.routing_counterfactual:
                for step, entry in enumerate(candidate):
                    entry["diagnostic_components"] = {
                        name: metrics(
                            expected_captures[name][step],
                            actual_captures[name][step],
                            torch.zeros_like(expected_captures[name][step]),
                        )
                        for name in ("ple", "pre_moe")
                    }
                    exact_indices = expected_captures["router"][step][2].sort(-1).values
                    actual_indices = actual_captures["router"][step][2].sort(-1).values
                    changed = (exact_indices != actual_indices).any(-1)
                    entry["changed_expert_set_per_row"] = changed.tolist()
                    entry["changed_expert_set_rows"] = int(changed.sum())
            if args.routing_counterfactual:
                forced_router = expected_captures["router"]
                forced, _ = run()
                forced_router = None
                for entry, target, observed, residual in zip(candidate, expected, forced, [prompt, *decode_inputs]):
                    entry["diagnostic_forced_exact_expert_sets"] = metrics(target, observed, residual)
            metadata = ((160 + size - 1) // size) * (2 if "affine" in method else 1)
            report["candidates"][selection] = {
                "group_size": size,
                "scales_per_row": (160 + size - 1) // size,
                "bf16_metadata_per_row": metadata,
                "extra_scale_bytes_over_row_scale": (metadata - 1) * 2 * 320001536,
                "phases": candidate,
                "passed": all(row["passed"] for row in candidate),
            }
            if "mixed" in method:
                extra_bits = 40 if "mixed40" in method else 64
                report["candidates"][selection]["extra_payload_bytes"] = extra_bits * 320001536 // 8
            if method == "scale8":
                report["candidates"][selection].update(
                    bf16_metadata_per_row=0,
                    scale_bytes_per_row=16,
                    extra_scale_bytes_over_row_scale=14 * 320001536,
                    full_embedding_payload_bytes=96 * 320001536,
                )
            additional = []
            for length in [int(value) for value in args.lengths.split(",") if value]:
                if not 1 <= length <= args.seq_len:
                    raise ValueError("Extra prefix length exceeds reference")
                result = reference(
                    prompt[:, :length],
                    position_embeddings=(None, None),
                    past_key_values=DynamicCache(config=config),
                    ple_input_ids=ids[:, :length],
                )
                additional.append(
                    dict(
                        kind="fresh_prefix",
                        length=length,
                        **metrics(expected[0][:, :length], result, prompt[:, :length]),
                    )
                )
            for start in [int(value) for value in args.continuation_starts.split(",") if value]:
                if not 1 <= start <= args.seq_len - 8:
                    raise ValueError("Continuation window exceeds reference")
                cache = DynamicCache(config=config)
                reference(
                    prompt[:, :start],
                    position_embeddings=(None, None),
                    past_key_values=cache,
                    ple_input_ids=ids[:, :start],
                )
                result = reference(
                    prompt[:, start : start + 8],
                    position_embeddings=(None, None),
                    past_key_values=cache,
                    ple_input_ids=ids[:, start : start + 8],
                )
                additional.append(
                    dict(
                        kind="continuation",
                        start=start,
                        rows=8,
                        **metrics(expected[0][:, start : start + 8], result, prompt[:, start : start + 8]),
                    )
                )
            report["candidates"][selection]["additional_contract_cases"] = additional
            report["candidates"][selection]["passed"] &= all(row["passed"] for row in additional)
            report["selected_real_rows"] = len(rows.ids)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(
                "CPU_FULL_PLE_QUANTIZATION",
                json.dumps({"method": selection, **report["candidates"][selection]}),
                flush=True,
            )


if __name__ == "__main__":
    main()
