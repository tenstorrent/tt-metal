# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Observe fused/optimized routing at selected steps of the unchanged traced harness."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import torch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tests import run_decoder, run_optimized_decoder
from models.autoports.google_gemma_4_26b_a4b_it.tt.fused_decoder import FusedDecoder
from models.autoports.google_gemma_4_26b_a4b_it.tt.optimized_decoder import OptimizedDecoder


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--probe-decoder", choices=("fused", "optimized"), default="optimized")
    parser.add_argument("--stage-report", type=Path, required=True)
    parser.add_argument("--inspect-positions", type=int, nargs="+", required=True)
    parser.add_argument("--compare-output-tensors", type=Path, required=True)
    parser.add_argument("--save-probe-inputs", type=Path)
    args, remaining = parser.parse_known_args()
    if "--save-output-tensors" not in remaining:
        parser.error("Output identity verification requires --save-output-tensors for the observed run")
    saved_path = Path(remaining[remaining.index("--save-output-tensors") + 1])
    if saved_path.resolve() == args.compare_output_tensors.resolve():
        parser.error("Observed and original output files must be different")
    original_load, original_pcc, original_topk = run_decoder.load_layer, run_decoder.comp_pcc, ttnn.topk
    decoder_class = OptimizedDecoder if args.probe_decoder == "optimized" else FusedDecoder
    original_factory = decoder_class.from_state_dict.__func__
    captured, references, rows, inputs = {}, {}, {}, []
    hf_layer, decode_position = None, 0
    in_decode_router = False

    def load(*a, **kw):
        nonlocal hf_layer
        hf_layer = original_load(*a, **kw)
        stages = {}

        def attention_hook(_module, _inputs, output):
            stages["attention"] = output[0].detach().clone()

        def router_hook(_module, router_inputs, output):
            stages["residual"] = router_inputs[0].detach().clone()
            stages["routes"] = output[2].detach().clone()

        def scores_hook(_module, _inputs, output):
            stages["scores"] = output.detach().clone()

        hf_layer.self_attn.register_forward_hook(attention_hook)
        hf_layer.router.register_forward_hook(router_hook)
        hf_layer.router.proj.register_forward_hook(scores_hook)
        forward = hf_layer.forward

        def observed_forward(value, *a, **kw):
            nonlocal decode_position
            output = forward(value, *a, **kw)
            if args.save_probe_inputs:
                inputs.append(value.detach().clone())
            if value.shape[-2] > 1:
                decode_position = value.shape[-2]
            else:
                references[output.data_ptr()] = {
                    "position": decode_position,
                    **{name: tensor.clone() for name, tensor in stages.items()},
                }
                decode_position += 1
            return output

        hf_layer.forward = observed_forward
        return hf_layer

    def observed_topk(value, *a, **kw):
        output = original_topk(value, *a, **kw)
        if in_decode_router:
            captured.update(scores=value, indices=output[1])
        return output

    def factory(cls, *a, **kw):
        result = original_factory(cls, *a, **kw)
        router, attention = result.layer.moe.router, result.layer.self_attn

        class RouterProbe:
            def __getattr__(self, name):
                return getattr(router, name)

            def __call__(self, value, *a, **kw):
                nonlocal in_decode_router
                in_decode_router = value.shape[-2] == 1
                try:
                    output = router(value, *a, **kw)
                    if in_decode_router:
                        captured.update(residual=value, normalized=kw.get("normalized"), routing=output)
                    return output
                finally:
                    in_decode_router = False

        class AttentionProbe:
            def __getattr__(self, name):
                return getattr(attention, name)

            def __call__(self, value, *a, **kw):
                output = attention(value, *a, **kw)
                if value.shape[-2] == 1:
                    captured["attention"] = output
                return output

        result.layer.moe.router = RouterProbe()
        result.layer.self_attn = AttentionProbe()
        return result

    def host(value):
        return ttnn.to_torch(value).float().reshape(1, -1)

    def correlation(a, b):
        return float(original_pcc(a.flatten().float(), b.flatten().float(), 0.995)[1])

    def ranked(scores):
        scores = scores.flatten()
        values, indices = scores.sort(descending=True)
        count = hf_layer.config.top_k_experts
        return {
            "experts": indices[:count].tolist(),
            "probability_topk_experts": scores.softmax(-1).topk(count).indices.tolist(),
            "rank8_rank9_gap": float(values[count - 1] - values[count]),
        }

    def compare(reference, actual, *a, **kw):
        passed, pcc = original_pcc(reference, actual, *a, **kw)
        state = references.get(reference.data_ptr())
        if state is None or (passed and state["position"] not in args.inspect_positions):
            return passed, pcc
        residual, scores = host(captured["residual"]), host(captured["scores"])
        normalized = host(captured["normalized"]) if captured["normalized"] is not None else None
        indices = ttnn.to_torch(captured["indices"]).long().flatten().tolist()
        routing = host(captured["routing"]).flatten()
        router = hf_layer.router
        with torch.no_grad():
            hf_normalized = router.norm(state["residual"])
            same_residual_norm = router.norm(residual)
            same_residual_scores = router.proj(same_residual_norm * router.scale * router.scalar_root_size)
            same_normalized_scores = (
                router.proj(normalized * router.scale * router.scalar_root_size) if normalized is not None else None
            )
        row = {
            "position": state["position"],
            "hf_pcc": float(pcc),
            "hf_passed": bool(passed),
            "attention_pcc": correlation(state["attention"], host(captured["attention"])),
            "residual_pcc": correlation(state["residual"], residual),
            "hf_routes": state["routes"].flatten().tolist(),
            "tt_routes": indices,
            "tt_route_weights": [{"expert": index, "weight": float(routing[index])} for index in indices],
            "hf_scores": ranked(state["scores"]),
            "tt_scores": ranked(scores),
            "cpu_same_residual": ranked(same_residual_scores),
        }
        if normalized is not None:
            row.update(
                common_normalized_pcc=correlation(hf_normalized, normalized),
                common_norm_same_residual_pcc=correlation(same_residual_norm, normalized),
                cpu_same_normalized=ranked(same_normalized_scores),
                scores_same_normalized_pcc=correlation(same_normalized_scores, scores),
                scores_same_normalized_max_abs_error=float((same_normalized_scores - scores).abs().max()),
            )
        score_sources = {"hf": state["scores"], "tt": scores, "cpu_same_residual": same_residual_scores}
        if same_normalized_scores is not None:
            score_sources["cpu_same_normalized"] = same_normalized_scores
        candidates = sorted(
            {int(index) for value in score_sources.values() for index in value.flatten().topk(9).indices}
        )
        row["candidate_scores"] = [
            {"expert": index, **{name: float(value.flatten()[index]) for name, value in score_sources.items()}}
            for index in candidates
        ]
        rows[state["position"]] = row
        print("STRESS_ROUTES", json.dumps(row), flush=True)
        return passed, pcc

    entrypoint = run_optimized_decoder.main if args.probe_decoder == "optimized" else run_decoder.main
    sys.argv = [sys.argv[0], *([] if args.probe_decoder == "optimized" else ["--decoder", "fused"]), *remaining]
    report = {
        "decoder": args.probe_decoder,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "inspect_positions": args.inspect_positions,
        "uninstrumented_outputs": str(args.compare_output_tensors),
        "observed_outputs": str(saved_path),
        "outputs_equal_to_uninstrumented": False,
        "limitations": "Retained intermediate tensors can change allocation lifetimes; exact output identity is required",
    }
    try:
        with (
            patch.object(run_decoder, "load_layer", load),
            patch.object(run_decoder, "comp_pcc", compare),
            patch.object(decoder_class, "from_state_dict", classmethod(factory)),
            patch.object(ttnn, "topk", observed_topk),
        ):
            entrypoint()
    finally:
        report["positions"] = [rows[position] for position in sorted(rows)]
        if saved_path.exists():
            before = torch.load(args.compare_output_tensors, map_location="cpu", weights_only=True)
            after = torch.load(saved_path, map_location="cpu", weights_only=True)
            exact = len(before["decode"]) == len(after["decode"]) and torch.equal(before["prefill"], after["prefill"])
            if exact:
                exact = all(torch.equal(a, b) for a, b in zip(before["decode"], after["decode"]))
            report["outputs_equal_to_uninstrumented"] = exact
            report["compared_decode_steps"] = len(after["decode"])
        if args.save_probe_inputs and inputs:
            torch.save({"x": inputs[0], "decode_inputs": torch.cat(inputs[1:], dim=1)}, args.save_probe_inputs)
            report["input_fixture"] = str(args.save_probe_inputs)
        args.stage_report.write_text(json.dumps(report, indent=2) + "\n")
        assert report["outputs_equal_to_uninstrumented"], "Observed outputs differ from the uninstrumented control"


if __name__ == "__main__":
    main()
