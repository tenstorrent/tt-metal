# SPDX-License-Identifier: Apache-2.0
"""Standalone full60 conditional logits and canonical token-out diagnostic (not a benchmark)."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

import ttnn

from ..tt.generator import Gemma4Generator
from ..tt.weights import HF_MODEL, HF_REVISION


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "readiness_vllm/qualitative_language_hf_probe.json")
    parser.add_argument("--output", type=Path, default=root / "readiness_vllm/qualitative_language_tt_control.json")
    args = parser.parse_args()
    reference = json.loads(args.input.read_text())
    tokens = reference["prompt_token_ids"] + reference["completion_prefix_token_ids"]
    assert reference["hf_revision"] == HF_REVISION and len(tokens) == 231
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root / "tt").glob("*.py"))}
    report = dict(
        status="running",
        scope="standalone full60 fresh prefill conditional next token",
        hf_model=HF_MODEL,
        hf_revision=HF_REVISION,
        input_source=str(args.input),
        input_source_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        input_token_ids=tokens,
        input_length=len(tokens),
        source_sha256=source_hashes,
        mesh=[1, 4],
        context_allocation=4096,
        batch_size=1,
        cases=[],
        limitations=[
            "Diagnostic full-logits readback only; not a serving or performance path.",
            "Exact reconstructed token IDs match the conditional HF and served probes; not a replay of the original unseeded generation.",
            "Fresh full prompt control; does not test resumed cache history.",
        ],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=128 * 1024 * 1024)
    gen = None
    try:
        gen = Gemma4Generator(mesh, batch_size=1, max_seq_len=4096)
        assert list(gen.model.layer_indices) == list(range(60))
        assert gen.model.precision_config["config_id"] == "head_bfp4"
        report["precision_config"] = gen.model.precision_config
        profiles = {
            "greedy": dict(temperature=0.0, top_k=1, top_p=1.0, seed=42),
            "sampled": dict(temperature=0.7, top_k=32, top_p=0.9, seed=42),
            "narrow_nucleus": dict(temperature=0.7, top_k=32, top_p=0.01, seed=42),
        }
        initial = None
        for name, sampling in profiles.items():
            print("CONDITIONAL_TT", name, flush=True)
            generated = gen.generate(tokens, 1, host_sampling=False, buffered=False, sampling_params=sampling)
            # Existing diagnostic helper reads the terminal prefill logits; no host sampling or feedback.
            logits = gen._host_logits(gen.logits).reshape(-1)
            values, ids = logits.topk(32)
            probabilities = torch.softmax(values / 0.7, dim=-1)
            candidates = {}
            for text, candidate in reference["candidates"].items():
                token_id = candidate["first_token"]["token_id"]
                candidates[text] = dict(
                    token_id=token_id, logit=float(logits[token_id]), rank=int((logits > logits[token_id]).sum()) + 1
                )
            entry = dict(
                profile=name,
                sampling=sampling,
                generated_token_ids=generated,
                completion=gen.tokenizer.decode(generated),
                candidates=candidates,
                japanese_minus_english_logit=float(logits[44839] - logits[5422]),
                top32=[
                    dict(
                        token_id=int(i),
                        text=gen.tokenizer.decode([int(i)]),
                        logit=float(v),
                        top32_probability_temperature_0_7=float(p),
                    )
                    for i, v, p in zip(ids, values, probabilities)
                ],
                logits_equal_first_case=initial is None or torch.equal(initial, logits),
            )
            if initial is None:
                initial = logits.clone()
            report["cases"].append(entry)
            save()
            print(
                json.dumps(
                    {
                        k: entry[k]
                        for k in (
                            "profile",
                            "generated_token_ids",
                            "completion",
                            "japanese_minus_english_logit",
                            "logits_equal_first_case",
                        )
                    }
                ),
                flush=True,
            )
        report["source_unchanged"] = source_hashes == {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root / "tt").glob("*.py"))
        }
        report["status"] = "complete"
        save()
        assert report["source_unchanged"], "Model source changed during diagnostic"
        assert all(
            c["logits_equal_first_case"] for c in report["cases"]
        ), "Fresh conditional logits changed across sampling profiles"
    finally:
        if gen is not None:
            gen.release_persistent_capture()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
