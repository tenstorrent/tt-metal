# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU diagnosis of fresh public PLE prefills using actual packed embedding rows.

The unchanged public fixture supplies independent full-HF fresh65/97/129-token
oracles. A second HF module changes only lookup values to actual cache-decoded
BF16 rows; checkpoint BF16 experts and every other HF operation stay unchanged.
Literal reruns must be bit-exact. This diagnostic does not change acceptance,
inject runtime values, or establish a matched TT gate without device outputs.
"""

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, DynamicCache
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextDecoderLayer

from models.autoports.qwen_qwen3_8_flash_next.tests import probe_public_continuation as public
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_ple_batch_quantization import CachedRows, tensor_sha
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_qsa_decoder import metric_passes, metrics


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.inference_mode()
def build_controls(snapshot, cache_dir, config):
    reference, weights = public.load_reference(snapshot, 1, config)
    fixtures = public.make_live_fixtures(reference, config, 1)
    del reference
    expert_dtypes = {str(value.dtype) for key, value in weights.items() if "experts" in key}
    assert expert_dtypes == {"torch.bfloat16"}, expert_dtypes
    with torch.device("meta"):
        control = Qwen4ExpTextDecoderLayer(config, 1)
    result = control.load_state_dict(weights, strict=False, assign=True)
    assert result.missing_keys == ["ple.ple_embedding.ngram_embedding.weight"] and not result.unexpected_keys
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    lookup = CachedRows(snapshot, index, "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.", cache_dir)
    control.ple.ple_embedding.ngram_embedding = lookup
    control.eval()
    captured = {}

    def router(_module, _inputs, output):
        captured["router_indices"] = output[2].clone()

    handle = control.mlp.gate.register_forward_hook(router)
    cases, records = [], []
    try:
        for fixture in fixtures:
            values = {}
            for policy in ("literal", "packed", "literal_repeat"):
                lookup.use_packed = policy == "packed"
                output = control(
                    fixture["prompt"],
                    position_embeddings=(None, None),
                    past_key_values=DynamicCache(config=config),
                    ple_input_ids=fixture["tokens"],
                )
                if policy != "packed":
                    assert torch.equal(output, fixture["expected"]), "Independent literal HF rerun must be bit-exact"
                values[policy] = {
                    "output": output.clone(),
                    "router_indices": captured["router_indices"].clone(),
                    "hash_ids": lookup.last_ids.clone(),
                }
            assert all(torch.equal(value, values["literal_repeat"][key]) for key, value in values["literal"].items())
            assert torch.equal(values["literal"]["hash_ids"], values["packed"]["hash_ids"])
            sensitivity = metrics(fixture["expected"], values["packed"]["output"], fixture["prompt"])
            route_changes = []
            for row, (literal, packed) in enumerate(
                zip(
                    values["literal"]["router_indices"].reshape(-1, config.num_experts_per_tok),
                    values["packed"]["router_indices"].reshape(-1, config.num_experts_per_tok),
                )
            ):
                if not torch.equal(literal, packed):
                    route_changes.append(
                        {
                            "row": row,
                            "literal": literal.tolist(),
                            "packed": packed.tolist(),
                            "symmetric_difference": sorted(set(literal.tolist()) ^ set(packed.tolist())),
                        }
                    )
            record = {
                "logical_length": fixture["length"],
                "seed": fixture["seed"],
                "prompt_sha256": tensor_sha(fixture["prompt"]),
                "tokens_sha256": tensor_sha(fixture["tokens"]),
                "literal_sha256": tensor_sha(fixture["expected"]),
                "packed_sha256": tensor_sha(values["packed"]["output"]),
                "literal_rerun_bit_exact": True,
                "literal_repeat_bit_exact": True,
                "hash_ids_identical": True,
                "packed_vs_literal": sensitivity,
                "packed_vs_literal_all_rows_pass": metric_passes(sensitivity),
                "router_changes": route_changes,
            }
            records.append(record)
            cases.append({**fixture, "reference": values})
            print(json.dumps(record), flush=True)
    finally:
        handle.remove()
    metadata = {
        "manifest": lookup.spec,
        "manifest_sha256": file_sha(cache_dir / "manifest.json"),
        "actual_selected_cache_vs_quantizer_exact_checks": lookup.parity_checks,
        "selected_rows": len(lookup.ids),
        "expert_dtypes": sorted(expert_dtypes),
    }
    return cases, records, metadata


def compare_device(cases, device_fixture):
    """Optional external readbacks; require exact input identity before comparing."""
    observed = torch.load(device_fixture, map_location="cpu", weights_only=False)
    assert observed["identity"]["layer"] == 1
    indexed = {case["length"]: case for case in cases}
    records = []
    seen = set()
    for value in observed["live_other_prefill"]:
        length = value["logical_length"]
        case = indexed[length]
        assert value["prompt_sha256"] == tensor_sha(case["prompt"])
        assert value["tokens_sha256"] == tensor_sha(case["tokens"])
        assert value["literal_sha256"] == tensor_sha(case["expected"])
        key = (length, value["repetition"])
        assert key not in seen
        seen.add(key)
        actual = value["actual"]
        literal = metrics(case["expected"], actual, case["prompt"])
        matched = metrics(case["reference"]["packed"]["output"], actual, case["prompt"])
        records.append(
            {
                "logical_length": length,
                "repetition": value["repetition"],
                "literal_metrics": literal,
                "matched_metrics": matched,
                "literal_passed": metric_passes(literal),
                "matched_passed": metric_passes(matched),
            }
        )
    assert seen == {(length, repetition) for length in public.LIVE_LENGTHS for repetition in (0, 1)}
    return records


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture-output", type=Path, required=True)
    parser.add_argument("--device-fixture", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(8)
    config = AutoConfig.from_pretrained(args.snapshot, local_files_only=True).text_config
    config._attn_implementation = "eager"
    cases, records, metadata = build_controls(args.snapshot, args.cache_dir, config)
    report = {
        "scope": __doc__,
        "device_executed": False,
        "acceptance_changed": False,
        "threshold": 0.995,
        "snapshot": str(args.snapshot),
        "config_sha256": file_sha(args.snapshot / "config.json"),
        "checkpoint_index_sha256": file_sha(args.snapshot / "model.safetensors.index.json"),
        "hf_source_sha256": file_sha(inspect.getfile(Qwen4ExpTextDecoderLayer)),
        "fixture_factory_source_sha256": file_sha(public.__file__),
        "diagnostic_source_sha256": file_sha(__file__),
        "cache_dir": str(args.cache_dir),
        "cache": metadata,
        "requests": records,
        "device_comparisons": compare_device(cases, args.device_fixture) if args.device_fixture else None,
        "device_comparison_pending": args.device_fixture is None,
        "device_fixture_sha256": file_sha(args.device_fixture) if args.device_fixture else None,
    }
    args.fixture_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"identity": report, "requests": cases}, args.fixture_output)
    report["fixture"] = str(args.fixture_output)
    report["fixture_sha256"] = file_sha(args.fixture_output)
    assert "ttnn" not in sys.modules, "CPU diagnostic must not import TTNN"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
