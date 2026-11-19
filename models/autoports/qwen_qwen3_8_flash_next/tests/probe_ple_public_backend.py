# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU projection-accumulation counterfactuals for public fresh PLE prefills.

Actual packed embedding rows and unchanged checkpoint BF16 experts are used.
Each named nn.Linear retains its operands, weight, bias, and BF16 output
boundary; only accumulation is changed to FP64. Single-module controls precede
combined controls. This diagnostic neither changes acceptance nor demonstrates
a TT cause without matched device-output/intermediate evidence.
"""

import argparse
import contextlib
import inspect
import json
import sys
from pathlib import Path
from unittest import mock

import torch
import torch.nn.functional as F
from transformers import AutoConfig, DynamicCache
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextDecoderLayer

from .probe_ple_batch_quantization import CachedRows, load_reference, tensor_sha
from .probe_ple_public_quantization import file_sha
from .probe_qsa_decoder import metric_passes, metrics

PROJECTIONS = {
    "attn_down": "attn_hyper_connection.input_mix_weight_down",
    "attn_up": "attn_hyper_connection.input_mix_weight_up",
    "mlp_down": "mlp_hyper_connection.input_mix_weight_down",
    "mlp_up": "mlp_hyper_connection.input_mix_weight_up",
    "gdn_qkv": "linear_attn.in_proj_qkv",
    "gdn_out": "linear_attn.out_proj",
}


@torch.inference_mode()
def public_backend_controls(config, weights, fixtures, snapshot, cache_dir):
    """Independent HF modules/caches with one fixed policy for every fresh case.

    No TT values enter this reference. Literal reruns before and after the
    control must match the untouched full-HF fixture bit for bit.
    """
    rng = torch.random.get_rng_state()
    with torch.device("meta"):
        reference = Qwen4ExpTextDecoderLayer(config, 1)
    torch.random.set_rng_state(rng)
    result = reference.load_state_dict(weights, strict=False, assign=True)
    assert result.missing_keys == ["ple.ple_embedding.ngram_embedding.weight"] and not result.unexpected_keys
    assert {value.dtype for name, value in weights.items() if "experts" in name} == {torch.bfloat16}
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    lookup = CachedRows(snapshot, index, "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.", cache_dir)
    reference.ple.ple_embedding.ngram_embedding = lookup
    reference.eval()
    captured = {}
    hook = reference.mlp.gate.register_forward_hook(
        lambda _module, _inputs, output: captured.update(indices=output[2].clone())
    )
    controls, records = [], []
    try:
        for case in fixtures:
            outputs, routes, ids = {}, {}, {}
            for policy in ("literal", "packed_six_fp64", "literal_repeat"):
                lookup.use_packed = policy == "packed_six_fp64"
                with contextlib.ExitStack() as stack:
                    if lookup.use_packed:
                        for path in PROJECTIONS.values():
                            module = reference.get_submodule(path)

                            def project(value, module=module):
                                assert value.dtype == module.weight.dtype == torch.bfloat16 and module.bias is None
                                return F.linear(value.double(), module.weight.double()).to(value.dtype)

                            stack.enter_context(mock.patch.object(module, "forward", project))
                    outputs[policy] = reference(
                        case["prompt"],
                        position_embeddings=(None, None),
                        past_key_values=DynamicCache(config=config),
                        ple_input_ids=case["tokens"],
                    )
                routes[policy] = captured["indices"].clone()
                ids[policy] = lookup.last_ids.clone()
                if policy != "packed_six_fp64":
                    assert torch.equal(
                        outputs[policy], case["expected"]
                    ), "Independent literal HF rerun must be bit-exact"
            assert torch.equal(routes["literal"], routes["literal_repeat"])
            assert all(torch.equal(ids["literal"], value) for value in ids.values())
            sensitivity = metrics(case["expected"], outputs["packed_six_fp64"], case["prompt"])
            record = {
                "logical_length": case["length"],
                "seed": case["seed"],
                "prompt_sha256": tensor_sha(case["prompt"]),
                "tokens_sha256": tensor_sha(case["tokens"]),
                "literal_sha256": tensor_sha(case["expected"]),
                "control_sha256": tensor_sha(outputs["packed_six_fp64"]),
                "independent_literal_rerun_bit_exact": True,
                "literal_repeat_bit_exact": True,
                "hash_ids_identical": True,
                "literal_router_indices": routes["literal"].tolist(),
                "control_router_indices": routes["packed_six_fp64"].tolist(),
                "sensitivity": sensitivity,
            }
            records.append(record)
            controls.append({"output": outputs["packed_six_fp64"], "sensitivity": sensitivity})
    finally:
        hook.remove()
    metadata = {
        "policy": "actual_packed_lookup_and_all_six_fp64_projections_to_original_bf16_boundary",
        "projection_scope": PROJECTIONS,
        "arithmetic_scope": "The six runtime FP32-SFPU projection boundaries use FP64 mathematical accumulation in the independent CPU control; this is not bitwise FP32-tree emulation. Other HF operations, router, BF16 experts, and cache semantics are unchanged.",
        "source_sha256": file_sha(__file__),
        "hf_source_sha256": file_sha(inspect.getfile(Qwen4ExpTextDecoderLayer)),
        "config_sha256": file_sha(snapshot / "config.json"),
        "checkpoint_index_sha256": file_sha(snapshot / "model.safetensors.index.json"),
        "manifest_sha256": file_sha(cache_dir / "manifest.json"),
        "manifest": lookup.spec,
        "actual_cache_parity_checks": lookup.parity_checks,
        "selected_rows": len(lookup.ids),
        "expert_dtype": "unchanged checkpoint BF16",
        "independent_literal_rerun_bit_exact": True,
        "requests": records,
    }
    return controls, metadata


def fresh_backend_gate(literal, matched, sensitivity):
    """Require all matched gates and reproduce every literal failing fresh row."""
    from .probe_batch import ple_quantization_control_gate

    rows = [item["row"] for item in literal["per_row"]]
    if any([item["row"] for item in value["per_row"]] != rows for value in (matched, sensitivity)):
        return {"passed": False, "policy": "row_identity_mismatch", "carried": []}
    result = ple_quantization_control_gate(literal, matched, sensitivity, phase="prefill")
    if result["carried"]:
        result["policy"] = "matched_ple_packed_and_six_projection_backend_control"
    return result


def host_gate_self_checks():
    from copy import deepcopy

    good = {
        "finite": True,
        "pcc": 1.0,
        "contribution_pcc": 1.0,
        "per_row": [{"row": row, "pcc": 1.0, "contribution_pcc": 1.0} for row in range(2)],
    }
    bad = deepcopy(good)
    bad["per_row"][1]["contribution_pcc"] = 0.994
    assert fresh_backend_gate(bad, good, bad)["carried"] == ["row:1"]
    assert not fresh_backend_gate(bad, good, good)["passed"]
    wrong_row = deepcopy(good)
    wrong_row["per_row"][0]["contribution_pcc"] = 0.994
    assert not fresh_backend_gate(bad, good, wrong_row)["passed"]
    assert not fresh_backend_gate(good, bad, good)["passed"]
    whole_failure = deepcopy(bad)
    whole_failure["per_row"][1]["pcc"] = 0.994
    assert not fresh_backend_gate(whole_failure, good, bad)["passed"]
    aggregate = deepcopy(bad)
    aggregate["contribution_pcc"] = 0.994
    assert not fresh_backend_gate(aggregate, good, bad)["passed"]
    assert fresh_backend_gate(aggregate, good, aggregate)["passed"]
    invalid = deepcopy(bad)
    invalid["per_row"][1]["contribution_pcc"] = float("nan")
    assert not fresh_backend_gate(invalid, good, bad)["passed"]
    mislabeled = deepcopy(good)
    mislabeled["per_row"][1]["row"] = 7
    assert not fresh_backend_gate(good, mislabeled, good)["passed"]
    assert fresh_backend_gate(good, good, good)["passed"]


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--cpu-fixture", type=Path, required=True)
    parser.add_argument("--lengths", default="65")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensor-output", type=Path, required=True)
    parser.add_argument("--device-fixture", type=Path)
    parser.add_argument(
        "--all-six-only",
        action="store_true",
        help="One fixed six-projection FP64 policy for every requested length, plus unchanged packed baseline/repeat controls",
    )
    args = parser.parse_args()
    torch.set_num_threads(8)
    saved = torch.load(args.cpu_fixture, map_location="cpu", weights_only=False)
    assert saved["identity"]["config_sha256"] == file_sha(args.snapshot / "config.json")
    assert saved["identity"]["cache"]["manifest_sha256"] == file_sha(args.cache_dir / "manifest.json")
    assert saved["identity"]["hf_source_sha256"] == file_sha(inspect.getfile(Qwen4ExpTextDecoderLayer))
    requested = {int(value) for value in args.lengths.split(",")}
    cases = [case for case in saved["requests"] if case["length"] in requested]
    assert {case["length"] for case in cases} == requested
    config = AutoConfig.from_pretrained(args.snapshot, local_files_only=True).text_config
    config._attn_implementation = "eager"
    reference = load_reference(args.snapshot, config, args.cache_dir)
    lookup = reference.ple.ple_embedding.ngram_embedding
    lookup.use_packed = True
    captured = {}

    def router(_module, _inputs, output):
        captured["indices"] = output[2].clone()

    hook = reference.mlp.gate.register_forward_hook(router)
    observed = torch.load(args.device_fixture, map_location="cpu", weights_only=False) if args.device_fixture else None
    policies = {"packed_native": (), **{name: (name,) for name in PROJECTIONS}}
    policies["all_gr"] = tuple(PROJECTIONS)[:4]
    policies["all_six"] = tuple(PROJECTIONS)
    policies["packed_native_repeat"] = ()
    if args.all_six_only:
        policies = {name: policies[name] for name in ("packed_native", "all_six", "packed_native_repeat")}
    report = {
        "scope": __doc__,
        "device_executed": False,
        "acceptance_changed": False,
        "cpu_fixture_sha256": file_sha(args.cpu_fixture),
        "source_sha256": file_sha(__file__),
        "hf_source_sha256": file_sha(inspect.getfile(Qwen4ExpTextDecoderLayer)),
        "device_fixture_sha256": file_sha(args.device_fixture) if args.device_fixture else None,
        "torch_config": torch.__config__.show(),
        "all_six_only": args.all_six_only,
        "projection_scope": PROJECTIONS,
        "arithmetic_scope": "FP64 controls the six GR down/up and GDN QKV/out boundaries implemented with FP32 SFPU trees; it is not a bitwise emulation of FP32 tree accumulation. HF router and all other modules remain unchanged.",
        "device_identity": observed["identity"] if observed else None,
        "cases": [],
    }
    if observed:
        assert observed["identity"]["layer"] == 1
        assert observed["identity"]["config_sha256"] == saved["identity"]["config_sha256"]
        assert observed["identity"]["hf_source_sha256"] == saved["identity"]["hf_source_sha256"]
        selected_outputs = [item for item in observed["live_other_prefill"] if item["logical_length"] in requested]
        keyed = {(item["logical_length"], item["repetition"]): item for item in selected_outputs}
        assert len(keyed) == len(selected_outputs) == 2 * len(requested)
        assert set(keyed) == {(length, repetition) for length in requested for repetition in (0, 1)}
        report["device_repeats_bit_exact"] = {
            str(length): torch.equal(keyed[length, 0]["actual"], keyed[length, 1]["actual"]) for length in requested
        }
    tensors = []
    try:
        for case in cases:
            for policy, selected in policies.items():
                same_operand = {}
                with contextlib.ExitStack() as stack:
                    for name in selected:
                        module = reference.get_submodule(PROJECTIONS[name])
                        original = module.forward

                        def project(value, module=module, original=original, name=name):
                            assert value.dtype == module.weight.dtype == torch.bfloat16
                            exact = F.linear(
                                value.double(),
                                module.weight.double(),
                                None if module.bias is None else module.bias.double(),
                            ).to(value.dtype)
                            native = original(value)
                            same_operand[name] = {
                                "changed_elements_by_row": (exact != native)
                                .reshape(case["length"], -1)
                                .sum(-1)
                                .tolist(),
                                "max_abs_error": (exact.float() - native.float()).abs().max().item(),
                                "operand_sha256": tensor_sha(value),
                            }
                            return exact

                        stack.enter_context(mock.patch.object(module, "forward", project))
                    actual = reference(
                        case["prompt"],
                        position_embeddings=(None, None),
                        past_key_values=DynamicCache(config=config),
                        ple_input_ids=case["tokens"],
                    )
                if not selected:
                    assert torch.equal(actual, case["reference"]["packed"]["output"])
                    assert torch.equal(captured["indices"], case["reference"]["packed"]["router_indices"])
                literal_metrics = metrics(case["expected"], actual, case["prompt"])
                record = {
                    "logical_length": case["length"],
                    "policy": policy,
                    "projections": [PROJECTIONS[name] for name in selected],
                    "packed_native_bit_exact": not selected,
                    "literal_metrics": literal_metrics,
                    "literal_passed": metric_passes(literal_metrics),
                    "packed_baseline_metrics": metrics(case["reference"]["packed"]["output"], actual, case["prompt"]),
                    "router_indices": captured["indices"].tolist(),
                    "same_operand_rounding": same_operand,
                    "device_comparisons": [],
                }
                if observed:
                    for item in observed["live_other_prefill"]:
                        if item["logical_length"] != case["length"]:
                            continue
                        assert item["prompt_sha256"] == tensor_sha(case["prompt"])
                        assert item["tokens_sha256"] == tensor_sha(case["tokens"])
                        assert item["literal_sha256"] == tensor_sha(case["expected"])
                        measured = metrics(actual, item["actual"], case["prompt"])
                        record["device_comparisons"].append(
                            {
                                "repetition": item["repetition"],
                                "metrics": measured,
                                "passed": metric_passes(measured),
                                "literal_metrics": metrics(case["expected"], item["actual"], case["prompt"]),
                            }
                        )
                report["cases"].append(record)
                tensors.append(
                    {
                        "logical_length": case["length"],
                        "policy": policy,
                        "output": actual,
                        "router_indices": captured["indices"],
                    }
                )
                print(json.dumps(record), flush=True)
    finally:
        hook.remove()
    report["actual_cache_parity_checks"] = lookup.parity_checks
    report["selected_rows"] = len(lookup.ids)
    args.tensor_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"identity": report, "cases": tensors}, args.tensor_output)
    report["tensor_output"] = str(args.tensor_output)
    report["tensor_sha256"] = file_sha(args.tensor_output)
    assert "ttnn" not in sys.modules
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
