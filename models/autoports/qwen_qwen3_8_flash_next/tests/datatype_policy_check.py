"""Host-only validation of selected JSON against constructed and captured evidence."""
import argparse
import json
from pathlib import Path

ROOT = Path("models/autoports/qwen_qwen3_8_flash_next")
OUT = ROOT / "doc/datatype_sweep"


def check(report, policy, *, require_default=True, serving=False):
    runtime = report["runtime"]
    assert runtime["policy"] == policy
    if require_default and not serving:
        assert report["args"]["config"] is None, "Qualification must use default construction"
    assert len(runtime["layers"]) == 48
    assert runtime["context"] == policy["max_context"] == 262144
    assert runtime["routing"] == {key: policy["routing"][key] for key in ("active_experts_per_token", "experts_total")}
    assert policy["activation_dtype"] == policy["residual_dtype"]
    checked = []
    omitted = []

    def dtype(actual, expected, label):
        assert actual == "DataType." + expected.upper(), (label, actual, expected)
        checked.append(label)

    def fidelity(actual, expected, label):
        assert actual == "MathFidelity." + expected, (label, actual, expected)
        checked.append(label)

    def row(actual, group, weights, computes, label):
        if serving and report["batch"] > 1 and actual["dtype"] is None:
            assert label.endswith((".decode", ".output")), label
            fidelity(actual["compute"]["math_fidelity"], computes[group], label + ".fidelity")
            omitted.append(label)
            return
        dtype(actual["dtype"], weights[group], label + ".weight")
        fidelity(actual["compute"]["math_fidelity"], computes[group], label + ".fidelity")

    for layer in runtime["layers"]:
        index = layer["layer"]
        prefix = f"layer{index}"
        weights = dict(policy["weight_groups"])
        computes = dict(policy["compute_fidelities"])
        exception = policy["layer_exceptions"].get(str(index), {})
        weights.update(exception.get("weight_groups", {}))
        computes.update(exception.get("compute_fidelities", {}))
        gdn = layer["kind"] == "linear_attention"
        if gdn:
            row(layer["input"], "gdn_input", weights, computes, prefix + ".input")
            fidelity(layer["recurrence_fidelity"], computes["gdn_recurrence"], prefix + ".recurrence")
            for state in ("convolution", "recurrent"):
                dtype(layer["state"][state], policy["state_dtypes"][state], prefix + "." + state)
        else:
            for key, group in (("prefill", "qsa_prefill"), ("decode", "qsa_decode")):
                row(layer[key], group, weights, computes, prefix + "." + key)
            dtype(layer["kv"], policy["kv_cache_dtype"], prefix + ".kv")
            dtype(layer["raw_index"], policy["state_dtypes"]["raw_index"], prefix + ".raw_index")
            fidelity(layer["attention_fidelity"], computes["qsa_attention"], prefix + ".attention")
            fidelity(
                layer["packed_singleton_fidelity"],
                computes[policy["qsa_execution"]["singleton_packed_fidelity_group"]],
                prefix + ".packed_singleton",
            )
        for key, group in (
            ("output", "gdn_output" if gdn else "qsa_output"),
            ("output_row", "gdn_output" if gdn else "qsa_output"),
            ("shared", "shared_expert"),
            ("shared_down", "shared_expert"),
            ("router", "router"),
            ("hc", "hc"),
            ("hc_up", "hc"),
            ("mlp_hc", "hc"),
            ("mlp_hc_up", "hc"),
        ):
            row(layer[key], group, weights, computes, prefix + "." + key)
        for group, key in (
            ("routed_expert_gate_up", "gate_dtypes"),
            ("routed_expert_gate_up", "up_dtypes"),
            ("routed_expert_down", "down_dtypes"),
        ):
            assert layer["experts"][key] == ["DataType." + weights[group].upper()]
            fidelity(layer["experts"]["fidelity"], computes[group], prefix + "." + group)
        assert layer["experts"]["local_experts"] * 4 == policy["routing"]["experts_total"]
        assert layer["ccl"] == policy["ccl_dtype"]
        fidelity(layer["norms"]["fidelity"], computes["norms"], prefix + ".norms")
        for key in ("attention_hc", "mlp_hc"):
            dtype(layer["norms"][key], "bfloat16", prefix + ".norms." + key)
        if "ple" in layer:
            row(layer["ple"], "ple_projection", weights, computes, prefix + ".ple")
            assert layer["ple_table_dtypes"] == ["DataType." + weights["ple_tables"].upper()]
            for key, value in layer["norms"]["ple"].items():
                dtype(value, "float32", prefix + "." + key)
    weights, computes = policy["weight_groups"], policy["compute_fidelities"]
    row(runtime["head"], "lm_head", weights, computes, "head")
    dtype(runtime["embedding"], weights["embedding"], "embedding")
    dtype(runtime["residual_dtype"], policy["residual_dtype"], "residual")
    dtype(runtime["logits_dtype"], policy["logits_dtype"], "logits")
    for key, value in runtime["terminal_mixer"].items():
        dtype(value, weights["terminal_mixer"], "terminal." + key)
    fidelity(runtime["terminal_fidelity"], computes["terminal_mixer"], "terminal.fidelity")
    sampler = report["generator_runtime"]
    dtype(sampler["logits_dtype"], policy["sampling"]["logits"], "sampler.logits")
    dtype(sampler["token_ids_dtype"], policy["sampling"]["token_ids"], "sampler.tokens")
    assert sampler["max_top_k"] == policy["sampling"]["max_top_k"]
    for key, values in sampler["parameter_dtypes"].items():
        dtype(values[1], policy["sampling"]["scores"], "sampler." + key + ".p")
        dtype(values[2], policy["sampling"]["scores"], "sampler." + key + ".temperature")
    capture = [r for r in report["captured_native_arguments"] if r["phase"] == "capture"]
    if omitted:
        assert not any(r.get("group") in ("attention.decode_projection", "output.column_weight") for r in capture)
    assert {r["op"] for r in capture} >= {
        "linear",
        "all_gather_async",
        "reduce_scatter_minimal_async",
        "sparse_sdpa",
        "unified_routed_expert_moe",
    }
    for record in capture:
        if record["op"] == "sparse_sdpa":
            dtype(record["kv_dtype"], policy["kv_cache_dtype"], "captured.kv")
            expected_format = "BF16" if policy["kv_cache_dtype"] == "bfloat16" else "FP8_E4M3"
            assert record["kv_format"].split(".")[-1] == expected_format
        elif record["op"] == "unified_routed_expert_moe":
            dtype(record["gate_dtype"], weights["routed_expert_gate_up"], "captured.expert.gate")
            dtype(record["down_dtype"], weights["routed_expert_down"], "captured.expert.down")
            fidelity(record["compute"]["math_fidelity"], computes["routed_expert_gate_up"], "captured.expert.fidelity")
        elif record["op"] == "linear" and record.get("group") == "terminal.lm_head":
            dtype(record["weight_dtype"], weights["lm_head"], "captured.head")
            fidelity(record["compute"]["math_fidelity"], computes["lm_head"], "captured.head.fidelity")
        elif record["op"] == "linear" and record["layers"]:
            group_name = record["group"]
            for index in record["layers"]:
                gdn = (index + 1) % 4 != 0
                groups = {
                    "attention.packed_weight": "gdn_input",
                    "attention.packed_projection": "qsa_prefill",
                    "attention.decode_projection": "qsa_decode",
                    "output.row_weight": "gdn_output" if gdn else "qsa_output",
                    "output.column_weight": "gdn_output" if gdn else "qsa_output",
                    "mlp.shared_gu": "shared_expert",
                    "mlp.shared_down": "shared_expert",
                    "mlp.router": "router",
                    "ple.projection": "ple_projection",
                    "attn_hc.read_weight": "hc",
                    "attn_hc.up": "hc",
                    "mlp_hc.read_weight": "hc",
                    "mlp_hc.up": "hc",
                }
                group = groups[group_name]
                local_weights = dict(weights)
                local_computes = dict(computes)
                exception = policy["layer_exceptions"].get(str(index), {})
                local_weights.update(exception.get("weight_groups", {}))
                local_computes.update(exception.get("compute_fidelities", {}))
                compute_group = group
                if group_name == "attention.packed_projection" and record["input_shape"][-2] == 1:
                    compute_group = policy["qsa_execution"]["singleton_packed_fidelity_group"]
                label = f"captured.layer{index}.{group_name}"
                dtype(record["weight_dtype"], local_weights[group], label + ".weight")
                fidelity(record["compute"]["math_fidelity"], local_computes[compute_group], label + ".fidelity")
    if policy["ccl_dtype"] == "bfloat8_b":
        assert any(
            r["op"] in ("all_gather_async", "reduce_scatter_minimal_async") and r["input_dtype"] == "DataType.BFLOAT8_B"
            for r in capture
        )
    if serving:
        assert report["context"] == runtime["context"]
        assert report["public_pool_tokens"] >= report["context"]
        assert report["counters"]["captures"] == len(report["trace_ids"])
        assert {"decode", "greedy", "sample"} <= set(report["trace_ids"])
    else:
        assert all(
            r["counter_delta"]["decode_replays"] == 99
            and r["counter_delta"].get("captures", 0) == 0
            and r["counter_delta"].get("releases", 0) == 0
            for r in report["teacher_forcing_repetitions"]
        )
    return dict(
        status="pass",
        config_id=policy["config_id"],
        checks=len(checked),
        constructed_layers=48,
        captured_argument_rows=len(capture),
        omitted_unused_singleton_weights=omitted,
        default_path=(
            "serving allocate_kv_cache -> QwenFlashNextModel -> load_policy(None) -> required selected_precision_config.json"
            if serving
            else "build_generator -> QwenFlashNextModel -> load_policy(None) -> required selected_precision_config.json"
            if require_default
            else "explicit candidate; default qualification pending"
        ),
        fixed_contracts="Unsupported activation/residual, native expert storage, state, router, norms and sampling choices are rejected before construction; captured/native intermediate formats remain explicit in README.",
        serving_path=(
            report["adapter"]
            if serving
            else "No vLLM adapter exists at this stage. Future callers of build_generator or the model constructor inherit the same required default artifact."
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--serving", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    policy = json.loads((OUT / "selected_precision_config.json").read_text())
    if args.serving and args.output is None:
        parser.error("--serving requires --output to preserve datatype-stage evidence")
    result = check(json.loads(args.report.read_text()), policy, serving=args.serving)
    result["report"] = str(args.report)
    (args.output or OUT / "selected_policy_propagation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
