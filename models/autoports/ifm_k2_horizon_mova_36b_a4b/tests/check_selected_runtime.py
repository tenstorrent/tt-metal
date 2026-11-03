# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks of observed default preparation operations and constructed tensors."""

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

from ..tt.precision import layer_options
from .check_datatype_policy import check, dtype

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/datatype_sweep"


def verify(proof):
    policy = proof["selected_policy"]
    count = check({**proof, "policy": policy})
    assert proof["sampling_input_dtype"] == dtype(policy["sampling_dtype"])
    assert proof["sampled_token_dtype"] == "DataType.UINT32"
    raw = gzip.decompress((ROOT / proof["operations"]["artifact"]).read_bytes())
    assert hashlib.sha256(raw).hexdigest() == proof["operations"]["uncompressed_sha256"]
    rows = json.loads(raw)
    assert len(rows) == proof["operations"]["rows"] > 0
    coverage = Counter()
    flags = Counter()
    projection_fields = {
        "attention_packed": "attention_dtype",
        "attention_decode": "attention_decode_dtype",
        "self_attn.o_proj": "output_weight_dtype",
        "mlp.gate_up": "mlp_dtype",
        "mlp.shared_router": "mlp_dtype",
        "mlp.down_proj": "down_weight_dtype",
        "mlp.shared_experts.down_proj": "down_weight_dtype",
    }
    for row in rows:
        op, fn = row["operation"], row["function"]
        inputs, output = row["inputs"], row["output"]
        options = layer_options(policy, row["layer"]) if row["layer"] is not None else {}

        def expect(field, actual, expected):
            nonlocal count
            assert actual == expected, (field, row)
            count += 1
            coverage[field] += row["count"]

        if row["compute_flags"] is not None:
            flags[
                json.dumps(
                    dict(function=fn, operation=op, fidelity=row["math_fidelity"], **row["compute_flags"]),
                    sort_keys=True,
                )
            ] += row["count"]
        if fn == "_linear" and op == "linear":
            name = row["role"]["name"]
            field = projection_fields[name]
            expect("weight_groups." + field, inputs[1]["dtype"], dtype(options[field]))
            group = "attention" if name.startswith(("attention", "self_attn")) else "mlp"
            expect("compute_fidelities." + group, row["math_fidelity"], "MathFidelity." + options[group + "_fidelity"])
        if op == "sparse_matmul":
            role = row["role"].get("role")
            value = role == "value" or fn == "_experts_ep_value"
            down = role == "down" or (fn == "mm" and inputs[1]["shape"][-2] != 2560)
            field = "value_dtype" if value else ("expert_down_dtype" if down else "experts_dtype")
            group = "value" if value else "experts"
            expect("weight_groups." + field, inputs[1]["dtype"], dtype(options[field]))
            expect("compute_fidelities." + group, row["math_fidelity"], "MathFidelity." + options[group + "_fidelity"])
            if fn == "_indexed_mm":
                activation = "value_activation" if value else "expert_activation"
            elif fn == "_experts_ep_value":
                activation = "ep_value_activation"
            elif fn == "mm":
                activation = "ep_expert_activation"
            else:
                activation = None if value else "expanded_expert_activation"
            if activation:
                expect("activations." + activation, output["dtype"], dtype(options[activation]))
            else:
                expect("fixed.expanded_value_single_tile", output["dtype"], dtype("bfloat16"))
        if fn == "_mlp" and op == "minimal_matmul":
            expect("weight_groups.mlp_dtype", inputs[1]["dtype"], dtype(options["mlp_dtype"]))
            expect("compute_fidelities.mlp", row["math_fidelity"], "MathFidelity." + options["mlp_fidelity"])
            expect(
                "activations.dense_prefill_hidden_dtype", output["dtype"], dtype(options["dense_prefill_hidden_dtype"])
            )
        if fn == "_norm" and op in ("mul", "typecast"):
            if op == "mul":
                expect("norm_dtype", inputs[1]["dtype"], dtype(policy["norm_dtype"]))
            field = "attention_activation" if row["role"]["name"] == "input_layernorm" else "mlp_activation"
            expect("activations." + field, output["dtype"], dtype(options[field]))
        if op == "rms_norm":
            if "weight" in row["tensor_kwargs"]:
                expect("norm_dtype", row["tensor_kwargs"]["weight"]["dtype"], dtype(policy["norm_dtype"]))
            expect("norm_fidelity", row["math_fidelity"], "MathFidelity." + policy["norm_fidelity"])
            if fn == "_norm" and row["caller"] == "tt/multichip_decoder.py":
                expect("activations.prefill_norm_dtype", inputs[0]["dtype"], dtype(options["prefill_norm_dtype"]))
        if fn == "_finish" and op == "typecast":
            expect("residual_dtype", output["dtype"], dtype(policy["residual_dtype"]))
        if op == "all_reduce_async":
            for value in inputs:
                expect("ccl_dtype", value["dtype"], dtype(policy["ccl_dtype"]))
        if op == "reduce_scatter_minimal_async":
            expect("prefill_ccl_dtype." + fn, inputs[0]["dtype"], dtype(policy["prefill_ccl_dtype"]))
        if op == "paged_fill_cache":
            for value in inputs[:2]:
                expect("kv_cache_dtype.fill", value["dtype"], dtype(policy["kv_cache_dtype"]))
        if op == "paged_fused_update_cache":
            for index in (0, 2):
                expect("kv_cache_dtype.update_storage", inputs[index]["dtype"], dtype(policy["kv_cache_dtype"]))
            for index in (1, 3):
                expect("fixed.cache_update_input", inputs[index]["dtype"], dtype("bfloat16"))
        if op in ("chunked_scaled_dot_product_attention", "paged_scaled_dot_product_attention_decode"):
            capacity = inputs[3]["shape"][-1] * inputs[1]["shape"][-2]
            phase = (
                "prefill"
                if op.startswith("chunked")
                else ("short" if capacity < 16384 else "middle" if capacity < 262144 else "long")
            )
            expect("sdpa_fidelities." + phase, row["math_fidelity"], "MathFidelity." + policy["sdpa_fidelities"][phase])
        if fn == "terminal" and op == "linear":
            expect("lm_head_dtype", inputs[1]["dtype"], dtype(policy["lm_head_dtype"]))
            expect("lm_head_fidelity", row["math_fidelity"], "MathFidelity." + policy["lm_head_fidelity"])
            expect("logits_dtype", output["dtype"], dtype(policy["logits_dtype"]))
    for field in (
        "lm_head_dtype",
        "lm_head_fidelity",
        "ccl_dtype",
        "kv_cache_dtype.fill",
        "residual_dtype",
        "activations.attention_activation",
        "activations.mlp_activation",
    ):
        assert coverage[field] > 0, field
    return dict(
        assertions=count,
        operation_rows=len(rows),
        operation_instances=sum(r["count"] for r in rows),
        executed_field_counts=dict(coverage),
        compute_flag_observations=[dict(**json.loads(key), count=value) for key, value in flags.items()],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--complete", action="store_true")
    args = parser.parse_args()
    policy = json.loads((DOC / "selected_precision_config.json").read_text())
    results = []
    for name in ("performance", "validation", "nonaligned", "contract"):
        path = DOC / f"selected_default_{name}_proof.json"
        if not path.exists() or json.loads(path.read_text())["status"] != "pass":
            continue
        proof = json.loads(path.read_text())
        assert proof["selected_policy"] == policy
        assert proof["environment"]["TT_METAL_TRACE_ALLOC_TRACKING"] == ("0" if name == "performance" else "1")
        assert proof["environment"]["TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"] == "0"
        assert proof["mesh"] == [1, 4]
        results.append(dict(mode=name, status="pass", artifact=path.name, **verify(proof)))
    if args.complete:
        assert len(results) == 4
    report = dict(
        status="pass" if len(results) == 4 else "partial; final jobs pending",
        config_id=policy["config_id"],
        results=results,
        scope={
            "embedding_norm_storage": "Constructed actual BF16 tensor dtypes checked; source consumers mapped in debug/precision_consumption_audit.md",
            "layer_exceptions": "Actual per-layer interleaved and separate decode banks plus executed projection weights; dense B1 decode bank differs from prefill alias",
            "prefill_value_activation": "Non-EP fallback only; not attributed to default results. Real-weight fallback exercised in debug/expert_aggregation.json",
            "sdpa_middle": "Constructed object and source capacity dispatch; not executed by default524288 or contract1024 capacities",
            "routing": "Fixed active value4/MLP8 and BF16 score/top-k interfaces; observed indexed shapes and source dispatch",
            "schema_max_context": "Required artifact metadata and constructor checks; physical capacity evidence is selected_validation.json",
            "timing": "Operations observed only during normal preparation; timed replay uninstrumented",
        },
    )
    (DOC / "selected_runtime_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print([(r["mode"], r["assertions"]) for r in results])


if __name__ == "__main__":
    main()
