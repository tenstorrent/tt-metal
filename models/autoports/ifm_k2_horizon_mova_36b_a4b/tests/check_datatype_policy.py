# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compare actual constructed weight/config objects against each measured precision policy."""

import json
from pathlib import Path

from ..tt.precision import layer_options

DOC = Path(__file__).resolve().parents[1] / "doc/datatype_sweep"


def dtype(value):
    return "DataType." + value.upper()


def check(record):
    policy, runtime = record["policy"], record["runtime"]
    assert runtime["embedding"] == dtype(policy["embedding_dtype"])
    assert runtime["head"] == dtype(policy["lm_head_dtype"])
    assert set(runtime["decode_head"]) == {dtype(policy["lm_head_dtype"])}
    assert runtime["head_fidelity"] == "MathFidelity." + policy["lm_head_fidelity"]
    if "head_in0_block_w" in runtime:
        assert runtime["head_in0_block_w"] == policy.get("lm_head_in0_block_w", 1)
    assert runtime["norm_fidelity"] == "MathFidelity." + policy["norm_fidelity"]
    assert record["cache_dtypes"] == [dtype(policy["kv_cache_dtype"])]
    groups = {
        "attention_packed.weight": "attention_dtype",
        "self_attn.o_proj.weight": "output_weight_dtype",
        "mlp.gate_up.weight": "mlp_dtype",
        "mlp.shared_router.weight": "mlp_dtype",
        "mlp.down_proj.weight": "down_weight_dtype",
        "mlp.shared_experts.down_proj.weight": "down_weight_dtype",
        "mlp.swiglu.weight": "mlp_dtype",
        "value_experts": "value_dtype",
        "value_ep": "value_dtype",
        "experts.gate_up": "experts_dtype",
        "experts.ep_gate_up": "experts_dtype",
        "experts.down_proj": "expert_down_dtype",
        "experts.ep_down": "expert_down_dtype",
        "input_layernorm.weight": "norm_dtype",
        "post_attention_layernorm.weight": "norm_dtype",
    }
    checks = 7
    for layer in runtime["layers"]:
        options = layer_options(policy, layer["index"])
        for name, field in groups.items():
            if name in layer["weights"]:
                assert layer["weights"][name] == dtype(options[field]), (layer["index"], name, field)
                checks += 1
            if name in layer["decode_weights"]:
                assert layer["decode_weights"][name] == dtype(options[field]), (layer["index"], name, field)
                checks += 1
        # The interleaved attention_decode alias deliberately reuses the prefill bank;
        # B1 decode consumes the separate DRAM-sharded decode_weights bank.
        if "attention_decode.weight" in layer["decode_weights"]:
            assert layer["decode_weights"]["attention_decode.weight"] == dtype(options["attention_decode_dtype"])
            checks += 1
        for group in policy["compute_fidelities"]:
            assert layer["fidelities"][group] == "MathFidelity." + options[group + "_fidelity"]
            checks += 1
        for phase, fidelity in policy["sdpa_fidelities"].items():
            assert layer["sdpa_fidelities"][phase] == "MathFidelity." + fidelity
            checks += 1
        assert layer["ccl"] == dtype(policy["ccl_dtype"])
        if "ccl_scratch" in layer:
            assert layer["ccl_scratch"] == [dtype(policy["ccl_dtype"])] * 2
        assert layer["cache"] == dtype(policy["kv_cache_dtype"])
        checks += 2
    return checks


def main():
    results = []
    for path in sorted((DOC / "runs").glob("*.json")):
        record = json.loads(path.read_text())
        if record["status"] not in ("pass", "accuracy_fail"):
            continue
        results.append(
            dict(
                config_id=record["policy"]["config_id"],
                assertions=check(record),
                status="pass",
                artifact=str(path.relative_to(DOC)),
            )
        )
    (DOC / "runtime_policy_check.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"Validated actual tensors and compute objects for {len(results)} full-model policies")


if __name__ == "__main__":
    main()
