# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only acceptance checks for selected precision and measured runtime fields."""

import hashlib
import json
import statistics
from pathlib import Path

import torch

from models.autoports.qwen_qwen3_8_27b.tt.precision import ROLES, decoder_policy, load_precision

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/datatype_sweep"


def main():
    for key in (0, "00"):
        invalid = load_precision("baseline")
        invalid["layer_exceptions"] = {key: {"weight_groups": {"attention": "bfloat8_b"}}}
        try:
            load_precision(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Ignored noncanonical layer key {key!r}")
    valid = load_precision("baseline")
    valid["layer_exceptions"] = {"0": {"weight_groups": {"attention": "bfloat8_b"}}}
    assert decoder_policy(load_precision(valid), 0)["attention_dtype"] == "bfloat8_b"
    selected = load_precision()
    artifact = json.loads((DOC / "selected_precision_config.json").read_text())
    assert selected == artifact, "Normal constructor policy differs from selected artifact"
    results = json.loads((DOC / "sweep_results.json").read_text())["results"]
    passing = [r for r in results if r["status"] == "pass"]
    fastest = max(passing, key=lambda r: r["traced_teacher_forcing_decode_t_s_u"])
    assert fastest["config_id"] == selected["config_id"]
    winners = [r for r in passing if r["config_id"] == selected["config_id"]]
    confirmation = json.loads((DOC / "selected_confirmation.json").read_text())
    assert confirmation["status"] == "pass"
    assert confirmation["runtime"]["policy"] == selected
    reproduced = statistics.median(p["tokens_per_second"] for p in confirmation["warmed_teacher_forcing_perf"][-2:])
    assert abs(reproduced / fastest["traced_teacher_forcing_decode_t_s_u"] - 1) < 0.02
    state = confirmation["runtime"]["allocated_state"]
    assert state["logits_dtype"].lower().endswith(selected["logits_dtype"])
    assert state["token_dtype"].lower().endswith(selected["token_dtype"])
    for layer in state["layers"]:
        for name, tensor in layer["tensors"].items():
            field = (
                "kv_cache_dtype"
                if name in ("key", "value")
                else "convolution_dtype"
                if name == "conv"
                else "recurrent_dtype"
            )
            assert tensor["dtype"].lower().endswith(selected[field])
    for result in winners:
        raw = json.loads((ROOT / result["artifact"]).read_text())
        runtime = raw["runtime"]
        assert runtime["policy"] == selected
        assert runtime["context"] == selected["max_context"] == 262144
        assert len(runtime["layers"]) == 64
        for layer in runtime["layers"]:
            expected = decoder_policy(selected, layer["layer"])
            assert all(layer["policy"][k] == v for k, v in expected.items())
            for role in ROLES:
                assert layer["compute"][role]["math_fidelity"].endswith(expected[role + "_fidelity"])
                assert layer["compute"][role]["fp32_dest_acc_en"] == selected["fp32_dest_acc_en"]
            for weight in layer["actual_weights"].values():
                assert weight["dtype"].lower().endswith(expected[weight["role"] + "_dtype"])
        assert runtime["head_dtype"].lower().endswith(selected["weight_groups"]["head"])
        assert runtime["head_compute"]["math_fidelity"].endswith(selected["compute_fidelities"]["head"])
    for role in (*ROLES, "head"):
        considered = [r for r in results if r["dtype_policy"]["weight_groups"][role] == "bfloat4_b"]
        if considered:
            assert any(r["compute_fidelity_policy"][role] == "LoFi" for r in considered), role
    contract = json.loads((ROOT / "doc/context_contract.json").read_text())
    assert contract["datatype_sweep"]["kv_dtype"] == selected["kv_cache_dtype"]
    assert contract["datatype_sweep"]["supported_context"] == selected["max_context"]
    token_out = json.loads((DOC / "selected_token_out.json").read_text())
    assert token_out["precision_policy"] == selected
    assert json.loads((DOC / "selected_readiness.json").read_text())["precision_policy"] == selected
    steady = token_out["queued_token_out"]["steady_state_counters"]
    assert steady == {"model_replays": 127, "sampling_replays": 127}
    assert token_out["full"] and len(token_out["layers"]) == 64
    assert token_out["queued_token_out"]["final_token_matches_immediate"]
    for name in ("top1_perf_pareto.png", "top5_perf_pareto.png"):
        assert (DOC / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    reference_path = ROOT / "readiness_aime24_chat.refpt"
    entry = torch.load(reference_path, weights_only=False)["entries"][0]
    greedy = entry["generated_tokens"].reshape(-1).tolist()
    topk = entry["topk_tokens"].tolist()
    audits = []
    for path in sorted(DOC.glob("*.json")):
        raw = json.loads(path.read_text())
        if not isinstance(raw, dict) or "teacher_forced_tokens" not in raw:
            continue
        predicted = raw["teacher_forced_tokens"]
        expected = raw["teacher_forcing_runs"][-1]["accuracy"][0]
        computed = {
            f"top{k}": sum(token in topk[i][:k] for i, token in enumerate(predicted)) / len(predicted)
            for k in (1, 5, 100)
        }
        assert all(computed[k] == expected[k] for k in computed), path
        audits.append(
            dict(
                artifact=path.name,
                computed=computed,
                reported={k: expected[k] for k in computed},
                greedy_sequence_match=sum(x == y for x, y in zip(predicted, greedy)) / len(predicted),
            )
        )
    audit = dict(
        status="pass",
        reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        reference_greedy_top1_differences=[
            dict(index=i, greedy=token, top1=topk[i][0], top5=topk[i][:5])
            for i, token in enumerate(greedy)
            if token != topk[i][0]
        ],
        metric_contract="Standard readiness scores stored topk_tokens; feedback uses generated_tokens. HF argmax/topk tie ordering differs at one position.",
        rows=audits,
    )
    (DOC / "reference_topk_order_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    report = dict(
        status="pass",
        selected_config=selected["config_id"],
        candidates=len(results),
        default_confirmation_traced_teacher_forcing_t_s_u=reproduced,
        construction_path="build_generator -> QwenModel -> load_precision (selected artifact by default)",
        serving_scope="No vLLM adapter exists in this stage; subsequent adapter construction inherits this shared default.",
    )
    (DOC / "propagation_check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
