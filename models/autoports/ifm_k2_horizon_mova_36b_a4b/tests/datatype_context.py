# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Recompute precision capacity against measured persistent allocations."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/datatype_sweep"


def allocated(view):
    return view["num_banks"] * view["total_bytes_allocated_per_bank"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--update-selected", action="store_true")
    args = ap.parse_args()
    contract = json.loads((ROOT / "doc/context_contract.json").read_text())
    baseline = json.loads((DOC / "baseline.json").read_text())
    validation = DOC / "selected_validation.json"
    validation_proof = DOC / "selected_default_validation_proof.json"
    plans = []
    for p in sorted((DOC / "runs").glob("*.json")):
        d = json.loads(p.read_text())
        if "constructed_memory" not in d:
            continue
        delta = allocated(d["constructed_memory"]) - allocated(baseline["constructed_memory"])
        kv = d["policy"]["kv_cache_dtype"]
        tile = {"bfloat16": 2048, "bfloat8_b": 1088, "bfloat4_b": 576}[kv]
        planned = contract["optimized_full_model"]["planned_total_bytes_per_device"] + delta
        physical = contract["full_model"]["physical_allocator_bytes"]
        plans.append(
            dict(
                config_id=d["policy"]["config_id"],
                kv_cache_dtype=kv,
                target_supported_context=524288,
                aggregate_cached_token_budget=524288,
                max_batch=32,
                capability_reduction=None,
                kv_bytes_per_device=48 * 2 * 524288 * 2 * 128 * tile // 1024,
                measured_persistent_delta_bytes=delta,
                planned_total_bytes_per_device=planned,
                planned_headroom_bytes_per_device=physical - planned,
                physical_allocator_bytes=physical,
                constructed_bytes_per_device=allocated(d["constructed_memory"]),
                captured_bytes_per_device=allocated(d["captured_memory"]) if "captured_memory" in d else None,
                reserve_policy="Retain Stage7 conservative workspace/trace/output reservation, adjust measured persistent allocation delta",
                evidence=str(p.relative_to(ROOT)),
                candidate_status=d["status"],
                status="capacity calculated; no selected-policy validation claimed",
            )
        )
    (DOC / "context_plans.json").write_text(json.dumps(plans, indent=2) + "\n")
    if args.update_selected:
        policy = json.loads((DOC / "selected_precision_config.json").read_text())
        plan = next(p for p in plans if p["config_id"] == policy["config_id"])
        plan["selected_precision_config"] = "doc/datatype_sweep/selected_precision_config.json"
        plan["status"] = "calculated; selected capability validation required"
        if validation.exists() and validation_proof.exists():
            checked = json.loads(validation.read_text())
            proof = json.loads(validation_proof.read_text())
            assert proof["selected_policy"] == policy and proof["status"] == "pass"
            assert checked["status"] == "passed" and checked["context"] == 524288
            assert checked["nonaligned_capacity"]["prompt_length"] == 524275
            assert len(checked["nonaligned_capacity"]["generated_tokens"]) == 13
            assert checked["full_extent_continuation"]["end"] == 524288
            assert checked["last_position_trace_smoke"]["position"] == 524287
            plan.update(
                status="validated; independent final stage review required",
                validated_supported_context=524288,
                largest_tested_prefill_context=524288,
                largest_tested_traced_decode_position=524287,
                full_context_tested_batch=1,
                physical_allocated_after_capacity_bytes_per_device=allocated(checked["final_memory"]),
                physical_free_after_capacity_bytes_per_device=(
                    checked["final_memory"]["num_banks"] * checked["final_memory"]["total_bytes_free_per_bank"]
                ),
                capacity_evidence=str(validation.relative_to(ROOT)),
                default_construction_evidence=str(validation_proof.relative_to(ROOT)),
                nonaligned_evidence="doc/datatype_sweep/selected_nonaligned.json",
                model_limit="524288 is the checkpoint's advertised maximum; no reduction from the accepted contract",
            )
        nonaligned = DOC / "selected_nonaligned.json"
        batch_contract = DOC / "selected_contract_b32_p256.json"
        continuation_accuracy = DOC / "selected_continuation_accuracy.json"
        if all(p.exists() for p in (nonaligned, batch_contract, continuation_accuracy)):
            boundary = json.loads(nonaligned.read_text())
            batch = json.loads(batch_contract.read_text())
            accuracy = json.loads(continuation_accuracy.read_text())
            assert boundary["status"] == "passed" and boundary["changed_inputs"]["bitwise_equal"]
            assert batch["batch"] == 32 and batch["external_cache"] and batch["page_size"] == 256
            assert batch["precision_config_id"] == policy["config_id"]
            assert batch["continuation"]["trace_ids_unchanged"]
            assert accuracy["status"] == "pass" and accuracy["policy"] == policy
            for label in ("nonaligned", "contract"):
                proof = json.loads((DOC / f"selected_default_{label}_proof.json").read_text())
                assert proof["status"] == "pass" and proof["selected_policy"] == policy
            plan.update(
                status="all selected capability checks validated; independent final review required",
                nonaligned_trace_oracle="bitwise trace/eager with identical segmentation; cross-segmentation diagnostics retained",
                continuation_accuracy_evidence=str(continuation_accuracy.relative_to(ROOT)),
                batch_contract_evidence=str(batch_contract.relative_to(ROOT)),
                batch_contract_scope="real layers0/3,32slots,page256,context1024; full aggregate external pool plus32output pages",
            )
        review = DOC / "stage_review_final.md"
        if review.exists() and "Verdict: clean-pass" in review.read_text():
            plan.update(
                status="validated; independent stage review clean-pass", stage_review=str(review.relative_to(ROOT))
            )
        contract["datatype_sweep"] = plan
        (ROOT / "doc/context_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
        (DOC / "context_plans.json").write_text(json.dumps(plans, indent=2) + "\n")


if __name__ == "__main__":
    main()
