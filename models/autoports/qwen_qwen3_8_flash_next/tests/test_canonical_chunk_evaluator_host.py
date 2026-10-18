# SPDX-License-Identifier: Apache-2.0
"""Synthetic CPU unit fixtures only; never creates serving readiness evidence."""
import copy
import importlib.util
import unittest
from pathlib import Path

SOURCE = Path(__file__).with_name("evaluate_vllm_canonical_chunks.py")
SPEC = importlib.util.spec_from_file_location("canonical_chunk_evaluator", SOURCE)
EVALUATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATOR)


def lifecycle():
    rows = []
    for index, label in enumerate(sorted(EVALUATOR.LABELS)):
        length = int(label.rsplit("_", 1)[1])
        tokens = [1] * 96
        if label == "mixed_513":
            tokens[60:] = [2] * 36
        rows.append(
            dict(
                label=label,
                prompt_length=length,
                prompt_token_ids=[7] * length,
                token_ids=tokens,
                output_tokens=96,
                sampling="sample" if label == "sample_130" else "greedy",
                start=float(index * 10),
                end=float(index * 10 + 5),
                usage=dict(prompt_tokens=length, completion_tokens=96),
            )
        )
    failed = dict(
        status="failed",
        requests=copy.deepcopy(rows),
        completed_requests=13,
        exact_greedy_differences=[dict(label="mixed_513", first_difference=60, control_token=1, candidate_token=2)],
    )
    return rows, failed


def trace():
    events, ids = [], {}
    for i, signature in enumerate(sorted(EVALUATOR.SIGNATURES)):
        ids[signature] = f"trace{i}"
        events += [
            dict(event="preparation", signature=signature, time=i, pid=42),
            dict(event="capture", signature=signature, trace_id=ids[signature], time=10 + i, pid=42),
        ]
    events.append(dict(event="prepared", time=20, pid=42, batch=32, traces=ids, buckets=[1, 2, 4, 8]))
    events.append(dict(event="replay", time=30, pid=42, signature="decode", trace_id=ids["decode"]))
    return events


class CanonicalChunkHostTests(unittest.TestCase):
    def reject(self, function, *args, **kwargs):
        try:
            function(*args, **kwargs)
        except ValueError:
            return
        self.fail("Malformed evidence was accepted")

    def test_preserves_original_failure_and_all_other_exact_comparisons(self):
        rows, failed = lifecycle()
        _, differences, unchanged = EVALUATOR.lifecycle_contract(rows, failed)
        self.assertEqual(differences, failed["exact_greedy_differences"])
        self.assertEqual(len(unchanged), 6)
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(rows, failed["requests"])

    def test_extra_mismatch_is_not_waived(self):
        rows, failed = lifecycle()
        next(r for r in rows if r["label"] == "mixed_129")["token_ids"][2] = 8
        failed["requests"] = copy.deepcopy(rows)
        self.reject(EVALUATOR.lifecycle_contract, rows, failed)

    def test_prompt_count_and_sampling_changes_fail(self):
        for field, value in (("prompt_token_ids", [8] * 513), ("token_ids", [1] * 95), ("sampling", "host")):
            rows, failed = lifecycle()
            next(r for r in rows if r["label"] == "mixed_513")[field] = value
            failed["requests"] = copy.deepcopy(rows)
            self.reject(EVALUATOR.lifecycle_contract, rows, failed)

    def test_missing_phase_or_relabelled_failure_fails(self):
        rows, failed = lifecycle()
        self.reject(EVALUATOR.lifecycle_contract, rows[:-1], failed)
        failed["status"] = "pass"
        self.reject(EVALUATOR.lifecycle_contract, rows, failed)

    def test_chunk_signature_requires_full_contiguous_coverage(self):
        row = dict(start=0, end=1000, prompt_length=513)
        events = [dict(event="prefill_schedule", time=1, ends=[513], physical_slots=[2])]
        events += [
            dict(
                event="prefill_chunk",
                time=2 + i,
                chunk_length=8 if i < 64 else 1,
                start_positions=[i * 8],
                slots=[2],
                active=[True],
            )
            for i in range(65)
        ]
        chunks = EVALUATOR.chunks_for_request(events, row)
        self.assertEqual(len(chunks), 65)
        self.assertEqual(chunks[-1], dict(start=512, length=1, slot=2))
        events[20]["start_positions"] = [999]
        self.reject(EVALUATOR.chunks_for_request, events, row)

    def test_trace_identity_retirement_and_recapture_fail_closed(self):
        events = trace()
        self.assertEqual(set(EVALUATOR.trace_contract(events, pid=42)["trace_ids"]), EVALUATOR.SIGNATURES)
        for mutation in ("recapture", "wrong_id", "retire", "wrong_pid"):
            changed = copy.deepcopy(events)
            if mutation == "recapture":
                changed.append(copy.deepcopy(changed[1]))
            elif mutation == "wrong_id":
                changed[-1]["trace_id"] = "unknown"
            elif mutation == "retire":
                changed.append(dict(event="retire", time=31, pid=42))
            else:
                changed[-1]["pid"] = 43
            self.reject(EVALUATOR.trace_contract, changed, pid=42)

    def test_token_comparison_detects_truncation_and_first_difference(self):
        self.assertIsNone(EVALUATOR.first_difference([1, 2], [1, 2]))
        self.assertEqual(EVALUATOR.first_difference([1, 2], [1]), 1)
        self.assertEqual(EVALUATOR.first_difference([1, 2], [1, 3]), 1)

    def test_incomplete_reduced_untracked_or_unclean_probe_fails(self):
        probe = dict(
            status="completed",
            matched_live_token_status="pass",
            closed_mesh=True,
            selected_layers="all",
            args=dict(layers="all", batch=32, steps=95),
            context=262144,
            tracking={"TT_METAL_TRACE_ALLOC_TRACKING": "1", "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE": "0"},
        )
        EVALUATOR.completed_full_probe_contract(probe)
        for key, value in (
            ("status", "running"),
            ("matched_live_token_status", "failed"),
            ("closed_mesh", False),
            ("selected_layers", "0,1,3"),
            ("args", dict(layers="all", batch=3, steps=95)),
            ("args", dict(layers="all", batch=32, steps=32)),
            ("context", 2048),
            ("tracking", {}),
        ):
            changed = copy.deepcopy(probe)
            changed[key] = value
            self.reject(EVALUATOR.completed_full_probe_contract, changed)


if __name__ == "__main__":
    unittest.main()
