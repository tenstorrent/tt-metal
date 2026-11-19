"""CPU controls for capture accounting and mutation-sensitive trace evidence."""

import unittest
from types import SimpleNamespace

import torch

from models.autoports.qwen_qwen3_8_flash_next.tests.probe_full_model_batch_hf import (
    FactoredTraceEvidence,
    run_resident_feedback,
)


class BatchEvidenceTest(unittest.TestCase):
    def fixture(self, fail=False):
        stats = {"allocated": 100, "programs": 7}
        owner = SimpleNamespace(
            metadata_uploads=3, slot_count=2, buckets=(1, 2), active_slots=(0, 1), _plan=(), _plan_slots=None
        )
        gen = SimpleNamespace(
            factored_decode=owner,
            traces={},
            sampling_traces={False: 100},
            prefill_traces={32: 200},
            model=SimpleNamespace(layers=[0]),
            counters={"captures": 0},
        )
        keys = [
            ("clear",),
            ("attention", 0, 0),
            ("attention", 0, 1),
            ("moe", 0, 1),
            ("moe", 0, 2),
            ("terminal", 0),
            ("terminal", 1),
        ]

        def capture():
            for key in keys:
                gen.traces[key] = len(gen.traces) + 1
                gen.counters["captures"] += 1
            stats["allocated"] += 25
            owner.metadata_uploads += 2
            if fail:
                raise RuntimeError("controlled_capture_failure")

        owner.capture = capture
        result = {}
        evidence = FactoredTraceEvidence(
            gen,
            lambda: {"num_banks": 8, "total_bytes_allocated_per_bank": stats["allocated"]},
            lambda: stats["programs"],
            lambda: None,
            result,
        )
        return evidence, gen, owner, stats, result

    def test_capture_delta_uses_per_rank_bytes_and_counts_roles(self):
        evidence, gen, owner, _, result = self.fixture()
        owner.capture()
        evidence.prepared()
        self.assertEqual(result["capture_windows"][0]["retained_capture_bytes_per_rank"], 200)
        self.assertEqual(result["native_capture_count"], 7)
        self.assertEqual(result["role_capture_counts"], {"clear": 1, "attention": 2, "moe": 2, "terminal": 2})
        self.assertEqual(result["metadata_warmup_uploads"], 3)
        self.assertEqual(result["metadata_capture_uploads"], 2)

    def test_failed_capture_retains_partial_memory_and_ids(self):
        evidence, gen, owner, _, result = self.fixture(fail=True)
        with self.assertRaisesRegex(RuntimeError, "controlled_capture_failure"):
            owner.capture()
        window = result["capture_windows"][0]
        self.assertEqual(window["retained_capture_bytes_per_rank"], 200)
        self.assertEqual(window["native_capture_count"], 7)

    def test_unchanged_migration_plan_and_shrink_upload_only_once(self):
        evidence, gen, owner, _, result = self.fixture()
        owner.capture()
        evidence.prepared()
        owner._plan_slots = (0, 1)
        owner._plan = (1, 2, 3)
        before = evidence.before()
        evidence.after(before, (0, 1), repeat=1, step=32, migration=True)
        self.assertEqual(result["membership_events"][-1]["metadata_upload_delta"], 0)
        self.assertEqual(result["unchanged_plan_identity_reuses"], 1)
        before = evidence.before()
        owner.active_slots = (1,)
        owner.metadata_uploads += 1
        owner._plan_slots = (1,)
        owner._plan = (1, 3)
        evidence.after(before, (1,), repeat=1, step=68)
        self.assertEqual(result["request_metadata_uploads"], 1)
        evidence.after(evidence.before(), (), repeat=1, step="all_inactive")

    def test_each_forbidden_mutation_is_detected(self):
        for mutation, message in (
            (lambda g, o, s: g.traces.__setitem__(("clear",), 999), "Native model IDs"),
            (lambda g, o, s: s.__setitem__("programs", 8), "New program"),
            (lambda g, o, s: setattr(o, "metadata_uploads", o.metadata_uploads + 1), ""),
            (lambda g, o, s: setattr(o, "_plan", tuple([1, 2, 3])), "dispatch plan"),
        ):
            with self.subTest(message=message):
                evidence, gen, owner, stats, _ = self.fixture()
                owner.capture()
                evidence.prepared()
                owner._plan_slots = (0, 1)
                owner._plan = tuple([1, 2, 3])
                before = evidence.before()
                mutation(gen, owner, stats)
                with self.assertRaisesRegex(AssertionError, message):
                    evidence.after(before, (0, 1), repeat=0, step=0)


class ResidentFeedbackTest(unittest.TestCase):
    def fixture(self, defect=None):
        counters = dict(
            token_refreshes=0,
            position_refreshes=0,
            page_table_refreshes=0,
            captures=0,
            releases=0,
            model_replays=0,
            sampling_replays=0,
        )
        gen = SimpleNamespace(
            max_batch_size=4,
            counters=counters,
            caches=[SimpleNamespace(position=17 + slot) for slot in range(4)],
            model=SimpleNamespace(config=SimpleNamespace(vocab_size=1000)),
        )
        feedback = [100 + slot for slot in range(4)]
        positions = {(slot, layer): 17 + slot for slot in range(4) for layer in (3, 7)}
        consumed = []

        def decode(tokens, start_pos, **kwargs):
            self.assertTrue((tokens == -12345).all())
            self.assertFalse(kwargs["reload_inputs"])
            self.assertFalse(kwargs["reload_page_table"])
            self.assertIsNone(kwargs["page_table"])
            self.assertTrue(kwargs["sample_on_device"])
            active = tuple(slot for slot, pos in enumerate(start_pos) if pos >= 0)
            consumed.append(tuple(feedback))
            result = torch.full((4,), -1, dtype=torch.long)
            for slot in active:
                feedback[slot] += slot + 1
                gen.caches[slot].position += 1
                for layer in (3, 7):
                    positions[slot, layer] += 1
                result[slot] = feedback[slot]
            counters["model_replays"] += 1
            counters["sampling_replays"] += 1
            if defect == "host_refresh":
                counters["token_refreshes"] += 1
            if 0 not in active:
                if defect == "inactive_token":
                    feedback[0] += 1
                if defect == "inactive_position":
                    positions[0, 3] += 1
            return result

        gen.decode_forward = decode
        evidence = SimpleNamespace(before=lambda: None, after=lambda *args, **kw: None)
        return gen, evidence, lambda: list(feedback), lambda: dict(positions), consumed

    def test_resident_chain_shrink_and_no_host_uploads(self):
        gen, evidence, tokens, positions, consumed = self.fixture()
        result = run_resident_feedback(gen, None, evidence, tokens, positions)
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["steps"]), 4)
        self.assertEqual([step["active_slots"] for step in result["steps"]], [[0, 1, 2, 3]] * 2 + [[1, 3]] * 2)
        for index in range(1, 4):
            self.assertEqual(list(consumed[index]), result["steps"][index - 1]["sampled_output_tokens"])
        self.assertEqual(gen.counters["token_refreshes"], 0)
        self.assertEqual(gen.counters["model_replays"], 4)
        self.assertEqual(result["steps"][2]["sampled_output_tokens"][0], result["steps"][1]["sampled_output_tokens"][0])

    def test_upload_or_dormant_state_mutation_fails(self):
        for defect in ("host_refresh", "inactive_token", "inactive_position"):
            with self.subTest(defect=defect):
                gen, evidence, tokens, positions, _ = self.fixture(defect)
                with self.assertRaises(AssertionError):
                    run_resident_feedback(gen, None, evidence, tokens, positions)


if __name__ == "__main__":
    unittest.main()
