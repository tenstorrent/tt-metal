# SPDX-License-Identifier: Apache-2.0
"""Host-only sizing checks against captured identity and launch-event fixtures."""
import ast
import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from collect_candidate_sizing import KV_TILE_BYTES, output_suffix, pool_geometry, size_candidate

FIXTURE = Path(__file__).with_name("fixtures") / "candidate_sizing_head_bfp8_b32.json"


def captured_fixture():
    return json.loads(FIXTURE.read_text())


def resized_fixture(*, batch, pool_ids, cache_dtype="bfloat8_b", head="bf16"):
    """Create a deterministic host-only variant of the captured pool geometry."""
    fixture = captured_fixture()
    identity = fixture["identity"]
    identity["max_num_seqs"] = batch
    policy = copy.deepcopy(identity["precision"])
    policy["config_id"] = f"host_test_b{batch}_{cache_dtype}_{head}"
    policy["weight_groups"]["attention"] = "bf16"
    policy["weight_groups"]["mlp"] = "bf16"
    policy["weight_groups"]["down"] = "bf16"
    policy["weight_groups"]["lm_head"] = head
    policy["kv_cache_dtype"] = cache_dtype
    identity["precision"] = policy
    identity["observed_registration"]["precision"] = copy.deepcopy(policy)

    allocation = next(e for e in fixture["events"] if e["event"] == "serving_cache_allocation")
    for row in allocation["layers"]:
        old_blocks = row["scheduler_shape"][0]
        physical_factor = row["physical_shape"][0] // old_blocks
        row["scheduler_shape"][0] = pool_ids
        row["physical_shape"][0] = pool_ids * physical_factor

    num_banks = next(
        e["allocator"]["num_banks"]
        for e in fixture["events"]
        if e["event"] == "serving_memory" and e.get("phase") == "before_serving_cache"
    )
    ids, expected_pool_bytes, _bytes_per_id = pool_geometry(allocation, cache_dtype, num_banks)
    assert ids == pool_ids
    validation = next(e for e in fixture["events"] if e["event"] == "serving_pool_launch_validation")
    for device in validation["per_device"]:
        device["pool_bytes"] = expected_pool_bytes
        device["remaining_bytes_after_reserves"] = (
            min(device["free_bytes"], device["contiguous_free_bytes"]) - device["pool_bytes"] - device["reserve_bytes"]
        )
    validation["fits"] = all(d["remaining_bytes_after_reserves"] >= 0 for d in validation["per_device"])
    if not validation["fits"]:
        raise AssertionError("Test fixture's synthesized pool must fit its captured free-memory envelope")
    return fixture


class CandidateSizingTests(unittest.TestCase):
    def test_captured_bfp8_b32_identity_reconstructs_measured_pool_geometry(self):
        fixture = captured_fixture()
        identity, events = fixture["identity"], fixture["events"]
        result = size_candidate(identity, events, task_count=85, host_slots=12)
        self.assertEqual((result["pool_ids"], result["pool_id_bytes"]), (36896, 696320))
        self.assertEqual(result["allocator_num_banks"], 8)
        self.assertEqual(result["capacity_agents"], 12)
        self.assertEqual(result["agents"], 12)
        self.assertEqual(result["batch"], 32)
        self.assertEqual(result["smallest_configured_batch_for_selected_agents"], 32)
        self.assertTrue(result["smallest_batch_is_measured"])

    def test_bf16_head_and_batch8_are_supported_and_agents_are_bounded(self):
        fixture = resized_fixture(batch=8, pool_ids=10248, cache_dtype="bfloat8_b", head="bf16")
        result = size_candidate(fixture["identity"], fixture["events"], task_count=85, host_slots=12)
        self.assertEqual(result["precision"]["weight_groups"]["lm_head"], "bf16")
        self.assertEqual(result["pool_cache_dtype"], "bfloat8_b")
        self.assertEqual(result["agents"], 3)
        self.assertLessEqual(result["agents"], result["max_num_seqs"])
        self.assertEqual(result["smallest_configured_batch_for_selected_agents"], 8)
        self.assertTrue(result["smallest_batch_is_measured"])

    def test_bf16_kv_bytes_follow_tile_geometry_not_a_hardcoded_bfp8_id_size(self):
        fixture = resized_fixture(batch=32, pool_ids=16416, cache_dtype="bfloat16", head="bf16")
        result = size_candidate(fixture["identity"], fixture["events"], task_count=85, host_slots=12)
        self.assertEqual(result["pool_id_bytes"], 1310720)
        self.assertAlmostEqual(result["pool_id_bytes"] / 696320, 2048 / 1088)
        self.assertEqual(result["capacity_agents"], 5)
        self.assertEqual(result["agents"], 5)
        self.assertEqual(result["smallest_configured_batch_for_selected_agents"], 8)
        self.assertFalse(result["smallest_batch_is_measured"])
        self.assertIn("separate launch is required", result["batch_selection_advisory"])

    def test_dtype_tile_bytes_match_serving_adapter_source_without_importing_it(self):
        source = ROOT / "tt" / "generator_vllm.py"
        tree = ast.parse(source.read_text())
        assignments = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "tile_bytes" for target in node.targets)
        ]
        self.assertEqual(len(assignments), 1)
        mapping = {
            ast.unparse(key): ast.literal_eval(value)
            for key, value in zip(assignments[0].value.keys, assignments[0].value.values)
        }
        self.assertEqual(mapping["ttnn.bfloat8_b"], KV_TILE_BYTES["bfloat8_b"])
        self.assertEqual(mapping["ttnn.bfloat16"], KV_TILE_BYTES["bfloat16"])

    def test_batch1_selection_cannot_exceed_configured_sequence_count(self):
        fixture = resized_fixture(batch=1, pool_ids=8449)
        result = size_candidate(fixture["identity"], fixture["events"], task_count=85, host_slots=12)
        self.assertEqual(result["agents"], 1)
        self.assertEqual(result["smallest_configured_batch_for_selected_agents"], 1)
        self.assertTrue(result["smallest_batch_is_measured"])

    def test_rejects_policy_mismatch_bad_batch_and_wrong_launch_pool_bytes(self):
        fixture = captured_fixture()
        identity = copy.deepcopy(fixture["identity"])
        identity["precision"]["weight_groups"]["lm_head"] = "bf16"
        with self.assertRaisesRegex(ValueError, "exactly match"):
            size_candidate(identity, fixture["events"], task_count=1, host_slots=1)

        fixture = captured_fixture()
        fixture["identity"]["max_num_seqs"] = 16
        with self.assertRaisesRegex(ValueError, "batch must be one of"):
            size_candidate(fixture["identity"], fixture["events"], task_count=1, host_slots=1)

        fixture = captured_fixture()
        validation = next(e for e in fixture["events"] if e["event"] == "serving_pool_launch_validation")
        validation["per_device"][0]["pool_bytes"] += 1
        with self.assertRaisesRegex(ValueError, "disagree on measured KV pool bytes"):
            size_candidate(fixture["identity"], fixture["events"], task_count=1, host_slots=1)

        fixture = captured_fixture()
        fixture["events"] = [event for event in fixture["events"] if event.get("phase") != "before_serving_cache"]
        with self.assertRaisesRegex(ValueError, "pre-cache serving_memory event"):
            size_candidate(fixture["identity"], fixture["events"], task_count=1, host_slots=1)

    def test_output_names_do_not_mislabel_new_precision_as_head_bfp8(self):
        fixture = captured_fixture()
        self.assertEqual(output_suffix(fixture["identity"]), "_head_bfp8")
        modern = resized_fixture(batch=8, pool_ids=10248)
        self.assertEqual(output_suffix(modern["identity"]), "")
        self.assertEqual(output_suffix(modern["identity"], "p17"), "_p17")
        with self.assertRaisesRegex(ValueError, "Output prefix"):
            output_suffix(modern["identity"], "../head_bfp8")


if __name__ == "__main__":
    unittest.main()
