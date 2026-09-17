# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for immutable BlackHole required-capability contracts."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from models.tttv2_validate_bh_required_capabilities import (
    _QWEN_DEMO_MANIFEST_SHA256,
    DEFAULT_CONTRACTS,
    DEFAULT_SCHEMA,
    CapabilityContractValidationError,
    _demo_manifest_digest,
    load_schema,
    validate_contract_data,
    validate_contracts,
)


class TestBhRequiredCapabilities(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = load_schema(DEFAULT_SCHEMA)
        cls.valid_contract = json.loads(DEFAULT_CONTRACTS[1].read_text(encoding="utf-8"))

    def _contract(self) -> dict:
        return copy.deepcopy(self.valid_contract)

    @staticmethod
    def _load_contract(index: int) -> dict:
        return json.loads(DEFAULT_CONTRACTS[index].read_text(encoding="utf-8"))

    def _assert_invalid(self, contract: dict, pattern: str) -> None:
        with self.assertRaisesRegex(CapabilityContractValidationError, pattern):
            validate_contract_data(contract, self.schema)

    def test_all_checked_in_contracts_validate(self) -> None:
        self.assertEqual(validate_contracts(), DEFAULT_CONTRACTS)

    def test_all_contract_identities_are_bound_to_canonical_model_entry_points(self) -> None:
        expected = (
            (
                "tttv2_llama3_8b_bh_required_capabilities_v1",
                "llama3_8b",
                "models/common/tests/demos/llama3_8b/demo.py",
            ),
            (
                "tttv2_qwen3_32b_bh_required_capabilities_v1",
                "qwen3_32b",
                "models/common/tests/demos/qwen3_32b/demo.py",
            ),
            (
                "tttv2_llama33_70b_bh_required_capabilities_v1",
                "llama33_70b",
                "models/common/tests/demos/llama33_70b/demo.py",
            ),
        )
        for index, (contract_id, package, entry_point) in enumerate(expected):
            with self.subTest(contract_id=contract_id):
                contract = self._load_contract(index)
                self.assertEqual(contract["contract_id"], contract_id)
                self.assertEqual(contract["model"]["package"], package)
                self.assertEqual(contract["model"]["demo_entry_point"], entry_point)

                for key, bad_value in (
                    ("package", "renamed_package"),
                    ("demo_entry_point", "models/common/tests/demos/wrong/demo.py"),
                ):
                    mutated = copy.deepcopy(contract)
                    mutated["model"][key] = bad_value
                    self._assert_invalid(mutated, rf"model\.{key} must equal")

                mutated = copy.deepcopy(contract)
                mutated["contract_id"] = f"{contract_id}_renamed"
                self._assert_invalid(mutated, "contract_id must identify one of the three immutable BH contracts")

    def test_every_demo_requires_exact_cold_write_then_warm_read_protocol(self) -> None:
        for index in range(len(DEFAULT_CONTRACTS)):
            contract = self._load_contract(index)
            self.assertTrue(contract["demo_requirements"])
            self.assertTrue(
                all(row["cache_protocol"] == ["cold_write", "warm_read"] for row in contract["demo_requirements"])
            )
            for cache_protocol in (["warm_read"], ["cold_write"], ["warm_read", "cold_write"]):
                with self.subTest(contract=contract["contract_id"], cache_protocol=cache_protocol):
                    mutated = copy.deepcopy(contract)
                    mutated["demo_requirements"][0]["cache_protocol"] = cache_protocol
                    self._assert_invalid(mutated, "cache_protocol must equal")

    def test_llama_cross_cardinality_nodes_and_p150x4_ids_are_canonical(self) -> None:
        llama = json.loads(DEFAULT_CONTRACTS[0].read_text(encoding="utf-8"))
        requirements = llama["demo_requirements"]
        cross_nodes = {
            requirement["node_id"]
            for requirement in requirements
            if requirement["demo_case"] == "seeded-cross-cardinality"
        }
        self.assertEqual(
            cross_nodes,
            {
                "models/common/tests/demos/llama3_8b/demo.py::"
                "test_llama3_8b_bh_seeded_cross_cardinality[blackhole-performance-P150]",
                "models/common/tests/demos/llama3_8b/demo.py::"
                "test_llama3_8b_bh_seeded_cross_cardinality[blackhole-accuracy-P150]",
                "models/common/tests/demos/llama3_8b/demo.py::"
                "test_llama3_8b_bh_seeded_cross_cardinality[blackhole-performance-P150X4]",
                "models/common/tests/demos/llama3_8b/demo.py::"
                "test_llama3_8b_bh_seeded_cross_cardinality[blackhole-accuracy-P150X4]",
            },
        )
        p150x4_nodes = [
            requirement["node_id"]
            for requirement in requirements
            if requirement["geometry_id"] in {"p150x4_tp4_dp1", "p150x4_tp1_dp4"}
        ]
        self.assertTrue(p150x4_nodes)
        self.assertTrue(all(node_id.endswith("-P150X4]") for node_id in p150x4_nodes))

    def test_llama_completed_negative_disposition_is_valid_and_policy_stays_sequential(self) -> None:
        contract = self._load_contract(0)
        cross_rows = [row for row in contract["demo_requirements"] if row["demo_case"] == "seeded-cross-cardinality"]
        self.assertEqual(len(cross_rows), 4)
        for row in cross_rows:
            self.assertIn("INVARIANT", row["acceptance_condition"])
            self.assertIn("BATCHED_PREFILL_REJECTED", row["acceptance_condition"])
            self.assertIn("does not pass invariance", row["acceptance_condition"])
            self.assertIn("Malformed or incomplete outputs", row["acceptance_condition"])

        experiment = next(
            row for row in contract["cross_cutting_requirements"] if row["id"] == "cross_cardinality_invariance"
        )
        policy = next(
            row for row in contract["cross_cutting_requirements"] if row["id"] == "disable_batched_prefill_policy"
        )
        self.assertIn("either disposition satisfies", experiment["acceptance_condition"])
        self.assertIn("does not pass invariance", experiment["acceptance_condition"])
        self.assertIn("remains sequential after BATCHED_PREFILL_REJECTED", policy["acceptance_condition"])
        self.assertIn("independent of completed experiment disposition", policy["acceptance_condition"])
        validate_contract_data(contract, self.schema)

        experiment["capability"] = "Only invariant execution is relevant."
        experiment["acceptance_condition"] = "Only INVARIANT is a completed experiment."
        self._assert_invalid(contract, "experiment disposition must preserve phrase 'BATCHED_PREFILL_REJECTED'")

        contract = self._load_contract(0)
        policy = next(
            row for row in contract["cross_cutting_requirements"] if row["id"] == "disable_batched_prefill_policy"
        )
        policy["capability"] = "Execution policy is unspecified."
        policy["acceptance_condition"] = "Any completed experiment may change production policy."
        self._assert_invalid(contract, "batched-prefill policy must follow verdict phrase")

    def test_llama_cross_cardinality_rejects_missing_malformed_execution_guard(self) -> None:
        contract = self._load_contract(0)
        row = next(
            row for row in contract["demo_requirements"] if row["id"] == "p150.performance.seeded_cross_cardinality"
        )
        row["acceptance_condition"] = row["acceptance_condition"].replace("Malformed or incomplete outputs", "Outputs")
        self._assert_invalid(contract, "including 'Malformed or incomplete outputs'")

    def test_llama_missing_performance_floor_allows_observation_but_not_acceptance(self) -> None:
        contract = self._load_contract(0)
        policy = next(row for row in contract["cross_cutting_requirements"] if row["id"] == "fail_closed_performance")
        policy_text = f"{policy['capability']} {policy['acceptance_condition']}"
        self.assertIn("must not block BH model execution or observational measurement", policy_text)
        self.assertIn("cannot establish performance acceptance", policy_text)
        self.assertIn("Every complete declared floor remains enforced", policy_text)
        self.assertIn("every failed meets_target result fails", policy_text)
        self.assertIn("every declared target passes", policy_text)
        validate_contract_data(contract, self.schema)

        policy["capability"] = (
            "A missing metric floor blocks model execution. Every complete declared floor remains enforced, "
            "every failed meets_target result fails, and performance acceptance requires an independently "
            "justified, frozen floor."
        )
        self._assert_invalid(contract, "must preserve observational execution and fail-closed acceptance")

    def test_llama_manifests_reject_missing_extra_and_renamed_rows(self) -> None:
        for index, package in ((0, "llama3_8b"), (2, "llama33_70b")):
            base = self._load_contract(index)
            with self.subTest(package=package, mutation="missing"):
                contract = copy.deepcopy(base)
                del contract["demo_requirements"][0]
                self._assert_invalid(contract, rf"{package} demo manifest missing immutable rows")

            with self.subTest(package=package, mutation="extra"):
                contract = copy.deepcopy(base)
                extra = copy.deepcopy(contract["demo_requirements"][0])
                extra["id"] = f"{package}.undeclared.row"
                extra["node_id"] = contract["model"]["demo_entry_point"] + "::test_undeclared[case]"
                contract["demo_requirements"].append(extra)
                self._assert_invalid(contract, rf"{package} demo manifest has undeclared rows")

            with self.subTest(package=package, mutation="renamed"):
                contract = copy.deepcopy(base)
                contract["demo_requirements"][0]["id"] += ".renamed"
                with self.assertRaises(CapabilityContractValidationError) as raised:
                    validate_contract_data(contract, self.schema)
                self.assertIn(f"{package} demo manifest missing immutable rows", str(raised.exception))
                self.assertIn(f"{package} demo manifest has undeclared rows", str(raised.exception))

    def test_llama_manifests_bind_exact_executable_workload_fields(self) -> None:
        mutations = (
            ("node_id", lambda row: row["node_id"].replace("[", "[renamed-", 1)),
            ("profile", lambda row: "accuracy" if row["profile"] == "performance" else "performance"),
            ("demo_case", lambda row: row["demo_case"] + "-renamed"),
            ("geometry_id", lambda row: row["geometry_id"] + "_renamed"),
            ("batch_size", lambda row: row["batch_size"] + 1),
            ("decode_tokens", lambda row: row["decode_tokens"] + 1),
            ("repeat_batches", lambda row: row["repeat_batches"] + 1),
            ("report_perf", lambda row: not row["report_perf"]),
            ("dp", lambda row: row["dp"] + 1),
            ("trace_mode", lambda row: "all" if row["trace_mode"] != "all" else "decode_only"),
            ("context_bucket", lambda row: row["context_bucket"] + 1),
            ("capabilities", lambda row: list(reversed(row["capabilities"])) + ["determinism"]),
            ("acceptance_condition", lambda row: row["acceptance_condition"] + " Mutated."),
        )
        for index, package in ((0, "llama3_8b"), (2, "llama33_70b")):
            base = self._load_contract(index)
            for field, replacement in mutations:
                with self.subTest(package=package, field=field):
                    contract = copy.deepcopy(base)
                    row = contract["demo_requirements"][0]
                    row[field] = replacement(row)
                    self._assert_invalid(contract, rf"{package} demo manifest does not match immutable")

    def test_llama70_performance_policy_preserves_observation_without_acceptance(self) -> None:
        contract = self._load_contract(2)
        policy = next(row for row in contract["cross_cutting_requirements"] if row["id"] == "fail_closed_performance")
        policy["acceptance_condition"] = "Missing targets block execution before measurements are collected."
        self._assert_invalid(
            contract,
            "llama33_70b performance policy must preserve observational execution and "
            "fail-closed complete-floor semantics",
        )

    def test_llama_source_ast_rejects_missing_functions_and_parameter_ids(self) -> None:
        for index, package in ((0, "llama3_8b"), (2, "llama33_70b")):
            base = self._load_contract(index)
            with self.subTest(package=package, mutation="function"):
                contract = copy.deepcopy(base)
                row = contract["demo_requirements"][0]
                path, _, suffix = row["node_id"].partition("::")
                row["node_id"] = f"{path}::test_missing_function[{suffix.split('[', 1)[-1]}"
                with self.assertRaises(CapabilityContractValidationError) as raised:
                    validate_contract_data(contract, self.schema)
                self.assertIn("declares missing test function", str(raised.exception))

            with self.subTest(package=package, mutation="source_parameter_id"):
                contract = copy.deepcopy(base)
                contract["demo_requirements"][0]["source_parameter_id"] = "not-a-declared-pytest-parameter"
                with self.assertRaises(CapabilityContractValidationError) as raised:
                    validate_contract_data(contract, self.schema)
                self.assertIn("source_parameter_id", str(raised.exception))
                self.assertIn("is not declared by", str(raised.exception))

    def test_demo_source_entry_point_must_exist(self) -> None:
        contract = self._load_contract(0)
        contract["model"]["demo_entry_point"] = "models/common/tests/demos/llama3_8b/missing.py"
        with self.assertRaises(CapabilityContractValidationError) as raised:
            validate_contract_data(contract, self.schema)
        self.assertIn("declared demo entry point does not exist", str(raised.exception))

    def test_qwen_full_trace_buckets_and_seeded_cross_cardinality_nodes_are_canonical(self) -> None:
        qwen = self._contract()
        eval_perf = [row for row in qwen["demo_requirements"] if row["demo_case"] == "eval-32-perf-report"]
        self.assertEqual({row["profile"] for row in eval_perf}, {"performance", "accuracy"})
        self.assertTrue(all(row["trace_mode"] == "all" for row in eval_perf))

        all_trace = [row for row in qwen["serving_requirements"] if row["profile"] == "all"]
        self.assertTrue(all_trace)
        self.assertTrue(all(row["trace_prefill_buckets"] == [128, 1024] for row in all_trace))

        cross_nodes = {
            row["source_parameter_id"]: row["node_id"]
            for row in qwen["demo_requirements"]
            if row["demo_case"] == "seeded-cross-cardinality"
        }
        self.assertEqual(
            cross_nodes,
            {
                "seeded-cross-cardinality": "models/common/tests/demos/qwen3_32b/demo.py::"
                "test_qwen3_32b_p150x4_seeded_cross_cardinality[P150x4]",
            },
        )

    def test_qwen_phase3_contract_invariants_fail_closed(self) -> None:
        mutations = (
            (
                lambda contract: next(
                    row for row in contract["demo_requirements"] if row["demo_case"] == "eval-32-perf-report"
                ).__setitem__("trace_mode", "decode_only"),
                "must declare full trace mode",
            ),
            (
                lambda contract: next(
                    row for row in contract["serving_requirements"] if row["profile"] == "all"
                ).__setitem__("trace_prefill_buckets", [128]),
                "must declare model-owned Q128/Q1024",
            ),
            (
                lambda contract: next(
                    row for row in contract["demo_requirements"] if row["demo_case"] == "seeded-cross-cardinality"
                ).__setitem__("node_id", "wrong-node"),
                "does not match its immutable identity",
            ),
            (
                lambda contract: next(
                    row for row in contract["demo_requirements"] if row["demo_case"] == "seeded-cross-cardinality"
                ).__setitem__(
                    "acceptance_condition",
                    "A passing node is enough without a recorded experiment verdict.",
                ),
                "acceptance must define exact-token executed verdict semantics",
            ),
            (
                lambda contract: next(
                    row for row in contract["cross_cutting_requirements"] if row["id"] == "fail_closed_performance"
                ).__setitem__(
                    "acceptance_condition",
                    "Missing targets block model execution before measurements are collected.",
                ),
                "must preserve observational execution and fail-closed complete-floor semantics",
            ),
        )
        for mutate, message in mutations:
            with self.subTest(message=message):
                contract = self._contract()
                mutate(contract)
                self._assert_invalid(contract, message)

    def test_qwen_manifest_rejects_missing_extra_and_renamed_rows(self) -> None:
        contract = self._contract()
        del contract["demo_requirements"][0]
        self._assert_invalid(contract, "demo manifest missing immutable rows")

        contract = self._contract()
        extra = copy.deepcopy(contract["demo_requirements"][0])
        extra["id"] = "p150x4.performance.unplanned"
        extra["node_id"] = "models/common/tests/demos/qwen3_32b/demo.py::test_unplanned[P150x4]"
        contract["demo_requirements"].append(extra)
        self._assert_invalid(contract, "demo manifest has undeclared rows")

        contract = self._contract()
        contract["demo_requirements"][0]["id"] = "p150x4.performance.token_accuracy_renamed"
        self._assert_invalid(contract, "demo manifest missing immutable rows")

        contract = self._contract()
        contract["demo_requirements"][0]["source_parameter_id"] = "token-accuracy-renamed"
        self._assert_invalid(contract, "does not match its immutable identity")

    def test_qwen_manifest_binds_every_executable_demo_field(self) -> None:
        mutations = (
            ("node_id", lambda row: row["node_id"].replace("[", "[renamed-", 1)),
            ("profile", lambda row: "accuracy" if row["profile"] == "performance" else "performance"),
            ("demo_case", lambda row: row["demo_case"] + "-renamed"),
            ("source_parameter_id", lambda row: row["source_parameter_id"] + "-renamed"),
            ("geometry_id", lambda row: row["geometry_id"] + "_renamed"),
            ("batch_size", lambda row: row["batch_size"] + 1),
            ("decode_tokens", lambda row: row["decode_tokens"] + 1),
            ("repeat_batches", lambda row: row["repeat_batches"] + 1),
            ("report_perf", lambda row: not row["report_perf"]),
            ("dp", lambda row: row["dp"] + 1),
            ("trace_mode", lambda row: "all" if row["trace_mode"] != "all" else "decode_only"),
            ("context_bucket", lambda row: row["context_bucket"] + 1),
            ("cache_protocol", lambda row: list(reversed(row["cache_protocol"]))),
            ("capabilities", lambda row: list(reversed(row["capabilities"]))),
            ("required_resolution", lambda _row: "PASS"),
            ("acceptance_condition", lambda row: row["acceptance_condition"] + " Mutated."),
        )
        base = self._contract()
        for field, replacement in mutations:
            with self.subTest(field=field):
                contract = copy.deepcopy(base)
                row = contract["demo_requirements"][0]
                row[field] = replacement(row)
                self.assertNotEqual(_demo_manifest_digest(contract["demo_requirements"]), _QWEN_DEMO_MANIFEST_SHA256)
                self._assert_invalid(contract, "capability contract validation failed")

    def test_qwen_completed_negative_disposition_is_valid_and_policy_follows_verdict(self) -> None:
        contract = self._contract()
        experiment = next(
            row for row in contract["cross_cutting_requirements"] if row["id"] == "cross_cardinality_invariance"
        )
        policy = next(
            row for row in contract["cross_cutting_requirements"] if row["id"] == "disable_batched_prefill_policy"
        )
        self.assertIn("either disposition satisfies", experiment["acceptance_condition"])
        self.assertIn("BATCHED_PREFILL_REJECTED", experiment["acceptance_condition"])
        self.assertIn("remains sequential", policy["acceptance_condition"])
        validate_contract_data(contract, self.schema)

        experiment["capability"] = "Only an invariant result is relevant."
        experiment["acceptance_condition"] = "Only INVARIANT is a completed experiment."
        self._assert_invalid(contract, "must permit completed verdict phrase 'BATCHED_PREFILL_REJECTED'")

    def test_qwen_checks_cannot_be_bypassed_by_renaming_package(self) -> None:
        contract = self._contract()
        contract["model"]["package"] = "qwen3_32b_typo"
        del contract["demo_requirements"][0]
        with self.assertRaisesRegex(CapabilityContractValidationError, "model.package must equal") as raised:
            validate_contract_data(contract, self.schema)
        self.assertIn("demo manifest missing immutable rows", str(raised.exception))

    def test_schema_requires_draft_2020_12(self) -> None:
        schema = copy.deepcopy(self.schema)
        schema["$schema"] = "http://json-schema.org/draft-07/schema#"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "schema.json"
            path.write_text(json.dumps(schema), encoding="utf-8")
            with self.assertRaisesRegex(CapabilityContractValidationError, "must declare JSON Schema Draft 2020-12"):
                load_schema(path)

    def test_schema_and_stdlib_required_fields_and_cache_protocol_stay_in_sync(self) -> None:
        self.assertIn("$schema", self.schema["required"])
        cache_schema = self.schema["$defs"]["demoRequirement"]["properties"]["cache_protocol"]
        self.assertEqual(
            cache_schema,
            {
                "type": "array",
                "prefixItems": [{"const": "cold_write"}, {"const": "warm_read"}],
                "items": False,
                "minItems": 2,
                "maxItems": 2,
            },
        )

        for mutate, message in (
            (lambda schema: schema["required"].remove("$schema"), "root required keys disagree"),
            (
                lambda schema: schema["$defs"]["demoRequirement"]["properties"]["cache_protocol"].__setitem__(
                    "minItems", 1
                ),
                "cache_protocol declaration disagrees",
            ),
        ):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                schema = copy.deepcopy(self.schema)
                mutate(schema)
                path = Path(directory) / "schema.json"
                path.write_text(json.dumps(schema), encoding="utf-8")
                with self.assertRaisesRegex(CapabilityContractValidationError, message):
                    load_schema(path)

    def test_duplicate_geometry_requirement_row_and_cross_cutting_ids_fail(self) -> None:
        mutations = (
            ("geometries", "id", "duplicate geometry IDs"),
            ("demo_requirements", "id", "duplicate demo requirement IDs"),
            ("serving_requirements", "row_id", "duplicate serving row IDs"),
            ("cross_cutting_requirements", "id", "duplicate cross-cutting requirement IDs"),
        )
        for collection, key, message in mutations:
            with self.subTest(collection=collection):
                contract = self._contract()
                contract[collection].append(copy.deepcopy(contract[collection][0]))
                self._assert_invalid(contract, message)

    def test_duplicate_demo_node_id_fails(self) -> None:
        contract = self._contract()
        contract["demo_requirements"][1]["node_id"] = contract["demo_requirements"][0]["node_id"]
        self._assert_invalid(contract, "duplicate demo node IDs")

    def test_null_demo_node_id_fails_schema(self) -> None:
        contract = self._contract()
        contract["demo_requirements"][0]["node_id"] = None
        self._assert_invalid(contract, r"demo_requirements\.0\.node_id must be a string")

    def test_missing_and_unknown_keys_fail_closed(self) -> None:
        contract = self._contract()
        del contract["model"]["hf_model_id"]
        self._assert_invalid(contract, "model missing required key hf_model_id")

        contract = self._contract()
        contract["model"]["mutable_status"] = "PASS"
        self._assert_invalid(contract, "model has unknown key mutable_status")

    def test_wrong_type_enum_and_constant_fail_closed(self) -> None:
        mutations = (
            (
                lambda contract: contract["demo_requirements"][0].__setitem__("batch_size", True),
                "batch_size must be an integer",
            ),
            (
                lambda contract: contract["geometries"][0].__setitem__("role", "experimental"),
                "role must be one of",
            ),
            (
                lambda contract: contract["serving_requirements"][0].__setitem__("tier1_performance", "optional"),
                "tier1_performance must equal 'required'",
            ),
            (lambda contract: contract.__setitem__("schema_version", 2), "schema_version must equal 1"),
        )
        for mutate, message in mutations:
            with self.subTest(message=message):
                contract = self._contract()
                mutate(contract)
                self._assert_invalid(contract, message)

    def test_unresolved_demo_and_serving_geometry_references_fail(self) -> None:
        for collection in ("demo_requirements", "serving_requirements"):
            with self.subTest(collection=collection):
                contract = self._contract()
                contract[collection][0]["geometry_id"] = "missing_geometry"
                self._assert_invalid(contract, "references unknown geometry_id missing_geometry")

    def test_unresolved_cross_cutting_geometry_reference_fails(self) -> None:
        contract = self._contract()
        contract["cross_cutting_requirements"][0]["applies_to"].append("missing_geometry")
        self._assert_invalid(contract, "references unknown geometry missing_geometry")

    def test_serving_profile_and_trace_mode_must_match(self) -> None:
        contract = self._contract()
        contract["serving_requirements"][0]["trace_mode"] = "all"
        self._assert_invalid(contract, "profile=decode_only but trace_mode=all")

    def test_decode_only_row_cannot_declare_prefill_trace_buckets(self) -> None:
        contract = self._contract()
        contract["serving_requirements"][0]["trace_prefill_buckets"] = [128]
        self._assert_invalid(contract, "must not declare prefill trace buckets")

    def test_all_trace_row_requires_prefill_trace_buckets(self) -> None:
        contract = self._contract()
        contract["serving_requirements"][1]["trace_prefill_buckets"] = []
        self._assert_invalid(contract, "must declare prefill trace buckets")


if __name__ == "__main__":
    unittest.main()
