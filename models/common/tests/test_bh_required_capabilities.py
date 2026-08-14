"""Host-only tests for immutable BlackHole required-capability contracts."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from models.tttv2_validate_bh_required_capabilities import (
    DEFAULT_CONTRACTS,
    DEFAULT_SCHEMA,
    CapabilityContractValidationError,
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

    def _assert_invalid(self, contract: dict, pattern: str) -> None:
        with self.assertRaisesRegex(CapabilityContractValidationError, pattern):
            validate_contract_data(contract, self.schema)

    def test_all_checked_in_contracts_validate(self) -> None:
        self.assertEqual(validate_contracts(), DEFAULT_CONTRACTS)

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

    def test_schema_requires_draft_2020_12(self) -> None:
        schema = copy.deepcopy(self.schema)
        schema["$schema"] = "http://json-schema.org/draft-07/schema#"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "schema.json"
            path.write_text(json.dumps(schema), encoding="utf-8")
            with self.assertRaisesRegex(CapabilityContractValidationError, "must declare JSON Schema Draft 2020-12"):
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
                lambda contract: contract["serving_requirements"][0].__setitem__(
                    "tier1_performance", "optional"
                ),
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
