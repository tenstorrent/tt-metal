#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for exabox.cluster_health.v1."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

TESTS_DIR = Path(__file__).resolve().parent
EXABOX_DIR = TESTS_DIR.parent
for _path in (EXABOX_DIR, TESTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import fixtures  # noqa: E402
from cluster_health_schema import (  # noqa: E402
    CLUSTER_HEALTH_JSON_SCHEMA,
    SCHEMA_ID,
    loads_and_validate,
    validate_record,
)

SCHEMA_SOURCE = EXABOX_DIR / "cluster_health_schema.py"


class TestValidFixtures(unittest.TestCase):
    def test_canonical_example_with_store(self):
        record = json.loads(fixtures.CANONICAL_WITH_STORE)
        validate_record(record, file_written=True)
        self.assertEqual(record["schema"], SCHEMA_ID)
        self.assertEqual(record["labels"]["quad"], "110-C-Q1")
        self.assertEqual(record["labels"]["superpod"], "SC36_3")

    def test_laptop_dry_run(self):
        text = fixtures.LAPTOP_DRY_RUN
        record = json.loads(text)
        self.assertNotIn("artifact_uri", record)
        self.assertNotIn("source", record)
        self.assertNotIn("record_id", record)
        self.assertNotIn("record_uri", record)
        self.assertNotIn("labels", record)
        self.assertNotIn("cluster", record)
        loads_and_validate(text, file_written=False)

    def test_recover_omits_analyzer_code(self):
        record = json.loads(fixtures.RECOVER_NO_ANALYZER_CODE)
        self.assertNotIn("analyzer_code", record)
        validate_record(record, file_written=False)

    def test_rank_binding_without_mesh_host_rank(self):
        record = json.loads(fixtures.LAPTOP_DRY_RUN)
        record["topology"] = {"rank_bindings": [{"rank": 0, "mesh_id": 0}]}
        validate_record(record, file_written=False)

    def test_host_from_diag(self):
        record = json.loads(fixtures.HOST_FROM_DIAG)
        validate_record(record, file_written=False)
        self.assertEqual(record["test_type"], "host")
        self.assertEqual(record["status"], "passed")
        self.assertEqual(record["analyzer_code"], 0)


class TestRejects(unittest.TestCase):
    def _dry_run(self) -> dict:
        return copy.deepcopy(json.loads(fixtures.LAPTOP_DRY_RUN))

    def test_missing_hosts(self):
        record = self._dry_run()
        del record["hosts"]
        with self.assertRaisesRegex(ValueError, r"^hosts: required"):
            validate_record(record)

    def test_empty_hosts(self):
        record = self._dry_run()
        record["hosts"] = []
        with self.assertRaisesRegex(ValueError, r"^hosts:"):
            validate_record(record)

    def test_unknown_test_type(self):
        record = self._dry_run()
        record["test_type"] = "ccl"
        with self.assertRaisesRegex(ValueError, r"^test_type:"):
            validate_record(record)

    def test_unknown_status(self):
        record = self._dry_run()
        record["status"] = "ok"
        with self.assertRaisesRegex(ValueError, r"^status:"):
            validate_record(record)

    def test_forbidden_scope(self):
        record = self._dry_run()
        record["scope"] = "quad"
        with self.assertRaisesRegex(ValueError, r"unknown keys|scope"):
            validate_record(record)

    def test_wrong_schema_id(self):
        record = self._dry_run()
        record["schema"] = "exabox.cluster_health.v0"
        with self.assertRaisesRegex(ValueError, r"^schema:"):
            validate_record(record)

    def test_record_id_without_file_written(self):
        record = self._dry_run()
        record["record_id"] = "abc"
        with self.assertRaisesRegex(ValueError, r"^record_id:"):
            validate_record(record, file_written=False)

    def test_missing_record_uri_when_file_written(self):
        record = self._dry_run()
        record["record_id"] = "abc"
        with self.assertRaisesRegex(ValueError, r"^record_uri:"):
            validate_record(record, file_written=True)

    def test_relative_record_uri_when_file_written(self):
        record = self._dry_run()
        record["record_id"] = "abc"
        record["record_uri"] = "cluster-health/abc.json"
        with self.assertRaisesRegex(ValueError, r"^record_uri:"):
            validate_record(record, file_written=True)

    def test_duration_s_rejects_nan_and_infinity(self):
        record = self._dry_run()
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                bad = copy.deepcopy(record)
                bad["duration_s"] = value
                with self.assertRaisesRegex(ValueError, r"^duration_s:"):
                    validate_record(bad)

    def test_rank_binding_missing_mesh_id(self):
        record = self._dry_run()
        record["topology"] = {"rank_bindings": [{"rank": 0}]}
        with self.assertRaisesRegex(ValueError, r"missing mesh_id"):
            validate_record(record)

    def test_rank_binding_non_integer_mesh_host_rank(self):
        record = self._dry_run()
        record["topology"] = {"rank_bindings": [{"rank": 0, "mesh_id": 0, "mesh_host_rank": "1"}]}
        with self.assertRaisesRegex(ValueError, r"mesh_host_rank:"):
            validate_record(record)

    def test_site_alias_keys_rejected_in_topology(self):
        record = self._dry_run()
        for alias_key in ("quad", "superpod", "SC36_3"):
            with self.subTest(alias_key=alias_key):
                bad = copy.deepcopy(record)
                bad["topology"] = {alias_key: "nope"}
                with self.assertRaisesRegex(ValueError, r"^topology:"):
                    validate_record(bad)


class TestNoTopologyYaml(unittest.TestCase):
    def test_schema_source_does_not_mention_topology_yaml(self):
        source = SCHEMA_SOURCE.read_text(encoding="utf-8")
        self.assertNotIn("topology.yaml", source)

    def test_validate_does_not_open_files(self):
        record = json.loads(fixtures.LAPTOP_DRY_RUN)
        with patch("builtins.open") as mocked_open:
            validate_record(record, file_written=False)
            mocked_open.assert_not_called()

    def test_hosts_only_without_topology_is_valid(self):
        record = json.loads(fixtures.LAPTOP_DRY_RUN)
        self.assertNotIn("topology", record)
        validate_record(record)


class TestJsonSchemaDocument(unittest.TestCase):
    def test_documented_schema_forbids_unknown_properties(self):
        self.assertFalse(CLUSTER_HEALTH_JSON_SCHEMA.get("additionalProperties"))
        self.assertEqual(CLUSTER_HEALTH_JSON_SCHEMA["properties"]["schema"]["const"], SCHEMA_ID)
        self.assertIn("host", CLUSTER_HEALTH_JSON_SCHEMA["properties"]["test_type"]["enum"])
        self.assertFalse(CLUSTER_HEALTH_JSON_SCHEMA["properties"]["topology"]["additionalProperties"])


if __name__ == "__main__":
    unittest.main()
