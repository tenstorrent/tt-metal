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

EXABOX_DIR = Path(__file__).resolve().parents[1]
if str(EXABOX_DIR) not in sys.path:
    sys.path.insert(0, str(EXABOX_DIR))

from cluster_health_schema import (  # noqa: E402
    CLUSTER_HEALTH_JSON_SCHEMA,
    SCHEMA_ID,
    loads_and_validate,
    validate_record,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SCHEMA_SOURCE = EXABOX_DIR / "cluster_health_schema.py"


def _load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class TestValidFixtures(unittest.TestCase):
    def test_canonical_example_with_store(self):
        record = _load_fixture("canonical_with_store.json")
        validate_record(record, file_written=True)
        self.assertEqual(record["schema"], SCHEMA_ID)
        self.assertEqual(record["labels"]["quad"], "110-C-Q1")
        self.assertEqual(record["labels"]["superpod"], "SC36_3")

    def test_laptop_dry_run(self):
        record = _load_fixture("laptop_dry_run.json")
        text = (FIXTURES / "laptop_dry_run.json").read_text(encoding="utf-8")
        self.assertNotIn("/data/stackstorm", text)
        self.assertNotIn("record_id", record)
        self.assertNotIn("record_uri", record)
        self.assertNotIn("labels", record)
        self.assertNotIn("cluster", record)
        loads_and_validate(text, file_written=False)

    def test_recover_omits_analyzer_code(self):
        record = _load_fixture("recover_no_analyzer_code.json")
        self.assertNotIn("analyzer_code", record)
        validate_record(record, file_written=False)


class TestRejects(unittest.TestCase):
    def _dry_run(self) -> dict:
        return copy.deepcopy(_load_fixture("laptop_dry_run.json"))

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
        record = _load_fixture("laptop_dry_run.json")
        with patch("builtins.open") as mocked_open:
            validate_record(record, file_written=False)
            mocked_open.assert_not_called()

    def test_hosts_only_without_topology_is_valid(self):
        record = _load_fixture("laptop_dry_run.json")
        self.assertNotIn("topology", record)
        validate_record(record)


class TestJsonSchemaDocument(unittest.TestCase):
    def test_documented_schema_forbids_unknown_properties(self):
        self.assertFalse(CLUSTER_HEALTH_JSON_SCHEMA.get("additionalProperties"))
        self.assertEqual(CLUSTER_HEALTH_JSON_SCHEMA["properties"]["schema"]["const"], SCHEMA_ID)
        self.assertFalse(CLUSTER_HEALTH_JSON_SCHEMA["properties"]["topology"]["additionalProperties"])


if __name__ == "__main__":
    unittest.main()
