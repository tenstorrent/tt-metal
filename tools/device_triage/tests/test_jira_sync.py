# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import base64
import tempfile
import unittest
from pathlib import Path

from device_triage import jira_sync

HOST = "bh-glx-110-d01u02"


def att(att_id: str, filename: str, created: str) -> dict:
    return {"id": att_id, "filename": filename, "size": 1, "mimeType": "", "created": created}


class AdfTest(unittest.TestCase):
    def test_paragraphs_breaks_and_cards(self):
        doc = {
            "type": "doc",
            "content": [
                {
                    "type": "paragraph",
                    "content": [
                        {"type": "text", "text": "Node: "},
                        {"type": "inlineCard", "attrs": {"url": "https://x/DIIM-1"}},
                    ],
                },
                {
                    "type": "codeBlock",
                    "content": [
                        {"type": "text", "text": "line1"},
                        {"type": "hardBreak"},
                        {"type": "text", "text": "line2"},
                    ],
                },
            ],
        }
        self.assertEqual(jira_sync.adf_text(doc), "Node: https://x/DIIM-1\nline1\nline2\n")


class HcAttachmentTest(unittest.TestCase):
    def test_roles_and_jobs(self):
        role = jira_sync.hc_attachment_role
        self.assertEqual(role(f"{HOST}-21863.log", HOST), ("suite", "21863"))
        self.assertEqual(role(f"{HOST}-131689-131689.log", HOST), ("suite", "131689"))
        self.assertEqual(role("diag_report-29848680-tkt4q.json", HOST), ("diag", "29848680-tkt4q"))
        self.assertEqual(role("gddr_fast-24936.log", HOST), ("check:gddr_fast", "24936"))
        self.assertEqual(role("triage_host_side-24936.txt", HOST), ("triage", "24936"))
        self.assertEqual(role("diag_report.json", HOST), ("diag", None))
        self.assertEqual(role("telemetry_bh-glx-110-d01u02-24936.prom", HOST), ("skip", None))
        self.assertEqual(role("console-24936.log", HOST), ("skip", None))

    def test_unsuffixed_files_pair_with_nearest_run(self):
        record = {
            "attachments": [
                att("1", f"{HOST}-20799.log", "2026-07-10T03:00:00.000-0700"),
                att("2", f"{HOST}-21166.log", "2026-07-12T03:00:00.000-0700"),
                att("3", "diag_report.json", "2026-07-10T03:00:05.000-0700"),
                att("4", "diag_report.json", "2026-07-12T03:00:04.000-0700"),
                att("5", "diag_report-21863.json", "2026-07-14T03:00:00.000-0700"),
                att("6", "eth_bandwidth.log", "2026-07-20T03:00:00.000-0700"),
            ]
        }
        runs = {r["job"]: {role: a["id"] for role, a in r["files"].items()} for r in jira_sync.hc_runs(record, HOST)}
        self.assertEqual(
            runs, {"20799": {"suite": "1", "diag": "3"}, "21166": {"suite": "2", "diag": "4"}, "21863": {"diag": "5"}}
        )

    def test_truncated_json_keeps_complete_values(self):
        text = '{"host": "x", "phases": {"snapshot": {"checks": [{"name": "a", "status": "FAIL"}, {"name": "b", "sta'
        doc, state = jira_sync.load_truncated_json(text)
        self.assertEqual(state, "repaired")
        self.assertEqual(doc["phases"]["snapshot"]["checks"][0], {"name": "a", "status": "FAIL"})
        self.assertEqual(jira_sync.load_truncated_json('{"a": 1}'), ({"a": 1}, "complete"))
        self.assertEqual(jira_sync.load_truncated_json('"abc'), (None, "unreadable"))

    def test_unwrap_base64(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "diag_report.json"
            path.write_bytes(b'"' + base64.b64encode(b'{"tool_version": "0.3.0"}') + b'"')
            jira_sync.unwrap_base64(path)
            self.assertEqual(path.read_bytes(), b'{"tool_version": "0.3.0"}')
            path.write_bytes(b'{"plain": true}')
            jira_sync.unwrap_base64(path)
            self.assertEqual(path.read_bytes(), b'{"plain": true}')

    def test_resolve_short_host(self):
        known = {"bh-glx-120-b05u02", "bh-glx-110-c08u02", "bh-glx-120-c08u02", "bh-glx-b05u02"}
        self.assertEqual(jira_sync._resolve_host("bh-glx-b05u02", known), "bh-glx-120-b05u02")
        self.assertIsNone(jira_sync._resolve_host("bh-glx-c08u02", known))
        self.assertEqual(jira_sync._resolve_host("qb2-120-p03t08", known), "qb2-120-p03t08")


if __name__ == "__main__":
    unittest.main()
