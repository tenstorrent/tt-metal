# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import unittest

from device_triage import jira


class JiraTest(unittest.TestCase):
    def test_adf_paragraphs_breaks_and_cards(self):
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
        self.assertEqual(jira.adf_text(doc), "Node: https://x/DIIM-1\nline1\nline2\n")

    def test_normalize_keeps_text_fields(self):
        issue = {
            "key": "DIIM-1",
            "fields": {
                "summary": "bh-glx-120-b02u08 only 24/32 chips",
                "status": {"name": "Backlog"},
                "created": "2026-10-13T06:12:01.123-0700",
                "description": {
                    "type": "doc",
                    "content": [{"type": "paragraph", "content": [{"type": "text", "text": "d"}]}],
                },
                "comment": {"comments": [{"body": "c"}]},
            },
        }
        self.assertEqual(
            jira.normalize(issue),
            {
                "key": "DIIM-1",
                "summary": "bh-glx-120-b02u08 only 24/32 chips",
                "status": "Backlog",
                "created": "2026-10-13T06:12:01.123-0700",
                "description": "d",
                "comments": [{"body": "c"}],
            },
        )


if __name__ == "__main__":
    unittest.main()
