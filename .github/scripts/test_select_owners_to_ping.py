# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import unittest
from unittest.mock import patch

from select_owners_to_ping import Selector


class ModelOwnerNotifications(unittest.TestCase):
    def select(self, files, **env):
        owners = "cglagovichTT,gwangTT,yieldthought,mtairum,uaydonat"
        with patch.dict(os.environ, {"INDIVIDUALS": f"pattern:{owners}:{files}", **env}, clear=True):
            return Selector().select()[0]

    def test_generator_notifies_all_five_current_owners(self):
        self.assertEqual(
            self.select("models/tt_transformers/tt/generator.py"),
            ["cglagovichTT", "gwangTT", "mtairum", "uaydonat", "yieldthought"],
        )

    def test_model_notifications_exclude_author_and_moreh(self):
        self.assertEqual(
            self.select("models/demo.py", PR_AUTHOR_LOGIN="yieldthought", MOREH_TEAM_MEMBERS="uaydonat"),
            ["cglagovichTT", "gwangTT", "mtairum"],
        )

    def test_approved_model_rule_needs_no_notification(self):
        self.assertEqual(self.select("models/demo.py", APPROVED_REVIEWERS="mtairum"), [])

    def test_non_model_rule_still_samples_two(self):
        with patch("select_owners_to_ping.Selector.pick_two", return_value=["gwangTT", "mtairum"]) as sample:
            self.assertEqual(self.select("ttnn/example.py"), ["gwangTT", "mtairum"])
            sample.assert_called_once()

    def test_model_file_in_shared_rule_notifies_all(self):
        self.assertEqual(len(self.select("ttnn/example.py,models/demo.py")), 5)


if __name__ == "__main__":
    unittest.main()
