# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reporting must work for source-tree TTNN installs without package metadata."""

import importlib.metadata
import unittest
from unittest.mock import patch

from models.experimental.voxcpm2.validation.replay_component import runtime_version


class TestRuntimeMetadata(unittest.TestCase):
    def test_source_tree_without_distribution_metadata(self):
        with patch(
            "importlib.metadata.version", side_effect=importlib.metadata.PackageNotFoundError("ttnn")
        ) as version:
            self.assertIsNone(runtime_version())
            version.assert_called_once_with("ttnn")

    def test_installed_distribution_reports_version(self):
        with patch("importlib.metadata.version", return_value="0.70.0") as version:
            self.assertEqual(runtime_version(), "0.70.0")
            version.assert_called_once_with("ttnn")
