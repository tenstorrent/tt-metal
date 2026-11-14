# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The consumer migration script: which checkouts move to the pinned action."""

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parent / "cachew_checkout.py"
spec = importlib.util.spec_from_file_location("cachew_checkout", SCRIPT)
migration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(migration)

PLAIN = "actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd # v6.0.2"
PINNED = f"tenstorrent/.github/actions/cachew-checkout@{migration.ACTION_SHA} # cachew checkout"
STALE = "tenstorrent/.github/actions/cachew-checkout@" + "0" * 40 + " # cachew checkout"
# The action's location before it moved to tenstorrent/.github.
LEGACY = "tenstorrent/tt-metal/.github/actions/cachew-checkout@" + "1" * 40 + " # cachew checkout"


class MigrationTests(unittest.TestCase):
    def test_self_hosted_jobs_move_to_the_pinned_action_and_hosted_jobs_keep_checkout(self):
        before = f"""on: push
jobs:
  build:
    runs-on: tt-ubuntu-2204-large-stable
    steps:
      - uses: {PLAIN}
  matrix:
    runs-on: >-
      ${{{{ inputs.label }}}}
    steps:
      - uses: {STALE}
  hosted:
    runs-on: ubuntu-22.04
    steps:
      - uses: {PLAIN}
      - uses: {STALE}
  arm:
    runs-on: "ubuntu-24.04-arm"
    steps:
      - uses: {PINNED}
  kept:
    runs-on: tt-ubuntu-2204-large-stable
    steps:
      - uses: {PLAIN} cachew: keep
"""
        expected = f"""on: push
jobs:
  build:
    runs-on: tt-ubuntu-2204-large-stable
    steps:
      - uses: {PINNED}
  matrix:
    runs-on: >-
      ${{{{ inputs.label }}}}
    steps:
      - uses: {PINNED}
  hosted:
    runs-on: ubuntu-22.04
    steps:
      - uses: {PLAIN}
      - uses: {PLAIN}
  arm:
    runs-on: "ubuntu-24.04-arm"
    steps:
      - uses: {PLAIN}
  kept:
    runs-on: tt-ubuntu-2204-large-stable
    steps:
      - uses: {PLAIN} cachew: keep
"""
        self.assertEqual(migration.rewrite(before), expected)

    def test_composite_actions_are_migrated_unless_kept(self):
        before = f"runs:\n  using: composite\n  steps:\n    - uses: {PLAIN}\n    - uses: {PLAIN} cachew: keep\n"
        expected = f"runs:\n  using: composite\n  steps:\n    - uses: {PINNED}\n    - uses: {PLAIN} cachew: keep\n"
        self.assertEqual(migration.rewrite(before), expected)

    def test_pins_to_the_old_tt_metal_location_move_to_tenstorrent_github(self):
        before = f"""on: push
jobs:
  build:
    runs-on: tt-ubuntu-2204-large-stable
    steps:
      - uses: {LEGACY}
  hosted:
    runs-on: ubuntu-22.04
    steps:
      - uses: {LEGACY}
"""
        expected = f"""on: push
jobs:
  build:
    runs-on: tt-ubuntu-2204-large-stable
    steps:
      - uses: {PINNED}
  hosted:
    runs-on: ubuntu-22.04
    steps:
      - uses: {PLAIN}
"""
        self.assertEqual(migration.rewrite(before), expected)

    def test_any_pin_reports_the_old_location(self):
        match = migration.ANY_PIN.search(LEGACY)
        self.assertEqual(match["repo"], migration.LEGACY_REPO)
        self.assertEqual(migration.ANY_PIN.search(PINNED)["repo"], migration.ACTION_REPO)

if __name__ == "__main__":
    unittest.main()
