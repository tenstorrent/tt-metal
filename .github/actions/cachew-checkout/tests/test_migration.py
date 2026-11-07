# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The consumer migration script: which checkouts move to the pinned action."""

import importlib.util
from pathlib import Path
import subprocess
import unittest


ACTION = Path(__file__).resolve().parents[1]
SCRIPT = ACTION.parents[1] / "scripts/utils/cachew_checkout.py"
spec = importlib.util.spec_from_file_location("cachew_checkout", SCRIPT)
migration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(migration)

PLAIN = "actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd # v6.0.2"
PINNED = f"tenstorrent/tt-metal/.github/actions/cachew-checkout@{migration.ACTION_SHA} # cachew checkout"
STALE = "tenstorrent/tt-metal/.github/actions/cachew-checkout@" + "0" * 40 + " # cachew checkout"


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

    def test_pins_cover_every_file_consumers_execute(self):
        tracked = subprocess.check_output(["git", "ls-files"], cwd=ACTION, text=True).split()
        executed = {path.split("/")[0] for path in tracked if path != "README.md" and not path.startswith("tests/")}
        self.assertEqual(executed, set(migration.RUNTIME_FILES))


if __name__ == "__main__":
    unittest.main()
