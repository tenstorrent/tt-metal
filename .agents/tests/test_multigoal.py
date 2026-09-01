from __future__ import annotations

import importlib.machinery
import importlib.util
import pathlib
import sys
import unittest
from unittest import mock


SCRIPT = pathlib.Path(__file__).parents[1] / "scripts" / "multigoal"
LOADER = importlib.machinery.SourceFileLoader("tt_metal_multigoal", str(SCRIPT))
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
assert SPEC is not None
MULTIGOAL = importlib.util.module_from_spec(SPEC)
sys.modules[LOADER.name] = MULTIGOAL
LOADER.exec_module(MULTIGOAL)


class ShellProfileConfigTests(unittest.TestCase):
    def test_parse_args_applies_the_default(self) -> None:
        with mock.patch.object(sys, "argv", ["multigoal", "goal.txt"]):
            args = MULTIGOAL.parse_args()

        self.assertEqual(args.config, ["shell_environment_policy.experimental_use_profile=false"])

    def test_shell_profile_loading_is_disabled_by_default(self) -> None:
        self.assertEqual(
            MULTIGOAL.with_shell_profile_default([]),
            ["shell_environment_policy.experimental_use_profile=false"],
        )

    def test_explicit_shell_profile_setting_is_preserved(self) -> None:
        config = ["model_reasoning_effort=high", "shell_environment_policy.experimental_use_profile=true"]

        self.assertEqual(MULTIGOAL.with_shell_profile_default(config), config)

    def test_unrelated_config_overrides_are_preserved(self) -> None:
        self.assertEqual(
            MULTIGOAL.with_shell_profile_default(["model_reasoning_effort=high"]),
            [
                "shell_environment_policy.experimental_use_profile=false",
                "model_reasoning_effort=high",
            ],
        )


if __name__ == "__main__":
    unittest.main()
