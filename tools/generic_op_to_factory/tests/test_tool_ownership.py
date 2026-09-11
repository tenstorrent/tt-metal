# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tool entry points need neither the evaluator package nor third-party drivers."""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "module",
    ["export_run", "prepare_baseline", "compare_baseline", "migration_workflow", "prepare_target", "validate_port"],
)
def test_cli_is_independent_of_evaluator(module, tmp_path):
    repository = Path(__file__).resolve().parents[3]
    bootstrap = """
import importlib.abc
import runpy
import sys

class RejectEvaluator(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'eval' or fullname.startswith('eval.'):
            raise AssertionError('migration tool imported the evaluator: ' + fullname)

sys.meta_path.insert(0, RejectEvaluator())
sys.path.insert(0, sys.argv.pop(1))
module = sys.argv.pop(1)
runpy.run_module(module, run_name='__main__')
"""
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            bootstrap,
            str(repository),
            f"tools.generic_op_to_factory.{module}",
            "--help",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "usage:" in result.stdout
