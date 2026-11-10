# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Run the tests of whichever pre-commit checker a change touches.

The checker-tests hook matches every path below and passes those names here. Each suite runs
only when one of its own paths changed, so a commit that touches one checker does not pay for
the others. A path listed under more than one suite (the shared helper, this runner) runs each
of them.
"""

import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNNER = "scripts/run_checker_tests.py"

# (test module, repo-relative paths that should run it). The hook's `files` filter has to
# match every path here; scripts/test_run_checker_tests.py checks that it does.
CHECKS = (
    (
        "scripts/test_run_checker_tests.py",
        frozenset({RUNNER, "scripts/test_run_checker_tests.py"}),
    ),
    (
        "scripts/test_check_bool_from_arithmetic.py",
        frozenset(
            {
                RUNNER,
                "scripts/precommit_common.py",
                "scripts/check_bool_from_arithmetic.py",
                "scripts/test_check_bool_from_arithmetic.py",
            }
        ),
    ),
    (
        "scripts/test_check_redefined_helpers.py",
        frozenset(
            {
                RUNNER,
                "scripts/precommit_common.py",
                "scripts/check_redefined_helpers.py",
                "scripts/test_check_redefined_helpers.py",
            }
        ),
    ),
    (
        "tt_metal/tt-llk/infra/tests/test_check_dest_to_src_wait.py",
        frozenset(
            {
                RUNNER,
                "tt_metal/tt-llk/infra/check_dest_to_src_wait.py",
                "tt_metal/tt-llk/infra/check_mutex_balance.py",
                "tt_metal/tt-llk/infra/tests/test_check_dest_to_src_wait.py",
            }
        ),
    ),
)

PYTEST_ARGS = ("-q", "-p", "no:cacheprovider", "--noconftest")


def _rel(path):
    """Repo-relative form of a pre-commit path. Absolute paths are accepted too."""
    path = path.replace("\\", "/")
    if os.path.isabs(path):
        path = os.path.relpath(path, REPO_ROOT)
    return os.path.normpath(path).replace("\\", "/")


def tests_for(paths):
    """Test modules to run for these changed paths, in a stable order.

    No paths means every suite, which is a manual run of this script. pre-commit always passes
    the paths that matched the hook.
    """
    rels = {_rel(path) for path in paths}
    if not rels:
        return [test for test, _triggers in CHECKS]
    return [test for test, triggers in CHECKS if rels & triggers]


def main(argv=None):
    selected = tests_for(sys.argv[1:] if argv is None else argv)
    if not selected:
        return 0
    return subprocess.call([sys.executable, "-m", "pytest", *PYTEST_ARGS, *selected])


if __name__ == "__main__":
    sys.exit(main())
