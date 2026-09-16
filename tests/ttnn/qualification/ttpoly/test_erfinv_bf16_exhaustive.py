# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Exhaustive BF16 test using the packaged canonical reference."""
from pathlib import Path
import sys


def test_erfinv_bf16_exhaustive(tmp_path):
    candidate = Path(__file__).resolve().parents[4]
    runtime = (candidate / "tests/ttnn/qualification/ttpoly/ttpoly_qualification/support").resolve(strict=True)
    if not runtime.is_relative_to(candidate):
        raise RuntimeError("packaged runtime escapes candidate root")
    sys.path.insert(0, str(runtime))
    from ttpoly import ttmetal_public_tests

    if not Path(ttmetal_public_tests.__file__).resolve().is_relative_to(runtime):
        raise RuntimeError("wrong packaged canonical runtime imported")
    ttmetal_public_tests.run_package_test("erfinv", tmp_path, candidate_root=candidate, support_root=runtime)
