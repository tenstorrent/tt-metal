# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# Experiment only: a kernel that hangs, so the next test in the same process shows whether the launch recovers.

import pytest
from helpers.test_config import BuildMode, TestConfig


@pytest.mark.parametrize("kind", ["stall", "spin"])
def test_hang_probe(kind):
    if TestConfig.BUILD_MODE == BuildMode.PRODUCE:
        pytest.skip()
    cfg = TestConfig(f"sources/hang_{kind}_test.cpp")
    cfg.generate_variant_hash()
    cfg.build_elfs()
    cfg.run_elf_files()
    with pytest.raises(Exception) as err:
        cfg.wait_for_tensix_operations_finished(timeout=2)
    print(f"HANG_RAISED {kind} {type(err.value).__name__}: {str(err.value)[:300]}")
