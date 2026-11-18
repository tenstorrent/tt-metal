# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F267: a fix job's default paths are the model's tt/ dir, also for a seeded run whose ledger is bringup/ledgers/<run>."""

from pathlib import Path

from models.demos.common.bringup.overseer import default_fix_paths

SEEDED = "models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml"


def test_seeded_ledger_fix_paths_are_the_models_tt_dir():
    assert Path(SEEDED).exists()
    assert default_fix_paths(SEEDED) == "models/demos/deepseek_v41_flash_d_p/tt"
