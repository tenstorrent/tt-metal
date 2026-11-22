# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""MB.4: the fork test of ttnn.bringup.ckv_bfp4_rows (ttnn/ttnn/bringup/ckv_bfp4_ttnn/tests/unit/test_ckv_bfp4_rows.py)
run on the resident runner's held mesh (only model test files share a batch process, core/batch.shareable).

    python -m models.demos.common.bringup.resident submit --spec $BRINGUP_SPEC --cmd "scripts/run_safe_pytest.sh
        --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/test_ckv_bfp4_rows.py"
"""

from ttnn.bringup.ckv_bfp4_ttnn.tests.unit.test_ckv_bfp4_rows import (  # noqa: F401
    test_ckv_bfp4_rows,
    test_ckv_bfp4_rows_refusals,
)

BATCH_SHARED = True  # (a mesh-only op test: 0 model loads)
