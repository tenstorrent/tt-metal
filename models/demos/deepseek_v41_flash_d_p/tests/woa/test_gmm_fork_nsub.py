# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""FIX.42: the fork tests of ttnn.grouped_matmul's bank-path column sub-blocks (GMM_N_SUB,
ttnn/ttnn/bringup/grouped_matmul_ttnn/tests/unit/test_grouped_matmul_nsub.py) on the resident runner's held mesh
(only model test files share a batch process, core/batch.shareable).

    python -m models.demos.common.bringup.resident submit --spec $BRINGUP_SPEC --cmd "scripts/run_safe_pytest.sh
        --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/woa/test_gmm_fork_nsub.py"
"""

from ttnn.bringup.grouped_matmul_ttnn.tests.unit.test_grouped_matmul_nsub import (  # noqa: F401
    test_nsub_same_bytes,
    test_nsub_wide_cores,
)

BATCH_SHARED = True  # (a mesh-only op test: 0 model loads)
