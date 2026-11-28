# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""WA.2: the fork tests of ttnn.bringup.grouped_matmul's bank path (option w_layout,
ttnn/ttnn/bringup/grouped_matmul_ttnn/tests/unit/test_grouped_matmul_bank.py) on the resident runner's held mesh
(only model test files share a batch process, core/batch.shareable).

    python -m models.demos.common.bringup.resident submit --spec $BRINGUP_SPEC --cmd "scripts/run_safe_pytest.sh
        --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/woa/test_woa_fork_bank.py"
"""

from ttnn.bringup.grouped_matmul_ttnn.tests.unit.test_grouped_matmul_bank import (  # noqa: F401
    test_bank_pack_cpu,
    test_grouped_matmul_bank,
    test_grouped_matmul_bank_refusals,
)

BATCH_SHARED = True  # (a mesh-only op test: 0 model loads)
