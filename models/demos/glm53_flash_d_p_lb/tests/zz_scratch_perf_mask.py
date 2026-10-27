# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Scratch: tests/test_perf.py with ttKDA's chunk-preparation bf16 output mask overridden (GLM_KC_MASK, e.g. 0x6 =
final_decay in fp32; ttKDA's default 0x26). A/B of what keeping exp(G_last) in fp32 costs.

    GLM_KC_MASK=0x6 ... scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/glm53_flash_d_p_lb/tests/zz_scratch_perf_mask.py -s
"""

import os

from models.demos.deepseek_v3_d_p.tt.kda import recurrence
from models.demos.glm53_flash_d_p_lb.tests import test_perf as _tp

if os.environ.get("GLM_KC_MASK"):
    recurrence.KDA_PREP_OUTPUT_BF16_MASK = int(os.environ["GLM_KC_MASK"], 0)
print(f"[perf-mask] KDA prep bf16 mask {recurrence.KDA_PREP_OUTPUT_BF16_MASK:#x}", flush=True)
pytestmark = _tp.pytestmark
test_perf = _tp.test_perf
