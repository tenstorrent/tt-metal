# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""RT-DETR end to end with every ttnn call that has a Quasar fork routed to ttnn.experimental.quasar.

Same reference, inputs and PCC checks as tests/pcc.py::test_rtdetr_model; only the TT model differs.
RTDETR_NUM_LAYERS reduces both models for simulator runs.
"""

import pytest
from transformers import RTDetrForObjectDetection

from models.demos.vision.detection.rtdetr.quasar.tt.model import TtRTDetrModel
from models.demos.vision.detection.rtdetr.tests.pcc import MODEL_NAME, run_rtdetr_model_test


@pytest.mark.timeout(14400)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_rtdetr_e2e(device):
    run_rtdetr_model_test(device, MODEL_NAME, RTDetrForObjectDetection, tt_model_class=TtRTDetrModel)
