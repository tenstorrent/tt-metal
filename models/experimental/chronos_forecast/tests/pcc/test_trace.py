# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch


@pytest.mark.parametrize("device_params", [{"trace_region_size": 20_000_000}], indirect=True)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_tt_forward_trace_pcc(mesh_device):
    ttnn = pytest.importorskip("ttnn")
    from tests.ttnn.utils_for_testing import assert_with_pcc

    from models.experimental.chronos_forecast.reference.chronos2.model import Chronos2Model as RefModel
    from models.experimental.chronos_forecast.tests.golden_helpers import DUMMY_MODEL_PATH
    from models.experimental.chronos_forecast.tt.model import TtChronos
    from models.experimental.chronos_forecast.tt.trace_runner import TtChronosTraceRunner

    if mesh_device.get_num_devices() != 1:
        pytest.skip("single-chip bring-up only (one chip)")

    reference = RefModel.from_pretrained(DUMMY_MODEL_PATH).eval()
    model = TtChronos.from_torch_model(mesh_device, reference)
    torch.manual_seed(0)
    context = torch.randn(2, 32)
    with torch.no_grad():
        expected = reference(context=context, num_output_patches=1).quantile_preds

    prepared = model.prepare_inputs(context=context, num_output_patches=1)
    runner = TtChronosTraceRunner(model, prepared)
    try:
        runner.capture()
        first = runner.execute().quantile_preds
        second = runner.execute().quantile_preds
    finally:
        runner.release()

    assert first.shape == expected.shape
    assert_with_pcc(expected.float(), first, pcc=0.99)
    assert_with_pcc(first, second, pcc=0.999)
