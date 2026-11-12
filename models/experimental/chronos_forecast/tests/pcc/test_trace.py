# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from models.experimental.chronos_forecast.tests.mesh_params import DATA_PARALLEL_MESHES


@pytest.mark.parametrize("device_params", [{"trace_region_size": 20_000_000}], indirect=True)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_tt_forward_trace_pcc(mesh_device):
    ttnn = pytest.importorskip("ttnn")
    from tests.ttnn.utils_for_testing import assert_with_pcc

    from models.experimental.chronos_forecast.reference.chronos2.model import Chronos2Model as RefModel
    from models.experimental.chronos_forecast.tests.golden_helpers import DUMMY_MODEL_PATH
    from models.experimental.chronos_forecast.tt.model import TtChronos
    from models.experimental.chronos_forecast.tt.trace_runner import TtChronosTraceRunner

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


@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 20_000_000, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_tt_forward_trace_two_cq_pcc(mesh_device):
    """Back-to-back pipelined replays on batches that differ from the captured one.

    The first replay's output is copied on CQ0 before the next call, so it is only
    correct if CQ1 waits for that replay to finish before overwriting its inputs.
    """
    ttnn = pytest.importorskip("ttnn")
    from tests.ttnn.utils_for_testing import assert_with_pcc

    from models.experimental.chronos_forecast.reference.chronos2.model import Chronos2Model as RefModel
    from models.experimental.chronos_forecast.tests.golden_helpers import DUMMY_MODEL_PATH
    from models.experimental.chronos_forecast.tt.model import TtChronos
    from models.experimental.chronos_forecast.tt.trace_runner import TtChronosTraceRunner

    reference = RefModel.from_pretrained(DUMMY_MODEL_PATH).eval()
    model = TtChronos.from_torch_model(mesh_device, reference)
    torch.manual_seed(7)
    contexts = [torch.randn(2, 32) for _ in range(3)]
    with torch.no_grad():
        expected = [reference(context=c, num_output_patches=1).quantile_preds for c in contexts]

    prepared = [model.prepare_inputs(context=c, num_output_patches=1) for c in contexts]
    runner = TtChronosTraceRunner(model, prepared[0])
    first_copy = None
    try:
        # Allocated before capture: a tensor allocated afterwards can share addresses with
        # the trace's intermediates and be overwritten by the next replay.
        first_copy = model.forward_device(runner.inputs)
        runner.capture()
        ttnn.copy(runner.execute_pipelined(prepared[1], readback=False), first_copy)
        second = runner.execute_pipelined(prepared[2]).quantile_preds
        first = model.postprocess_output(
            first_copy,
            prepared[1].loc_scale,
            num_output_patches=prepared[1].num_output_patches,
            output_rows=prepared[1].output_rows,
        )
    finally:
        if first_copy is not None:
            ttnn.deallocate(first_copy)
        runner.release()

    assert_with_pcc(expected[1].float(), first, pcc=0.99)
    assert_with_pcc(expected[2].float(), second, pcc=0.99)
    # Different batches must give different forecasts, or the refresh is not being tested.
    assert not torch.allclose(first, second)


@pytest.mark.parametrize("device_params", [{"trace_region_size": 20_000_000}], indirect=True)
@pytest.mark.parametrize("mesh_device", DATA_PARALLEL_MESHES, indirect=True)
def test_tt_forward_trace_stream_pcc(mesh_device):
    """Every streamed batch matches the reference; 6 series also exercises mesh padding."""
    pytest.importorskip("ttnn")
    from tests.ttnn.utils_for_testing import assert_with_pcc

    from models.experimental.chronos_forecast.reference.chronos2.model import Chronos2Model as RefModel
    from models.experimental.chronos_forecast.tests.golden_helpers import DUMMY_MODEL_PATH
    from models.experimental.chronos_forecast.tt.model import TtChronos
    from models.experimental.chronos_forecast.tt.trace_runner import TtChronosTraceRunner

    reference = RefModel.from_pretrained(DUMMY_MODEL_PATH).eval()
    model = TtChronos.from_torch_model(mesh_device, reference)
    torch.manual_seed(3)
    contexts = [torch.randn(6, 32) for _ in range(3)]

    def prepare(context):
        return model.prepare_inputs(context=context, num_output_patches=1)

    runner = TtChronosTraceRunner(model, prepare(contexts[0]))
    try:
        runner.capture()
        results = [result.quantile_preds for result in runner.stream(contexts, prepare)]
    finally:
        runner.release()

    assert len(results) == len(contexts)
    for context, got in zip(contexts, results):
        with torch.no_grad():
            expected = reference(context=context, num_output_patches=1).quantile_preds
        assert got.shape == expected.shape
        assert_with_pcc(expected.float(), got, pcc=0.99)


@pytest.mark.parametrize("device_params", [{"trace_region_size": 20_000_000}], indirect=True)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_tt_forward_trace_repeated_capture_release(mesh_device):
    pytest.importorskip("ttnn")

    from models.experimental.chronos_forecast.reference.chronos2.model import Chronos2Model as RefModel
    from models.experimental.chronos_forecast.tests.golden_helpers import DUMMY_MODEL_PATH
    from models.experimental.chronos_forecast.tt.model import TtChronos
    from models.experimental.chronos_forecast.tt.trace_runner import TtChronosTraceRunner

    reference = RefModel.from_pretrained(DUMMY_MODEL_PATH).eval()
    model = TtChronos.from_torch_model(mesh_device, reference)
    prepared = model.prepare_inputs(context=torch.randn(2, 32), num_output_patches=1)

    for _ in range(3):
        runner = TtChronosTraceRunner(model, prepared)
        runner.capture()
        output = runner.execute().quantile_preds
        assert output.shape == (2, 21, 16)
        runner.release()
        runner.release()
