# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Device-resident forward at the paper context length vs the PyTorch reference.

Exercises the tile-aligned (d_model=768, head_dim=64) fast paths that the dummy
checkpoint cannot reach, starting with the folded-batch matmul program configs.
"""

from __future__ import annotations

import pytest
import torch

from models.experimental.chronos_forecast.tests.perf.test_paper_forward import (
    CONTEXT,
    NUM_OUTPUT_PATCHES,
    _load_reference,
)


def _normalized(preds: torch.Tensor, loc_scale) -> torch.Tensor:
    loc, scale = loc_scale
    return torch.asinh((preds.float() - loc[:, None, :]) / scale[:, None, :])


def _run_device(model, context, group_ids):
    import ttnn

    prepared = model.prepare_inputs(context=context, group_ids=group_ids, num_output_patches=NUM_OUTPUT_PATCHES)
    inputs = model.upload_inputs(prepared)
    output_device = None
    try:
        output_device = model.forward_device(inputs)
        got = model.postprocess_output(output_device, prepared.loc_scale, num_output_patches=NUM_OUTPUT_PATCHES)
    finally:
        if output_device is not None:
            ttnn.deallocate(output_device)
        model.deallocate_inputs(inputs)
    return got, prepared.loc_scale


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "batch, group_size",
    [
        pytest.param(64, 1, id="b64_unique_groups"),
        pytest.param(64, 4, id="b64_groups_of_4"),
    ],
)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_device_resident_paper_context_pcc(mesh_device, batch, group_size):
    pytest.importorskip("ttnn")
    from tests.ttnn.utils_for_testing import assert_with_pcc

    from models.experimental.chronos_forecast.tt.model import TtChronos

    if mesh_device.get_num_devices() != 1:
        pytest.skip("single-chip bring-up only (one chip)")

    reference, weight_source = _load_reference()
    model = TtChronos.from_torch_model(mesh_device, reference)

    torch.manual_seed(0)
    context = torch.cumsum(torch.randn(batch, CONTEXT), dim=-1) + 5.0 * torch.randn(batch, 1)
    group_ids = torch.arange(batch, dtype=torch.long) // group_size
    with torch.no_grad():
        expected = reference(
            context=context,
            group_ids=group_ids,
            num_output_patches=NUM_OUTPUT_PATCHES,
        ).quantile_preds

    got, loc_scale = _run_device(model, context, group_ids)
    assert got.shape == expected.shape
    expected_n = _normalized(expected, loc_scale)
    got_n = _normalized(got, loc_scale)
    _, pcc_n = assert_with_pcc(expected_n, got_n, pcc=0.99)
    _, pcc = assert_with_pcc(expected.float(), got, pcc=0.95)
    mae_n = (expected_n - got_n).abs().mean().item()
    print(
        f"\n[PCC] {weight_source} batch={batch} group_size={group_size} "
        f"pcc_norm={pcc_n} pcc={pcc} mae_norm={mae_n:.5f}"
    )
