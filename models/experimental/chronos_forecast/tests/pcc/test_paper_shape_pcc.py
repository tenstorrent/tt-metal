# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Device-resident forward at the paper context length vs the PyTorch reference.

Exercises the tile-aligned (d_model=768, head_dim=64) fast paths that the dummy
checkpoint cannot reach: folded-batch matmuls, fused RoPE, SDPA configs, and the
opt-in reduced-precision modes (gated on quantile accuracy against held-out data).
"""

from __future__ import annotations

import math

import pytest
import torch

from models.experimental.chronos_forecast.tests.perf.test_paper_forward import (
    _QUANTILES,
    CONTEXT,
    NUM_OUTPUT_PATCHES,
    PREDICTION_LENGTH,
    _load_reference,
)


def _normalized(preds: torch.Tensor, loc_scale) -> torch.Tensor:
    loc, scale = loc_scale
    return torch.asinh((preds.float() - loc[:, None, :]) / scale[:, None, :])


def _seasonal_series(batch: int, length: int) -> torch.Tensor:
    """Forecastable series: level + trend + one seasonality + noise."""
    t = torch.arange(length, dtype=torch.float32)
    period = torch.randint(12, 200, (batch, 1)).float()
    phase = 2 * math.pi * torch.rand(batch, 1)
    amplitude = 1.0 + 3.0 * torch.rand(batch, 1)
    trend = 1e-3 * torch.randn(batch, 1) * t
    level = 5.0 * torch.randn(batch, 1)
    return level + trend + amplitude * torch.sin(2 * math.pi * t / period + phase) + 0.3 * torch.randn(batch, length)


def _wql(preds: torch.Tensor, target: torch.Tensor) -> float:
    """Weighted quantile loss (mean over quantiles), as in fev-bench / the Chronos papers."""
    q = torch.tensor(_QUANTILES, dtype=torch.float32)[None, :, None]
    err = target[:, None, :] - preds.float()
    pinball = torch.maximum(q * err, (q - 1) * err)
    return (2 * pinball.sum(dim=(0, 2)) / target.abs().sum()).mean().item()


def _group_ids(batch: int, groups) -> torch.Tensor:
    """``groups`` is a uniform group size, or "mixed" for shuffled groups of 1..7 series."""
    if groups != "mixed":
        return torch.arange(batch, dtype=torch.long) // groups
    sizes, total = [], 0
    while total < batch:
        sizes.append(min(1 + len(sizes) % 7, batch - total))
        total += sizes[-1]
    group_ids = torch.repeat_interleave(torch.arange(len(sizes)), torch.tensor(sizes))
    return group_ids[torch.randperm(batch, generator=torch.Generator().manual_seed(1))]


def _run_device(model, context, group_ids):
    import ttnn

    prepared = model.prepare_inputs(context=context, group_ids=group_ids, num_output_patches=NUM_OUTPUT_PATCHES)
    inputs = model.upload_inputs(prepared)
    output_device = None
    try:
        output_device = model.forward_device(inputs)
        got = model.postprocess_output(
            output_device,
            prepared.loc_scale,
            num_output_patches=NUM_OUTPUT_PATCHES,
            output_rows=prepared.output_rows,
        )
    finally:
        if output_device is not None:
            ttnn.deallocate(output_device)
        model.deallocate_inputs(inputs)
    return got, prepared.loc_scale


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "batch, groups, l1_resident",
    [
        pytest.param(64, 1, False, id="b64_unique_groups"),
        pytest.param(64, 4, False, id="b64_groups_of_4"),
        # 48-series chunks: one full chunk plus a 16-series remainder.
        pytest.param(64, 1, True, id="b64_unique_groups_l1"),
        # Chunks round down to whole 32-series group blocks.
        pytest.param(64, 4, True, id="b64_groups_of_4_l1"),
        # Non-uniform blocks with dummy series; the L1 case slices the block mask per chunk.
        pytest.param(100, "mixed", False, id="b100_mixed_groups"),
        pytest.param(100, "mixed", True, id="b100_mixed_groups_l1"),
        # A 128-series block exceeds one L1 chunk, so it runs from DRAM.
        pytest.param(100, 33, True, id="b100_groups_of_33_l1_falls_back"),
    ],
)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_device_resident_paper_context_pcc(mesh_device, batch, groups, l1_resident):
    pytest.importorskip("ttnn")
    from tests.ttnn.utils_for_testing import assert_with_pcc

    from models.experimental.chronos_forecast.tt.model import TtChronos
    from models.experimental.chronos_forecast.tt.program_configs import TtChronosPrecision

    if mesh_device.get_num_devices() != 1:
        pytest.skip("single-chip bring-up only (one chip)")

    reference, weight_source = _load_reference()
    l1_chunk_tokens = TtChronosPrecision().l1_chunk_tokens() if l1_resident else None
    model = TtChronos.from_torch_model(mesh_device, reference, l1_chunk_tokens=l1_chunk_tokens)

    torch.manual_seed(0)
    context = torch.cumsum(torch.randn(batch, CONTEXT), dim=-1) + 5.0 * torch.randn(batch, 1)
    group_ids = _group_ids(batch, groups)
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
        f"\n[PCC] {weight_source} batch={batch} groups={groups} l1={l1_resident} "
        f"pcc_norm={pcc_n} pcc={pcc} mae_norm={mae_n:.5f}"
    )


# Relative WQL increase allowed over the PyTorch reference on the same held-out window.
_MAX_REL_WQL_INCREASE = 0.01


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "precision_name, group_size, l1_resident",
    [
        pytest.param("default", 1, False, id="default"),
        pytest.param("default", 4, False, id="default_groups_of_4"),
        pytest.param("bf8_attention", 1, False, id="bf8_attention"),
        pytest.param("bf8_weights", 1, False, id="bf8_weights"),
        pytest.param("bf8_ff_hidden", 1, False, id="bf8_ff_hidden"),
        pytest.param("bf8_sublayer_out", 1, False, id="bf8_sublayer_out"),
        pytest.param("lofi_ff", 1, False, id="lofi_ff"),
        pytest.param("performance", 1, False, id="performance"),
        pytest.param("performance", 4, False, id="performance_groups_of_4"),
        pytest.param("default", 1, True, id="default_l1"),
        pytest.param("performance", 1, True, id="performance_l1"),
        pytest.param("performance", 4, True, id="performance_l1_groups_of_4"),
    ],
)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_precision_quantile_accuracy(mesh_device, precision_name, group_size, l1_resident):
    pytest.importorskip("ttnn")
    from tests.ttnn.utils_for_testing import assert_with_pcc

    from models.experimental.chronos_forecast.tt.model import TtChronos
    from models.experimental.chronos_forecast.tt.program_configs import TtChronosPrecision

    if mesh_device.get_num_devices() != 1:
        pytest.skip("single-chip bring-up only (one chip)")

    if precision_name == "default":
        precision = TtChronosPrecision()
    elif precision_name == "performance":
        precision = TtChronosPrecision.performance()
    else:
        precision = TtChronosPrecision(**{precision_name: True})
    reference, weight_source = _load_reference()
    if weight_source != "checkpoint":
        pytest.skip("quantile accuracy is only meaningful with the real checkpoint")
    l1_chunk_tokens = precision.l1_chunk_tokens() if l1_resident else None
    model = TtChronos.from_torch_model(mesh_device, reference, precision, l1_chunk_tokens=l1_chunk_tokens)

    batch = 256
    torch.manual_seed(0)
    series = _seasonal_series(batch, CONTEXT + PREDICTION_LENGTH)
    context, target = series[:, :CONTEXT], series[:, CONTEXT:]
    group_ids = torch.arange(batch, dtype=torch.long) // group_size
    with torch.no_grad():
        expected = reference(context=context, group_ids=group_ids, num_output_patches=NUM_OUTPUT_PATCHES).quantile_preds

    got, loc_scale = _run_device(model, context, group_ids)
    _, pcc_n = assert_with_pcc(_normalized(expected, loc_scale), _normalized(got, loc_scale), pcc=0.99)
    wql_ref, wql_tt = _wql(expected, target), _wql(got, target)
    rel = (wql_tt - wql_ref) / wql_ref
    print(
        f"\n[WQL] precision={precision_name} group_size={group_size} l1={l1_resident} "
        f"wql_ref={wql_ref:.5f} wql_tt={wql_tt:.5f} rel={rel:+.4%} pcc_norm={pcc_n:.6f}"
    )
    assert rel <= _MAX_REL_WQL_INCREASE, f"WQL {wql_tt:.5f} is {rel:.2%} above the reference {wql_ref:.5f}"
