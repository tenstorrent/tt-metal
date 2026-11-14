# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Token packing (``pack_tokens``) on device against the FP32 reference and the unpacked path.

Packing moves tokens, so a packed forecast must track the reference as closely as the unpacked
one: normalized PCC >= 0.99 (as in ``test_paper_shape_pcc``) and a worst-row quantile NRMSE at
most ``MAX_NRMSE_INCREASE`` above the unpacked path of the same model. The envelope covers the
short contexts and horizons where packing puts many series in one sequence, at batches large
enough to fill L1 chunks.
"""

from __future__ import annotations

import math

import pytest
import torch

from models.experimental.chronos_forecast.tests.mesh_params import DATA_PARALLEL_MESHES
from models.experimental.chronos_forecast.tests.pcc.test_paper_shape_pcc import _normalized
from models.experimental.chronos_forecast.tests.perf.test_paper_forward import CKPT

OUTPUT_PATCH = 16
MAX_NRMSE_INCREASE = 0.02
DEVICE_PARAMS = [{"trace_region_size": 200_000_000}]
# (context, horizon, batch): packing engages in every row but the last (paper shape, 133 -> 160 tokens).
ENVELOPE = [
    (17, 16, 256),
    (64, 64, 256),
    (100, 48, 64),
    (160, 32, 1024),
    (256, 64, 1024),
    (512, 16, 1024),
    (512, 64, 64),
    (512, 64, 1024),
    (2048, 64, 64),
]


def _require_checkpoint():
    if not (CKPT / "config.json").is_file() or not (CKPT / "model.safetensors").is_file():
        pytest.skip(f"Chronos-2 checkpoint not found at {CKPT}")


def _max_row_nrmse(pred: torch.Tensor, ref: torch.Tensor) -> float:
    pred, ref = pred.float().flatten(1), ref.float().flatten(1)
    rmse = (pred - ref).pow(2).mean(-1).sqrt()
    return float((rmse / ref.pow(2).mean(-1).sqrt().clamp_min(1e-6)).max())


def _series(batch: int, length: int, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    t = torch.arange(length, dtype=torch.float32)
    period = torch.randint(8, 64, (batch, 1), generator=gen).float()
    level = torch.randn(batch, 1, generator=gen) * 5
    return level + torch.sin(2 * torch.pi * t / period) + 0.01 * t + 0.1 * torch.randn(batch, length, generator=gen)


def _forecast(model, context, patches, *, pack_tokens, group_ids=None):
    from models.experimental.chronos_forecast.tt.trace_runner import TtChronosTraceRunner

    model.pack_tokens = pack_tokens
    prepared = model.prepare_inputs(context, None, group_ids, None, None, num_output_patches=patches)
    runner = TtChronosTraceRunner(model, prepared)
    try:
        runner.capture()
        first = runner.execute(prepared).quantile_preds
        second = runner.execute(prepared).quantile_preds
    finally:
        runner.release()
    assert torch.equal(first, second), "repeated identical calls must be bit-identical"
    return first, prepared


def _models(mesh_device, precision_name):
    from models.experimental.chronos_forecast.reference.chronos2.model import Chronos2Model as RefModel
    from models.experimental.chronos_forecast.tt.model import TtChronos
    from models.experimental.chronos_forecast.tt.program_configs import TtChronosPrecision

    _require_checkpoint()
    reference = RefModel.from_pretrained(str(CKPT)).eval()
    precision = TtChronosPrecision.performance() if precision_name == "performance" else TtChronosPrecision()
    model = TtChronos.from_torch_model(mesh_device, reference, precision, l1_chunk_tokens=precision.l1_chunk_tokens())
    return reference, model


def _check_packed(reference, model, context, patches):
    """Packed vs reference and vs the unpacked path; returns (k, packed NRMSE, unpacked NRMSE)."""
    from tt_py_test_utils_common.utils_for_testing import assert_with_pcc

    with torch.no_grad():
        expected = reference(context=context, num_output_patches=patches).quantile_preds.float()
    packed, prepared = _forecast(model, context, patches, pack_tokens=None)
    unpacked, _ = _forecast(model, context, patches, pack_tokens=False)
    assert packed.shape == unpacked.shape == expected.shape
    assert_with_pcc(_normalized(expected, prepared.loc_scale), _normalized(packed, prepared.loc_scale), pcc=0.99)
    packed_error, unpacked_error = _max_row_nrmse(packed, expected), _max_row_nrmse(unpacked, expected)
    assert packed_error <= unpacked_error + MAX_NRMSE_INCREASE, (packed_error, unpacked_error)
    return prepared.pack, packed_error, unpacked_error


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize("precision_name", ["default", "performance"])
def test_packed_forecast_over_envelope(mesh_device, precision_name):
    reference, model = _models(mesh_device, precision_name)
    packs = []
    for context_len, horizon, batch in ENVELOPE:
        patches = math.ceil(horizon / OUTPUT_PATCH)
        k, packed_error, unpacked_error = _check_packed(
            reference, model, _series(batch, context_len, seed=context_len + batch), patches
        )
        packs.append(k)
        print(
            f"\n[TOKEN PACKING] {precision_name} ctx={context_len} h={horizon} b={batch} k={k} "
            f"max_row_nrmse packed={packed_error:.4f} unpacked={unpacked_error:.4f}"
        )
    assert max(packs) > 1, "packing never engaged"
    assert packs[-1] == 1, "the paper shape must stay unpacked"


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_groups_are_never_packed(mesh_device):
    from tt_py_test_utils_common.utils_for_testing import assert_with_pcc

    reference, model = _models(mesh_device, "default")
    context = _series(32, 512, seed=2)
    group_ids = torch.arange(32) // 4
    with torch.no_grad():
        expected = reference(context=context, group_ids=group_ids, num_output_patches=4).quantile_preds.float()
    out, prepared = _forecast(model, context, 4, pack_tokens=None, group_ids=group_ids)
    assert prepared.pack == 1 and not prepared.unique_groups
    assert_with_pcc(_normalized(expected, prepared.loc_scale), _normalized(out, prepared.loc_scale), pcc=0.99)


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [DATA_PARALLEL_MESHES[1]], indirect=True)
def test_packed_forecast_on_mesh(mesh_device):
    reference, model = _models(mesh_device, "performance")
    for batch in (6, 1024):  # 6 does not split evenly over the chips
        k, packed_error, unpacked_error = _check_packed(reference, model, _series(batch, 512, seed=batch), 4)
        print(f"\n[TOKEN PACKING] mesh b={batch} k={k} packed={packed_error:.4f} unpacked={unpacked_error:.4f}")


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_trace_rejects_horizon_change(mesh_device, expect_error):
    """ctx512 b64 packs 6 per 224 rows at every horizon up to 64, so only the horizon check catches it."""
    from models.experimental.chronos_forecast.tt.trace_runner import TtChronosTraceRunner

    _, model = _models(mesh_device, "default")
    model.pack_tokens = None
    context = _series(64, 512, seed=3)
    prepared = model.prepare_inputs(context, num_output_patches=4)
    shorter = model.prepare_inputs(context, num_output_patches=3)
    assert prepared.pack == shorter.pack > 1
    assert prepared.patched_tokens.shape == shorter.patched_tokens.shape
    runner = TtChronosTraceRunner(model, prepared)
    try:
        runner.capture()
        with expect_error(ValueError, "trace horizon changed"):
            runner.execute(shorter)
    finally:
        runner.release()


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_default_does_not_pack(mesh_device):
    from models.experimental.chronos_forecast.reference.chronos2.model import Chronos2Model as RefModel
    from models.experimental.chronos_forecast.tt.model import TtChronos

    _require_checkpoint()
    model = TtChronos.from_torch_model(mesh_device, RefModel.from_pretrained(str(CKPT)).eval())
    prepared = model.prepare_inputs(_series(1024, 512), num_output_patches=4)
    assert prepared.pack == 1
