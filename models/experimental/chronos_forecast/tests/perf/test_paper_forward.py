# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Paper-shape perf: 1024 series, context 2048, 64-step forecast, one chip.

A10G bar from Chronos-2 (arXiv:2510.15821): 300 series/s on the 120M model,
which is 1024/300 ≈ 3.41 s of wall clock. This times ``TtChronos.forward``
after one warmup and does not fail when the device is slower than that bar.

Weights are the local ``amazon/chronos-2`` checkpoint when it is present.
Until then the test builds a randomly initialized reference model with the
published 120M geometry (d_model=768, 12 layers, 12 heads, d_kv=64).
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
import torch

BATCH = 1024
CONTEXT = 2048
PREDICTION_LENGTH = 64
NUM_OUTPUT_PATCHES = 4
NUM_QUANTILES = 21
A10G_SERIES_PER_S = 300.0
A10G_WALL_S = BATCH / A10G_SERIES_PER_S
CKPT = Path("models/experimental/chronos_forecast/weights/chronos-2")

# Published amazon/chronos-2 config (huggingface.co/amazon/chronos-2).
_QUANTILES = [
    0.01,
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
    0.99,
]


def _paper_config():
    from models.experimental.chronos_forecast.reference.chronos2.config import Chronos2CoreConfig

    return Chronos2CoreConfig(
        d_model=768,
        d_kv=64,
        d_ff=3072,
        num_layers=12,
        num_heads=12,
        dropout_rate=0.1,
        layer_norm_epsilon=1e-6,
        rope_theta=10000.0,
        vocab_size=2,
        reg_token_id=1,
        chronos_config={
            "context_length": 8192,
            "input_patch_size": 16,
            "input_patch_stride": 16,
            "max_output_patches": 64,
            "output_patch_size": 16,
            "quantiles": list(_QUANTILES),
            "time_encoding_scale": 8192,
            "use_arcsinh": True,
            "use_reg_token": True,
        },
    )


def _checkpoint_ready() -> bool:
    if not (CKPT / "config.json").is_file():
        return False
    return (CKPT / "model.safetensors").is_file() or (CKPT / "pytorch_model.bin").is_file()


def _load_reference():
    from models.experimental.chronos_forecast.reference.chronos2.model import Chronos2Model as RefModel

    if _checkpoint_ready():
        return RefModel.from_pretrained(str(CKPT)).eval(), "checkpoint"
    return RefModel(_paper_config()).eval(), "random"


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_paper_forward_perf(mesh_device):
    """Time one steady-state forward at the Chronos-2 paper shape."""
    pytest.importorskip("ttnn")

    from models.experimental.chronos_forecast.tt.model import TtChronos, tt_chronos_config_from_torch_model

    if mesh_device.get_num_devices() != 1:
        pytest.skip("single-chip bring-up only (one chip)")

    model, weight_source = _load_reference()
    cfg = tt_chronos_config_from_torch_model(model)
    if cfg.d_model != 768 or model.config.num_layers != 12 or model.config.num_heads != 12:
        raise AssertionError(
            f"expected 120M geometry d_model=768 layers=12 heads=12, "
            f"got d_model={cfg.d_model} layers={model.config.num_layers} heads={model.config.num_heads}"
        )
    if cfg.output_patch_size * NUM_OUTPUT_PATCHES != PREDICTION_LENGTH:
        raise AssertionError(f"output_patch_size {cfg.output_patch_size} * {NUM_OUTPUT_PATCHES} != {PREDICTION_LENGTH}")
    if cfg.num_quantiles != NUM_QUANTILES:
        raise AssertionError(f"expected {NUM_QUANTILES} quantiles, model has {cfg.num_quantiles}")

    tt = TtChronos.from_torch_model(mesh_device, model)

    torch.manual_seed(0)
    context = torch.randn(BATCH, CONTEXT)
    expected_shape = (BATCH, NUM_QUANTILES, PREDICTION_LENGTH)

    cold_start = time.perf_counter()
    cold = tt.forward(context=context, num_output_patches=NUM_OUTPUT_PATCHES)
    cold_s = time.perf_counter() - cold_start
    assert cold.shape == expected_shape

    steady_start = time.perf_counter()
    got = tt.forward(context=context, num_output_patches=NUM_OUTPUT_PATCHES)
    steady_s = time.perf_counter() - steady_start
    assert got.shape == expected_shape

    series_per_s = BATCH / steady_s
    print(
        "\n[PERF] paper forward"
        f"\n  weights:            {weight_source}"
        f"\n  shape:              context {tuple(context.shape)} -> quantile_preds {expected_shape}"
        f"\n  num_output_patches: {NUM_OUTPUT_PATCHES}"
        f"\n  cold_s:             {cold_s:.3f}"
        f"\n  steady_s:           {steady_s:.3f}"
        f"\n  series_per_s:       {series_per_s:.2f}"
        f"\n  a10g_wall_s:        {A10G_WALL_S:.3f}"
        f"\n  a10g_series_per_s:  {A10G_SERIES_PER_S:.0f}"
    )
