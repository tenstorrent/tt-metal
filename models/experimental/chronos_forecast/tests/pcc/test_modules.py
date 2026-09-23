# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""PCC tests comparing TTNN Chronos against the PyTorch reference (single-chip)."""

from __future__ import annotations

import pytest
import torch

from models.experimental.chronos_forecast.reference.chronos2.layers import ResidualBlock as RefRB
from models.experimental.chronos_forecast.tests.golden_helpers import log_golden
from models.experimental.chronos_forecast.tt.residual_block import (
    TtResidualBlock,
    TtResidualBlockWeights,
)


@pytest.mark.parametrize(
    "shape",
    [pytest.param((2, 4, 48), id="tiny_2x4x48"), pytest.param((2, 2, 48), id="real_2x2x48")],
)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_residual_block_pcc(mesh_device, shape):
    """TT ResidualBlock (48d patched input) vs reference oracle.

    tiny covers the golden stub geometry (tests/test_residual.py); real covers
    the T=32 patched context (B=2, P=2, d_model=6). Single-chip only.
    """
    pytest.importorskip("ttnn")
    from tests.ttnn.utils_for_testing import assert_with_pcc

    if mesh_device.get_num_devices() != 1:
        pytest.skip("single-chip bring-up only (one chip)")

    torch.manual_seed(0)
    block = RefRB(in_dim=48, h_dim=8, out_dim=6, act_fn_name="relu", dropout_p=0.0).eval()
    weights = TtResidualBlockWeights.from_torch_block(block)
    tt = TtResidualBlock(device=mesh_device, weights=weights)

    torch.manual_seed(1)
    x = torch.randn(*shape)
    expected = block(x).float()
    got = tt.forward(x)
    assert got.shape == (*shape[:2], 6)
    log_golden(f"tt_residual/device_{shape[1]}p", got)
    assert_with_pcc(expected, got, pcc=0.99)


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_time_attention_pcc(mesh_device):
    """TT TimeSelfAttention (tiny golden dims) vs reference oracle. Single-chip only."""
    pytest.importorskip("ttnn")
    from tests.ttnn.utils_for_testing import assert_with_pcc

    from models.experimental.chronos_forecast.reference.chronos2.layers import (
        TimeSelfAttention as RefTSA,
    )
    from models.experimental.chronos_forecast.tests.golden_helpers import tiny_config
    from models.experimental.chronos_forecast.tt.time_attention import (
        TtTimeAttention,
        TtTimeAttentionWeights,
        build_rope_cache,
    )

    if mesh_device.get_num_devices() != 1:
        pytest.skip("single-chip bring-up only (one chip)")

    cfg = tiny_config()
    torch.manual_seed(0)
    layer = RefTSA(cfg).eval()
    weights = TtTimeAttentionWeights.from_torch_layer(layer)
    tt = TtTimeAttention(device=mesh_device, weights=weights)

    torch.manual_seed(1)
    x = torch.randn(2, 8, cfg.d_model)
    position_ids = torch.arange(8).unsqueeze(0).expand(2, -1)
    expected = layer(
        x,
        attention_mask=torch.zeros(2, cfg.num_heads, 8, 8),
        position_ids=position_ids,
    ).hidden_states

    cos, sin = build_rope_cache(position_ids, weights.inv_freq)
    got = tt.forward(x, cos, sin, torch.zeros(1, 1, 8, 8))
    assert got.shape == x.shape
    log_golden("tt_time_attn/device_8", got)
    assert_with_pcc(expected.float(), got, pcc=0.99)


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_group_attention_pcc(mesh_device):
    """TT GroupSelfAttention (tiny golden dims) vs reference oracle. Single-chip only."""
    pytest.importorskip("ttnn")
    from tests.ttnn.utils_for_testing import assert_with_pcc

    from models.experimental.chronos_forecast.reference.chronos2.layers import (
        GroupSelfAttention as RefGSA,
    )
    from models.experimental.chronos_forecast.tests.golden_helpers import tiny_config
    from models.experimental.chronos_forecast.tt.group_attention import (
        TtGroupAttention,
        TtGroupAttentionWeights,
    )

    if mesh_device.get_num_devices() != 1:
        pytest.skip("single-chip bring-up only (one chip)")

    cfg = tiny_config()
    torch.manual_seed(0)
    layer = RefGSA(cfg).eval()
    weights = TtGroupAttentionWeights.from_torch_layer(layer)
    tt = TtGroupAttention(device=mesh_device, weights=weights)

    torch.manual_seed(1)
    x = torch.randn(2, 8, cfg.d_model)
    expected = layer(x, attention_mask=torch.zeros(8, 1, 2, 2)).hidden_states

    got = tt.forward(x, torch.zeros(8, 1, 2, 2))
    assert got.shape == x.shape
    log_golden("tt_group_attn/device_8", got)
    assert_with_pcc(expected.float(), got, pcc=0.99)


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_encoder_block_pcc(mesh_device):
    """TT Chronos2EncoderBlock (tiny golden dims) vs reference oracle. Single-chip only."""
    pytest.importorskip("ttnn")
    from tests.ttnn.utils_for_testing import assert_with_pcc

    from models.experimental.chronos_forecast.reference.chronos2.model import (
        Chronos2EncoderBlock as RefBlock,
    )
    from models.experimental.chronos_forecast.tests.golden_helpers import tiny_config
    from models.experimental.chronos_forecast.tt.encoder_block import (
        TtEncoderBlock,
        TtEncoderBlockWeights,
    )
    from models.experimental.chronos_forecast.tt.time_attention import build_rope_cache

    if mesh_device.get_num_devices() != 1:
        pytest.skip("single-chip bring-up only (one chip)")

    cfg = tiny_config()
    torch.manual_seed(0)
    block = RefBlock(cfg).eval()
    weights = TtEncoderBlockWeights.from_torch_block(block)
    tt = TtEncoderBlock(device=mesh_device, weights=weights)

    torch.manual_seed(1)
    x = torch.randn(2, 8, cfg.d_model)
    position_ids = torch.arange(8).unsqueeze(0).expand(2, -1)
    expected = block(
        x,
        position_ids=position_ids,
        attention_mask=torch.zeros(2, cfg.num_heads, 8, 8),
        group_time_mask=torch.zeros(8, 1, 2, 2),
    ).hidden_states

    cos, sin = build_rope_cache(position_ids, weights.time.inv_freq)
    got = tt.forward(x, cos, sin, torch.zeros(1, 1, 8, 8), torch.zeros(8, 1, 2, 2))
    assert got.shape == x.shape
    log_golden("tt_encoder_block/device_8", got)
    assert_with_pcc(expected.float(), got, pcc=0.99)


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_encoder_pcc(mesh_device):
    pytest.importorskip("ttnn")
    from tests.ttnn.utils_for_testing import assert_with_pcc

    from models.experimental.chronos_forecast.reference.chronos2.model import (
        Chronos2Encoder as RefEncoder,
    )
    from models.experimental.chronos_forecast.tests.golden_helpers import tiny_config
    from models.experimental.chronos_forecast.tt.encoder import TtEncoder, TtEncoderWeights
    from models.experimental.chronos_forecast.tt.time_attention import build_rope_cache

    if mesh_device.get_num_devices() != 1:
        pytest.skip("single-chip bring-up only (one chip)")

    cfg = tiny_config()
    torch.manual_seed(0)
    encoder = RefEncoder(cfg).eval()
    weights = TtEncoderWeights.from_torch_encoder(encoder)
    tt = TtEncoder(device=mesh_device, weights=weights)

    torch.manual_seed(1)
    x = torch.randn(2, 8, cfg.d_model)
    position_ids = torch.arange(8).unsqueeze(0).expand(2, -1)
    expected = encoder(
        inputs_embeds=x,
        group_ids=torch.zeros(2, dtype=torch.long),
        attention_mask=torch.ones(2, 8),
    ).last_hidden_state

    inv_freq = weights.blocks[0].time.inv_freq
    cos, sin = build_rope_cache(position_ids, inv_freq)
    got = tt.forward(x, cos, sin, torch.zeros(1, 1, 8, 8), torch.zeros(8, 1, 2, 2))
    assert got.shape == x.shape
    log_golden("tt_encoder/device_8", got)
    assert_with_pcc(expected.float(), got, pcc=0.99)


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_output_embedding_pcc(mesh_device):
    """TT output patch embedding (TtResidualBlock reuse, dummy 6->336) vs oracle."""
    pytest.importorskip("ttnn")
    from tests.ttnn.utils_for_testing import assert_with_pcc

    from models.experimental.chronos_forecast.reference.chronos2.model import Chronos2Model as RefModel
    from models.experimental.chronos_forecast.tests.golden_helpers import DUMMY_MODEL_PATH

    if mesh_device.get_num_devices() != 1:
        pytest.skip("single-chip bring-up only (one chip)")

    model = RefModel.from_pretrained(DUMMY_MODEL_PATH).eval()
    weights = TtResidualBlockWeights.from_torch_block(model.output_patch_embedding)
    tt = TtResidualBlock(device=mesh_device, weights=weights)

    torch.manual_seed(1)
    x = torch.randn(2, 1, model.config.d_model)
    expected = model.output_patch_embedding(x).float()
    got = tt.forward(x)
    assert got.shape == (2, 1, expected.shape[-1])
    log_golden("tt_output_embed/device_1", got)
    assert_with_pcc(expected, got, pcc=0.99)


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_tt_encode_pcc(mesh_device):
    """TT Chronos encode (dummy checkpoint, C=32, O=1) vs reference oracle. Single-chip only."""
    pytest.importorskip("ttnn")
    from tests.ttnn.utils_for_testing import assert_with_pcc

    from models.experimental.chronos_forecast.reference.chronos2.model import Chronos2Model as RefModel
    from models.experimental.chronos_forecast.tests.golden_helpers import DUMMY_MODEL_PATH
    from models.experimental.chronos_forecast.tt.model import TtChronos

    if mesh_device.get_num_devices() != 1:
        pytest.skip("single-chip bring-up only (one chip)")

    model = RefModel.from_pretrained(DUMMY_MODEL_PATH).eval()
    tt = TtChronos.from_torch_model(mesh_device, model)

    torch.manual_seed(0)
    context = torch.randn(2, 32)
    with torch.no_grad():
        ref_out, _, _, _ = model.encode(context=context, num_output_patches=1)
        expected = ref_out[0]
    got, _, _ = tt.encode(context=context, num_output_patches=1)
    assert got.shape == expected.shape
    log_golden("tt_encode/device_hidden", got)
    assert_with_pcc(expected.float(), got, pcc=0.99)


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_tt_forward_pcc(mesh_device):
    """TT Chronos forward (dummy checkpoint, C=32, O=1) vs reference oracle. Single-chip only."""
    pytest.importorskip("ttnn")
    from tests.ttnn.utils_for_testing import assert_with_pcc

    from models.experimental.chronos_forecast.reference.chronos2.model import Chronos2Model as RefModel
    from models.experimental.chronos_forecast.tests.golden_helpers import DUMMY_MODEL_PATH
    from models.experimental.chronos_forecast.tt.model import TtChronos

    if mesh_device.get_num_devices() != 1:
        pytest.skip("single-chip bring-up only (one chip)")

    model = RefModel.from_pretrained(DUMMY_MODEL_PATH).eval()
    tt = TtChronos.from_torch_model(mesh_device, model)

    torch.manual_seed(0)
    context = torch.randn(2, 32)
    with torch.no_grad():
        expected = model(context=context, num_output_patches=1).quantile_preds
    got = tt.forward(context=context, num_output_patches=1)
    assert got.shape == expected.shape
    log_golden("tt_forward/device_quantiles", got)
    assert_with_pcc(expected.float(), got, pcc=0.99)


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_tt_forward_pretrained_pcc(mesh_device):
    """TT forward with real amazon/chronos-2 weights via preprocess_model_parameters.

    Skipped when weights/chronos-2 is absent. Short context (C=512 -> 32
    patches) keeps L small on the single chip.
    """
    pytest.importorskip("ttnn")
    from pathlib import Path

    from tests.ttnn.utils_for_testing import assert_with_pcc

    from models.experimental.chronos_forecast.reference.chronos2.model import Chronos2Model as RefModel
    from models.experimental.chronos_forecast.tt.model import TtChronos, tt_chronos_config_from_torch_model
    from models.experimental.chronos_forecast.tt.model_preprocessing import preprocess_model_parameters

    ckpt = Path("models/experimental/chronos_forecast/weights/chronos-2")
    if not (ckpt / "config.json").is_file():
        pytest.skip("weights/chronos-2 absent")

    if mesh_device.get_num_devices() != 1:
        pytest.skip("single-chip bring-up only (one chip)")

    model = RefModel.from_pretrained(str(ckpt)).eval()
    weights = preprocess_model_parameters(
        model.state_dict(),
        head_dim=model.config.d_kv,
        rope_theta=model.config.rope_theta,
        eps=model.config.layer_norm_epsilon,
    )
    tt = TtChronos(mesh_device, weights, tt_chronos_config_from_torch_model(model))

    torch.manual_seed(0)
    context = torch.randn(2, 512)
    with torch.no_grad():
        expected = model(context=context, num_output_patches=1).quantile_preds
    got = tt.forward(context=context, num_output_patches=1)
    assert got.shape == expected.shape
    log_golden("tt_forward_pretrained/device_quantiles", got)
    assert_with_pcc(expected.float(), got, pcc=0.99)
