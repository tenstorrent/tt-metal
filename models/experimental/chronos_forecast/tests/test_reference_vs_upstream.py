# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""CPU golden tests: vendored Chronos-2 reference vs amazon-science submodule.

No Tenstorrent device. Dummy weights stay in the submodule (root gitignore excludes *.bin).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from models.experimental.chronos_forecast.common.chronos_src import (
    CHRONOS_SUBMODULE_ROOT,
    ensure_chronos_on_path,
)
from models.experimental.chronos_forecast.reference.chronos2 import model as ref_model_module
from models.experimental.chronos_forecast.reference.chronos2.config import Chronos2CoreConfig as RefConfig
from models.experimental.chronos_forecast.reference.chronos2.layers import (
    MHA as RefMHA,
)
from models.experimental.chronos_forecast.reference.chronos2.layers import (
    MLP as RefMLP,
)
from models.experimental.chronos_forecast.reference.chronos2.layers import (
    Chronos2LayerNorm as RefLayerNorm,
)
from models.experimental.chronos_forecast.reference.chronos2.layers import (
    Chronos2RotaryEmbedding as RefRotary,
)
from models.experimental.chronos_forecast.reference.chronos2.layers import (
    FeedForward as RefFeedForward,
)
from models.experimental.chronos_forecast.reference.chronos2.layers import (
    GroupSelfAttention as RefGroupSelfAttention,
)
from models.experimental.chronos_forecast.reference.chronos2.layers import (
    ResidualBlock as RefResidualBlock,
)
from models.experimental.chronos_forecast.reference.chronos2.layers import (
    TimeSelfAttention as RefTimeSelfAttention,
)
from models.experimental.chronos_forecast.reference.chronos2.model import Chronos2Encoder as RefEncoder
from models.experimental.chronos_forecast.reference.chronos2.model import Chronos2EncoderBlock as RefEncoderBlock
from models.experimental.chronos_forecast.reference.chronos2.model import Chronos2Model as RefModel
from models.experimental.chronos_forecast.reference.chronos_bolt_ops import InstanceNorm as RefInstanceNorm
from models.experimental.chronos_forecast.reference.chronos_bolt_ops import Patch as RefPatch

ensure_chronos_on_path()

from chronos.chronos2.config import Chronos2CoreConfig as UpConfig  # noqa: E402
from chronos.chronos2.layers import MHA as UpMHA  # noqa: E402
from chronos.chronos2.layers import MLP as UpMLP  # noqa: E402
from chronos.chronos2.layers import Chronos2LayerNorm as UpLayerNorm  # noqa: E402
from chronos.chronos2.layers import Chronos2RotaryEmbedding as UpRotary  # noqa: E402
from chronos.chronos2.layers import FeedForward as UpFeedForward  # noqa: E402
from chronos.chronos2.layers import GroupSelfAttention as UpGroupSelfAttention  # noqa: E402
from chronos.chronos2.layers import ResidualBlock as UpResidualBlock  # noqa: E402
from chronos.chronos2.layers import TimeSelfAttention as UpTimeSelfAttention  # noqa: E402
from chronos.chronos2.model import Chronos2Encoder as UpEncoder  # noqa: E402
from chronos.chronos2.model import Chronos2EncoderBlock as UpEncoderBlock  # noqa: E402
from chronos.chronos2.model import Chronos2Model as UpModel  # noqa: E402
from chronos.chronos_bolt import InstanceNorm as UpInstanceNorm  # noqa: E402
from chronos.chronos_bolt import Patch as UpPatch  # noqa: E402

SEED = 0
B, S = 2, 8
PATCH_SIZE = 16
DUMMY_MODEL_PATH = CHRONOS_SUBMODULE_ROOT / "test" / "dummy-chronos2-model"
REAL_WEIGHTS_PATH = Path(__file__).resolve().parents[1] / "weights" / "chronos-2"
REF_MODEL_FILE = Path(ref_model_module.__file__).resolve()


def _tiny_kwargs() -> dict:
    return dict(
        d_model=6,
        d_kv=4,
        d_ff=8,
        num_layers=2,
        num_heads=4,
        dropout_rate=0.0,
        attn_implementation="eager",
    )


def _ref_config() -> RefConfig:
    return RefConfig(**_tiny_kwargs())


def _up_config() -> UpConfig:
    return UpConfig(**_tiny_kwargs())


def _assert_same(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0, equal_nan=True)


def _sync_eval(src: torch.nn.Module, dst: torch.nn.Module) -> None:
    dst.load_state_dict(src.state_dict())
    src.eval()
    dst.eval()


def _hidden(config) -> torch.Tensor:
    torch.manual_seed(SEED)
    return torch.randn(B, S, config.d_model)


def _time_mask(config) -> torch.Tensor:
    return torch.zeros(B, config.num_heads, S, S)


def _group_mask() -> torch.Tensor:
    # Encoder layout after rearrange: (time, 1, batch, batch)
    return torch.zeros(S, 1, B, B)


def _position_ids() -> torch.Tensor:
    return torch.arange(S).unsqueeze(0).expand(B, -1)


def test_vendored_model_file_is_under_reference():
    assert "reference/chronos2/model.py" in str(REF_MODEL_FILE).replace("\\", "/")


def test_layer_norm():
    up = UpLayerNorm(6)
    ref = RefLayerNorm(6)
    _sync_eval(up, ref)
    torch.manual_seed(SEED)
    x = torch.randn(B, S, 6)
    _assert_same(ref(x), up(x))


def test_rotary():
    up_cfg, ref_cfg = _up_config(), _ref_config()
    up, ref = UpRotary(up_cfg), RefRotary(ref_cfg)
    _sync_eval(up, ref)
    torch.manual_seed(SEED)
    x = torch.randn(B, up_cfg.num_heads, S, up_cfg.d_kv)
    pos = _position_ids()
    ref_cos, ref_sin = ref(x, pos)
    up_cos, up_sin = up(x, pos)
    _assert_same(ref_cos, up_cos)
    _assert_same(ref_sin, up_sin)


def test_mlp():
    up_cfg, ref_cfg = _up_config(), _ref_config()
    up, ref = UpMLP(up_cfg), RefMLP(ref_cfg)
    _sync_eval(up, ref)
    x = _hidden(up_cfg)
    _assert_same(ref(x), up(x))


def test_feed_forward():
    up_cfg, ref_cfg = _up_config(), _ref_config()
    up, ref = UpFeedForward(up_cfg), RefFeedForward(ref_cfg)
    _sync_eval(up, ref)
    x = _hidden(up_cfg)
    _assert_same(ref(x), up(x))


def test_mha_no_rope():
    up_cfg, ref_cfg = _up_config(), _ref_config()
    up, ref = UpMHA(up_cfg, use_rope=False), RefMHA(ref_cfg, use_rope=False)
    _sync_eval(up, ref)
    x = _hidden(up_cfg)
    mask = _time_mask(up_cfg)
    _assert_same(ref(x, mask=mask).hidden_states, up(x, mask=mask).hidden_states)


def test_mha_rope():
    up_cfg, ref_cfg = _up_config(), _ref_config()
    up, ref = UpMHA(up_cfg, use_rope=True), RefMHA(ref_cfg, use_rope=True)
    _sync_eval(up, ref)
    x = _hidden(up_cfg)
    mask = _time_mask(up_cfg)
    pos = _position_ids()
    _assert_same(
        ref(x, mask=mask, position_ids=pos).hidden_states,
        up(x, mask=mask, position_ids=pos).hidden_states,
    )


def test_time_self_attention():
    up_cfg, ref_cfg = _up_config(), _ref_config()
    up, ref = UpTimeSelfAttention(up_cfg), RefTimeSelfAttention(ref_cfg)
    _sync_eval(up, ref)
    x = _hidden(up_cfg)
    mask = _time_mask(up_cfg)
    pos = _position_ids()
    _assert_same(
        ref(x, attention_mask=mask, position_ids=pos).hidden_states,
        up(x, attention_mask=mask, position_ids=pos).hidden_states,
    )


def test_group_self_attention():
    up_cfg, ref_cfg = _up_config(), _ref_config()
    up, ref = UpGroupSelfAttention(up_cfg), RefGroupSelfAttention(ref_cfg)
    _sync_eval(up, ref)
    x = _hidden(up_cfg)
    mask = _group_mask()
    _assert_same(ref(x, attention_mask=mask).hidden_states, up(x, attention_mask=mask).hidden_states)


def test_residual_block():
    in_dim = PATCH_SIZE * 3
    up = UpResidualBlock(in_dim=in_dim, h_dim=8, out_dim=6, act_fn_name="relu", dropout_p=0.0)
    ref = RefResidualBlock(in_dim=in_dim, h_dim=8, out_dim=6, act_fn_name="relu", dropout_p=0.0)
    _sync_eval(up, ref)
    torch.manual_seed(SEED)
    x = torch.randn(B, 4, in_dim)
    _assert_same(ref(x), up(x))


def test_patch():
    up, ref = UpPatch(patch_size=PATCH_SIZE, patch_stride=PATCH_SIZE), RefPatch(
        patch_size=PATCH_SIZE, patch_stride=PATCH_SIZE
    )
    torch.manual_seed(SEED)
    x = torch.randn(B, 20)
    _assert_same(ref(x), up(x))


def test_instance_norm():
    up, ref = UpInstanceNorm(use_arcsinh=True), RefInstanceNorm(use_arcsinh=True)
    torch.manual_seed(SEED)
    x = torch.randn(B, 20)
    ref_y, ref_ls = ref(x)
    up_y, up_ls = up(x)
    _assert_same(ref_y, up_y)
    _assert_same(ref_ls[0], up_ls[0])
    _assert_same(ref_ls[1], up_ls[1])
    _assert_same(ref.inverse(ref_y, ref_ls), up.inverse(up_y, up_ls))


def test_encoder_block():
    up_cfg, ref_cfg = _up_config(), _ref_config()
    up, ref = UpEncoderBlock(up_cfg), RefEncoderBlock(ref_cfg)
    _sync_eval(up, ref)
    x = _hidden(up_cfg)
    pos = _position_ids()
    time_mask = _time_mask(up_cfg)
    group_mask = _group_mask()
    _assert_same(
        ref(x, position_ids=pos, attention_mask=time_mask, group_time_mask=group_mask).hidden_states,
        up(x, position_ids=pos, attention_mask=time_mask, group_time_mask=group_mask).hidden_states,
    )


def test_encoder():
    up_cfg, ref_cfg = _up_config(), _ref_config()
    up, ref = UpEncoder(up_cfg), RefEncoder(ref_cfg)
    _sync_eval(up, ref)
    x = _hidden(up_cfg)
    group_ids = torch.arange(B)
    attention_mask = torch.ones(B, S)
    _assert_same(
        ref(inputs_embeds=x, group_ids=group_ids, attention_mask=attention_mask).last_hidden_state,
        up(inputs_embeds=x, group_ids=group_ids, attention_mask=attention_mask).last_hidden_state,
    )


def test_model_forward():
    assert (DUMMY_MODEL_PATH / "config.json").is_file()
    up = UpModel.from_pretrained(DUMMY_MODEL_PATH).eval()
    ref = RefModel.from_pretrained(DUMMY_MODEL_PATH).eval()
    up.config._attn_implementation = "eager"
    ref.config._attn_implementation = "eager"
    _sync_eval(up, ref)
    torch.manual_seed(SEED)
    context = torch.randn(2, 32)
    up_out = up(context=context, num_output_patches=1)
    ref_out = ref(context=context, num_output_patches=1)
    assert up_out.quantile_preds is not None and ref_out.quantile_preds is not None
    _assert_same(ref_out.quantile_preds, up_out.quantile_preds)


def test_model_forward_pretrained():
    """Accuracy lock: vendored Chronos2Model vs submodule on amazon/chronos-2 weights."""
    if not (REAL_WEIGHTS_PATH / "model.safetensors").is_file():
        pytest.skip(
            f"Missing {REAL_WEIGHTS_PATH}. Download with: "
            "hf download amazon/chronos-2 --local-dir "
            "models/experimental/chronos_forecast/weights/chronos-2"
        )
    up = UpModel.from_pretrained(REAL_WEIGHTS_PATH).eval()
    ref = RefModel.from_pretrained(REAL_WEIGHTS_PATH).eval()
    up.config._attn_implementation = "eager"
    ref.config._attn_implementation = "eager"
    _sync_eval(up, ref)
    torch.manual_seed(SEED)
    context = torch.randn(2, 64)
    with torch.no_grad():
        up_out = up(context=context, num_output_patches=1)
        ref_out = ref(context=context, num_output_patches=1)
    assert up_out.quantile_preds is not None and ref_out.quantile_preds is not None
    assert torch.isfinite(ref_out.quantile_preds).all()
    _assert_same(ref_out.quantile_preds, up_out.quantile_preds)
