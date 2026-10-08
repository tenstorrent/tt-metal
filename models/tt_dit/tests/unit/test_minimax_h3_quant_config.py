# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for the MiniMax-H3 8-bit matmul config: the environment knobs, the presets' contents, the
`to_out` carve-out, and what `apply_quant_config` sets on a block. A wrong knob here is silent on device -- a
quantized linear that was meant to stay bf16 produces video, not an error."""

from types import SimpleNamespace

import pytest

import ttnn
from models.tt_dit.models.transformers.minimax_h3 import quant_config as qc

KNOBS = (
    qc.ENV_FLAG,
    "FAST_H3_FP8_LINEARS",
    "FAST_H3_FP8_ACTIVATIONS",
    "FAST_H3_FP8_FIDELITY",
    "FAST_H3_FP8_FP32_ACC",
    "FAST_H3_FP8_SDPA",
    "FAST_H3_FP8_OUT_WEIGHT",
    "FAST_H3_FP8_BLOCKS",
)


@pytest.fixture
def clean_env(monkeypatch):
    for name in KNOBS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_unset_is_off(clean_env):
    config = qc.quant_config_from_env()
    assert not config.active
    assert config == qc.MiniMaxH3QuantConfig.default()
    assert config.describe() == "off"


@pytest.mark.parametrize("value", ["0", "false", "off", ""])
def test_false_values_are_off(clean_env, value):
    clean_env.setenv(qc.ENV_FLAG, value)
    assert not qc.quant_config_from_env().active


def test_one_means_w8a8_lofi_with_an_unfused_out(clean_env):
    clean_env.setenv(qc.ENV_FLAG, "1")
    config = qc.quant_config_from_env()
    assert config == qc.MiniMaxH3QuantConfig.preset("w8a8_lofi")
    for name in qc.LINEARS:
        quant = config.linear(name)
        assert quant.weight_dtype == ttnn.bfloat8_b
        assert quant.activation_dtype == ttnn.bfloat8_b
        assert quant.math_fidelity == ttnn.MathFidelity.LoFi
        assert quant.fp32_dest_acc
    # At LoFi to_out's epilogue is un-fused (it would multiply the residual at LoFi), so its weight is quantized too.
    assert config.out_weight
    assert config.sdpa_input_dtype is None


def test_w8a8_keeps_the_fused_out_epilogue(clean_env):
    clean_env.setenv(qc.ENV_FLAG, "w8a8")
    config = qc.quant_config_from_env()
    for name in ("qkv", "ff1", "ff2"):
        quant = config.linear(name)
        assert quant.weight_dtype == ttnn.bfloat8_b
        assert quant.activation_dtype == ttnn.bfloat8_b
        assert quant.math_fidelity == ttnn.MathFidelity.HiFi2
    # to_out feeds the fused addcmul epilogue, whose weight tile format must match the bf16 residual.
    assert config.out.weight_dtype is None
    assert config.out.activation_dtype == ttnn.bfloat8_b
    assert not config.out_weight
    clean_env.setenv("FAST_H3_FP8_OUT_WEIGHT", "0")
    clean_env.setenv(qc.ENV_FLAG, "w8a8_lofi")
    assert not qc.quant_config_from_env().out_weight  # an explicit 0 keeps the epilogue fused at LoFi too


def test_w8_is_weights_only(clean_env):
    clean_env.setenv(qc.ENV_FLAG, "w8")
    config = qc.quant_config_from_env()
    for name in ("qkv", "ff1", "ff2"):
        assert config.linear(name).weight_dtype == ttnn.bfloat8_b
        assert config.linear(name).activation_dtype is None
    assert not config.out.quantized


def test_lofi_presets_keep_fp32_accumulation(clean_env):
    clean_env.setenv(qc.ENV_FLAG, "w8a8_lofi")
    config = qc.quant_config_from_env()
    for name in qc.LINEARS:
        assert config.linear(name).math_fidelity == ttnn.MathFidelity.LoFi
        assert config.linear(name).fp32_dest_acc
    clean_env.setenv(qc.ENV_FLAG, "w8_lofi")
    config = qc.quant_config_from_env()
    assert config.qkv.math_fidelity == ttnn.MathFidelity.LoFi
    assert config.qkv.weight_dtype == ttnn.bfloat8_b and config.qkv.activation_dtype is None
    assert config.describe() == "qkv:w8a16/LoFi out:w8a16/LoFi ff1:w8a16/LoFi ff2:w8a16/LoFi"


def test_overrides(clean_env):
    clean_env.setenv(qc.ENV_FLAG, "w8a8")
    clean_env.setenv("FAST_H3_FP8_LINEARS", "qkv,ff1")
    clean_env.setenv("FAST_H3_FP8_FIDELITY", "LoFi")
    clean_env.setenv("FAST_H3_FP8_FP32_ACC", "0")
    clean_env.setenv("FAST_H3_FP8_SDPA", "1")
    config = qc.quant_config_from_env()
    assert config.qkv.quantized and config.ff1.quantized
    assert not config.out.quantized and not config.ff2.quantized
    assert config.qkv.math_fidelity == ttnn.MathFidelity.LoFi and not config.qkv.fp32_dest_acc
    assert config.sdpa_input_dtype == ttnn.bfloat8_b
    assert config.describe() == "qkv:w8a8/LoFi/no-fp32-acc ff1:w8a8/LoFi/no-fp32-acc sdpa:in8"
    assert not config.out_weight  # "out" is not in the restricted linears


def test_out_weight_requires_the_unfused_epilogue(clean_env):
    clean_env.setenv(qc.ENV_FLAG, "w8a8")
    clean_env.setenv("FAST_H3_FP8_OUT_WEIGHT", "1")
    config = qc.quant_config_from_env()
    assert config.out.weight_dtype == ttnn.bfloat8_b
    assert config.out_weight


def test_block_range(clean_env, monkeypatch):
    clean_env.setenv(qc.ENV_FLAG, "1")
    clean_env.setenv("FAST_H3_FP8_BLOCKS", "2-46")
    config = qc.quant_config_from_env()
    assert config.blocks == (2, 46) and config.describe().endswith("blocks:2-46")
    assert not config.covers(1) and config.covers(2) and config.covers(46) and not config.covers(47)
    monkeypatch.setattr(qc.ttnn, "typecast", lambda data, dtype: SimpleNamespace(dtype=dtype))
    model = SimpleNamespace(transformer_blocks=[_fake_block() for _ in range(4)])
    qc.apply_quant_config(model, qc.MiniMaxH3QuantConfig.preset("w8a8", blocks=(1, 2)))
    cast = [b.attn.to_qkv.weight._data.dtype == ttnn.bfloat8_b for b in model.transformer_blocks]
    assert cast == [False, True, True, False]
    assert model.transformer_blocks[0].attn.to_qkv.activation_dtype is None


@pytest.mark.parametrize(
    ("var", "value"),
    [
        (qc.ENV_FLAG, "fp8"),
        ("FAST_H3_FP8_BLOCKS", "46-2"),
        ("FAST_H3_FP8_BLOCKS", "all"),
        ("FAST_H3_FP8_LINEARS", "qkv,adaln"),
        ("FAST_H3_FP8_FIDELITY", "VeryHiFi"),
        ("FAST_H3_FP8_SDPA", "maybe"),
    ],
)
def test_bad_values_are_refused(clean_env, var, value):
    clean_env.setenv(qc.ENV_FLAG, "1")
    clean_env.setenv(var, value)
    with pytest.raises(ValueError):
        qc.quant_config_from_env()


class _Param:
    def __init__(self, dtype):
        self._data = SimpleNamespace(dtype=dtype)
        self.dtype = dtype


class _Linear:
    def __init__(self):
        self.weight = _Param(ttnn.bfloat16)
        self.bias = None
        self.activation_dtype = None
        self.pin_output_bf16 = False


def _fake_block():
    attn = SimpleNamespace(to_qkv=_Linear(), to_out=_Linear(), sdpa_input_dtype=None, fuse_out_addcmul=True)
    ff = SimpleNamespace(ff1=_Linear(), ff2=_Linear(), ff1_output_dtype=None)
    return SimpleNamespace(attn=attn, ff=ff, mesh_device=SimpleNamespace(arch=lambda: ttnn.device.Arch.BLACKHOLE))


def test_apply_sets_the_block_attributes(monkeypatch):
    casts = []
    monkeypatch.setattr(qc.ttnn, "typecast", lambda data, dtype: casts.append(dtype) or SimpleNamespace(dtype=dtype))
    block = _fake_block()
    qc.apply_quant_config(block, qc.MiniMaxH3QuantConfig.preset("w8a8_lofi", sdpa=True))
    # Weights: all four cast; at LoFi to_out's epilogue is un-fused so its weight is quantized too.
    assert casts == [ttnn.bfloat8_b] * 4
    assert block.attn.to_out.weight._data.dtype == ttnn.bfloat8_b
    assert not block.attn.fuse_out_addcmul
    # Inputs: the three ColParallel linears cast before their gather; ff2 takes ff1's bfloat8_b output instead.
    for linear in (block.attn.to_qkv, block.attn.to_out, block.ff.ff1):
        assert linear.activation_dtype == ttnn.bfloat8_b
    assert block.ff.ff2.activation_dtype is None
    assert block.ff.ff1_output_dtype == ttnn.bfloat8_b
    # Outputs feeding a norm or the residual are pinned to bf16; ff1's is not.
    assert block.attn.to_qkv.pin_output_bf16 and block.attn.to_out.pin_output_bf16
    assert not block.ff.ff1.pin_output_bf16
    assert block.attn.qkv_compute_kernel_config.math_fidelity == ttnn.MathFidelity.LoFi
    assert block.ff_compute_kernel_config.math_fidelity == ttnn.MathFidelity.LoFi
    assert block.attn.sdpa_input_dtype == ttnn.bfloat8_b
    # Re-applying is a no-op on already-cast weights.
    qc.apply_quant_config(block, qc.MiniMaxH3QuantConfig.preset("w8a8_lofi", sdpa=True))
    assert casts == [ttnn.bfloat8_b] * 4
    # At HiFi2 the fused epilogue stays and to_out's weight is carved out.
    hifi = _fake_block()
    qc.apply_quant_config(hifi, qc.MiniMaxH3QuantConfig.preset("w8a8"))
    assert hifi.attn.fuse_out_addcmul and hifi.attn.to_out.weight._data.dtype == ttnn.bfloat16


def test_apply_out_weight_unfuses_the_epilogue(monkeypatch):
    monkeypatch.setattr(qc.ttnn, "typecast", lambda data, dtype: SimpleNamespace(dtype=dtype))
    block = _fake_block()
    qc.apply_quant_config(block, qc.MiniMaxH3QuantConfig.preset("w8a8", out_weight=True))
    assert block.attn.to_out.weight._data.dtype == ttnn.bfloat8_b
    assert not block.attn.fuse_out_addcmul
