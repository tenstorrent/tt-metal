# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host-only tests of the MiniMax-H3 8-bit matmul config: environment knobs, presets, the `to_out` carve-out and
what `apply_quant_config` sets on a block."""

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
    assert config.out_weight and not config.fuse_out_addcmul
    assert config.sdpa_input_dtype is None


def test_w8a8_keeps_the_fused_out_epilogue(clean_env):
    clean_env.setenv(qc.ENV_FLAG, "w8a8")
    config = qc.quant_config_from_env()
    for name in ("qkv", "ff1", "ff2"):
        quant = config.linear(name)
        assert quant.weight_dtype == ttnn.bfloat8_b
        assert quant.activation_dtype == ttnn.bfloat8_b
        assert quant.math_fidelity == ttnn.MathFidelity.HiFi2
    assert config.out.weight_dtype is None
    assert config.out.activation_dtype == ttnn.bfloat8_b
    assert not config.out_weight and config.fuse_out_addcmul
    clean_env.setenv("FAST_H3_FP8_OUT_WEIGHT", "0")
    clean_env.setenv(qc.ENV_FLAG, "w8a8_lofi")
    assert not qc.quant_config_from_env().out_weight


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
    assert not config.out_weight


def test_out_weight_requires_the_unfused_epilogue(clean_env):
    clean_env.setenv(qc.ENV_FLAG, "w8a8")
    clean_env.setenv("FAST_H3_FP8_OUT_WEIGHT", "1")
    config = qc.quant_config_from_env()
    assert config.out.weight_dtype == ttnn.bfloat8_b
    assert config.out_weight and not config.fuse_out_addcmul


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


_MODEL_CONFIG = SimpleNamespace(math_fidelity=ttnn.MathFidelity.HiFi2)


def _fake_block():
    attn = SimpleNamespace(
        to_qkv=_Linear(),
        to_out=_Linear(),
        mm_compute_kernel_config=_MODEL_CONFIG,
        qkv_compute_kernel_config=_MODEL_CONFIG,
        out_compute_kernel_config=_MODEL_CONFIG,
        sdpa_input_dtype=None,
        fuse_out_addcmul=True,
    )
    ff = SimpleNamespace(
        ff1=_Linear(),
        ff2=_Linear(),
        ff1_output_dtype=None,
        ff1_compute_kernel_config=None,
        ff2_compute_kernel_config=None,
    )
    return SimpleNamespace(attn=attn, ff=ff, mesh_device=SimpleNamespace(arch=lambda: ttnn.device.Arch.BLACKHOLE))


def test_apply_sets_the_block_attributes(monkeypatch):
    casts = []
    monkeypatch.setattr(qc.ttnn, "typecast", lambda data, dtype: casts.append(dtype) or SimpleNamespace(dtype=dtype))
    block = _fake_block()
    qc.apply_quant_config(block, qc.MiniMaxH3QuantConfig.preset("w8a8_lofi", sdpa=True))
    assert casts == [ttnn.bfloat8_b] * 4
    assert block.attn.to_out.weight._data.dtype == ttnn.bfloat8_b
    assert not block.attn.fuse_out_addcmul
    for linear in (block.attn.to_qkv, block.attn.to_out, block.ff.ff1):
        assert linear.activation_dtype == ttnn.bfloat8_b
    assert block.ff.ff2.activation_dtype is None
    assert block.ff.ff1_output_dtype == ttnn.bfloat8_b
    assert block.attn.to_qkv.pin_output_bf16 and block.attn.to_out.pin_output_bf16
    assert not block.ff.ff1.pin_output_bf16
    assert block.attn.qkv_compute_kernel_config.math_fidelity == ttnn.MathFidelity.LoFi
    assert block.attn.out_compute_kernel_config.math_fidelity == ttnn.MathFidelity.LoFi
    assert block.ff.ff1_compute_kernel_config.math_fidelity == ttnn.MathFidelity.LoFi
    assert block.ff.ff2_compute_kernel_config.math_fidelity == ttnn.MathFidelity.LoFi
    assert block.attn.sdpa_input_dtype == ttnn.bfloat8_b
    qc.apply_quant_config(block, qc.MiniMaxH3QuantConfig.preset("w8a8_lofi", sdpa=True))
    assert casts == [ttnn.bfloat8_b] * 4
    hifi = _fake_block()
    qc.apply_quant_config(hifi, qc.MiniMaxH3QuantConfig.preset("w8a8"))
    assert hifi.attn.fuse_out_addcmul and hifi.attn.to_out.weight._data.dtype == ttnn.bfloat16


def test_apply_out_weight_unfuses_the_epilogue(monkeypatch):
    monkeypatch.setattr(qc.ttnn, "typecast", lambda data, dtype: SimpleNamespace(dtype=dtype))
    block = _fake_block()
    qc.apply_quant_config(block, qc.MiniMaxH3QuantConfig.preset("w8a8", out_weight=True))
    assert block.attn.to_out.weight._data.dtype == ttnn.bfloat8_b
    assert not block.attn.fuse_out_addcmul


def test_fusion_follows_the_out_weight_dtype(monkeypatch):
    monkeypatch.setattr(qc.ttnn, "typecast", lambda data, dtype: SimpleNamespace(dtype=dtype))
    block = _fake_block()
    qc.apply_quant_config(block, qc.MiniMaxH3QuantConfig(out=qc.LinearQuant(weight_dtype=ttnn.bfloat8_b)))
    assert not block.attn.fuse_out_addcmul
    block = _fake_block()
    qc.apply_quant_config(block, qc.MiniMaxH3QuantConfig(out=qc.LinearQuant(activation_dtype=ttnn.bfloat8_b)))
    assert block.attn.fuse_out_addcmul and block.attn.to_out.weight._data.dtype == ttnn.bfloat16


def test_ff1_only_keeps_ff2_at_full_precision(monkeypatch):
    monkeypatch.setattr(qc.ttnn, "typecast", lambda data, dtype: SimpleNamespace(dtype=dtype))
    block = _fake_block()
    qc.apply_quant_config(block, qc.MiniMaxH3QuantConfig.preset("w8a8_lofi", linears=("ff1",)))
    assert block.ff.ff1.weight._data.dtype == ttnn.bfloat8_b
    assert block.ff.ff2.weight._data.dtype == ttnn.bfloat16
    assert block.ff.ff1.activation_dtype == ttnn.bfloat8_b
    assert block.ff.ff1_output_dtype == ttnn.bfloat16
    assert block.ff.ff1_compute_kernel_config.math_fidelity == ttnn.MathFidelity.LoFi
    assert block.ff.ff2_compute_kernel_config is None
    assert block.attn.qkv_compute_kernel_config is _MODEL_CONFIG
    assert block.attn.fuse_out_addcmul


def test_ff2_only_runs_at_the_requested_fidelity(monkeypatch):
    monkeypatch.setattr(qc.ttnn, "typecast", lambda data, dtype: SimpleNamespace(dtype=dtype))
    block = _fake_block()
    qc.apply_quant_config(block, qc.MiniMaxH3QuantConfig.preset("w8a8_lofi", linears=("ff2",)))
    assert block.ff.ff2.weight._data.dtype == ttnn.bfloat8_b
    assert block.ff.ff1.weight._data.dtype == ttnn.bfloat16
    assert block.ff.ff1.activation_dtype is None
    assert block.ff.ff1_output_dtype == ttnn.bfloat8_b
    assert block.ff.ff1_compute_kernel_config is None
    assert block.ff.ff2_compute_kernel_config.math_fidelity == ttnn.MathFidelity.LoFi
