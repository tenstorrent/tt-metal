# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the GDN configuration invariants."""

import hashlib
from dataclasses import asdict, fields
from pathlib import Path

import pytest

from models.demos.deepseek_v3_d_p.reference.gdn import qwen_models
from models.demos.deepseek_v3_d_p.reference.gdn.config import GDNConfig
from models.demos.deepseek_v3_d_p.reference.gdn.qwen_models import QWEN_GDN_MODELS, qwen_gdn_config, qwen_model_config
from models.demos.deepseek_v3_d_p.reference.gdn.tests.helpers import TINY


def test_derived_dimensions() -> None:
    # 2 K heads x 16 + 2 K heads x 16 + 6 V heads x 16 = 32 + 32 + 96.
    assert (TINY.group, TINY.q_dim, TINY.k_dim, TINY.v_dim, TINY.conv_dim) == (3, 32, 32, 96, 160)


@pytest.mark.parametrize(
    "field", ["hidden_size", "num_key_heads", "num_value_heads", "head_k_dim", "head_v_dim", "conv_kernel_size"]
)
def test_rejects_nonpositive_dimensions(field: str, expect_error) -> None:
    with expect_error(ValueError, field):
        GDNConfig(**(asdict(TINY) | {field: 0}))


def test_rejects_value_heads_not_a_multiple_of_key_heads(expect_error) -> None:
    with expect_error(ValueError, "multiple of num_key_heads"):
        GDNConfig(**(asdict(TINY) | {"num_value_heads": 5}))


def test_rejects_unsupported_numerical_policy(expect_error) -> None:
    with expect_error(ValueError, "conv_kernel_size=4"):
        GDNConfig(**(asdict(TINY) | {"conv_kernel_size": 3}))
    for norm_eps in (0.0, float("nan"), float("inf")):
        with expect_error(ValueError, "norm_eps"):
            GDNConfig(**(asdict(TINY) | {"norm_eps": norm_eps}))


@pytest.mark.parametrize("activation", ["swish", "gelu", None])
def test_rejects_noncanonical_output_gate_activation(activation, expect_error) -> None:
    """Aliases (swish) are resolved by the model-config boundary; the config takes only canonical names."""
    with expect_error(ValueError, "output_gate_activation"):
        GDNConfig(**(asdict(TINY) | {"output_gate_activation": activation}))


def test_reference_package_does_not_import_ttnn() -> None:
    import models.demos.deepseek_v3_d_p.reference.gdn.config as config_module
    import models.demos.deepseek_v3_d_p.reference.gdn.layer as layer_module

    assert "ttnn" not in config_module.__dict__
    assert "ttnn" not in layer_module.__dict__


# Every GDNConfig field as stated by each pinned config.json (values read off the files by hand), and the
# config's GDN output-gate setting that the activation is resolved from.
_REAL_MODELS = {
    "qwen38_27b": dict(hidden_size=5120, num_key_heads=16, num_value_heads=48, output_gate_activation="silu"),
    "qwen36_35b": dict(hidden_size=2048, num_key_heads=16, num_value_heads=32, output_gate_activation="silu"),
    "qwen38_2_4t": dict(hidden_size=8192, num_key_heads=16, num_value_heads=128, output_gate_activation="silu"),
    "qwen38_flash_next": dict(hidden_size=2560, num_key_heads=16, num_value_heads=48, output_gate_activation="sigmoid"),
}
_COMMON = dict(head_k_dim=128, head_v_dim=128, conv_kernel_size=4, norm_eps=1e-6)


def _text(model_config: dict) -> dict:
    return model_config.get("text_config", model_config)


@pytest.mark.parametrize("model", _REAL_MODELS)
def test_real_config_covers_every_field(model: str) -> None:
    expected = _REAL_MODELS[model] | _COMMON
    assert set(expected) == {field.name for field in fields(GDNConfig)}, "a GDNConfig field has no source check"
    assert asdict(qwen_gdn_config(model)) == expected


@pytest.mark.parametrize("model", QWEN_GDN_MODELS)
def test_pinned_config_is_the_upstream_file(model: str) -> None:
    """The in-tree copy is the hub file at the pinned revision plus the final newline pre-commit adds."""
    data = (Path(qwen_models.__file__).with_name("model_configs") / f"{model}.json").read_bytes()
    assert data.endswith(b"\n")
    assert hashlib.sha256(data[:-1]).hexdigest() == QWEN_GDN_MODELS[model].config_sha256


def test_text_config_nesting_follows_the_checkpoint() -> None:
    """2.4T is the one flat (text-only) config; the others nest the text tower in text_config."""
    nested = {model: "text_config" in qwen_model_config(model) for model in QWEN_GDN_MODELS}
    assert nested == {"qwen38_27b": True, "qwen36_35b": True, "qwen38_2_4t": False, "qwen38_flash_next": True}


@pytest.mark.parametrize("model", _REAL_MODELS)
def test_real_config_rejects_unmodeled_linear_key(model: str, expect_error) -> None:
    model_config = qwen_model_config(model)
    _text(model_config)["linear_chunk_size"] = 64
    with expect_error(ValueError, "linear_chunk_size"):
        GDNConfig.from_model_config(model_config)


def test_rejects_unknown_or_missing_model_type(expect_error) -> None:
    model_config = qwen_model_config("qwen38_27b")
    model_config["text_config"]["model_type"] = "qwen3_next"
    with expect_error(ValueError, "qwen3_next"):
        GDNConfig.from_model_config(model_config)
    del model_config["text_config"]["model_type"]
    with expect_error(ValueError, "model_type"):
        GDNConfig.from_model_config(model_config)


@pytest.mark.parametrize("model", ["qwen38_27b", "qwen36_35b", "qwen38_2_4t"])
@pytest.mark.parametrize("output_gate_type", [None, "silu", "swish"])
def test_qwen3_5_output_gate_is_silu(model: str, output_gate_type) -> None:
    """qwen3_5 / qwen3_5_moe: unset, silu and swish all mean the hard-wired silu gate."""
    model_config = qwen_model_config(model)
    _text(model_config).pop("output_gate_type", None)
    if output_gate_type is not None:
        _text(model_config)["output_gate_type"] = output_gate_type
    assert GDNConfig.from_model_config(model_config).output_gate_activation == "silu"


@pytest.mark.parametrize("model", ["qwen38_27b", "qwen38_2_4t"])
def test_qwen3_5_rejects_sigmoid_output_gate(model: str, expect_error) -> None:
    """transformers qwen3_5 would still apply silu, vLLM sigmoid: the references disagree, so reject."""
    model_config = qwen_model_config(model)
    _text(model_config)["output_gate_type"] = "sigmoid"
    with expect_error(ValueError, "output_gate_type 'sigmoid'"):
        GDNConfig.from_model_config(model_config)


def test_qwen4_exp_output_gate_resolution(expect_error) -> None:
    """qwen4_exp: output_gate_type or hidden_act, only silu and sigmoid accepted (swish raises in transformers)."""
    model_config = qwen_model_config("qwen38_flash_next")
    assert GDNConfig.from_model_config(model_config).output_gate_activation == "sigmoid"
    del model_config["text_config"]["output_gate_type"]
    assert GDNConfig.from_model_config(model_config).output_gate_activation == "silu"
    model_config["text_config"]["output_gate_type"] = "swish"
    with expect_error(ValueError, "'swish'"):
        GDNConfig.from_model_config(model_config)


@pytest.mark.parametrize(
    "update, message",
    [
        pytest.param({"hidden_act": "gelu"}, "hidden_act", id="hidden_act-not-silu"),
        pytest.param({"linear_conv_kernel_dim": 3}, "conv_kernel_size=4", id="conv-not-4"),
        pytest.param({"mamba_ssm_dtype": "bfloat16"}, "mamba_ssm_dtype", id="state-dtype-not-fp32"),
        pytest.param({"linear_num_value_heads": 40}, "multiple of num_key_heads", id="value-heads-not-multiple"),
    ],
)
@pytest.mark.parametrize("model", ["qwen38_27b", "qwen38_2_4t", "qwen38_flash_next"])
def test_real_config_rejects_unsupported_values(model: str, update: dict, message: str, expect_error) -> None:
    model_config = qwen_model_config(model)
    _text(model_config).update(update)
    with expect_error(ValueError, message):
        GDNConfig.from_model_config(model_config)


def test_missing_field_is_named(expect_error) -> None:
    model_config = qwen_model_config("qwen38_2_4t")
    del model_config["linear_key_head_dim"]
    with expect_error(ValueError, "linear_key_head_dim"):
        GDNConfig.from_model_config(model_config)
