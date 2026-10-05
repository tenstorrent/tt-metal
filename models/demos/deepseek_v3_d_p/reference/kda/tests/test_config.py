# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for KDA semantic configuration."""

import copy
from dataclasses import asdict, fields
from typing import Any, Callable

import pytest

from models.demos.deepseek_v3_d_p.reference.glm_5_3_flash_config import glm_5_3_flash_model_config
from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig
from models.demos.deepseek_v3_d_p.reference.kda.tests.helpers import make_config
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import kimi_k3_model_config

# Every KDAConfig field, as stated by each pinned config.json (values read off the files by hand).
_REAL_MODELS: dict[str, tuple[Callable[[], dict[str, Any]], dict[str, Any]]] = {
    "kimi_k3": (
        kimi_k3_model_config,
        {
            "hidden_size": 7168,
            "num_heads": 96,
            "head_k_dim": 128,
            "head_v_dim": 128,
            "conv_kernel_size": 4,
            "norm_eps": 1e-5,
            "use_full_rank_gate": True,
            "gate_lower_bound": -5.0,
        },
    ),
    "glm_5_3_flash": (
        glm_5_3_flash_model_config,
        {
            "hidden_size": 4096,
            "num_heads": 64,
            "head_k_dim": 128,
            "head_v_dim": 128,
            "conv_kernel_size": 4,
            "norm_eps": 1e-5,
            "use_full_rank_gate": False,
            "gate_lower_bound": -5.0,
        },
    ),
}


def _linear_attn_config(model_config: dict[str, Any]) -> dict[str, Any]:
    return model_config["text_config"]["linear_attn_config"]


def test_model_config_mapping() -> None:
    config = KDAConfig.from_model_config(
        {
            "model_type": "kimi_linear",
            "hidden_size": 2304,
            "rms_norm_eps": 1e-5,
            "linear_attn_config": {
                "head_dim": 128,
                "num_heads": 32,
                "short_conv_kernel_size": 4,
                "use_full_rank_gate": True,
                "gate_lower_bound": -5.0,
            },
        }
    )
    assert (config.q_dim, config.k_dim, config.v_dim) == (4096, 4096, 4096)
    assert config.use_full_rank_gate
    assert config.gate_lower_bound == -5.0


def test_nested_text_config_mapping() -> None:
    base = make_config()
    mapped = KDAConfig.from_model_config(
        {
            "text_config": {
                "model_type": "kimi_linear",
                "hidden_size": base.hidden_size,
                "rms_norm_eps": base.norm_eps,
                "linear_attn_config": {
                    "head_dim": base.head_k_dim,
                    "num_heads": base.num_heads,
                    "short_conv_kernel_size": base.conv_kernel_size,
                },
            }
        }
    )
    assert mapped == base


@pytest.mark.parametrize("field", ["hidden_size", "num_heads", "head_k_dim", "head_v_dim"])
def test_config_rejects_nonpositive_dimensions(field: str, expect_error) -> None:
    values = make_config().__dict__.copy()
    values[field] = 0
    with expect_error(ValueError, field):
        KDAConfig(**values)


def test_config_rejects_invalid_numerical_policy(expect_error) -> None:
    values = make_config().__dict__.copy()
    with expect_error(ValueError, "conv_kernel_size=4"):
        KDAConfig(**(values | {"conv_kernel_size": 3}))
    with expect_error(ValueError, "norm_eps"):
        KDAConfig(**(values | {"norm_eps": 0.0}))
    for norm_eps in (float("nan"), float("inf")):
        with expect_error(ValueError, "norm_eps"):
            KDAConfig(**(values | {"norm_eps": norm_eps}))
    with expect_error(ValueError, "gate_lower_bound"):
        KDAConfig(**(values | {"gate_lower_bound": 0.0}))


@pytest.mark.parametrize("model", _REAL_MODELS)
def test_real_config_covers_every_field(model: str) -> None:
    load, expected = _REAL_MODELS[model]
    assert set(expected) == {field.name for field in fields(KDAConfig)}, "a KDAConfig field has no source check"
    assert asdict(KDAConfig.from_model_config(load())) == expected


@pytest.mark.parametrize("model", _REAL_MODELS)
def test_real_config_rejects_unmodeled_linear_attn_key(model: str, expect_error) -> None:
    model_config = _REAL_MODELS[model][0]()
    _linear_attn_config(model_config)["chunk_size"] = 64
    with expect_error(ValueError, "chunk_size"):
        KDAConfig.from_model_config(model_config)


def test_rejects_unknown_or_missing_model_type(expect_error) -> None:
    model_config = kimi_k3_model_config()
    model_config["text_config"]["model_type"] = "qwen3_next"
    with expect_error(ValueError, "qwen3_next"):
        KDAConfig.from_model_config(model_config)
    del model_config["text_config"]["model_type"]
    with expect_error(ValueError, "model_type"):
        KDAConfig.from_model_config(model_config)


def test_glm_rejects_full_rank_gate_key(expect_error) -> None:
    """GLM-5.3-Flash always uses the low-rank gate, so the Kimi gate switch is not a GLM key."""
    model_config = glm_5_3_flash_model_config()
    _linear_attn_config(model_config)["use_full_rank_gate"] = True
    with expect_error(ValueError, "use_full_rank_gate"):
        KDAConfig.from_model_config(model_config)


@pytest.mark.parametrize(
    "linear_attn_update, expected_bound",
    [
        pytest.param({"gate_lower_bound": None}, -5.0, id="null_bound-safe_gate_default"),
        pytest.param({"gate_lower_bound": None, "safe_gate": True}, -5.0, id="null_bound-safe_gate"),
        pytest.param({"gate_lower_bound": None, "safe_gate": False}, None, id="null_bound-unsafe_gate"),
        pytest.param({"gate_lower_bound": -2.0, "safe_gate": False}, -2.0, id="explicit_bound-unsafe_gate"),
    ],
)
def test_glm_gate_lower_bound_follows_glm_defaults(
    linear_attn_update: dict[str, Any], expected_bound: float | None
) -> None:
    model_config = glm_5_3_flash_model_config()
    _linear_attn_config(model_config).update(linear_attn_update)
    assert KDAConfig.from_model_config(model_config).gate_lower_bound == expected_bound


@pytest.mark.parametrize(
    "load, expected_bound",
    [
        pytest.param(glm_5_3_flash_model_config, -5.0, id="glm_5_3_flash-default_bound"),
        pytest.param(kimi_k3_model_config, None, id="kimi_k3-softplus"),
    ],
)
def test_missing_gate_lower_bound_uses_model_default(
    load: Callable[[], dict[str, Any]], expected_bound: float | None
) -> None:
    model_config = load()
    del _linear_attn_config(model_config)["gate_lower_bound"]
    assert KDAConfig.from_model_config(model_config).gate_lower_bound == expected_bound


def test_glm_conv_activation_must_be_silu(expect_error) -> None:
    model_config = glm_5_3_flash_model_config()
    assert model_config["text_config"]["hidden_act"] == "silu"
    model_config["text_config"]["hidden_act"] = "gelu"
    with expect_error(ValueError, "hidden_act"):
        KDAConfig.from_model_config(model_config)


def test_kimi_hidden_act_does_not_select_conv_activation() -> None:
    """K3's hidden_act is its MLP activation; the KDA convolution is SiLU regardless."""
    model_config = kimi_k3_model_config()
    assert model_config["text_config"]["hidden_act"] == "situ"
    expected = KDAConfig.from_model_config(model_config)
    changed = copy.deepcopy(model_config)
    changed["text_config"]["hidden_act"] = "gelu"
    assert KDAConfig.from_model_config(changed) == expected


def test_config_module_does_not_import_ttnn() -> None:
    import models.demos.deepseek_v3_d_p.reference.kda.config as config_module

    assert "ttnn" not in config_module.__dict__
