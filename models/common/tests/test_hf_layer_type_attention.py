# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for ``models.common.hf_layer_type_attention.layer_type_attention``.

tt-metal pins a transformers release that stores the full attention values in
global fields, while the vLLM plugin installs one that stores them in
``per_layer_config``. The stand-in configs cover both layouts on any
transformers version; the checked-in Gemma4 configs cover the layout of the
installed transformers.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from models.common.hf_layer_type_attention import LayerTypeAttention, layer_type_attention

GEMMA4_CONFIGS = Path(__file__).resolve().parents[2] / "demos" / "gemma4" / "configs"


class _HeterogeneousConfig:
    """Stand-in for a transformers >= 5.15 text config with ``per_layer_config``.

    Top-level per-layer attributes raise, and ``per_layer_config`` accepts layer
    type keys only.
    """

    is_heterogeneous = True

    def __init__(self, layer_types, per_type, num_hidden_layers=None):
        self.layer_types = layer_types
        self.num_hidden_layers = len(layer_types) if num_hidden_layers is None else num_hidden_layers
        self.per_layer_config = {
            layer_type: SimpleNamespace(num_key_value_heads=kv_heads, head_dim=head_dim)
            for layer_type, (kv_heads, head_dim) in per_type.items()
        }

    @property
    def head_dim(self):
        raise RuntimeError("head_dim is a per-layer attribute")

    @property
    def num_key_value_heads(self):
        raise RuntimeError("num_key_value_heads is a per-layer attribute")


LAYER_TYPES = ["sliding_attention"] * 5 + ["full_attention"]


def test_global_fields():
    config = SimpleNamespace(num_key_value_heads=8, head_dim=256, num_global_key_value_heads=2, global_head_dim=512)
    assert layer_type_attention(config) == LayerTypeAttention(8, 256, 2, 512)


def test_global_fields_unset_full_values_take_sliding_values():
    config = SimpleNamespace(num_key_value_heads=8, head_dim=256, num_global_key_value_heads=None)
    assert layer_type_attention(config) == LayerTypeAttention(8, 256, 8, 256)


def test_per_layer_config():
    config = _HeterogeneousConfig(LAYER_TYPES, {"sliding_attention": (8, 256), "full_attention": (2, 512)})
    assert layer_type_attention(config) == LayerTypeAttention(8, 256, 2, 512)


def test_per_layer_config_with_fewer_hidden_layers_than_layer_types():
    # Unit tests set num_hidden_layers = 1 and keep the full layer_types list.
    config = _HeterogeneousConfig(
        LAYER_TYPES, {"sliding_attention": (8, 256), "full_attention": (2, 512)}, num_hidden_layers=1
    )
    assert layer_type_attention(config) == LayerTypeAttention(8, 256, 2, 512)


@pytest.mark.parametrize(
    "layer_type, values",
    [("sliding_attention", (8, 256)), ("full_attention", (2, 512))],
)
def test_per_layer_config_with_one_layer_type(layer_type, values):
    config = _HeterogeneousConfig([layer_type] * 4, {layer_type: values})
    assert layer_type_attention(config) == LayerTypeAttention(*values, *values)


def test_per_layer_config_without_attention_layer_types(expect_error):
    config = _HeterogeneousConfig(["linear_attention"] * 2, {})
    with expect_error(ValueError, "layer_types"):
        layer_type_attention(config)


@pytest.mark.parametrize(
    "name, expected",
    [
        ("gemma-4-12B-it", LayerTypeAttention(8, 256, 1, 512)),
        ("gemma-4-26B-A4B-it", LayerTypeAttention(8, 256, 2, 512)),
        ("gemma-4-31B-it", LayerTypeAttention(16, 256, 4, 512)),
        ("gemma-4-E2B-it", LayerTypeAttention(1, 256, 1, 512)),
        ("gemma-4-E4B-it", LayerTypeAttention(2, 256, 2, 512)),
    ],
)
@pytest.mark.parametrize("num_hidden_layers", [None, 1])
def test_checked_in_gemma4_configs(name, expected, num_hidden_layers):
    transformers = pytest.importorskip("transformers")
    text_config = transformers.AutoConfig.from_pretrained(GEMMA4_CONFIGS / name).get_text_config()
    if num_hidden_layers is not None:
        text_config.num_hidden_layers = num_hidden_layers
    assert layer_type_attention(text_config) == expected
