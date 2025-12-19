# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

from models.experimental.dpt_large.tt.config import DPTLargeConfig
from models.experimental.dpt_large.tt.weights import convert_weights_for_tt

transformers = pytest.importorskip("transformers")
from transformers import DPTConfig, DPTForDepthEstimation  # noqa: E402


def test_weight_conversion_passthrough():
    cfg = DPTLargeConfig(
        image_size=64,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
    )
    hf_cfg = DPTConfig(**cfg.to_hf_kwargs())
    model = DPTForDepthEstimation(hf_cfg)
    state_dict = model.state_dict()
    converted = convert_weights_for_tt(state_dict)
    assert len(converted) == len(state_dict)
    # sanity check: pick a parameter and compare
    any_key = next(iter(state_dict))
    assert (converted[any_key] - state_dict[any_key]).abs().max() == 0
