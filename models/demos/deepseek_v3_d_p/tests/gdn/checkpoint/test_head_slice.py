# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real Qwen GDN layers (local only): one layer loads from its checkpoint and its LB-B K-group slices are the Galaxy
TP4 rank oracles. Fetch first, one model at a time, and delete after:

    python -m models.demos.deepseek_v3_d_p.tests.gdn.prepare --fetch-layer <model>
    TT_METAL_MOCK_CLUSTER_DESC_PATH=<yaml> pytest models/demos/deepseek_v3_d_p/tests/gdn/checkpoint/test_head_slice.py -k <model>
    python -m models.demos.deepseek_v3_d_p.tests.gdn.prepare --delete-layer <model>
"""

from pathlib import Path

import pytest
import torch

from models.demos.deepseek_v3_d_p.reference.gdn.head_slice import gdn_head_slice_config, slice_gdn_heads
from models.demos.deepseek_v3_d_p.reference.gdn.layer import gdn_forward_reference
from models.demos.deepseek_v3_d_p.reference.gdn.qwen_models import (
    QWEN_FIRST_GDN_LAYER,
    QWEN_GDN_MODELS,
    qwen_gdn_config,
)
from models.demos.deepseek_v3_d_p.reference.gdn.tests.test_head_slice import (
    _nonzero_state,
    _rank_partial_reference,
    _restrict_state,
)
from models.demos.deepseek_v3_d_p.tests.gdn.checkpoint_utils import (
    QWEN_LAYER_0_SHA256,
    gdn_checkpoint_dir,
    gdn_state_dict_sha256,
    load_gdn_layer_state_dict,
)

# LoudBox LB-B runs on each chip the heads one Galaxy TP4 rank owns.
_GALAXY_TP = 4
# K / V heads per LB-B chip: 16 K heads / 4 with their V-head groups.
_LB_B_CHIP_HEADS = {"qwen38_27b": (4, 12), "qwen36_35b": (4, 8), "qwen38_2_4t": (4, 32), "qwen38_flash_next": (4, 12)}


@pytest.mark.parametrize("model", QWEN_GDN_MODELS)
def test_lb_b_chip_config_is_galaxy_tp4_k_group(model: str) -> None:
    config = qwen_gdn_config(model)
    chip = gdn_head_slice_config(config, config.num_key_heads // _GALAXY_TP)
    assert (chip.num_key_heads, chip.num_value_heads) == _LB_B_CHIP_HEADS[model]


def _checkpoint(model: str) -> Path:
    directory = gdn_checkpoint_dir(model, QWEN_FIRST_GDN_LAYER)
    if not (directory / "model.safetensors.index.json").is_file():
        pytest.skip(f"fetch first: python -m models.demos.deepseek_v3_d_p.tests.gdn.prepare --fetch-layer {model}")
    return directory


@pytest.mark.parametrize("model", QWEN_GDN_MODELS)
def test_real_layer_loads_pinned_weights(model: str) -> None:
    config = qwen_gdn_config(model)
    state_dict = load_gdn_layer_state_dict(_checkpoint(model), QWEN_FIRST_GDN_LAYER, config)
    assert gdn_state_dict_sha256(state_dict) == QWEN_LAYER_0_SHA256[model]


@pytest.mark.parametrize("model", QWEN_GDN_MODELS)
def test_real_layer_slice_is_tp_rank_partial(model: str) -> None:
    """Galaxy TP4 ranks 0 and 3: the K-group slice's reference equals the full layer's rank partial and restricted
    state (nonzero carried state); a different rank's partial does not match (negative control)."""
    config = qwen_gdn_config(model)
    state_dict = load_gdn_layer_state_dict(_checkpoint(model), QWEN_FIRST_GDN_LAYER, config)
    weights = {name: tensor.float() for name, tensor in state_dict.items()}
    hidden = (
        torch.randn(32, config.hidden_size, generator=torch.Generator().manual_seed(1607)).to(torch.bfloat16).float()
    )
    state = _nonzero_state(config, seed=8)
    _, full_state = gdn_forward_reference(hidden, weights, config, state)
    heads = config.num_key_heads // _GALAXY_TP
    partials = {}
    for rank in (0, _GALAXY_TP - 1):
        start = rank * heads
        output, sliced_state = gdn_forward_reference(
            hidden,
            slice_gdn_heads(weights, config, key_head_start=start, num_key_heads=heads),
            gdn_head_slice_config(config, heads),
            _restrict_state(state, config, start, heads),
        )
        partials[rank] = _rank_partial_reference(hidden, weights, config, state, start, heads)
        expected_state = _restrict_state(full_state, config, start, heads)
        torch.testing.assert_close(output, partials[rank], atol=1e-4, rtol=1e-4, msg=f"rank {rank} output")
        torch.testing.assert_close(sliced_state.recurrent, expected_state.recurrent, atol=1e-4, rtol=1e-4)
        assert torch.equal(sliced_state.conv, expected_state.conv)
    assert not torch.allclose(partials[0], partials[_GALAXY_TP - 1], atol=1e-3)
