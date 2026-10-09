# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks of the layer-streamed HF reference (no device).

Each module is built from only its own snapshot tensors with ``strict=True``; the full 54 GB
model is never instantiated.
"""

import pytest
import torch

from models.demos.pplx_decider_v1_27b.reference import hf_reference as ref
from models.demos.pplx_decider_v1_27b.tests.test_utils import reader


@pytest.mark.parametrize("layer_idx, kind", [(0, "linear_attention"), (3, "full_attention")])
def test_layer_strict_load(layer_idx, kind):
    sd = reader().layer_state_dict(layer_idx)
    layer = ref.build_decoder_layer(reader(), layer_idx)  # load_state_dict(strict=True) inside
    assert layer.layer_type == kind
    assert set(layer.state_dict()) == set(sd)
    for key, value in layer.state_dict().items():
        assert torch.equal(value, sd[key].float()), key
    # no key leaks across layers: exactly this layer's prefix
    expected = {k for k in reader().weight_map if k.startswith(ref.LAYER_PREFIX.format(layer_idx))}
    assert {ref.LAYER_PREFIX.format(layer_idx) + k for k in sd} == expected


def test_non_layer_modules_strict_load():
    cfg = reader().text_config
    assert ref.build_final_norm(reader()).weight.shape == (cfg.hidden_size,)
    readout = ref.build_readout(reader())
    assert readout.weight.shape == (ref.NUM_OPTIONS, cfg.hidden_size) and readout.bias is None
    emb = ref.build_embedding(reader(), dtype=torch.bfloat16)
    assert emb.weight.shape == (cfg.vocab_size, cfg.hidden_size)
    assert reader().temperature > 0
