# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One complete decoder layer, with residuals, vs the torch reference — both layer types.

The composition test, run after every piece passes alone. What it adds over the piece tests is the
residual bookkeeping and the layer-type dispatch: the layer must pick a Gated DeltaNet mixer for
index 0 and an attention mixer for index 3, and the two must plug into an identical residual
shape.
"""

from __future__ import annotations

import pytest
import torch

from models.demos.qwen_3_8_27b_d_p.reference.config import FULL_ATTENTION, LINEAR_ATTENTION
from models.demos.qwen_3_8_27b_d_p.reference.modeling import (
    Qwen35DecoderLayer,
    Qwen35RotaryEmbedding,
    init_random_weights,
)
from models.demos.qwen_3_8_27b_d_p.tt.attention.prefill import Attention
from models.demos.qwen_3_8_27b_d_p.tt.caches import allocate_prefill_caches
from models.demos.qwen_3_8_27b_d_p.tt.context import ChunkContext
from models.demos.qwen_3_8_27b_d_p.tt.gdn.prefill import GatedDeltaNet
from models.demos.qwen_3_8_27b_d_p.tt.layer import DecoderLayer
from models.demos.qwen_3_8_27b_d_p.tt.rope import RotarySetup

from ..test_factory import mesh_setup, parametrize_mesh, unit_test_config
from .helpers import CACHE_DTYPE, WEIGHT_DTYPE, assert_tp_replicated, check_pcc, from_sp_sharded, randn, to_sp_sharded

S_LOCAL = 128
LAYER_OF_TYPE = {LINEAR_ATTENTION: 0, FULL_ATTENTION: 3}


@parametrize_mesh()
@pytest.mark.parametrize("layer_type", [LINEAR_ATTENTION, FULL_ATTENTION])
def test_decoder_layer_vs_ref(mesh, submesh_shape, device_params, layer_type):
    cfg = unit_test_config()
    mesh_config, ccl = mesh_setup(mesh)
    layer_idx = LAYER_OF_TYPE[layer_type]
    total = S_LOCAL * mesh_config.sp

    ref = Qwen35DecoderLayer(cfg, layer_idx).eval()
    init_random_weights(ref, seed=61)
    x = randn(1, total, cfg.hidden_size, seed=62, scale=0.5)
    cos, sin = Qwen35RotaryEmbedding(cfg)(torch.arange(total)[None, :])
    with torch.no_grad():
        expected, _state = ref(x, (cos, sin))

    layer = DecoderLayer(
        mesh,
        cfg,
        ref.state_dict(),
        layer_idx,
        mesh_config=mesh_config,
        ccl_manager=ccl,
        weight_dtype=WEIGHT_DTYPE,
        cache_dtype=CACHE_DTYPE,
    )
    expected_mixer = GatedDeltaNet if layer_type == LINEAR_ATTENTION else Attention
    assert isinstance(layer.mixer, expected_mixer), f"layer {layer_idx} picked the wrong mixer"

    caches = allocate_prefill_caches(
        mesh,
        cfg,
        mesh_config=mesh_config,
        max_seq_len=total,
        layer_indices=[layer_idx],
        cache_dtype=CACHE_DTYPE,
    )
    tt_cos, tt_sin = RotarySetup(mesh, cfg, mesh_config).chunk_mats(0, total)
    ctx = ChunkContext(cos=tt_cos, sin=tt_sin, caches=caches, cached_len=0)
    tt_x = to_sp_sharded(x.reshape(1, 1, total, cfg.hidden_size), mesh, mesh_config)
    out = layer(tt_x, ctx)
    assert_tp_replicated(out, mesh_config, f"decoder layer[{layer_type}] output")
    got = from_sp_sharded(out, mesh_config)
    check_pcc(f"decoder_layer[{layer_type}]", expected.reshape(1, 1, total, cfg.hidden_size), got)
