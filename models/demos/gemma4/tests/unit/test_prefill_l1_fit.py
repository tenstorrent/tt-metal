# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Regression test for ``dram_sharded.linear_l1_safe``.

The tuned prefill matmul program configs were swept at the shipped tensor
parallelism, and their gate is a row-count band that does not check the L1
footprint. At TP=1/2 on Wormhole the un-fractured weight makes the same block
sizes demand 1.7-4.3 MB against a 1.46 MB L1, and the op throws before computing
anything; ``linear_l1_safe`` catches that once per shape and falls back to ttnn
auto.

The property that matters here is the *other* half: on the meshes the configs
were tuned for, the fallback must never fire, so those programs — and their
measured perf — are byte-for-byte what they always were. This test asserts that
directly.
"""

import pytest
import torch

import ttnn
from models.demos.gemma4.tt.dram_sharded import l1_fallback_shapes, reset_l1_fallback_shapes
from models.demos.gemma4.tt.layer import Gemma4DecoderLayer

from ...tests.test_factory import TestFactory, parametrize_mesh_with_fabric
from .test_layer import (
    _create_gemma4_model_args,
    _create_hf_reference_layer,
    _create_hf_text_config,
    _hf_state_to_tt_state,
)

# Below the cutoff the tuned 1D config is what ships; above it the cutoff-reshape
# path is. Cover both so a regression in either is caught.
SEQ_LENS = [1024, 4096]


@parametrize_mesh_with_fabric()
@pytest.mark.parametrize("seq_len", SEQ_LENS, ids=[f"prefill_{s}" for s in SEQ_LENS])
def test_tuned_prefill_config_fits_l1(seq_len, mesh_device, reset_seeds):
    """On a tuned mesh (tp >= 4) no prefill matmul may fall back to ttnn auto.

    A fallback here means a tuned config stopped fitting L1 — i.e. the shipped
    program silently changed, taking its perf with it.
    """
    tp = mesh_device.shape[1] if hasattr(mesh_device, "shape") else 1
    layer_idx = 0

    hf_text_config = _create_hf_text_config(num_experts=4, top_k=2)
    hf_layer = _create_hf_reference_layer(hf_text_config, layer_idx)
    tt_state = _hf_state_to_tt_state(hf_layer.state_dict(), layer_idx)
    model_args = _create_gemma4_model_args(hf_text_config)

    from models.demos.gemma4.config import MeshConfig, ModeConfig
    from models.demos.gemma4.tt.ccl import CCLManager

    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=tp))
    ccl_manager = CCLManager(mesh_device, num_links=1) if tp > 1 else None

    tt_layer = Gemma4DecoderLayer(
        mesh_device=mesh_device,
        hf_config=model_args,
        state_dict=tt_state,
        layer_idx=layer_idx,
        ccl_manager=ccl_manager,
        dtype=ttnn.bfloat16,
        tensor_cache_path=None,
        mesh_config=mesh_config,
        max_seq_len=seq_len,
        max_local_batch_size=1,
    )

    # Start clean: a verdict cached by an earlier mesh in the same session would
    # skip the tuned config outright and make the assertion below pass vacuously.
    reset_l1_fallback_shapes()

    x_torch = torch.randn(1, seq_len, hf_text_config.hidden_size, dtype=torch.float32)
    x_tt = ttnn.from_torch(
        x_torch.unsqueeze(0).to(torch.bfloat16),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if tp > 1 else None,
    )
    cos_tt, sin_tt = TestFactory.create_tt_rope_cache(mesh_device, hf_text_config, max(seq_len, 128), layer_idx)

    tt_layer(
        x_tt,
        rope_mats=(cos_tt, sin_tt),
        position_idx=None,
        page_table=None,
        kv_cache=None,
        is_decode=False,
    )

    fired = l1_fallback_shapes()
    if tp >= 4:
        assert not fired, (
            f"Tuned prefill matmul config no longer fits L1 at tp={tp}, seq={seq_len}: "
            f"shapes {sorted(fired)} fell back to ttnn auto. The shipped program — and its "
            f"measured perf — changed."
        )
