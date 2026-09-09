# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One complete decoder layer, residuals included, vs the torch reference.

Target mesh (8, 4), random weights, identical on both sides. Structure follows
`minimax_m3/tests/unit/test_decoder_layer_vs_ref.py`.

The composition test — run only after every piece above it passes alone. What it adds beyond the
sum of its parts is the residual stream: whether the two adds see correctly-sharded operands, and
whether the per-norm all-gather and the two closing collectives leave the stream in the layout the
next block expects. Those are exactly the errors that a per-block test cannot see, because each
block is fed a correctly-shaped input by its own test.

Both residual schemes are covered: they place different collectives in different places, and the
package ships with sharded as the default.

All 32 Llama layers are structurally identical, so one layer here is representative of every layer
— unlike the donor, whose dense and sparse layers each needed their own composition test.
"""

import pytest
import torch

import ttnn
from models.demos.llama_3_1_8b_d_p.reference.model import REF_DTYPE, RefDecoderLayer, RefRotaryEmbedding, causal_mask
from models.demos.llama_3_1_8b_d_p.tt.layer import DecoderLayer
from models.demos.llama_3_1_8b_d_p.tt.rope import create_rope_setup

from ..test_factory import ACT_DTYPE, CHUNK_SIZE, WEIGHT_DTYPE, assert_pcc, parametrize_target_mesh, sp_shard_rope
from .test_attention_vs_ref import meta_rope_tables

SEQ = 2048


@parametrize_target_mesh()
@pytest.mark.parametrize("sharded_residual", [True, False], ids=["sharded_residual", "replicated_residual"])
def test_decoder_layer_vs_ref(
    mesh_device, device_params, config, hf_config, mesh_config, ccl_manager, topology_name, sharded_residual, monkeypatch
):
    """Layer output vs the torch reference, for both residual-stream layouts."""
    monkeypatch.setenv("LLAMA31_8B_SHARDED_RESIDUAL", "1" if sharded_residual else "0")

    torch.manual_seed(0)
    x = torch.randn(1, SEQ, config.hidden_size, dtype=REF_DTYPE)
    ref = RefDecoderLayer(config)
    rope = RefRotaryEmbedding(config)
    cos, sin = rope(torch.arange(SEQ)[None, :], dtype=REF_DTYPE)
    with torch.no_grad():
        golden = ref(x, (cos, sin), causal_mask(SEQ))
    state_dict = {k: v.detach().clone() for k, v in ref.state_dict().items()}

    rope_setup = create_rope_setup(mesh_device, hf_config, max_seq_len=SEQ)
    layer = DecoderLayer(
        mesh_device,
        hf_config,
        state_dict,
        layer_idx=0,
        ccl_manager=ccl_manager,
        mesh_config=mesh_config,
        max_seq_len=SEQ,
        chunk_size=CHUNK_SIZE,
        weight_dtype=WEIGHT_DTYPE,
        transformation_mats=rope_setup.transformation_mat_prefill,
    )
    assert layer.sharded_residual == sharded_residual

    def to_dev(t, dims):
        return ttnn.from_torch(
            t,
            device=mesh_device,
            dtype=ACT_DTYPE,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims),
        )

    # The residual enters in whatever layout the layer ships with: emb/tp when sharded, full emb
    # when replicated.
    in_dims = [2, 3] if sharded_residual else [2, None]
    tt_x = to_dev(x.reshape(1, 1, SEQ, config.hidden_size), in_dims)
    cos_meta, sin_meta = meta_rope_tables(mesh_device, rope_setup, config.head_dim, 0, SEQ)
    # SP-sharded, not replicated: each SP row needs the cos/sin rows for ITS positions.
    rope_mats = (sp_shard_rope(mesh_device, mesh_config, cos_meta), sp_shard_rope(mesh_device, mesh_config, sin_meta))

    out = layer(tt_x, rope_mats=rope_mats, kv_cache=None, cached_len=0, logical_n=SEQ)
    ttnn.synchronize_device(mesh_device)

    if sharded_residual:
        got = ttnn.to_torch(
            out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 3), mesh_shape=tuple(mesh_device.shape))
        )
    else:
        got = ttnn.to_torch(
            out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 0), mesh_shape=tuple(mesh_device.shape))
        )[:1]

    assert_pcc(
        f"decoder_layer[{'sharded' if sharded_residual else 'replicated'}_residual]",
        golden.reshape(1, 1, SEQ, config.hidden_size),
        got.to(REF_DTYPE),
        topology_name,
    )


@parametrize_target_mesh()
def test_all_layers_are_structurally_identical(mesh_device, device_params, hf_config, mesh_config, ccl_manager):
    """No per-layer type dispatch: layer 0 and layer 31 build the same blocks.

    Cheap, and it pins the one architectural simplification this package makes over its donor. If a
    hybrid schedule ever appeared, this is where it would have to be handled deliberately rather
    than by a `getattr(hf_config, "moe_layer_freq", None)` that silently defaults.
    """
    layers = [
        DecoderLayer(
            mesh_device,
            hf_config,
            {},
            layer_idx=idx,
            ccl_manager=ccl_manager,
            mesh_config=mesh_config,
            max_seq_len=SEQ,
            chunk_size=CHUNK_SIZE,
        )
        for idx in (0, hf_config.num_hidden_layers - 1)
    ]
    kinds = [(type(l.self_attn).__name__, type(l.mlp).__name__, type(l.input_layernorm).__name__) for l in layers]
    assert kinds[0] == kinds[1], f"layer 0 and layer 31 differ: {kinds}"
    assert not hasattr(hf_config, "moe_layer_freq"), "a hybrid schedule appeared; per-layer dispatch is now needed"
