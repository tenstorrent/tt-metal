# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Distributed weight sharding tests for Qwen3.5-27B TTNN modules.

Tests run on mesh_device (T3K) to validate:
1. Weights are col-sharded across devices (each device has out_features/num_devices columns)
2. Forward passes produce correct output (PCC >= 0.90 vs PyTorch reference)
3. All-gather correctly reconstructs replicated tensors from col-sharded projections
"""

import os

import pytest
import torch

from .conftest import (
    assert_with_pcc,
    get_config_attr,
    get_layer_type,
    skip_no_ttnn,
    skip_no_transformers,
    skip_no_symbiote,
    TTNN_AVAILABLE,
    TT_SYMBIOTE_AVAILABLE,
)

if TTNN_AVAILABLE:
    import ttnn

if TT_SYMBIOTE_AVAILABLE:
    from models.experimental.tt_symbiote.modules.qwen35_gated_deltanet import TTNNQwen35GatedDeltaNet
    from models.experimental.tt_symbiote.modules.qwen35_attention import TTNNQwen35FullAttention
    from models.experimental.tt_symbiote.modules.qwen35_mlp import TTNNQwen35MLP
    from models.experimental.tt_symbiote.modules.qwen35_decoder_layer import TTNNQwen35DecoderLayer
    from models.experimental.tt_symbiote.modules.attention import PagedAttentionConfig
    from models.experimental.tt_symbiote.modules.qwen_attention import TTNNQwenPagedAttentionKVCache
    from models.experimental.tt_symbiote.utils.device_management import set_device
    from models.experimental.tt_symbiote.modules.linear import TTNNLinearIReplicatedWColSharded


def _mesh_to_torch(tensor, mesh_device, batch_size=1):
    """Convert a replicated mesh tensor to a single torch tensor."""
    mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    t = ttnn.to_torch(tensor, mesh_composer=mesh_composer)
    return t[:batch_size]


def _col_sharded_to_torch(tensor, mesh_device):
    """Convert a col-sharded mesh tensor to a full torch tensor.

    Each device has [batch, seq, hidden/N]. Concatenate on dim=-1 to
    reconstruct [batch, seq, hidden].
    """
    mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=-1)
    return ttnn.to_torch(tensor, mesh_composer=mesh_composer)


MESH_DEVICE_MAP = {
    "N150": (1, 1),
    "N300": (1, 2),
    "N150x4": (1, 4),
    "T3K": (1, 8),
    "TG": (8, 4),
    "P150": (1, 1),
    "P300": (1, 2),
    "P150x4": (1, 4),
    "P150x8": (1, 8),
    "BHGLX": (8, 4),
}


@skip_no_ttnn
@skip_no_transformers
@skip_no_symbiote
@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 200000000, "num_command_queues": 1, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [MESH_DEVICE_MAP.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))],
    indirect=True,
)
def test_attention_distributed_weight_sharding(mesh_device, model_4_layers):
    """Full attention with distributed col-sharded weights produces correct output.

    Verifies:
    - Weights use TTNNLinearIReplicatedWColSharded (col-sharded across devices)
    - Prefill output matches PyTorch reference (PCC >= 0.90)
    """
    model, config = model_4_layers
    torch_attn = model.model.layers[3].self_attn

    hidden_size = get_config_attr(config, "hidden_size")
    num_kv_heads = get_config_attr(config, "num_key_value_heads")
    head_dim = get_config_attr(config, "head_dim")
    batch_size, seq_len = 1, 32

    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)
    position_ids = torch.arange(seq_len).unsqueeze(0)
    cos, sin = model.model.rotary_emb(x, position_ids)

    # PyTorch reference
    with torch.no_grad():
        torch_out = torch_attn(x, attention_mask=None, position_embeddings=(cos, sin))[0]

    # TTNN with distributed weights
    ttnn_attn = TTNNQwen35FullAttention.from_torch(torch_attn, distributed=True)
    set_device(ttnn_attn, mesh_device)
    ttnn_attn.preprocess_weights()
    ttnn_attn.move_weights_to_device()

    # Verify weights are col-sharded
    assert isinstance(
        ttnn_attn.q_proj, TTNNLinearIReplicatedWColSharded
    ), f"Expected TTNNLinearIReplicatedWColSharded, got {type(ttnn_attn.q_proj)}"

    # Create paged cache for layer 3
    paged_config = PagedAttentionConfig(block_size=64, max_num_blocks=32, batch_size=batch_size)
    paged_cache = TTNNQwenPagedAttentionKVCache(
        num_layers=1,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        config=paged_config,
        device=None,
        layer_indices=[3],
    ).to_device(mesh_device)

    # Run on mesh device
    tt_input = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_cos = ttnn.from_torch(
        cos,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_sin = ttnn.from_torch(
        sin,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    ttnn_out, _ = ttnn_attn.forward(
        tt_input,
        position_embeddings=(tt_cos, tt_sin),
        past_key_values=paged_cache,
    )

    # Output is col-sharded (no all-gather in o_proj) — concat on dim=-1
    ttnn_out_torch = _col_sharded_to_torch(ttnn_out, mesh_device)
    pcc_val = assert_with_pcc(torch_out, ttnn_out_torch, 0.90)
    print(f"  Distributed attention PCC = {pcc_val:.6f}")


@skip_no_ttnn
@skip_no_transformers
@skip_no_symbiote
@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 200000000, "num_command_queues": 1, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [MESH_DEVICE_MAP.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))],
    indirect=True,
)
def test_mlp_distributed_weight_sharding(mesh_device, model_4_layers):
    """MLP with distributed col-sharded weights produces correct output.

    Verifies:
    - MLP weights use TTNNLinearIReplicatedWColSharded
    - Forward output matches PyTorch reference (PCC >= 0.90)
    """
    model, config = model_4_layers
    torch_mlp = model.model.layers[0].mlp

    hidden_size = get_config_attr(config, "hidden_size")
    batch_size, seq_len = 1, 32

    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)

    # PyTorch reference
    with torch.no_grad():
        torch_out = torch_mlp(x)

    # TTNN with distributed weights
    ttnn_mlp = TTNNQwen35MLP.from_torch(torch_mlp, distributed=True)
    set_device(ttnn_mlp, mesh_device)
    ttnn_mlp.preprocess_weights()
    ttnn_mlp.move_weights_to_device()

    # Verify weights are col-sharded
    assert isinstance(
        ttnn_mlp.gate_proj, TTNNLinearIReplicatedWColSharded
    ), f"Expected TTNNLinearIReplicatedWColSharded, got {type(ttnn_mlp.gate_proj)}"

    tt_input = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    ttnn_out = ttnn_mlp.forward(tt_input)

    # Output is col-sharded (no all-gather in down_proj) — concat on dim=-1
    ttnn_out_torch = _col_sharded_to_torch(ttnn_out, mesh_device)
    pcc_val = assert_with_pcc(torch_out, ttnn_out_torch, 0.90)
    print(f"  Distributed MLP PCC = {pcc_val:.6f}")


@skip_no_ttnn
@skip_no_transformers
@skip_no_symbiote
@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 200000000, "num_command_queues": 1, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [MESH_DEVICE_MAP.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))],
    indirect=True,
)
def test_gdn_distributed_weight_sharding(mesh_device, model_4_layers):
    """GDN with distributed col-sharded weights produces correct output.

    Verifies:
    - GDN weights use TTNNLinearIReplicatedWColSharded
    - Prefill output matches PyTorch reference (PCC >= 0.90)
    """
    model, config = model_4_layers
    layer_0 = model.model.layers[0]

    assert get_layer_type(layer_0) == "linear_attention"
    torch_gdn = layer_0.linear_attn

    hidden_size = get_config_attr(config, "hidden_size")
    batch_size, seq_len = 1, 32

    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)

    # PyTorch reference
    with torch.no_grad():
        torch_out = torch_gdn(x)[0]

    # TTNN with distributed weights
    ttnn_gdn = TTNNQwen35GatedDeltaNet.from_torch(torch_gdn, distributed=True)
    set_device(ttnn_gdn, mesh_device)
    ttnn_gdn.preprocess_weights()
    ttnn_gdn.move_weights_to_device()

    # Verify weights are col-sharded
    assert isinstance(
        ttnn_gdn.in_proj_qkv, TTNNLinearIReplicatedWColSharded
    ), f"Expected TTNNLinearIReplicatedWColSharded, got {type(ttnn_gdn.in_proj_qkv)}"

    tt_input = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    ttnn_out = ttnn_gdn.forward(tt_input)

    # Output is col-sharded (no all-gather in out_proj) — concat on dim=-1
    ttnn_out_torch = _col_sharded_to_torch(ttnn_out, mesh_device)
    pcc_val = assert_with_pcc(torch_out, ttnn_out_torch, 0.90)
    print(f"  Distributed GDN PCC = {pcc_val:.6f}")


@skip_no_ttnn
@skip_no_transformers
@skip_no_symbiote
@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 200000000, "num_command_queues": 1, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [MESH_DEVICE_MAP.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))],
    indirect=True,
)
def test_decoder_layer_distributed_forward(mesh_device, model_4_layers):
    """Full decoder layer forward on mesh_device with distributed weights.

    Verifies output is valid (no NaN/Inf) and has correct shape.
    """
    model, config = model_4_layers

    hidden_size = get_config_attr(config, "hidden_size")
    num_kv_heads = get_config_attr(config, "num_key_value_heads")
    head_dim = get_config_attr(config, "head_dim")
    batch_size, seq_len = 1, 32

    # Test linear attention layer (layer 0)
    torch_layer = model.model.layers[0]
    ttnn_layer = TTNNQwen35DecoderLayer.from_torch(torch_layer)
    set_device(ttnn_layer, mesh_device)
    ttnn_layer.preprocess_weights()
    ttnn_layer.move_weights_to_device()

    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)
    position_ids = torch.arange(seq_len).unsqueeze(0)
    cos, sin = model.model.rotary_emb(x, position_ids)

    tt_input = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_cos = ttnn.from_torch(
        cos,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_sin = ttnn.from_torch(
        sin,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    output = ttnn_layer.forward(tt_input, position_embeddings=(tt_cos, tt_sin))

    # Output is col-sharded — concat on dim=-1 to reconstruct full hidden dim
    out_torch = _col_sharded_to_torch(output, mesh_device)

    assert not torch.isnan(out_torch).any(), "Output contains NaN"
    assert not torch.isinf(out_torch).any(), "Output contains Inf"
    assert out_torch.shape == (
        batch_size,
        seq_len,
        hidden_size,
    ), f"Expected shape {(batch_size, seq_len, hidden_size)}, got {out_torch.shape}"
    print(
        f"  Distributed decoder layer output: shape={out_torch.shape}, "
        f"mean={out_torch.float().mean():.6f}, std={out_torch.float().std():.6f}"
    )


@pytest.fixture(scope="module")
def model_4_layers():
    """Load Qwen3.5-27B-FP8 with 4 hidden layers (module-scoped for reuse)."""
    from .conftest import load_model

    model, config = load_model(num_hidden_layers=4)
    return model, config
