# Test RoPE operator

import os

import pytest
import torch
from transformers import AutoConfig

import ttnn
from models.demos.gpt_oss.reference.modeling_gpt_oss import GptOssRotaryEmbedding, apply_rotary_pos_emb
from models.demos.gpt_oss.tt.rope import ApplyRotaryPosEmb
from tests.ttnn.utils_for_testing import comp_pcc


@pytest.fixture
def hf_config():
    """Load GPT-OSS config for testing"""
    path = os.getenv("HF_MODEL", "models/demos/gpt_oss/reference")
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    return config


@pytest.mark.parametrize(
    "mesh_device",
    [
        (1, 2),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize(
    "seq_len",
    [1, 32, 64, 128, 512, 1024],
)
def test_rope_op(
    mesh_device,
    device_params,
    seq_len,
    hf_config,
):
    position_ids = torch.arange(seq_len).unsqueeze(0)

    RopeEmbeddings = GptOssRotaryEmbedding(hf_config)
    torch_inputs = torch.randn(1, seq_len, hf_config.hidden_size)
    cos, sin = RopeEmbeddings(torch_inputs, position_ids)
    q_torch = torch.randn(1, hf_config.num_attention_heads, seq_len, hf_config.head_dim)
    k_torch = torch.randn(1, hf_config.num_key_value_heads, seq_len, hf_config.head_dim)

    q_rope_torch, k_rope_torch = apply_rotary_pos_emb(q_torch, k_torch, cos, sin)

    q_tt = ttnn.from_torch(
        q_torch.permute(0, 2, 1, 3),
        device=mesh_device,
        # Shard along the num_attention_heads dimension
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, dims=(None, -2)),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    k_tt = ttnn.from_torch(
        k_torch.permute(0, 2, 1, 3),
        device=mesh_device,
        # Shard along the num_key_value_heads dimension
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, dims=(None, -2)),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    cos_tt = ttnn.from_torch(
        cos.unsqueeze(-2),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, dims=(None, None)),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )
    sin_tt = ttnn.from_torch(
        sin.unsqueeze(-2),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, dims=(None, None)),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    apply_rope = ApplyRotaryPosEmb(hf_config)

    q_tt_rotated = apply_rope(q_tt, cos_tt, sin_tt)
    k_tt_rotated = apply_rope(k_tt, cos_tt, sin_tt)

    q_tt_rotated_torch = ttnn.to_torch(
        q_tt_rotated, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_device.shape, dims=(0, -2))
    ).permute(0, 2, 1, 3)
    k_tt_rotated_torch = ttnn.to_torch(
        k_tt_rotated, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_device.shape, dims=(0, -2))
    ).permute(0, 2, 1, 3)

    passing, pcc_message = comp_pcc(q_tt_rotated_torch, q_rope_torch)
    mse = torch.nn.functional.mse_loss(q_tt_rotated_torch, q_rope_torch)
    print(f"Q: {pcc_message}, mse: {mse}")
    assert passing, f"q_tt_rotated_torch: {pcc_message}"
    passing, pcc_message = comp_pcc(k_tt_rotated_torch, k_rope_torch)
    mse = torch.nn.functional.mse_loss(k_tt_rotated_torch, k_rope_torch)
    print(f"K: {pcc_message}, mse: {mse}")
    assert passing, f"k_tt_rotated_torch: {pcc_message}"
