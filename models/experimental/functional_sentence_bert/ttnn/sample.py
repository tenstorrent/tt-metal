import ttnn
import torch
import pytest
from tests.ttnn.utils_for_testing import assert_with_pcc

SDPAProgramConfig = ttnn._ttnn.operations.transformer.SDPAProgramConfig


def p(x, a="x"):
    print(f"{a}'s  shape: {x.shape}")
    print(f"{a}'s  layout: {x.layout}")
    print(f"{a}'s  dtype: {x.dtype}")
    print(f"{a}'s config: {x.memory_config()}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_emb(device):
    # torch
    emb_mod = torch.nn.Embedding(32000, 768, padding_idx=0)
    input_tens = torch.randint(low=0, high=31999, size=[2, 8], dtype=torch.int64)
    in_weights = torch.randn((32000, 768), dtype=torch.bfloat16)
    emb_mod.weight = torch.nn.Parameter(in_weights)
    torch_out = emb_mod(input_tens)
    print("torch out", torch_out.shape)
    # ttnn
    tt_input_tens = ttnn.from_torch(
        input_tens, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    in_weights = ttnn.from_torch(in_weights, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
    tt_out = ttnn.embedding(tt_input_tens, weight=in_weights)
    tt_out = ttnn.to_torch(tt_out)
    assert_with_pcc(tt_out, torch_out, 1.0)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
@pytest.mark.parametrize(
    "inputs",
    [
        [[2, 12, 32, 64], [2, 1, 32, 32]],  # BS-2, varying seq_length
        [[2, 12, 8, 64], [2, 1, 8, 8]],  # failing
        [[2, 12, 64, 64], [2, 1, 64, 64]],
        [[2, 12, 128, 64], [2, 1, 128, 128]],
    ],
)
def test_sdpa(device, inputs):
    query = torch.randn(inputs[0], dtype=torch.bfloat16)
    key = torch.randn((inputs[0]), dtype=torch.bfloat16)
    value = torch.randn((inputs[0]), dtype=torch.bfloat16)
    att_mask = torch.randn((inputs[1]), dtype=torch.bfloat16)
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=att_mask.shape[-1],
        k_chunk_size=att_mask.shape[-1],
        exp_approx_mode=True,
    )
    torch_attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=att_mask,
        dropout_p=0.0,
        is_causal=False,
    )
    tt_query = ttnn.from_torch(
        query, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_key = ttnn.from_torch(
        key, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_value = ttnn.from_torch(
        value, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_att_mask = ttnn.from_torch(
        att_mask, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    attn_output = ttnn.transformer.scaled_dot_product_attention(
        tt_query, tt_key, tt_value, attn_mask=tt_att_mask, is_causal=False, program_config=program_config
    )
    attn_output = ttnn.to_torch(attn_output)
    assert_with_pcc(torch_attn_output, attn_output, 0.999)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_sharded_add(device):
    torch_input_tensor_a = torch.rand((8, 384, 768), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((1, 384, 768), dtype=torch.bfloat16)
    shard_shape = (384, 96)

    torch_output_tensor = torch_input_tensor_a + torch_input_tensor_b
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    # input_tensor_a = ttnn.to_layout(input_tensor_a, ttnn.TILE_LAYOUT)
    block_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=input_tensor_a.shape,
        core_grid=ttnn.CoreGrid(y=8, x=8),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.COL_MAJOR,
        # use_height_and_width_as_shard_shape=True,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    # input_tensor_b = ttnn.to_layout(input_tensor_b, ttnn.TILE_LAYOUT)
    input_tensor_a = ttnn.to_memory_config(input_tensor_a, block_sharded_mem_config)
    p(input_tensor_a, "1st")
    input_tensor_b = ttnn.to_memory_config(input_tensor_b, block_sharded_mem_config)
    p(input_tensor_b, "2nd")
    ttnn_out = input_tensor_a + input_tensor_b
    p(ttnn_out, "result")
    ttnn_out = ttnn.to_torch(ttnn_out)
    assert_with_pcc(torch_output_tensor, ttnn_out, 1.0)
