import ttnn
import torch
from models.utility_functions import (
    skip_for_wormhole_b0,
    comp_allclose_and_pcc,
    comp_pcc,
    comp_allclose,
)

# create test for layer_norm for sharded input tensor [32, 2048]  and weight and bias tensors [2048]
# sharded on 32 cores


def rms_norm(x, gamma, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * gamma


def test_post_allgather_layernorm(device, use_program_cache):
    torch.manual_seed(0)

    torch_input_tensor = torch.randn((1, 1, 32, 1024), dtype=torch.bfloat16)
    torch_weight = torch.randn((1, 1, 1, 1024), dtype=torch.bfloat16)
    eps = 1e-6
    torch_output_tensor = rms_norm(torch_input_tensor, torch_weight, eps=eps)
    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )
    # shard to 32 cores
    tt_sharded_config = ttnn.create_sharded_memory_config(
        shape=(1, 1, 32, 1024),
        core_grid=ttnn.CoreGrid(y=2, x=8),
        strategy=ttnn.ShardStrategy.WIDTH,
    )
    tt_input_tensor = ttnn.experimental.tensor.interleaved_to_sharded(
        tt_input_tensor, sharded_mem_config=tt_sharded_config
    )

    SHARDED_NORM_PRGM_CFG = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[8, 2],
        subblock_w=(1024 // 16) // 32,
        block_h=1,
        block_w=(1024 // 16) // 32,
        inplace=False,
    )

    tt_weights = ttnn.as_tensor(
        torch_weight,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        # cache_file_name="rms_weights_cache_1024",
    )

    tt_output_tensor = ttnn.rms_norm(
        tt_input_tensor,
        epsilon=eps,
        weight=tt_weights,
        program_config=SHARDED_NORM_PRGM_CFG,
        memory_config=tt_sharded_config,
    )
    tt_output_torch = ttnn.to_torch(tt_output_tensor)

    rtol = atol = 0.1
    pcc = 0.99
    passing, out = comp_allclose_and_pcc(torch_output_tensor, tt_output_torch, pcc=pcc, rtol=rtol, atol=atol)
    print(torch_output_tensor)
    print(tt_output_torch)
    print(out)
    assert passing
