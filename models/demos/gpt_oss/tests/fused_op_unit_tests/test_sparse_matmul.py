import math
from dataclasses import dataclass

import torch
from tqdm import tqdm

import ttnn
from models.demos.gpt_oss.tests.test_factory import parametrize_mesh_with_fabric
from models.demos.gpt_oss.tt.experts_throughput.config import create_expert_mapping_tensors


@dataclass
class Config:
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG
    core_x: int = 5
    core_y: int = 9
    in0_block_w: int = 1
    out_subblock_h: int = 1
    out_subblock_w: int = 1
    per_core_M: int = 1
    output_tile: ttnn.Tile = None
    per_core_N: int = 1

    def __post_init__(self):
        if self.output_tile is None:
            self.output_tile = ttnn.Tile([32, 32])
        out_subblock_w = min(self.per_core_N, 8)

        n_tiles = math.ceil(2880 / ttnn.TILE_SIZE)
        # per_core_N = n_tiles // (self.core_x * self.core_y)

        self.program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(self.core_x, self.core_y),
            in0_block_w=self.in0_block_w,
            out_subblock_h=self.out_subblock_h,
            out_subblock_w=self.out_subblock_w,
            per_core_M=self.per_core_M,
            per_core_N=max(1, self.per_core_N),
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
        )


@parametrize_mesh_with_fabric()
# @pytest.mark.parametrize("in0_block_w", [1, 2, 3, 5, 6, 9, 10, 15, 18, 30, 45, 90])
def test_sparse_matmul(mesh_device):
    num_tokens = 32
    total_experts = 128
    experts_per_token = 4
    experts_per_device = 4
    num_devices = 32
    num_rows = 4
    hidden_dim = 2880
    sparse_block_size = 32

    expert_indices = torch.randint(total_experts, [1, 1, num_tokens, experts_per_token])
    dispatch_metadata_mock = torch.concat([expert_indices] * num_rows, dim=-2)
    dispatch_metadata_mock_tt = ttnn.from_torch(
        dispatch_metadata_mock, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint16
    )

    # remap_mask = ttnn.repeat(topk_mask, ttnn.Shape((1, 1, num_tokens, 1)))
    # remap_mask = ttnn.reshape(remap_mask, (1, 1, num_tokens*num_rows, total_experts))
    topk_mask = torch.ones(
        (1, 1, num_tokens * num_rows, total_experts),
        dtype=torch.bfloat16,
    )

    topk_mask_tt = ttnn.from_torch(topk_mask, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)

    expert_mapping_tensors = create_expert_mapping_tensors(
        num_devices=num_devices, num_experts_per_device=experts_per_device, mesh_device=mesh_device
    )
    _, sparsity = ttnn.moe_expert_token_remap(
        topk_mask_tt,
        expert_mapping_tensors,
        dispatch_metadata_mock_tt,
        reduction_size=sparse_block_size,
    )

    num_sparse_blocks = (num_tokens * num_rows) // sparse_block_size
    in_0_shape = (1, num_sparse_blocks, sparse_block_size, hidden_dim)  # tokens = [1, 16, 32, 2880]
    in_0 = torch.randn(in_0_shape, dtype=torch.bfloat16)
    in_0_tt = ttnn.from_torch(in_0, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    in_1_shape = (1, experts_per_device, hidden_dim, hidden_dim)  # weight = [1, 4, 2880, 2880]
    in_1 = torch.randn(in_1_shape, dtype=torch.bfloat16)
    in_1_tt = ttnn.from_torch(in_1, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat4_b)

    sparsity_shape = (1, 1, num_sparse_blocks, experts_per_token)  # sparsity = [1, 1, 16, 4]
    assert sparsity.shape == sparsity_shape

    # for in0_block_w in [1, 2, 3, 5, 6, 9, 10, 15, 18, 30, 45, 90]:
    #     logger.info(f"Testing sparse matmul with in0_block_w = {in0_block_w}")
    in0_block_w = 1
    # for per_core_N in [90, 45, 30, 18, 9, 6, 3, 1]:
    # for per_core_N in [2]:
    # logger.info(f"Testing sparse matmul with per_core_N = {per_core_N}")
    # config = Config(in0_block_w=in0_block_w, per_core_N=per_core_N)
    from models.demos.gpt_oss.tt.experts_throughput.config import ThroughputProgramConfig

    config = ThroughputProgramConfig()
    mock_bias = torch.randn((1, 1, 1, 4, 1, 2880), dtype=torch.bfloat16)
    mock_bias_tt = ttnn.from_torch(mock_bias, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat4_b)
    for i in tqdm(range(10000000)):
        # (1, 128, 32, 2880) x (1, 4, 2880, 2880)
        # sparse matmul of shapes (1, 16, 32, 2880) and (1, 4, 2880, 2880) M = 32, N = 2880, K = 2880
        output = ttnn.sparse_matmul(
            in_0_tt,
            in_1_tt,
            sparsity=sparsity,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=config.get_gate_up_config(n=2880, m=32),
            is_input_a_sparse=False,
            is_input_b_sparse=True,
            output_tile=ttnn.Tile([32, ttnn.TILE_SIZE]),
        )
        ttnn.synchronize_device(mesh_device)
        output = ttnn.add(output, mock_bias_tt, output_tensor=output)
        ttnn.synchronize_device(mesh_device)
