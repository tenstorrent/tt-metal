# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.demo.pipeline import create_fabric_router_config
from models.demos.deepseek_v3_b1.micro_ops.flash_mla.op import FlashMLADecode
from models.demos.deepseek_v3_b1.tests.unit_tests.test_pre_sdpa import deinterleave_kv_cache


class KVCacheMetadata:
    def __init__(
        self,
        noc_addr: int,
        fabric_node_ids: list[int],
        kv_chunk_size: int,
        num_tokens_in_chunk: int,
        is_dram: bool = True,
        dram_bank_id: int | None = None,
    ):
        self.is_dram = is_dram
        self.dram_bank_id = dram_bank_id
        self.noc_addr = noc_addr
        self.fabric_node_ids = fabric_node_ids
        self.kv_chunk_size = kv_chunk_size
        self.num_tokens_in_chunk = num_tokens_in_chunk

    def print(self):
        if self.is_dram:
            print(f"location: DRAM (bank_id={self.dram_bank_id})")
            print(f"noc_addr: 0x{self.noc_addr:X}")
        else:
            print(f"location: L1")
            print(f"noc_addr: 0x{self.noc_addr:X}")
        print(f"fabric_node_ids: {self.fabric_node_ids}")
        print(f"kv_chunk_size: {self.kv_chunk_size}")
        print(f"num_tokens_in_chunk: {self.num_tokens_in_chunk}")


def get_kv_cache_metadata(
    mesh_device: ttnn.MeshDevice,
    layer_id: int,
    position_id: int,
    slot_id: int,
    ttnn_kv_cache_tensor: ttnn.Tensor = None,
) -> dict:
    """
    Get KV cache metadata.
    """
    # kv_cache_tensor should come from the pipeline, but for testing pass in a tensor

    if ttnn_kv_cache_tensor is None:
        raise ValueError("ttnn_kv_cache_tensor is required")

    # Row 0: SP0.TP0, SP0.TP1
    fabric_id_sp0_tp0 = mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(0, 0))
    fabric_id_sp0_tp1 = mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(0, 1))
    # Row 1: SP1.TP0, SP1.TP1
    fabric_id_sp1_tp0 = mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(1, 0))
    fabric_id_sp1_tp1 = mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(1, 1))
    # Row 2: SP2.TP0, SP2.TP1
    fabric_id_sp2_tp0 = mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(2, 0))
    fabric_id_sp2_tp1 = mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(2, 1))
    # Row 3: SP3.TP0, SP3.TP1
    fabric_id_sp3_tp0 = mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(3, 0))
    fabric_id_sp3_tp1 = mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(3, 1))

    sp_idx_to_fabric_ids = {
        0: [fabric_id_sp0_tp0, fabric_id_sp0_tp1],
        1: [fabric_id_sp1_tp0, fabric_id_sp1_tp1],
        2: [fabric_id_sp2_tp0, fabric_id_sp2_tp1],
        3: [fabric_id_sp3_tp0, fabric_id_sp3_tp1],
    }

    tokens_per_kv_tile = ttnn_kv_cache_tensor.get_tile().tile_shape[0]
    k_tile_size = ttnn_kv_cache_tensor.get_tile().get_tile_size(ttnn_kv_cache_tensor.dtype)
    k_base_addr = ttnn_kv_cache_tensor.buffer_address()

    kv_cache_shape = ttnn_kv_cache_tensor.shape
    per_device_seq_len = kv_cache_shape[2]
    max_kv_cache_slots = kv_cache_shape[0]
    kv_cache_dim = kv_cache_shape[3]

    flash_mla_program_config = FlashMLADecode.ProgramConfig(k_chunk_size=128)
    flash_mla_optimal_grid = flash_mla_program_config.grid

    assert (
        per_device_seq_len == flash_mla_program_config.max_seq_len // flash_mla_program_config.sp_dim
    ), "KV cache sequence length must match max seq len"
    assert (
        max_kv_cache_slots == flash_mla_program_config.max_kv_cache_slots
    ), "KV cache slots must match max kv cache slots"

    # a bit confusing, the k_chunk_size is the number of tokens for a block of compute
    # for migration purposes, we call a kv_chunk the 32x576 unit to transfer

    # Calculate slot size
    slot_size_in_bytes = per_device_seq_len // tokens_per_kv_tile * k_tile_size

    # Calculate location of kv chunk that includes position_id
    assert layer_id == 0, "Only layer 0 is supported for now, use layer id to find correct devices"
    blocks_per_device = position_id // flash_mla_program_config.k_chunk_size

    block_id = position_id // flash_mla_program_config.k_chunk_size
    block_size_in_bytes = flash_mla_program_config.k_chunk_size * k_tile_size // tokens_per_kv_tile
    print(f"block_size_in_bytes: {block_size_in_bytes}")
    # round down to nearest kv_tile
    offset_in_block = (position_id % flash_mla_program_config.k_chunk_size) // tokens_per_kv_tile * tokens_per_kv_tile
    print(f"offset_in_block: {offset_in_block}")

    noc_addr = (
        ttnn_kv_cache_tensor.buffer_address()
        + (block_id % flash_mla_program_config.sp_dim) * block_size_in_bytes
        + offset_in_block * k_tile_size
        + slot_size_in_bytes * slot_id
    )

    # Get DRAM/SRAM noc address/bank id
    dram_bank_id = flash_mla_optimal_grid.OPTIMAL_DRAM_BANK_ORDER[block_id % flash_mla_optimal_grid.NUM_BLOCKS]

    # Get devices that hold the kv chunk
    print(f"block_id: {block_id}")
    sp_idx = (block_id // flash_mla_optimal_grid.NUM_BLOCKS) % flash_mla_program_config.sp_dim
    print(f"sp_idx: {sp_idx}")
    fabric_node_ids = sp_idx_to_fabric_ids[sp_idx]

    kv_cache_slot_size_device = per_device_seq_len * k_tile_size // tokens_per_kv_tile
    print(f"kv_cache_slot_size_device: {kv_cache_slot_size_device}")
    print(f"per_device_seq_len: {per_device_seq_len}")
    print(f"k_tile_size: {k_tile_size}")
    print(f"tokens_per_kv_tile: {tokens_per_kv_tile}")
    is_dram = ttnn_kv_cache_tensor.memory_config().buffer_type == ttnn.BufferType.DRAM
    if is_dram:
        return KVCacheMetadata(
            noc_addr=noc_addr,
            fabric_node_ids=fabric_node_ids,
            kv_chunk_size=k_tile_size,
            num_tokens_in_chunk=tokens_per_kv_tile,
            is_dram=True,
            dram_bank_id=dram_bank_id,
        )
    else:
        return KVCacheMetadata(
            noc_addr=noc_addr,
            fabric_node_ids=fabric_node_ids,
            kv_chunk_size=k_tile_size,
            num_tokens_in_chunk=tokens_per_kv_tile,
            is_dram=False,
        )


@pytest.mark.parametrize(
    "mesh_device",
    [(4, 8)],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(15232),
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "position_id",
    [11664],
)
def test_decode_kv_cache_metadata(mesh_device, device_params, position_id):
    torch.manual_seed(0)
    mesh_rows = 4
    mesh_cols = 2
    submesh = mesh_device.create_submesh(ttnn.MeshShape((mesh_rows, mesh_cols)))
    # model_pipeline = ModelPipeline(mesh_device, weights_mode="synthetic")
    flash_mla_program_config = FlashMLADecode.ProgramConfig(k_chunk_size=128)
    kvpe_dim = 576
    cache_shape = (flash_mla_program_config.max_kv_cache_slots, 1, flash_mla_program_config.max_seq_len, kvpe_dim)

    device_chunk_size = flash_mla_program_config.device_chunk_size
    num_sp = flash_mla_program_config.sp_dim

    torch_kv_cache = torch.zeros(cache_shape, dtype=torch.bfloat16)
    torch_kv_cache[:, :, :position_id, :] = torch.randn(1, 1, position_id, kvpe_dim, dtype=torch.bfloat16)
    print(f"torch_kv_cache: {torch_kv_cache.shape}")
    torch_kv_cache_shuffled = deinterleave_kv_cache(torch_kv_cache, device_chunk_size, num_sp)
    print(f"torch_kv_cache_shuffled: {torch_kv_cache_shuffled.shape}")

    grid = flash_mla_program_config.grid
    kv_nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=[1, 1, flash_mla_program_config.k_chunk_size, kvpe_dim],
        grid=grid.optimal_dram_grid(),
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    )
    kv_mem_config = ttnn.MemoryConfig(
        buffer_type=ttnn.BufferType.DRAM,
        nd_shard_spec=kv_nd_shard_spec,
    )
    num_chunks = flash_mla_program_config.max_seq_len // flash_mla_program_config.k_chunk_size
    num_banks = len(grid.OPTIMAL_DRAM_BANK_ORDER)
    logger.info(
        f"KV cache: ND sharded, DRAM banks: {num_banks} (optimal order: {grid.OPTIMAL_DRAM_BANK_ORDER}), chunks: {num_chunks}, shard_shape: [1, 1, {flash_mla_program_config.k_chunk_size}, {kvpe_dim}]"
    )

    ttnn_kv_cache = ttnn.from_torch(
        torch_kv_cache_shuffled,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=kv_mem_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(submesh, mesh_shape=(mesh_rows, mesh_cols), dims=(2, None)),
    )

    def get_local_seq_len(sp_idx):
        """Return how many KV positions SP device sp_idx holds for the current global position_id."""
        sp_block = device_chunk_size * num_sp
        num_full_blocks = position_id // sp_block
        remainder = position_id % sp_block
        dev_start = sp_idx * device_chunk_size
        dev_end = dev_start + device_chunk_size
        dev_contrib = max(0, min(remainder, dev_end) - dev_start)
        return num_full_blocks * device_chunk_size + dev_contrib

    print(f"local_seq_len: {get_local_seq_len(0)}")
    print(f"local_seq_len: {get_local_seq_len(1)}")
    print(f"local_seq_len: {get_local_seq_len(2)}")
    print(f"local_seq_len: {get_local_seq_len(3)}")
    layer_id = 0
    slot_id = 0
    kv_cache_metadata = get_kv_cache_metadata(submesh, layer_id, position_id, slot_id, ttnn_kv_cache)
    kv_cache_metadata.print()
