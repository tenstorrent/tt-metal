# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn

from models.demos.t3000.falcon40b.tt.model_utils import (
    matmul_2d_config_from_tensor_shapes,
    matmul_1d_config_from_tensor_shapes,
)
from models.utility_functions import nearest_32

MAX_SEQ_LEN = 4096
MAX_SEQ_LEN_LLAMA3 = 8192
MAX_SEQ_LEN_LLAMA3_1 = 128 * 1024


def num_to_corerange(x):
    assert x < 8 or x % 8 == 0
    num_x = min(x, 8)
    num_y = x // num_x
    assert num_x * num_y == x
    return ttnn.CoreRange(
        ttnn.CoreCoord(0, 0),
        ttnn.CoreCoord(num_x - 1, num_y - 1),
    )


def num_to_corerange_set(x):
    assert x < 8 or x % 8 == 0
    num_x = min(x, 8)
    num_y = x // num_x
    assert num_x * num_y == x
    return ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(num_x - 1, num_y - 1),
            ),
        }
    )


def get_model_config(llama_version="llama3-tg", max_batch_size=32, max_context_len=4096, cluster_shape=(4, 8)):
    assert max_batch_size in (1, 16, 32)

    if max_context_len == 8192:
        assert max_batch_size == 16
    elif max_context_len == 128 * 1024:
        assert max_batch_size == 1
    else:
        assert max_batch_size == 32

    model_config = {
        "MAX_GRID_SIZE": (8, 8),
        "CLUSTER_SHAPE": cluster_shape,
        "HIDDEN_SIZE": model_config_entries["hidden_size"],
        "MAX_BATCH_SIZE": max_batch_size,
        "MAX_CONTEXT_LEN": max_context_len,
        "llama3-tg": MAX_SEQ_LEN_LLAMA3,
        "llama3.1-tg": MAX_SEQ_LEN_LLAMA3_1,
        "NUM_DEVICES": 32,
        "PADDING_LENGTH": 32,
        "MAX_MM_SEQ_LEN": lambda seq_len: min(seq_len, 1024),  # Used to support seq len greater than 2k
        "CORE_GRID_Y": lambda seq_len: 4
        if min(seq_len, 1024) // 32 >= 4
        else min(seq_len, 1024) // 32,  # Core grid must be ratio of seq_len // 32
        "COMPUTE_KERNEL_CONFIG": ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
    }

    if llama_version == "llama3" or llama_version == "llama3-tg":
        model_config["FFN_EXPANDED_HIDDEN_SIZE"] = 28 * 1024
    elif llama_version == "llama3-405b":
        model_config["FFN_EXPANDED_HIDDEN_SIZE"] = 52 * 1024

    # Set attention config
    model_config["attention"] = set_attention_config(model_config, max_batch_size)
    # Set mlp config
    model_config["mlp"] = set_mlp_config(model_config, cluster_shape)
    # Set decoder config
    model_config["decoder"] = set_decoder_config(model_config)
    # Set core model config
    model_config["core_model"] = set_core_model_config(model_config, cluster_shape)
    return model_config


def get_batch_grid_size(batch_size):
    if batch_size == 1:
        return [1, 1]
    elif batch_size == 16:
        return [8, 2]
    elif batch_size == 32:
        return [8, 4]
    else:
        raise ValueError(f"Unsupported batch size: {batch_size}")


def set_attention_config(model_config, max_batch_size):
    # Set decode config first
    decode_config = {}

    decode_config["ROT_MAT_MM_PROGCFG"] = lambda batch_size: ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=get_batch_grid_size(batch_size),
        in0_block_w=4,  # 128 // TILE_SIZE (dynamic)
        out_subblock_h=1,
        out_subblock_w=4,
        per_core_M=1,
        per_core_N=4,
    )

    decode_config["FUSED_QKV_MM_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 5),
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=1,
        per_core_N=1,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )

    decode_config["COMPUTE_KERNEL_QKV"] = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    decode_config["COMPUTE_KERNEL_SELFOUT"] = decode_config["COMPUTE_KERNEL_QKV"]
    n_local_heads = model_config_entries["num_attention_heads"] // model_config_entries["num_kv_heads"]
    n_local_kv_heads = 1
    head_dim = model_config_entries["head_dim"]
    total_cores = (n_local_heads + n_local_kv_heads * 2) * head_dim // 32  # 1280 / 32 = 40
    assert total_cores == 40, f"total_cores: {total_cores}"
    shard_spec_n_cores_grid = ttnn.CoreRangeSet({num_to_corerange(total_cores)})

    decode_config["CREATE_HEAD_INPUT_MEMCFG"] = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            shard_spec_n_cores_grid,
            [
                32,
                32,
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )

    decode_config["COMPUTE_KERNEL_ROTARY"] = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    decode_config["ROTARY_PROGCFG"] = ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=[8, 1],
        in0_block_w=4,
        out_subblock_h=1,
        out_subblock_w=4,
        per_core_M=1,
        per_core_N=4,
    )

    decode_config["COMPUTE_KERNEL_SDPA"] = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    padded_local_heads = 32

    decode_config["SDPA_HEIGHT_SHARDED_MEMCFG"] = lambda batch_size_per_device_group: ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({num_to_corerange(batch_size_per_device_group)}),
            (padded_local_heads, head_dim),
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )

    decode_config["QKV_OUT_GATHERED_MEMCFG"] = lambda mesh_cols: ttnn.create_sharded_memory_config(
        shape=(32 * mesh_cols, 1280 // 40),  # mesh_cols = 4
        core_grid=ttnn.CoreGrid(y=5, x=8),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    decode_config["SELF_OUT_GATHERED_MEMCFG"] = lambda mesh_rows: ttnn.create_sharded_memory_config(
        shape=(32 * mesh_rows, 2048 // 32),  # mesh_rows = 8
        core_grid=ttnn.CoreGrid(y=4, x=8),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    decode_config["GATHER_USERS_MEMCFG"] = lambda mesh_cols: ttnn.create_sharded_memory_config(
        shape=(32 * mesh_cols, 1024 // 32),  # mesh_cols = 4
        core_grid=ttnn.CoreGrid(y=4, x=8),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    # Set prefill config
    prefill_config = {}

    prefill_config["COMPUTE_KERNEL_QKV"] = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    prefill_config["COMPUTE_KERNEL_SELFOUT"] = prefill_config["COMPUTE_KERNEL_QKV"]
    prefill_config["COMPUTE_KERNEL_ROTARY"] = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    prefill_config["COMPUTE_KERNEL_SDPA"] = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    prefill_config["SDPA_PROG_CFG"] = lambda seq_len: ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=[8, 7],
        q_chunk_size=256 if seq_len % 256 == 0 else 32,
        k_chunk_size=256 if seq_len % 256 == 0 else 32,
    )

    prefill_config["FUSED_QKV_MM_PROGCFG"] = lambda seq_len: matmul_2d_config_from_tensor_shapes(
        (1, 1, model_config["MAX_MM_SEQ_LEN"](seq_len), 2048),
        (1, 1, 2048, 1280),
        grid=ttnn.CoreGrid(x=8, y=model_config["CORE_GRID_Y"](seq_len)),
        overwrite_subblock_h=1,
        overwrite_subblock_w=1,
        fuse_batch=False,
    )

    prefill_config["SELFOUT_PROGCFG"] = lambda seq_len: matmul_2d_config_from_tensor_shapes(
        (1, 1, model_config["MAX_MM_SEQ_LEN"](seq_len), 1024),
        (1, 1, 1024, 2048),
        grid=ttnn.CoreGrid(x=8, y=model_config["CORE_GRID_Y"](seq_len)),
        overwrite_subblock_h=1,
        overwrite_subblock_w=1,
        fuse_batch=False,
    )

    return {"prefill": prefill_config, "decode": decode_config}


def set_mlp_config(model_config, cluster_shape):
    decode_config = {}

    decode_config["COMPUTE_KERNEL_LOFI"] = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    M, K, N = 32, model_config["HIDDEN_SIZE"], model_config["FFN_EXPANDED_HIDDEN_SIZE"]
    K = K // cluster_shape[0]
    N = N // cluster_shape[1]
    decode_config["W1_MEM_CONFIG"] = lambda mesh_device: ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(
            setup_weight_grid(mesh_device),
            (K, nearest_32(N // 12)),
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )

    decode_config["W2_MEM_CONFIG"] = lambda mesh_device: ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(
            setup_weight_grid(mesh_device),
            (N, nearest_32(K // 12)),
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )

    decode_config["FF1_DRAM_SHARDED_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=K // 8 // 32,  # K = 8192 / TILE_WIDTH=32 / Grid_Size is based on compute_with_storage_grid_size
        per_core_M=M // 32,  # M / TILE_HEIGHT = 32 / 32
        per_core_N=N // 8 // 32,  # N / TILE_WIDTH / Grid_Size is based on compute_with_storage_grid_size
        fused_activation=None,
    )

    decode_config["FF2_DRAM_SHARDED_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=N // 8 // 32,  # K = 8192 / TILE_WIDTH=32 / Grid_Size is based on compute_with_storage_grid_size
        per_core_M=M // 32,  # M / TILE_HEIGHT = 32 / 32
        per_core_N=K // 8 // 32,  # N / TILE_WIDTH / Grid_Size is based on compute_with_storage_grid_size
        fused_activation=None,
    )

    full_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(7, 7),
            )
        }
    )
    decode_config["FULL_GRID_MEMCFG"] = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            full_grid,
            [
                32,
                nearest_32(56),
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )
    decode_config["FF2_ACT_MEMCFG"] = ttnn.create_sharded_memory_config(
        shape=(M, N // 8),
        core_grid=ttnn.CoreGrid(y=1, x=8),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    decode_config["FF1_ACT_MEMCFG"] = ttnn.create_sharded_memory_config(
        shape=(32, 2048 // 8),
        core_grid=ttnn.CoreGrid(y=1, x=8),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    decode_config["FF1_OUT_GATHERED_MEMCFG"] = ttnn.create_sharded_memory_config(
        shape=(M * cluster_shape[0], N // 8),
        core_grid=ttnn.CoreGrid(y=1, x=8),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    decode_config["FF2_OUT_GATHERED_MEMCFG"] = ttnn.create_sharded_memory_config(
        shape=(32 * cluster_shape[1], 2048 // 8),
        core_grid=ttnn.CoreGrid(y=1, x=8),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    prefill_config = {}

    prefill_config["COMPUTE_KERNEL_LOFI"] = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    hidden_dim_per_chip = model_config_entries["hidden_size"] // cluster_shape[0]  # 2048
    ff_outer_dim_per_chip = model_config["FFN_EXPANDED_HIDDEN_SIZE"] // cluster_shape[1]  # 3584
    prefill_config["FF1_PROGCFG"] = lambda seq_len: matmul_2d_config_from_tensor_shapes(
        (
            1,
            1,
            model_config["MAX_MM_SEQ_LEN"](seq_len),
            hidden_dim_per_chip,
        ),  # (1, 1, model_config["MAX_MM_SEQ_LEN"], 2048)
        (1, 1, hidden_dim_per_chip, ff_outer_dim_per_chip),  # (1, 1, 2048, 3584)
        grid=ttnn.CoreGrid(x=8, y=model_config["CORE_GRID_Y"](seq_len)),
        overwrite_subblock_h=1,
        overwrite_subblock_w=1,
        fuse_batch=False,
    )
    prefill_config["FF2_PROGCFG"] = lambda seq_len: matmul_2d_config_from_tensor_shapes(
        (
            1,
            1,
            model_config["MAX_MM_SEQ_LEN"](seq_len),
            ff_outer_dim_per_chip,
        ),  # (1, 1, self.model_config["MAX_MM_SEQ_LEN"], 3584)
        (1, 1, ff_outer_dim_per_chip, hidden_dim_per_chip),  # (1, 1, 3584, 2048)
        grid=ttnn.CoreGrid(x=8, y=model_config["CORE_GRID_Y"](seq_len)),
        overwrite_subblock_h=1,
        overwrite_subblock_w=1,
        fuse_batch=False,
    )

    return {"prefill": prefill_config, "decode": decode_config}


def set_decoder_config(model_config):
    decode_config = {}

    decode_config["LN_COMPUTE_KERNEL_CONFIG"] = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    decode_config["LN_PROGCFG"] = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[8, 4],
        subblock_w=8,
        block_h=32 // 32,
        block_w=8,
        inplace=False,
    )

    shard_spec_32_cores_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(7, 3),
            ),
        }
    )

    decode_config["LN_OUTPUT_MEMCFG"] = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            shard_spec_32_cores_grid,
            [
                32,
                8192 // 32,
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )
    decode_config["ATTN_ACT_MEMCFG"] = ttnn.create_sharded_memory_config(
        shape=(32, 2048 // 32),
        core_grid=ttnn.CoreGrid(y=4, x=8),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    decode_config["MLP_ACT_MEMCFG"] = ttnn.create_sharded_memory_config(
        shape=(32, 2048 // 8),
        core_grid=ttnn.CoreGrid(y=1, x=8),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    prefill_config = {}

    prefill_config["LN_COMPUTE_KERNEL_CONFIG"] = decode_config["LN_COMPUTE_KERNEL_CONFIG"]

    return {"prefill": prefill_config, "decode": decode_config}


def set_core_model_config(model_config, cluster_shape):
    decode_config = {}
    decode_config["LN_COMPUTE_KERNEL_CONFIG"] = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    decode_config["COMPUTE_KERNEL_CONFIG"] = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    decode_config["LM_HEAD_ACT_MEMCFG"] = ttnn.create_sharded_memory_config(
        shape=(32, 2048 // 32),
        core_grid=ttnn.CoreGrid(y=4, x=8),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    prefill_config = {}
    prefill_config["LN_COMPUTE_KERNEL_CONFIG"] = decode_config["LN_COMPUTE_KERNEL_CONFIG"]
    prefill_config["COMPUTE_KERNEL_CONFIG"] = decode_config["COMPUTE_KERNEL_CONFIG"]
    prefill_config["LM_HEAD_ACT_MEMCFG"] = decode_config["LM_HEAD_ACT_MEMCFG"]

    hidden_size_per_chip = model_config_entries["hidden_size"] // cluster_shape[0]
    prefill_config["LM_HEAD_PROGCFG"] = matmul_1d_config_from_tensor_shapes(
        (
            1,
            1,
            model_config["PADDING_LENGTH"],
            hidden_size_per_chip,
        ),  # get only last padding_length (32) tokens
        (
            1,
            1,
            hidden_size_per_chip,
            model_config_entries["padded_vocab_size"] // cluster_shape[1],
        ),  # (1, 1, 2048, 16 * 1024)
        grid=ttnn.CoreGrid(x=8, y=4),
        overwrite_subblock_h=1,
        overwrite_subblock_w=1,
    )

    return {"prefill": prefill_config, "decode": decode_config}


def setup_weight_grid(mesh_device):
    weight_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(
                    mesh_device.dram_grid_size().x - 1,
                    mesh_device.dram_grid_size().y - 1,
                ),
            )
        }
    )
    return weight_grid


model_config_entries = {
    "hidden_size": 8192,
    "head_dim": 128,
    "num_attention_heads": 64,
    "num_kv_heads": 8,
    "num_layers": 80,
    "weight_cache": True,
    "vocab_size": 128256,
    "padded_vocab_size": 128 * 1024,
    "mlp_dim": 28672,
    "padded_mlp_dim": 32768,
    "layer_norm_epsilon": 1e-05,
}
