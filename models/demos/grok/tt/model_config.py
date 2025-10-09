import math
import os

import torch
from loguru import logger
from safetensors.torch import load_file

import ttnn
from models.common.utility_functions import nearest_32
from models.tt_transformers.tt.model_config import DecodersPrecision


def num_to_coregrid(x):
    if x % 8 == 0:
        return ttnn.CoreGrid(y=x // 8, x=8)
    if x == 12:
        return ttnn.CoreGrid(y=2, x=6)
    if x == 20:
        return ttnn.CoreGrid(y=4, x=5)


def num_to_corerange(x):
    assert x < 8 or x % 8 == 0
    num_x = min(x, 8)
    num_y = x // num_x
    assert num_x * num_y == x
    return ttnn.CoreRange(
        ttnn.CoreCoord(0, 0),
        ttnn.CoreCoord(num_x - 1, num_y - 1),
    )


class TtModelArgs:
    def __init__(self, mesh_device):
        self.mesh_device = mesh_device
        self.num_devices = mesh_device.get_num_devices()
        self.galaxy_num_links = 4
        self.model_cache_path = "/localdev/ricozhu/tt-metal/model_cache_grok/"

        self.vocab_size = 128 * 1024

        # Model hyperparameters
        self.num_hidden_layers = 64

        # Embedding hyperparameters
        self.embedding_multiplier_scale = 90.50966799187809

        # Attention hyperparameters
        self.num_attention_heads = 64
        self.num_key_value_heads = 8
        self.head_dim = 128

        # MLP hyperparameters
        self.dim = 8192
        self.hidden_size = 8192
        self.intermediate_size = 32768
        self.expert_intermediate_size = 16384

        # MoE hyperparameters
        self.num_local_experts = 8
        self.num_experts_per_tok = 2

        self.num_devices_per_group = 8

        # LM head hyperparameters
        self.vocab_size = 128 * 1024
        self.padded_vocab_size = 128 * 1024

        # Add missing properties
        self.n_heads = self.num_attention_heads
        self.n_kv_heads = self.num_key_value_heads
        self.max_seq_len = 131072  # Grok-2 max sequence length
        self.max_batch_size = 32
        self.cluster_shape = (8, 4)  # TG mesh shape
        self.rope_theta = 208533496  # Grok-2 rope theta
        self.tile_size = 32
        self.ccl_dtype = ttnn.bfloat8_b
        self.num_reduce_scatter_links = 2
        self.num_all_gather_links = 2
        self.dummy_weights = True  # For testing
        self.hidden_dim = self.intermediate_size  # For MLP
        self.unpadded_hidden_dim = self.intermediate_size
        self.tile_padded_batch_rows = self.tile_size * int(math.ceil(self.max_batch_size / self.tile_size))

        self.n_local_heads = self.n_heads // self.cluster_shape[1]
        grid = self.mesh_device.compute_with_storage_grid_size()
        self.max_grid_size = ttnn.CoreGrid(x=grid.x, y=grid.y)

        self.compute_kernel_config_hifi2 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
            dst_full_sync_en=True,
        )

        self.model_config = self.get_model_config()
        self.rope_scaling = {
            "rope_type": "original",
            "original_max_position_embeddings": 8192,
            "scaling_factor": 16.0,
            "extrapolation_factor": 1.0,
            "attn_factor": 1.0,
            "beta_fast": 8,
            "beta_slow": 1,
        }

        lm_head_num_rows = 4
        while self.dim % (32 * 32 * lm_head_num_rows) != 0:
            lm_head_num_rows -= 1

        lm_head_cores_per_row = 8
        while self.dim % (32 * lm_head_num_rows * lm_head_cores_per_row) != 0:
            lm_head_num_rows -= 1
            if lm_head_num_rows == 0:
                lm_head_cores_per_row -= 1
                if lm_head_cores_per_row == 0:
                    raise ValueError(
                        f"Could not find a lm_head_num_rows such that self.dim(={self.dim}) % (lm_head_num_rows * 8) == 0"
                    )
                lm_head_num_rows = 8
        self.lm_head_core_grid = ttnn.CoreGrid(y=lm_head_num_rows, x=lm_head_cores_per_row)
        self.max_columns_per_device_lm_head = self.vocab_size // 8

    def prepare_residual_tensor_decode(self, x, input_mem_cfg, force_replicated=False, on_host=False):
        """
        Prepare inputs for decode mode.
        x: (batch, seq, dim)
        """
        dims = (None, None) if force_replicated else (None, -1)
        mesh_mapper = ttnn.ShardTensor2dMesh(self.mesh_device, dims=dims, mesh_shape=self.cluster_shape)

        if len(x.shape) == 3:
            batch = x.shape[0]
            seq_len = x.shape[1]
            assert x.shape[2] == self.dim
        elif len(x.shape) == 4:
            seq_len = x.shape[0]
            assert x.shape[1] == 1
            batch = x.shape[2]
            assert x.shape[3] == self.dim

        assert seq_len == 1, "Only supporting decode mode"

        # Support input on device
        if torch.is_tensor(x):  # Input on host -> Use torch
            x = x.transpose(0, 1).unsqueeze(1)  # [seq_len, 1, batch, dim]
            # Pad small batches to 32
            if batch < 32:
                zeros = torch.zeros(1, seq_len, 32, self.dim)
                zeros[:, :, :batch, :] = x
                x = zeros
        elif len(x.shape) == 3:  # Input on device -> Use ttnn
            x = ttnn.reshape(x, (batch, seq_len, 1, self.dim))  # [batch, seqlen, dim] -> [batch, seqlen, 1, dim]
            x = ttnn.permute(x, (1, 2, 0, 3))  # [seq_len, 1, batch, dim]
        elif len(x.shape) == 4:
            pass  # already in [seq_len, 1, batch, dim]

        if torch.is_tensor(x):
            x = ttnn.from_torch(
                x,
                device=self.mesh_device if not on_host else None,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=mesh_mapper,
                memory_config=input_mem_cfg if not on_host else None,
            )
        else:  # Convert the row major layout from embedding back to tile layout
            x = ttnn.to_layout(x, layout=ttnn.TILE_LAYOUT)
        return x

    def matmul_1d_config_from_tensor_shapes(
        self,
        in0_shape,
        in1_shape,
        grid=ttnn.CoreGrid(x=8, y=2),
        act=None,
        is_fp32_accumulate=False,
        overwrite_subblock_w=None,
        overwrite_subblock_h=None,
    ):
        m, k, n = in0_shape[0] * in0_shape[1] * in0_shape[2], in0_shape[3], in1_shape[3]
        return self.matmul_1d_config(
            m,
            k,
            n,
            grid,
            act,
            is_fp32_accumulate,
            overwrite_subblock_w=overwrite_subblock_w,
            overwrite_subblock_h=overwrite_subblock_h,
        )

    def matmul_1d_config(
        self,
        m,
        k,
        n,
        grid=ttnn.CoreGrid(x=8, y=2),
        act=None,
        is_fp32_accumulate=False,
        overwrite_per_core_k=None,
        overwrite_subblock_w=None,
        overwrite_subblock_h=None,
    ):
        tile_width = 32
        tile_height = 32

        if (
            n // tile_width // grid.num_cores < 1
        ):  # use less number of cores in case we have more N num tiles than cores
            grid_y = n // tile_width // grid.x
            grid = ttnn.CoreGrid(x=grid.x, y=grid_y)

        per_core_m = m // tile_height
        per_core_k = self.find_largest_divisor(k // (32 * grid.num_cores))
        per_core_n = math.ceil(n / tile_width / grid.num_cores)

        if is_fp32_accumulate:
            max_subblock_w_h = 4
        else:
            max_subblock_w_h = 8

        # find the largest value between 1 and 8 that is a factor of per_core_n
        out_subblock_w = max([i for i in range(1, max_subblock_w_h + 1) if per_core_n % i == 0])

        # find the largest value that is a factor of per_core_m such that
        # out_subblock_w * out_subblock_h <= 8
        out_subblock_h = max(
            [
                i
                for i in range(1, max_subblock_w_h + 1)
                if per_core_m % i == 0 and i * out_subblock_w <= max_subblock_w_h
            ]
        )

        if overwrite_per_core_k is not None:
            per_core_k = overwrite_per_core_k

        if overwrite_subblock_w is not None:
            out_subblock_w = overwrite_subblock_w

        if overwrite_subblock_h is not None:
            out_subblock_h = overwrite_subblock_h

        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(grid.x, grid.y),
            in0_block_w=per_core_k,
            out_subblock_h=out_subblock_h,
            out_subblock_w=out_subblock_w,
            per_core_M=per_core_m,
            per_core_N=per_core_n,
            fuse_batch=True,
            fused_activation=act,
            mcast_in0=True,
        )

    def find_largest_divisor(self, n, max_divisor=8):
        for i in range(max_divisor, 0, -1):
            if n % i == 0:
                return i
        return 1  # Fallback to 1 if no divisor found

    def create_sharded_norm_config(self, grid):
        """Helper function to create LayerNormShardedMultiCoreProgramConfig for RMS NORM.

        Args:
            grid (ttnn.CoreGrid): Grid specification for the norm operation
        """
        block_w = self.dim // grid.num_cores // self.tile_size
        # Find largest value <= 4 that evenly divides block_w
        subblock_w = 4
        while subblock_w > 0:
            if block_w % subblock_w == 0:
                break
            subblock_w -= 1
        return ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[grid.x, grid.y],
            subblock_w=subblock_w,
            block_h=self.tile_padded_batch_rows // self.tile_size,
            block_w=block_w,
            inplace=False,
        )

    def create_dram_sharded_mem_config(self, k, n):
        """Create DRAM-sharded memory config for width-sharded tensors"""
        dram_cores = self.mesh_device.dram_grid_size().x  # WH has 12 dram cores, P150 has 8, P100 has 7
        assert self.mesh_device.dram_grid_size().y == 1, "Current dram sharding assumes y dim is 1"
        padded_size = math.ceil(n / (self.tile_size * dram_cores)) * (self.tile_size * dram_cores)
        self.dram_weight_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(self.mesh_device.dram_grid_size().x - 1, self.mesh_device.dram_grid_size().y - 1),
                )
            }
        )
        shard_spec = ttnn.ShardSpec(
            self.dram_weight_grid, (k, padded_size // dram_cores), ttnn.ShardOrientation.ROW_MAJOR
        )
        return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)

    def dram_matmul_config(self, m: int, k: int, n: int, num_cores=None, fused_activation=None):
        # in0_block_w must evenly divide k and be no larger than tile_size * num_cores
        return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=self.find_largest_divisor(k // (self.tile_size * num_cores)),
            per_core_M=math.ceil(m / self.tile_size),
            per_core_N=math.ceil(n / (self.tile_size * num_cores)),
            fused_activation=fused_activation,
        )

    def get_model_config(self):
        """Create and return the model configuration dictionary with all required configs"""
        model_config = {}

        # Basic memory and layout configs
        model_config["EMB_WEIGHTS_MEMCFG"] = ttnn.DRAM_MEMORY_CONFIG
        model_config["DECODE_RESIDUAL_MEMCFG"] = ttnn.L1_MEMORY_CONFIG
        model_config["ATTN_W_LAYOUT_TILE"] = ttnn.TILE_LAYOUT

        model_config["MOE_INPUT_MEMCFG"] = ttnn.DRAM_MEMORY_CONFIG
        model_config["MLP_ACT_MEMCFG"] = ttnn.create_sharded_memory_config(
            shape=(32, self.dim // 4 // 16),  # dim / num devices / 16 cores
            core_grid=ttnn.CoreGrid(x=8, y=2),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        # Decoder optimizations - simplified for TG=True, dim=8192
        model_config["DECODERS_OPTIMIZATIONS"] = DecodersPrecision.accuracy(
            num_decoders=self.num_hidden_layers, model_name="grok-2"
        )

        # TG-specific program configs for MLP
        model_config["FF1_3_TG_PROGCFG"] = self.matmul_1d_config_from_tensor_shapes(
            (1, 1, 32, self.hidden_size // 4),
            (1, 1, self.hidden_size // 4, self.intermediate_size // 8),
            grid=ttnn.CoreGrid(x=8, y=2),
            overwrite_subblock_h=1,
            overwrite_subblock_w=1,
        )

        model_config["FF1_3_TG_PROGCFG_SINGLE_EXPERT"] = self.matmul_1d_config_from_tensor_shapes(
            (1, 8, 32, self.hidden_size // 4),
            (1, 8, self.hidden_size // 4, self.expert_intermediate_size // 8),
            grid=ttnn.CoreGrid(x=8, y=8),
            overwrite_subblock_h=1,
            overwrite_subblock_w=1,
        )

        model_config["FF2_TG_PROGCFG"] = self.matmul_1d_config_from_tensor_shapes(
            (1, 1, 32, self.intermediate_size // 8),
            (1, 1, self.intermediate_size // 8, self.hidden_size // 4),
            grid=ttnn.CoreGrid(x=8, y=2),
            overwrite_subblock_h=1,
            overwrite_subblock_w=1,
        )

        model_config["FF2_TG_PROGCFG_SINGLE_EXPERT"] = self.matmul_1d_config_from_tensor_shapes(
            (1, 8, 32, self.expert_intermediate_size // 8),
            (1, 8, self.expert_intermediate_size // 8, self.hidden_size // 4),
            grid=ttnn.CoreGrid(x=8, y=8),
            overwrite_subblock_h=1,
            overwrite_subblock_w=1,
        )

        # Memory configs for MLP operations
        model_config["FF1_OUT_REDUCE_SCATTER_MEMCFG"] = ttnn.create_sharded_memory_config(
            shape=(32, self.intermediate_size // 64 // 8),  # shard_grid_cores = 28, num_devices=8
            core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))}),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        # [1, 8, 32, 16384 // 8] -> [256, 2048]

        model_config["FF1_OUT_REDUCE_SCATTER_MEMCFG_SINGLE_EXPERT"] = ttnn.create_sharded_memory_config(
            shape=(256, self.expert_intermediate_size // 64 // 8),  # shard_grid_cores = 64, num_devices=8
            core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))}),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        model_config["FF2_OUT_REDUCE_SCATTER_MEMCFG"] = ttnn.create_sharded_memory_config(
            shape=(32, self.hidden_size // 8 // 4),  # shard_grid_cores = 8, num_devices=4
            core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))}),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        model_config["FF2_OUT_REDUCE_SCATTER_MEMCFG_SINGLE_EXPERT"] = ttnn.create_sharded_memory_config(
            shape=(256, self.hidden_size // 8 // 4),  # shard_grid_cores = 8, num_devices=4
            core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))}),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        # Attention input memory config for TG
        lm_head_num_rows = 4  # For TG with dim=8192
        model_config["SHARDED_ATTN_INPUT_MEMCFG"] = ttnn.create_sharded_memory_config(
            shape=(32, nearest_32(self.hidden_size // (8 * lm_head_num_rows) // 4)),
            core_grid=ttnn.CoreGrid(y=lm_head_num_rows, x=8),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        model_config["SHARDED_NORM_ATTN_PRGM_CFG"] = self.create_sharded_norm_config(
            ttnn.CoreGrid(y=lm_head_num_rows, x=8)
        )
        model_config["GATHER_IN_MEMCFG"] = ttnn.create_sharded_memory_config(
            shape=(1, 1, 32, 8192 // 4),
            core_grid=ttnn.CoreGrid(y=4, x=8),
            strategy=ttnn.ShardStrategy.WIDTH,
        )
        model_config["SHARDED_ATTN_INPUT_MEMCFG_SINGLE_EXPERT"] = ttnn.create_sharded_memory_config(
            shape=(256, nearest_32(self.hidden_size // (8 * lm_head_num_rows) // 4)),
            core_grid=ttnn.CoreGrid(y=lm_head_num_rows, x=8),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        # Attention-specific configs for TG
        model_config["XQKV_DECODE_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 5),
            in0_block_w=1,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=1,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )

        # TG attention memory configs
        num_cores = 40  # For dim=8192 on TG
        model_config["QKV_OUT_GATHERED_MEMCFG"] = lambda mesh_cols: ttnn.create_sharded_memory_config(
            shape=(32 * mesh_cols, 32),  # mesh_cols = 4
            core_grid=num_to_coregrid(num_cores),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        model_config["CREATE_HEAD_INPUT_MEMCFG"] = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet({num_to_corerange(40)}),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

        model_config["CREATE_QKV_DECODE_SHARD"] = ttnn.create_sharded_memory_config(
            shape=(32, self.head_dim),
            core_grid=ttnn.CoreGrid(y=4, x=8),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        model_config["SDPA_DECODE_PROGCFG"] = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            exp_approx_mode=False,
            q_chunk_size=256,
            k_chunk_size=256,
        )

        model_config[
            "SCORES_BATCHED_MM_OUTPUT_MEMCFG"
        ] = lambda batch_size_per_device_group: ttnn.create_sharded_memory_config(
            shape=(math.ceil((self.num_attention_heads // self.num_devices_per_group) / 32) * 32, self.head_dim),
            core_grid=ttnn.CoreRangeSet({num_to_corerange(batch_size_per_device_group)}),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        model_config["GATHER_USERS_MEMCFG"] = lambda mesh_cols: ttnn.create_sharded_memory_config(
            shape=(32 * mesh_cols, 32),  # mesh_cols = 4
            core_grid=num_to_coregrid(min(32, self.hidden_size // 8 // 32)),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        model_config["SELF_OUT_REDUCE_SCATTER_MEMCFG"] = ttnn.create_sharded_memory_config(
            shape=(32, 2048 // 8 // 8),  # mesh_rows = 8, num_cores=8
            core_grid=ttnn.CoreGrid(y=1, x=8),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        # Expert MLP configurations (adapted from Mixtral)
        model_config["MLP_W_LAYOUT_TILE_EXPERTS"] = ttnn.TILE_LAYOUT
        model_config["MLP_WEIGHTS_MEMCFG_EXPERTS"] = ttnn.DRAM_MEMORY_CONFIG
        model_config["FF1_OUTPUT_MEMCFG_EXPERTS"] = ttnn.L1_MEMORY_CONFIG
        model_config["FF2_OUTPUT_MEMCFG_EXPERTS"] = ttnn.L1_MEMORY_CONFIG
        model_config["FF3_OUTPUT_MEMCFG_EXPERTS"] = ttnn.L1_MEMORY_CONFIG

        # Expert MLP program configs (adapted from Mixtral for Grok dimensions)
        model_config["FF1_OUTPUT_PROGCFG_EXPERTS"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=4,  # K = 8192 / TILE_WIDTH=32 / Grid_Size
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=8,  # Adjusted for Grok's intermediate_size
            fuse_batch=True,
            fused_activation=ttnn.UnaryOpType.GELU,
            mcast_in0=True,
        )

        model_config["FF3_OUTPUT_PROGCFG_EXPERTS"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=4,  # K = 8192 / TILE_WIDTH=32 / Grid_Size
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=8,  # Adjusted for Grok's intermediate_size
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )

        model_config["FF2_OUTPUT_PROGCFG_EXPERTS"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=8,  # K = 32768 / TILE_WIDTH=32 / Grid_Size
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=4,  # Adjusted for Grok's hidden_size
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )

        # MoE Gate configurations (adapted from Mixtral)
        model_config["GATE_W_LAYOUT_TILE_EXPERTS"] = ttnn.TILE_LAYOUT
        model_config["GATE_WEIGHTS_MEMCFG_EXPERTS"] = ttnn.DRAM_MEMORY_CONFIG
        model_config["GATE_MM_OUTPUT_MEMCFG_EXPERTS"] = ttnn.L1_MEMORY_CONFIG

        model_config["GATE_MM_OUTPUT_KERNEL_CONFIG_EXPERTS"] = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        model_config["NORM_COMPUTE_KERNEL_CONFIG"] = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        return model_config

    def weight_cache_path(self, dtype):
        dtype_string = {
            ttnn.bfloat16: "tensor_cache_bf16",
            ttnn.bfloat8_b: "tensor_cache_bfp8",
            ttnn.bfloat4_b: "tensor_cache_bfp4",
        }[dtype]
        return f"{self.model_cache_path}/{dtype_string}"

    def load_weights_to_state_dict_no_experts(self, weights_path="/localdev/ricozhu/grok_2_weights/", fuse_qkv=True):
        """
        Weights are stored as follows:  (all bfloat16)                                                dims:
          - 00000-TP-common: embedding                                                                      [128*1024, 8192]
          - 00001-TP-common: lm head                                                                        [128*1024, 8192]
          - 00002-TP-common: model.norm                                                                     [8192]
          - 00003-TP-common: shared_mlp.gate_proj, 64 layers                                                [32768, 8192]
          - 00004-TP-common: shared_mlp.down_proj, 64 layers                                                [8192, 32768]
          - 00005-TP-common: shared_mlp.up_proj, 64 layers                                                  [32768, 8192]

          - 00006-TP-00[0-7]: block_sparse_moe.experts.[0-7].w1, shard [0-7], 64 layers                     [2048 * 8, 8192]
          - 00007-TP-00[0-7]: block_sparse_moe.experts.[0-7].w2, shard [0-7], 64 layers                     [8192, 2048 * 8]
          - 00008-TP-00[0-7]: block_sparse_moe.experts.[0-7].w3, shard [0-7], 64 layers                     [2048 * 8, 8192]

          - 00009-TP-common: self_attn.k_proj, 64 layers                                                    [8192, 1024]
          - 00010-TP-common: self_attn.o_proj, 64 layers                                                    [8192, 8192]
          - 00011-TP-common: self_attn.q_proj, 64 layers                                                    [8192, 8192]
          - 00012-TP-common: self_attn.v_proj, 64 layers                                                    [8192, 1024]

          - 00013-TP-common: pre_attn_norm, 64 layers                                                       [8192]
          - 00014-TP-common: post_attn_norm, 64 layers                                                      [8192]
          - 00015-TP-common: pre_moe_norm, 64 layers                                                        [8192]
          - 00016-TP-common: post_moe_norm, 64 layers                                                       [8192]

          - 00017-TP-common: block_sparse_moe.gate, 64 layers                                               [8, 8192]
        """

        state_dict = {}

        # --- embeddings / head / final norm ---
        state_dict["tok_embeddings.weight"] = load_file(
            os.path.join(weights_path, "pytorch_model-00000-TP-common.safetensors")
        )["model.embed_tokens.weight"]
        state_dict["model.lm_head.weight"] = load_file(
            os.path.join(weights_path, "pytorch_model-00001-TP-common.safetensors")
        )["lm_head.weight"]
        state_dict["model.norm.weight"] = load_file(
            os.path.join(weights_path, "pytorch_model-00002-TP-common.safetensors")
        )["model.norm.weight"]

        # --- shared MLP projections (64 layers) ---
        for proj, fname, pname in [
            ("w1", "pytorch_model-00003-TP-common.safetensors", "gate_proj"),
            ("w2", "pytorch_model-00004-TP-common.safetensors", "down_proj"),
            ("w3", "pytorch_model-00005-TP-common.safetensors", "up_proj"),
        ]:
            arr = load_file(os.path.join(weights_path, fname))
            for layer_idx in range(64):
                grok_weight_key = f"model.layers.{layer_idx}.mlp.{pname}.weight"
                weight_key = f"model.layers.{layer_idx}.mlp.{proj}.weight"
                state_dict[weight_key] = arr[grok_weight_key]

        # --- self-attention projections ---
        for proj, fname, pname in [
            ("k_proj", "pytorch_model-00009-TP-common.safetensors", "self_attn.k_proj"),
            ("o_proj", "pytorch_model-00010-TP-common.safetensors", "self_attn.o_proj"),
            ("q_proj", "pytorch_model-00011-TP-common.safetensors", "self_attn.q_proj"),
            ("v_proj", "pytorch_model-00012-TP-common.safetensors", "self_attn.v_proj"),
        ]:
            arr = load_file(os.path.join(weights_path, fname))
            for layer_idx in range(64):
                weight_key = f"model.layers.{layer_idx}.self_attn.{proj}.weight"
                state_dict[weight_key] = arr[weight_key]

        for layer_idx in range(64):
            q_proj = state_dict[f"model.layers.{layer_idx}.self_attn.q_proj.weight"]
            # q_proj = reverse_permute(q_proj, 64, q_proj.shape[0], q_proj.shape[1])
            k_proj = state_dict[f"model.layers.{layer_idx}.self_attn.k_proj.weight"]
            # k_proj = reverse_permute(k_proj, 8, k_proj.shape[0], k_proj.shape[1])
            v_proj = state_dict[f"model.layers.{layer_idx}.self_attn.v_proj.weight"]

            if fuse_qkv:
                qkv_list = []
                for i in range(self.num_devices_per_group):
                    # Chunk weights
                    wq_selected = torch.chunk(q_proj, self.num_devices_per_group, dim=0)[i]
                    wk_selected = torch.chunk(k_proj, self.num_devices_per_group, dim=0)[i]
                    wv_selected = torch.chunk(v_proj, self.num_devices_per_group, dim=0)[i]

                    # Transpose the selected chunks
                    wq = torch.transpose(wq_selected, -2, -1)
                    wk = torch.transpose(wk_selected, -2, -1)
                    wv = torch.transpose(wv_selected, -2, -1)

                    qkv = torch.cat([wq, wk, wv], dim=-1)
                    qkv_list.append(qkv)

                qkv_cat = torch.cat(qkv_list, dim=-1)
                state_dict[f"model.layers.{layer_idx}.self_attn.wqkv.weight"] = qkv_cat

            del state_dict[f"model.layers.{layer_idx}.self_attn.q_proj.weight"]
            del state_dict[f"model.layers.{layer_idx}.self_attn.k_proj.weight"]
            del state_dict[f"model.layers.{layer_idx}.self_attn.v_proj.weight"]

            state_dict[f"model.layers.{layer_idx}.self_attn.wo.weight"] = state_dict[
                f"model.layers.{layer_idx}.self_attn.o_proj.weight"
            ]
            del state_dict[f"model.layers.{layer_idx}.self_attn.o_proj.weight"]

        # --- norms ---
        for norm_key, fname, pname in [
            ("pre_attn_norm", "pytorch_model-00013-TP-common.safetensors", "pre_attn_norm"),
            ("post_attn_norm", "pytorch_model-00014-TP-common.safetensors", "post_attn_norm"),
            ("pre_moe_norm", "pytorch_model-00015-TP-common.safetensors", "pre_moe_layernorm"),
            ("post_moe_norm", "pytorch_model-00016-TP-common.safetensors", "post_moe_layernorm"),
        ]:
            arr = load_file(os.path.join(weights_path, fname))
            for layer_idx in range(64):
                weight_key = f"model.layers.{layer_idx}.{norm_key}.weight"
                state_dict[weight_key] = arr[weight_key]

        # --- gating for MoE ---
        gate = load_file(os.path.join(weights_path, "pytorch_model-00017-TP-common.safetensors"))
        for layer_idx in range(64):
            weight_key = f"model.layers.{layer_idx}.block_sparse_moe.gate.weight"
            state_dict[weight_key] = gate[weight_key]

        return state_dict

    def load_experts_weights_to_state_dict(self, state_dict, weights_path="/localdev/ricozhu/grok_2_weights/"):
        # if os.path.exists(f"{weights_path}/experts"):
        #     for file in os.listdir(f"{weights_path}/experts"):
        #         logger.info(f"Loading experts weights to state dict for {file}...")
        #         state_dict[file.replace(".pt", "")] = torch.load(f"{weights_path}/experts/{file}")
        #     return state_dict

        # os.makedirs(f"{weights_path}/experts", exist_ok=True)

        # --- experts (3 matrices, 8 shards each) ---
        for proj, base, pname in [
            ("w1", "pytorch_model-00006-TP-00", "block_sparse_moe.experts"),
            ("w2", "pytorch_model-00007-TP-00", "block_sparse_moe.experts"),
            ("w3", "pytorch_model-00008-TP-00", "block_sparse_moe.experts"),
        ]:
            logger.info(f"Loading experts weights to state dict for block_sparse_moe.experts.{proj}...")

            # List of shards for each of the 8 experts
            shard_list = [[[] for _ in range(64)] for _ in range(8)]

            for tp_shard_idx in range(8):
                f = f"{base}{tp_shard_idx}.safetensors"
                arr = load_file(os.path.join(weights_path, f))
                for expert_idx in range(8):
                    for layer_idx in range(64):
                        weight_key = f"model.layers.{layer_idx}.{pname}.{expert_idx}.{proj}.weight"
                        shard_list[expert_idx][layer_idx].append(arr[weight_key])
                logger.info(
                    f"Finished loading experts weights to state dict for block_sparse_moe.experts.{proj} for tp_shard_idx {tp_shard_idx}..."
                )

            full = [[[] for _ in range(64)] for _ in range(8)]
            for expert_idx in range(8):
                for layer_idx in range(64):
                    full[expert_idx][layer_idx] = torch.cat(
                        shard_list[expert_idx][layer_idx], dim=0 if proj in ["w1", "w3"] else 1
                    )

            for expert_idx in range(8):
                for layer_idx, chunk in enumerate(full[expert_idx]):
                    state_dict[f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.{proj}.weight"] = chunk
                    # torch.save(chunk, f"{weights_path}/experts/model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.{proj}.weight.pt")

            logger.info(f"Finished loading experts weights to state dict for block_sparse_moe.experts.{proj}...")

        logger.info(f"Loaded experts weights to state dict...")
        return state_dict

    def prune_experts_except_layers(self, state_dict, layers):
        for layer_idx in range(self.num_hidden_layers):
            if layer_idx not in layers:
                for expert_idx in range(8):
                    del state_dict[f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w1.weight"]
                    del state_dict[f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w2.weight"]
                    del state_dict[f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w3.weight"]

    def get_state_dict_prefix(self, module_name, layer_num):
        layer_prefix = f"model.layers.{layer_num}" if layer_num is not None else ""
        module_map = {
            "MLP": ".mlp",
            "ExpertMLP": ".block_sparse_moe",
            "Attention": ".self_attn",
            "TtTransformerBlock": "",
            "": "",  # If no module is given, just get layer prefix
        }
        return layer_prefix + module_map.get(module_name, "")

    def ccl_topology(self):
        """Return the CCL topology for the mesh configuration"""
        return ttnn.Topology.Ring


def reverse_permute(tensor, n_heads, dim1, dim2):
    return tensor.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)


def permute(tensor, n_heads, dim1, dim2):
    return tensor.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)
