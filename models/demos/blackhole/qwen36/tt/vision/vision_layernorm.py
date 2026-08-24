# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0
import ttnn
from models.common.lightweightmodule import LightweightModule

TILE = 32
SHARD_HEIGHT = TILE


class LayerNorm(LightweightModule):
    def __init__(
        self,
        device,
        dim,
        state_dict,
        state_dict_prefix,
        weight_cache_path=None,
        weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        weight_dtype=ttnn.bfloat8_b,
        model_config=None,
        eps: float = 1e-05,
        sharded_fp32_acc: bool = False,
    ):
        super().__init__()
        self.device = device
        self.eps = eps
        # fp32 dest accumulation for the SHARDED forward path. Defaults OFF so every config keeps
        # the previously shipped behavior; the caller opts in (vision_block gates it on
        # tp_common.wh_9b_n300_vision). See the sharded branch in forward() for why it matters.
        self.sharded_fp32_acc = sharded_fp32_acc

        torch_weight = (
            state_dict[f"{state_dict_prefix}.weight"].unsqueeze(0).view(1, 1, dim).expand([1, SHARD_HEIGHT, dim])
        )
        torch_bias = state_dict[f"{state_dict_prefix}.bias"].unsqueeze(0).view(1, 1, dim).expand([1, SHARD_HEIGHT, dim])
        if weight_cache_path is None:
            cache_name = lambda *_: None
        else:
            cache_name = lambda suffix: weight_cache_path / (state_dict_prefix + f"{suffix}")

        is_mesh_device = device.__class__.__name__ == "MeshDevice"
        self.weight = ttnn.as_tensor(
            torch_weight,
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_memory_config,
            cache_file_name=cache_name("weight"),
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
        )

        self.bias = ttnn.as_tensor(
            torch_bias,
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_memory_config,
            cache_file_name=cache_name("bias"),
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
        )

        if model_config:
            self.sharded_input_config = model_config["SHARDED_NORM_INPUT_MEMCFG"]
            self.sharded_program_config = model_config["SHARDED_NORM_PRGM_CFG"]
            self.sharded_output_config = model_config["SHARDED_NORM_OUTPUT_MEMCFG"]
        else:
            assert (
                dim % SHARD_HEIGHT == 0
            ), f"Input dimension dim ({dim}) must be a multiple of SHARD_HEIGHT ({SHARD_HEIGHT})"
            shard_width_hidden_dim_across_32_cores = dim // SHARD_HEIGHT
            core_grid = ttnn.CoreGrid(x=8, y=SHARD_HEIGHT // 8)
            # core_grid = ttnn.CoreGrid(x=8, y=8)
            self.sharded_input_config = ttnn.create_sharded_memory_config(
                shape=(SHARD_HEIGHT, shard_width_hidden_dim_across_32_cores),
                core_grid=core_grid,
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            self.sharded_program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=[core_grid.x, core_grid.y],
                subblock_w=shard_width_hidden_dim_across_32_cores // TILE,
                block_h=SHARD_HEIGHT // TILE,
                block_w=shard_width_hidden_dim_across_32_cores // TILE,
                inplace=False,
            )
            self.sharded_output_config = self.sharded_input_config

    def forward(self, x: ttnn.Tensor, in_sharded=False, out_sharded=False, memory_config=None) -> ttnn.Tensor:
        if in_sharded:
            x = ttnn.layer_norm(
                x,
                epsilon=self.eps,
                weight=self.weight,
                bias=self.bias,
                program_config=self.sharded_program_config,
                memory_config=self.sharded_output_config,
                compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.HiFi4,
                    math_approx_mode=False,
                    # SCOPED via self.sharded_fp32_acc, which vision_block gates on
                    # tp_common.wh_9b_n300_vision (Wormhole 9B on N300 only). Off everywhere else,
                    # which preserves the previously shipped behavior on Blackhole / the 27B /
                    # N150 / T3K.
                    #
                    # Same reason as the interleaved branch below: the tower's outlier activations
                    # (absmax 354 vs rms 0.65) swamp a bf16 running sum over 1152 channels. This
                    # branch is currently unused in qwen36 (nothing passes in_sharded=True) and so
                    # had drifted without the flag; wiring it keeps the two paths numerically
                    # equivalent on the gated config, so enabling sharding later cannot silently
                    # give back the +0.005 PCC the interleaved branch calls non-optional.
                    fp32_dest_acc_en=self.sharded_fp32_acc,
                    packer_l1_acc=False,
                ),
            )
            if out_sharded:
                return x
            x_interleaved = ttnn.sharded_to_interleaved(x)
            x.deallocate(True)
            return x_interleaved
        else:  # Interleaved rmsnorm does not need program or memory configs
            assert not out_sharded, "Non-sharded version of RMSNorm cannot output a sharded tensor"
            x = ttnn.layer_norm(
                x,
                weight=self.weight,
                bias=self.bias,
                epsilon=self.eps,
                compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.HiFi4,
                    math_approx_mode=False,
                    # fp32 accumulation is not optional here: from block 9 the vision tower's hidden
                    # states carry massive activations (9B: absmax 354 against an rms of 0.65), and
                    # this norm reduces the mean and the variance over all 1152 of them. In a bf16
                    # dest the outlier swamps the running sum and the ordinary channels stop
                    # contributing. Worth +0.005 PCC at full depth on the 9B with real weights.
                    fp32_dest_acc_en=True,
                    packer_l1_acc=False,
                ),
                # The consuming matmul may want its input 0 in L1 and cannot move it itself, so
                # write it where asked (None keeps ttnn's default).
                memory_config=memory_config,
            )
            return x
