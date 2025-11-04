"""
This is the FeedForward submodule for vision block in Gemma-3-4b-it
We have reused the TtLlamaImageFeedForward with few changes in CoreGrid and program_config configurations
"""

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


def get_out_subblock_w(per_core_N, out_subblock_h):
    """
    Helper function to calculate the out_subblock_w based on the per_core_N and out_subblock_h
    """
    out_subblock_w = 4  # TODO: Check with LLK team if this is the true bound, might be 8 now
    while out_subblock_w > 1:
        if out_subblock_w * out_subblock_h <= 4 and per_core_N % out_subblock_w == 0:
            break
        out_subblock_w -= 1
    return out_subblock_w


def matmul_config(
    self,
    m: int,
    k: int,
    n: int,
    grid_size,
    in0_block_w: int = None,
    fuse_batch: bool = False,
    fused_activation=None,
) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
    self.tile_size = ttnn.TILE_SIZE

    per_core_M = math.ceil(m / (self.tile_size * grid_size[1]))
    per_core_N = math.ceil(n / (self.tile_size * grid_size[0]))

    out_subblock_h = 1
    out_subblock_w = get_out_subblock_w(per_core_N, out_subblock_h)

    if in0_block_w is None:
        in0_block_w = min(4, max(1, k // (self.tile_size * grid_size[0])))

    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=fused_activation,
        fuse_batch=fuse_batch,
    )


class TtGemmaImageFeedForward(LightweightModule):
    def __init__(
        self,
        mesh_device,
        tt_ccl,
        args,
        state_dict,
        state_dict_prefix,
        weight_cache_path,
        dtype,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.args = args
        self.model_config = args.get_model_config()
        torch_weight = lambda name, suffix: torch.transpose(
            self.state_dict[f"{state_dict_prefix}{name}.{suffix}"], -2, -1
        )
        torch_bias = lambda name, suffix: self.state_dict[f"{state_dict_prefix}{name}.{suffix}"]

        if args.dummy_weights:
            cache_name = lambda *_: None
        else:
            cache_name = lambda name, suffix: weight_cache_path / (state_dict_prefix + f"{name}.{suffix}")

        as_interleaved_tensor = lambda name, suffix, type, dim: ttnn.as_tensor(
            (
                torch_weight(name, suffix) if suffix == "weight" else torch_bias(name, suffix)
            ),  # Grab only the wX part of the name
            dtype=type,
            device=self.mesh_device,
            mesh_mapper=(
                ttnn.ShardTensorToMesh(self.mesh_device, dim=dim)
                if dim is not None
                else ttnn.ReplicateTensorToMesh(self.mesh_device)
            ),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name(name, suffix),
        )

        # Sharded weights
        self.c_fc_weight = as_interleaved_tensor("c_fc", "weight", dtype, dim=-1)
        self.c_fc_bias = as_interleaved_tensor("c_fc", "bias", ttnn.bfloat16, dim=-1)
        self.c_fc_bias = ttnn.reshape(self.c_fc_bias, [1, -1])
        self.c_proj_weight = as_interleaved_tensor("c_proj", "weight", dtype, dim=-2)
        self.c_proj_bias = as_interleaved_tensor("c_proj", "bias", ttnn.bfloat16, dim=None)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        w1 -> gate_proj
        w2 -> down_proj
        w3 -> up_proj
        HF reference: self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        """
        seq_len = x.shape[-2]
        batch_size = x.shape[0]

        # Depends on whether we are padding or not
        MAX_MM_SEQ_LEN = seq_len if self.args.is_gemma else self.args.VISION_MAX_MM_SEQ

        x_in = x
        if seq_len >= MAX_MM_SEQ_LEN:  # Too big to compute. Set different program configs based on seqlen
            # Reshape input to to fit on device and parallelize computation
            x_in = ttnn.reshape(x_in, [batch_size, seq_len // MAX_MM_SEQ_LEN, MAX_MM_SEQ_LEN, -1])

        # These use HiFi2; this drops 1 bit of the activations but would be FLOP-bound on 12 cores with HiFi4
        c_fc_out = ttnn.linear(
            x_in,
            self.c_fc_weight,
            bias=self.c_fc_bias,
            compute_kernel_config=self.args.compute_kernel_config_hifi4,
            dtype=ttnn.bfloat16,
            program_config=matmul_config(
                self, x_in.shape[-2], x_in.shape[-1], self.c_fc_weight.shape[-1], (8, 8), fuse_batch=False
            ),
            activation="gelu",  # NOTE: activation must be passed to linear here, not in program config! Bad output otherwise
        )

        c_proj_out = ttnn.linear(
            c_fc_out,
            self.c_proj_weight,
            compute_kernel_config=self.args.compute_kernel_config_hifi4,
            dtype=ttnn.bfloat8_b,
            program_config=matmul_config(
                self, c_fc_out.shape[-2], c_fc_out.shape[-1], self.c_proj_weight.shape[-1], (8, 8), fuse_batch=False
            ),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # NOTE: Need to reshape to 4D so that fast_reduce_nc hsa a dim1 to work on
        c_proj_out = ttnn.reshape(c_proj_out, [batch_size, 1, seq_len, -1])

        # All reduce
        if self.args.num_devices > 1:  # replace with reduce_scatter and all_gather
            w2_out_gathered = ttnn.experimental.all_gather_async(
                c_proj_out,
                persistent_output_buffer=None,
                dim=1,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
                num_links=1,
                topology=ttnn.Topology.Ring,
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                chunks_per_sync=10,
                num_workers_per_link=2,
                num_buffers_per_channel=2,
            )

            pre_bias_output = ttnn.experimental.fast_reduce_nc(
                w2_out_gathered, dims=[1], output=None, compute_kernel_config=None
            )
        else:
            pre_bias_output = c_proj_out

        output = ttnn.add(pre_bias_output, self.c_proj_bias)
        return output
