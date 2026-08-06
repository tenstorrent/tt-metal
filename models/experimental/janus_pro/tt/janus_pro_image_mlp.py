"""
Feed-forward network for the Janus-Pro-7B vision tower.

HF reference: `vision_model.encoder.layers[i].mlp` (`ModelArgs.reference_vision_mlp`).
"""

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


class TtJanusProImageFeedForward(LightweightModule):
    def __init__(
        self,
        mesh_device,
        tt_ccl,
        args,
        state_dict,
        state_dict_prefix,
        weight_cache_path,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.args = args
        torch_weight = lambda name, suffix: torch.transpose(
            self.state_dict[f"{state_dict_prefix}{name}.{suffix}"], -2, -1
        )
        torch_bias = lambda name, suffix: self.state_dict[f"{state_dict_prefix}{name}.{suffix}"]

        if weight_cache_path is None:
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
        self.c_fc_weight = as_interleaved_tensor("c_fc", "weight", ttnn.bfloat8_b, dim=-1)
        self.c_fc_bias = as_interleaved_tensor("c_fc", "bias", ttnn.bfloat16, dim=-1)
        self.c_fc_bias = ttnn.reshape(self.c_fc_bias, [1, -1])
        self.c_proj_weight = as_interleaved_tensor("c_proj", "weight", ttnn.bfloat8_b, dim=-2)
        self.c_proj_bias = as_interleaved_tensor("c_proj", "bias", ttnn.bfloat16, dim=None)
        self.c_proj_bias = ttnn.reshape(self.c_proj_bias, [1, -1])

        # c_proj's bias may ride inside its matmul only when nothing reduces after it. With more
        # than one device the all-reduce would fold the bias in once per device.
        self.fuse_c_proj_bias = self.args.num_devices == 1

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        FeedForward: c_proj(gelu(c_fc(x) + b_fc)) + b_proj
        HF reference (SigLIP MLP): fc2(act_fn(fc1(x)))  ->  c_fc = fc1, c_proj = fc2, act_fn = gelu.
        This is a plain two-layer FFN, not a gated/SwiGLU FFN: no gate/up split and no third projection.
        c_fc's bias and gelu ride inside its linear. c_proj's bias does too on a single device;
        with a mesh it stays a separate add after the all-reduce, so it is not scaled by the
        device count.
        """
        seq_len = x.shape[-2]
        batch_size = x.shape[0]

        # Depends on whether we are padding or not
        MAX_MM_SEQ_LEN = seq_len

        x_in = x
        if seq_len >= MAX_MM_SEQ_LEN:  # Too big to compute. Set different program configs based on seqlen
            # Reshape input to to fit on device and parallelize computation
            x_in = ttnn.reshape(x_in, [batch_size, seq_len // MAX_MM_SEQ_LEN, MAX_MM_SEQ_LEN, -1])

        # The activation must ride inside the config -- `activation=` alongside an explicit one
        # appends a second gelu op (matmul.cpp). Its parameter is gelu's APPROXIMATION_MODE
        # (ckernel_sfpu_gelu.h:209-215 vs :301-311).
        gelu = (ttnn.UnaryOpType.GELU, True)

        # 2D reads the shard ln_2 hands over in place. It is no faster than 1D on this shape, but it
        # is what removes the unshard, and that pays several times the difference.
        c_fc_program_config = self.args.vision_c_fc_program_config(batch_size, seq_len, fused_activation=gelu)

        # Without a config ttnn derives its own, which on an L1 in0 picks a 1D strategy an order of
        # magnitude slower, and the shard ln_2 hands over would need undoing. Fail instead: the only
        # way here is a sequence long enough that the output block overruns L1.
        assert c_fc_program_config is not None, f"no 2D c_fc config for seq_len {seq_len}"

        # A sharded output is taken only on one device: otherwise it flows into the all-gather,
        # which would carry a shard spec covering a fraction of the gathered tensor.
        c_proj_program_config = self.args.vision_c_proj_program_config(batch_size, seq_len)
        shard_mlp_outputs = c_proj_program_config is not None and self.args.num_devices == 1

        # The intermediate stays in L1 rather than taking a DRAM round trip to c_proj, and is
        # block-sharded rather than interleaved so each core writes its own output into its own L1
        # instead of scattering it across every core's banks over the NOC. The writer loop stays --
        # BRISC still runs for the whole matmul -- but its transactions become local.
        # The shard is 16 tiles wide, which c_proj's in0_block_w divides, so c_proj reads it in place.
        c_fc_out = ttnn.linear(
            x_in,
            self.c_fc_weight,
            bias=self.c_fc_bias,
            compute_kernel_config=self.args.compute_kernel_config_lofi,
            program_config=c_fc_program_config,
            # c_proj is the only consumer, and halving it also halves the L1 residency the budget
            # above does not account for.
            dtype=ttnn.bfloat8_b,
            memory_config=(ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG if shard_mlp_outputs else ttnn.L1_MEMORY_CONFIG),
        )
        ttnn.deallocate(x_in)

        c_proj_out = ttnn.linear(
            c_fc_out,
            self.c_proj_weight,
            bias=self.c_proj_bias if self.fuse_c_proj_bias else None,
            compute_kernel_config=self.args.compute_kernel_config_lofi,
            program_config=c_proj_program_config,
            # Read once by the block's add; the residual it lands in stays bfloat16.
            dtype=ttnn.bfloat8_b,
            # A sharded output compiles the writer loop out of the kernel that also reads in1.
            memory_config=(ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG if shard_mlp_outputs else ttnn.DRAM_MEMORY_CONFIG),
        )
        ttnn.deallocate(c_fc_out)

        # NOTE: Need to reshape to 4D so that fast_reduce_nc hsa a dim1 to work on
        c_proj_out = ttnn.reshape(c_proj_out, [batch_size, 1, seq_len, -1])

        # All reduce
        if self.args.num_devices > 1:  # replace with reduce_scatter and all_gather
            w2_out_gathered = ttnn.experimental.all_gather_async(
                c_proj_out,
                persistent_output_buffer=None,
                dim=1,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
                num_links=4 if self.args.is_galaxy else 1,
                topology=ttnn.Topology.Ring,
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                chunks_per_sync=10,
                num_workers_per_link=2,
                num_buffers_per_channel=2,
            )

            ttnn.deallocate(c_proj_out)

            pre_bias_output = ttnn.experimental.fast_reduce_nc(
                w2_out_gathered, dims=[1], output=None, compute_kernel_config=None
            )
            ttnn.deallocate(w2_out_gathered)
        else:
            pre_bias_output = c_proj_out

        if self.fuse_c_proj_bias:
            return pre_bias_output

        output = ttnn.add(pre_bias_output, self.c_proj_bias)
        ttnn.deallocate(pre_bias_output)
        return output
