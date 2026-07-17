# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Tensor-parallel ("Megatron-style") qwen35_27b vision PatchMerger.

Mirrors the final-stretch convention from `tt_transformers.tt.model` (i.e. the
LLM `DistributedNorm -> LMHead` pair), adapted to LayerNorm and to the
two-matmul merger MLP:

  in:  fractured along dim=3 (1/TP of hidden on each device)
  norm: DistributedLayerNorm  (all-gather then LayerNorm)         -> replicated
  reshape: merge spatial_merge_size^2 patches into one row        -> replicated
           [B, 1, S, hidden]  ->  [B, 1, S/sms^2, mlp_size]
  fc1: column-sharded W1, b1  + GELU                              -> fractured
       per-device [B, 1, S/sms^2, mlp_size/TP]
  fc2: row-sharded   W2       -> partial sums of out_hidden_size
       tt_all_reduce(dim=3)   (reduce_scatter on T3K/QB2)             -> fractured
       + b2 (sharded along dim=3)
  out: fractured along dim=3, per-device [B, 1, S/sms^2, out_hidden_size/TP]

This produces a naturally-fractured output, matching the LLM `lm_head` contract
where logits are left sharded along the vocab dim.
"""

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.ccl import tt_all_reduce

from .vision_distributed_layernorm import DistributedLayerNorm


class PatchMerger(LightweightModule):
    def __init__(
        self,
        mesh_device,
        args,
        state_dict,
        state_dict_prefix,
        weight_cache_path,
        dtype,
        tt_ccl,
        postshuffle_norm: bool = False,
    ):
        super().__init__()

        # `postshuffle_norm=True` is only used by the deepstack mergers, which
        # the qwen35_27b vision tower does not use. Leaving it unsupported keeps
        # the I/O contract simple (input always fractured along the raw hidden
        # dim, not the post-shuffle mlp_size dim).
        assert not postshuffle_norm, "PatchMerger does not support postshuffle_norm=True."

        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.args = args
        self.tt_ccl = tt_ccl
        self.postshuffle_norm = postshuffle_norm
        self.cluster_shape = args.cluster_shape
        # TP across cluster axis 1 (the row axis on T3K/QB2).
        self.tp = self.cluster_shape[1]
        # On QB2 (1, 4) `tt_all_reduce` short-circuits the cluster_axis arg and
        # falls into `reduce_scatter_minimal_async` over the non-1 axis. Any
        # cluster_axis value other than 1 works; mirror MLP/VisionAttention.
        self.ccl_cluster_axis = 0

        vision_cfg = args.hf_config.vision_config
        self.hidden_size = vision_cfg.hidden_size
        self.out_hidden_size = vision_cfg.out_hidden_size
        self.mlp_size = self.hidden_size * (vision_cfg.spatial_merge_size**2)

        assert (
            self.mlp_size % self.tp == 0
        ), f"PatchMerger: mlp_size ({self.mlp_size}) must be divisible by TP={self.tp}"
        assert (
            self.out_hidden_size % self.tp == 0
        ), f"PatchMerger: out_hidden_size ({self.out_hidden_size}) must be divisible by TP={self.tp}"

        state_dict_prefix = (
            args.get_state_dict_prefix(self.__class__.__name__) if state_dict_prefix is None else state_dict_prefix
        )

        # Norm: gather fractured input back to replicated hidden_size and run
        # local LayerNorm. Mirrors the LLM's DistributedNorm just before LMHead.
        self.norm = DistributedLayerNorm(
            device=mesh_device,
            dim=self.hidden_size,
            state_dict=state_dict,
            state_dict_prefix=state_dict_prefix + ".norm",
            tt_ccl=tt_ccl,
            weight_cache_path=None if args.dummy_weights else weight_cache_path,
            weight_dtype=ttnn.bfloat16,
            eps=1e-6,  # Qwen3_VLPatchMerger hard-codes this
            ccl_topology=args.ccl_topology(),
        )

        torch_weight = lambda name: torch.transpose(self.state_dict[f"{state_dict_prefix}.{name}.weight"], -2, -1)
        torch_bias = lambda name: self.state_dict[f"{state_dict_prefix}.{name}.bias"]

        if args.dummy_weights or weight_cache_path is None:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / f"{state_dict_prefix}.{name}.tp{self.tp}"

        # ---- fc1: column-sharded (mlp_size -> mlp_size on output dim) -------------
        # torch_weight gives [mlp_size, mlp_size]. Unsqueeze to 4D before mapping.
        fc1_w = torch_weight("linear_fc1").unsqueeze(0).unsqueeze(0)
        self.w1 = ttnn.as_tensor(
            fc1_w,
            dtype=dtype,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(None, -1), mesh_shape=self.cluster_shape),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("linear_fc1_w_col"),
        )
        # 1-D bias [mlp_size] sharded along its only dim to match the col-sharded
        # output. ttnn.linear folds this in during the matmul.
        self.b1 = ttnn.as_tensor(
            torch_bias("linear_fc1"),
            dtype=ttnn.bfloat16,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(None, -1), mesh_shape=self.cluster_shape),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("linear_fc1_b_col"),
        )

        # ---- fc2: row-sharded (mlp_size -> out_hidden_size on input dim) ----------
        # torch_weight gives [mlp_size, out_hidden_size]; shard along the input dim.
        fc2_w = torch_weight("linear_fc2").unsqueeze(0).unsqueeze(0)
        self.w2 = ttnn.as_tensor(
            fc2_w,
            dtype=dtype,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(None, -2), mesh_shape=self.cluster_shape),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("linear_fc2_w_row"),
        )
        # The fc2 output is fractured along dim=3 after the reduce_scatter, so
        # we shard the bias along the same axis and fold it in *after* the
        # all-reduce, matching the MLP / VisionAttention pattern.
        self.b2 = ttnn.as_tensor(
            torch_bias("linear_fc2"),
            dtype=ttnn.bfloat16,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(None, -1), mesh_shape=self.cluster_shape),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("linear_fc2_b_frac"),
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # x enters fractured along dim=3 (block I/O contract). DistributedLayerNorm
        # all-gathers internally and returns a replicated tensor on hidden_size.
        x_norm = self.norm(x)

        # Merge spatial_merge_size^2 consecutive rows into one row of mlp_size.
        # The reshape workaround through ROW_MAJOR matches the replicated PatchMerger
        # (see tt-metal#29932 for the underlying tilized reshape hang).
        x_norm = ttnn.to_layout(x_norm, ttnn.ROW_MAJOR_LAYOUT)
        x_norm = ttnn.reshape(x_norm, (x_norm.shape[0], x_norm.shape[1], -1, self.mlp_size))
        x_norm = ttnn.to_layout(x_norm, ttnn.TILE_LAYOUT)

        # fc1 column-sharded: replicated in -> fractured-along-dim=3 out
        # (each device owns mlp_size/TP of the GELU activations).
        w1_out = ttnn.linear(
            x_norm,
            self.w1,
            bias=self.b1,
            activation="gelu",
            compute_kernel_config=self.args.compute_kernel_config_hifi2_fp16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(x_norm)

        # fc2 row-sharded: each device produces partial sums for the full
        # out_hidden_size. Bias is folded in after the reduce_scatter.
        w2_partial = ttnn.linear(
            w1_out,
            self.w2,
            compute_kernel_config=self.args.compute_kernel_config_hifi2_fp16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(w1_out)

        # On T3K/QB2 `tt_all_reduce(dim=3)` is a reduce_scatter, so the result
        # is fractured along dim=3 -- the same contract as the rest of the TP
        # path and as the LLM's lm_head output.
        w2_frac = tt_all_reduce(
            w2_partial,
            self.mesh_device,
            self.tt_ccl,
            cluster_axis=self.ccl_cluster_axis,
            dim=3,
            sharded=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=self.args.ccl_dtype,
            topology=self.args.ccl_topology(),
        )
        if w2_frac is not w2_partial:
            ttnn.deallocate(w2_partial)

        out = ttnn.add(w2_frac, self.b2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if out is not w2_frac:
            ttnn.deallocate(w2_frac)
        return out
