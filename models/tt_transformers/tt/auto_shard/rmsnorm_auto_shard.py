# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""RMSNorm that matches the auto-shard residual stream's actual layout.

The stream is only ever one of two layouts, and this norms each in place -- no forced full-activation
gather like the stock DistributedNorm, and no axis-blind ring:

    axis=None   replicated activation (full dim on every chip): a plain local rms_norm, zero collectives.
    axis=a      activation fractured on mesh axis a: compute partial stats on the local shard, all-gather
                just the STATS over that named axis (cheap, and 2D-safe), then normalize the shard. The
                activation stays fractured; the weight is sharded on the same axis to match.
"""

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.auto_shard.ccl_auto_shard import all_gather

TILE = 32  # rms_norm weight is laid out [1, 1, dim // TILE, TILE]


class RMSNorm(LightweightModule):
    def __init__(
        self,
        device,
        dim,
        state_dict,
        weight_key,
        axis,
        layer_num=None,
        state_dict_prefix=None,
        weight_dtype=ttnn.bfloat16,
        eps: float = 1e-05,
        add_unit_offset=False,
        fp32_dest_acc_en=True,
    ):
        super().__init__()
        self.mesh_device = device
        self.eps = eps
        self.axis = axis  # mesh axis the hidden dim is fractured on, or None (replicated)

        if state_dict_prefix:
            weight_name = f"{state_dict_prefix}{weight_key}.weight"
        elif layer_num is None:
            weight_name = f"{weight_key}.weight"
        else:
            weight_name = f"layers.{layer_num}.{weight_key}.weight"

        w = state_dict[weight_name].unsqueeze(0).view(1, 1, dim).reshape([1, 1, dim // TILE, TILE])
        if add_unit_offset:  # Gemma-style: weights are stored as (1 + w)
            w = w + 1.0

        # Replicated stream -> replicate the weight; fractured stream -> shard the weight's hidden dim
        # (tensor dim 2) on the same mesh axis, so each chip normalizes its shard with its weight slice.
        if axis is None:
            mesh_mapper = ttnn.ReplicateTensorToMesh(device)
        else:
            dims = [None, None]
            dims[axis] = 2
            mesh_mapper = ttnn.ShardTensor2dMesh(device, dims=tuple(dims), mesh_shape=tuple(device.shape))

        self.weight = ttnn.as_tensor(
            w,
            device=device,
            dtype=weight_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )

        # fp32_dest_acc_en doubles the norm's intermediate circular buffers, which scale with the
        # hidden dim: 168 tiles for Gemma-3-27B's 5376 vs 128 for Llama-8B's 4096. That is enough to
        # push a replicated norm past the 1499136 B L1 budget, so wide models pass False here. Same
        # workaround decoder.py:102-120 applies to Qwen and to Llama-8B on a Galaxy row submesh.
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=fp32_dest_acc_en,
            packer_l1_acc=True,
        )

    def forward(self, x, mode=None, norm_config=None):
        # mode/norm_config are accepted so this norm is a drop-in for the stock DistributedNorm at the
        # inherited trace/batched-prefill helpers (_apply_norm_and_lm_head, process_*). They don't
        # affect an axis=None local norm or a stats-gather fractured norm, so they're ignored here.
        if self.axis is None:
            # Replicated: every chip has the full hidden vector, so norm locally.
            return ttnn.rms_norm(x, epsilon=self.eps, weight=self.weight, compute_kernel_config=self.compute_kernel_config)

        # Fractured: local partial stats -> gather just the stats over the shard axis -> normalize the shard.
        # rms_norm_pre_all_gather (like the stock DistributedNorm) only takes an interleaved input; in
        # decode the residual is L1 width-sharded, so pull it back to DRAM interleaved first.
        if x.is_sharded():
            x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)
        stats = ttnn.rms_norm_pre_all_gather(x, compute_kernel_config=self.compute_kernel_config, dtype=ttnn.bfloat16)
        stats = all_gather(stats, self.mesh_device, self.axis, label="rmsnorm stats all_gather")
        out = ttnn.rms_norm_post_all_gather(
            x, stats, epsilon=self.eps, weight=self.weight, compute_kernel_config=self.compute_kernel_config
        )
        stats.deallocate(True)
        return out
