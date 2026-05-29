# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""TTNN implementation of the dots.ocr DotsVisionTransformer PatchMerger.

Reference: models/demos/rednote_hilab_dots.ocr/reference/functional.py
           :func:`vision_patch_merger_forward`

DotsPatchMerger (pre_norm = 'layernorm'):

    x = layer_norm(x, ln_q.weight, ln_q.bias, eps=1e-6)   # NB: LayerNorm w/ bias
    x = x.view(-1, context_dim * spatial_merge_size**2)    # group merge**2 patches
    x = linear(x, mlp.0.weight, mlp.0.bias)                # [hidden, hidden]
    x = gelu(x)
    x = linear(x, mlp.2.weight, mlp.2.bias)                # [hidden, out_dim]

context_dim 1536, spatial_merge_size 2 -> hidden_size = 1536*4 = 6144,
out_dim 1536. All Linears are biased; the pre-norm is a true LayerNorm (weight
AND bias), not an RMSNorm.

The patch-grouping reshape is done on device via ttnn.reshape with no host
fallback: the reference ``view(-1, hidden_size)`` is a plain row-major regroup
of ``merge**2`` consecutive patch tokens, so a contiguous reshape matches it
exactly.

Reference TTNN impl this follows: models/demos/qwen25_vl/tt/patch_merger.py
"""

import ttnn
from models.common.lightweightmodule import LightweightModule


class TtVisionPatchMerger(LightweightModule):
    """dots.ocr vision PatchMerger.

    Args:
        device: ttnn Device or MeshDevice.
        ln_weight: torch.Tensor [context_dim] (LayerNorm gamma).
        ln_bias:   torch.Tensor [context_dim] (LayerNorm beta).
        fc1_weight: torch.Tensor [hidden, hidden] (mlp.0, biased).
        fc1_bias:   torch.Tensor [hidden].
        fc2_weight: torch.Tensor [out_dim, hidden] (mlp.2, biased).
        fc2_bias:   torch.Tensor [out_dim].
        context_dim: per-patch feature width (1536).
        spatial_merge_size: 2 -> groups 2x2 = 4 patches.
        ln_eps: LayerNorm epsilon (1e-6).
        dtype: activation/weight dtype (bf16).
    """

    def __init__(
        self,
        device,
        ln_weight,
        ln_bias,
        fc1_weight,
        fc1_bias,
        fc2_weight,
        fc2_bias,
        context_dim: int = 1536,
        spatial_merge_size: int = 2,
        ln_eps: float = 1e-6,
        dtype=ttnn.bfloat16,
        weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ):
        super().__init__()
        self.device = device
        self.context_dim = context_dim
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_eps = ln_eps

        is_mesh_device = device.__class__.__name__ == "MeshDevice"
        mesh_mapper = ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None

        def _as(t, layout=ttnn.TILE_LAYOUT):
            return ttnn.as_tensor(
                t,
                device=device,
                dtype=dtype,
                layout=layout,
                memory_config=weight_memory_config,
                mesh_mapper=mesh_mapper,
            )

        # LayerNorm gamma/beta: ttnn.layer_norm wants them laid out one tile high,
        # reshaped to [1, 1, dim // TILE, TILE] in row-major (matches the rmsnorm block).
        TILE = 32
        self.ln_weight = _as(ln_weight.reshape([1, 1, context_dim // TILE, TILE]), layout=ttnn.ROW_MAJOR_LAYOUT)
        self.ln_bias = _as(ln_bias.reshape([1, 1, context_dim // TILE, TILE]), layout=ttnn.ROW_MAJOR_LAYOUT)

        # ttnn.linear computes x @ W with W as [in, out]; transpose torch weights.
        self.fc1_weight = _as(fc1_weight.transpose(0, 1).contiguous())  # [hidden, hidden]
        self.fc1_bias = _as(fc1_bias.reshape(1, -1))
        self.fc2_weight = _as(fc2_weight.transpose(0, 1).contiguous())  # [hidden, out_dim]
        self.fc2_bias = _as(fc2_bias.reshape(1, -1))

        # fp32 compute to match the reference float path.
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x: [num_patches, context_dim] (TILE layout) -> [num_patches // merge**2, out_dim]."""
        # Pre-norm: true LayerNorm (weight AND bias).
        x = ttnn.layer_norm(
            x,
            epsilon=self.ln_eps,
            weight=self.ln_weight,
            bias=self.ln_bias,
            compute_kernel_config=self.compute_kernel_config,
        )

        # Group merge**2 consecutive patch tokens: [N, context_dim] -> [N//m^2, hidden].
        x = ttnn.reshape(x, (-1, self.hidden_size))

        # mlp.0 -> GELU -> mlp.2 (both biased).
        #
        # NB: this block is compute-bound on the two K=6144 matmuls (~80% of
        # device-kernel time, already on 96 cores). Forcing the fc1/GELU/fc2
        # chain to L1_MEMORY_CONFIG was measured under traced tracy and is a
        # no-op on latency (368 vs 372 us): the small merged-token M makes the
        # intermediate DRAM round-trips cheap relative to the matmul compute,
        # and an explicit L1 output without core_grid splits the matmul's fused
        # bias into a separate BinaryNg add (+6 us). The default DRAM-interleaved
        # output keeps the bias fused into the matmul, so it is left in place.
        x = ttnn.linear(
            x,
            self.fc1_weight,
            bias=self.fc1_bias,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
        )
        x = ttnn.gelu(x)
        x = ttnn.linear(
            x,
            self.fc2_weight,
            bias=self.fc2_bias,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
        )
        return x
