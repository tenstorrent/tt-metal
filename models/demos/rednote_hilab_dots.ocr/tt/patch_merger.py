# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TTNN patch merger for dots.ocr.

DotsPatchMerger (modeling_dots_vision):
``LayerNorm(eps=1e-6) -> view(-1, C*m^2) -> Linear -> GELU -> Linear``
with hidden 1536, spatial_merge_size 2, so 1536 -> view 6144 -> 6144 -> 1536.
All four affine params present (ln_q weight+bias, both Linear biases).

Structure mirrors reference_impl models/demos/qwen25_vl/tt/patch_merger.py:
norm -> ROW_MAJOR reshape workaround (tilized ttnn.reshape hang, tt-metal
issue #29932) -> linear -> gelu -> linear. Differences vs qwen25_vl: dots.ocr
uses LayerNorm with bias (qwen used RMSNorm) and biased Linears, so we pass
``bias=`` to ttnn.linear and use ttnn.layer_norm (TILE gamma/beta per
models/tt_transformers/tt/multimodal/llama_layernorm.py).

KB ttnn_gelu cited: standalone exact ttnn.gelu(fast_and_approximate_mode=False)
after ttnn.linear replaces the torch linear->gelu subsequence (entry notes that
fusing the activation into the matmul cost PCC).

Parallelism plan (ARCHITECTURE.md): vision tower placement=replicate — all
weights ``ReplicateTensorToMesh`` on the 1x4 mesh, activations stay replicated,
no CCL. On a single device the mesh_mapper degenerates gracefully.
"""

import ttnn
from models.common.lightweightmodule import LightweightModule

TILE = 32


class TtPatchMerger(LightweightModule):
    """dots.ocr patch merger: LayerNorm -> view(-1, dim*m^2) -> Linear -> GELU -> Linear.

    Args:
        mesh_device: ttnn mesh device handle (weights replicated).
        state_dict: {"ln_q.weight": [dim], "ln_q.bias": [dim],
            "mlp.0.weight": [dim*m^2, dim*m^2], "mlp.0.bias": [dim*m^2],
            "mlp.2.weight": [out, dim*m^2], "mlp.2.bias": [out]} torch tensors
            (HF keys vision_tower.merger.*).
        spatial_merge_size: spatial merge factor m (default 2).
        eps: LayerNorm epsilon (DotsPatchMerger hard-codes 1e-6).
        dtype: on-device weight dtype.
    """

    def __init__(self, mesh_device, state_dict, spatial_merge_size=2, eps=1e-6, dtype=ttnn.bfloat16):
        super().__init__()
        self.mesh_device = mesh_device
        self.eps = eps
        dim = state_dict["ln_q.weight"].shape[0]
        self.merged_dim = dim * spatial_merge_size**2

        replicate = lambda t, layout: ttnn.from_torch(
            t,
            dtype=dtype,
            layout=layout,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        # LayerNorm gamma/beta: [1, TILE, dim] TILE_LAYOUT, per
        # models/tt_transformers/tt/multimodal/llama_layernorm.py.
        norm_param = lambda name: replicate(state_dict[name].view(1, 1, dim).expand(1, TILE, dim), ttnn.TILE_LAYOUT)
        self.norm_weight = norm_param("ln_q.weight")
        self.norm_bias = norm_param("ln_q.bias")

        # Linear weights transposed [out, in] -> [in, out] for x @ W^T; biases [1, out].
        as_weight = lambda name: replicate(state_dict[name].transpose(-2, -1).contiguous(), ttnn.TILE_LAYOUT)
        as_bias = lambda name: replicate(state_dict[name].reshape(1, -1), ttnn.TILE_LAYOUT)
        self.w1 = as_weight("mlp.0.weight")
        self.b1 = as_bias("mlp.0.bias")
        self.w2 = as_weight("mlp.2.weight")
        self.b2 = as_bias("mlp.2.bias")

        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x: [seq, dim] (or [..., seq, dim]) TILE_LAYOUT, replicated across the mesh.

        Returns: [seq / m^2, out_dim], replicated.
        """
        x = ttnn.layer_norm(
            x,
            epsilon=self.eps,
            weight=self.norm_weight,
            bias=self.norm_bias,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Merge m^2 adjacent patch rows into the feature dim: [seq, dim] ->
        # [seq/m^2, dim*m^2]. Tilized ttnn.reshape can hang (issue #29932) —
        # use the qwen25_vl ROW_MAJOR round-trip workaround.
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.reshape(x, (-1, self.merged_dim))
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        # Tracy: the up-projection keeps the default heuristic (96-core
        # DRAM-output fused-bias path; forcing core_grid 10x10 + L1 output
        # measured WORSE, 431.9 -> 448.8us — fp32 [224,6144] is matmul-bound,
        # not write-bound).
        h = ttnn.linear(
            x,
            self.w1,
            bias=self.b1,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(x)
        # KB ttnn_gelu: exact (erf) GELU, standalone after the linear.
        # Writes DRAM so the next matmul is DRAM-fed (large L1-interleaved
        # matmul operands stall the kernel, see vision_block notes).
        h = ttnn.gelu(h, fast_and_approximate_mode=False, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # Tracy-driven: the down-projection's default heuristic ran at only
        # 48 cores (249.6us). core_grid 10x10 + L1 output engages the
        # fused-bias L1 matmul variant on a fuller grid (vision_patch_embed
        # recipe): 142.4us @ 70 cores.
        out = ttnn.linear(
            h,
            self.w2,
            bias=self.b2,
            core_grid=ttnn.CoreGrid(y=10, x=10),
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(h)
        # Block-output contract: hand the (small, [seq/m^2, out]) result back
        # in DRAM interleaved like the other tower blocks.
        out_dram = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(out)
        return out_dram
