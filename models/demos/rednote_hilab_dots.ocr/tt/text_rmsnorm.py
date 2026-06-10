# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TTNN text RMSNorm for dots.ocr (Qwen2RMSNorm, eps=1e-6).

Qwen2RMSNorm: fp32 variance — x * rsqrt(mean(x^2, -1) + eps) — cast back, then
scale by the learned weight. The decomposed pow -> mean -> rsqrt -> mul chain
maps onto the single fused ``ttnn.rms_norm`` op (cf. KB entry ttnn_pow: the
chain "is a fusion candidate into ttnn.rms_norm"); reference_impl
models/common/rmsnorm.py uses the identical fused op with a
[1, 1, dim//32, 32] ROW_MAJOR gamma.

Parallelism plan (ARCHITECTURE.md / inventory): decoder_stack is sharded
4-way; text_rmsnorm placement=shard via the distributed norm recipe of
models/common/rmsnorm.py / tt_transformers distributed_norm.py, with the
gamma replicated logically (each device holds its hidden slice). Mirroring
the reference_impl, this block carries BOTH paths:

- ``forward``: replicated activation -> fused ``ttnn.rms_norm`` with a
  replicated gamma (the tt_transformers non-TG path; correct whenever the
  decoder keeps a replicated residual stream between CCLs).
- ``forward_distributed``: hidden-sharded activation [.., dim/N per device]
  -> ``ttnn.rms_norm_pre_all_gather`` (per-device sum(x^2) stats) ->
  ``ttnn.all_gather(dim=3, Topology.Linear)`` (sync variant, acceptable for
  first-pass correctness per tp-guidance; async in optimization phase) ->
  ``ttnn.rms_norm_post_all_gather`` with the dim-2-sharded gamma (KB entry
  ttnn_rms_norm_post_all_gather). Output stays hidden-sharded.
"""

import ttnn
from models.common.lightweightmodule import LightweightModule

TILE = 32


class TtTextRMSNorm(LightweightModule):
    """dots.ocr text RMSNorm (Qwen2, eps=1e-6) with replicated + distributed paths.

    Args:
        mesh_device: ttnn mesh device handle.
        state_dict: {"weight": [dim]} torch tensor (HF key e.g.
            model.layers.N.input_layernorm.weight).
        dtype: on-device weight dtype.
        eps: RMSNorm epsilon (Qwen2 text decoder uses 1e-6).
    """

    def __init__(self, mesh_device, state_dict, dtype=ttnn.bfloat16, eps=1e-6):
        super().__init__()
        self.mesh_device = mesh_device
        self.eps = eps

        weight = state_dict["weight"]
        dim = weight.shape[-1]
        # ttnn.rms_norm gamma format: [1, 1, dim//32, 32] in ROW_MAJOR for
        # bf16. The ROW_MAJOR gamma path is bf16-only (an fp32 ROW_MAJOR
        # gamma is misread on device, PCC ~0) — fp32 gammas use TILE
        # [1, 1, 1, dim] instead (cf. tt/vision_rmsnorm.py).
        if dtype == ttnn.float32:
            gamma, gamma_layout = weight.reshape(1, 1, 1, dim), ttnn.TILE_LAYOUT
        else:
            gamma, gamma_layout = weight.reshape(1, 1, dim // TILE, TILE), ttnn.ROW_MAJOR_LAYOUT
        # Replicated gamma for the fused single-op path.
        self.weight = ttnn.from_torch(
            gamma,
            dtype=dtype,
            layout=gamma_layout,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        # Hidden-sharded gamma for the distributed path: shard the
        # [1, 1, dim//32, 32] ROW_MAJOR gamma on dim 2 — per-device
        # [1, 1, (dim/N)//32, 32], matching the per-device hidden slice
        # (models/common/rmsnorm.py weight_distributed, ShardTensor2dMesh
        # dims=(None, 2); 1xN line -> plain dim-2 shard).
        self.weight_distributed = ttnn.from_torch(
            weight.reshape(1, 1, dim // TILE, TILE),
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=2),
        )
        # Reference_impl models/common/rmsnorm.py runs HiFi2 + fp32 dest acc
        # for the norm; init_device_compute_kernel_config picks the
        # arch-correct config struct (Blackhole here).
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Replicated path. x: [..., dim] TILE_LAYOUT, replicated across the mesh.

        Returns: same shape, replicated.
        """
        return ttnn.rms_norm(
            x,
            epsilon=self.eps,
            weight=self.weight,
            compute_kernel_config=self.compute_kernel_config,
        )

    def forward_distributed(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Distributed path. x: [1, 1, seq, dim/N] per device (hidden-sharded).

        Per-device partial sum(x^2) stats are all-gathered so every device
        normalizes by the FULL-hidden variance, then scales by its local
        gamma slice. Returns: same shape, still hidden-sharded.
        """
        stats = ttnn.rms_norm_pre_all_gather(x, compute_kernel_config=self.compute_kernel_config, dtype=ttnn.bfloat16)
        stats = ttnn.all_gather(stats, dim=3, topology=ttnn.Topology.Linear)
        out = ttnn.rms_norm_post_all_gather(
            x,
            stats,
            epsilon=self.eps,
            weight=self.weight_distributed,
            compute_kernel_config=self.compute_kernel_config,
        )
        stats.deallocate(True)
        return out
