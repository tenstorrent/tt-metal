# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Distributed RMSNorm for Qwen3.6-27B on BH GLX 8×4 mesh.

Forked from models/demos/llama3_70b_galaxy/tt/distributed_norm.py with one
addition: the ``zero_centered`` constructor flag.

When ``zero_centered=True`` the weight preprocessing step applies ``w += 1.0``
**before** storing the tensor on device.  The forward path is identical to the
standard case — it relies on ``ttnn.rms_norm_pre_all_gather`` /
``ttnn.rms_norm_post_all_gather``.  This implements the HF Qwen3NextRMSNorm
convention::

    output = (1 + w) * x * rsqrt(var + eps)

without any extra on-device arithmetic: the +1 is baked into the stored weight.

Architecture
------------
The distributed norm is used for the *residual-stream* norms
(``input_layernorm``, ``post_attention_layernorm``, final ``model.norm``), all
of which use zero_centered=True in Qwen3.6-27B.

Input layout
------------
The input tensor is expected to be sharded across cluster_axis=1 (the 4 mesh
columns).  Each column holds a 1280-wide (= 5120 / 4) slice.  This matches the
layout produced by the preceding attention / MLP blocks.

Weight layout
-------------
The weight ``[dim]`` is reshaped to ``[1, 1, dim//32, 32]`` (tile-aligned row-
major) and then sharded across column (cluster_axis=1) so that each of the 4
columns owns a ``[1, 1, dim//4//32, 32]`` slice.  The weight is replicated
across the 8 rows.

AllGather pattern
-----------------
Pre-gather stats are all-gathered across cluster_axis=1 (dim=3) using
``ttnn.experimental.all_gather_matmul`` / ``ttnn.all_gather``, then
post-gather normalization is applied.

This module targets the **prefill** path (mode-agnostic forward).  A sharded
decode path can be added later by wrapping ``ttnn.fused_rms_minimal`` via
``tt_ccl``, as done in the parent llama3_70b_galaxy file.
"""

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule

_TILE = 32


class DistributedNorm(LightweightModule):
    """Distributed RMSNorm for Qwen3.6-27B on BH GLX 8×4 mesh.

    Parameters
    ----------
    mesh_device : ttnn.MeshDevice
        The full 8×4 mesh.
    weight_torch : torch.Tensor
        1-D float32 tensor of shape ``[dim]``.  Typically loaded directly from
        the HF safetensors weight (e.g. ``input_layernorm.weight``).
    eps : float
        RMSNorm epsilon (Qwen3.6 uses 1e-6).
    zero_centered : bool
        When True, apply ``w += 1.0`` before storing on device, implementing
        the HF ``Qwen3NextRMSNorm`` zero-centred convention
        ``output = (1+w) * norm(x)``.
        When False (standard), the stored weight is used as-is:
        ``output = w * norm(x)``.
    weight_dtype : ttnn.DataType
        On-device weight dtype.  Defaults to bfloat16.
    weight_memory_config : ttnn.MemoryConfig
        On-device weight memory config.  Defaults to DRAM.
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        weight_torch: torch.Tensor,
        eps: float = 1e-6,
        zero_centered: bool = False,
        weight_dtype=ttnn.bfloat16,
        weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ):
        super().__init__()
        self.eps = eps
        self.zero_centered = zero_centered

        cluster_shape = list(mesh_device.shape)  # [8, 4]
        assert (
            len(cluster_shape) == 2 and cluster_shape[1] == 4
        ), f"DistributedNorm expects an 8×4 mesh; got shape {cluster_shape}"

        # ------------------------------------------------------------------
        # Weight preprocessing
        # ------------------------------------------------------------------
        # weight_torch: [dim] float32
        dim = weight_torch.numel()
        assert dim % _TILE == 0, f"dim={dim} must be divisible by tile size {_TILE}"

        # Convert to [1, 1, dim/TILE, TILE] tile-aligned row-major layout
        w = weight_torch.clone().float()

        if zero_centered:
            # Bake +1 into the weight so forward path is unchanged
            w = w + 1.0

        # Tile reshape: [1, 1, dim//32, 32]
        w = w.reshape(1, 1, dim // _TILE, _TILE)

        # ------------------------------------------------------------------
        # Store weight distributed across columns (cluster_axis=1)
        # Rows are irrelevant for the norm weight — replicate across rows.
        # ShardTensor2dMesh(dims=(None, 2)) shards dim-2 across mesh columns.
        # ------------------------------------------------------------------
        self.weight = ttnn.as_tensor(
            w,
            device=mesh_device,
            dtype=weight_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=weight_memory_config,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 2), mesh_shape=cluster_shape),
        )

        # ------------------------------------------------------------------
        # Compute kernel config (matches llama3_70b_galaxy convention)
        # BH uses WormholeComputeKernelConfig (BlackholeComputeKernelConfig
        # does not exist in this build).
        # ------------------------------------------------------------------
        # HiFi4 + fp32 dest accumulation: the pre-all-gather stats step computes
        # sum-of-squares which compounds rounding for small-magnitude activations
        # (e.g., embedding output std≈0.013 → rsqrt of small variance is precision-
        # critical).  Float32 accumulation preserves PCC on real activations.
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Apply distributed RMSNorm.

        Parameters
        ----------
        x : ttnn.Tensor
            Input tensor sharded across cluster_axis=1 (cols), shape
            ``[B, 1, T, dim/4]`` per device, dtype bfloat16.

        Returns
        -------
        ttnn.Tensor
            Normalised tensor, same layout as input (still sharded across cols).
        """
        use_2d_grid = False

        # Step 1: compute local sum-of-squares statistics per shard
        tt_stats = ttnn.rms_norm_pre_all_gather(
            x,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            use_2d_core_grid=use_2d_grid,
        )

        # Step 2: all-gather stats across cluster_axis=1 (cols)
        tt_stats_gathered = ttnn.all_gather(
            tt_stats,
            dim=3,
            num_links=1,
            cluster_axis=1,
            topology=ttnn.Topology.Linear,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_stats.deallocate(True)

        # Step 3: apply normalisation with gathered stats + per-col weight shard
        tt_out = ttnn.rms_norm_post_all_gather(
            x,
            tt_stats_gathered,
            epsilon=self.eps,
            weight=self.weight,
            compute_kernel_config=self.compute_kernel_config,
            use_2d_core_grid=use_2d_grid,
        )
        tt_stats_gathered.deallocate(True)

        return tt_out
