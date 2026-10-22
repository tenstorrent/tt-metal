# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Plain Llama-3.1 RMSNorm for sequence-parallel Galaxy prefill."""

import math
import os
from numbers import Real

import torch

import ttnn
from models.demos.llama_3p1_8b_d_p.reference.llama_3p1_8b_config import Llama31_8BConfig

_FUSED_RESOURCES = {}


class RMSNorm:
    """Normalize a full hidden row locally on every SP/TP mesh chip."""

    def __init__(self, mesh_device, weight, *, eps=Llama31_8BConfig.RMS_NORM_EPS):
        if not isinstance(weight, torch.Tensor) or weight.ndim != 1:
            shape = getattr(weight, "shape", None)
            raise ValueError(f"RMSNorm weight must be a one-dimensional host tensor, got shape {shape}")
        if weight.shape[0] != Llama31_8BConfig.EMB_SIZE:
            raise ValueError(f"RMSNorm weight must have width {Llama31_8BConfig.EMB_SIZE}, got {weight.shape[0]}")
        if not isinstance(eps, Real) or not math.isfinite(eps) or eps <= 0:
            raise ValueError(f"RMSNorm epsilon must be finite and positive, got {eps!r}")

        self.mesh_device = mesh_device
        self.eps = float(eps)
        self.weight = ttnn.from_torch(
            weight.to(torch.bfloat16).reshape(1, 1, Llama31_8BConfig.EMB_SIZE // ttnn.TILE_SIZE, ttnn.TILE_SIZE),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        self.sp_weight = None
        if os.getenv("LOCAL_LLAMA_SP") == "1":
            # TP-sharded gamma for the distributed norm: 4096/8 = 512 columns = 16 tile rows per chip.
            self.sp_weight = ttnn.from_torch(
                weight.to(torch.bfloat16).reshape(1, 1, Llama31_8BConfig.EMB_SIZE // ttnn.TILE_SIZE, ttnn.TILE_SIZE),
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, dims=(None, 2)),
            )
        self.sp_fused_weight = None
        if self.sp_weight is not None and os.getenv("LOCAL_LLAMA_FNORM") == "1":
            # The fused distributed op takes a tiled row of gamma: 512 local columns.
            local_width = Llama31_8BConfig.EMB_SIZE // mesh_device.shape[1]
            self.sp_fused_weight = ttnn.to_layout(
                ttnn.reshape(self.sp_weight, (1, 1, 1, local_width)), ttnn.TILE_LAYOUT
            )
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def __call__(self, x):
        if not isinstance(x, ttnn.Tensor) or not ttnn.is_tensor_storage_on_device(x):
            raise ValueError("RMSNorm input must be a device ttnn.Tensor")
        if x.device() != self.mesh_device:
            raise ValueError("RMSNorm input must reside on the constructor mesh")
        if len(ttnn.get_device_tensors(x)) != self.mesh_device.get_num_devices():
            raise ValueError("RMSNorm input must cover every device in the constructor mesh")
        if x.shape[-1] != Llama31_8BConfig.EMB_SIZE:
            raise ValueError(
                f"RMSNorm input must contain the full hidden width {Llama31_8BConfig.EMB_SIZE} on every chip; "
                f"got {x.shape[-1]}"
            )
        if x.dtype != ttnn.bfloat16:
            raise ValueError(f"RMSNorm input must be bfloat16, got {x.dtype}")
        if x.layout != ttnn.TILE_LAYOUT:
            raise ValueError(f"RMSNorm input must use TILE_LAYOUT, got {x.layout}")
        if x.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
            raise ValueError(f"RMSNorm input must use interleaved DRAM, got {x.memory_config()}")

        return ttnn.rms_norm(
            x,
            weight=self.weight,
            epsilon=self.eps,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )

    def _fused_norm(self, x_shard):
        """Mistral/Kimi/GLM fused distributed RMSNorm (stats gather inside the op), as tt_distributed_rms_norm.py."""
        key = tuple(x_shard.padded_shape)
        resources = _FUSED_RESOURCES.get(key)
        if resources is None or resources["device"] is not self.mesh_device:
            grid = self.mesh_device.compute_with_storage_grid_size()
            cores = ttnn.num_cores_to_corerangeset(grid.x * grid.y, grid, row_wise=True)
            links = int(os.getenv("LOCAL_LLAMA_FNORM_LINKS", "1"))
            pairs = [
                (
                    [ttnn.create_global_semaphore(self.mesh_device, cores, 0)],
                    ttnn.experimental.dit_fused_distributed_rmsnorm_create_stats_buffer(
                        x_shard, 1, self.mesh_device, num_links=links, weight=self.sp_fused_weight
                    ),
                )
                for _ in range(2)
            ]
            ttnn.synchronize_device(self.mesh_device)
            resources = _FUSED_RESOURCES[key] = {"pairs": pairs, "next": 0, "device": self.mesh_device}
        semaphores, stats = resources["pairs"][resources["next"]]
        resources["next"] = 1 - resources["next"]
        return ttnn.experimental.dit_fused_distributed_rmsnorm(
            x_shard,
            1,
            self.mesh_device,
            semaphores,
            topology=getattr(ttnn.Topology, os.getenv("LOCAL_LLAMA_FNORM_TOPO", "Linear")),
            epsilon=self.eps,
            weight=self.sp_fused_weight,
            persistent_output_buffer=stats,
            num_preferred_links=int(os.getenv("LOCAL_LLAMA_FNORM_LINKS", "1")),
            dtype=x_shard.dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def sp_full(self, x_shard):
        """Distributed norm of a TP-sharded [1,1,256,512] residual, returned all-gathered to full width."""
        if self.sp_fused_weight is not None:
            normalized = self._fused_norm(x_shard)
        else:
            stats = ttnn.rms_norm_pre_all_gather(
                x_shard, dtype=ttnn.bfloat16, compute_kernel_config=self.compute_kernel_config
            )
            gathered = ttnn.all_gather(
                stats,
                dim=3,
                cluster_axis=1,
                num_links=1,
                topology=ttnn.Topology.Ring,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            stats.deallocate(True)
            normalized = ttnn.rms_norm_post_all_gather(
                x_shard,
                gathered,
                epsilon=self.eps,
                weight=self.sp_weight,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.compute_kernel_config,
            )
            gathered.deallocate(True)
        full = ttnn.all_gather(
            normalized,
            dim=3,
            cluster_axis=1,
            num_links=2,
            topology=ttnn.Topology.Ring,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        normalized.deallocate(True)
        return full
