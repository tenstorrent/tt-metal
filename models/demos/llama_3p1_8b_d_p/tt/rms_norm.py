# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Plain Llama-3.1 RMSNorm for sequence-parallel Galaxy prefill."""

import math
from numbers import Real

import torch

import ttnn
from models.demos.llama_3p1_8b_d_p.reference.llama_3p1_8b_config import Llama31_8BConfig


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
