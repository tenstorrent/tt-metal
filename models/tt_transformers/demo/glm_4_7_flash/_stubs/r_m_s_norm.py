# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN RMSNorm for GLM-4.7-Flash. Replicated (no sharding)."""
from __future__ import annotations

import torch

import ttnn


class TtRMSNorm:
    def __init__(self, device, torch_module) -> None:
        self.device = device
        sd = torch_module.state_dict()
        w = sd["weight"].float()
        self._dim = w.shape[0]
        w4d = w.reshape(1, 1, -1, 32)
        kw = dict(
            dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        try:
            if isinstance(device, ttnn.MeshDevice):
                kw["mesh_mapper"] = ttnn.ReplicateTensorToMesh(device)
        except AttributeError:
            pass
        self._w = ttnn.from_torch(w4d, **kw)
        self._eps = getattr(torch_module, "variance_epsilon", 1e-6)
        # HiFi4 + fp32 accumulation: the default (approx) config under-scales the output by ~1%.
        self._ckc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )

    @classmethod
    def build(cls, device, torch_module):
        return cls(device, torch_module)

    def __call__(self, x, **kwargs):
        if isinstance(x, torch.Tensor):
            kw = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
            try:
                if isinstance(self.device, ttnn.MeshDevice):
                    kw["mesh_mapper"] = ttnn.ReplicateTensorToMesh(self.device)
            except AttributeError:
                pass
            x = ttnn.from_torch(x.to(torch.bfloat16), **kw)
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        return ttnn.rms_norm(x, weight=self._w, epsilon=self._eps, compute_kernel_config=self._ckc)


def build(device, torch_module=None):
    return TtRMSNorm.build(device, torch_module)


def r_m_s_norm(*args, **kwargs):
    return TtRMSNorm.build(*args, **kwargs)
