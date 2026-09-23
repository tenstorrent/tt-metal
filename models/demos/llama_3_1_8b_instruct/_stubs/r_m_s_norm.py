# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN RMSNorm for Llama-3.1-8B-Instruct.

Mirrors the Hugging Face ``LlamaRMSNorm``::

    weight * x * rsqrt(mean(x^2, dim=-1) + eps)

The scale is a per-element vector over the model dim, so this is a
replicate-only role in every tensor-parallel scheme: on a mesh the weight is
replicated and each chip normalises its own copy of the (replicated)
activation. There is no collective, and no axis it could usefully split on --
a width-sharded activation would have to be all-gathered to compute the
variance, which costs more than the norm itself.
"""

from __future__ import annotations

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.llama_3_1_8b_instruct.tt._invocation import record


def _num_devices(device) -> int:
    try:
        return int(device.get_num_devices())
    except AttributeError:
        return 1


def _compute_kernel_config(device):
    """fp32 accumulate: the variance reduction over the model dim is the accuracy-critical step."""
    try:
        return ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
    except Exception:
        return ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )


class TtLlamaRMSNorm(LightweightModule):
    def __init__(self, device, torch_module):
        super().__init__()
        self.device = device
        self.num_devices = _num_devices(device)
        self.eps = float(getattr(torch_module, "variance_epsilon", 1e-5))
        self.compute_kernel_config = _compute_kernel_config(device)

        host = torch_module.weight.detach().to(torch.float32).reshape(1, -1).contiguous().to(torch.bfloat16)
        kwargs = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        if self.num_devices > 1:
            kwargs["mesh_mapper"] = ttnn.ReplicateTensorToMesh(device)
        self.weight = ttnn.from_torch(host, **kwargs)

    def __call__(self, hidden_states, **kwargs):
        return self.forward(hidden_states)

    def forward(self, hidden_states):
        record("r_m_s_norm")  # Gate 2: proof of invocation, recorded from INSIDE the real forward
        x = hidden_states
        in_rank = len(x.shape)
        batch = int(x.shape[0]) if in_rank >= 3 else 1
        seq_len = int(x.shape[-2])
        hidden = int(x.shape[-1])
        if in_rank == 3:
            x = ttnn.reshape(x, (batch, 1, seq_len, hidden))

        out = ttnn.rms_norm(
            x,
            epsilon=self.eps,
            weight=self.weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )

        if in_rank == 3:
            out = ttnn.reshape(out, (batch, seq_len, hidden))
        return out


def build(device, torch_module):
    """Entry point used by the per-component PCC harness."""
    return TtLlamaRMSNorm(device, torch_module)
