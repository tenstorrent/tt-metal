# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN port of `r_m_s_norm` (Qwen3RMSNorm) for FLUX.2-klein-9B's text encoder.

Bound to `model.layers.0.self_attn.q_norm`: Qwen3's per-head query norm, so it
normalizes over the trailing head_dim (128) axis, not the model dim.

    x * rsqrt(mean(x^2) + eps) * gamma

`ttnn.rms_norm` is exactly that op, so the whole component is one call. The only
real work is gamma's layout: `ttnn.rms_norm` wants it as `[1, 1, dim // 32, 32]`
in ROW_MAJOR (see `models/common/rmsnorm.py`), not as the flat `[dim]` vector
torch stores.

A norm has no matmul weight to split — gamma is per-element over the normalized
axis — so under any TP scheme it stays REPLICATED and every chip computes the
same answer from the same input. That is why this component is single-phase.

The canonical `models/common/rmsnorm.py::RMSNorm` was not reusable directly: its
`__init__` takes no `mesh_device` and expects a `ModelArgs`-shaped configuration
plus a state-dict prefix, none of which the per-component PCC harness has (it
hands the stub a bare device plus the torch module). This is that class's math,
expressed against the harness's inputs.
"""
from __future__ import annotations

import torch

import ttnn


class TtRMSNorm:
    def __init__(self, device, gamma, dim, norm_eps) -> None:
        self.device = device
        self.gamma = gamma
        self.dim = dim
        self.norm_eps = norm_eps
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    # ---------------------------------------------------------------- build

    @classmethod
    def build(cls, device, torch_module):
        if torch_module is None:
            raise RuntimeError("r_m_s_norm stub needs the torch module to source its weights")

        weight = torch_module.weight.detach()
        dim = int(weight.shape[-1])
        if dim % ttnn.TILE_SIZE:
            raise RuntimeError(f"rms_norm gamma must be a whole number of tiles wide, got dim={dim}")

        return cls(
            device=device,
            gamma=_norm_gamma(weight, dim, device),
            dim=dim,
            norm_eps=torch_module.variance_epsilon,
        )

    # -------------------------------------------------------------- forward

    def __call__(self, hidden_states, *args, **kwargs):
        in_shape = list(hidden_states.shape)

        # `ttnn.rms_norm` normalizes the LAST axis, so a rank-4 tensor needs no
        # reshuffling at all: folding its leading dims would compute the same
        # numbers but would corrupt the decode layout [1, batch, heads, head_dim],
        # where the "rows" axis is heads and folding it changes the element count.
        # Only rank != 4 inputs are folded into the [rows, 1, seq, dim] form.
        if len(in_shape) == 4:
            return ttnn.rms_norm(
                hidden_states,
                epsilon=self.norm_eps,
                weight=self.gamma,
                compute_kernel_config=self.compute_kernel_config,
            )

        seq_len = int(in_shape[-2])
        rows = 1
        for d in in_shape[:-2]:
            rows *= int(d)
        x = ttnn.reshape(hidden_states, (rows, 1, seq_len, self.dim))

        out = ttnn.rms_norm(
            x,
            epsilon=self.norm_eps,
            weight=self.gamma,
            compute_kernel_config=self.compute_kernel_config,
        )
        return ttnn.reshape(out, tuple(in_shape))


# ------------------------------------------------------------------ helpers


def _num_devices(device):
    try:
        return int(device.get_num_devices())
    except AttributeError:
        return 1


def _replicate_mapper(device):
    if _num_devices(device) <= 1:
        return None
    return ttnn.ReplicateTensorToMesh(device)


def _norm_gamma(weight, dim, device):
    """ttnn.rms_norm wants gamma as [1, 1, dim // TILE, TILE] in ROW_MAJOR
    (see models/common/rmsnorm.py). Per-element over the normalized axis, so it
    is REPLICATED on every chip."""
    return ttnn.from_torch(
        weight.to(torch.bfloat16).reshape(1, 1, dim // ttnn.TILE_SIZE, ttnn.TILE_SIZE),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=_replicate_mapper(device),
    )


# Module-level `build` — primary test entry point.
def build(device, torch_module=None):
    return TtRMSNorm.build(device, torch_module)


# Module-level shim with the component's lowercase slug name, for legacy SMOKE/PCC tests.
def r_m_s_norm(device, torch_module=None):
    return TtRMSNorm.build(device, torch_module)
