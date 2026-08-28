# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native ttnn port of `Qwen3NextRMSNormGated`.

Reference: `transformers/models/qwen3_next/modeling_qwen3_next.py::Qwen3NextRMSNormGated`:

    h = x * rsqrt(mean(x^2, -1) + eps)
    h = weight * h
    out = h * silu(gate)

Contrast with the PLAIN `Qwen3NextRMSNorm` (see `_stubs/r_m_s_norm.py`), which stores its scale as
a residual and applies `(1.0 + weight)`. This one is initialised to ONES and applied directly, so
the weight goes to the device untouched -- folding a `1.0 +` in here would be a silent 2x error.

The gate arrives from the PCC harness as a HOST torch tensor (the harness only stages the primary
arg), so `__call__` stages it with `ttnn.from_torch` exactly like `attention.py` stages its rope
tables; a caller that already holds a ttnn gate (the delta net) passes it straight through.

Tensor-parallel: the normalisation reduces over the FULL head_v_dim, so there is nothing to split
and no collective to place -- the weight is REPLICATED and each chip reproduces the golden output.
"""
from __future__ import annotations

import ttnn

from models.demos.qwen3_coder_next._stubs.gated_delta_net import (
    TILE,
    num_devices,
    replicate_mapper,
    to_device,
)


class TtQwen3NextRMSNormGated:
    """Native ttnn Qwen3-Next gated RMSNorm: `rms_norm(x) * weight * silu(gate)`."""

    def __init__(self, device, *, weight, hidden_size, eps) -> None:
        self.device = device
        self.weight = weight
        self.hidden_size = hidden_size
        self.eps = eps
        self.num_devices = num_devices(device)
        self.replicate = replicate_mapper(device, self.num_devices)
        self.compute_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    @classmethod
    def build(cls, device, torch_module):
        if torch_module is None:
            raise RuntimeError("r_m_s_norm_gated stub needs the torch reference module for its weights")

        w = torch_module.state_dict()["weight"].detach().float()
        hidden = int(w.shape[-1])
        eps = float(getattr(torch_module, "variance_epsilon", None) or getattr(torch_module, "eps", 1e-6))
        if hidden % TILE:
            raise NotImplementedError(f"r_m_s_norm_gated expects a tile-aligned head dim; got {hidden}")

        replicate = replicate_mapper(device, num_devices(device))
        return cls(
            device,
            weight=to_device(
                w.view(1, 1, hidden // TILE, TILE),
                device,
                mesh_mapper=replicate,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            ),
            hidden_size=hidden,
            eps=eps,
        )

    def _stage(self, host_tensor, seq):
        """Put a host-side gate on device, replicated, shaped (1, 1, seq, hidden)."""
        return to_device(
            host_tensor.reshape(1, 1, seq, self.hidden_size).float(),
            self.device,
            mesh_mapper=self.replicate,
        )

    def __call__(self, hidden_states, gate=None, *args, **kwargs):
        if gate is None:
            raise RuntimeError("r_m_s_norm_gated requires a `gate`; the golden dereferences it unconditionally")

        # The delta net hands this a per-head tensor `(1, heads, S, head_v_dim)`. Both the norm and
        # the gate multiply are per-row over the LAST axis, so the head axis just rides along --
        # flattening it into `seq` would be wrong the moment `S` stops being tile-aligned.
        if (
            isinstance(hidden_states, ttnn.Tensor)
            and len(hidden_states.shape) == 4
            and int(hidden_states.shape[-1]) == self.hidden_size
        ):
            h = ttnn.rms_norm(
                hidden_states,
                weight=self.weight,
                epsilon=self.eps,
                compute_kernel_config=self.compute_config,
            )
            g = gate if isinstance(gate, ttnn.Tensor) else self._stage(gate, int(hidden_states.shape[-2]))
            return ttnn.multiply(h, ttnn.silu(g))

        seq = int(hidden_states.shape[-2])
        x = ttnn.reshape(hidden_states, (1, 1, seq, self.hidden_size))
        h = ttnn.rms_norm(
            x,
            weight=self.weight,
            epsilon=self.eps,
            compute_kernel_config=self.compute_config,
        )

        g = gate if isinstance(gate, ttnn.Tensor) else self._stage(gate, seq)
        g = ttnn.reshape(g, (1, 1, seq, self.hidden_size))
        out = ttnn.multiply(h, ttnn.silu(g))
        return ttnn.reshape(out, (1, seq, self.hidden_size))


def build(device, torch_module=None):
    return TtQwen3NextRMSNormGated.build(device, torch_module)


def r_m_s_norm_gated(device, torch_module=None):
    return TtQwen3NextRMSNormGated.build(device, torch_module)
