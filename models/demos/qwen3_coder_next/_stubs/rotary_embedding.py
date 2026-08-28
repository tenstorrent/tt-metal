# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native ttnn port of `Qwen3NextRotaryEmbedding`.

Reference: `transformers/models/qwen3_next/modeling_qwen3_next.py::Qwen3NextRotaryEmbedding`:

    freqs = outer(position_ids, inv_freq)          # (B, S, dim/2)
    emb   = cat(freqs, freqs, dim=-1)              # (B, S, dim)
    cos, sin = emb.cos() * scaling, emb.sin() * scaling

`inv_freq` and `attention_scaling` are read off the reference module rather than re-derived from
the config: this is the "default" rope type here, but reading them keeps the port correct if the
checkpoint ever switches to a scaled variant (yarn/linear), where HF folds the scaling into both.

PRECISION: this is the one op in the model that must NOT run in bf16. `freqs` reaches
`(S-1) * inv_freq[0]` radians -- tens of radians at the sequence lengths this harness exercises --
and bf16's 8-bit mantissa quantises an angle of ~63 rad to ~0.25 rad, which is a quarter-radian
phase error straight into `cos`/`sin`. The whole table is therefore built and evaluated in
float32; only the consumer (attention) casts down, after the transcendentals are done.

Tensor-parallel: rope is a per-position table with no model-dim reduction and no weights to split.
`inv_freq` is REPLICATED and every chip computes the identical table -- no collective.
"""
from __future__ import annotations

import ttnn

from models.demos.qwen3_coder_next._stubs.gated_delta_net import (
    num_devices,
    replicate_mapper,
    to_device,
)


class TtQwen3NextRotaryEmbedding:
    """Native ttnn rope table generator: `(cos, sin)` for a batch of position ids."""

    def __init__(self, device, *, inv_freq, rotary_dim, attention_scaling) -> None:
        self.device = device
        self.inv_freq = inv_freq
        self.rotary_dim = rotary_dim
        self.attention_scaling = attention_scaling
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
            raise RuntimeError("rotary_embedding stub needs the torch reference module for its inv_freq table")

        inv_freq = torch_module.inv_freq.detach().float().flatten()
        half = int(inv_freq.shape[0])
        scaling = float(getattr(torch_module, "attention_scaling", 1.0))

        replicate = replicate_mapper(device, num_devices(device))
        return cls(
            device,
            # Row vector (1, 1, 1, dim/2): the right operand of the outer product below.
            inv_freq=to_device(
                inv_freq.view(1, 1, 1, half),
                device,
                mesh_mapper=replicate,
                dtype=ttnn.float32,
            ),
            rotary_dim=2 * half,
            attention_scaling=scaling,
        )

    def __call__(self, x, position_ids=None, *args, **kwargs):
        if position_ids is None:
            raise RuntimeError("rotary_embedding requires `position_ids`; the golden has no default table")

        seq = int(position_ids.shape[-1])
        # Column vector (1, 1, S, 1) x row vector (1, 1, 1, dim/2) -> outer product (1, 1, S, dim/2),
        # which is exactly HF's `inv_freq_expanded @ position_ids_expanded` after its transpose.
        # In the real pipeline the position ids are already a resident device buffer; the host
        # branch is the per-component PCC harness, which hands over the golden's torch tensor.
        if isinstance(position_ids, ttnn.Tensor):
            pos = ttnn.typecast(ttnn.reshape(position_ids, (1, 1, seq, 1)), ttnn.float32)
        else:
            pos = to_device(
                position_ids.reshape(1, 1, seq, 1).float(),
                self.device,
                mesh_mapper=self.replicate,
                dtype=ttnn.float32,
            )
        freqs = ttnn.matmul(pos, self.inv_freq, compute_kernel_config=self.compute_config)
        emb = ttnn.concat([freqs, freqs], dim=-1)

        cos = ttnn.cos(emb)
        sin = ttnn.sin(emb)
        if self.attention_scaling != 1.0:
            cos = ttnn.multiply(cos, self.attention_scaling)
            sin = ttnn.multiply(sin, self.attention_scaling)

        shape = (1, seq, self.rotary_dim)
        return ttnn.reshape(cos, shape), ttnn.reshape(sin, shape)


def build(device, torch_module=None):
    return TtQwen3NextRotaryEmbedding.build(device, torch_module)


def rotary_embedding(device, torch_module=None):
    return TtQwen3NextRotaryEmbedding.build(device, torch_module)
