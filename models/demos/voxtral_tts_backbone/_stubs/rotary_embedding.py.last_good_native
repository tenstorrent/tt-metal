# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN port of `rotary_embedding` (`MistralRotaryEmbedding`) for
`/localdev/lserbedzija/hf_models/voxtral-tts-backbone`.

    freqs    = outer(position_ids, inv_freq)       # [s, head_dim/2]
    emb      = cat(freqs, freqs, dim=-1)           # [s, head_dim]
    cos, sin = cos(emb) * scaling, sin(emb) * scaling

The plan's reuse target `models/tt_transformers/tt/rope.py::RotaryEmbedding` is
not usable here: it is a torch `nn.Module` that precomputes host-side caches and
hands them back in Meta's interleaved layout, whereas this component's golden is
HF's half-split layout — and returning torch tensors would make the forward
non-native by definition. So the table is computed on device from the same
`inv_freq` buffer the HF module holds.

Everything stays in ttnn `float32`: RoPE angles reach `position * inv_freq[0]`
(~63 rad already at seq 64, far more at real context lengths) and bfloat16
cannot carry that argument into `cos`/`sin` accurately.

`build` touches torch only to stage `inv_freq` from the checkpoint module;
`__call__` is pure ttnn — `models/common/native_probe.py` counts what actually
executes.
"""
from __future__ import annotations

import torch
import ttnn

from models.demos.voxtral_tts_backbone._stubs.attention import _replicate_mapper


class TtRotaryEmbedding:
    def __init__(self, device, inv_freq, head_dim, scaling, compute_kernel_config=None):
        self.device = device
        self.inv_freq = inv_freq  # ttnn float32, [1, 1, head_dim/2, 1]
        self.head_dim = int(head_dim)
        self.scaling = float(scaling)
        self.compute_kernel_config = compute_kernel_config

    @classmethod
    def build(cls, device, torch_module):
        if torch_module is None:
            raise RuntimeError("rotary_embedding build needs the HF MistralRotaryEmbedding to read inv_freq from")
        inv_freq = getattr(torch_module, "inv_freq", None)
        if inv_freq is None:
            raise RuntimeError("HF rotary module exposes no `inv_freq` buffer")
        host = inv_freq.detach().to(torch.float32).reshape(1, 1, -1, 1)
        head_dim = int(host.shape[-2]) * 2
        scaling = float(getattr(torch_module, "attention_scaling", 1.0) or 1.0)
        mapper = _replicate_mapper(device)
        stage_kwargs = {"mesh_mapper": mapper} if mapper is not None else {}
        staged = ttnn.from_torch(
            host,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            **stage_kwargs,
        )
        try:
            compute_kernel_config = ttnn.init_device_compute_kernel_config(
                device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )
        except Exception:  # noqa: BLE001 - accuracy tuning is best-effort
            compute_kernel_config = None
        return cls(device, staged, head_dim, scaling, compute_kernel_config)

    def __call__(self, x=None, position_ids=None, **_ignored):
        if position_ids is None:
            raise RuntimeError(
                "rotary_embedding forward needs position_ids; the harness must stage the same positions "
                "it gave the HF reference"
            )
        mm = {"compute_kernel_config": self.compute_kernel_config} if self.compute_kernel_config else {}

        # position_ids: [b, s] (or already [b, 1, 1, s]) -> [b, 1, 1, s]
        pos_shape = list(position_ids.shape)
        batch, seq = pos_shape[0], pos_shape[-1]
        positions = ttnn.reshape(position_ids, (batch, 1, 1, seq))
        if positions.dtype != self.inv_freq.dtype:
            positions = ttnn.typecast(positions, self.inv_freq.dtype)

        # [1, 1, hd/2, 1] @ [b, 1, 1, s] -> [b, 1, hd/2, s] -> [b, 1, s, hd/2]
        freqs = ttnn.matmul(self.inv_freq, positions, **mm)
        freqs = ttnn.permute(freqs, (0, 1, 3, 2))
        emb = ttnn.concat([freqs, freqs], dim=-1)

        cos = ttnn.cos(emb)
        sin = ttnn.sin(emb)
        if self.scaling != 1.0:
            cos = ttnn.multiply(cos, self.scaling)
            sin = ttnn.multiply(sin, self.scaling)
        return (
            ttnn.reshape(cos, (batch, seq, self.head_dim)),
            ttnn.reshape(sin, (batch, seq, self.head_dim)),
        )


# Module-level `build` — primary test entry point.
def build(device, torch_module=None):
    return TtRotaryEmbedding.build(device, torch_module)


# Module-level shim with the component's lowercase slug name. Kept for
# backward compatibility with legacy SMOKE/PCC tests that import the
# slug directly.
def rotary_embedding(device, torch_module=None):
    return TtRotaryEmbedding.build(device, torch_module)
