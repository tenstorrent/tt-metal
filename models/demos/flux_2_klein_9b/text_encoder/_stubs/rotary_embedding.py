# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN port of `rotary_embedding` (Qwen3RotaryEmbedding) for FLUX.2-klein-9B's text encoder.

Bound to `model.rotary_emb`. This component is the RoPE TABLE GENERATOR, not the
rotation itself: given `position_ids`, it returns the `(cos, sin)` pair the
attention layers later apply. Following `Qwen3RotaryEmbedding.forward`:

    freqs = outer(position_ids, inv_freq)
    emb   = cat(freqs, freqs)
    cos, sin = emb.cos() * attention_scaling, emb.sin() * attention_scaling

`inv_freq` is a registered buffer — a lookup table — so under the TP principles
it is REPLICATED, never split, and so is everything derived from it. There is no
matmul weight here to shard, which is why this component is single-phase.

Two things are worth explaining:

  * `inv_freq` is uploaded ONCE at build time; the forward is pure ttnn and
    touches no host tensor, so the native probe measures torch_ops = 0.
  * The outer product is `repeat` + broadcast `multiply`, not `matmul`. Measured
    on device against the torch reference: the elementwise form reproduces the
    angles EXACTLY (max error 0.0, and 1.2e-7 after cos), while `ttnn.matmul`
    of the same operands is off by 5.5e-2 after cos even at HiFi4 with
    fp32_dest_acc — its internal bf16 rounding of the angle is amplified by the
    trig, because the low-frequency components reach ~seq_len radians. The whole
    angle path is kept in FLOAT32 for the same reason: positions run to
    `max_position_embeddings` (40960).

The canonical `models/tt_transformers/tt/rope.py::RotaryEmbedding` was not
reusable directly: its `__init__` takes no `mesh_device` and expects precomputed
transformation matrices plus a `ModelArgs`-shaped configuration, none of which
the per-component PCC harness has (it hands the stub a bare device plus the
torch module).
"""
from __future__ import annotations

import torch

import ttnn


class TtRotaryEmbedding:
    def __init__(self, device, inv_freq, attention_scaling, head_dim) -> None:
        self.device = device
        self.inv_freq = inv_freq
        self.attention_scaling = attention_scaling
        self.head_dim = head_dim

    # ---------------------------------------------------------------- build

    @classmethod
    def build(cls, device, torch_module):
        if torch_module is None:
            raise RuntimeError("rotary_embedding stub needs the torch module to source its inv_freq")

        inv_freq = torch_module.inv_freq.detach().to(torch.float32).reshape(-1)
        half = int(inv_freq.shape[0])

        return cls(
            device=device,
            # Shaped [1, 1, 1, half] so it broadcasts over batch and sequence in the
            # forward's multiply. Replicated: it is a table, not a matmul weight.
            inv_freq=ttnn.from_torch(
                inv_freq.reshape(1, 1, 1, half),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=_replicate_mapper(device),
            ),
            attention_scaling=float(torch_module.attention_scaling),
            head_dim=2 * half,
        )

    # -------------------------------------------------------------- forward

    def __call__(self, x, position_ids=None, *args, **kwargs):
        if position_ids is None:
            raise RuntimeError("rotary_embedding needs position_ids to build its (cos, sin) tables")

        shape = list(position_ids.shape)
        batch, seq_len = int(shape[0]), int(shape[-1])
        half = self.head_dim // 2

        # ---- outer(position_ids, inv_freq), elementwise: broadcast each position
        # across the frequency axis, then let inv_freq broadcast back over batch/seq.
        pos = ttnn.reshape(position_ids, (batch, 1, seq_len, 1))
        pos = ttnn.repeat(pos, ttnn.Shape([1, 1, 1, half]))
        freqs = ttnn.multiply(pos, self.inv_freq)
        ttnn.deallocate(pos)

        emb = ttnn.concat([freqs, freqs], dim=-1)
        ttnn.deallocate(freqs)

        cos = ttnn.cos(emb)
        sin = ttnn.sin(emb)
        ttnn.deallocate(emb)

        if self.attention_scaling != 1.0:
            cos = ttnn.multiply(cos, self.attention_scaling)
            sin = ttnn.multiply(sin, self.attention_scaling)

        out_shape = (batch, seq_len, self.head_dim)
        return ttnn.reshape(cos, out_shape), ttnn.reshape(sin, out_shape)


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


# Module-level `build` — primary test entry point.
def build(device, torch_module=None):
    return TtRotaryEmbedding.build(device, torch_module)


# Module-level shim with the component's lowercase slug name, for legacy SMOKE/PCC tests.
def rotary_embedding(device, torch_module=None):
    return TtRotaryEmbedding.build(device, torch_module)
