# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN rotary embedding (Llama-3 scaled RoPE) for Llama-3.1-8B-Instruct.

Mirrors the Hugging Face ``LlamaRotaryEmbedding.forward``::

    freqs = position_ids^T (x) inv_freq        # outer product
    emb   = cat(freqs, freqs)
    cos, sin = cos(emb) * scaling, sin(emb) * scaling

``inv_freq`` carries the ``llama3`` rope-scaling already applied by the
checkpoint's rope parameters; it is a small static table, so it is built once at
construction and staged to the device. The forward is pure ttnn: the outer
product is a broadcast multiply of ``position_ids`` ``[b, 1, s, 1]`` against the
duplicated frequency row ``[1, 1, 1, head_dim]``, then ``ttnn.cos`` / ``ttnn.sin``.

This is a REPLICATE-ONLY role under tensor parallelism: it is a per-position
lookup table, not a matmul, so there is no weight to split. On a mesh the table
and the result are replicated on every chip, which is exactly what the sharded
attention wants -- each chip rotates its own heads with the same angles.

The tables are kept in float32 on device: ``position_ids`` reach 10^5 while
``inv_freq`` reaches 10^-5, and rounding that product to bfloat16 before the
cosine would move the angle by far more than the cosine's own error.
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


class TtLlamaRotaryEmbedding(LightweightModule):
    def __init__(self, device, torch_module):
        super().__init__()
        self.device = device
        self.num_devices = _num_devices(device)
        self.attention_scaling = float(getattr(torch_module, "attention_scaling", 1.0))

        inv_freq = torch_module.inv_freq.detach().to(torch.float32).reshape(-1)
        self.head_dim = int(inv_freq.numel()) * 2
        # `emb = cat(freqs, freqs)` is the same outer product twice, so duplicate the
        # frequency row once here and take the product straight to head_dim width.
        half = inv_freq.numel()
        inv_freq_dup = torch.empty(1, 1, 1, self.head_dim, dtype=torch.float32)
        inv_freq_dup[0, 0, 0, :half] = inv_freq
        inv_freq_dup[0, 0, 0, half:] = inv_freq
        self.inv_freq = self._to_device(inv_freq_dup)

    def _to_device(self, host_tensor):
        """Stage a host tensor as float32/TILE. The dtype cast happens INSIDE
        ``ttnn.from_torch`` so the forward runs no torch compute of its own."""
        kwargs = dict(dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device)
        if self.num_devices > 1:
            kwargs["mesh_mapper"] = ttnn.ReplicateTensorToMesh(self.device)
        return ttnn.from_torch(host_tensor, **kwargs)

    def __call__(self, x, position_ids=None, **kwargs):
        return self.forward(x, position_ids)

    def forward(self, x, position_ids=None):
        record("rotary_embedding")  # Gate 2: proof of invocation, from INSIDE the real forward
        if position_ids is None:
            # Default to consecutive positions, matching how the model would call this.
            batch, seq_len = 1, int(x.shape[-2])
            pos = ttnn.arange(0, seq_len, 1, dtype=ttnn.float32, device=self.device)
        elif isinstance(position_ids, ttnn.Tensor):
            shape = position_ids.shape
            batch = int(shape[0]) if len(shape) >= 2 else 1
            seq_len = int(shape[-1])
            pos = position_ids
            if pos.dtype != ttnn.float32:
                pos = ttnn.typecast(pos, ttnn.float32)
        else:
            # Stage the host positions with ttnn's own marshalling call and do every
            # reshape on device -- no torch compute in the forward.
            shape = position_ids.shape
            batch = int(shape[0]) if len(shape) >= 2 else 1
            seq_len = int(shape[-1])
            pos = self._to_device(position_ids)

        # -> [b, 1, s, 1] so the frequency row broadcasts across the position axis.
        if pos.layout != ttnn.TILE_LAYOUT:
            pos = ttnn.to_layout(pos, ttnn.TILE_LAYOUT)
        pos = ttnn.reshape(pos, (batch, 1, 1, seq_len))
        pos = ttnn.transpose(pos, -2, -1)

        # Outer product: [b, 1, s, 1] * [1, 1, 1, head_dim] -> [b, 1, s, head_dim]
        emb = ttnn.multiply(pos, self.inv_freq)
        ttnn.deallocate(pos)

        cos = ttnn.cos(emb)
        sin = ttnn.sin(emb)
        ttnn.deallocate(emb)

        if self.attention_scaling != 1.0:
            cos = ttnn.multiply(cos, self.attention_scaling)
            sin = ttnn.multiply(sin, self.attention_scaling)

        # HF returns [b, s, head_dim] (broadcast over heads by the attention module).
        cos = ttnn.reshape(cos, (batch, seq_len, self.head_dim))
        sin = ttnn.reshape(sin, (batch, seq_len, self.head_dim))
        return cos, sin


def build(device, torch_module):
    """Entry point used by the per-component PCC harness."""
    return TtLlamaRotaryEmbedding(device, torch_module)
