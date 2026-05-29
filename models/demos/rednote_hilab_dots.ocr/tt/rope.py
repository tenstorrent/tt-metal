# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""TTNN implementation of the dots.ocr Qwen2 language-model rotary embedding.

Reference: models/demos/rednote_hilab_dots.ocr/reference/functional.py
           :func:`rope_forward`

The Qwen2RotaryEmbedding builds the rotary cos/sin tables from an inverse
frequency vector and the integer position ids::

    inv_freq = 1.0 / (rope_theta ** (arange(0, head_dim, 2) / head_dim))
    freqs    = outer(position_ids, inv_freq)          # [seq, head_dim/2]
    emb      = cat(freqs, freqs, dim=-1)              # [seq, head_dim]
    cos, sin = emb.cos(), emb.sin()                    # [seq, head_dim]

For the dots.ocr LM ``rope_theta = 1e6`` and ``head_dim = 128`` (default rope
type, attention_scaling = 1.0).  The reference computes ``freqs`` in fp32 then
casts to the activation dtype.

The ``inv_freq`` vector depends only on ``head_dim`` / ``rope_theta`` (not on the
runtime positions), so it is materialised once on the host in ``__init__`` (a
parameter-style precompute, exactly like the weight tables of the norm blocks).
The forward path runs entirely on the device: it forms the outer product of the
(device-resident) position ids with ``inv_freq`` via a broadcasting multiply,
concatenates the two halves, and applies the elementwise cos / sin.  The
positions and inv_freq are kept in fp32 so the table generation matches the
reference float-then-cast path.
"""
import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


class TtRoPE(LightweightModule):
    """dots Qwen2 LM rotary cos/sin table generator.

    Produces the ``(cos, sin)`` tables of shape ``[batch, seq_len, head_dim]``
    consumed by the LM attention.  The application to Q/K (rotate_half) lives in
    the attention block; this module owns only the table generation that the
    ``rope`` golden validates.

    Args:
        device: ttnn Device or MeshDevice.
        head_dim: per-head dimension (128).
        rope_theta: rotary base (1e6 for the dots.ocr LM).
        attention_scaling: default-rope scaling factor (1.0).
        inv_freq_dtype: dtype the precomputed inv_freq table is stored in.
    """

    def __init__(
        self,
        device,
        head_dim: int = 128,
        rope_theta: float = 1000000.0,
        attention_scaling: float = 1.0,
        inv_freq_dtype=ttnn.float32,
        inv_freq_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ):
        super().__init__()
        self.device = device
        self.head_dim = head_dim
        self.attention_scaling = attention_scaling

        # inv_freq depends only on head_dim / rope_theta -> precompute on host.
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))
        # Shape [1, 1, 1, head_dim/2] so it broadcasts against [b, 1, seq, 1] positions.
        inv_freq = inv_freq.reshape(1, 1, 1, head_dim // 2)

        is_mesh_device = device.__class__.__name__ == "MeshDevice"
        self.inv_freq = ttnn.from_torch(
            inv_freq,
            device=device,
            dtype=inv_freq_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=inv_freq_memory_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
        )

    def forward(self, position_ids: ttnn.Tensor) -> tuple:
        """Build the rotary cos/sin tables for the given positions.

        Args:
            position_ids: device tensor of shape ``[batch, 1, seq_len, 1]``
                (float) holding the integer positions, broadcastable against the
                ``[1, 1, 1, head_dim/2]`` inverse-frequency table.

        Returns:
            ``(cos, sin)`` device tensors, each ``[batch, 1, seq_len, head_dim]``.
        """
        # Outer product positions x inv_freq via broadcast multiply -> [b, 1, seq, head_dim/2].
        # fp32 inputs keep the product in fp32 to match the reference float-then-cast path.
        # The whole table-generation chain is pinned to L1: tracy showed every op
        # (multiply, concat, cos, sin) coalescing its output back to DRAM-interleaved by
        # default, so each downstream op re-reads the freqs from DRAM. The tables are a few
        # hundred KB at fp32 and split cleanly across the core grid, so keeping them L1
        # resident lets the cos/sin reads stay on-chip. PCC is unaffected (same math).
        l1 = ttnn.L1_MEMORY_CONFIG
        freqs = ttnn.multiply(position_ids, self.inv_freq, dtype=ttnn.float32, memory_config=l1)

        # emb = cat(freqs, freqs) duplicates the half-rotation across the head dim.
        emb = ttnn.concat([freqs, freqs], dim=-1, memory_config=l1)

        cos = ttnn.cos(emb, memory_config=l1)
        sin = ttnn.sin(emb, memory_config=l1)

        if self.attention_scaling != 1.0:
            cos = ttnn.multiply(cos, self.attention_scaling, memory_config=l1)
            sin = ttnn.multiply(sin, self.attention_scaling, memory_config=l1)

        return cos, sin
