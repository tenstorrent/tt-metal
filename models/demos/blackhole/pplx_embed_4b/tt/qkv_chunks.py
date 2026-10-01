# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""QKV projection + fused heads op in batch chunks (QWEN_QKV_CHUNKS=2 or 4, bs32 at ISL 512; 4 is the default).

At bs32 the QKV output (892 KB per core) cannot live in L1, so the heads op streams it from DRAM at the op's DRAM
floor. Run in two half-batch chunks, each chunk's QKV output (446 KB per core) fits in L1 like bs16's and the heads op
reads it from L1 with the v3 compute; each chunk writes its Q / K / V into full-batch tensors at a batch offset, so
SDPA sees one bs32 tensor. Four quarter-batch chunks (223 KB per core) also leave room for K / V in L1.

The chunks come from the previous layer's post-MLP fused add+RMSNorm, which writes its normalised output as that many
batch-chunk tensors (tt/decoder_fusion.py). The decoder still reshapes the attention input and hands it to the
attention module as one tensor, so the next layer's ``attention_norm`` returns a never-written DRAM stand-in of the
full shape instead, registered here against the halves; the attention (tt/attention.py) looks its input up and, if it
is a stand-in, runs the chunks inside its QKV matmul call.
"""
import os

import ttnn

# stand-in buffer address -> (stand-in tensor, (chunk tensors, in batch order))
_STANDINS = {}


def chunks() -> int:
    return int(os.getenv("QWEN_QKV_CHUNKS", "1") or 1)


def register(halves, shape, dtype, device) -> ttnn.Tensor:
    """A DRAM tensor of the full normalised shape standing in for ``halves`` (allocation only, never written)."""
    standin = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape), dtype, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    _STANDINS[standin.buffer_address()] = (standin, tuple(halves))
    return standin


def take(x: ttnn.Tensor):
    """The halves ``x`` stands in for (``x`` may be a reshaped view of the stand-in), or None."""
    try:
        hit = _STANDINS.pop(x.buffer_address(), None)
    except Exception:  # noqa: BLE001  (no device buffer: not a stand-in)
        return None
    return None if hit is None else hit[1]


def clear():
    """Drop stand-ins a forward left behind (a layer that never consumed its input's halves)."""
    for standin, halves in _STANDINS.values():
        for t in halves:
            ttnn.deallocate(t)
    _STANDINS.clear()
