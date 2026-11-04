# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Persistent zero-initialised mHC stream buffers for the valid-row expand (mhc_expand3, DSV41_MHC_EXPAND_VALID=1).

The expand output [T,1,4,D] fp32 tile tensor has data only in rows 0..3 of every tile (two 256 B pieces: face 0 at byte 0, face 1 at byte 1024);
the remaining 3.5 KB of each 4 KB tile are zero (consumers such as the head matmul need them to be exactly 0). mhc_expand3 writes ONLY the two
valid pieces, so it needs an output buffer whose padding is already zero and stays zero: this pool.

Slot = (tag, T, role). Within one decoder layer the attention-side expand (role "attn") produces x2 and the FFN-side expand (role "ffn") produces
the layer output x3; x2 is dead after the same layer's FFN expand, x3 is dead after the NEXT layer's FFN expand has started (the next layer's
attention expand reads it as its residual, nothing else holds it), so one buffer per role suffices and no parity counter is needed (the assignment is
identical in eager and traced execution). ``tag`` separates independent chains of the same T (main decoder / drafter).

Buffers are allocated on first use: call the layer once eagerly (warm-up / compile run, which every trace capture has) BEFORE begin_trace_capture so
the buffers are not allocated inside a trace. ``capturing`` can be set by a caller to make a first use during capture an error."""

import os

import torch

import ttnn

_POOL = {}
capturing = False  # set True by a caller while a trace is being captured: first use then raises instead of allocating inside the trace


def get(mesh, T, D, tag, role):
    key = (id(mesh), tag, int(T), int(D), role)
    t = _POOL.get(key)
    if (
        t is not None and not t.is_allocated()
    ):  # a caller freed the pooled stream (must not happen in a traced model): re-create, never inside a capture
        t = None
    if t is None:
        if capturing:
            raise RuntimeError(
                f"mhc_stream_pool: buffer {key} first used during trace capture; warm up the layer eagerly first"
            )
        t = ttnn.from_torch(
            torch.zeros(int(T), 1, 4, int(D)),
            device=mesh,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG
            if os.environ.get("DSV41_MHC_POOL_L1") == "1"
            else ttnn.DRAM_MEMORY_CONFIG,  # DSV41_MHC_POOL_L1=1: the residual stream buffers in L1 interleaved (persistent, 43 KB per core per buffer)
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        _POOL[key] = t
    return t


def reset(mesh=None):
    """drop the buffers (tests / mesh close)"""
    for k in [k for k in _POOL if mesh is None or k[0] == id(mesh)]:
        ttnn.deallocate(_POOL.pop(k))
