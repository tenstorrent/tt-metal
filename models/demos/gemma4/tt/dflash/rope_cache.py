# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""On-device RoPE cos/sin for the DFlash drafter, replacing generate.py's former host-torch
per-iteration trig computation (build_noise_inputs). Mirrors the target model's own
``create_rope_caches`` 2D/embedding-gather pattern (models/demos/gemma4/tt/model.py) for
the gather side; the table itself is now ALSO built entirely on device (arange, outer,
exp/log for scalar**tensor, cos, sin) -- no host torch computation or upload at all, not
even the one-time setup step.

The drafter needs its OWN cache, not the target's: its rope_theta (from the DFlash
checkpoint's own config) is independent of whatever the target model's layers use.
"""

from __future__ import annotations

import math

import torch

import ttnn


def build_dflash_rope_cache_2d(mesh_device, head_dim: int, rope_theta: float, max_seq_len: int):
    """Returns (cos_2d, sin_2d): ROW_MAJOR bf16 [max_seq_len, head_dim] ttnn tensors,
    replicated across the mesh. Same formula as the reference's ``Qwen3RotaryEmbedding``
    (``1/theta**(arange(0,head_dim,2)/head_dim)``, outer with positions, cos/sin), built
    with on-device ttnn ops throughout. ``rope_theta**exponent`` has no direct ttnn op for
    "scalar base, tensor exponent", so it's computed as ``exp(exponent * log(rope_theta))``
    -- ``log(rope_theta)`` is a plain Python float constant, not a tensor operation.

    Confirmed against the previous host-torch implementation: cos bit-identical (bf16,
    PCC 1.0); sin PCC 1.0 with 4/16384 values (at max_seq_len=128) differing by ~2e-6,
    right at sin's near-zero crossings -- an ordinary host-libm-vs-device-kernel ULP
    difference, not a correctness issue."""
    log_theta = math.log(rope_theta)

    exponent = ttnn.arange(0, head_dim, 2, device=mesh_device, dtype=ttnn.float32)  # [half]
    exponent = ttnn.multiply(exponent, 1.0 / head_dim)
    theta_pow = ttnn.exp(ttnn.multiply(exponent, log_theta))  # rope_theta ** exponent
    inv_freq = ttnn.reciprocal(theta_pow)  # [half]

    positions = ttnn.arange(0, max_seq_len, 1, device=mesh_device, dtype=ttnn.float32)  # [max_seq_len]
    freqs = ttnn.outer(positions, inv_freq)  # [max_seq_len, half]
    emb = ttnn.concat([freqs, freqs], dim=-1)  # [max_seq_len, head_dim]
    cos = ttnn.typecast(ttnn.cos(emb), ttnn.bfloat16)
    sin = ttnn.typecast(ttnn.sin(emb), ttnn.bfloat16)

    # ttnn.embedding (the gather side, below) needs its weight table in ROW_MAJOR.
    cos_2d = ttnn.to_layout(cos, ttnn.ROW_MAJOR_LAYOUT)
    sin_2d = ttnn.to_layout(sin, ttnn.ROW_MAJOR_LAYOUT)
    return cos_2d, sin_2d


def gather_rope_on_device(mesh_device, positions: list[int], cos_2d, sin_2d, head_dim: int):
    """Gather cos/sin rows for ``positions`` (a contiguous absolute-position range) via
    ttnn.embedding, entirely on device -- no host trig VALUES computed, only a small
    integer index tensor built on host and freshly allocated on device.

    Used for prefill's one-off (larger, non-repeating) gather, where a fresh allocation
    doesn't matter -- it happens exactly once per generation session. The steady-state
    per-iteration gather uses ``gather_rope_on_device_buffered`` below instead, which
    reuses one persistent index buffer.

    Returns (cos_tt, sin_tt): [1,1,len(positions),head_dim] bf16, matching what
    dflash_attention_forward expects for cos_full/sin_full."""
    is_mesh = hasattr(mesh_device, "shape")
    replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None

    idx = torch.tensor(positions, dtype=torch.int32).reshape(1, -1)
    idx_tt = ttnn.from_torch(
        idx, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32, mesh_mapper=replicate
    )

    cos = ttnn.embedding(idx_tt, cos_2d, layout=ttnn.TILE_LAYOUT)
    sin = ttnn.embedding(idx_tt, sin_2d, layout=ttnn.TILE_LAYOUT)
    ttnn.deallocate(idx_tt)
    return ttnn.unsqueeze_to_4D(cos), ttnn.unsqueeze_to_4D(sin)


def make_rope_gather_index_buffer(mesh_device, size: int) -> ttnn.Tensor:
    """Persistent [1,size] uint32 index buffer for gather_rope_on_device_buffered, built
    once per generation session and refreshed in place via
    ``ttnn.copy_host_to_device_tensor`` every call -- not a fresh device allocation each
    iteration (see verify.py's module docstring for why that matters for eventual trace
    capture). ``size`` must be large enough to cover every steady-state call's
    ``len(positions)``; DFlash's own steady state (every generation iteration after the
    first) never needs more than ``2 * block_size`` (context_len maxes out at block_size
    once past the initial prefill-seeded block, per generate.py's sliding-context
    mechanism)."""
    is_mesh = hasattr(mesh_device, "shape")
    mapper = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None
    return ttnn.from_torch(
        torch.zeros((1, size), dtype=torch.int64),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
        mesh_mapper=mapper,
    )


def refresh_rope_gather_buffer(mesh_device, idx_buffer: ttnn.Tensor, positions: list[int]) -> None:
    """Refresh ``idx_buffer`` in place (``copy_host_to_device_tensor``) with ``positions``
    -- the host-side half of the buffered gather, split out from the device-side half
    (``gather_rope_from_buffer`` below) so a trace can capture ONLY the gather (a pure
    device op reading from the now-current buffer contents) while this refresh runs
    eagerly before each replay, exactly like ``verify.py``'s
    ``refresh_verify_positions``/``run_verify_forward`` split. ``len(positions)`` must be
    <= ``idx_buffer.shape[-1]``; padding slots beyond the real positions are filled by
    repeating the last real position (harmless -- see ``gather_rope_from_buffer``)."""
    max_size = idx_buffer.shape[-1]
    n = len(positions)
    assert n <= max_size, f"{n} positions don't fit in a size-{max_size} rope gather buffer"
    is_mesh = hasattr(mesh_device, "shape")
    mapper = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None

    padded = positions + [positions[-1]] * (max_size - n)
    idx = torch.tensor(padded, dtype=torch.int64).reshape(1, max_size)
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(idx, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32, mesh_mapper=mapper), idx_buffer
    )


def gather_rope_from_buffer(idx_buffer: ttnn.Tensor, cos_2d, sin_2d, head_dim: int, n: int | None = None):
    """Pure on-device gather from ``idx_buffer``'s CURRENT contents -- no host tensor
    construction, no ``copy_host_to_device_tensor``, safe to capture inside a Metal
    trace. Always gathers at the buffer's fixed full size; pass ``n < idx_buffer.shape[-1]``
    to additionally slice down to just the first ``n`` (real) rows -- DFlash's own
    steady-state usage never needs this (context+noise positions always exactly fill the
    buffer), so the default (``n=None``, no slicing) is the trace-safe fixed-shape path."""
    max_size = idx_buffer.shape[-1]
    cos = ttnn.unsqueeze_to_4D(ttnn.embedding(idx_buffer, cos_2d, layout=ttnn.TILE_LAYOUT))
    sin = ttnn.unsqueeze_to_4D(ttnn.embedding(idx_buffer, sin_2d, layout=ttnn.TILE_LAYOUT))
    if n is not None and n != max_size:
        cos = ttnn.slice(cos, [0, 0, 0, 0], [1, 1, n, head_dim])
        sin = ttnn.slice(sin, [0, 0, 0, 0], [1, 1, n, head_dim])
    return cos, sin


def gather_rope_on_device_buffered(
    mesh_device, idx_buffer: ttnn.Tensor, positions: list[int], cos_2d, sin_2d, head_dim: int
):
    """Convenience wrapper combining ``refresh_rope_gather_buffer`` + ``gather_rope_from_buffer``
    for eager (non-traced) callers -- same result as before this function was split."""
    refresh_rope_gather_buffer(mesh_device, idx_buffer, positions)
    return gather_rope_from_buffer(idx_buffer, cos_2d, sin_2d, head_dim, n=len(positions))
