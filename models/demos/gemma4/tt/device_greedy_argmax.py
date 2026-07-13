# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared on-device greedy argmax helpers for Gemma4 decode paths.

Used by speculative fused decode and plain fused greedy decode. The multicore
``ttnn.argmax`` path requires ROW_MAJOR input with a tile-aligned (32) row dim;
otherwise it returns garbage. See ``probe_device_argmax`` / ``argmax_last``.
"""

from __future__ import annotations

import torch

import ttnn

_device_argmax_ok: bool | None = None


def probe_device_argmax(mesh_device, mapper) -> bool:
    """One-time correctness probe for multicore untilize+argmax (cached)."""
    global _device_argmax_ok
    if _device_argmax_ok is not None:
        return _device_argmax_ok
    try:
        torch.manual_seed(0)
        rows, vocab = 4, 8192
        th = torch.randn(1, 1, rows, vocab, dtype=torch.float32)
        ref = torch.argmax(th, dim=-1).to(torch.int32)

        tt = ttnn.from_torch(
            th,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=mapper,
        )
        r32 = 32
        padded = ttnn.pad(tt, [(0, 0), (0, 0), (0, r32 - rows), (0, 0)], value=0.0)
        tt.deallocate(True)
        u = ttnn.untilize(padded, use_multicore=True)
        padded.deallocate(True)
        idx = ttnn.argmax(u, dim=-1, keepdim=False)
        u.deallocate(True)
        idx = ttnn.slice(idx, [0, 0, 0], [1, 1, rows])

        if hasattr(mesh_device, "shape"):
            got = ttnn.to_torch(ttnn.get_device_tensors(idx)[0]).reshape(-1)[:rows].to(torch.int32)
        else:
            got = ttnn.to_torch(idx).reshape(-1)[:rows].to(torch.int32)
        idx.deallocate(True)
        _device_argmax_ok = bool(torch.equal(got, ref.reshape(-1)))
    except Exception:
        _device_argmax_ok = False
    return _device_argmax_ok


def reset_device_argmax_probe() -> None:
    """Test helper — clear the cached probe result."""
    global _device_argmax_ok
    _device_argmax_ok = None


def logits_to_host(logits, *, tp: int, vocab_size: int | None = None):
    """Read logits to a host torch tensor (device-0 replica under TP)."""
    if tp > 1:
        t = ttnn.to_torch(ttnn.get_device_tensors(logits)[0])
    else:
        t = ttnn.to_torch(logits)
    if vocab_size is not None:
        return t[..., :vocab_size]
    return t


def argmax_last_host(logits, rows, *, mesh_device, mapper, tp: int, vocab_size: int):
    """Host argmax — correct on all builds; fallback when device probe fails."""
    lh = logits_to_host(logits, tp=tp, vocab_size=vocab_size).reshape(rows, -1)
    ids = torch.argmax(lh, dim=-1).to(torch.int32).reshape(1, 1, rows)
    return ttnn.from_torch(
        ids,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
        mesh_mapper=mapper,
    )


def argmax_last(logits, rows, *, mesh_device, mapper, tp: int, vocab_size: int):
    """Argmax over the last (vocab) dim. Returns ``[1,1,rows]`` uint32 ROW_MAJOR.

    Pads rows to 32 for multicore untilize+argmax when the device probe passed;
    otherwise falls back to host argmax.
    """
    if not probe_device_argmax(mesh_device, mapper):
        return argmax_last_host(logits, rows, mesh_device=mesh_device, mapper=mapper, tp=tp, vocab_size=vocab_size)

    r32 = 32
    pad_rows = rows < r32
    src = logits
    padded = None
    if pad_rows:
        padded = ttnn.pad(logits, [(0, 0), (0, 0), (0, r32 - rows), (0, 0)], value=0.0)
        src = padded
    u = ttnn.untilize(src, use_multicore=True)
    if padded is not None:
        padded.deallocate(True)
    idx = ttnn.argmax(u, dim=-1, keepdim=False)
    u.deallocate(True)
    if pad_rows:
        sliced = ttnn.slice(idx, [0, 0, 0], [1, 1, rows])
        idx.deallocate(True)
        idx = sliced
    return idx


def id_to_host(id_tt, *, tp: int, sanitize) -> int:
    """``[*,1]`` uint32 device id → python int (TP: device-0 replica)."""
    t = ttnn.to_torch(ttnn.get_device_tensors(id_tt)[0]) if tp > 1 else ttnn.to_torch(id_tt)
    return sanitize(t.reshape(-1)[0])


def ids_to_host(ids_tt, n: int, *, tp: int, sanitize) -> list[int]:
    """``[1,1,n]`` uint32 device ids → list[int]."""
    t = ttnn.to_torch(ttnn.get_device_tensors(ids_tt)[0]) if tp > 1 else ttnn.to_torch(ids_tt)
    flat = t.reshape(-1)
    return [sanitize(flat[j]) for j in range(n)]
