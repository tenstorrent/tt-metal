# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DEVICE check (a free 8x4 Blackhole Galaxy, FABRIC_2D as the prefill runner; never on a host whose chips a stack holds):
``V41_PREFILL_RESTORE_GATHER`` restores the same rows as the replicated uploads (tt-blaze DS41F-0161).

The real ``V41PrefixReuse._gather_rows`` / ``_upload_rows`` / ``_upload`` on stand-in states with the serving geometry (chunk
4096; layers 2 / 8 / 14 at ratio 2 -> 2048 entry rows, layer 20 KV-only at ratio 1 -> 4096; entries [n, 512] and index keys
[n, 128], bf16 caches): a chunk's random rows are restored both ways at entry row e0 = 2 chunks in, through
``fill_cache_for_user_`` as ``_restore_rows`` does, and every chip's cache rows must be byte-identical. Also times both (sync,
median of 10) and checks the 128-row tip carry falls back to the replicated upload.

    PREFILL_FABRIC_MODE=2d python -m pytest -c /dev/null --noconftest -p no:cacheprovider -s \\
        models/demos/deepseek_v3_d_p/tests/v41_prefix_reuse/device_check/test_v41_restore_gather_device.py
"""

from __future__ import annotations

import statistics
import time
from types import SimpleNamespace

import torch

import ttnn
from models.demos.common.prefill.runners.runner_utils import open_mesh_device
from models.demos.deepseek_v3_d_p.tt.runners.adapters.deepseek_v4_1_flash import DeepSeekV41FlashConfig
from models.demos.deepseek_v3_d_p.tt.v41 import prefix_reuse as PR

CHUNK = 4096
CAP = 4 * CHUNK


def _cache(mesh, W):
    return ttnn.from_torch(
        torch.zeros(1, 1, CAP, W, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _stand_in(mesh):
    """A ``V41PrefixReuse`` without its __init__ (no runtime, no store): what the restore upload reads."""
    roles = {L: SimpleNamespace(mode="full", compress_ratio=2) for L in (2, 8, 14)}
    roles[20] = SimpleNamespace(mode="kv_only", compress_ratio=1)

    def state():
        return SimpleNamespace(compressed_kv=_cache(mesh, 512), index_k=_cache(mesh, 128))

    pf = SimpleNamespace(
        blocks=[SimpleNamespace(layer=L, attn=None, state=state()) for L in (2, 8, 14)],
        cfg=SimpleNamespace(role=lambda L: roles[L]),
        kv_attn=object(),
        kv_only_layer=20,
        kv_state=state(),
    )
    obj = PR.V41PrefixReuse.__new__(PR.V41PrefixReuse)
    obj.rt, obj.pf, obj.chunk = SimpleNamespace(mesh_device=mesh), pf, CHUNK
    return obj


def _restore(obj, parts, start, gather):
    """``_restore_rows``' model-row half (uploads + fill_cache_for_user_), the upload path chosen by ``gather``."""
    obj.gather_rows = gather
    for i, (_L, _attn, st, ratio, _blk) in enumerate(PR.entry_sources(obj.pf)):
        e0 = start // ratio
        rows = obj._upload_rows(parts[2 * i], st.compressed_kv.dtype)
        keys = obj._upload_rows(parts[2 * i + 1], st.index_k.dtype)
        ttnn.kv_cache.fill_cache_for_user_(st.compressed_kv, rows, 0, update_idx=e0)
        ttnn.kv_cache.fill_cache_for_user_(st.index_k, keys, 0, update_idx=e0)
        ttnn.deallocate(rows)
        ttnn.deallocate(keys)


def _rows(obj, start):
    out = []
    for _L, _attn, st, ratio, _blk in PR.entry_sources(obj.pf):
        e0, n = start // ratio, CHUNK // ratio
        for src in (st.compressed_kv, st.index_k):
            W = int(src.shape[3])
            part = ttnn.slice(src, [0, 0, e0, 0], [1, 1, e0 + n, W])
            out.append(torch.stack([ttnn.to_torch(d) for d in ttnn.get_device_tensors(part)]))
            ttnn.deallocate(part)
    return out


def _zero(obj):
    for _L, _attn, st, _r, _b in PR.entry_sources(obj.pf):
        for name in ("compressed_kv", "index_k"):
            W = int(getattr(st, name).shape[3])
            ttnn.deallocate(getattr(st, name))
            setattr(st, name, _cache(obj.rt.mesh_device, W))


def test_restore_gather_byte_identical(monkeypatch):
    mesh = open_mesh_device((8, 4), DeepSeekV41FlashConfig)
    try:
        obj = _stand_in(mesh)
        torch.manual_seed(0)
        parts = []
        for _L, _attn, _st, ratio, _blk in PR.entry_sources(obj.pf):
            n = CHUNK // ratio
            parts += [torch.randn(n, 512).to(torch.bfloat16), torch.randn(n, 128).to(torch.bfloat16)]
        start = 2 * CHUNK

        monkeypatch.setenv(PR.ENV_RESTORE_GATHER, "1")
        assert obj._gather_rows() == 8  # builds the gather programs for the 4 block shapes
        _restore(obj, parts, start, gather=0)
        ttnn.synchronize_device(mesh)
        ref = _rows(obj, start)
        want = [
            p.reshape(1, 1, *p.shape).unsqueeze(0).expand(ref[i].shape[0], 1, 1, *p.shape) for i, p in enumerate(parts)
        ]
        assert all(torch.equal(r, w) for r, w in zip(ref, want)), "the replicated restore does not hold the host rows"
        _zero(obj)
        _restore(obj, parts, start, gather=8)
        ttnn.synchronize_device(mesh)
        got = _rows(obj, start)
        bad = [i for i, (g, r) in enumerate(zip(got, ref)) if not torch.equal(g, r)]
        assert not bad, f"gathered restore differs from the replicated one in tensors {bad}"
        print(f"\n[restore-gather] {len(got)} tensors x {got[0].shape[0]} chips byte-identical")

        # the 128-row tip carry does not split into tiles over 8 chips: replicated upload, same result
        obj.gather_rows = 8
        carry = torch.randn(128, 512).to(torch.bfloat16)
        t = obj._upload_rows(carry, ttnn.bfloat16)
        assert all(torch.equal(ttnn.to_torch(d).reshape(128, 512), carry) for d in ttnn.get_device_tensors(t))
        ttnn.deallocate(t)

        for gather in (0, 8):
            ts = []
            for i in range(13):
                t0 = time.perf_counter()
                _restore(obj, parts, start, gather)
                ttnn.synchronize_device(mesh)
                if i >= 3:
                    ts.append(time.perf_counter() - t0)
            print(
                f"[restore-gather] gather_rows {gather}: {statistics.median(ts) * 1e3:.2f} ms per chunk (uploads + fills, "
                f"sync, median of {len(ts)})"
            )
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
