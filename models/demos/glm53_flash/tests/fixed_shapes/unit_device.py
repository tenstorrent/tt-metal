# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Device check of ``GLM53_PREFILL_FIXED_SHAPES`` on ONE chip (tt-blaze G53F-0313), the index maintenance alone:
``TtDSAIndexer.maintain_prefill`` of random index stems, per-length path vs fixed path, on two caches holding the SAME
random stale content (a previous prompt's pools and ring rows): the pool caches (both users) and the rings must be
bit-identical after every call, and the fixed path's calls at new ends of one (start, padded length) must add no
program. The device twin of ``tests/test_prefill_fixed_shapes_host.py`` (which checks the composition on an emulation).

    TT_VISIBLE_DEVICES=0 python models/demos/glm53_flash/tests/fixed_shapes/unit_device.py
"""

from __future__ import annotations

import os
import sys
import time
import types

import torch

import ttnn
from models.demos.glm53_flash.tt import indexer as IX

P, D, TILE = 4, 128, 32


def main() -> int:
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    ok = True
    try:
        cfg = types.SimpleNamespace(index_head_dim=D, index_kpool=P, index_select_pools=512)
        idx = IX.TtDSAIndexer.__new__(IX.TtDSAIndexer)
        g = torch.Generator().manual_seed(0)
        idx.device, idx.mesh, idx.cfg, idx.P, idx.D = mesh, None, cfg, P, D
        idx.ape = IX._upload(mesh, torch.randn(1, 1, P, D, generator=g), ttnn.float32)
        idx.fp32 = IX._compute(mesh)
        idx._const_cache = {}
        nprog = mesh.num_program_cache_entries

        def cache_pair(max_seq_len):
            caches = [IX.TtIndexCache(mesh, cfg, max_batch=2, max_seq_len=max_seq_len) for _ in range(2)]
            stale = [torch.randn(tuple(caches[0].pool_cache.shape), generator=g).to(torch.bfloat16)]
            stale += [torch.randn(tuple(caches[0].ring_k.shape), generator=g).to(torch.bfloat16) for _ in range(2)]
            for c in caches:
                for name, host in zip(("pool_cache", "ring_k", "ring_g"), stale):
                    old = getattr(c, name)
                    setattr(c, name, ttnn.from_torch(host, dtype=old.dtype, layout=ttnn.TILE_LAYOUT, device=mesh))
                    ttnn.deallocate(old)
            return caches

        def run(c, k, gg, start, n, fixed):
            os.environ["GLM53_PREFILL_FIXED_SHAPES"] = "1" if fixed else "0"
            kt = ttnn.from_torch(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
            gt = ttnn.from_torch(gg, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
            ttnn.synchronize_device(mesh)
            p0, t0 = nprog(), time.perf_counter()
            idx.maintain_prefill(None, c, 1, n, start=start, kg=(kt, gt))
            ttnn.synchronize_device(mesh)
            return nprog() - p0, time.perf_counter() - t0

        def bits(t):
            return ttnn.to_torch(t).view(torch.int16)

        for start, T in ((0, 256), (128, 512), (36864, 5120), (12288, 12288)):
            a, b = cache_pair(start + T + 4096)
            k = torch.randn(1, 1, T, D, generator=g).to(torch.bfloat16)
            gg = torch.randn(1, 1, T, D, generator=g).to(torch.bfloat16)
            lens = sorted({1, 2, 3, 4, 5, 127, 128, 129, 130, 131, 132, T - 1, T, T // 2 + 3, T - 100, max(1, T - 129)})
            bad = []
            for n in [x for x in lens if 0 < x <= T]:
                run(a, k, gg, start, n, False)
                run(b, k, gg, start, n, True)
                for name in ("pool_cache", "ring_k", "ring_g"):
                    if not torch.equal(bits(getattr(a, name)), bits(getattr(b, name))):
                        bad.append((n, name))
            ok &= not bad
            print(
                f"[unit] start {start} T_pad {T}: {len(lens)} lengths, caches {'IDENTICAL' if not bad else bad}",
                flush=True,
            )

        # programs: ends 100 tokens apart in one (start, T_pad) = (36864, 5120) chunk, fresh lengths for both paths
        start, T = 36864, 5120
        a, b = cache_pair(start + T)
        k = torch.randn(1, 1, T, D, generator=g).to(torch.bfloat16)
        gg = torch.randn(1, 1, T, D, generator=g).to(torch.bfloat16)
        on, off = [], []
        for n in range(4167, 5120, 100):  # tails 3: every tail-row slice program on the first call
            on.append(run(b, k, gg, start, n, True))
        for n in range(4133, 5120, 100):
            off.append(run(a, k, gg, start, n, False))
        print(f"[unit] fixed path, ends 100 apart: programs {[d for d, _ in on]}, s {[round(s, 4) for _, s in on]}")
        print(
            f"[unit] per-length path, ends 100 apart: programs {[d for d, _ in off]}, s {[round(s, 4) for _, s in off]}"
        )
        ok &= all(d == 0 for d, _ in on[1:])
        # warmed: the same lengths again
        on_w = [run(b, k, gg, start, n, True) for n in range(4167, 5120, 300)]
        off_w = [run(a, k, gg, start, n, False) for n in range(4133, 5120, 300)]
        print(f"[unit] warmed s: fixed {[round(s, 4) for _, s in on_w]}, per-length {[round(s, 4) for _, s in off_w]}")
        print(f"[unit] tail selector entries {len(IX.TtDSAIndexer._tail_sel_shared)}, consts {list(idx._const_cache)}")
    finally:
        ttnn.close_mesh_device(mesh)
    print(f"[unit] RESULT {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
