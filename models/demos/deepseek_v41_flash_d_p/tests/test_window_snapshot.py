# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""WINDOW_KV_KEEP=snapshot128 (P.42): the served 128-row window snapshot at cap (device; run_safe_pytest.sh, the
spec's mesh).

One window cache (tt/attention.TtWindowCache, keep snapshot128, its own snapshot slab: the contract's ND-sharded
[1, 1, 128, 512] bf16 ROW_MAJOR, 32-row records ROUND_ROBIN_1D over the DRAM banks) takes random window rows through
both ring write paths (``write``: the chunk's natural rows on every chip; ``write_own``: each chip its own query rows,
the ring blocks owned by one chip, so the snapshot must gather them), then ``write_snapshot(end)``. After every write,
on every chip: snapshot row p % 128 == the row of position p for p in [cap - 128, cap), cap = (end - 1) // 64 x 64
(KV_BLOCK_SIZE 64), and every row of no position < cap zero, byte for byte. Ends: 100 (cap 64 < 128), a whole chunk,
a mid-chunk end, a follow-up off the chunk period (start 32 mod 64) whose window spans two chunks, an end mid-record
over owned blocks, the server's pulled-back last chunk (its carry from the pin: TtWindowCache.pin), 40 (cap 0: all
zero). 8x4: ~10 s.
"""

import pytest
import torch

import ttnn
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec
from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

WIDTH = 512


def _set(monkeypatch, **kw):
    for k, v in kw.items():
        monkeypatch.setenv(settings.env_name(k), str(v))
    settings.refresh()


def test_snapshot_cap():
    from models.demos.deepseek_v41_flash_d_p.tt.attention import snapshot_cap

    assert [snapshot_cap(e, 64) for e in (1, 64, 65, 100, 128, 129, 3000, 5120, 5121)] == [
        0,
        0,
        64,
        64,
        64,
        128,
        2944,
        5056,
        5120,
    ]


@mesh_parametrize
def test_window_snapshot(mesh_device, monkeypatch):
    if mesh_device is None:
        pytest.skip("device test")
    from models.demos.deepseek_v41_flash_d_p.tt.attention import (
        SNAP,
        TtWindowCache,
        WindowTables,
        _upload,
        snapshot_cap,
    )

    _set(monkeypatch, WINDOW_KV_KEEP="snapshot128", KV_BLOCK_SIZE=64)
    mesh = mesh_device
    chunk = int(spec().get("target.chunk"))
    tables = WindowTables(mesh, chunk)
    g = tables.geo
    cache = TtWindowCache(mesh, tables, 4 * chunk)
    assert cache.snap is not None and cache.ring is not None
    gen = torch.Generator().manual_seed(42)
    full = (torch.randn(4 * chunk, WIDTH, generator=gen) * 2).to(torch.bfloat16)

    def nat_rows(start, end):
        nat = torch.zeros(chunk, WIDTH, dtype=torch.bfloat16)
        nat[: end - start] = full[start:end]
        return nat

    def write_rep(start, end):
        t = _upload(mesh, nat_rows(start, end).view(1, 1, chunk, WIDTH), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        cache.write(t, start, end)
        ttnn.deallocate(t)

    def write_own(start, end):
        nat = nat_rows(start, end).view(g.sp, g.tp, g.q_rows, WIDTH)
        t = _upload(mesh, nat, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, (0, 1))
        cache.write_own(t, start, end)
        ttnn.deallocate(t)

    fails, checked = [], []

    def check(end, what):
        cache.write_snapshot(end)
        ttnn.synchronize_device(mesh)
        cap = snapshot_cap(end, 64)
        want = torch.zeros(SNAP, WIDTH, dtype=torch.bfloat16)
        for p in range(max(0, cap - SNAP), cap):
            want[p % SNAP] = full[p]
        chips = cache.snapshot_host()
        for q, got in enumerate(chips):
            if not torch.equal(got.view(torch.int16), want.view(torch.int16)):
                bad = torch.nonzero((got != want).any(dim=-1)).flatten()
                fails.append(f"{what} end {end} cap {cap}: chip {q} rows {bad[:8].tolist()} differ")
                break
        checked.append((what, end, cap, len(chips)))

    for start, end, how, what in (
        (0, 100, write_rep, "short prompt, cap < 128"),
        (0, chunk, write_own, "whole chunk, owned blocks"),
        (chunk, chunk + 1000, write_own, "mid-chunk end"),
        (chunk + 992, 2 * chunk + 70, write_rep, "follow-up at 32 mod 64, window over two chunks"),
        (2 * chunk + 64, 3 * chunk - 23, write_own, "owned blocks, end mid-record"),
    ):
        how(start, end)
        check(end, what)
    # the server's pulled-back last chunk (prefill_writer.cpp:63-67): (P, max_seq) with P = max_seq - chunk after a chunk
    # that ended past P; its carry [P - 128, P) is out of the 192-row ring, kept in the pin
    pin_at = 3 * chunk
    assert cache.pin_at == pin_at, (cache.pin_at, pin_at)
    cache.truncate(2 * chunk + 640)  # (its own carry is not the point here: masked, as begin_chunk does for a new slot)
    write_own(2 * chunk + 640, 3 * chunk + 640)
    cache.truncate(None)
    assert not cache.ring.holds(pin_at - 128, pin_at) and cache.carry_ready(pin_at)
    carry = cache._ring_carry(pin_at)
    for q, got in enumerate(ttnn.get_device_tensors(carry)):
        got = ttnn.to_torch(got).reshape(-1, WIDTH)
        if not torch.equal(got.view(torch.int16), full[pin_at - 128 : pin_at].view(torch.int16)):
            fails.append(f"pulled-back carry [{pin_at - 128}, {pin_at}): chip {q} differs")
            break
    ttnn.deallocate(carry)
    write_own(pin_at, 4 * chunk)
    check(4 * chunk, "pulled-back last chunk (max_seq - chunk, max_seq)")
    write_rep(0, 40)
    check(40, "cap 0: no position below cap")
    cache.free()
    print(f"[window snapshot] checked (what, end, cap, chips): {checked}")
    assert not fails, "\n".join(fails)
