# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""DS41-216 stress on the model's own window-ring gathers (device; run_safe_pytest.sh, the spec's mesh).

One window cache (tt/attention.TtWindowCache, WINDOW_KV_KEEP=snapshot128) takes ITERS chunks [0, E) through
``write_own`` (each chip its own query rows: the ring's last blocks are held by one chip each, so every read gathers
them from their owner), E a random multiple of 32 in [1024, chunk] (the owners move over the mesh), the rows one of K
random versions. After each write: ``write_snapshot(E)`` (the served 128-row snapshot, ``_snapshot_rows``) and
``_ring_carry(E)`` (P.32's carry, ``_gather_all``: the model's CCL_IMPL gathers), then a full mesh sync, then every
chip's snapshot and carry against the rows written, byte for byte. ``skew``: a per-chip sparse_matmul (one random mesh
row or column or a random fifth of the chips dense, tests/test_gather_stress._Skew) before each of the two reads, so the chips reach the gathers at different
times as in the model. ``pairs``: axis = the shipped tt/ccl.py (a high_bw semaphore pair per mesh axis); shared = the
pre-fix one pair for both axes (xfail: the stress must see DS41-216 there). Prints the wrong replicas per iteration
(chip, mesh row / column, the positions).
"""

import pytest
import torch

import ttnn
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec
from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

WIDTH = 512
K = 8
ITERS = 512


def _set(monkeypatch, **kw):
    for k, v in kw.items():
        monkeypatch.setenv(settings.env_name(k), str(v))
    settings.refresh()


@pytest.mark.parametrize(
    "pairs",
    [
        "axis",
        pytest.param(
            "shared",
            marks=pytest.mark.xfail(
                reason="DS41-216: one high_bw semaphore pair for both axes (pre-fix)", strict=False
            ),
        ),
    ],
)
@pytest.mark.parametrize("skew", ["even", "skew"])
@mesh_parametrize(timeout_s=0)
def test_window_gather_stress(mesh_device, monkeypatch, skew, pairs):
    if mesh_device is None:
        pytest.skip("device test")
    from models.demos.deepseek_v41_flash_d_p.tests.test_gather_stress import _Skew
    from models.demos.deepseek_v41_flash_d_p.tt.attention import (
        SNAP,
        WINDOW,
        TtWindowCache,
        WindowTables,
        _upload,
        snapshot_cap,
    )

    _set(monkeypatch, WINDOW_KV_KEEP="snapshot128", KV_BLOCK_SIZE=64)
    mesh = mesh_device
    if pairs == "shared":  # (the pre-fix tt/ccl.py: every gather on one semaphore pair, whatever its axis)
        from models.demos.deepseek_v41_flash_d_p.tt import ccl

        real = ccl._sems
        monkeypatch.setattr(ccl, "_sems", lambda m, axis: real(m, 1))
    chunk = int(spec().get("target.chunk"))
    tables = WindowTables(mesh, chunk)
    g = tables.geo
    cache = TtWindowCache(mesh, tables, 4 * chunk)
    busy = _Skew(mesh) if skew == "skew" else None
    gen = torch.Generator().manual_seed(216)
    full = [(torch.randn(chunk, WIDTH, generator=gen) * 2).to(torch.bfloat16) for _ in range(K)]

    bad = {"snapshot": 0, "carry": 0}
    detail = []

    def compare(what, k, end, chips, want):
        wrong = []
        for q, got in enumerate(chips):
            got = got.reshape(-1, WIDTH)
            if not torch.equal(got.view(torch.int16), want.view(torch.int16)):
                rows = torch.nonzero((got.view(torch.int16) != want.view(torch.int16)).any(dim=-1)).flatten()
                wrong.append((q, rows))
        if wrong:
            bad[what] += len(wrong)
            if len(detail) < 40:
                chips_s = [f"{q} (r {q // g.tp} c {q % g.tp})" for q, _ in wrong]
                detail.append(
                    f"iter {k} end {end} {what}: {len(wrong)} chips wrong: {chips_s}; rows {wrong[0][1][:6].tolist()}"
                )

    for k in range(ITERS):
        v = k % K
        end = 32 * int(torch.randint(32, chunk // 32 + 1, (1,), generator=gen))
        own = full[v].clone()
        own[end:] = 0  # (rows at or past end are zero, write_own's contract)
        t = _upload(mesh, own.view(g.sp, g.tp, g.q_rows, WIDTH), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, (0, 1))
        cache.write_own(t, 0, end)
        ttnn.deallocate(t)
        if busy is not None:
            busy()
        cache.write_snapshot(end)
        if busy is not None:  # (the carry's gathers write fresh outputs: an early write there lands on live bytes)
            busy()
        carry = cache._ring_carry(end)
        ttnn.synchronize_device(mesh)
        cap = snapshot_cap(end, 64)
        want = torch.zeros(SNAP, WIDTH, dtype=torch.bfloat16)
        for p in range(max(0, cap - SNAP), cap):
            want[p % SNAP] = own[p]
        compare("snapshot", k, end, cache.snapshot_host(), want)
        compare("carry", k, end, [ttnn.to_torch(t) for t in ttnn.get_device_tensors(carry)], own[end - WINDOW : end])
        ttnn.deallocate(carry)
    cache.free()
    print(
        f"[window gather stress] {skew} pairs={pairs} CCL_IMPL={settings.get('CCL_IMPL')}: {ITERS} iterations x {g.n} chips: "
        f"snapshot replicas wrong {bad['snapshot']}, carry replicas wrong {bad['carry']}"
    )
    for d in detail:
        print("  " + d)
    assert not any(bad.values()), f"wrong replicas: {bad}"
