# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""INDEX_K_SOURCE=served (P.76): the indexer's keys are the served 192-byte rows decoded (tt/index_keys.py,
ttnn.bringup.unpack_fp8_rows), not a private copy (device; run_safe_pytest.sh, the spec's mesh).

For ratio 2 and ratio 1, a served index-key cache S (INDEX_K_SOURCE=served, INDEX_K_SERVED=fp8_row192) gets chunk
writes through the device write path (aligned starts, a follow-up at 32 mod 64, a 64-aligned follow-up, the last W
block). After every write, on every chip, the key buffer index_keys.keys returns holds, in local rows [0, hi) (the
chunk's W blocks' end), exactly the served rows decoded on the host (e4m3 x scale, rounded to nearest even onto
bfloat8_b in float64): once by a full decode, once by a prefetch of [0, c W / r) before the write and the chunk's
blocks after it (the model's path). The served bytes equal those of a private-path cache U (INDEX_K_SOURCE=private,
INDEX_K_COPY=bf16) written with the same keys (both encode the bf16 key). A reused slot: a fresh served cache T on S's
engine slab, never written by this process, returns the same keys as S (a private copy would hold zeros: shown).
"""

import pytest
import torch

import ttnn
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec
from models.demos.deepseek_v41_flash_d_p.tt import compressor as C
from models.demos.deepseek_v41_flash_d_p.tt import index_keys
from models.demos.deepseek_v41_flash_d_p.tt.served import CacheSlab
from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

WIDTH = 128


def _set(monkeypatch, **kw):
    for k, v in kw.items():
        monkeypatch.setenv(settings.env_name(k), v)
    settings.refresh()


def test_source_rule(monkeypatch):
    _set(monkeypatch, INDEX_K_SOURCE="served", INDEX_K_FORMAT="bfp8", INDEX_K_SERVED="fp8_row192")
    assert C.index_k_source() == "served"
    for fmt, served in (("bfp4", "fp8_row192"), ("bf16", "fp8_row192"), ("bfp8", "bf16")):
        _set(monkeypatch, INDEX_K_FORMAT=fmt, INDEX_K_SERVED=served)
        assert C.index_k_source() == "private", (fmt, served)
    _set(monkeypatch, INDEX_K_SOURCE="private", INDEX_K_FORMAT="bfp8", INDEX_K_SERVED="fp8_row192")
    assert C.index_k_source() == "private"


def _bfp8_exact(x: torch.Tensor) -> torch.Tensor:
    """[n, 128] float -> rounded to nearest even onto the bfloat8_b grid (16 values a group, float64)."""
    g = x.double().reshape(-1, 16)
    amax = g.abs().amax(-1, keepdim=True)
    _, e = torch.frexp(amax)
    step = torch.ldexp(torch.ones_like(amax), e - 7)
    q = torch.round(g.abs() / step).clamp(max=127)
    return torch.where(amax == 0, torch.zeros_like(g), torch.sign(g) * q * step).float().reshape(x.shape)


def _rows(mesh, nat: torch.Tensor) -> ttnn.Tensor:
    return C._upload(mesh, nat.reshape(1, 1, -1, WIDTH).to(torch.bfloat16), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)


def _chips(t: ttnn.Tensor) -> list[torch.Tensor]:
    return [ttnn.to_torch(d).float().reshape(-1, WIDTH) for d in ttnn.get_device_tensors(t)]


def _check(what, cache, start, buf, exp_from=None):
    """buf's local rows [0, hi) on every chip == the served rows of ``exp_from`` (default ``cache``) decoded."""
    _, hi = index_keys._span(cache, start)
    shards = (exp_from or cache).served.host_rows()
    got = _chips(buf)
    for chip, (g, raw) in enumerate(zip(got, shards)):
        exp = _bfp8_exact(C.fp8_rows_decode(raw[:hi]))
        bad = (g[:hi] != exp).any(dim=-1).nonzero().flatten()
        assert not bad.numel(), f"{what}: chip {chip}: {bad.numel()} of {hi} key rows differ, first {bad[:4].tolist()}"
    return len(got), hi


@mesh_parametrize
def test_index_k_served(mesh_device, monkeypatch):
    if mesh_device is None:
        pytest.skip("device test")
    from models.demos.deepseek_v41_flash_d_p.tt.attention import WindowTables

    mesh = mesh_device
    chunk = int(spec().get("target.chunk"))
    tables = WindowTables(mesh, chunk)
    writes = [
        (0, chunk - 500),
        (0, chunk),
        (chunk, 2 * chunk),
        (chunk + 1312, 2 * chunk + 1312 - 200),  # 1312 % 64 == 32: ratio 2 -> 16 compressed rows off a tile
        (2 * chunk + 2944, 3 * chunk + 2944),
        (3 * chunk + 1312, 4 * chunk),  # the last W block
    ]
    _set(monkeypatch, INDEX_K_FORMAT="bfp8", INDEX_K_SERVED="fp8_row192")
    gen = torch.Generator().manual_seed(76)
    for ratio in (2, 1):
        max_seq = 4 * chunk
        _set(monkeypatch, INDEX_K_SOURCE="served")
        s = C.TtServedRows(mesh, tables, max_seq, WIDTH, ratio, dedup=False, fp8=True, mirrors=[None], layer=ratio)
        assert s.k_served and s.k8 is None and s.slab is None
        _set(monkeypatch, INDEX_K_SOURCE="private", INDEX_K_COPY="bf16")
        u = C.TtServedRows(mesh, tables, max_seq, WIDTH, ratio, dedup=False, fp8=True, mirrors=[None])
        assert not u.k_served
        for k, (start, end) in enumerate(writes):
            prefetched = k % 2 == 1  # (alternately the model's path: the history first, the chunk's blocks after)
            index_keys.new_chunk()
            if prefetched:
                index_keys.prefetch(s, start)
            n = chunk // ratio
            keys = (torch.randn(n, WIDTH, generator=gen) * 3.0).to(torch.bfloat16).float()
            keys[end // ratio - start // ratio :] = 0
            kb = _rows(mesh, keys)
            s.write(kb, start, end)
            u.write(kb, start, end)
            ttnn.deallocate(kb)
            st0 = dict(index_keys.STATS)
            buf = index_keys.keys(s, start)
            path = "tail" if index_keys.STATS["tail"] > st0["tail"] else "full"
            assert path == ("tail" if prefetched and start >= chunk else "full"), (path, start, prefetched)
            what = f"ratio {ratio}, write [{start}, {end}) ({path})"
            chips, hi = _check(what, s, start, buf)
            for chip, (g, e) in enumerate(zip(s.served.host_rows(), u.served.host_rows())):
                assert torch.equal(g, e), f"{what}: chip {chip}: served bytes differ from the private path's"
            assert index_keys.keys(s, start) is buf and index_keys.STATS["hit"] > st0["hit"]  # (a memo reader)
            print(f"[index_k_served] {what}: keys of rows [0, {hi}) exact on {chips} chips")
        # a reused slot: T reads S's engine slab, written by "another process" (S); T never wrote a row
        _set(monkeypatch, INDEX_K_SOURCE="served")
        t = C.TtServedRows(mesh, tables, max_seq, WIDTH, ratio, dedup=False, fp8=True, slab=CacheSlab(s.served.tensor))
        _set(monkeypatch, INDEX_K_SOURCE="private", INDEX_K_COPY="bfp8")
        p = C.TtServedRows(mesh, tables, max_seq, WIDTH, ratio, dedup=False, fp8=True, slab=CacheSlab(s.served.tensor))
        index_keys.new_chunk()
        start = writes[-1][0]
        chips, hi = _check(f"ratio {ratio}, reused slot", t, start, index_keys.keys(t, start), exp_from=s)
        stale = sum(int((g[:hi] == 0).all(dim=-1).sum()) for g in _chips(p.k8))
        print(
            f"[index_k_served] ratio {ratio}, reused slot: served keys exact on {chips} chips; a private copy "
            f"would score {stale} all-zero rows of {hi * chips}"
        )
        for c in (t, p, s, u):
            c.free()
