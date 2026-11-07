# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""INDEX_K_COPY=bfp8 (P.41): the index keys' one bfloat8_b copy and the served rows built from it (device;
run_safe_pytest.sh, the spec's mesh).

For ratio 2 and ratio 1, two served index-key caches (INDEX_K_SERVED=fp8_row192) get the same chunk writes through
the device write path (TtServedRows.write): B with INDEX_K_COPY=bfp8 and random bf16 keys, R with INDEX_K_COPY=bf16
and the same keys already rounded on the host to bfloat8_b by tt-metal's host rule (ckv_bfp4_ttnn/tests/reference.py
round_values, nearest even per 16 values of a row; ttnn.bringup.bfp_round equals it for bf16 inputs). Writes: chunk-
aligned starts, a follow-up at a 32-but-not-64-aligned start (ratio 2: 16 compressed rows off a tile on the c_start
row) with a partial end, a 64-aligned follow-up, and a start in the last W block (the region shorter than two
blocks). After every write, on every chip: B's bfp8 copy == R's bf16 copy (the rounded keys), exactly, and B's served
192-byte rows == R's, byte for byte (R encodes the rounded keys: the encode of the bfp8 keys). Plus the fallback rule.
"""

import pytest
import torch
from ttnn.bringup.ckv_bfp4_ttnn.tests.reference import round_values

import ttnn
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec
from models.demos.deepseek_v41_flash_d_p.tt import compressor as C
from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

WIDTH = 128


def _set(monkeypatch, **kw):
    for k, v in kw.items():
        monkeypatch.setenv(settings.env_name(k), v)
    settings.refresh()


def test_fallback_rule(monkeypatch):
    _set(monkeypatch, INDEX_K_COPY="bfp8", INDEX_K_FORMAT="bfp8", INDEX_K_SERVED="fp8_row192")
    assert C.index_k_copy() == "bfp8"
    for fmt, served in (("bfp4", "fp8_row192"), ("bf16", "fp8_row192"), ("bfp8", "bf16")):
        _set(monkeypatch, INDEX_K_FORMAT=fmt, INDEX_K_SERVED=served)
        assert C.index_k_copy() == "bf16", (fmt, served)
    _set(monkeypatch, INDEX_K_COPY="bf16", INDEX_K_FORMAT="bfp8", INDEX_K_SERVED="fp8_row192")
    assert C.index_k_copy() == "bf16"


def _rows(mesh, nat: torch.Tensor) -> ttnn.Tensor:
    return C._upload(mesh, nat.reshape(1, 1, -1, WIDTH).to(torch.bfloat16), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)


def _chips(t: ttnn.Tensor) -> list[torch.Tensor]:
    return [ttnn.to_torch(d).float().reshape(-1, WIDTH) for d in ttnn.get_device_tensors(t)]


@mesh_parametrize
def test_index_k_copy_bfp8(mesh_device, monkeypatch):
    if mesh_device is None:
        pytest.skip("device test")
    from models.demos.deepseek_v41_flash_d_p.tt.attention import WindowTables

    mesh = mesh_device
    chunk = int(spec().get("target.chunk"))
    tables = WindowTables(mesh, chunk)
    writes = [
        (0, chunk - 500),  # aligned, a partial end
        (0, chunk),
        (chunk, 2 * chunk),
        (chunk + 1312, 2 * chunk + 1312 - 200),  # 1312 % 64 == 32: ratio 2 -> 656 compressed rows (16 off a tile)
        (2 * chunk + 2944, 3 * chunk + 2944),  # 64-aligned follow-up
        (3 * chunk + 1312, 4 * chunk),  # the last W block: the region is one block
    ]
    # (max_seq, writes): capacities of one and two chunks make the copy's slices full-range (K-1, DS41-52)
    cases = [
        (4 * chunk, writes),
        (2 * chunk, [(0, chunk), (1312, chunk + 1312)]),
        (chunk, [(0, chunk - 64), (0, chunk)]),
    ]
    _set(monkeypatch, INDEX_K_FORMAT="bfp8", INDEX_K_SERVED="fp8_row192")
    gen = torch.Generator().manual_seed(41)
    for (max_seq, writes), ratio in [(c, r) for c in cases for r in (2, 1)]:
        _set(monkeypatch, INDEX_K_COPY="bfp8")
        b = C.TtServedRows(mesh, tables, max_seq, WIDTH, ratio, dedup=False, fp8=True, mirrors=[None])
        _set(monkeypatch, INDEX_K_COPY="bf16")
        ref = C.TtServedRows(mesh, tables, max_seq, WIDTH, ratio, dedup=False, fp8=True, mirrors=[None])
        assert b.k8 is not None and b.slab is None and ref.k8 is None
        assert b.k8.dtype == ttnn.bfloat8_b and tuple(b.k8.shape) == (1, 1, b.crows, WIDTH)
        for start, end in writes:
            n = chunk // ratio
            keys = torch.randn(n, WIDTH, generator=gen) * 3.0
            keys = keys.to(torch.bfloat16).float()
            valid = end // ratio - start // ratio
            keys[valid:] = 0  # rows of groups not complete by end (TtCompressor.write's mask)
            kb, kr = _rows(mesh, keys), _rows(mesh, round_values(keys, mbits=7))
            b.write(kb, start, end)
            ref.write(kr, start, end)
            for t in (kb, kr):
                ttnn.deallocate(t)
            what = f"max_seq {max_seq}, ratio {ratio}, write [{start}, {end})"
            got, exp = _chips(b.k8), [s.float() for s in ref.slab.host_rows()]
            for chip, (g, e) in enumerate(zip(got, exp)):
                bad = (g != e).any(dim=-1).nonzero().flatten()
                assert not bad.numel(), f"{what}: chip {chip}: {bad.numel()} copy rows differ, first {bad[:4].tolist()}"
            for sb, sr, name in ((b.served, ref.served, "served"), (b.mirrors[0], ref.mirrors[0], "mirror")):
                for chip, (g, e) in enumerate(zip(sb.host_rows(), sr.host_rows())):
                    bad = (g != e).any(dim=-1).nonzero().flatten()
                    assert (
                        not bad.numel()
                    ), f"{what}: chip {chip}: {bad.numel()} {name} rows differ, first {bad[:4].tolist()}"
            print(f"[index_k_copy] {what}: bfp8 copy and served rows exact on {len(got)} chips")
        assert b.version == ref.version == len(writes)
        for c in (b, ref):
            c.free()
