# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks of the regions-sourced streaming decode (``GLM53_REGIONS_DECODE=stream``; no ttnn, no device):
``custom_ops/expert_stream/regions.py`` against an independent model of the DRAM-interleaved page map, on 8 banks
(Blackhole) and 12 (Wormhole) and at the three per-chip widths (256 on the 8x4, 1024 on a (2,4) block, 2048 on one
chip); the (bank, column) maps of the writers; the packet model against the stacked layout; the byte model that
decided the finding (9.41 GiB per chip); the knob and the mesh model's keep-stacked rule; the REGIONS_SRC kernel
ABI (defines, common-arg indexing, the appended compile-time tail, the k-major compute index, the exit barrier) and
the op / model plumbing by source inspection (``doc/mesh/regions_decode.md``).

    pytest --noconftest models/demos/glm53_flash/tests/test_regions_decode_host.py -q
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from pathlib import Path

import pytest
from loguru import logger

from models.demos.glm53_flash.custom_ops.expert_stream import regions as R
from models.demos.glm53_flash.custom_ops.expert_stream.plan import TILE_BYTES

PKG = Path(R.__file__).parents[2]  # models/demos/glm53_flash
KERNELS = Path(R.__file__).parent / "kernels"
OP_SRC = (Path(R.__file__).parent / "op.py").read_text()
MOE_SRC = (PKG / "tt" / "moe.py").read_text()
MESH_MODEL_SRC = (PKG / "tt" / "mesh_model.py").read_text()
K, H = 4096, 4096
WIDTHS = (256, 1024, 2048)  # I_local on the 8x4, a (2,4) block, one chip
BANKS = (8, 12)  # Blackhole p150, Wormhole


@contextmanager
def expect_error(error, message):
    """The root conftest's ``expect_error`` fixture (the CI triager's [EXPECTED_ERROR] markers), local to this module:
    the host test runs with ``--noconftest`` because the root conftest imports ttnn at collection."""
    logger.info(f'[EXPECTED_ERROR BEGIN] {error.__name__} message="{message}"')
    try:
        with pytest.raises(error, match=message) as exc_info:  # allow-pytest.raises: the fixture body, no conftest here
            yield exc_info
    finally:
        logger.info(f'[EXPECTED_ERROR END] {error.__name__} message="{message}"')


# ---------------------------------------------------------------------------------------------- independent model
def _page_map(rows_t: int, cols_t: int, banks: int, tile_bytes: int) -> dict[tuple[int, int], tuple[int, int]]:
    """Tile (r, c) of a [rows_t x cols_t] tile tensor -> (bank, offset): the allocator's round-robin over the
    row-major page ids, written out independently of ``regions.py``."""
    out = {}
    page = 0
    for r in range(rows_t):
        for c in range(cols_t):
            out[(r, c)] = (page % banks, (page // banks) * tile_bytes)
            page += 1
    return out


def _tiles_of(packets: list[tuple[int, int]], tile_bytes: int) -> list[int]:
    """The tile offsets a packet list covers, in issue order (every packet a whole number of tiles)."""
    offs = []
    for off, nbytes in packets:
        assert nbytes % tile_bytes == 0, (off, nbytes, tile_bytes)
        offs += [off + j * tile_bytes for j in range(nbytes // tile_bytes)]
    return offs


# ---------------------------------------------------------------------------------------------- knob
def test_decode_mode_knob():
    assert R.regions_decode_mode({}) == "stream"  # the default since 2026-10-27
    for m in R.REGIONS_DECODE_MODES:
        assert R.regions_decode_mode({R.REGIONS_DECODE_ENV: m}) == m
    with expect_error(ValueError, "GLM53_REGIONS_DECODE"):
        R.regions_decode_mode({R.REGIONS_DECODE_ENV: "fast"})


def test_resolve_keep_stacked():
    """The mesh model's auto rule: the planner's verdict, except that the stream mode never keeps a second copy."""
    assert R.resolve_keep_stacked(True, "off") is True
    assert R.resolve_keep_stacked(False, "off") is False
    assert R.resolve_keep_stacked(True, "regions") is True
    assert R.resolve_keep_stacked(True, "stream") is False
    assert R.resolve_keep_stacked(False, "stream") is False


# ---------------------------------------------------------------------------------------------- layout math
@pytest.mark.parametrize("I", WIDTHS)
@pytest.mark.parametrize("w13", ["bfp8", "bfp4"])
def test_gate_up_walk_matches_page_map(I, w13):
    """The gate|up reader's packets for every (bank, pair) cover exactly the K tiles of tile column
    bank + banks * q, in K order, all in that bank, per the independent page map; whole-column packets at ppb = 1,
    single tiles otherwise."""
    banks = 8
    p = R.plan_regions_stream(K, I, H, banks=banks, w13=w13)
    assert p.bank_local, p.reason
    tb = TILE_BYTES[w13]
    pm = _page_map(p.kt, p.nt, banks, tb)
    for b in range(banks):
        for q in range(p.ppb):
            n = b + banks * q
            want = [pm[(k, n)] for k in range(p.kt)]
            assert all(bank == b for bank, _ in want), (b, q)
            got = _tiles_of(p.gate_col_packets(q), tb)
            assert got == [off for _, off in want], (I, w13, b, q)
            assert [p.gate_tile_offset(k, q) for k in range(p.kt)] == got
    if p.ppb == 1:  # the stacked op's packet rule: the largest divisor of K/32 within the 16 KB burst
        want_tiles = 8 if w13 == "bfp8" else 16  # 8.7 KB bfp8 / 9.2 KB bfp4 packets
        assert p.gate_packet_tiles == want_tiles and p.gate_pages_per_col == p.kt // want_tiles
        assert p.gate_packet_bytes == want_tiles * tb and p.gate_src_step == p.gate_packet_bytes
    else:
        assert p.gate_packet_tiles == 1 and p.gate_pages_per_col == p.kt and p.gate_src_step == p.ppb * tb
    # a pair slot of the dispatch reader: the gate column then the up column, kt tiles each, contiguous in L1
    assert p.gate_pages_per_col * p.gate_packet_bytes == p.kt * tb


@pytest.mark.parametrize("I", WIDTHS)
def test_down_walk_matches_page_map(I):
    """The down reader's packets for a core (bank b, columns [c0, c0 + cols)) cover, per K tile, the cols tiles of
    the output columns b + banks * (c0 + c), contiguous, all in bank b (k-major in the CB)."""
    banks = 8
    p = R.plan_regions_stream(K, I, H, banks=banks)
    assert p.bank_local, p.reason
    tb = TILE_BYTES["bfp8"]
    pm = _page_map(p.seg_tiles, p.ht, banks, tb)
    assert p.C == 16 and p.cols * p.down_readers == p.C and p.cols <= 8
    assert p.cols * p.seg_tiles <= 128, "the expert block must fit the three CB slots (op.py _down_readers)"
    for b in range(banks):
        for r in range(p.down_readers):
            c0 = r * p.cols
            packets = p.down_block_packets(c0)
            assert len(packets) == p.seg_tiles and all(n == p.cols * tb for _, n in packets)
            for i, (off, _) in enumerate(packets):
                for c in range(p.cols):
                    bank, want = pm[(i, b + banks * (c0 + c))]
                    assert bank == b and want == off + c * tb, (I, b, r, i, c)
                    assert p.down_tile_offset(i, c0 + c) == want


@pytest.mark.parametrize("I", WIDTHS)
def test_column_maps_are_permutations(I):
    """The writers' (bank, column) -> global column maps of both layouts enumerate every act / out tile column
    exactly once (the act tensor and the output are the same tensors, filled through a different map)."""
    banks = 8
    p = R.plan_regions_stream(K, I, H, banks=banks)
    act_regions = sorted(p.act_tile_column(b, q) for b in range(banks) for q in range(p.ppb))
    act_stacked = sorted(p.stacked_act_tile_column(b, q, p.ppb) for b in range(banks) for q in range(p.ppb))
    assert act_regions == act_stacked == list(range(p.nt))
    cols0 = [b * p.C + r * p.cols for b in range(banks) for r in range(p.down_readers)]
    out_regions = sorted(p.out_tile_column(c0, c) for c0 in cols0 for c in range(p.cols))
    out_stacked = sorted(p.stacked_out_tile_column(c0, c) for c0 in cols0 for c in range(p.cols))
    assert out_regions == out_stacked == list(range(p.ht))
    # the header's arithmetic (regions_src.hpp regions_out_col) written out
    for c0 in cols0:
        bank, m0 = c0 // p.C, c0 - (c0 // p.C) * p.C
        assert [p.out_tile_column(c0, c) for c in range(p.cols)] == [bank + banks * (m0 + c) for c in range(p.cols)]


def test_wormhole_banks_refused_at_these_widths():
    """12 banks: I_local = 256 (8 tile columns) is not bank-local (the stacked layout refuses it too); I = 384 makes
    the gate|up bank-local but H/32 = 128 does not split over 12 banks; a 3072-wide hidden would."""
    p = R.plan_regions_stream(K, 256, H, banks=12)
    assert not p.bank_local and "12 banks" in p.reason and "I/32" in p.reason
    p = R.plan_regions_stream(K, 384, H, banks=12)
    assert not p.bank_local and "H/32" in p.reason and "I/32" not in p.reason
    ok, reason = R.regions_stream_feasible(3072, 384, 3072, banks=12)
    assert ok, reason
    ok, reason = R.regions_stream_feasible(K, 256, H, banks=8)
    assert ok and reason == ""


def test_packet_model_vs_stacked():
    """8x4 (ppb = 1): the regions-sourced stream issues the same packets as the stacked one (32 per expert and
    gate|up core, 8 per expert and down core); the 1024-wide block slice pays single-tile gate|up packets (8x)."""
    p = R.plan_regions_stream(K, 256, H, banks=8)
    s = R.stacked_packets_per_expert(K, 256, H, banks=8)
    assert p.packets_per_expert() == {"gate_up": 32, "down": 8}
    assert s["gate_up"] == 32 and s["down"] == 8 and s["cols"] == p.cols == 8
    p = R.plan_regions_stream(K, 1024, H, banks=8)
    s = R.stacked_packets_per_expert(K, 1024, H, banks=8)
    assert p.packets_per_expert() == {"gate_up": 2 * 4 * 128, "down": 32} and s == {
        "gate_up": 128,
        "down": 16,
        "cols": 4,
        "readers": 4,
    }
    logger.info(f"packets per expert and core at I = 1024: regions {p.packets_per_expert()} vs stacked {s}")
    p = R.plan_regions_stream(K, 2048, H, banks=8)
    assert p.ppb == 8 and p.cols == 2 and p.down_readers == 8 and p.packets_per_expert()["down"] == 64


def test_packet_knob_rules():
    p = R.plan_regions_stream(K, 256, H, banks=8, packet_tiles=4)
    assert p.bank_local and p.gate_packet_tiles == 4 and p.gate_pages_per_col == 32
    p = R.plan_regions_stream(K, 1024, H, banks=8, packet_tiles=4)
    assert not p.bank_local and "one tile" in p.reason
    p = R.plan_regions_stream(K, 256, H, banks=8, packet_tiles=3)
    assert not p.bank_local and "do not divide" in p.reason


def test_byte_model_8x4():
    """The stacked copy the finding paid for: 42 MoE layers x 72 experts x (gate|up [4096, 512] + down [256, 4096])
    in bfp8 = 9.41 GiB per chip; the bytes a decode step streams do not change (the active slices once)."""
    one = R.one_copy_bytes(72, K, 256, H, n_moe=42)
    assert abs(one / 2**30 - 9.41) < 0.01, one / 2**30
    per_expert = R.expert_slice_bytes(K, 256, H)
    assert per_expert == 128 * 16 * 1088 + 8 * 128 * 1088  # 3.34 MB
    active = R.active_read_bytes(28, K, 256, H)
    logger.info(
        f"stacked copy {one / 2**30:.2f} GiB per chip; 28 active experts read {active / 1e6:.1f} MB per layer per chip "
        f"({active / 300e3:.0f} us at 300 GB/s)"
    )
    assert R.one_copy_bytes(72, K, 256, H, n_moe=42, w13="bfp4") < one


def test_common_args_within_limit():
    p = R.plan_regions_stream(K, 256, H, banks=8)
    ca = p.common_args(72)
    assert ca["gate_up"] == 144 and ca["down"] == 72 and max(ca["gate_up"], ca["down"]) <= ca["limit"]
    hdr = PKG.parents[2] / "tt_metal" / "api" / "tt-metalium" / "kernel_types.hpp"
    if hdr.exists():  # pin the limit to the runtime's constant when the tree is around
        m = re.search(r"max_runtime_args\s*=\s*(\d+)", hdr.read_text())
        assert m and int(m.group(1)) == R.MAX_RUNTIME_ARGS, (m and m.group(1), R.MAX_RUNTIME_ARGS)


def test_interleaved_place_rule():
    assert R.interleaved_place(0, 8, 1088) == (0, 0)
    assert R.interleaved_place(7, 8, 1088) == (7, 0)
    assert R.interleaved_place(8, 8, 1088) == (0, 1088)
    assert R.interleaved_place(8 * 5 + 3, 8, 1088) == (3, 5 * 1088)
    assert R.tile_page(3, 5, 8) == 29


# ---------------------------------------------------------------------------------------------- kernel ABI
READERS = ("reader_gate_up", "reader_gate_up_dispatch", "reader_down", "reader_down_dispatch")
WRITERS = ("writer_gate_up", "writer_gate_up_dispatch", "writer_down", "writer_down_dispatch")
COMPUTES = ("compute_down", "compute_down_dispatch")


def _src(name: str) -> str:
    return (KERNELS / f"{name}.cpp").read_text()


def test_kernel_abi_regions_src():
    """Every REGIONS_SRC kernel: the define, the shared header, the exit barrier; readers take the per-expert
    addresses from the common runtime args (gate e, up E + e; down e) under the bank's NOC base at offset 0;
    writers map (bank, column) through regions_src.hpp; the down writers read the appended compile-time tail after
    the last accessor; the down computes index the k-major block."""
    hdr = (KERNELS / "regions_src.hpp").read_text()
    assert "regions_act_col" in hdr and "regions_out_col" in hdr
    for name in READERS + WRITERS + COMPUTES:
        assert "#ifdef REGIONS_SRC" in _src(name), name
    for name in READERS + WRITERS:  # the data-movement exit contract
        assert "noc_async_full_barrier();" in _src(name), name
    for name in READERS:
        src = _src(name)
        assert '#include "regions_src.hpp"' in src, name
        assert "get_noc_addr_from_bank_id<true>(bank_id, 0)" in src, name
        assert "(void)w_addr;" in src, name
        assert "src += src_step;" in src, name
    for name in ("reader_gate_up", "reader_gate_up_dispatch"):
        src = _src(name)
        assert "get_common_arg_val<uint32_t>(E + " in src, name
        assert "ppb == 1 ? page_bytes : ppb * tile_bytes" in src, name
    for name in ("reader_down", "reader_down_dispatch"):
        src = _src(name)
        assert "get_common_arg_val<uint32_t>(" in src and "C = bank_expert_stride / seg_bytes" in src, name
        assert "page_bytes == cols * tile_bytes" in src, name
    for name in ("writer_gate_up", "writer_gate_up_dispatch"):
        assert "regions_act_col(bank_id, q, i_tiles / ppb)" in _src(name), name
    for name in ("writer_down", "writer_down_dispatch"):
        assert "regions_out_col(col0, C, banks, c)" in _src(name), name
    wd = _src("writer_down")
    assert "get_compile_time_arg_val(CT_TAIL)" in wd and "get_compile_time_arg_val(CT_TAIL + 1)" in wd
    assert "CT_TAIL = base_args.next_compile_time_args_offset()" in wd
    assert "CT_TAIL = out_args.next_compile_time_args_offset()" in wd
    dw = _src("writer_down_dispatch")
    assert "get_compile_time_arg_val(base_args.next_compile_time_args_offset())" in dw
    for name in COMPUTES:
        src = _src(name)
        assert "matmul_tiles(cb_a, cb_w, k, k * cols + c, c);" in src, name
        assert "matmul_tiles(cb_a, cb_w, k, c * seg_tiles + k, c);" in src, name
    # the units kernels stay stacked-only (no define, no header)
    for name in ("reader_gate_up_dispatch_units", "writer_gate_up_dispatch_units"):
        assert "REGIONS_SRC" not in _src(name), name


def test_op_plumbing():
    """op.py packs what the kernels read: the common runtime args on the readers, the compute define on the down
    computes, the appended tail [h_tiles, banks] / [banks] on the down writers, the units refusal, the io lists
    with the regions lists first (today's order kept)."""
    assert 'REGIONS_DEFINE = ("REGIONS_SRC", "1")' in OP_SRC
    assert OP_SRC.count("reader_common=src.gate_up_addrs if src is not None else None") == 2
    assert OP_SRC.count("reader_common=src.down_addrs if src is not None else None") == 2
    assert OP_SRC.count("compute_defines=[REGIONS_DEFINE] if src is not None else ()") == 2
    assert "writer_ct += [self.H // TILE, self.banks]" in OP_SRC and "writer_ct += [self.banks]" in OP_SRC
    assert 'assert not (self.regions_src and units), "the units kernels take the stacked layout only"' in OP_SRC
    assert 'reader_extra["common_runtime_args"] = [int(a) for a in reader_common]' in OP_SRC
    assert "weights + [x, routing, act] + ([base] if base is not None else [])" in OP_SRC
    assert "weights + [act, routing, out] + ([base] if base is not None else [])" in OP_SRC
    assert "weights + [x_rm, routing, counts, act, table] + ([base] if self.ep else [])" in OP_SRC
    assert "weights + [act, table, counts, out] + ([base] if self.ep else [])" in OP_SRC
    # the gate|up ops plan the gate side only (H unknown there); the down ops pass their readers (whitespace-
    # insensitive: black wraps the longer call)
    sq = "".join(OP_SRC.split())
    assert sq.count("plan_regions_stream(K,I,None,banks=self.banks,w13=w_dtype") == 2
    assert sq.count("plan_regions_stream(H,I,H,banks=self.banks,down_readers=readers)") == 2


def test_model_plumbing():
    """tt/moe.py: the knob is read per call in ``_stream_serves`` (trace-safe like the glue knobs), the one-copy
    configuration skips the stacked upload and the stream tensorbins, the regions source is built after the lists,
    the regions fallback is forced by ``regions``; tt/mesh_model.py applies ``resolve_keep_stacked``."""
    assert (
        "def one_copy_config(" in MOE_SRC and "self.one_copy = one_copy_config(dense, stream, prefill_mode)" in MOE_SRC
    )
    assert "files = block_files if one_copy_config(dense, stream, mode) else files + block_files" in MOE_SRC
    assert "self.w13 = self.w2 = None" in MOE_SRC
    assert "def _init_regions_source(self):" in MOE_SRC and "RegionsSource(" in MOE_SRC
    assert 'if mode == "regions" and self._dispatch is not None:' in MOE_SRC
    assert "if self.w13 is None and not self._regions_serves(mode):" in MOE_SRC
    assert "elif self._dispatch is not None and (T > TILE or not self._stream_serves(T)):" in MOE_SRC
    assert "if not regions_keep_stacked() and self.w13 is not None:" in MOE_SRC
    assert "regions_src=regions, **kw" in MOE_SRC
    assert "keep = resolve_keep_stacked(fits, mode)" in MESH_MODEL_SRC
