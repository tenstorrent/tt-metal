# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host tests of the qsa_block kernel variants (mt_row, sel_spread, own-position q1 tails) without a device."""

from __future__ import annotations

import inspect
import os
import re
from types import SimpleNamespace

import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import program as fp
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import qsa_block

# ============================================================================
# was test_fused_qsa_mt_row_static.py
# QWEN38_QSA_MT_ROW without a device: on by default, the dispatch of main_tail's staging core kernels (flag off ->
# the default kernels; flag on with one lane -> the one-row kernels; more lanes, paged or a KV stage -> the default
# kernels), and the one-row kernels' text.
# ============================================================================

# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


KERNELS = qsa_block.fp.REPO_ROOT / "models/demos/blackhole/qwen38_flash_next/ttnn/fused/qsa_block/kernels"
SOURCE = inspect.getsource(qsa_block.main_tail)


def test_on_by_default():
    assert qsa_block.MT_ROW_ENV == "QWEN38_QSA_MT_ROW"
    assert qsa_block.mt_row_enabled({})
    assert not qsa_block.mt_row_enabled({qsa_block.MT_ROW_ENV: "0"})
    assert not qsa_block.mt_row_enabled({qsa_block.MT_ROW_ENV: "true"})
    assert qsa_block.mt_row_enabled({qsa_block.MT_ROW_ENV: "1"})


def test_admits_only_the_single_lane_decode():
    assert qsa_block.mt_row_admits(1, False, None)
    for rows in (2, 8, 32):
        assert not qsa_block.mt_row_admits(rows, False, None)
    assert qsa_block.mt_row_admits(1, True, None)
    assert not qsa_block.mt_row_admits(8, True, None)
    assert not qsa_block.mt_row_admits(1, False, object())


def test_dispatch_in_main_tail():
    # the kernel sets: the flag and the admission pick the one-row pair, everything else the default three kernels
    assert "mt_row = mt_row_enabled() and mt_row_admits(rows, paged, kv_stage)" in SOURCE
    assert "kernels += staging_row_kernels() if mt_row else staging_kernels()" in SOURCE
    assert 'MAIN_TAIL["reader_staging"]' in SOURCE and 'MAIN_TAIL["compute_staging"]' in SOURCE
    assert 'MAIN_TAIL_ROW["reader"]' in SOURCE and 'MAIN_TAIL_ROW["writer"]' in SOURCE
    row_kernels = SOURCE[
        SOURCE.index("def staging_row_kernels") : SOURCE.index("if kv_stage is None:\n        kernels +=")
    ]
    assert "compute_kernel" not in row_kernels and "MAIN_TAIL[" not in row_kernels
    assert row_kernels.count("fp.reader_kernel(") == 1 and row_kernels.count("fp.writer_kernel(") == 1
    assert qsa_block.MAIN_TAIL_ROW["reader"].endswith("main_tail_reader_staging_row.cpp")
    assert qsa_block.MAIN_TAIL_ROW["writer"].endswith("main_tail_writer_staging_row.cpp")
    assert qsa_block.MAIN_TAIL["reader_staging"].endswith("main_tail_reader_staging.cpp")
    assert qsa_block.CB_MT_ROW == 30


def test_default_kernels_are_untouched_by_the_flag():
    for path in (
        KERNELS.parent / "kernels_q1" / "main_tail_reader_staging.cpp",
        KERNELS / "main_tail_compute_staging.cpp",
        KERNELS.parent / "kernels_q1" / "main_tail_writer_staging.cpp",
    ):
        assert "CB_ROW" not in path.read_text()
    assert "CB_ROW = 30" in (KERNELS / "main_tail_cbs.h").read_text()


def _code(text):
    return re.sub(r"//[^\n]*|/\*.*?\*/", "", text, flags=re.S)


def test_one_row_kernels_text():
    reader = _code((KERNELS / "main_tail_reader_staging_row.cpp").read_text())
    writer = _code((KERNELS / "main_tail_writer_staging_row.cpp").read_text())
    assert "sem_krot.wait_min(1)" in reader and "sem_knorm.wait_min(1)" in reader
    header = _code((KERNELS / "main_tail_row.h").read_text())
    assert "sem_done.wait(1)" in reader and "SEM_ROW_DONE" in writer and "HEAD_TILES / 2" in reader
    assert "MT_ROW_SPLIT" not in reader + writer
    assert reader.count("cb_push_back(CB_ROW, 1)") == 2 and "canonical_word(s[k])" in header
    assert reader.index("build_tiles(pack_l1, 0, HEAD_TILES, lane") < reader.index(
        "sem_krot.wait_min(1)"
    )  # the V half first
    assert "copy_tile_row" not in reader and "CB_STG" not in reader and "CB_ONES" not in reader
    # per half one 512 B cache write and the 32-byte staging chunks, one barrier, no 32-row loop
    assert writer.count("noc_async_write(") == 2 and writer.count("noc_async_write_barrier()") == 1
    assert "cache.get_noc_addr(cache_row, h * HALF_BYTES)" in writer and "cb_wait_front(CB_ROW, h + 1)" in writer
    assert "chunk_offset(slot, half)" in writer and "HALF_BYTES = QUERY_ROW_BYTES / 2" in writer
    assert "TILE_ROWS" not in writer and "CB_RM" not in writer and "CB_STGW" not in writer


# ============================================================================
# was test_fused_qsa_sel_spread_static.py
# QWEN38_QSA_SEL_SPREAD without a device: off by default, the core count knob, the segment cut (whole 16-column blocks,
# every column once, in order), and the kernel and program wiring.
# ============================================================================

# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


def test_flag_default_on_and_cores():
    assert qsa_block.SEL_SPREAD_ENV == "QWEN38_QSA_SEL_SPREAD"
    assert qsa_block.sel_spread_enabled({}) and qsa_block.sel_spread_enabled({qsa_block.SEL_SPREAD_ENV: "1"})
    assert not qsa_block.sel_spread_enabled({qsa_block.SEL_SPREAD_ENV: "0"})
    assert qsa_block.sel_spread_cores({}) == qsa_block.SEL_SPREAD_CORES
    assert qsa_block.sel_spread_cores({qsa_block.SEL_SPREAD_CORES_ENV: "65"}) == 65
    assert qsa_block.sel_spread_cores({qsa_block.SEL_SPREAD_CORES_ENV: "500"}) == 65
    assert qsa_block.sel_spread_cores({qsa_block.SEL_SPREAD_CORES_ENV: "0"}) == 1


def test_segments_cover_every_column_once():
    for cores in (1, 8, 16, 32, 65):
        segments = qsa_block.sel_spread_segments(cores)
        assert len(segments) == 2 * cores
        position = 0
        for first, count in segments:
            assert first == position and count > 0 and first % 16 == 0 and count % 16 == 0
            position += count
        assert position == qsa_block.SELECTION_WIDTH
        widths = {count for _, count in segments}
        assert max(widths) - min(widths) <= 16


def test_wiring():
    source = inspect.getsource(qsa_block.selection_row)
    assert "rows == 1 and sel_spread_enabled()" in source
    assert source.count("SEL_HALF") == 1 and "fp.reader_kernel, 0" in source and "fp.writer_kernel, 1" in source
    kernel = (fp.REPO_ROOT / qsa_block.SELECTION_ROW_SPREAD).read_text()
    assert "SEL_HALF" in kernel and "SEL_MAXC" in kernel and "#ifdef PAGED" in kernel
    assert "(ids_l1[i >> 2] << 2)" not in kernel and "const uint32_t id4 = ids_l1[i >> 2] << 2;" in kernel


# ============================================================================
# was test_qsa_tail_q1_static.py
# The own-position kernels of index_tail / main_tail and the two-RISC selection row without a device: the programs of
# index_tail, main_tail and selection_row (rows 1, 8, 32; paged / unpaged / live) use the kernels in kernels_q1/ for the
# lane-position reads and the row loop.  The programs are built on fake tensors (``fp.run_program`` captures the
# descriptor).
# ============================================================================

# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


KERNELS_Q1 = fp.REPO_ROOT / fp.KERNEL_ROOT / "qsa_block" / "kernels_q1"
BF16, U32, I32 = ttnn.bfloat16, ttnn.uint32, ttnn.int32
TILE, RM = ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT


class Fake:
    _next = 0x1000

    def __init__(self, shape, dtype, layout, mesh):
        self.shape, self.padded_shape, self.dtype, self.layout, self._mesh = (
            tuple(shape),
            tuple(shape),
            dtype,
            layout,
            mesh,
        )
        if layout == TILE:
            padded = list(shape)
            padded[-2] = -(-padded[-2] // 32) * 32
            self.padded_shape = tuple(padded)
        Fake._next += 0x10000
        self._addr = Fake._next

    def device(self):
        return self._mesh

    def buffer_address(self):
        return self._addr


def _mesh():
    return SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=11, y=10), id=1)


def _capture(module, monkeypatch):
    captured = []
    monkeypatch.setattr(module.fp, "accessor_args", lambda t: [t.dtype == U32, len(t.shape)] if False else [1, 2, 3])
    monkeypatch.setattr(module.fp, "allocate", lambda shape, dtype, layout, mesh, *a: Fake(shape, dtype, layout, mesh))
    monkeypatch.setattr(module.fp, "program_meta", lambda *a, **k: None)
    monkeypatch.setattr(
        module.fp, "run_program", lambda io, descriptor, meta=None: captured.append(descriptor) or io[-1]
    )
    monkeypatch.setattr(
        module, "noc_coords", lambda mesh, cores: {(c.x, c.y): (c.x + 1, c.y + 2) for c in cores} | {(0, 0): (1, 2)}
    )
    return captured


def _ranges(crs):
    return [(r.start.x, r.start.y, r.end.x, r.end.y) for r in crs.ranges()]


def _runtime_args(view):
    """{(x, y): [args]} of a kernel's runtime args (a 2-D view indexed [x][y]; cores of the 11 x 10 fake grid)."""

    out = {}
    for x in range(len(view)):
        column = view[x]
        for y in range(10):
            try:
                args = list(column[y])
            except (IndexError, RuntimeError):
                continue
            if args:
                out[(x, y)] = args
    return out


def _parts(descriptor):
    kernels = [
        (
            k.kernel_source,
            _ranges(k.core_ranges),
            list(k.compile_time_args),
            sorted(map(tuple, k.defines)),
            _runtime_args(k.runtime_args),
            type(k.config).__name__,
        )
        for k in descriptor.kernels
    ]
    cbs = []
    for cb in descriptor.cbs:
        cbs.append(
            (
                cb.total_size,
                _ranges(cb.core_ranges),
                [(f.buffer_index, f.page_size) for f in cb.format_descriptors],
            )
        )
    return kernels, cbs


def _it_args(mesh, rows):
    t = lambda shape, dtype=BF16, layout=TILE: Fake(shape, dtype, layout, mesh)
    position = SimpleNamespace(kv_block_start=t((1, 1, 1, rows), U32, RM), kv_row_hit=t((1, rows, 32, 1)))
    return (
        t((1, 1, rows, 128)),
        t((1, 1, rows, 128)),
        position,
        t((1, 1, 1, 128)),
        t((1, 1, 1, 128)),
        *[t((1, 1, rows, 64)) for _ in range(4)],
        t((1, rows, 32, 128)),
        t((rows, 1, 64, 128)),
    )


def _mt_args(mesh, rows, paged):
    t = lambda shape, dtype=BF16, layout=TILE: Fake(shape, dtype, layout, mesh)
    position = SimpleNamespace(kv_block_start=t((1, 1, 1, rows), U32, RM), kv_row_hit=t((1, rows, 32, 1)))
    heads = qsa_block.LOCAL_HEADS
    kv = t((1, 1, rows * 512, 512), BF16, RM)
    kwargs = dict(lane_rows=512) if not paged else dict(page_table=t((1, 1, 32, 16), I32, RM), page_size=32)
    args = (
        t((1, 1, rows, 2 * heads * 256)),
        t((1, 1, rows, 256)),
        t((1, 1, rows, 256)),
        position,
        t((1, 1, 1, 256)),
        t((1, 1, 1, 256)),
        t((1, 1, rows, 64)),
        t((1, 1, rows, 64)),
        t((1, rows, 32, 512)),
        kv,
    )
    return args, kwargs


def _sel_args(mesh, rows, paged):
    t = lambda shape, dtype=U32, layout=RM: Fake(shape, dtype, layout, mesh)
    args = (t((1, 1, rows, 512)), t((1, 1, 1, 32)), t((1, 1, 32, 2048)), t((1, 1, 32, 2080)), t((1, 1, 32, 2080)))
    kwargs = dict(page_table=t((1, 1, 32, 16), I32), page_size=32) if paged else {}
    return args, kwargs


def _programs(module, monkeypatch):
    captured = _capture(module, monkeypatch)
    Fake._next = 0x1000  # the same addresses in every build
    mesh = _mesh()
    out = {}
    for rows in (1, 8, 32):
        module.index_tail(*_it_args(mesh, rows))
        out[("index_tail", rows)] = _parts(captured[-1])
        for paged in (False, True):
            args, kwargs = _mt_args(mesh, rows, paged)
            module.main_tail(*args, **kwargs)
            out[("main_tail", rows, paged)] = _parts(captured[-1])
            args, kwargs = _sel_args(mesh, rows, paged)
            module.selection_row(*args, **kwargs)
            out[("selection_row", rows, paged)] = _parts(captured[-1])
        if rows > 1:
            args, _ = _sel_args(mesh, rows, False)
            module.selection_row(*args, live_rows=rows // 2)
            out[("selection_row", rows, "live")] = _parts(captured[-1])
    return out


def test_programs_use_the_q1_kernels(monkeypatch):
    programs = _programs(qsa_block, monkeypatch)
    for key, (kernels, _) in programs.items():
        sources = sorted(os.path.basename(k[0]) for k in kernels if "kernels_q1" in k[0])
        name = key[0]
        if name == "index_tail":
            assert {"index_tail_reader_norm.cpp", "index_tail_writer_norm.cpp", "index_tail_writer_rope.cpp"} <= set(
                sources
            ), key
        elif name == "main_tail" and key[1] == 1:
            assert sources == [], key  # the one-row staging kernels are main_tail's own
        elif name == "main_tail":
            assert {"main_tail_reader_staging.cpp", "main_tail_writer_staging.cpp"} <= set(sources), key
        elif key[1] == 1:
            assert sources == [], key  # selection_row rows == 1: the spread form
        else:
            assert sources == ["selection_row_2r.cpp"] * 2, key


def test_selection_row_splits_the_row_loop_over_both_risc(monkeypatch):
    programs = _programs(qsa_block, monkeypatch)
    for key, (kernels, cbs) in programs.items():
        if key[0] != "selection_row" or key[1] == 1:
            continue
        k_on, k_on2 = kernels
        assert k_on[0] == k_on2[0] and k_on[0].endswith("kernels_q1/selection_row_2r.cpp")
        assert k_on[1:3] == k_on2[1:3] and k_on[4] == k_on2[4]  # cores, compile args, runtime args
        assert k_on[5] == "ReaderConfigDescriptor" and k_on2[5] == "WriterConfigDescriptor"
        assert {dict(k[3])["SEL2_HALF"] for k in (k_on, k_on2)} == {"0", "1"}
        stride = int(dict(k_on[3])["SEL2_STRIDE"])
        assert stride % 64 == 0 and stride >= 256 + 1024 + 128 + 3 * 1152
        assert cbs[0][0] == 2 * stride


def test_kernel_copies_include_the_own_position_header():
    for name in (
        "index_tail_reader_norm",
        "index_tail_writer_norm",
        "index_tail_writer_rope",
        "main_tail_reader_staging",
        "main_tail_writer_staging",
    ):
        new = (KERNELS_Q1 / f"{name}.cpp").read_text()
        assert "read_own_position" in new and "read_positions" not in new
        assert new.count("read_own_position(") == 1
        assert new.count('#include "own_position.h"') == 1 and '"positions.h"' not in new


def test_own_position_header_reads_only_its_lane():
    import re

    header = (KERNELS_Q1 / "own_position.h").read_text()
    code = lambda s: re.sub(r"\s+", "", re.sub(r"//[^\n]*", "", s))
    for piece in (
        "if(e[0]==0x3F80){row=r;}",
        "for(uint32_tr=0;r<TILE_ROWS;++r){",
        "chunk_offset(r,0)",
        "noc_async_read_page(0,block_start,scratch);",
    ):
        assert piece in code(header), piece
    assert "noc_async_read_page(lane,hit,scratch+128);" in code(header)
    assert "starts[lane]+row" in code(header) and "positions[lane]=p;" in code(header)
