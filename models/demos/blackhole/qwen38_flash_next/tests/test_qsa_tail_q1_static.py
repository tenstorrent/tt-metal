# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""QWEN38_QSA_OWNPOS and QWEN38_QSA_SELROW2 without a device: both off by default; with the flags off the programs of
index_tail, main_tail and selection_row (rows 1, 8, 32; paged / unpaged / live) are what the builders made before the
flags (kernels in kernels/, same defines, runtime args, core ranges, CBs); with the flags on exactly the named kernels
change.  The programs are built on fake tensors (``fp.run_program`` captures the descriptor)."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import program as fp
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import qsa_block

PKG = "models.demos.blackhole.qwen38_flash_next.ttnn.fused"
# the upstream builder the flags extend (9bb1f0f698c, with the paged index_tail, spread and one-row staging forms)
ORIG = Path(
    os.environ.get(
        "Q1_ORIG_INIT", "/home/ttuser/atupe/qwen38_work/runs/plan_nov06/p1_pull/q1_orig/qsa_block__init__.py"
    )
)
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


def _load_orig(monkeypatch):
    if not ORIG.is_file():
        return None
    name = f"{PKG}.qsa_block.orig_q1"
    spec = importlib.util.spec_from_file_location(name, ORIG, submodule_search_locations=None)
    module = importlib.util.module_from_spec(spec)
    import sys

    from models.demos.blackhole.qwen38_flash_next.ttnn.fused import registry

    monkeypatch.setattr(registry, "register", lambda *a, **k: None)  # the copy must not register its kernels twice

    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


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


def _programs(module, monkeypatch, flags):
    for name, value in flags.items():
        monkeypatch.setenv(name, value)
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


FLAGS_OFF = {qsa_block.OWNPOS_ENV: "0", qsa_block.SELROW2_ENV: "0"}
FLAGS_ON = {qsa_block.OWNPOS_ENV: "1", qsa_block.SELROW2_ENV: "1"}


def test_flags_default_on():
    assert qsa_block.OWNPOS_ENV == "QWEN38_QSA_OWNPOS" and qsa_block.SELROW2_ENV == "QWEN38_QSA_SELROW2"
    for enabled in (qsa_block.ownpos_enabled, qsa_block.selrow2_enabled):
        assert enabled({}) and not enabled({"QWEN38_QSA_OWNPOS": "0", "QWEN38_QSA_SELROW2": "0"})
    assert not qsa_block.ownpos_enabled({qsa_block.OWNPOS_ENV: "0"}) and qsa_block.ownpos_enabled(
        {qsa_block.SELROW2_ENV: "0"}
    )
    assert not qsa_block.selrow2_enabled({qsa_block.SELROW2_ENV: "0"}) and qsa_block.selrow2_enabled(
        {qsa_block.OWNPOS_ENV: "0"}
    )


def test_flags_off_programs_use_the_original_kernels(monkeypatch):
    programs = _programs(qsa_block, monkeypatch, FLAGS_OFF)
    for key, (kernels, _) in programs.items():
        assert all("/kernels/" in k[0] and "kernels_q1" not in k[0] for k in kernels), key
    assert len(programs[("selection_row", 8, False)][0]) == 1
    assert len(programs[("selection_row", 32, "live")][0]) == 1


def test_flags_off_programs_equal_the_pre_flag_builders(monkeypatch):
    orig = _load_orig(monkeypatch)
    if orig is None:
        pytest.skip(f"{ORIG} is absent")
    assert not hasattr(orig, "ownpos_enabled")
    assert _programs(orig, monkeypatch, {}) == _programs(qsa_block, monkeypatch, FLAGS_OFF)


def _same_but_sources(a, b):
    ka, ca = a
    kb, cb = b
    assert ca == cb and len(ka) == len(kb)
    return [i for i, (x, y) in enumerate(zip(ka, kb)) if x != y], [(x[1:], y[1:]) for x, y in zip(ka, kb)]


def test_ownpos_on_swaps_only_the_lane_position_kernels(monkeypatch):
    off = _programs(qsa_block, monkeypatch, FLAGS_OFF)
    on = _programs(qsa_block, monkeypatch, {**FLAGS_OFF, qsa_block.OWNPOS_ENV: "1"})
    for key in off:
        name = key[0]
        if name == "selection_row":
            assert off[key] == on[key], key
            continue
        changed = [i for i, (x, y) in enumerate(zip(off[key][0], on[key][0])) if x != y]
        assert off[key][1] == on[key][1] and len(off[key][0]) == len(on[key][0])
        for i in changed:  # only the source differs
            assert off[key][0][i][1:] == on[key][0][i][1:], key
            assert "kernels_q1" in on[key][0][i][0] and "kernels_q1" not in off[key][0][i][0]
        swapped = sorted(os.path.basename(on[key][0][i][0]) for i in changed)
        if name == "index_tail":
            assert swapped == [
                "index_tail_reader_norm.cpp",
                "index_tail_writer_norm.cpp",
                "index_tail_writer_rope.cpp",
            ], key
        elif key[1] == 1:
            # the one-row staging kernels (rows == 1, paged or not since upstream's paged staging row) are the ones main_tail already uses: unchanged
            assert swapped == [], key
        else:
            assert swapped == ["main_tail_reader_staging.cpp", "main_tail_writer_staging.cpp"], key


def test_selrow2_on_splits_the_row_loop_over_both_risc(monkeypatch):
    off = _programs(qsa_block, monkeypatch, FLAGS_OFF)
    on = _programs(qsa_block, monkeypatch, {**FLAGS_OFF, qsa_block.SELROW2_ENV: "1"})
    for key in off:
        if key[0] != "selection_row":
            assert off[key] == on[key], key
            continue
        if key[1] == 1:  # the spread form (rows == 1, paged or not since upstream) keeps its own kernels
            assert off[key] == on[key], key
            continue
        (k_off,), (k_on, k_on2) = off[key][0], on[key][0]
        assert k_on[0] == k_on2[0] and k_on[0].endswith("kernels_q1/selection_row_2r.cpp")
        assert k_on[1:3] == k_off[1:3] and k_on[4] == k_off[4] == k_on2[4]  # cores, compile args, runtime args
        assert k_on[5] == "ReaderConfigDescriptor" and k_on2[5] == "WriterConfigDescriptor"
        halves = {dict(k[3])["SEL2_HALF"] for k in (k_on, k_on2)}
        assert halves == {"0", "1"}
        extra = set(k_on[3]) - set(k_off[3])
        assert {n for n, _ in extra} == {"SEL2_HALF", "SEL2_STRIDE"} and set(k_off[3]) <= set(k_on[3])
        stride = int(dict(k_on[3])["SEL2_STRIDE"])
        assert stride % 64 == 0 and stride >= 256 + 1024 + 128 + 3 * 1152
        assert on[key][1][0][0] == 2 * stride and on[key][1][0][1] == off[key][1][0][1]


def test_kernel_copies_differ_only_in_the_position_call():
    import re

    base = fp.REPO_ROOT / fp.KERNEL_ROOT / "qsa_block" / "kernels"
    for name in (
        "index_tail_reader_norm",
        "index_tail_writer_norm",
        "index_tail_writer_rope",
        "main_tail_reader_staging",
        "main_tail_writer_staging",
    ):
        old = (base / f"{name}.cpp").read_text()
        new = (KERNELS_Q1 / f"{name}.cpp").read_text()
        norm = lambda s: re.sub(r"\s+", "", s.replace('"positions.h"', "").replace('"own_position.h"', ""))
        assert "read_positions" in old and "read_own_position" in new and "read_positions" not in new
        strip = lambda s: re.sub(
            r"(tile_rows::read_positions\(.*?\);)|(for\(uint32_ti=0;i<lane_count;\+\+i\)\{//QWEN.*?\}\n?)",
            "",
            s,
            flags=re.S,
        )
        assert new.count("read_own_position(") == 1
        assert new.count('#include "own_position.h"') == 1 and '"positions.h"' not in new


def test_own_position_header_matches_read_positions_scan():
    import re

    header = (KERNELS_Q1 / "own_position.h").read_text()
    old = (fp.REPO_ROOT / fp.KERNEL_ROOT / "qsa_block/kernels/positions.h").read_text()
    code = lambda s: re.sub(r"\s+", "", re.sub(r"//[^\n]*", "", s))
    for piece in (
        "if(e[0]==0x3F80){row=r;}",
        "for(uint32_tr=0;r<TILE_ROWS;++r){",
        "chunk_offset(r,0)",
        "noc_async_read_page(0,block_start,scratch);",
    ):
        assert piece in code(header) and piece in code(old), piece
    assert "noc_async_read_page(lane,hit,scratch+128);" in code(header)
    assert "starts[lane]+row" in code(header) and "positions[lane]=p;" in code(header)
