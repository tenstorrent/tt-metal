# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The sort-core router top-k (``router_tail_sc``, QWEN38_ROUTER_SORTCORE=1) without a device: flag off -> today's
``router_tail.program_parts`` unchanged; flag on (rows == 1) -> the two-core form; its kernels keep today's sort, softmax,
sum and division verbatim; the CB / argument / semaphore contract between the Python side and the kernels."""

from __future__ import annotations

import inspect
import re
import types

import pytest

import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn import fused
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import program as fp
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import router_tail as rt
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import router_tail_sc as sc

SOURCES = {name: (fp.REPO_ROOT / path).read_text() for name, path in sc.KERNELS.items()}
TODAY = (fp.REPO_ROOT / rt.KERNELS["compute"]).read_text()


def _code(text):
    return re.sub(r"\s+", "", re.sub(r"//[^\n]*|/\*.*?\*/", "", text, flags=re.S))


class _Logits:
    dtype = ttnn.float32

    def device(self):
        return "mesh"

    def buffer_address(self):
        return 0x1000


@pytest.fixture
def stubs(monkeypatch):
    monkeypatch.setattr(fp, "accessor_args", lambda tensor: [])
    monkeypatch.setattr(
        sc.gr_read, "noc_map", lambda mesh: {(x, y): (x + 18, y + 18) for x in range(8) for y in range(8)}
    )
    monkeypatch.setattr(sc.fp, "tensor_bytes", lambda tensor: 1)
    return types.SimpleNamespace(rect=ttnn.CoreRange(ttnn.CoreCoord(2, 3), ttnn.CoreCoord(5, 7)))


def test_flag_off_is_todays_program_parts_unchanged(monkeypatch):
    seen = {}

    def today(*args, **kwargs):
        seen["call"] = (args, kwargs)
        return "today"

    monkeypatch.setattr(rt, "program_parts", today)
    assert sc.enabled(1, {})  # on by default
    for env in ({sc.SORTCORE_ENV: "0"}, {sc.SORTCORE_ENV: ""}):
        assert not sc.enabled(1, env)
    monkeypatch.setenv(sc.SORTCORE_ENV, "0")
    got = sc.program_parts("l", "i", "s", "x", rows=1, top_k=10, rectangle="r", sem_base=3)
    assert got == "today"
    assert seen["call"] == ((("l", "i", "s", "x")), dict(rows=1, top_k=10, rectangle="r", sem_base=3))


@pytest.mark.parametrize("rows", [2, 5, 8, 32, 128])
def test_flag_on_other_rows_fall_back(monkeypatch, rows):
    monkeypatch.setenv(sc.SORTCORE_ENV, "1")
    monkeypatch.setattr(rt, "program_parts", lambda *a, **k: "today")
    assert not sc.enabled(rows)
    assert sc.program_parts("l", "i", "s", "x", rows=rows, top_k=10) == "today"


def test_flag_on_needs_the_lane_form(monkeypatch):
    monkeypatch.setenv(sc.SORTCORE_ENV, "1")
    monkeypatch.setenv(rt.LANES_ENV, "0")
    assert not sc.enabled(1)
    monkeypatch.delenv(rt.LANES_ENV)
    assert sc.enabled(1)


def test_flag_on_rows_1_is_the_two_core_form(monkeypatch, stubs):
    monkeypatch.setenv(sc.SORTCORE_ENV, "1")
    monkeypatch.setenv(sc.FAST_ENV, "0")  # the base two-core form (the fast tail is the default since x42)
    monkeypatch.setattr(rt, "program_parts", lambda *a, **k: pytest.fail("fell back"))
    kernels, cbs, semaphores = sc.program_parts(
        _Logits(), _Logits(), _Logits(), _Logits(), rows=1, top_k=10, rectangle=stubs.rect, sem_base=2
    )
    assert len(kernels) == 5 and len(semaphores) == 1
    assert semaphores[0].id == 2  # sem_base
    by_source = {k.kernel_source.rsplit("/", 1)[-1]: k for k in kernels}
    assert set(by_source) == {
        "reader_rt_main.cpp",
        "writer_router_tail.cpp",
        "compute_rt_main.cpp",
        "reader_rt_sort.cpp",
        "compute_rt_sort.cpp",
    }
    cores = {}
    for name, kernel in by_source.items():
        ranges = list(kernel.core_ranges.ranges())
        assert len(ranges) == 1 and ranges[0].start == ranges[0].end, name
        cores[name] = (ranges[0].start.x, ranges[0].start.y)
    a, b = (2, 3), (3, 3)  # the first two cores of the rectangle's first row
    assert {
        cores[n] for n in cores if n in ("reader_rt_main.cpp", "writer_router_tail.cpp", "compute_rt_main.cpp")
    } == {a}
    assert {cores[n] for n in cores if n in ("reader_rt_sort.cpp", "compute_rt_sort.cpp")} == {b}
    assert [(r.start.x, r.start.y) for r in semaphores[0].core_ranges.ranges()] == [a]
    # core B's reader gets core A's NoC coordinates (the stub map adds 18); both compute kernels get pass 1 / token 1
    sort_reader = by_source["reader_rt_sort.cpp"]
    assert list(sort_reader.runtime_args[3][3]) == [0x1000, 0x1000, 0, 2 + 18, 3 + 18]
    named = dict(by_source["reader_rt_main.cpp"].named_compile_time_args)
    assert named["sem_id"] == 2 and named["top_k"] == 10 and named["Wt"] == 16
    # today's CB table on both cores
    assert [(c.total_size) for c in cbs] == [
        (rt.TOKEN_PAGE_BYTES if n in rt.TOKEN_CBS else fp.TILE_BYTES[ttnn.float32 if n == "cb_in0" else d]) * p
        for n, _i, d, p in rt.CBS
    ]
    assert all(len(list(c.core_ranges.ranges())) == 1 for c in cbs)


def test_python_side_matches_the_kernels():
    names = {}
    for key, text in SOURCES.items():
        names[key] = set(re.findall(r'get_named_compile_time_arg_val\("([A-Za-z_0-9]+)"\)', text))
        assert names[key], key
        assert names[key] <= set(rt.CB_INDEX) | {"Wt", "top_k", "stage_pages", "sem_id"}, key
    args = lambda key: [int(i) for i in re.findall(r"get_arg_val<uint32_t>\((\d)\)", SOURCES[key])]  # noqa: E731
    assert args("reader_main") == [0, 1]  # logits_addr, tile_row
    assert args("reader_sort") == [0, 1, 2, 3, 4]  # logits, index, tile_row, a_x, a_y
    assert args("writer") == list(range(len(rt.WRITER_ARGS)))
    assert 1 in args("compute_main") and 0 in args("compute_sort")  # token_mask / pass_mask
    source = inspect.getsource(sc.two_core_parts)
    assert "[(core_a, [logits_addr, 0, 1])]" in source
    assert "[(core_b, [logits_addr, index_addr, 0, a_x, a_y])]" in source
    assert "[(core_a, [SORT_PASS_MASK, 1])]" in source and "[(core_b, [SORT_PASS_MASK, 1])]" in source
    assert sc.SORT_PASS_MASK == 1 and 0 in rt.PASS_TOKENS[0]  # pass 0 holds token 0
    assert rt.lanes_plan(1) == [(sc.SORT_PASS_MASK, 1)]  # today's one-core lane plan for one row


def test_sort_core_keeps_todays_sort_verbatim():
    sort = SOURCES["compute_sort"]
    for piece in (
        "FORCE_INLINE void frt_load_tile(uint32_t cb_probs, uint32_t cb_index, uint32_t w, uint32_t slot) {\n"
        "    reconfig_data_format_srca(cb_probs);\n    transpose_init(cb_probs);\n    transpose_tile(cb_probs, w, slot);\n"
        "    reconfig_data_format_srca(cb_index);\n    copy_init(cb_index);\n    copy_tile(cb_index, w, slot + 2);\n}",
        "FORCE_INLINE void frt_sort64(uint32_t pass_mask) {\n    if (pass_mask == 0) {\n"
        "        ckernel::topk_local_sort<false>(0, 0 /* largest */, 5 /* end_phase */);\n    } else {\n"
        "        topk_local_sort_lanes<false>(0, 0 /* largest */, 5 /* end_phase */, pass_mask);\n    }\n}",
        "compute_kernel_hw_startup(cb_probs, cb_index, cb_vals_t);\n    ckernel::topk_tile_init();",
        "tile_regs_acquire();\n        for (uint32_t w = 0; w < Wt; ++w) {\n"
        "            frt_load_tile(cb_probs, cb_index, w, (w == 0) ? 0 : 1);\n            if (w != 0) {\n"
        "                frt_sort64(pass_mask);\n            }\n        }\n        tile_regs_commit();",
        "pack_tile(2, cb_idx_t);",
    ):
        assert _code(piece) in _code(sort), piece
    # x18's raw-logit copy pass is the same block as today's FRT_RAWSORT one
    copy_pass = "tile_regs_acquire();\n            for (uint32_t wt8 = 0; wt8 < ndst; wt8++) {\n                copy_tile(cb_in0, wt + wt8, wt8);"
    assert _code(copy_pass) in _code(sort) and _code(copy_pass) in _code(TODAY)
    assert '#include "../../router_tail/kernels/topk_lanes.h"' in sort
    assert "softmax" not in _code(sort).replace("fz_rt_c_softmax", "")  # no softmax on the sort core
    assert "wait_front(1)" not in sort and "cb_max_scaler" not in sort
    # the sort keys are logit + 128 (positive like the probabilities: the SFPSWAP breaks negative ties differently)
    assert "SORT_SHIFT_BITS = 0x43000000" in sort and "add_unary_tile(wt8, SORT_SHIFT_BITS)" in sort


def test_main_core_keeps_todays_softmax_sum_and_div_verbatim():
    main = SOURCES["compute_main"]
    softmax_begin = TODAY.index('FUSED_ZONE("fz_rt_c_softmax");')
    softmax_end = TODAY.index("#else", TODAY.index("probs.push_back(ndst);"))
    softmax = TODAY[softmax_begin:softmax_end]
    softmax = softmax[: softmax.rindex("    }")]
    assert _code(softmax) in _code(main)
    tail = TODAY[TODAY.index('FUSED_ZONE("fz_rt_c_sum");') :]
    assert _code(tail) in _code(main)
    assert "topk" not in _code(main) and "frt_sort64" not in main
    assert "transpose" not in _code(main)


def test_reader_gathers_by_the_tile_face_formula():
    reader = SOURCES["reader_main"]
    assert "idx[k * 16]" in reader  # [k, token] tile, token 0 (row k of face 0)
    assert "p[(e >> 5) * 1024 + ((e >> 4) & 1u) * 256 + (e & 15u)]" in reader
    assert "noc_semaphore_wait_min(sem, 1)" in reader and "noc_semaphore_set(sem, 0)" in reader
    sender = SOURCES["reader_sort"]
    assert (
        sender.index("noc_async_write(")
        < sender.index("noc_async_write_barrier()")
        < sender.index("noc_semaphore_inc(")
    )
    assert "get_noc_addr(a_x, a_y, src)" in sender  # the same L1 address on core A


def _simulate_gather(probs_row, idx_k):
    """Torch-free model of the reader's gather over the tile layout (16 tiles x 1024 words, 4 faces of 16 x 16)."""

    tiles = [[0.0] * 1024 for _ in range(16)]
    for e, value in enumerate(probs_row):
        face = (e >> 4) & 1
        tiles[e >> 5][face * 256 + (e & 15)] = value  # row 0 of the tile
    flat = [w for t in tiles for w in t]
    return [flat[(e >> 5) * 1024 + ((e >> 4) & 1) * 256 + (e & 15)] for e in idx_k]


def test_gather_model_picks_probability_of_each_index():
    probs = [float(i) + 0.5 for i in range(512)]
    idx = [511, 0, 17, 33, 256, 300, 31, 32, 64, 480]
    assert _simulate_gather(probs, idx) == [probs[i] for i in idx]


def test_not_a_registered_kernel_and_off_by_default():
    assert "router_tail_sc" not in {e.name for e in fused.REGISTRY.values()} if hasattr(fused, "REGISTRY") else True
    assert sc.SORTCORE_ENV == "QWEN38_ROUTER_SORTCORE"
    moe = (fp.REPO_ROOT / "models/demos/blackhole/qwen38_flash_next/ttnn/fused/moe_dense/__init__.py").read_text()
    assert "rt_sc.program_parts(" in moe and "rt.program_parts(" not in moe
