# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The shortened sort-core tail (QWEN38_ROUTER_TAIL_FAST=1) without a device: off by default, option plumbing, the
kernels keep the sort-core form's softmax / sum / div / sort text, and the fp32-index decode."""

from __future__ import annotations

import re

import pytest

import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import program as fp
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import router_tail_sc as sc

K = lambda name: (fp.REPO_ROOT / sc.FAST_KERNELS[name]).read_text()  # noqa: E731
BASE = lambda name: (fp.REPO_ROOT / sc.KERNELS[name]).read_text()  # noqa: E731


def _code(text):
    return re.sub(r"\s+", "", re.sub(r"//[^\n]*|/\*.*?\*/", "", text, flags=re.S))


def test_on_by_default_and_option_plumbing():  # default on since x42; QWEN38_ROUTER_TAIL_FAST=0 restores the sort-core tail
    assert sc.fast_opts({}) == sc.FAST_DEFAULT_OPTS and sc.fast_opts({sc.FAST_ENV: "0"}) == ()
    assert sc.fast_opts({sc.FAST_ENV: "1"}) == sc.FAST_DEFAULT_OPTS
    assert sc.fast_opts({sc.FAST_ENV: "1", sc.FAST_OPTS_ENV: "row0,split"}) == ("row0", "split")
    with pytest.raises(ValueError):  # allow-pytest.raises: no-device Python-level check, no device error text to triage
        sc.fast_opts({sc.FAST_ENV: "1", sc.FAST_OPTS_ENV: "bogus"})
    assert set(sc.FAST_DEFAULT_OPTS) <= set(sc.FAST_OPTS) and "incr" not in sc.FAST_DEFAULT_OPTS


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
    monkeypatch.setattr(sc, "float_index_template", lambda mesh: _Logits())
    return ttnn.CoreRange(ttnn.CoreCoord(2, 3), ttnn.CoreCoord(5, 7))


def test_fast_program_has_the_split_index_kernel_and_defines(monkeypatch, stubs):
    monkeypatch.setenv(sc.FAST_ENV, "1")
    kernels, cbs, semaphores = sc.two_core_parts(
        _Logits(), _Logits(), _Logits(), _Logits(), rows=1, top_k=10, rectangle=stubs
    )
    names = sorted(k.kernel_source.rsplit("/", 1)[-1] for k in kernels)
    assert names == sorted(
        [
            "reader_rtf_main.cpp",
            "writer_rtf.cpp",
            "compute_rtf_main.cpp",
            "reader_rtf_sort.cpp",
            "compute_rtf_sort.cpp",
            "writer_rtf_index.cpp",
        ]
    )
    defines = {k.kernel_source.rsplit("/", 1)[-1]: dict(k.defines) for k in kernels}
    assert defines["compute_rtf_sort.cpp"].keys() >= {"RTF_DIRECT", "RTF_INLINE", "RTF_IDXF"}
    assert defines["reader_rtf_sort.cpp"].get("RTF_SPLIT") == "1" and defines["writer_rtf.cpp"].get("RTF_IDXF") == "1"
    assert defines["reader_rtf_main.cpp"].keys() >= {"RTF_ROW0", "RTF_NOTOKEN", "RTF_IDXF"}


def test_fast_off_builds_the_sort_core_sources(monkeypatch, stubs):
    monkeypatch.setenv(sc.FAST_ENV, "0")  # the fast tail is the default; "0" builds the sort-core sources
    kernels, _cbs, _sem = sc.two_core_parts(
        _Logits(), _Logits(), _Logits(), _Logits(), rows=1, top_k=10, rectangle=stubs
    )
    assert sorted(k.kernel_source.rsplit("/", 1)[-1] for k in kernels) == sorted(
        [
            "reader_rt_main.cpp",
            "writer_router_tail.cpp",
            "compute_rt_main.cpp",
            "reader_rt_sort.cpp",
            "compute_rt_sort.cpp",
        ]
    )
    assert all(not k.defines or all(not n.startswith("RTF_") for n, _v in k.defines) for k in kernels)


def test_fast_compute_main_is_the_sort_core_one_plus_switches():
    fast, base = K("compute_main"), BASE("compute_main")
    # the same code with the two switches' #if lines removed
    stripped = re.sub(r"#if !RTF_(NOSTART|NOTOKEN)\n(.*?)#endif\n", r"\2", fast, flags=re.S)
    stripped = re.sub(r"#ifndef RTF_\w+\n#define RTF_\w+ 0[^\n]*\n#endif\n", "", stripped)
    assert _code(base.split("#include", 1)[1]) == _code(stripped.split("#include", 1)[1])


def test_fast_sort_keeps_the_network_and_the_shift():
    fast = K("compute_sort")
    assert (
        "frt_sort64(pass_mask)" in fast
        and "topk_local_sort_lanes<false>(0, 0 /* largest */, 5 /* end_phase */, pass_mask)" in fast
    )
    assert "SORT_SHIFT_BITS = 0x43000000" in fast and "VectorMode::C" in fast  # faces 0 and 2 hold token 0
    assert "(w == 0) ? 0 : 1" in fast and "frt_load_tile(cb_src, cb_index, w" in fast


def test_fp32_index_decode_matches_python():
    import struct

    for e in range(512):
        bits = struct.unpack("<I", struct.pack("<f", float(e)))[0]
        assert (0 if bits == 0 else (((bits & 0x7FFFFF) | 0x800000) >> (150 - (bits >> 23)))) == e
    for name in ("reader_main", "writer"):
        assert "(((bits & 0x7FFFFFu) | 0x800000u) >> (150u - (bits >> 23)))" in K(name)
