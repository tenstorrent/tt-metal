# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The barrier-free row gather (``ttnn/fused/row_gather2``) without a device: its admission, the argument layout between the
builder and the kernel, the parity protocol pins and the QWEN38_ROW_GATHER switch of the registry entry."""

from __future__ import annotations

import inspect
import re
from types import SimpleNamespace

import pytest

import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn import fused
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import program as fp
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import row_gather as rg
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import row_gather2 as rg2

KERNEL = (fp.REPO_ROOT / rg2.KERNEL).read_text()
MODULE = inspect.getsource(rg2)


def _hidden(rows=1, width=320, dtype=ttnn.bfloat16, ring=8, layout=ttnn.TILE_LAYOUT):
    mesh = SimpleNamespace(shape=(1, ring))
    return SimpleNamespace(
        shape=(1, 1, rows, width), padded_shape=(1, 1, 32, width), dtype=dtype, layout=layout, device=lambda: mesh
    )


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setenv("QWEN38_ROW_GATHER", "1")
    monkeypatch.setattr(rg2.gr_fold, "line", lambda mesh: object())
    monkeypatch.setattr(rg2, "TP_SIZE", 8)
    monkeypatch.setattr(rg2.ttnn, "get_tt_fabric_max_payload_size_bytes", lambda: 4352, raising=False)


def test_switch_modes(monkeypatch):
    monkeypatch.setenv("QWEN38_ROW_GATHER", "0")
    assert not rg.enabled_by_env() and not rg.admits(_hidden())
    assert rg.enabled_by_env({})  # on by default (x38 A/B)
    for value, enabled in (("1", True), ("v1", True), ("0", False), ("", False)):
        assert rg.enabled_by_env({"QWEN38_ROW_GATHER": value}) is enabled
    assert "row_gather" in fused.enabled_names({"QWEN38_ROW_GATHER": "1"}) or fused.registry.TP_SIZE == 4
    assert "row_gather" in fused.enabled_names({}) or fused.registry.TP_SIZE == 4
    assert "row_gather" not in fused.enabled_names({"QWEN38_ROW_GATHER": "0"})
    assert fused.kernel("row_gather").tolerance == fused.BITWISE and not fused.kernel("row_gather").default_on


def test_admission(on, monkeypatch):
    assert rg2.admits(_hidden()) and rg2.admits(_hidden(rows=2))
    monkeypatch.setenv(rg2.ROWS_ENV, "2")  # rows 3..8 are the rows program's: off with the knob at 2
    assert not rg2.admits(_hidden(rows=3))
    assert not rg2.admits(_hidden(dtype=ttnn.float32)) and not rg2.admits(_hidden(layout=ttnn.ROW_MAJOR_LAYOUT))
    assert not rg2.admits(_hidden(ring=4)) and not rg2.admits(_hidden(), cluster_axis=0)
    assert not rg2.admits(_hidden(width=320 * 20))  # 2 x 64 B per tile of 100 tiles over the packet payload


def test_kernel_compile_args_match_the_builder():
    ids = sorted({int(i) for i in re.findall(r"get_compile_time_arg_val\((\d+)\)", KERNEL)})
    base = int(re.search(r"ACCESSOR_BASE = (\d+);", KERNEL).group(1))
    extra = re.search(r"extra_args=\(([^)]*)\)", inspect.getsource(rg2.row_gather)).group(1).split(",")
    assert base == 11 + len([e for e in extra if e.strip()]) == max(ids) + 1


def test_parity_protocol_pins():
    # one counter word read before anything else, stepped after the other RISC's read (its done.set), data semaphore reset
    # by the owner, one packet per direction, no barrier semaphore
    assert KERNEL.index("const uint32_t parity = *counter & 1;") < KERNEL.index(
        "noc_semaphore_wait_min(data, RING - 1);"
    )
    assert (
        KERNEL.index("done.wait(1);")
        < KERNEL.index("noc_semaphore_set(data, 0);")
        < KERNEL.index("*counter = *counter + 1;")
    )
    assert KERNEL.count("fabric_multicast_noc_fused_unicast_with_atomic_inc(") == 1 and "arrive(" not in KERNEL
    assert "parity ? DATA1_ADDR : data0_addr" in KERNEL
    # both RISCs share the sources; BRISC owns the own chunks
    assert "own || (k++ & 1)" in KERNEL and "!own && !(k++ & 1)" in KERNEL


def test_zones():
    names = re.findall(r'FUSED_ZONE\("([^"]+)"\)', KERNEL)
    assert len(names) == len(set(names)) == 4 and all(n.startswith("fz_rg2_") for n in names)


def test_rows_admission_and_knob(on, monkeypatch):
    monkeypatch.delenv(rg2.ROWS_ENV, raising=False)
    assert rg2.rows_max() == 8
    for rows in (3, 5, 8):
        assert rg2.admits(_hidden(rows=rows)) and rg2.admits_rows(_hidden(rows=rows))
        assert rg2.admits(_hidden(rows=32), live_rows=rows)  # the verify form: live rows on the whole tile
    assert not rg2.admits(_hidden(rows=9)) and not rg2.admits(_hidden(rows=32), live_rows=9)
    assert not rg2.admits(_hidden(rows=32)) and not rg2.admits(_hidden(rows=5), live_rows=6)
    assert not rg2.admits(_hidden(rows=32), live_rows=2) and not rg2.admits(
        _hidden(rows=32), live_rows=1
    )  # rows 1, 2 unchanged
    assert not rg2.admits(_hidden(rows=5, dtype=ttnn.float32)) and not rg2.admits(_hidden(rows=5, ring=4))
    assert not rg2.admits(_hidden(rows=5), cluster_axis=0)
    for value, expected in (("2", 2), ("5", 5), ("99", 8), ("0", 2), ("x", 2)):
        monkeypatch.setenv(rg2.ROWS_ENV, value)
        assert rg2.rows_max() == expected
    monkeypatch.setenv(rg2.ROWS_ENV, "5")
    assert rg2.admits(_hidden(rows=5)) and not rg2.admits(_hidden(rows=6))
    monkeypatch.setenv(rg2.ROWS_ENV, "2")
    assert not any(rg2.admits(_hidden(rows=r)) for r in range(3, 9))
    assert rg2.admits(_hidden()) and rg2.admits(_hidden(rows=2))  # the 1- and 2-row program is untouched by the knob
    monkeypatch.setenv("QWEN38_ROW_GATHER", "0")
    assert not rg2.admits(_hidden(rows=5))


def test_rows_geometry_fits_one_packet_and_never_grows_the_persistent_buffer(on):
    for rows, pitch in ((3, 128), (4, 128), (5, 192), (6, 192), (7, 256), (8, 256)):
        links, got_pitch, slot = rg2._geometry(_hidden(rows=rows), rows)
        assert (links, got_pitch, slot) == (2, pitch, 5 * 2 * pitch)
        assert slot <= 4352
    # the rows program lands in its own CB: the persistent landing buffer of the 1- and 2-row program keeps its size
    assert "land =" in inspect.getsource(rg2.prepare) and "2 * TP_SIZE * slot" in inspect.getsource(rg2.prepare)
    assert "prepare(" not in inspect.getsource(rg2.row_gather_rows)


ROWS_KERNEL = (fp.REPO_ROOT / rg2.KERNEL_ROWS).read_text()


def test_rows_kernel_compile_args_match_the_builder():
    ids = sorted({int(i) for i in re.findall(r"get_compile_time_arg_val\((\d+)\)", ROWS_KERNEL)})
    base = int(re.search(r"ACCESSOR_BASE = (\d+);", ROWS_KERNEL).group(1))
    extra = re.search(r"extra_args=\(([^)]*)\)", inspect.getsource(rg2.row_gather_rows)).group(1).split(",")
    assert base == 11 + len([e for e in extra if e.strip()]) == max(ids) + 1
    assert (
        "get_compile_time_arg_val(11)" in ROWS_KERNEL
        and "constexpr uint32_t ROWS = get_compile_time_arg_val(11)" in ROWS_KERNEL
    )


def test_rows_kernel_protocol_pins():
    # the barrier (v1's FACT 1/2): arrive, then BRISC's wait + reset before the release, the single fused send after it
    assert ROWS_KERNEL.count("fabric_multicast_noc_fused_unicast_with_atomic_inc(") == 1
    assert ROWS_KERNEL.count("line.arrive(barrier_addr);") == 1
    assert (
        ROWS_KERNEL.index("line.arrive(barrier_addr);")
        < ROWS_KERNEL.index("noc_semaphore_wait_min(barrier, RING - 1);")
        < ROWS_KERNEL.index("noc_semaphore_set(barrier, 0);")
        < ROWS_KERNEL.index("go.set(1);")
        < ROWS_KERNEL.index("fabric_multicast_noc_fused_unicast_with_atomic_inc(")
    )
    assert ROWS_KERNEL.index("go.wait(1);") < ROWS_KERNEL.index("fabric_multicast_noc_fused_unicast_with_atomic_inc(")
    # the dead rows are zeroed while the barrier is in flight (after the arrive, before the wait); data reset after `done`
    assert (
        ROWS_KERNEL.index("line.arrive(barrier_addr);")
        < ROWS_KERNEL.index("fz_rg2_r8_zero")
        < ROWS_KERNEL.index("noc_semaphore_wait_min(barrier, RING - 1);")
    )
    assert ROWS_KERNEL.index("done.wait(1);") < ROWS_KERNEL.index("noc_semaphore_set(data, 0);")
    assert (
        "noc_semaphore_wait_min(data, RING - 1);" in ROWS_KERNEL
        and "LAND_ADDR" not in ROWS_KERNEL
        and "call_counter" not in ROWS_KERNEL
    )
    # both RISCs share the sources; BRISC owns the own chunks; zero and data writes of a page are the same RISC's
    assert "own || (k++ & 1)" in ROWS_KERNEL and "!own && !(k++ & 1)" in ROWS_KERNEL
    assert "ZERO_A_LEN = FACE_BYTES - CHUNK_LEN" in ROWS_KERNEL and "ZERO_B_OFF = FACE_BYTES + CHUNK_LEN" in ROWS_KERNEL
    names = re.findall(r'FUSED_ZONE\("([^"]+)"\)', ROWS_KERNEL)
    assert len(names) == len(set(names)) == 5 and all(n.startswith("fz_rg2_r8_") for n in names)


def test_one_and_two_row_program_is_untouched():
    # the original kernel does not know the rows mode and the 1- / 2-row call keeps its extra args
    assert "arrive(" not in KERNEL and "go.set" not in KERNEL and "barrier, RING" not in KERNEL
    assert 'extra_args=(rows, state["land_addr"], state["data1"], tile_bytes)' in inspect.getsource(rg2.row_gather)
    assert "ROWS_MAX = 2" in MODULE and "2 * TP_SIZE * slot" in MODULE


def test_call_sites_pass_the_live_rows():
    gdn_src = (fp.REPO_ROOT / "models/demos/blackhole/qwen38_flash_next/ttnn/gdn.py").read_text()
    qsa_src = (fp.REPO_ROOT / "models/demos/blackhole/qwen38_flash_next/ttnn/qsa.py").read_text()
    assert (
        'fused.resolve_admitted("row_gather")(\n                padded, cluster_axis=TP_AXIS, memory_config=gather_memory, live_rows=rows'
        in gdn_src
    )
    assert "hidden_rows, cluster_axis=TP_AXIS, memory_config=gather_memory, live_rows=live_rows" in qsa_src
    assert "self._all_gather_hidden_rows(hidden_rows, constants, live_rows=live_rows)" in qsa_src
    assert "2 < live_rows < CHUNK_ROWS" in qsa_src and "2 < rows < CHUNK_SIZE" in gdn_src
    for module in (rg, rg2):  # every entry point takes live_rows
        for fn in ("admits", "composed", "row_gather", "row_gather_or_all_gather"):
            assert "live_rows" in inspect.signature(getattr(module, fn)).parameters, (module.__name__, fn)
