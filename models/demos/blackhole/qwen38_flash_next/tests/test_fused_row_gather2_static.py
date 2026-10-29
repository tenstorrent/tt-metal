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


def test_admission(on):
    assert rg2.admits(_hidden()) and rg2.admits(_hidden(rows=2))
    assert not rg2.admits(_hidden(rows=3))
    assert not rg2.admits(_hidden(dtype=ttnn.float32)) and not rg2.admits(_hidden(layout=ttnn.ROW_MAJOR_LAYOUT))
    assert not rg2.admits(_hidden(ring=4)) and not rg2.admits(_hidden(), cluster_axis=0)
    assert not rg2.admits(_hidden(width=320 * 20))  # 2 x 64 B per tile of 100 tiles over the packet payload


def test_kernel_compile_args_match_the_builder():
    ids = sorted({int(i) for i in re.findall(r"get_compile_time_arg_val\((\d+)\)", KERNEL)})
    base = int(re.search(r"ACCESSOR_BASE = (\d+);", KERNEL).group(1))
    extra = re.search(r"extra_args=\(([^)]*)\)", MODULE).group(1).split(",")
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
