# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The GR gate + gather program (``ttnn/fused/row_gather3``) without a device: switch, hand-off tables, kernel argument layout
against the builder, the gate kernels it reuses unchanged, the hooks in gate3 / row_gather2 / gr.py / layer.py."""

from __future__ import annotations

import inspect
import re

from models.demos.blackhole.qwen38_flash_next.ttnn.fused import program as fp
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import row_gather2 as rg2
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import row_gather3 as rg3
from models.demos.blackhole.qwen38_flash_next.ttnn.fused.gr_read import gate3

MODULE = inspect.getsource(rg3)
TRANSPORT = (fp.REPO_ROOT / rg3.TRANSPORT_KERNEL).read_text()
WRITER = (fp.REPO_ROOT / rg3.WRITER_FWD).read_text()


def test_switch_and_handoff_tables():
    assert not rg3.enabled_by_env({}) and rg3.enabled_by_env({rg3.ENV: "1"}) and not rg3.enabled_by_env({rg3.ENV: "0"})
    block, cfg = object(), object()
    rg3.GATHERED[id(block)] = (block, "full", cfg)
    assert rg3.take_gathered(block, object()) is None  # another memory config: ordinary gather
    rg3.GATHERED[id(block)] = (block, "full", cfg)
    assert rg3.take_gathered(block, cfg) == "full" and rg3.take_gathered(block, cfg) is None
    rg3.CONTEXT["memory_config"] = cfg
    assert rg3.take_context() is cfg and rg3.take_context() is None


def test_transport_compile_args_match_the_builder():
    ids = sorted({int(i) for i in re.findall(r"get_compile_time_arg_val\((\d+)\)", TRANSPORT)})
    base = int(re.search(r"ACCESSOR_BASE = (\d+);", TRANSPORT).group(1))
    extra = MODULE.split("extra_args=(", 1)[1].split("\n    )", 1)[0]
    scalars = [e for e in re.sub(r"\s+", " ", extra).split(",") if e.strip()]
    assert base == 11 + len(scalars) == max(ids) + 1
    assert rg3.SEM_ARRIVE not in (rg3.SEM_GATE, rg3.SEM_GO, rg3.SEM_SCRATCH, rg3.SEM_DONE)


def test_gate_kernels_are_gate3s_and_the_writer_keeps_the_stock_write():
    assert (
        "g3.LOWRANK3" in MODULE
        and "g3.GATE3" in MODULE
        and "g3._mcast_writer" in MODULE
        and "g3._mcast_reader" in MODULE
    )
    stock = (fp.REPO_ROOT / fp.KERNEL_ROOT / "gr_read/kernels/writer.cpp").read_text()
    assert "noc.async_write(\n                dfb, accessor, tile_bytes" in stock
    assert (
        "noc.async_write(dfb, accessor, tile_bytes, {.offset_bytes = 0}, {.page_id = first});" in WRITER
    )  # the DRAM block write stays
    assert WRITER.index("noc.async_write_barrier();") < WRITER.index(
        "arrive.up(noc, tx, ty, 1);"
    )  # chunks landed, then the step
    # parity protocol as row_gather2
    assert TRANSPORT.index("arrive.wait_min(TILES);") < TRANSPORT.index(
        "fabric_multicast_noc_fused_unicast_with_atomic_inc("
    )
    assert (
        TRANSPORT.index("done.wait(1);") < TRANSPORT.index("*counter = *counter + 1;") and "arrive.set(0);" in TRANSPORT
    )


def test_hooks_are_in_place():
    assert "row_gather3.take_context()" in inspect.getsource(gate3.low_rank_gate3)
    assert "row_gather3.take_gathered(hidden, memory_config)" in inspect.getsource(rg2.row_gather)
    import models.demos.blackhole.qwen38_flash_next.ttnn.gr as gr
    import models.demos.blackhole.qwen38_flash_next.ttnn.layer as layer

    assert "set_context(self.gather_memory_config)" in inspect.getsource(gr)
    assert "gather_memory_config" in inspect.getsource(layer)
    names = re.findall(r'FUSED_ZONE\("([^"]+)"\)', TRANSPORT + WRITER)
    assert len(names) == len(set(names)) == 5 and all(n.startswith("fz_rg3_") for n in names)
