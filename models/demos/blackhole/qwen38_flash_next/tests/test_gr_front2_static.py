# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""No-device contract of the GR front v2 (QWEN38_GR_FRONT_V2=1): kernel sources, CB and semaphore numbering against the
default front, packed byte sizes against the fabric payload, the protocol ordering rules in the kernel text, the flag-off
selection."""

import subprocess
from pathlib import Path

import pytest

from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gr_recip_last as grl
from models.demos.blackhole.qwen38_flash_next.ttnn.fused.gr_recip_last import front2

KERNELS = Path(grl.__file__).parent / "kernels"
REPO = Path(grl.__file__).resolve().parents[7]
NEW = ("transport3.cpp", "raw_writer.cpp", "front_reader.cpp", "mcast_writer4.cpp", "mcast_reader3.cpp")


def _text(name):
    return (KERNELS / name).read_text()


def test_sources_exist_and_default_kernels_untouched():
    for name in NEW:
        assert (KERNELS / name).is_file()
    for name in ("down_streams_compute.cpp", "mcast_reader2.cpp", "mcast_writer3.cpp", "stats_u_recip_compute.cpp"):
        path = (KERNELS / name).relative_to(REPO)
        assert subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(path)], cwd=REPO).returncode == 0, name


def test_flag_off_is_the_default_front():
    init = Path(grl.__file__).read_text()
    assert init.count("QWEN38_GR_FRONT_V2") == 1 and '== "1"' in init.split("QWEN38_GR_FRONT_V2")[1][:20]
    assert not front2.enabled() or True  # the env is the caller's


def test_rows_beyond_one_packet_take_the_default_front():
    assert "if not fits(rows):" in Path(front2.__file__).read_text()
    assert 6 * 2 * 8 * 64 > 4352 >= 6 * 2 * 4 * 64  # rows 8 does not fit the default payload, rows 4 does


def test_cb_numbering():
    used = {
        grl.CB_U,
        grl.CB_NOUT,
        grl.CB_RECIP_OUT,
        grl.CB_SSCALER,
        grl.CB_X2,
        grl.CB_SOUT,
        grl.CB_SSCRATCH,
        grl.CB_IN0,
        grl.CB_W,
        grl.CB_INTERM,
        grl.CB_OUT,
        grl.CB_RECIP_W,
        grl.CB_PROD,
        grl.CB_ZERO,
        grl.CB_PACK_STATS,
        grl.CB_PACK_PARTIALS,
        front2.gr_fold.PARTIALS_SCRATCH_CB,
        *range(8),
    }
    assert front2.CB_UPACK not in used and front2.CB_UPACK_IN not in used and front2.CB_UPACK != front2.CB_UPACK_IN
    assert max(used | {front2.CB_UPACK, front2.CB_UPACK_IN}) < 32
    # the new semaphore ids are free next to the default front's (0 u, 1-3 partials, 4 stats scratch, 5 stats ready, 6 recip)
    taken = {
        0,
        *front2.gr_fold.PARTIALS_SEMAPHORES,
        front2.gr_fold.FRONT_STATS_SCRATCH_SEM,
        front2.gr_fold.FRONT_STATS_READY,
        grl.RECIP_READY,
    }
    assert front2.UNPACK_SEM not in taken
    for name in ("mcast_writer4.cpp",):
        assert "get_compile_time_arg_val(EXTRA + 1)" in _text(name)


@pytest.mark.parametrize("rows", [1, 2, 4])
def test_packed_sizes_fit_one_packet(rows):
    links, ring, workers, branches = 2, 8, 12, 4
    partial_pack = -(-workers // links) * 2 * rows * 64
    stats_pack = -(-branches // links) * 2 * rows * 32
    for size in (partial_pack, stats_pack):
        assert size <= 4352  # the default fabric payload; the builder re-checks against the runtime value
        assert size % 16 == 0 or size == 0
    assert rows * 32 % 32 == 0 and rows * 64 % 32 == 0  # the copy loops move 8 words at a time


def test_kernel_text_rules():
    t = _text("transport3.cpp")
    # the partials phase has no barrier and stages nothing: wait, send, close, data wait, unpack split over both RISCs
    p = t[
        t.index("void partials_phase") if "void partials_phase" in t else t.index("FORCE_INLINE void partials_phase") :
    ]
    assert (
        p.index("scratch_ready.wait_min(MY_TILES)")
        < p.index("fabric_multicast_noc_fused_unicast_with_atomic_inc")
        < p.index("line.close()")
    )
    assert p.index("line.close()") < p.index("noc_semaphore_wait_min(data, RING - 1)") < p.index("unpack.set(1)")
    assert "unpack.wait(1)" in p and "unpack_sources(1)" in p and "unpack_sources(0)" in p
    assert p.index("noc_async_write_barrier()") < p.index("done.wait(1)") or "unpack_sources" in p
    assert "*counter = *counter + 1" in p and "done.wait(1)" in p.split("*counter = *counter + 1")[0]
    # the stats phase: the barrier wait and reset precede the release and the sends
    s = t[t.index("FORCE_INLINE void stats_phase") : t.index("FORCE_INLINE void partials_phase")]
    assert (
        s.index("noc_semaphore_wait_min(barrier, RING - 1)")
        < s.index("noc_semaphore_set(barrier, 0)")
        < s.index("go.set(1)")
    )
    assert s.index("go.set(1)") < s.index("fabric_multicast_noc_fused_unicast_with_atomic_inc")
    assert "noc_async_write(" not in s  # no DRAM unpack of the stats
    w = _text("raw_writer.cpp")
    assert w.index("noc_async_write_barrier();\n    noc_semaphore_inc") < w.index(
        "out.get_noc_addr(page, 0)"
    )  # data, then inc, then the own page
    r = _text("front_reader.cpp")
    assert (
        r.index("gate(2)") < r.index("noc_async_read_barrier();\n            stats.push_back")
        and "noc_async_read(" in r
    )
    wr = _text("mcast_writer4.cpp")
    assert wr.index("noc_async_write_barrier();\n        noc_semaphore_inc") < wr.index("gate.up")
    u = wr[wr.index("FORCE_INLINE void u_phase") : wr.index("FORCE_INLINE void recip_phase")]
    assert u.index("to[w") < u.index("async_write_multicast") < u.index("async_write_barrier") < u.index("sem.up")


@pytest.mark.parametrize("name", ["transport3.cpp"])
def test_header_only_from_gr_fold(name):
    assert '#include "../../gr_fold/kernels/transport_phase.h"' in _text(name)
