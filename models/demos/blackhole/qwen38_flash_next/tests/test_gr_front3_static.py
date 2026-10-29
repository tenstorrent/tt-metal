# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""No-device contract of the GR front v3 (QWEN38_GR_FRONT_V3=1): sources, the selection hook, the token protocol's ordering rules, the
slot size that carries the token, the reader's reorder."""

from pathlib import Path

from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gr_recip_last as grl
from models.demos.blackhole.qwen38_flash_next.ttnn.fused.gr_recip_last import front3

KERNELS = Path(grl.__file__).parent / "kernels"


def _text(name):
    return (KERNELS / name).read_text()


def test_sources_and_hook():
    for name in ("transport4.cpp", "front_reader3.cpp", "mcast_writer5.cpp", "raw_writer3.cpp", "mcast_reader4.cpp"):
        assert (KERNELS / name).is_file()
    init = Path(grl.__file__).read_text()
    assert init.count("QWEN38_GR_FRONT_V3") == 1 and 'os.environ.get("QWEN38_GR_FRONT_V3") == "1"' in init
    assert front3.ENV == "QWEN38_GR_FRONT_V3" and not front3._STATE3 is None
    src = Path(front3.__file__).read_text()
    assert "+ 16  # + the call token" in src and "my_tiles_b * pack_tile + 16 >" in src


def test_token_protocol_text():
    t = _text("transport4.cpp")
    p = t[t.index("FORCE_INLINE void partials_phase") : t.index("void kernel_main()")]
    assert "const uint32_t token = count + 1" in p
    assert p.index("= token;") < p.index("fabric_multicast_noc_fused_unicast_with_atomic_inc") < p.index("line.close()")
    assert "TOKEN_OFF + 16," in p  # the payload carries the token as its last 16 B
    assert p.index("if (*flag != token)") < p.index("noc_async_write(") < p.index("noc_async_write_barrier()")
    assert "invalidate_l1_cache()" in p
    # the data semaphore is still counted and reset, and the counter stepped only after NCRISC finished
    assert (
        p.index("noc_semaphore_wait_min(data, RING - 1)")
        < p.index("noc_semaphore_set(data, 0)")
        < p.index("done.wait(1)")
    )
    assert p.index("done.wait(1)") < p.index("*counter = *counter + 1")
    assert "FIRST_TILE = 0" in p and "FIRST_TILE = 1" in p  # BRISC even tiles, NCRISC odd tiles
    assert "unpack.wait" not in p and "unpack.set" not in p


def test_reader_reorder():
    r = _text("front_reader3.cpp")
    k = r[r.index("void kernel_main()") :]
    assert (
        k.index("issue_stream(args0")
        < k.index("make_const<")
        < k.index("noc_async_read_barrier()")
        < k.index("gate(2)")
    )
