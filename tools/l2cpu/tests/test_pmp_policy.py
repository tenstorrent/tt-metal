# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
# SPDX-License-Identifier: Apache-2.0
"""Host-side PMP policy (host/l2cpu/pmp.py): entry count, table layout, first-match decisions, refusals. No device."""
import os
import struct
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "host"))

from l2cpu import layout as A  # noqa: E402
from l2cpu import pmp as P  # noqa: E402

REGION = 0x4000_305A_0000  # measured ttnn region base (tile 0)


def test_fits_the_x280_and_table_layout():
    for size in (A.L2CPU_REGION_MIN_SIZE, 64 << 20):
        pol = P.build(REGION, size)
        assert len(pol.entries) == P.HW_ENTRIES == 8
        assert all(e.cfg & P.L for e in pol.entries)
        assert pol.entries[-1].addr == P.HW_ADDR_MASK and pol.entries[-1].cfg & 7 == 0
        t = pol.table()
        assert len(t) == A.L2CPU_RES_PMP_CFG + A.L2CPU_PMP_MAX - A.L2CPU_RES_PMP
        magic, n = struct.unpack_from("<II", t, 0)
        assert (magic, n) == (A.L2CPU_PMP_MAGIC, 8)
        addr0 = struct.unpack_from("<Q", t, A.L2CPU_RES_PMP_ADDR - A.L2CPU_RES_PMP)[0]
        assert addr0 == pol.entries[0].addr
        assert A.L2CPU_RES_PMP + len(t) <= A.L2CPU_RES_PROTECTED


@pytest.mark.parametrize(
    "pa,n,perm,ok",
    [
        (REGION, 8, P.R | P.W | P.X, True),
        (REGION + (64 << 20) - 8, 8, P.W, True),
        (REGION + (64 << 20) - 4, 8, P.R, False),  # straddles the region end
        (REGION + (64 << 20), 4, P.R, False),
        (REGION + A.L2CPU_OFF_RESIDENT + 0x10, 4, P.W, False),  # resident code
        (REGION + A.L2CPU_OFF_RESIDENT + 0x10, 4, P.X, True),
        (REGION + A.L2CPU_OFF_RESIDENT + A.L2CPU_RES_REC, 8, P.W, True),  # records stay writable
        (REGION - P.UNCACHED_DELTA + 0x200_0000, 64, P.R, True),
        (REGION - P.UNCACHED_DELTA + 0x200_0000, 8, P.W, False),
        (0x0201_0200, 4, P.W, True),  # L3 FLUSH64
        (0x0C00_2000, 4, P.W, True),  # PLIC enable
        (0x2000_0010, 4, P.W, True),  # TLB window config
        (A.L2CPU_RNMI_TRIGGER, 4, P.R, True),
        (A.L2CPU_RNMI_HANDLER, 4, P.W, False),
        (A.L2CPU_BOOT_RECORD_PA + A.L2CPU_BOOT_PMP, 4, P.W, False),
        (0x2006_0000, 4, P.R, True),  # MSI FIFO pop
        (0x2008_0000, 4, P.R, False),  # DMA controller
        (P.TLB2M_UNCACHED + 31 * P.TLB2M_SIZE, 8, P.W, True),
        (P.TLB2M_UNCACHED + 32 * P.TLB2M_SIZE, 8, P.R, False),
        (P.TLB2M_CACHED, 8, P.R, False),
    ],
)
def test_decisions(pa, n, perm, ok):
    assert P.build(REGION, 64 << 20).allows(pa, n, perm) is ok


def refused(*args, **kw):
    try:
        P.build(*args, **kw)
    except P.PmpError:
        return True
    return False


def test_refusals():
    assert refused(REGION, 64 << 20, windows=((0, 32, False), (64, 32, False)))  # 9 entries
    assert refused(REGION, 64 << 20, windows=((0, 24, False),))  # not a power-of-two block
    assert refused(0x4000_0000_0000 + 0x0FF0_0000, 64 << 20)  # uncached block would leave the local DRAM
    assert refused(REGION + 0x100, 64 << 20)  # not granule aligned
    assert len(P.build(0x4000_0000_0000 + 0x0FF0_0000, 64 << 20, uncached=False).entries) == 8
