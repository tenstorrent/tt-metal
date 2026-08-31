# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Does the DST tile budget follow a 32-bit Dest?

A 32-bit Dest register holds half as many tiles -- 4 rather than 8 -- and `kMaxDstTiles`
is what the model sizes every DST allocation against. It also picks between two code
paths: `Strategy<FPUFusion>::run` takes a single acquire when the output block fits and
`run_banded()` when it does not, and the two emit DIFFERENT buffer protocol.

So the boundary is the test. rt=2 x ct=4 is exactly 8 output tiles: it fits in a 16-bit
Dest and does not fit in a 32-bit one, which is the only shape where a budget that has
not followed the mode gives a different answer from one that has.

L1 accumulate is the row that carries the information. math.hpp records why: a too-large
subblock still round-trips in Dst mode, because math and pack share the same wrong
mapping and it cancels, while L1 mode exposes it -- the packer's read-modify-write
depends on the absolute DST->L1 mapping, and only the upper half is right.

Sabotage that proves it: pin `kMaxDstTiles` back to a literal 8 and the two dst_32bit
rows fail while the 16-bit rows keep passing.

    export TT_METAL_HOME=$PWD
    source python_env/bin/activate
    python test_unified_dst32.py
"""

import sys

import ttnn
from loguru import logger

from test_unified_matmul import pcc, run

# 8 output tiles: the fit/no-fit boundary between a 16-bit and a 32-bit Dest.
RT, CT, KT = 2, 4, 2


def main():
    device = ttnn.open_device(device_id=0)
    failed = []
    try:
        for mode in ("dst", "l1"):
            for dst_32bit in (False, True):
                got, want = run(device, RT, CT, KT, mode=mode, dst_32bit=dst_32bit)
                measured = pcc(got, want)
                ok = measured >= 0.99
                logger.info(
                    f"  {mode:3s} accumulate, {RT}x{CT}=8 tiles, "
                    f"32-bit Dest={str(dst_32bit):5s}: PCC = {measured:.6f}  {'ok' if ok else 'FAIL'}"
                )
                if not ok:
                    failed.append(f"{mode}/dst32={dst_32bit}")
    finally:
        ttnn.close_device(device)

    if failed:
        logger.error(f"FAIL: {failed}")
        return 1
    logger.info("all ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
