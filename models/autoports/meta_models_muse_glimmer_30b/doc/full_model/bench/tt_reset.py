# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reset the Tenstorrent devices through the kernel driver's own ioctl.

``$tt-device-usage`` prescribes ``tt-smi -r`` for recovery, and ``tt-smi`` is not
installed in this environment (no network to install it either).  This does the
same thing the tool does: the ``TENSTORRENT_IOCTL_RESET_DEVICE`` ioctl that
``/usr/src/tenstorrent-2.10.0/ioctl.h`` defines, with the Blackhole
``ASIC_RESET`` then ``POST_RESET`` flag pair.

Every device must be free first -- kill anything holding ``/dev/tenstorrent/*``
before running this, or the driver refuses.

Usage::

    python doc/full_model/bench/tt_reset.py            # reset every device
    python doc/full_model/bench/tt_reset.py --devices 0,1
"""

from __future__ import annotations

import argparse
import fcntl
import glob
import os
import struct
import sys
import time

TENSTORRENT_IOCTL_MAGIC = 0xFA
# _IO(TENSTORRENT_IOCTL_MAGIC, 6)
TENSTORRENT_IOCTL_RESET_DEVICE = (TENSTORRENT_IOCTL_MAGIC << 8) | 6

RESET_DEVICE_ASIC_RESET = 4
RESET_DEVICE_POST_RESET = 6
# struct tenstorrent_reset_device { in{u32 output_size_bytes; u32 flags;}
#                                  out{u32 output_size_bytes; u32 result;} }
_STRUCT = "=IIII"
_OUT_SIZE = 8


def reset_one(path: str, flags: int) -> int:
    payload = struct.pack(_STRUCT, _OUT_SIZE, flags, 0, 0)
    fd = os.open(path, os.O_RDWR | os.O_CLOEXEC)
    try:
        result = fcntl.ioctl(fd, TENSTORRENT_IOCTL_RESET_DEVICE, payload)
    finally:
        os.close(fd)
    _, _, _, code = struct.unpack(_STRUCT, result)
    return int(code)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", default="", help="comma-separated device indices; default all")
    parser.add_argument("--settle-s", type=float, default=3.0)
    args = parser.parse_args()

    if args.devices:
        paths = [f"/dev/tenstorrent/{index.strip()}" for index in args.devices.split(",") if index.strip()]
    else:
        paths = sorted(glob.glob("/dev/tenstorrent/[0-9]*"))
    if not paths:
        print("no /dev/tenstorrent devices found", file=sys.stderr)
        return 3

    failures = 0
    for flags, label in ((RESET_DEVICE_ASIC_RESET, "asic"), (RESET_DEVICE_POST_RESET, "post")):
        for path in paths:
            try:
                code = reset_one(path, flags)
                print(f"RESET {label} {path} result={code}", flush=True)
                if code != 0:
                    failures += 1
            except OSError as exc:
                print(f"RESET {label} {path} FAILED {exc}", flush=True)
                failures += 1
        time.sleep(args.settle_s)
    print("RESET_DONE failures=" + str(failures), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
