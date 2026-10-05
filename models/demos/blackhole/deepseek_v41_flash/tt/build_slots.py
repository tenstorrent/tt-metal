# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Cluster-wide cap on concurrent FULL model builds.

A 40-layer build reads ~640 GB of bfp8 expert weights over the shared NFS (~100 s per layer per host when only a few hosts build); a dozen hosts building at once
saturate the NFS server (builds of 65+ minutes, 5 MB/s per host seen). ``build_slot()`` holds one of N slot files (flock, which Linux/NFSv4 turns into byte-range
locks, so it excludes across hosts) while the weights are loaded, and releases it when the build is done. Short builds (few layers) do not need a slot.

Env: ``DSV41_BUILD_SLOTS`` (default 5, 0 disables), ``DSV41_BUILD_SLOT_DIR`` (default /mnt/tt-data/ssinghal/buildslots),
``DSV41_BUILD_SLOT_MIN_LAYERS`` (builds with fewer layers skip the cap, default 9).
"""

import contextlib
import fcntl
import os
import time


@contextlib.contextmanager
def build_slot(n_layers, log=print):
    slots = int(os.environ.get("DSV41_BUILD_SLOTS", "5"))
    if slots <= 0 or n_layers < int(os.environ.get("DSV41_BUILD_SLOT_MIN_LAYERS", "9")):
        yield None
        return
    d = os.environ.get("DSV41_BUILD_SLOT_DIR", "/mnt/tt-data/ssinghal/buildslots")
    os.makedirs(d, exist_ok=True)
    fds = [os.open(os.path.join(d, f"slot{i}.lock"), os.O_RDWR | os.O_CREAT, 0o666) for i in range(slots)]
    held, t0, last = None, time.time(), 0.0
    try:
        while held is None:
            for fd in fds:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    held = fd
                    break
                except OSError:
                    continue
            if held is None:
                if time.time() - last > 300:
                    log(f"build slot: all {slots} full-build slots busy, waiting ({time.time() - t0:.0f} s so far)")
                    last = time.time()
                time.sleep(10)
        log(f"build slot acquired after {time.time() - t0:.0f} s (cap {slots} concurrent full builds)")
        yield held
    finally:
        for fd in fds:
            try:
                if fd == held:
                    fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
        if held is not None:
            log(f"build slot released after {time.time() - t0:.0f} s")
