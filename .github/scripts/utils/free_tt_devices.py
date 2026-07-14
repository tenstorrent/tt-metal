#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Kill stray host processes still holding a /dev/tenstorrent handle.

A stale or crashed process from a previous CI run can keep the device's
hugepage sysmem NOC space and TLB windows claimed. A board reset
(``tt-smi -glx_reset_auto``) resets the ASIC but does NOT kill that host
process or release its kernel-side resources, so the next cluster open fails
with "tt_tlb_alloc failed with error code -12" / "another process is already
holding the sysmem NOC address space". The UMD warning explicitly says to find
and kill those processes, then retry -- which is what this script does.

Run as root (e.g. via sudo) so /proc/<pid>/fd of processes owned by other users
(such as leftover privileged containers) is readable.
"""

import os
import signal
import sys


def self_and_ancestors():
    """PIDs of this process and its ancestor chain -- never to be killed."""
    pids = set()
    pid = os.getpid()
    while pid and pid > 1 and pid not in pids:
        pids.add(pid)
        try:
            with open(f"/proc/{pid}/stat") as f:
                # ppid is the 4th field, but comm (2nd field) may contain
                # spaces/parens -- parse after the closing ')'.
                data = f.read()
                pid = int(data[data.rindex(")") + 1 :].split()[1])
        except Exception:
            break
    return pids


def device_holder_pids():
    """PIDs (excluding self + ancestors) holding a /dev/tenstorrent fd."""
    protected = self_and_ancestors()
    holders = []
    try:
        entries = os.listdir("/proc")
    except OSError:
        return holders
    for entry in entries:
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid in protected:
            continue
        fd_dir = f"/proc/{entry}/fd"
        try:
            fds = os.listdir(fd_dir)
        except OSError:
            continue
        for fd in fds:
            try:
                target = os.readlink(f"{fd_dir}/{fd}")
            except OSError:
                continue
            if "tenstorrent" in target:
                holders.append(pid)
                break
    return holders


def main():
    holders = device_holder_pids()
    if not holders:
        print("No stray processes are holding /dev/tenstorrent.")
        return 0
    print(f"Killing stray process(es) holding /dev/tenstorrent: {holders}")
    for pid in holders:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError as e:
            # already gone, or not permitted (need root) -- report and continue
            print(f"  pid {pid}: could not kill ({e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
