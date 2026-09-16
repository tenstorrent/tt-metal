#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Recover a Galaxy (6U) cluster on the runner host. Two subcommands:

free   SIGKILL stray processes still holding a /dev/tenstorrent handle. A
       leftover process from a previous CI run keeps the device's sysmem/TLB
       windows claimed, which a board reset does not release, so the next
       cluster open fails with "tt_tlb_alloc failed with error code -12". Needs
       root (via sudo) to read other users' /proc/<pid>/fd, e.g. a leftover
       privileged container.

reset  Reset the cluster with tt-smi. The per-board PCIe reset (-r) is invalid
       on topology-6u runners and fails instantly, so this uses -glx_reset_auto
       (matching the topology-6u branch of TT_SMI_RESET_COMMAND in
       ttnn-run-sweeps.yaml).

The sweeps runner does the same cleanup in-process; see ResetUtil._free_device in
tests/sweep_framework/framework/tt_smi_util.py. This stays a standalone copy
rather than an import because sudo runs outside the job's setup-python env, so it
must have no third-party dependencies.
"""

import os
import signal
import subprocess
import sys

# The host's tt-smi is a venv console script, not on PATH. Its shebang points at
# the venv python, so the absolute path needs no activation.
HOST_TT_SMI = "/opt/tt_metal_infra/provisioning/provisioning_env/bin/tt-smi"


def self_and_ancestors():
    """PIDs of this process and its ancestor chain -- never to be killed."""
    pids = set()
    pid = os.getpid()
    while pid > 1 and pid not in pids:
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


def describe(pid):
    """Best-effort '<pid> (<cmdline>)' label, read before the process is killed."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            cmdline = f.read().replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        cmdline = ""
    return f"{pid} ({cmdline or '?'})"


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


def reset_cluster():
    return subprocess.call([HOST_TT_SMI, "-glx_reset_auto"])


def free_devices():
    holders = device_holder_pids()
    if not holders:
        print("No stray processes are holding /dev/tenstorrent.")
        return 0
    # Log the cmdlines before killing -- afterwards there is no way to tell from
    # the CI log what a bare pid was.
    print("Killing stray process(es) holding /dev/tenstorrent:")
    for pid in holders:
        print(f"  {describe(pid)}")
    for pid in holders:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError as e:
            # already gone, or not permitted (need root) -- report and continue
            print(f"  pid {pid}: could not kill ({e})")
    return 0


def main(argv):
    if len(argv) != 1 or argv[0] not in ("free", "reset"):
        print(f"usage: {sys.argv[0]} free|reset", file=sys.stderr)
        return 2
    return free_devices() if argv[0] == "free" else reset_cluster()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
