# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers of the read-only ethernet counter probes: core listing, register and L1 reads driven by the
tt-llk tables, card busy check. Nothing here writes a device register or L1.
"""

import os
import sys
from pathlib import Path

# tt_llk_perf is a sibling of this directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from tt_llk_perf import headers  # noqa: E402

BLACKHOLE_PCS_STATUS = 0xFFB9800C
BLACKHOLE_ERR_STAT = 0xFFB980D8
MAC_TX_CFG = 0xFFBA2200
MAC_RX_CFG = 0xFFBA2400
FAILED_READ = 0xFFFFFFFF


def card_busy(device_path):
    """True when another process has the device node open."""
    target = os.path.realpath(device_path)
    me = str(os.getpid())
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or pid == me:
            continue
        fd_dir = os.path.join("/proc", pid, "fd")
        try:
            fds = os.listdir(fd_dir)
        except OSError:
            continue
        for fd in fds:
            try:
                if os.path.realpath(os.path.join(fd_dir, fd)) == target:
                    return True
            except OSError:
                continue
    return False


def executable(path):
    """Absolute path of an existing executable file, or exit."""
    resolved = Path(path).resolve()
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise SystemExit(f"not an executable file: {path}")
    return str(resolved)


def default_device_path():
    return os.environ.get("TT_DEVICE_PATH", "/dev/tenstorrent/1")


def fail(msg):
    print(msg, file=sys.stderr)
    raise SystemExit(2)


class Cards:
    """Every ethernet core of every chip exalens sees, read through exalens with its safe mode on."""

    def __init__(self):
        from ttexalens.tt_exalens_init import init_ttexalens

        self.context = init_ttexalens(safe_mode=True)
        self.devices = list(self.context.devices.values())

    def arch(self, device):
        if device.is_blackhole():
            return "blackhole"
        if device.is_wormhole():
            return "wormhole"
        fail(f"unsupported architecture of device {device.id}")

    def eth_cores(self, device):
        from ttexalens.coordinate import OnChipCoordinate

        return [
            OnChipCoordinate.create(loc.to_str("noc0"), device, "noc0")
            for loc in device.get_block_locations("eth")
        ]


def link_up(arch, core):
    """Live link state; Blackhole only, from the PCS status register (1 means up)."""
    if arch != "blackhole":
        return None
    return core.noc_read32(BLACKHOLE_PCS_STATUS) == 1


def queue_registers(arch):
    """[(name, queue, address)] of every queue register of the tables."""
    layout = headers.eth_layout(arch)
    out = []
    for name, offset in headers.eth_tx_queue_table(arch):
        for q in range(layout.num_txq):
            out.append((name, q, layout.txq_base + q * layout.queue_stride + offset))
    for name, offset in headers.eth_rx_queue_table(arch):
        for q in range(layout.num_rxq):
            out.append((name, q, layout.rxq_base + q * layout.queue_stride + offset))
    return out


def read_u64(core, address, high_first=False):
    first = core.noc_read32(address)
    second = core.noc_read32(address + 4)
    return (first << 32 | second) if high_first else (second << 32 | first)


def snapshot(arch, core):
    """{(name, queue): value} of every queue register and L1 value of one core; queue is -1 for no queue."""
    values = {}
    for name, q, address in queue_registers(arch):
        values[(name, q)] = core.noc_read32(address)
    for entry in headers.eth_l1_table(arch):
        if entry.words == 1:
            values[(entry.name, entry.queue)] = core.noc_read32(entry.address)
        else:
            values[(entry.name, entry.queue)] = read_u64(
                core, entry.address, entry.high_first
            )
    return values
