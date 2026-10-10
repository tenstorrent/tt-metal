# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""UMD simulation host that all test processes attach to; run by SimulationServer.

Prints READY_MARKER and the server directory once serving, then serves until "exit"
on stdin, stdin EOF or SIGTERM.

Usage: sim_host.py <RTL simulator build directory>
"""

import signal
import sys
from pathlib import Path

import tt_umd

READY_MARKER = "SIM_HOST_READY"


def _park_tensix_cores(arch, devices) -> None:
    """Same RTL bring-up tt-exalens does as host: release every Tensix onto a harmless loop."""
    for device in devices.values():
        soc_descriptor = device.get_soc_descriptor()
        for core in soc_descriptor.get_cores(tt_umd.CoreType.TENSIX):
            if arch == tt_umd.ARCH.BLACKHOLE:
                core_noc0 = soc_descriptor.translate_coord_to(
                    core, tt_umd.CoordSystem.NOC0
                )
                # jal x0, 0
                device.noc_write32(core_noc0.x, core_noc0.y, 0, 0x6F)
                device.deassert_risc_reset(core, tt_umd.RiscType.BRISC)
            elif arch == tt_umd.ARCH.QUASAR:
                device.deassert_risc_reset(core, tt_umd.RiscType.ALL)


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2

    # SystemExit runs the finally below, which releases the emulator.
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(0))

    options = tt_umd.SimulationConnectorOptions()
    options.simulator_directory = Path(sys.argv[1])
    options.serve_over_sockets = True
    connection, devices = tt_umd.SimulationConnector.discover(options)
    try:
        if connection.role != tt_umd.SimulationConnector.Role.HOST:
            print(
                f"{sys.argv[1]} is the directory of a running simulation server, "
                "not a simulator build",
                file=sys.stderr,
            )
            return 1
        if connection.backend == tt_umd.SimulationBackendType.RTL:
            _park_tensix_cores(connection.arch, devices)

        print(f"{READY_MARKER} {connection.server_directory}", flush=True)
        for line in sys.stdin:
            if line.strip() == "exit":
                break
    finally:
        devices.clear()
    return 0


if __name__ == "__main__":
    sys.exit(main())
