# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host-side guards for ``helpers/sim_host.py``: the simulation host must accept what the
pinned tt-umd's ``SimulationConnector.discover()`` returns."""

import io
import sys
from types import SimpleNamespace

import tt_umd
from helpers import sim_host


def test_sim_host_unpacks_discover_result(monkeypatch, capsys):
    """tt-umd >= 0.9.14 returns (connection, devices, cluster_descriptor) from discover()."""
    devices = {0: object()}
    connection = SimpleNamespace(
        role=tt_umd.SimulationConnector.Role.HOST,
        # Not RTL, so main() does not try to park cores on the fake device.
        backend=None,
        arch=None,
        server_directory="/tmp/fake-sim-server",
    )
    cluster_descriptor = object()

    class FakeConnector:
        Role = tt_umd.SimulationConnector.Role

        @staticmethod
        def discover(options):
            return connection, devices, cluster_descriptor

    monkeypatch.setattr(tt_umd, "SimulationConnector", FakeConnector)
    monkeypatch.setattr(sys, "argv", ["sim_host.py", "/tmp/fake-sim-build"])
    monkeypatch.setattr(sys, "stdin", io.StringIO("exit\n"))
    # main() installs a SIGTERM handler; keep it from leaking out of the test.
    monkeypatch.setattr(sim_host.signal, "signal", lambda *args: None)

    assert sim_host.main() == 0
    out = capsys.readouterr().out
    assert f"{sim_host.READY_MARKER} /tmp/fake-sim-server" in out
    # The host releases the devices on exit.
    assert devices == {}
