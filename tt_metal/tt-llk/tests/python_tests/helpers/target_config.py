# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""Where the tests run, and what that target can do.

The backend is resolved at import time from the environment, because the pytest
plugin has to know whether tt-exalens runs in-process before it imports anything
that reaches for a device context -- long before pytest options are parsed. The
pytest options then refine the same singleton in update_from_pytest_config().

Capabilities are named here rather than re-derived at each call site, because no
single predicate separates the backends the same way: ttsim groups with VCS on
register access, with silicon on the Tensix dump GPRs, and with Versim on process
model. Asking "can I reach the register bus" says why a branch exists and survives
a backend gaining the capability later; asking "is this Versim" does neither.

This module must stay dependency-free (stdlib only): it is imported before the
tt-exalens context exists.
"""

import glob
import os
import sys
from enum import Enum

# TT_METAL_SIMULATOR is the canonical env var (matches the tt-metal runtime and
# the ttsim README); TT_UMD_SIMULATOR_PATH is the alias the RTL-simulator
# workflow already used.
_SIMULATOR_PATH_ENV_VARS = ("TT_METAL_SIMULATOR", "TT_UMD_SIMULATOR_PATH")


class Backend(Enum):
    """Execution target. Silicon plus the three simulators reached via tt-exalens."""

    SILICON = "silicon"
    TTSIM = "ttsim"  # libttsim_*.so, models a whole chip
    VCS = "vcs"  # RTL simulation, behaves like silicon
    VERSIM = "versim"  # functional Tensix model, no host-side register bus

    def __str__(self):
        return self.value


def get_simulator_path() -> str | None:
    """Path of the simulator to run against, or None if no env var is set."""
    for name in _SIMULATOR_PATH_ENV_VARS:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _is_versim_path(path: str) -> bool:
    """Whether path is a Versim build directory.

    tt-umd-simulators produces the run.sh that UMD spawns plus the versim-<arch>
    executable it launches. Both are checked so a VCS build directory, which also
    has a run.sh, is not mistaken for one.
    """
    if not os.path.isfile(os.path.join(path, "run.sh")):
        return False
    return any(
        os.access(candidate, os.X_OK)
        for candidate in glob.glob(os.path.join(path, "versim-*"))
    )


def detect_backend(asked_for_simulator: bool | None = None) -> Backend:
    """Resolve the backend from the environment. Safe to call at import time.

    The simulator env var alone is not enough: it is often left set while running
    on silicon, so --run-simulator has to agree. At import time that can only be
    read off argv, which misses an option supplied any other way (pytest.ini
    addopts, for one); pass asked_for_simulator once pytest has parsed the real
    value. --compile-producer only builds ELFs and never talks to a device. xdist
    workers inherit env vars but not the controller's argv, so the env var is
    trusted there.
    """
    if asked_for_simulator is None:
        is_xdist_worker = "PYTEST_XDIST_WORKER" in os.environ
        asked_for_simulator = is_xdist_worker or (
            "--run-simulator" in sys.argv and "--compile-producer" not in sys.argv
        )
    path = get_simulator_path()
    if not asked_for_simulator or not path:
        return Backend.SILICON
    if path.endswith(".so"):
        return Backend.TTSIM
    if os.path.isdir(path) and _is_versim_path(path):
        return Backend.VERSIM
    return Backend.VCS


class TestTargetConfig:
    """Where the tests run (silicon or a simulator) and what that target supports."""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TestTargetConfig, cls).__new__(cls)
        return cls._instance

    def __init__(
        self,
        run_simulator=None,
        simulator_port=5555,
        device_id=0,
        log_level="INFO",
    ):
        """
            Initializes the test configuration in regards to using the simulator.
        Args:
            run_simulator (bool): True if the test is run on simulator, False if on
                silicon. Defaults to whatever detect_backend() found in the env.
            simulator_port (int): Simulator server port number
            device_id (int): ID number of the device to send message to.
            log_level (str): Log level
        """
        # Only initialize once
        if not TestTargetConfig._initialized:
            self.backend: Backend = detect_backend()
            self.simulator_path: str | None = get_simulator_path()
            self.run_simulator: bool = (
                self.backend is not Backend.SILICON
                if run_simulator is None
                else run_simulator
            )
            self.simulator_port: int = simulator_port
            self.device_id: int = device_id
            self.log_level: str = log_level
            self.reset_simulator_per_test: bool = False
            TestTargetConfig._initialized = True

    def update_from_pytest_config(self, config):
        """Update only the simulator related settings from pytest config"""
        self.run_simulator = config.getoption("--run-simulator", default=False)
        self.simulator_port = config.getoption("--port", default=5555)
        self.reset_simulator_per_test = config.getoption(
            "--reset-simulator-per-test", default=False
        )
        # Re-resolve now that --run-simulator has been parsed properly, rather than
        # sniffed off argv at import time.
        self.backend = detect_backend(asked_for_simulator=self.run_simulator)

    # --- capabilities -----------------------------------------------------
    # Derived from the backend rather than stored, so they cannot drift out of
    # sync with it when update_from_pytest_config() re-resolves.

    @property
    def runs_in_process(self) -> bool:
        """Whether tt-exalens drives the simulator in-process, with no server.

        ttsim always has. Versim joins it because UMD's simulator reset API takes
        a tt_umd.RiscType, which tt-exalens cannot serialize over its Pyro5 server
        (RiscType is absent from ttexalens.server.UMD_SERIALIZABLE_TYPES). UMD
        spawns the Versim process itself, so no server is needed either way.
        """
        return self.backend in (Backend.TTSIM, Backend.VERSIM)

    @property
    def has_host_register_access(self) -> bool:
        """Whether the host can reach the Tensix register bus over the NOC.

        False on Versim: host reads and writes resolve straight into L1, so
        accesses to the 0xFFB..... aperture silently alias -- a read returns
        whatever L1 holds at the aliased offset and a write lands somewhere in L1.
        Register access from inside the Tensix (the RISCs' own reg_read/reg_write)
        is unaffected, which is why BRISC firmware can still drive TRISC reset.
        """
        return self.backend is not Backend.VERSIM

    @property
    def has_risc_debug(self) -> bool:
        """Whether the RISC debug hardware (halt, ebreak, callstack) is reachable.

        It sits behind the same register bus as has_host_register_access.
        """
        return self.has_host_register_access

    @property
    def models_tensix_dump_gprs(self) -> bool:
        """Whether the target models the GPRs the Tensix dump helper overrides.

        Silicon and ttsim do; the RTL simulators do not.
        """
        return self.backend in (Backend.SILICON, Backend.TTSIM)

    @property
    def can_reset_card(self) -> bool:
        """Whether tt-smi can reset the device. Simulators have no card to reset."""
        return self.backend is Backend.SILICON

    @property
    def kernel_timeout_s(self) -> int:
        """Seconds to wait for a kernel to report completion.

        Simulators are orders of magnitude slower than silicon, and the wait is a
        poll on an L1 mailbox rather than anything backend-specific.
        """
        return 600 if self.run_simulator else 2

    @property
    def brisc_command_timeout_s(self) -> int:
        """Seconds to wait for BRISC to acknowledge a command mailbox write."""
        return 600 if self.run_simulator else 1
