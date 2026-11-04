# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os


def simulator_path() -> str:
    """TT_METAL_SIMULATOR, or its alias TT_UMD_SIMULATOR_PATH; empty without a simulator."""
    return os.environ.get("TT_METAL_SIMULATOR") or os.environ.get("TT_UMD_SIMULATOR_PATH") or ""


def is_ttsim(path: str | None) -> bool:
    """ttsim is a shared library run in process; RTL simulators run behind a simulation server."""
    return bool(path) and path.endswith(".so")


class TestTargetConfig:
    """TestTargetConfig class represents where the test will be run: simulator or silicon."""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TestTargetConfig, cls).__new__(cls)
        return cls._instance

    def __init__(
        self,
        run_simulator=False,
        device_id=0,
        log_level="INFO",
    ):
        """
            Initializes the test configuration in regards to using the simulator.
        Args:
            run_simulator (bool): True if the test is run on simulator, False if on silicon
            device_id (int): ID number of the device to send message to.
            log_level (str): Log level
        """
        # Only initialize once
        if not TestTargetConfig._initialized:
            self.run_simulator: bool = run_simulator
            self.device_id: int = device_id
            self.log_level: str = log_level
            self.reset_simulator_per_test: bool = False
            TestTargetConfig._initialized = True

    def runs_on_ttsim(self) -> bool:
        return self.run_simulator and is_ttsim(simulator_path())

    def update_from_pytest_config(self, config):
        """Update only the simulator related settings from pytest config"""
        self.run_simulator = config.getoption("--run-simulator", default=False)
        self.reset_simulator_per_test = config.getoption(
            "--reset-simulator-per-test", default=False
        )
