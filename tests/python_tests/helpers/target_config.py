# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC


class TestTargetConfig:
    """TestTargetConfig class represents where the test will be run: simulator or silicon."""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TestTargetConfig, cls).__new__(cls)
        return cls._instance

    def __init__(
        self, run_simulator=False, simulator_port=5555, device_id=0, log_level="INFO"
    ):
        """
            Initializes the test configuration in regards to using the simulator.
        Args:
            run_simulator (bool): True if the test is run on simulator, False if on silicon
            simulator_port (int): Simulator server port number
            device_id (int): ID number of the device to send message to.
            log_level (str): Log level
        """
        # Only initialize once
        if not TestTargetConfig._initialized:
            self.run_simulator: bool = run_simulator
            self.simulator_port: int = simulator_port
            self.device_id: int = device_id
            self.log_level: str = log_level
            TestTargetConfig._initialized = True

    def update_from_pytest_config(self, config):
        """Update only the simulator related settings from pytest config"""
        self.run_simulator = config.getoption("--run_simulator", default=False)
        self.simulator_port = config.getoption("--port", default=5555)


def initialize_test_target_from_pytest(config):
    """Initialize the global test configuration from pytest command line options."""
    test_target = TestTargetConfig()
    test_target.update_from_pytest_config(config)
