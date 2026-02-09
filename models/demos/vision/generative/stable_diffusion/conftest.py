# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


def pytest_configure(config):
    """Override global timeout setting for Stable Diffusion tests

    The device performance tests run the actual model tests as subprocesses via tracy,
    and those subprocesses would otherwise be subject to the global 300s timeout from
    pytest.ini. This disables the timeout for all stable diffusion tests to allow
    longer-running performance tests to complete.
    """
    config.option.timeout = 0
