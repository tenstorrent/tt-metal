# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


# These inputs override the default inputs used by ethernet microbenchmarks
def pytest_addoption(parser):
    parser.addoption("--num-iterations", action="store", type=int, help="Number of iterations to run each test config")
    parser.addoption("--num-packets", action="store", type=int, help="Number of packets to send during each iteration")
    parser.addoption("--packet-size", action="store", type=int, help="Packet size in bytes")
    parser.addoption(
        "--extended",
        action="store_true",
        default=False,
        help="Run extended/exhaustive sweep tests (skipped by default)",
    )
