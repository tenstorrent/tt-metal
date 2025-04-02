# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


# These inputs override the default inputs used by ethernet microbenchmarks
def pytest_addoption(parser):
    parser.addoption("--num-iterations", action="store", type=int, help="Number of iterations to run each test config")
    parser.addoption("--num-packets", action="store", type=int, help="Number of packets to send during each iteration")
    parser.addoption("--packet-size", action="store", type=int, help="Packet size in bytes")
