# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# shellcheck shell=bash
# Extra argv for GoogleTest binaries launched under MPI (mpirun / tt-run).
# When gtest catches C++ exceptions (default), the throwing rank can continue while
# peers block in MPI → deadlock. Disabling catch lets std::terminate run
# mpi_terminate_handler (see ttnn/ttnn/distributed/ULFM.md).
# shellcheck disable=SC2034  # array is consumed by scripts that source this file
MULTIHOST_GTEST_FLAGS=(--gtest_catch_exceptions=0)
