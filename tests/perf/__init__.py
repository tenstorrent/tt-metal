# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Generic runtime performance test framework.

A benchmark binary reports results in a shared JSON format (perf_contract.hpp), suites.yaml says how to run and
judge it, and this package compares every case against a checked-in golden, reports, and updates goldens.
Nothing here is specific to one benchmark. See README.md.
"""
