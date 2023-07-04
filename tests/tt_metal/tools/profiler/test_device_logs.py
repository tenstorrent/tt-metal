# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

# THIS FILE IS AUTO-GENERATED
# Refer to the profiler README to learn how to generate this file using populate_golden.py

from tests.tt_metal.tools.profiler.device_log_run import run_test


@run_test
def test_add_two_ints():
    pass


@run_test
def test_circular_trace_order():
    pass


@run_test
def test_compute():
    pass


@run_test
def test_full_buffer_nc_b_risc():
    pass


@run_test
def test_full_buffer_nc_b_t0_t1_t2_risc():
    pass


@run_test
def test_full_buffer_nc_risc():
    pass


@run_test
def test_grayskull_new_header():
    pass


@run_test
def test_large():
    pass


@run_test
def test_matmul():
    pass


@run_test
def test_matmul_small():
    pass


@run_test
def test_missing_brisc():
    pass


@run_test
def test_missing_ncrisc():
    pass


@run_test
def test_missing_all_defualt_markers():
    pass


@run_test
def test_multi_core_multi_launch():
    pass


@run_test
def test_very_long_launch_delta():
    pass


@run_test
def test_wormhole_multi_core():
    pass


@run_test
def test_wormhole_single_core():
    pass
