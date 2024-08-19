# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from tests.tt_metal.tools.profiler import test_device_profiler


def test_multi_op_gs_no_reset():
    test_device_profiler.test_multi_op()
