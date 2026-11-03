# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""common.zero_dram on device (long_ctx.check_zero_dram), without building a model: one tensor per call, 28 per call
(as PPPrefill.reset) twice and one per call again, each zeroed buffer read back zero and its neighbours untouched.

usage: python -m qwen36_resident.tests.zero_dram_check
"""
import faulthandler
import sys

from qwen36_resident.tests.long_ctx import check_zero_dram, close_mesh, open_mesh

faulthandler.dump_traceback_later(300, repeat=True, file=sys.stderr)
mesh = open_mesh()
try:
    check_zero_dram(mesh)
    print("ZERO_DRAM CHECK PASS", flush=True)
finally:
    close_mesh(mesh)
