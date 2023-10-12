# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import sys
import os

sys.path.insert(0, os.environ["TT_METAL_HOME"] + "/scripts/profiler")
from postproc_unary import parse_device_log_for_opprofiler, TRISC
import tempfile
import numpy as np


def test_basic():
    data = """Chip clock is at 1202 Mhz; ARCH: grayskull, CHIP_FREQ[MHz]: 1202;
PCIe slot, core_x, core_y, RISC processor type, timer_id, time[cycles since reset]
0, 0, 0, NCRISC, 1, 344477331766589
0, 0, 0, NCRISC, 2, 344477331770357
0, 0, 0, NCRISC, 3, 344477331770821
0, 0, 0, NCRISC, 4, 344477331770874
0, 0, 0, BRISC, 1, 344477331764845
0, 0, 0, BRISC, 2, 344477331771479
0, 0, 0, BRISC, 3, 344477331772669
0, 0, 0, BRISC, 4, 344477331772756
0, 0, 0, TRISC_0, 1, 344477331764931
0, 0, 0, TRISC_0, 2, 344477331765495
0, 0, 0, TRISC_0, 9997, 344477331765540
0, 0, 0, TRISC_0, 9998, 344477331770911
0, 0, 0, TRISC_0, 3, 344477331770962
0, 0, 0, TRISC_0, 4, 344477331771040
0, 0, 0, TRISC_1, 1, 344477331764932
0, 0, 0, TRISC_1, 2, 344477331765093
0, 0, 0, TRISC_1, 9997, 344477331765148
0, 0, 0, TRISC_1, 9998, 344477331772153
0, 0, 0, TRISC_1, 3, 344477331772217
0, 0, 0, TRISC_1, 4, 344477331772297
0, 0, 0, TRISC_2, 1, 344477331764930
0, 0, 0, TRISC_2, 2, 344477331765551
0, 0, 0, TRISC_2, 9997, 344477331765595
0, 0, 0, TRISC_2, 9998, 344477331767430
0, 0, 0, TRISC_2, 3, 344477331767487
0, 0, 0, TRISC_2, 4, 344477331772257
"""
    with tempfile.TemporaryDirectory() as td:
        fname = td + "/demo.csv"
        fp = open(fname, "w")
        fp.write(data)
        fp.close()
        trisc: TRISC = parse_device_log_for_opprofiler(fname)
        assert isinstance(trisc, TRISC)
        assert np.isclose(trisc.start_exec_end, np.array([5371.0, 7005.0, 1835.0])).all()
