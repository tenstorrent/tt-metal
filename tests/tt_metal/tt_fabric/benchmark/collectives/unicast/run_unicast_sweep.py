# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
import subprocess

BIN = "build/test/tt_metal/tt_fabric/bench_unicast"
SRC = "0:0"
DST = "0:1"
PAGE = "2048"
ITERS = "5"
WARMUP = "1"
CSV = "artifacts/unicast_sweep.csv"

sizes = [65536, 131072, 262144, 524288]  # bytes
recv_cores = [(0, 0), (1, 0), (2, 0)]  # x,y list

for size in sizes:
    for rx, ry in recv_cores:
        cmd = [
            BIN,
            "--src-dev",
            SRC,
            "--dst-dev",
            DST,
            "--size",
            str(size),
            "--page",
            PAGE,
            "--send-core",
            "0,0",
            "--recv-core",
            f"{rx},{ry}",
            "--iters",
            ITERS,
            "--warmup",
            WARMUP,
            "--format",
            "csv",
            "--csv",
            CSV,
        ]
        print(">>", " ".join(cmd))
        subprocess.run(cmd, check=True)
