# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Per-core cycle counters of one sparse-row launch (kernels built with KDEBUG), mean and max over cores.

usage: python models/experimental/hpcg/demo/kernel_counters.py nx op [--per-core]
op: gs<l> (post-smoother of level l), gspre<l> (pre-smoother), spmv<l>, cg (the CG SpMV).
"""

import sys

import numpy as np
import ttnn

from models.experimental.hpcg.tt.hpcg import HPCG

READER = ["window", "phase_wait", "-", "gather", "dram_wait", "cb_reserve", "issue", "total"]
WRITER = ["ready", "phase_wait", "issue", "gather", "dram_wait", "cb_reserve", "emit", "total"]
TRISC = {0: "pack_wait_offsets", 1: "pack_wait_space", 2: "pack_gather", 4: "unpack_wait_offsets", 5: "unpack_gather"}


def main(nx, which, per_core):
    device = ttnn.open_device(device_id=0)
    try:
        h = HPCG(device, nx, kernel_debug=True)
        if which == "cg":
            key = "spmv_cg"
        elif which.startswith("gspre"):
            key = ("gs_pre", int(which[5:]))
        elif which.startswith("gs"):
            key = ("gs", int(which[2:]))
        else:
            key = None
        op = h.spmv(int(which[4:])) if key is None else h.ops[key]
        op()
        op()
        ttnn.synchronize_device(device)
        d = ttnn.to_torch(h.debug).numpy().view(np.uint32).astype(np.int64)  # 32-bit counters under an fp32 label
        rd, wr = d[0::2, :8], d[1::2, :8]
        act = rd[:, 7] > 0
        print(which, "cores", int(act.sum()), "(cycles at the AI clock; mean / max over cores)")
        print(
            "reader", {READER[i]: (int(rd[act, i].mean()), int(rd[act, i].max())) for i in range(8) if READER[i] != "-"}
        )
        print("writer", {WRITER[i]: (int(wr[act, i].mean()), int(wr[act, i].max())) for i in range(8)})
        tc = d[1::2, 8:16]
        print("trisc", {n: (int(tc[act, i].mean()), int(tc[act, i].max())) for i, n in TRISC.items()})
        if per_core:
            g = device.compute_with_storage_grid_size()
            for name, col in (("window load", 0), ("total", 7)):
                print(f"per-core reader {name} kcycles (rows y = 0..{g.y - 1}, columns x = 0..{g.x - 1}):")
                for y in range(g.y):
                    print("  ", " ".join(f"{rd[y * g.x + x, col] // 1000:5d}" for x in range(g.x)))
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2], "--per-core" in sys.argv)
