# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""kernel-track: build the TP32 expert weight cache (layout v1) layer by layer on the 4x8 mesh (needs the whole mesh: 32 chips).
Env: L0 (first layer, default 1), L1 (last+1, default 78), OUTDIR (default /hy4cache/tp32), HY4_CKPT. UNTESTED dump/load round trip: run L0=1 L1=2 first and compare via dev_moe_tp32 (TP32_CACHE=...).
Per layer: ~23 s checkpoint read (NFS) + ~55 s pack/upload + dump; 5.44 GB on disk per layer."""
import os
import time

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import close_mesh, open_mesh
from models.demos.blackhole.hy4_preview.tt.hy4_moe_tp32 import Tp32Experts
from models.demos.blackhole.hy4_preview.tt.hy4_moe_weights import load_moe_layer


def main():
    out = os.environ.get("OUTDIR", "/hy4cache/tp32")
    md = open_mesh(trace=10_000_000)
    try:
        for L in range(int(os.environ.get("L0", "1")), int(os.environ.get("L1", "78"))):
            d = f"{out}/layer_{L}"
            if os.path.exists(f"{d}/w2.tensorbin"):
                continue
            t0 = time.time()
            W = load_moe_layer(L, experts=list(range(256)))
            t = Tp32Experts(md, W["w0"][0], W["w1"][0], W["w2"][0])
            os.makedirs(d, exist_ok=True)
            ttnn.dump_tensor(f"{d}/w01.tensorbin", t.w01)
            ttnn.dump_tensor(f"{d}/w2.tensorbin", t.w2)
            print(f"layer {L} done {time.time()-t0:.0f}s", flush=True)
            del t, W
    finally:
        close_mesh(md)


if __name__ == "__main__":
    main()
