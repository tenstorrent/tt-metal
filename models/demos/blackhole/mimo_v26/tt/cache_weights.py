# SPDX-License-Identifier: Apache-2.0
"""Build the converted-weight cache (bfp8, sharded for the 4x8 mesh) for a range of layers; run one process per host on disjoint ranges.

    python -m models.demos.blackhole.mimo_v26.tt.cache_weights --first 2 --last 17
The mesh is opened only to create the shard mappers (no device tensors are made). Existing files are kept (hit), so the tool is resumable.
"""
import argparse
import time

import ttnn
from models.demos.blackhole.mimo_v26.reference.loader import Ckpt
from models.demos.blackhole.mimo_v26.tt import weights as W
from models.demos.blackhole.mimo_v26.tt.layer import MiMoDecoderLayer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, required=True)
    ap.add_argument("--last", type=int, required=True)
    ap.add_argument("--expert-op", default="mm")
    a = ap.parse_args()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(4, 8))
    try:
        ck = Ckpt()
        wc = W.WeightCache(mesh, build_only=True)
        for L in range(a.first, a.last + 1):
            t0 = time.time()
            MiMoDecoderLayer(mesh, L, ck, wc, max_pos=128, expert_op=a.expert_op, build_only=True)
            print(f"WCACHE layer {L} done in {time.time() - t0:.1f}s stats={wc.stats}", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
