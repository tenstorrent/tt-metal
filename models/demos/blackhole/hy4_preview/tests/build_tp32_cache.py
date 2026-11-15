# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Build the TP32 expert weight cache (layout tp32-v1, W2 v1) for layers L0..L1-1 of the bf16 checkpoint. Packing is host-side (torch); ttnn needs ONE visible device only to initialise
(no mesh is opened), so several processes with disjoint layer ranges run in parallel, one slot each.
Env: L0 (first layer, default 1), L1 (last+1, default 78), OUTDIR (cache root, default $HY4_TP32_CACHE_DIR or /hy4tp32), HY4_CKPT. Layers whose dir exists are skipped (complete-or-absent: tmp dir + rename).
Per layer: ~23 s checkpoint read (NFS) + pack + dump + sha256; 5.44 GB on disk per layer (32 chips x (113.2 + 56.6) MB / ... see LAYOUT.md)."""
import os
import time

from models.demos.blackhole.hy4_preview.tt import hy4_tp32_cache as C
from models.demos.blackhole.hy4_preview.tt.hy4_moe_weights import load_moe_layer


def main():
    out = os.environ.get("OUTDIR") or C.cache_root() or "/hy4tp32"
    for L in range(int(os.environ.get("L0", "1")), int(os.environ.get("L1", "78"))):
        if os.path.exists(os.path.join(C.layer_dir(out, L), "moe_meta_tp32.json")):
            print(f"layer {L} exists, skip", flush=True)
            continue
        t0 = time.time()
        W = load_moe_layer(L, experts=list(range(256)))
        t1 = time.time()
        w0, w1, w2 = (W[k].reshape(256, *W[k].shape[-2:]) for k in ("w0", "w1", "w2"))
        d = C.build_layer(w0, w1, w2, L, out)
        print(f"layer {L} done: read {t1-t0:.0f}s pack+dump {time.time()-t1:.0f}s -> {d}", flush=True)
        del W, w0, w1, w2


if __name__ == "__main__":
    main()
