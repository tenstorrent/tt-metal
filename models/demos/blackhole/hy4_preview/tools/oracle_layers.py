# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-layer oracle streams for the 4 short dump prompts (all 78 layers, fp32 CPU, layer streaming): OUT/layerXX_pP.pt = {'out': [1,S,4,D]} (device chain compares layer by layer).
usage: python -m ...tools.oracle_layers OUT [--prompts 0,1,2,3]; needs the ref-oracle package (wt/hy4_ref) on PYTHONPATH; resumable."""
import os
import sys
import time

import torch

from models.demos.blackhole.hy4_preview.reference.oracle import Conv, Hy4Oracle, decoder_layers_multi


def main():
    out = sys.argv[1]
    sel = (
        [int(x) for x in sys.argv[sys.argv.index("--prompts") + 1].split(",")]
        if "--prompts" in sys.argv
        else [0, 1, 2, 3]
    )
    torch.set_num_threads(int(os.environ.get("HY4_THREADS", "16")))
    os.makedirs(out, exist_ok=True)
    ids_all = torch.load("/mnt/tt-data/ssinghal/hy4-logs/program/chain/prompt_ids.pt")
    ids = [torch.tensor([ids_all[i]]) for i in sel]
    o, conv = Hy4Oracle(), Conv()
    st_f = os.path.join(out, "state.pt")
    if os.path.exists(st_f):
        st = torch.load(st_f)
        x4s, start, idx = st["x4s"], st["layer"], st["idx"]
    else:
        x4s, start, idx = [o.embed_ids(i) for i in ids], 0, [{} for _ in ids]
    t0 = time.time()
    for L in range(start, o.cfg.n_layers):
        x4s = decoder_layers_multi(o.ck, L, x4s, [conv] * len(ids), idx, None)
        for p, x in zip(sel, x4s):
            torch.save({"out": x.clone()}, os.path.join(out, f"layer{L:02d}_p{p}.pt"))
        torch.save({"x4s": x4s, "layer": L + 1, "idx": idx}, st_f + ".tmp")
        os.replace(st_f + ".tmp", st_f)
        print(f"layer {L} done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
