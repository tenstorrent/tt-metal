# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Teacher-forced oracle check of the device decode tokens (CPU, fp32, layer streaming; ref-oracle code from wt/hy4_ref on PYTHONPATH).
usage: python oracle_tf.py decode_tokens_<tag>.json OUT.json [--steps N]
For every row: ids = prompt + generated[:N-1]; one oracle forward; at position len(prompt)-1+j the oracle's greedy token must equal generated[j] (j < N); logs top-1/top-2 gap (near-tie flips)."""
import json
import os
import sys
import time

import torch

from models.demos.blackhole.hy4_preview.reference.oracle import Conv, Hy4Oracle, decoder_layers_multi


def main():
    d = json.load(open(sys.argv[1]))
    out = sys.argv[2]
    N = int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else 16
    torch.set_num_threads(int(os.environ.get("HY4_THREADS", "32")))
    o, conv = Hy4Oracle(), Conv()
    seqs = [torch.tensor([p + g[: N - 1]]) for p, g in zip(d["prompts"], d["generated"])]
    x4s = [o.embed_ids(s) for s in seqs]
    idx = [{} for _ in seqs]
    t0 = time.time()
    for L in range(o.cfg.n_layers):
        x4s = decoder_layers_multi(o.ck, L, x4s, [conv] * len(seqs), idx, None)
        print(f"layer {L} {time.time()-t0:.0f}s", flush=True)
    res = []
    for r, (x4, p, g) in enumerate(zip(x4s, d["prompts"], d["generated"])):
        _, hn = o.head(x4, conv)
        lg = o.logits(hn[:, len(p) - 1 :])[0]  # [N, V]
        top = lg.topk(2, dim=-1)
        orc = top[1][:, 0].tolist()
        gap = (top[0][:, 0] - top[0][:, 1]).tolist()
        match = [orc[j] == g[j] for j in range(N)]
        res.append(
            {
                "row": r,
                "oracle": orc,
                "device": g[:N],
                "match": match,
                "gap": [round(x, 3) for x in gap],
                "n_match": sum(match),
            }
        )
        print(
            f"row {r}: {sum(match)}/{N} match; mismatches at {[j for j in range(N) if not match[j]]} gaps {[round(gap[j],2) for j in range(N) if not match[j]]}",
            flush=True,
        )
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
