# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real-activation dump for the layer-moe task: runs the fp32 CPU oracle (read-only ref worktree) over real prompts through
layers 0 (dense) and 1 (MoE) and saves, per layer, the MLP-sublayer input (post_attention_layernorm output, what the MLP/MoE
block receives), the block output y (before the iHC expand), the oracle router (weights, ids) and the shared-expert output.

    python tools/dump_lmoe_acts.py OUT_DIR [REF_PKG_DIR]
"""
import json
import os
import sys

import torch

REF = "/mnt/tt-data/ssinghal/wt/hy4_ref/models/demos/blackhole/hy4_preview/reference"
import models.demos.blackhole.hy4_preview as _p  # noqa: E402

_p.__path__.append(os.path.dirname(REF))  # exposes .reference from the read-only oracle tree
from models.demos.blackhole.hy4_preview.reference import oracle as O  # noqa: E402
from models.demos.blackhole.hy4_preview.reference.dump_ref import LONG_UNIT, SHORT  # noqa: E402


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("HY4_THREADS", "24")))
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file("/mnt/tt-data/Hy4-Preview/tokenizer.json")
    mid = "".join(LONG_UNIT.format(i=i, a=(i * 7) % 23 + 1, b=(i * 5) % 17 + 1) for i in range(14))
    texts = SHORT + [mid]
    ids = [torch.tensor([[120000] + tok.encode(t).ids]) for t in texts]
    o = O.Hy4Oracle()
    conv = O.Conv()
    rec = {}
    orig_dense, orig_multi, orig_router, orig_shared = O.dense_mlp, O.moe_multi, O.router, O.shared_expert

    def dense(h, sd, cfg, c):
        y = orig_dense(h, sd, cfg, c)
        rec.setdefault("dense", []).append((h.clone(), y.clone()))
        return y

    def multi(ck, layer, hs, sds, cfg, convs):
        ys, routes = orig_multi(ck, layer, hs, sds, cfg, convs)
        for h, y, r in zip(hs, ys, routes):
            rec.setdefault("moe", []).append((h.clone(), y.clone(), r.clone()))
        return ys, routes

    O.dense_mlp, O.moe_multi = dense, multi
    x4s = [o.embed_ids(i) for i in ids]
    idx_states = [{} for _ in ids]
    for L in (0, 1):
        x4s = O.decoder_layers_multi(o.ck, L, x4s, [conv] * len(ids), idx_states, None)
        print("layer", L, "done", flush=True)
    torch.save(
        {"prompts": [i.tolist() for i in ids], "dense": rec["dense"], "moe": rec["moe"]},
        os.path.join(out, "acts_l0_l1.pt"),
    )
    # layer-1 router weights / bias / shared-expert weights for the CPU router test
    sd = o.ck.layer_state(1)
    keep = {k: v for k, v in sd.items() if k.startswith("mlp.gate.") or k.startswith("mlp.shared_experts.")}
    torch.save(keep, os.path.join(out, "l1_gate_shared.pt"))
    sd0 = o.ck.layer_state(0)
    torch.save({k: v for k, v in sd0.items() if k.startswith("mlp.")}, os.path.join(out, "l0_dense.pt"))
    json.dump({"n_tokens": [i.shape[1] for i in ids]}, open(os.path.join(out, "meta.json"), "w"))


if __name__ == "__main__":
    main()
