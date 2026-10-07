# SPDX-License-Identifier: Apache-2.0
"""Is the KV cache itself run-to-run deterministic? Prefill DEC_N prompts twice (compare every layer's k/v cache bitwise: live slots / dead scratch), then (DEC_TRACE) run STEPS teacher-forced decode steps
twice from a fresh prefill and compare the caches again (positions written by decode). Reports the first layer and slot that differ."""
import json
import os
import sys

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decode_common import gsm_prompts  # noqa: E402
from harness import close_mesh, open_mesh_full  # noqa: E402

from models.demos.blackhole.mimo_v26.reference.prompts import tokenizer  # noqa: E402
from models.demos.blackhole.mimo_v26.tt.generator import Generator  # noqa: E402
from models.demos.blackhole.mimo_v26.tt.kv_pool import KVLayout  # noqa: E402
from models.demos.blackhole.mimo_v26.tt.model import MiMoModel  # noqa: E402

E = os.environ.get
OUT = E("DEC_OUT", "/tmp/kv_det.jsonl")


def rec(case, **kw):
    kw["case"] = case
    s = json.dumps(kw, default=str)
    print("DEC_RESULT " + s, flush=True)
    open(OUT, "a").write(s + "\n")


def snap(model, mesh):
    rows, cols = tuple(mesh.shape)
    out = []
    for lyr in model.layers:
        a = lyr.attn
        k = ttnn.get_device_tensors(a.k_cache)
        v = ttnn.get_device_tensors(a.v_cache)
        out.append(
            (torch.stack([ttnn.to_torch(t) for t in k]), torch.stack([ttnn.to_torch(t) for t in v]))
        )  # [32dev, nblocks,1,64,192]
    return out


def compare(model, lay, a, b, label):
    res = []
    for L, ((ka, va), (kb, vb)) in enumerate(zip(a, b)):
        swa = model.layers[L].swa
        nb = lay.ring_blocks if swa else lay.ga_nb
        U = lay.users
        bad = {}
        for name, x, y in (("k", ka, kb), ("v", va, vb)):
            d = (x != y).any(dim=(3, 4)) if x.dim() == 5 else None  # [dev, nblocks, 1]
            if d is not None and d.any():
                blocks = d.any(dim=(0, 2)).nonzero().flatten().tolist()
                slots = sorted({b // nb for b in blocks})
                bad[name] = dict(n_bad_blocks=len(blocks), slots=slots[:10])
        if bad:
            res.append((L, "swa" if swa else "ga", bad))
    rec(label, n_layers_with_diff=len(res), first=res[:3])
    return res


def main():
    NL, N, STEPS = int(E("DEC_LAYERS", "48")), int(E("DEC_N", "32")), int(E("DEC_STEPS", "12"))
    tok = tokenizer()
    prompts = [it["prompt_ids"] for it in gsm_prompts(N, tok)]
    S = 256
    lay = KVLayout(users=-(-N // 4), max_pos=512, chunk=S)
    mesh = open_mesh_full()
    try:
        model = MiMoModel(mesh, n_layers=NL, layout=lay, log=lambda s: None)
        g = Generator(model, 512, trace=E("DEC_TRACE", "1") == "1", keep_logits=False)
        first, _ = g.prefill(prompts, S)
        s1 = snap(model, mesh)
        g.prefill(prompts, S)
        s2 = snap(model, mesh)
        compare(model, lay, s1, s2, "prefill_vs_prefill")
        # decode twice from fresh prefills (teacher = tokens of a first free run)
        a = g.generate(prompts, STEPS, S=S)
        sa = snap(model, mesh)
        b = g.generate(prompts, STEPS, S=S, teacher=a["tokens"])
        sb = snap(model, mesh)
        compare(model, lay, sa, sb, "after_decode_run1_vs_run2")
        rec("tokens", equal=a["tokens"] == b["tokens"])
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
