# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic: chained-stack accuracy layer by layer. One prompt runs through the TT prefill stack (the generator's
own chunk path) and through the HF bf16 model; every layer's output is compared (whole tensor and worst row PCC, and
at the last position). Shows whether the stack error grows gradually (precision policy) or jumps at a layer.

  python models/autoports/google_gemma_4_12b_it/tests/diag_layers.py --length 1000 --out doc/full_model/logs/diag_layers.json
"""

import argparse
import json

import torch

import ttnn
from models.autoports.google_gemma_4_12b_it.tests.harness import row_pcc
from models.autoports.google_gemma_4_12b_it.tests.hf_reference import pcc
from models.autoports.google_gemma_4_12b_it.tests.test_full_model import PROMPTS
from models.autoports.google_gemma_4_12b_it.tt.generator import build_generator
from models.autoports.google_gemma_4_12b_it.tt.multichip_decoder import close_qb2_mesh, open_qb2_mesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=1000)
    ap.add_argument("--text", default="prompts", choices=["prompts", "llms"])
    ap.add_argument("--policy", default="{}", help="JSON policy override for every layer (precision controls)")
    ap.add_argument("--lm_head", default="{}")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.set_num_threads(16)
    n = a.length
    mesh = open_qb2_mesh(trace_region_size=0)
    tt_layers = []
    try:
        kw = {"prepare": False, "num_blocks": 64}
        names = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}

        def dtypes(v):
            if isinstance(v, dict):
                return {k: dtypes(x) for k, x in v.items()}
            return names.get(v, v) if isinstance(v, str) else v

        if json.loads(a.policy):
            kw["policy"] = dtypes(json.loads(a.policy))
        gen = build_generator(None, mesh, **kw)
        m = gen.model
        tok = gen.tokenizer
        text = " ".join(PROMPTS) * 200 if a.text == "prompts" else open("tech_reports/LLMs/llms.md").read()
        base = tok.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True, return_dict=False
        )
        ids = base[:n]
        assert n <= m.prefill_chunk
        row = gen._alloc_row(n)
        plan = m.chunk_plan(0, n)
        (c0, real, bucket) = plan[0]
        table = m.device_ints(m.prefill_page_table(row, n, bucket))
        chunk = torch.zeros(bucket, dtype=torch.int32)
        chunk[:real] = torch.tensor(ids)
        tokd = m.device_ints(chunk.reshape(1, bucket), dtype=ttnn.uint32)
        x = ttnn.embedding(tokd, m.embedding, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        x = ttnn.reshape(x, [1, 1, bucket, m.hidden_size // 4])

        def host(t):
            return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=3)).float().reshape(bucket, -1)[:n]

        emb = host(x)
        for layer in m.layers:
            y = layer.prefill_forward(x, page_table=table, start_pos=0, user_id=0)
            ttnn.deallocate(x)
            x = y
            tt_layers.append(host(x))
    finally:
        close_qb2_mesh(mesh)
    from models.autoports.google_gemma_4_12b_it.tests.full_model_ref import hf_causal_lm

    def hf_layers(dtype):
        hf = hf_causal_lm(dtype=dtype)
        outs = []
        hooks = [
            l.register_forward_hook(
                lambda mod, inp, out: outs.append((out[0][0] if isinstance(out, tuple) else out[0]).float())
            )
            for l in hf.model.layers
        ]
        with torch.no_grad():
            hf(torch.tensor([ids]))
        for h in hooks:
            h.remove()
        del hf
        return outs

    ref32 = hf_layers(torch.float32)
    ref16 = hf_layers(torch.bfloat16)
    res = {"length": n, "reference": "HF float32 (and HF bf16 as a control)", "layers": []}
    for i, (want, got, b16) in enumerate(zip(ref32, tt_layers, ref16)):
        r = {
            "layer": i,
            "tt_pcc": round(pcc(want, got), 5),
            "tt_worst_row": round(float(row_pcc(want, got).min()), 5),
            "tt_last_row": round(pcc(want[-1], got[-1]), 5),
            "bf16_pcc": round(pcc(want, b16), 5),
            "bf16_worst_row": round(float(row_pcc(want, b16).min()), 5),
            "bf16_last_row": round(pcc(want[-1], b16[-1]), 5),
        }
        rp = row_pcc(want, got)
        w = int(rp.argmin())
        r["tt_worst_row_index"] = w
        r["worst_row_norm_hf_tt"] = [round(float(want[w].norm()), 1), round(float(got[w].norm()), 1)]
        r["median_row_norm_hf"] = round(float(want.norm(dim=-1).median()), 1)
        r["rows_below_0.99"] = int((rp < 0.99).sum())
        res["layers"].append(r)
        print(r, flush=True)
    print("DIAG", json.dumps(res), flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
