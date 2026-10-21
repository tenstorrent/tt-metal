# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic: logits of one position computed through different prefill buckets (short 2D-matmul path at <= 1280
rows vs the long minimal_matmul / BFP8-MLP path) against the HF bf16 model.

  python models/autoports/google_gemma_4_12b_it/tests/diag_prefill_paths.py --position 999 --lengths 1000,1037,1300
"""

import argparse
import json

import torch

from models.autoports.google_gemma_4_12b_it.tests.hf_reference import pcc
from models.autoports.google_gemma_4_12b_it.tests.test_full_model import PROMPTS, row_logits
from models.autoports.google_gemma_4_12b_it.tt.generator import build_generator
from models.autoports.google_gemma_4_12b_it.tt.multichip_decoder import close_qb2_mesh, open_qb2_mesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--position", type=int, default=999)
    ap.add_argument("--lengths", default="1000,1037,1300")
    ap.add_argument("--text", default="prompts", choices=["prompts", "llms"])
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.set_num_threads(16)
    lengths = [int(v) for v in a.lengths.split(",")]
    mesh = open_qb2_mesh(trace_region_size=0)
    res = {"position": a.position}
    try:
        gen = build_generator(None, mesh, prepare=False)
        tok = gen.tokenizer
        text = " ".join(PROMPTS) * 200 if a.text == "prompts" else open("tech_reports/LLMs/llms.md").read()
        base = tok.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True, return_dict=False
        )
        ids = (base * (2 + max(lengths) // len(base)))[: max(lengths)]
        tt = {n: row_logits(gen, ids[:n], a.position) for n in lengths}
        # stack output (pre final norm) at the position, through the first length's bucket
        import ttnn

        m = gen.model
        n0 = lengths[0]
        row = gen._alloc_row(n0)
        plan = m.chunk_plan(0, n0)
        table = m.device_ints(m.prefill_page_table(row, n0, plan[-1][0] + plan[-1][2]))
        tt_hidden = None
        for c0, real, bucket in plan:
            chunk = torch.zeros(bucket, dtype=torch.int32)
            chunk[:real] = torch.tensor(ids[c0 : c0 + real])
            h = m.prefill_chunk_hidden(chunk, table, c0)
            if c0 <= a.position < c0 + real:
                full = ttnn.to_torch(h, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=3)).float()
                tt_hidden = full.reshape(-1, m.hidden_size)[a.position - c0]
            ttnn.deallocate(h)
    finally:
        close_qb2_mesh(mesh)
    from models.autoports.google_gemma_4_12b_it.tests.full_model_ref import hf_causal_lm

    hf = hf_causal_lm(dtype=torch.bfloat16)
    captured = {}
    hook = hf.model.norm.register_forward_hook(lambda mod, inp, out: captured.update(h=inp[0][0, -1].float()))
    with torch.no_grad():
        want = hf(torch.tensor([ids[: a.position + 1]])).logits[0, -1].float()
    hook.remove()
    # host terminal (fp32 final norm + fp32 tied LM head + soft-cap) on the TT stack output and on the HF one
    w = hf.model.embed_tokens.weight.float()
    g = hf.model.norm.weight.float()

    def host_logits(h):
        hn = h * torch.rsqrt(h.pow(2).mean() + 1e-6) * g
        return 30.0 * torch.tanh((w @ hn) / 30.0)

    res["stack_hidden_pcc_tt_vs_hf"] = round(pcc(captured["h"], tt_hidden), 5)
    res["host_terminal_on_tt_hidden_vs_hf"] = round(pcc(want, host_logits(tt_hidden)), 5)
    res["host_terminal_on_hf_hidden_vs_hf"] = round(pcc(want, host_logits(captured["h"])), 5)
    res["tt_logits_vs_host_terminal_on_tt_hidden"] = round(pcc(host_logits(tt_hidden), tt[lengths[0]]), 5)
    top = set(want.topk(5).indices.tolist())
    for n, got in tt.items():
        res[n] = {
            "pcc_vs_hf_bf16": round(pcc(want, got), 5),
            "argmax_tt_hf": [int(got.argmax()), int(want.argmax())],
            "top5_overlap": len(top & set(got.topk(5).indices.tolist())),
        }
    names = list(tt)
    res["pcc_between"] = {f"{x}-{y}": round(pcc(tt[x], tt[y]), 5) for x in names for y in names if x < y}
    print("DIAG", json.dumps(res), flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
