# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""HF-referenced control of the user-split full-layer decode traces at the full-model level.

The long chat reference (8293-token prompt, 64 HF tokens with their top-100) is prefilled into every row of a decode
bucket (B = 8 / 16: past the split break-even), then teacher-forced through the traced decode with the reference
tokens, once with the user-split trace and once with the unsplit trace (``decode_forward(user_split=...)``, same
cache state per step). Per row: top-1 / top-5 / top-100 against the HF reference and the logits PCC between the two
placements, with the chip that attends the row on the split path (chip d: rows [d B/4, (d+1) B/4)). The batch-1
generator's numbers on the same reference are in readiness_long8k_final.json.

  python models/autoports/google_gemma_4_12b_it/tests/split_tf_control.py --out doc/full_model/logs/split_tf_control.json
"""

import argparse
import json

import torch
from readiness_check.schema import load_reference

from models.autoports.google_gemma_4_12b_it.tests.hf_reference import pcc
from models.autoports.google_gemma_4_12b_it.tt.generator import build_generator
from models.autoports.google_gemma_4_12b_it.tt.multichip_decoder import close_qb2_mesh, open_qb2_mesh


def control(gen, reference="bringup/references/gemma4_12b_it_long8k_chat.refpt", batches=(8, 16)):
    """Run the control on a live generator; returns the per-batch, per-row results."""
    ref = load_reference(reference).entries[0]
    prompt = ref.prompt_tokens[0].tolist()
    forced = ref.generated_tokens[0].tolist()
    topk = ref.topk_tokens
    steps = len(forced)
    res = {"reference": str(reference), "prompt_tokens": len(prompt), "steps": steps}
    for B in batches:
        assert gen.model.use_user_split(B, len(prompt))
        rows = [gen._alloc_row(len(prompt) + steps + 8) for _ in range(B)]
        table = torch.stack(rows)
        gen.prefill_forward(torch.tensor([prompt] * B), page_table=table, prompt_lens=[len(prompt)] * B)
        stats = {m: torch.zeros(B, 3) for m in ("split", "unsplit")}
        pccs = torch.ones(B)
        for i in range(steps):
            if i == 0:
                continue  # step 0 is the prefill's prediction (identical for both placements)
            tok = torch.tensor([forced[i - 1]] * B)
            pos = torch.tensor([len(prompt) + i - 1] * B)
            out = {}
            for mode in ("split", "unsplit"):
                out[mode] = gen.decode_forward(
                    tok, pos, page_table=table, return_logits=True, user_split=(mode == "split")
                )
                pred = out[mode].argmax(-1)
                ref_k = topk[i]
                for b in range(B):
                    p = int(pred[b])
                    stats[mode][b] += torch.tensor(
                        [p == int(ref_k[0]), p in ref_k[:5].tolist(), p in ref_k.tolist()], dtype=torch.float32
                    )
            for b in range(B):
                pccs[b] = min(pccs[b], pcc(out["split"][b], out["unsplit"][b]))
        n = steps - 1
        res[f"b{B}"] = {
            "positions": [len(prompt), len(prompt) + n - 1],
            "per_row": [
                {
                    "row": b,
                    "split_chip": b // (B // 4),
                    "split_top1_top5_top100": [round(float(v) / n, 4) for v in stats["split"][b]],
                    "unsplit_top1_top5_top100": [round(float(v) / n, 4) for v in stats["unsplit"][b]],
                    "min_step_logits_pcc_split_vs_unsplit": round(float(pccs[b]), 5),
                }
                for b in range(B)
            ],
        }
        print(f"B={B}", json.dumps(res[f"b{B}"]), flush=True)
        for r in rows:
            gen._release(r)
        gen.reset()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="bringup/references/gemma4_12b_it_long8k_chat.refpt")
    ap.add_argument("--batches", default="8,16")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.set_num_threads(16)
    mesh = open_qb2_mesh(trace_region_size=0)
    try:
        gen = build_generator(None, mesh)
        res = control(gen, a.reference, [int(v) for v in a.batches.split(",")])
        gen.teardown()
    finally:
        close_qb2_mesh(mesh)
    print("SPLITTF", json.dumps(res), flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
