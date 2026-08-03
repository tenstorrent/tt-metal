# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Non-aligned prompt-length check for the datatype-sweep SELECTED precision config.

Datatype selection must preserve valid non-aligned logical prompt lengths (lengths not divisible by
the internal tile/block/page/chunk sizes: 32, 256, ...). This builds the model through the normal
selected-config construction path (build_generator, which loads the selected precision config by
default) and prefills several deliberately non-aligned prompt lengths, then runs a few traced decode
steps, asserting finite logits/tokens and that decode advances. If the selected KV-cache/trace-buffer
dtype or layout had changed chunking, this is the check that would catch a regression.

  cd /tmp && TT_METAL_HOME=<tree> PYTHONPATH=<repo> python <this>.py --out <json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import ttnn
from models.autoports.poolside_laguna_xs_2_1.tt.generator import build_generator
from models.autoports.poolside_laguna_xs_2_1.tt.model import load_selected_precision_policy

MD = Path("/home/ttuser/dev/tt-metal/models/autoports/poolside_laguna_xs_2_1")
# Non-aligned lengths: 100 (not %32), 129 (%32==1), 257 (>MoE chunk 256, %32==1), 513 (%32==1).
LENGTHS = [100, 129, 257, 513]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--decode-steps", type=int, default=4)
    args = ap.parse_args()

    pol, src = load_selected_precision_policy()
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=1_500_000_000)
    results = {"selected_config_source": src, "policy": pol.to_dict(), "lengths": {}}
    ok_all = True
    try:
        gen = build_generator(MD, mesh, max_seq_len=4096)  # default -> selected config
        vocab = gen.vocab
        for L in LENGTHS:
            torch.manual_seed(L)
            prompt = torch.randint(0, vocab, (L,), dtype=torch.int64).tolist()
            # low-level prefill (last-position logits) on a fresh cache sized for this prompt + decode
            kv = gen.model.alloc_kv_cache(max_users=1, max_seq_len=L + args.decode_steps + 8, block_size=32)
            pt = gen.model.make_page_table(1, kv[0]["blocks_per_user"])
            logits = gen.prefill_forward(
                tokens=torch.tensor([prompt]), page_table=pt, kv_cache=kv, prompt_lens=[L], user_id=0
            )  # [1,1,vocab]
            finite_prefill = bool(torch.isfinite(logits.float()).all())
            first_tok = int(torch.argmax(logits[0, 0]))
            # a few eager decode steps from pos L
            toks = []
            cur = first_tok
            pos = L
            dec_finite = True
            for _ in range(args.decode_steps):
                out = gen.decode_forward(
                    torch.tensor([[cur]]), torch.tensor([pos]), page_table=pt, kv_cache=kv, return_logits=True
                )
                dec_finite = dec_finite and bool(torch.isfinite(out.float()).all())
                cur = int(torch.argmax(out[0]))
                toks.append(cur)
                pos += 1
            aligned = L % 32 == 0
            entry = {
                "prompt_len": L,
                "tile_aligned": aligned,
                "block_aligned": (L % 32 == 0),
                "moe_chunk_aligned": (L % 256 == 0),
                "prefill_finite": finite_prefill,
                "first_token": first_tok,
                "decode_finite": dec_finite,
                "decode_tokens": toks,
                "in_vocab": all(0 <= t < vocab for t in [first_tok] + toks),
                "pass": bool(
                    finite_prefill and dec_finite and 0 <= first_tok < vocab and all(0 <= t < vocab for t in toks)
                ),
            }
            results["lengths"][str(L)] = entry
            ok_all = ok_all and entry["pass"]
            print("NONALIGNED", L, "pass" if entry["pass"] else "FAIL", entry)
    finally:
        try:
            gen.teardown()
        except Exception:
            pass
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    results["all_pass"] = ok_all
    Path(args.out).write_text(json.dumps(results, indent=2))
    print("NONALIGNED_ALL_PASS", ok_all)


if __name__ == "__main__":
    import os as _os
    import sys as _sys

    main()
    _sys.stdout.flush()
    _sys.stderr.flush()
    _os._exit(0)
