# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Direct (no vLLM) DFlash2 reference of the serving prompts (prompts.py): lane dfl's ResidentDFlash decoded the way
Qwen36ResidentDFlashForCausalLM serves it (start, then next_block(8, max_len) until max_tokens), built as the serving
class builds it (max_pos = max_len + DFLASH_POS_MARGIN). The served dflash2 profile at temperature 0 must reproduce
these tokens exactly (chat_client parity / compare read token_ids).
With --plain (greedy_ref.py output): the first token where DFlash2 differs from plain greedy (information: the R=8
verify model's arithmetic is not the B=1 decode's), with plain's top-1 - top-2 logit gap there.

usage: python -m qwen36_resident.serve.dflash_ref --out dflash_ref.json [--plain greedy_ref.json] [--max-tokens 128]
"""
import argparse
import faulthandler
import json
import sys
import time

from transformers import AutoTokenizer

from qwen36_resident import weights as Q
from qwen36_resident.generator_vllm import DFLASH_BLOCK, DFLASH_POS_MARGIN
from qwen36_resident.serve.prompts import PROMPTS, chat_ids
from qwen36_resident.tests.e0_bringup import close_mesh, open_mesh


def first_diff(a, b):
    return next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--plain")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=32768)
    a = ap.parse_args()
    faulthandler.dump_traceback_later(900, repeat=True, file=sys.stderr)
    from qwen36_resident.dflash import ResidentDFlash

    path = Q.checkpoint_dir()
    ck = Q.Checkpoint(path)
    tok = AutoTokenizer.from_pretrained(path)
    plain = {r["name"]: r for r in json.load(open(a.plain))["prompts"]} if a.plain else {}
    mesh = open_mesh()
    rows = []
    try:
        t0 = time.time()
        spec = ResidentDFlash(mesh, ck, a.max_len + DFLASH_POS_MARGIN, prefill=True)
        print(f"built in {time.time() - t0:.0f} s", flush=True)
        N = a.max_tokens
        for p in PROMPTS:
            ids = chat_ids(tok, p)
            out = [int(spec.start(ids).argmax())]
            while len(out) < N:
                out += spec.next_block(DFLASH_BLOCK, a.max_len)
            stats = spec.stats()
            spec.stop()
            out = out[:N]
            row = dict(name=p["name"], prompt_ids=ids, token_ids=out, stats=stats)
            msg = f"{p['name']}: {len(out)} tokens, {stats}"
            if p["name"] in plain:
                ref = plain[p["name"]]
                k = row["vs_plain"] = first_diff(out, ref["host_tokens"])
                msg += f", vs plain greedy first diff {k}"
                if k is not None:
                    row["plain_gap"] = ref["host_gaps"][k]
                    msg += f" (plain gap there {row['plain_gap']:.4f})"
            rows.append(row)
            print(msg, flush=True)
    finally:
        close_mesh(mesh)
    json.dump(dict(model=path, max_tokens=a.max_tokens, prompts=rows), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
