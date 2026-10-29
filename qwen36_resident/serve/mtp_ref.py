# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Direct (no vLLM) MTP references of the serving prompts (prompts.py), next to greedy_ref.py's plain greedy.

Per prompt, max_tokens each:
verify: plain greedy on the MTP main model's verify step (SpecDecoder.greedy after the prefill): the tokens
    speculative decode must reproduce.
spec:   speculative decode as Qwen36ResidentMTPForCausalLM serves it (start, then next_block(block, depth)).
With --plain (greedy_ref.py output): the first token where verify differs from plain greedy, with plain's
top-1 - top-2 logit gap there.

usage: python -m qwen36_resident.serve.mtp_ref --out mtp_ref.json [--plain greedy_ref.json] [--max-tokens 128]
"""
import argparse
import faulthandler
import json
import sys
import time

from transformers import AutoTokenizer

from qwen36_resident import weights as Q
from qwen36_resident.mtp import SpecDecoder
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
    ap.add_argument("--block", type=int, default=4)
    ap.add_argument("--depth", type=int, default=4)
    a = ap.parse_args()
    faulthandler.dump_traceback_later(900, repeat=True, file=sys.stderr)
    path = Q.checkpoint_dir()
    ck = Q.Checkpoint(path)
    tok = AutoTokenizer.from_pretrained(path)
    plain = {r["name"]: r for r in json.load(open(a.plain))["prompts"]} if a.plain else {}
    mesh = open_mesh()
    rows = []
    try:
        t0 = time.time()
        spec = SpecDecoder(mesh, ck, ck.config["num_hidden_layers"], a.max_len, prefill=True)
        print(f"built in {time.time() - t0:.0f} s", flush=True)
        N = a.max_tokens
        for p in PROMPTS:
            ids = chat_ids(tok, p)
            verify = spec.greedy(ids, N, prefill=True)[len(ids) :]
            out = [int(spec.start(ids).argmax())]
            while len(out) < N:
                out += spec.next_block(a.block, a.depth, a.max_len)
            spec.stop()
            out = out[:N]
            row = dict(name=p["name"], prompt_ids=ids, verify_tokens=verify, spec_tokens=out)
            row["spec_vs_verify"] = first_diff(out, verify)
            msg = f"{p['name']}: spec vs verify first diff {row['spec_vs_verify']}"
            if p["name"] in plain:
                ref = plain[p["name"]]
                k = row["verify_vs_plain"] = first_diff(verify, ref["host_tokens"])
                row["spec_vs_plain"] = first_diff(out, ref["host_tokens"])
                msg += f", verify vs plain {k}, spec vs plain {row['spec_vs_plain']}"
                if k is not None:
                    row["plain_gap"] = ref["host_gaps"][k]
                    msg += f" (plain gap there {row['plain_gap']:.4f})"
            rows.append(row)
            print(msg, flush=True)
        spec.stop()
    finally:
        close_mesh(mesh)
    json.dump(dict(model=path, max_tokens=a.max_tokens, prompts=rows), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
