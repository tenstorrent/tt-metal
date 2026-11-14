# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Post-norm main hidden dump (fp32 CPU oracle, layer-streaming) on natural prose / code / dialogue text
(for MTP acceptance at depth, spec/alpha_cpu.py). Usage: python -m ...reference.text_dump OUT_DIR [N_TOK=400]
Texts come from files on NFS (docs, source, READMEs) cut at N_TOK tokens; resumable (state.pt)."""
import json
import os
import sys
import time

import torch

from models.demos.blackhole.hy4_preview.reference.oracle import Conv, Hy4Oracle, decoder_layers_multi

REPO = "/mnt/tt-data/ssinghal/tests/tt-metal"
SRC = [
    ("prose", f"{REPO}/docs/source/ttnn/ttnn/tensor.rst"),
    ("prose", "/mnt/tt-data/Hy4-Preview/README.md"),
    ("prose", f"{REPO}/README.md"),
    ("prose", f"{REPO}/CONTRIBUTING.md"),
    ("code", f"/mnt/tt-data/ssinghal/wt/hy4_mtp/models/demos/blackhole/hy4_preview/reference/oracle.py"),
    ("code", f"/mnt/tt-data/ssinghal/wt/hy4_mtp/models/demos/blackhole/hy4_preview/spec/accept.py"),
    ("code", f"{REPO}/ttnn/ttnn/operations/core.py"),
    ("prose", "/mnt/tt-data/ssinghal/hy4-logs/program/MODEL_PROFILE.md"),
    ("prose", "/mnt/tt-data/ssinghal/skills/claude_skills/skills/tt-moe-serving/SKILL.md"),
    ("prose", "/mnt/tt-data/ssinghal/hy4-logs/program/CHARTER.md"),
]
DIALOGUES = [
    "User: Can you explain how a hash map handles collisions?\nAssistant: Sure. A hash map stores key-value pairs in an array of buckets. When two keys hash to the same bucket, "
    "that is a collision. The two common strategies are separate chaining, where each bucket holds a linked list of entries, and open addressing, where the map probes other slots in the array "
    "until it finds an empty one. Chaining is simple and degrades gracefully, while open addressing is more cache friendly but needs a lower load factor.\nUser: Which one does Python use?\nAssistant: Python's dict uses open addressing with a "
    "perturbed probe sequence, and since version 3.6 it also keeps a compact array of entries so that iteration follows insertion order.\nUser: And what happens when it fills up?\nAssistant:",
    "User: I have a list of orders and I want the total revenue per customer in pandas. How do I do it?\nAssistant: You can group by the customer column and sum the revenue column:\n\n```python\n"
    "import pandas as pd\n\ndf = pd.read_csv('orders.csv')\ntotals = df.groupby('customer_id')['revenue'].sum().reset_index()\nprint(totals.sort_values('revenue', ascending=False).head(10))\n```\n\n"
    "This returns one row per customer with the summed revenue, sorted so the biggest customers come first.\nUser: Great, and how do I also get the number of orders?\nAssistant:",
]


def main():
    out = sys.argv[1]
    ntok = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    torch.set_num_threads(int(os.environ.get("HY4_THREADS", "24")))
    os.makedirs(out, exist_ok=True)
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file("/mnt/tt-data/Hy4-Preview/tokenizer.json")
    texts, kinds = [], []
    for kind, f in SRC:
        if os.path.exists(f):
            t = open(f, errors="ignore").read()
            t = t[len(t) // 8 :] if len(t) > 20000 else t  # skip license headers / front matter
            ids = tok.encode(t).ids[:ntok]
            if len(ids) > 100:
                texts.append(ids)
                kinds.append(kind + ":" + os.path.basename(f))
    for i, d in enumerate(DIALOGUES):
        texts.append(tok.encode(d).ids)
        kinds.append(f"chat:{i}")
    ids = [torch.tensor([[120000] + t]) for t in texts]
    o, conv = Hy4Oracle(), Conv()
    state_f = os.path.join(out, "state.pt")
    if os.path.exists(state_f):
        st = torch.load(state_f)
        x4s, start, idx_states = st["x4s"], st["layer"], st["idx"]
    else:
        x4s, start, idx_states = [o.embed_ids(i) for i in ids], 0, [{} for _ in ids]
    json.dump(
        {"kinds": kinds, "n_tokens": [int(i.shape[1]) for i in ids], "ids": [i[0].tolist() for i in ids]},
        open(os.path.join(out, "meta.json"), "w"),
    )
    t0 = time.time()
    for L in range(start, o.cfg.n_layers):
        x4s = decoder_layers_multi(o.ck, L, x4s, [conv] * len(ids), idx_states, {})
        torch.save({"x4s": x4s, "layer": L + 1, "idx": idx_states}, state_f + ".tmp")
        os.replace(state_f + ".tmp", state_f)
        print(f"layer {L} done {time.time()-t0:.0f}s", flush=True)
    for p, x4 in enumerate(x4s):
        h, hn = o.head(x4, conv)
        torch.save({"hc_head": h, "final_normed": hn}, os.path.join(out, f"final_hidden_p{p}.pt"))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
