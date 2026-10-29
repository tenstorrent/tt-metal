# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Accuracy of the multi-user rows build on the Qwen3.8 checkpoint (1 x 4 mesh), for comparing build settings
(e.g. RESIDENT_ATTN_WORKERS / RESIDENT_HEAD_GROUPS) in separate runs.

Built as the served rows profile (PP prefill at max_len, rows model of 8 slots on a paged pool, warm-ups, traces).
Every user is prefilled into its own slot (model.row_order()), then decoded teacher-forced with host token states,
all of them in the same steps:
  hf:    the Hugging Face reference (hf_teacher_ref.py: sequences of `prefix` prompt tokens, `scored` positions):
         top-1 / top-5 against HF's argmax, and HF's / the device's margin at each miss;
  long:  prompts of the long_ctx corpus (P tokens), the logits at P - 1 .. P - 1 + steps, saved as <tag>_P<P>.pt
         (the long_ctx teacher format) and compared with <ref-dir>/ttnn_P<P>.pt when present.
Compare two runs' long logits with `python -m qwen36_resident.tests.long_ctx compare a.pt b.pt`.

usage: python -m qwen36_resident.tests.rows_accuracy --corpus corpus_ids.pt --ref hf_ref_qwen38.pt
           [--long 31872,131712] [--steps 128] [--ref-dir .../256k] --out-dir DIR --tag TAG
"""

import argparse
import faulthandler
import json
import os
import sys
import time

import torch

import ttnn
from qwen36_resident import weights as Q
from qwen36_resident.model import Dims, ResidentModel, State, is_attn
from qwen36_resident.prefill.pp_prefill import PPPrefill
from qwen36_resident.tests.e0_bringup import PREFILL_CHUNKS, close_mesh, open_mesh
from qwen36_resident.tests.long_ctx import compare

B = 8
BLOCK = 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--long", default="31872,131712")
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=262144)
    ap.add_argument("--ref-dir", default="")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    faulthandler.dump_traceback_later(300, repeat=True, file=sys.stderr)
    corpus = torch.load(a.corpus)["ids"]
    ref = torch.load(a.ref)
    longs = [int(p) for p in a.long.split(",") if p]
    t_start = time.time()
    out = dict(args=vars(a), env={k: v for k, v in os.environ.items() if k.startswith("RESIDENT_")})

    def phase(name):
        print(f"[{time.time() - t_start:7.1f} s] {name}", flush=True)

    mesh = open_mesh()
    try:
        ck = Q.Checkpoint(Q.checkpoint_dir())
        cfg = ck.config
        layers, interval, vocab = cfg["num_hidden_layers"], cfg["full_attention_interval"], cfg["vocab_size"]
        d = Dims(mesh.get_num_devices(), mesh.dram_grid_size().x, vocab=vocab)
        emb = Q.embedding(ck)
        C = max(PREFILL_CHUNKS)
        max_len = -(-a.max_len // C) * C
        n_attn = sum(is_attn(l, interval) for l in range(layers))
        empty = lambda: State(d, layers - n_attn, n_attn, 0, max_pos=max_len + 64, zero=True, kv=False)
        pp = PPPrefill(mesh, ck, layers, interval, chunk=list(PREFILL_CHUNKS), max_len=max_len)
        phase(f"prefill model (max_len {max_len})")

        # users: the HF sequences, then the long prompts; each a slot and its own blocks (shuffled pool order)
        P_hf, N_hf = ref["prefix"], ref["scored"]
        users = [dict(kind="hf", seq=i, ids=s["ids"], P=P_hf, n=N_hf) for i, s in enumerate(ref["seqs"])]
        users += [dict(kind="long", ids=corpus[: P + a.steps + 1], P=P, n=a.steps + 1) for P in longs]
        assert len(users) <= B
        need = [-(-(u["P"] + u["n"] + 64) // BLOCK) for u in users]
        nb = sum(need) + 64
        perm = torch.randperm(nb - 1, generator=torch.Generator().manual_seed(5)).add(1).tolist()  # block 0 kept free
        m = ResidentModel(
            mesh,
            d,
            Q.load(ck, d, layers, interval),
            [empty() for _ in range(B)],
            layers,
            interval,
            lm_head=True,
            embed=pp.embed,
            kv_pool_blocks=nb,
            block_tables=[[0]] * B,
            rows=True,
        )
        phase(f"rows decode model (G={m.head_groups}, pool {nb} blocks)")
        out["head_groups"] = m.head_groups
        pp.handoff(m, 1, block_table=[0], slot=0)
        m.clear_slot(0)
        m.capture_trace()
        pp.capture()
        phase("traces")

        order = m.row_order()
        tables = [[0] for _ in range(B)]
        for i, u in enumerate(users):
            u["slot"] = order[i]
            tables[u["slot"]] = perm[sum(need[:i]) : sum(need[: i + 1])]
        m.set_page_table(tables)
        for u in users:  # prefill all but the last prompt token into the slot
            pp.reset()
            pp.run(u["ids"][: u["P"] - 1])
            pp.handoff(m, u["P"] - 1, block_table=tables[u["slot"]], slot=u["slot"])
            phase(f"{u['kind']} P={u['P']}: prefilled into slot {u['slot']}")

        # teacher-forced: step k feeds every user its token at P - 1 + k (users with fewer steps go inactive)
        logits = {u["slot"]: [] for u in users}
        for k in range(max(u["n"] for u in users)):
            x0, pos, active = torch.zeros(B, 5120), [0] * B, [False] * B
            for u in users:
                if k < u["n"]:
                    p = u["P"] - 1 + k
                    x0[u["slot"]], pos[u["slot"]], active[u["slot"]] = emb[int(u["ids"][p])].float(), p, True
            # (a user whose steps are done goes inactive: its slot is never read again, as an empty one)
            m.step(m.token_state(x0, pos, active=active))
            ttnn.synchronize_device(mesh)
            lg = m.logits()[:, :, : d.vocab_chip].reshape(B, -1)[:, :vocab].float()
            for u in users:
                if k < u["n"]:
                    logits[u["slot"]].append(lg[u["slot"]].clone())
        phase("teacher-forced steps")

        # hf: top-1 / top-5 against HF's argmax
        hf = dict(top1=0, top5=0, scored=0, mismatch=[])
        for u in (u for u in users if u["kind"] == "hf"):
            s = ref["seqs"][u["seq"]]
            for k, lg in enumerate(logits[u["slot"]]):
                p = u["P"] - 1 + k
                want, dev1 = int(s["top_i"][p][0]), int(lg.argmax())
                hf["top1"] += dev1 == want
                hf["top5"] += want in lg.topk(5).indices.tolist()
                hf["scored"] += 1
                if dev1 != want:
                    hit = (s["top_i"][p] == dev1).nonzero()
                    hf_gap = float(s["top_v"][p, 0] - s["top_v"][p, hit[0, 0]]) if len(hit) else float("inf")
                    hf["mismatch"].append(
                        dict(seq=u["seq"], pos=p, dev=dev1, hf=want, hf_gap=hf_gap, dev_gap=float(lg[dev1] - lg[want]))
                    )
        out["hf"] = hf
        print(
            f"HF teacher-forced: top-1 {hf['top1']}/{hf['scored']} = {hf['top1'] / hf['scored']:.4f}, top-5 "
            f"{hf['top5'] / hf['scored']:.4f}; misses (HF gap, device gap): "
            f"{[(round(x['hf_gap'], 3), round(x['dev_gap'], 3)) for x in hf['mismatch']]}",
            flush=True,
        )

        # long: save in the long_ctx teacher format, compare with the ttnn reference
        out["long"] = []
        for u in (u for u in users if u["kind"] == "long"):
            P = u["P"]
            lg = torch.stack(logits[u["slot"]])
            nxt = corpus[P : P + a.steps + 1]
            torch.save(dict(P=P, steps=a.steps, logits=lg), f"{a.out_dir}/{a.tag}_P{P}.pt")
            res = dict(P=P, slot=u["slot"], teacher_acc=float((lg.argmax(-1) == nxt).double().mean()))
            path = f"{a.ref_dir}/ttnn_P{P}.pt"
            if a.ref_dir and os.path.exists(path):
                res["vs_ttnn"] = {
                    k: v for k, v in compare(torch.load(path)["logits"], lg, nxt).items() if k != "gaps_ref"
                }
            out["long"].append(res)
            print(f"long P={P}: {json.dumps(res)}", flush=True)
        m.release_trace()
    finally:
        close_mesh(mesh)
        json.dump(out, open(f"{a.out_dir}/{a.tag}.json", "w"), indent=1)
    phase("done")


if __name__ == "__main__":
    main()
