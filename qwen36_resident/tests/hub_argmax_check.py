# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The verify-rows hub's device greedy (ResidentDFlashRows.greedy_rows: the hub's argmax of each row's logits) vs the
host argmax of the same step's logits (ResidentModel.logits), every row of every step of a greedy decode of 8 users.
A mismatch with a clear top-2 gap is a device argmax error (not a near-tie). --share: the build as DB3 phase 2 makes it
(l1_union for a plain build, which is then built with share=).

usage: python -m qwen36_resident.tests.hub_argmax_check [--share] [--prompt-len 256] [--new 64]
"""
import argparse
import faulthandler

import torch

from qwen36_resident import weights as Q
from qwen36_resident.dflash import ResidentDFlashRows
from qwen36_resident.model import ATTN_TAILS, TOP_K, Dims, ResidentModel, State, is_attn
from qwen36_resident.tests.e0_bringup import close_mesh, open_mesh

CORPUS = "/home/ttuser/experiments/qwen38/results/native/corpus_ids_40k.pt"


def main():
    faulthandler.enable()
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--prompt-len", type=int, default=256)
    ap.add_argument("--new", type=int, default=64)
    a = ap.parse_args()
    ids = torch.load(CORPUS)["ids"]
    mesh = open_mesh()
    try:
        ck = Q.Checkpoint(Q.checkpoint_dir())
        cfg = ck.config
        layers, interval = cfg["num_hidden_layers"], cfg["full_attention_interval"]
        d = Dims(mesh.get_num_devices(), mesh.dram_grid_size().x, vocab=cfg["vocab_size"])
        users, R, P = 8, 4, a.prompt_len
        max_pos = P + a.new + 256
        bpu = -(-(max_pos + 72) // 64) + 2
        blocks = 40 * bpu + 8
        union = dict(cand_words=d.n * 2 * TOP_K * 32, attn_tails=ATTN_TAILS) if a.share else None
        eng = ResidentDFlashRows(
            mesh, ck, users=users, max_pos=max_pos, R=R, prefill=True, kv_pool_blocks=blocks, l1_union=union
        )
        eng.m.set_page_table([list(range(u * bpu, (u + 1) * bpu)) for u in range(users) for _ in range(R)])
        if a.share:
            n_attn = sum(is_attn(l, interval) for l in range(layers))
            sts = [State(d, layers - n_attn, n_attn, 0, eng.max_pos, zero=True, kv=False) for _ in range(32)]
            plain = ResidentModel(
                mesh,
                d,
                Q.load(ck, d, layers, interval),
                sts,
                layers,
                interval,
                lm_head=True,
                embed=eng.pp.embed,
                kv_pool_blocks=blocks,
                block_tables=[[0]] * 32,
                rows=True,
                groups=4,
                share=eng.m,
            )
            plain.reset()
            plain.capture_trace()
            eng.pp.capture()
        prompts = [ids[1000 * u + 101 : 1000 * u + 101 + P].tolist() for u in range(users)]
        t = eng._prompt(prompts, draft_context=False)
        p = [P] * users
        bad, total, gaps = [], 0, []
        for step in range(a.new):
            eng._step([[t[u]] * R for u in range(users)], p)
            g = eng.greedy_rows()
            lg = eng.m.logits()[:, :, : d.vocab_chip].reshape(users * R, -1)[:, : cfg["vocab_size"]].float()
            for r in range(users * R):
                total += 1
                h = int(lg[r].argmax())
                if g[r // R][r % R] != h:
                    top = lg[r].topk(2)
                    gap = float(top.values[0] - top.values[1])
                    dev_v = float(lg[r][g[r // R][r % R]])
                    bad.append((step, r, g[r // R][r % R], h, round(gap, 4), round(float(top.values[0]) - dev_v, 4)))
            t = [g[u][0] for u in range(users)]
            eng._commit([0] * users)
            p = [x + 1 for x in p]
        print(
            f"hub argmax ({'shared build' if a.share else 'standalone'}): {len(bad)} of {total} row-steps differ "
            f"from the host argmax",
            flush=True,
        )
        for b in bad[:20]:
            print(
                f"  step {b[0]} row {b[1]}: device {b[2]} host {b[3]} (top-2 gap {b[4]}, host max - device token's "
                f"logit {b[5]})",
                flush=True,
            )
        by_chip = {}
        for b in bad:
            by_chip[(b[2] // d.vocab_chip, b[3] // d.vocab_chip)] = (
                by_chip.get((b[2] // d.vocab_chip, b[3] // d.vocab_chip), 0) + 1
            )
        print(f"  (device token's chip, host token's chip): {by_chip}", flush=True)
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
