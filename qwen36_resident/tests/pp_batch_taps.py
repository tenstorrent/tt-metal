# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PPPrefill.run_batch on the drafter-tapped prefill (DB3 phase 2: plain-geometry prompts gang-prefilled while the
tap writes go to the scratch blocks): 4 prompts batched into plain rows 0..3 vs the same prompts one at a time into
rows 4..7 — every slot's GDN states, conv history and KV pages must be bit-identical; then a spec user started after
the batch must speculate normally (its drafter context from its own prefill's taps).

usage: python -m qwen36_resident.tests.pp_batch_taps
"""
import faulthandler

import torch

import ttnn
from qwen36_resident import weights as Q
from qwen36_resident.adaptive import AdaptiveEngine
from qwen36_resident.tests.e0_bringup import close_mesh, open_mesh

CORPUS = "/home/ttuser/experiments/qwen38/results/native/corpus_ids_40k.pt"


def host(t):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(t.device(), dim=0))


def main():
    faulthandler.enable()
    ids = torch.load(CORPUS)["ids"]
    mesh = open_mesh()
    ok = True
    try:
        max_pos = 2048
        bpu = -(-(max_pos + 72) // 64) + 2
        eng = AdaptiveEngine(mesh, Q.Checkpoint(Q.checkpoint_dir()), max_pos, kv_pool_blocks=20 * bpu + 8)
        pp, plain = eng.pp, eng.plain
        lens = [190, 230, 251, 206]
        prompts = [ids[700 * k + 50 : 700 * k + 50 + n].tolist() for k, n in enumerate(lens)]
        table = lambda s: list(range(s * bpu, (s + 1) * bpu))  # noqa: E731
        for s in range(8):
            eng._set_plain_table(s, table(s))
        C = pp.pick_chunk(max(lens) - 1)
        pp.run_batch([(p[:-1], table(k), k) for k, p in enumerate(prompts)], plain, chunk=C)
        for k, p in enumerate(prompts):
            pp.reset()
            pp.run(torch.as_tensor(p[:-1]), chunk=C)
            pp.handoff(plain, len(p) - 1, block_table=table(4 + k), slot=4 + k)
        for k, p in enumerate(prompts):
            a = plain.snapshot_slot(k, n_tokens=len(p) - 1)
            b = plain.snapshot_slot(4 + k, n_tokens=len(p) - 1)
            same = {x: torch.equal(host(a[x]), host(b[x])) for x in ("state", "hist", "kv")}
            print(f"prompt {k} ({len(p)} tokens): batched vs alone {same}", flush=True)
            ok &= all(same.values())
        # a spec user after the batch: its own drafter context
        P = 300
        out = eng.add([(100, ids[20000 : 20000 + P].tolist(), table(9))], "spec")
        for _ in range(16):
            eng.spec_step()
        us = eng.users[100]
        tps = (len(us.tokens) - P) / 17
        print(f"spec user after the batch: {len(us.tokens) - P} tokens in 17 steps ({tps:.2f} per step)", flush=True)
        ok &= tps > 1.8
        print(f"pp batch taps: {'PASS' if ok else 'FAIL'}", flush=True)
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
