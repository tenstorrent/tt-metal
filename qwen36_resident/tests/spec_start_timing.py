# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Where a batch-spec start's time goes: SpecRowsSession.start_many's phases per prompt (PP reset, PP run, handoff,
drafter fill) and the shared first verify step, each synchronized, for short prompts (the served c = 8 TTFT case).

usage: python -m qwen36_resident.tests.spec_start_timing [--users 8] [--prompt-len 64] [--reps 3]
"""
import argparse
import faulthandler
import time

import torch

import ttnn
from qwen36_resident import weights as Q
from qwen36_resident.dflash import ResidentDFlashRows, SpecRowsSession
from qwen36_resident.tests.e0_bringup import close_mesh, open_mesh

CORPUS = "/home/ttuser/experiments/qwen38/results/native/corpus_ids_40k.pt"


def main():
    faulthandler.enable()
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=8)
    ap.add_argument("--prompt-len", type=int, default=64)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    ids = torch.load(CORPUS)["ids"]
    mesh = open_mesh()
    try:
        e = ResidentDFlashRows(mesh, Q.Checkpoint(Q.checkpoint_dir()), users=a.users, max_pos=4096, R=4, prefill=True)
        sess = SpecRowsSession(e)
        sync = lambda: ttnn.synchronize_device(mesh)  # noqa: E731
        for rep in range(a.reps):
            ph = {"reset": 0.0, "run": 0.0, "handoff": 0.0, "fill": 0.0}
            t_all = time.perf_counter()
            for u in range(a.users):
                p = ids[1000 * u + 101 : 1000 * u + 101 + a.prompt_len].tolist()
                P = len(p)
                for name, fn in (
                    ("reset", lambda: e.pp.reset()),
                    ("run", lambda: e.pp.run(torch.as_tensor(p[: P - 1]))),
                    ("handoff", lambda: e.pp.handoff(e.m, P - 1, slot=u * e.R, block_table=e.m.block_tables[u * e.R])),
                    ("fill", lambda: e._fill_context(P - 1, phys=u)),
                ):
                    sync()
                    t0 = time.perf_counter()
                    fn()
                    sync()
                    ph[name] += time.perf_counter() - t0
            # the shared first verify step (as start_many: every user's last prompt token)
            t0 = time.perf_counter()
            sess.start_many([(u, [int(ids[1000 * u + 101 + a.prompt_len - 1])], None) for u in range(a.users)])
            sync()
            t_step = time.perf_counter() - t0
            total = time.perf_counter() - t_all
            print(
                f"rep {rep}: {a.users} starts of {a.prompt_len} tokens in {total * 1000:.0f} ms; per prompt: "
                + ", ".join(f"{k} {v / a.users * 1000:.1f} ms" for k, v in ph.items())
                + f"; shared verify step {t_step * 1000:.1f} ms",
                flush=True,
            )
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
