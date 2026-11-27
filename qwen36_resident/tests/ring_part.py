# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Strata S4-1b on the real spec build: a speculating user's drafter context ring read as it stands
(ResidentDFlashRows.ring_part -> DFlash2DrafterTP.read_ring) and written into another user's slot through the P2d fill
(fill_from_taps with a ring part -> write_ring): the second slot's ring byte-identical to the first's, and its drafts
from the same anchor and position identical. Also the read / write times.
usage: python -m qwen36_resident.tests.ring_part [--prompt-len 3000] [--steps 24]
"""
import argparse
import sys
import time

import torch

import ttnn
from qwen36_resident import weights as Q
from qwen36_resident.dflash import ResidentDFlashRows, SpecRowsSession
from qwen36_resident.tests.drafter_refill import CORPUS
from qwen36_resident.tests.e0_bringup import close_mesh, open_mesh


def ring_torch(eng, phys):
    comp = ttnn.ConcatMeshToTensor(eng.mesh, dim=1)
    return [torch.stack([ttnn.to_torch(h, mesh_composer=comp) for h in pair]) for pair in eng.drafter.read_ring(phys)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-len", type=int, default=3000)
    ap.add_argument("--steps", type=int, default=24)
    a = ap.parse_args()
    ids = torch.load(CORPUS)["ids"][1000 : 1000 + a.prompt_len].tolist()
    mesh = open_mesh()
    ok = False
    try:
        eng = ResidentDFlashRows(
            mesh, Q.Checkpoint(Q.checkpoint_dir()), users=4, max_pos=a.prompt_len + 512, R=4, prefill=True
        )
        s, U = SpecRowsSession(eng), 4
        lg = s.start_many([(0, ids, None)])[0]
        anchor, pos, acc = int(lg.argmax()), len(ids), 1
        act = [True] + [False] * (U - 1)
        for _ in range(a.steps):
            dr = s.propose([anchor] + [0] * 3, [pos] + [0] * 3, [acc] + [1] * 3, act)
            row = [anchor] + list(dr[0])
            g = s.verify([row] + [[0] * 4] * 3, [pos] + [0] * 3, act)
            k = 0
            while k < 3 and dr[0][k] == g[0][k]:
                k += 1
            new = list(dr[0][:k]) + [g[0][k]]
            pos, anchor, acc = pos + len(new), new[-1], len(new)
        d0 = s.propose([anchor] + [0] * 3, [pos] + [0] * 3, [acc] + [1] * 3, act)  # (the context extended to pos)
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        part = eng.ring_part(0, pos // 64 * 64)
        t_read = time.perf_counter() - t0
        t0 = time.perf_counter()
        eng.fill_from_taps(1, dict(parts=[part], n=part["hb"], tail=None), part["hb"])
        ttnn.synchronize_device(mesh)
        t_write = time.perf_counter() - t0
        r0, r1 = ring_torch(eng, 0), ring_torch(eng, 1)
        same_ring = all(torch.equal(x, y) for x, y in zip(r0, r1))
        d = eng.drafter.draft([anchor, anchor, 0, 0], [pos, pos, 0, 0])
        same_draft = list(d[0]) == list(d[1])  # (d0: the propose's, for reference)
        nbytes = sum(2 * x[0].numel() * x[0].element_size() for x in r0)
        print(
            f"ring part at {part['hb']} (window from {part['s0']}): read {1e3 * t_read:.0f} ms, write {1e3 * t_write:.0f} ms, "
            f"{nbytes / 1e6:.1f} MB; ring byte-identical {same_ring}; drafts {list(d[0])} / {list(d[1])} "
            f"(the propose's {list(d0[0])}) equal {same_draft}",
            flush=True,
        )
        ok = same_ring and same_draft
        print(f"ring part: {'PASS' if ok else 'FAIL'}", flush=True)
    finally:
        close_mesh(mesh)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
