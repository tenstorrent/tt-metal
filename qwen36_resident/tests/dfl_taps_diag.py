# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostics of ResidentDFlash's prefill path: acceptance of a corpus prompt through verify steps only, before and
after runs of the prefill path with and without the drafter's context fill from the prefill's taps.

usage: python -m qwen36_resident.tests.dfl_taps_diag [--max-pos 8576] [--order verify,prefill,verify]
"""
import argparse
import faulthandler
import sys

import torch

from qwen36_resident import weights as Q
from qwen36_resident.dflash import ResidentDFlash
from qwen36_resident.tests.long_ctx import close_mesh, log, open_mesh

ap = argparse.ArgumentParser()
ap.add_argument("--max-pos", type=int, default=8576)
ap.add_argument("--order", default="prefill,verify")
ap.add_argument("--T", type=int, default=3000)
ap.add_argument("--no-pp", action="store_true")
ap.add_argument("--save", help="the first run's prompt and generation taps and tokens (.pt)")
a = ap.parse_args()
faulthandler.dump_traceback_later(600, repeat=True, file=sys.stderr)
mesh = open_mesh()
try:
    ck = Q.Checkpoint(Q.checkpoint_dir())
    spec = ResidentDFlash(mesh, ck, max_pos=a.max_pos, prefill=not a.no_pp)
    ids = torch.load("/home/ttuser/experiments/qwen36_27b/profiles/resident/256k/corpus_ids.pt")["ids"][: a.T].tolist()
    nofill = getattr(spec, "_fill_context", None)
    for name in a.order.split(","):
        spec.use_prefill = name.startswith("prefill")
        if name == "prefill-nofill":
            spec._fill_context = lambda *a, **k: None
        spec.tap_log = [] if a.save else None
        spec.draft_log = []
        gen, st = spec.generate(ids, 128)
        if a.save:
            loop_taps = [(p, t.bfloat16()) for p, t in spec.tap_log]
            spec.tap_log = []
            ref = spec.greedy(ids, 128)  # the same tokens one per step: row 0's taps
            torch.save(
                dict(
                    gen=gen,
                    ids=ids,
                    taps=loop_taps,
                    drafts=spec.draft_log,
                    ref_gen=ref,
                    ref_taps=[(p, t.bfloat16()) for p, t in spec.tap_log],
                ),
                a.save,
            )
            a.save = None
        spec.tap_log = None
        spec._fill_context = nofill
        log(f"{name}: mean accept {st['mean_accept']:.2f} accepts {st['accepts']}")
finally:
    close_mesh(mesh)
