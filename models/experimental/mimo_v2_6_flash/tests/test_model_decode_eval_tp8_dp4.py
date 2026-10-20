# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Teacher-forced decode accuracy over many positions: test_model_tp8_dp4.py's decode rows, at scale.

The golden is reference/make_golden_large.py's (MIMO_GOLDEN): C4 sequences and the fp32 reference's last hidden
states, whose logits the reference's final norm and LM head give on the host. Every replica takes 4 sequences at a
time: the first ``PREFILL`` tokens are prefilled, then every later token is fed through decode (the device's own
traced-step path, logits read back in full), and each step's logits are scored against the reference's at that
position: top-1 agreement, top-1 where the reference's top-2 margin is at least 0.5, and KL(reference || device).

MIMO_DECODE_EVAL_DUMP=<file> writes the per-position rows as JSON.
"""

import json
import os

import pytest
import torch
from loguru import logger

from models.experimental.mimo_v2_6_flash.reference.model import rms_norm
from models.experimental.mimo_v2_6_flash.tests.common import (
    DP,
    GALAXY_DEVICE_PARAMS,
    cache_dir,
    golden_path,
    load_config_and_ckpt,
)
from models.experimental.mimo_v2_6_flash.tt.generator import MiMoGalaxyGenerator

PREFILL = 8
USERS = 4


@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_model_decode_eval_tp8_dp4(mesh_device):
    golden = torch.load(golden_path())
    assert "final_hidden" in golden, "needs a reference/make_golden_large.py golden (MIMO_GOLDEN)"
    cfg, ckpt = load_config_and_ckpt()
    norm, lm_head = ckpt.final_norm(), ckpt.lm_head().float()
    gen = MiMoGalaxyGenerator(
        mesh_device, cfg, ckpt, cache_dir("model_bfp4experts"), dp=DP, users_per_replica=USERS, max_seq_len=1024
    )
    seqs = golden["token_ids"]
    per_round = DP * USERS
    rows = []
    for first in range(0, len(seqs), per_round):
        group = {u: seqs[first + u] for u in range(min(per_round, len(seqs) - first))}
        gen.prefill({u: t[:PREFILL] for u, t in group.items()})
        length = min(t.numel() for t in group.values())
        refs = {}
        with torch.no_grad():
            for u in group:
                h = golden["final_hidden"][first + u][:length].float()
                refs[u] = rms_norm(h, norm, cfg.layernorm_epsilon) @ lm_head.T
        for pos in range(PREFILL, length - 1):
            out = gen.decode({u: int(t[pos]) for u, t in group.items()}, {u: pos for u in group})
            for u, lg in out.items():
                ref = refs[u][pos]
                lp_ref, lp_dev = ref.log_softmax(-1), lg.float().log_softmax(-1)
                top = ref.topk(2)
                rows.append(
                    dict(
                        seq=first + u,
                        pos=pos,
                        match=int(lg.argmax()) == int(top.indices[0]),
                        ref_margin=float(top.values[0] - top.values[1]),
                        kl=float((lp_ref.exp() * (lp_ref - lp_dev)).sum()),
                    )
                )
    n = len(rows)
    top1 = sum(r["match"] for r in rows) / n
    confident = [r for r in rows if r["ref_margin"] >= 0.5]
    logger.info(
        f"teacher-forced decode over {n} positions of {len(seqs)} sequences: top-1 {top1:.4f}, top-1 where the "
        f"reference margin >= 0.5 {sum(r['match'] for r in confident) / len(confident):.4f} ({len(confident)} rows), "
        f"mean KL {sum(r['kl'] for r in rows) / n:.4f}"
    )
    if os.environ.get("MIMO_DECODE_EVAL_DUMP"):
        with open(os.environ["MIMO_DECODE_EVAL_DUMP"], "w") as f:
            json.dump(rows, f)
    assert top1 >= 0.8, top1
