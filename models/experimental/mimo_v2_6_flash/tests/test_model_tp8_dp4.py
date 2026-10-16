# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Full 48-layer MiMo-V2.6-Flash on the Galaxy as TP=8 x DP=4, checked against the fp32 reference.

Golden activations come from ``reference/make_golden.py`` (MIMO_GOLDEN, default
/home/zni/models/mimo-golden/<checkpoint dir name>/golden_48l.pt). Each replica (one Galaxy row) serves:

* slots 0-1: two golden prompts. Prefill all but the last 4 tokens, then teacher-force those 4
  through decode. Every logits row is compared with the reference (top-1 and PCC).
* slot 2: the same shared prompt on every replica. All four rows must agree with each other.

Then every user keeps generating greedily, and the text is logged.
"""

import os

import pytest
import torch
from loguru import logger
from transformers import AutoTokenizer

from models.experimental.mimo_v2_6_flash.tests.common import (
    DP,
    GALAXY_DEVICE_PARAMS,
    cache_dir,
    load_config_and_ckpt,
    model_path,
    pcc,
)
from models.experimental.mimo_v2_6_flash.tt.generator import MiMoGalaxyGenerator

TEACHER_FORCED = 4
GREEDY_TOKENS = 16
SHARED_PROMPT = "Tenstorrent Blackhole is a"


def golden_path():
    ckpt_name = os.path.basename(os.path.normpath(model_path()))
    path = os.environ.get("MIMO_GOLDEN", os.path.join("/home/zni/models/mimo-golden", ckpt_name, "golden_48l.pt"))
    if not os.path.exists(path):
        pytest.skip(f"golden file {path} not found; run reference/make_golden.py")
    return path


@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_model_tp8_dp4(mesh_device):
    golden = torch.load(golden_path())
    assert golden["num_layers"] == 48
    cfg, ckpt = load_config_and_ckpt()
    tok = AutoTokenizer.from_pretrained(model_path())
    gen = MiMoGalaxyGenerator(
        mesh_device, cfg, ckpt, cache_dir("model_bfp4experts"), dp=DP, users_per_replica=4, max_seq_len=1024
    )

    # ---- users: replica r gets golden prompts 2r and 2r + 1, plus the shared prompt in slot 2
    shared = torch.tensor(tok(SHARED_PROMPT)["input_ids"])
    users = {}
    for r in range(DP):
        for s in range(2):
            g = 2 * r + s
            ids = golden["token_ids"][g]
            users[r * 4 + s] = dict(golden=g, ids=ids, prefill_len=ids.numel() - TEACHER_FORCED)
        users[r * 4 + 2] = dict(golden=None, ids=shared, prefill_len=shared.numel())

    first = gen.prefill({u: usr["ids"][: usr["prefill_len"]] for u, usr in users.items()})

    top1, pccs = [], []
    for u, usr in users.items():
        if usr["golden"] is not None:
            ref = golden["logits"][usr["golden"]][usr["prefill_len"] - 1]
            top1.append(int(first[u].argmax()) == int(ref.argmax()))
            pccs.append(pcc(first[u], ref))

    # ---- teacher-forced decode over the last 4 prompt tokens
    positions = {u: usr["prefill_len"] for u, usr in users.items() if usr["golden"] is not None}
    last_logits = {}
    for t in range(TEACHER_FORCED):
        # decode feeds the token at the current position and predicts the next one
        feed = {u: int(users[u]["ids"][positions[u]]) for u in positions}
        out = gen.decode(feed, positions)
        for u, lg in out.items():
            ref = golden["logits"][users[u]["golden"]][positions[u]]
            top1.append(int(lg.argmax()) == int(ref.argmax()))
            pccs.append(pcc(lg, ref))
            last_logits[u] = lg
            positions[u] += 1

    top1_rate = sum(top1) / len(top1)
    logger.info(
        f"vs fp32 reference over {len(top1)} logits rows (4 replicas): top-1 {top1_rate:.3f}, "
        f"PCC min {min(pccs):.4f} mean {sum(pccs) / len(pccs):.4f}"
    )

    # ---- DP consistency: the shared prompt must give the same answer on every row
    shared_users = [r * 4 + 2 for r in range(DP)]
    shared_argmax = [int(first[u].argmax()) for u in shared_users]
    shared_pcc = [pcc(first[u], first[shared_users[0]]) for u in shared_users]
    logger.info(
        f"shared prompt next token per replica: {[tok.decode([t]) for t in shared_argmax]}, PCC vs row 0 {shared_pcc}"
    )

    # ---- greedy continuation for everyone
    gen_tokens = {u: [int(last_logits[u].argmax())] for u in last_logits}
    for u in shared_users:
        gen_tokens[u] = [int(first[u].argmax())]
        positions[u] = users[u]["prefill_len"]
    for _ in range(GREEDY_TOKENS - 1):
        out = gen.decode({u: v[-1] for u, v in gen_tokens.items()}, positions)
        for u, lg in out.items():
            gen_tokens[u].append(int(lg.argmax()))
            positions[u] += 1
    for u in sorted(gen_tokens):
        r, s = gen.locate(u)
        prompt_text = tok.decode(users[u]["ids"])
        logger.info(f"replica {r} slot {s}: {prompt_text!r} -> {tok.decode(gen_tokens[u])!r}")
    shared_texts = [gen_tokens[u] for u in shared_users]

    assert all(t == shared_texts[0] for t in shared_texts), "replicas disagree on the shared prompt"
    assert min(shared_pcc) > 0.9999, shared_pcc
    assert top1_rate >= 0.8, top1_rate
    assert sum(pccs) / len(pccs) > 0.95, pccs
