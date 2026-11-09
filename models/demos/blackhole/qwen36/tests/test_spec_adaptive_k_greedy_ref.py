# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Plain-greedy quality of QWEN36_SPEC_ADAPTIVE_K: the greedy streams the production-loop harness wrote
(test_spec_adaptive_k_devidle.py, AK_OUT/<arm>_ctx<CTX>_u<n>.json) teacher-forced through the PLAIN decode path, the
method of test_spec_lossless.py: at every generated position the plain single-token decode is fed the spec stream's own
prefix and its argmax and top-2 logit gap are recorded, so every position is checked (no stop at the first divergence).

A spec token that differs from the plain argmax where plain was CONFIDENT (top-2 gap >= NEAR_TIE_GAP) is a speculation
bug; a difference at a near tie is bf16 noise between two kernels (either token is a faithful greedy continuation). Run
for the flag-off (K = 3) and flag-on (wide family K = 7 at <= 4 users) streams of the same prompts, so the two arms are
judged against the same reference.

Env: AK_REF_DIRS (comma list of harness AK_OUT dirs), AK_CTX (prompt tokens, default 32768), AK_REF_TOKENS (positions
checked per user, default 160), AK_REF_RUNS (comma list of user counts, default "1,2,4").
"""

import glob
import json
import math
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, BLOCK_SIZE, DEVICE_PARAMS, _get_prompt
from models.demos.blackhole.qwen36.tests.test_spec_lossless import NEAR_TIE_GAP, _reference_greedy
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

CTX = int(os.environ.get("AK_CTX", "32768"))
NTOK = int(os.environ.get("AK_REF_TOKENS", "160"))
RUNS = [int(x) for x in os.environ.get("AK_REF_RUNS", "1,2,4").split(",") if x.strip()]
DIRS = [d for d in os.environ.get("AK_REF_DIRS", "").split(",") if d.strip()]


@run_for_blackhole()
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_adaptive_k_streams_match_plain_greedy(mesh_device):
    from transformers import AutoTokenizer

    assert DIRS, "set AK_REF_DIRS"
    nb = math.ceil((CTX + NTOK + 64) / BLOCK_SIZE) + 1
    mesh_device.enable_program_cache()
    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=nb * BLOCK_SIZE, enable_mtp=None)
    tokenizer = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    model.set_gdn_fused_decode(True)  # the verify's GDN arithmetic for plain decode too (test_spec_lossless.py)
    kv_shape = [nb + 1, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim]
    pt = torch.arange(1, nb + 1, dtype=torch.int32).reshape(1, nb)
    base = _get_prompt(CTX, tokenizer)[0].tolist()
    summary = {}
    bad = []
    for d in DIRS:
        for f in sorted(glob.glob(f"{d}/*.json")):
            j = json.load(open(f))
            if j.get("churn") or j.get("sampled") or int(j.get("users", 0)) not in RUNS or int(j.get("ctx", 0)) != CTX:
                continue
            tag = j["tag"]
            tot = mism = conf = ties = 0
            for u, toks in sorted(j["gens"].items(), key=lambda kv: int(kv[0])):
                spec = [int(t) for t in toks[:NTOK]]
                prompt = base[13 * int(u) :]  # the harness prompts (_prompts): user u drops the first 13 * u tokens
                ref, gaps = _reference_greedy(model, prompt, pt, kv_shape, spec)
                m = [i for i in range(len(spec)) if spec[i] != ref[i]]
                c = [i for i in m if gaps[i] >= NEAR_TIE_GAP]
                nt = sum(g < NEAR_TIE_GAP for g in gaps)
                tot, mism, conf, ties = tot + len(spec), mism + len(m), conf + len(c), ties + nt
                logger.info(
                    f"[ak-ref] {tag} user {u}: {len(spec)} positions, {len(m)} differ from the teacher-forced plain argmax "
                    f"at {m[:12]}{' ...' if len(m) > 12 else ''} (gaps {[round(gaps[i], 3) for i in m[:12]]}); "
                    f"confident mismatches {c}; near-tie positions {nt}"
                )
                bad += [(tag, u, i, spec[i], ref[i], round(gaps[i], 3)) for i in c]
            summary[tag] = dict(positions=tot, mismatches=mism, confident=conf, near_ties=ties)
    logger.info(f"[ak-ref] summary (NEAR_TIE_GAP {NEAR_TIE_GAP}): {summary}")
    assert summary, f"no greedy harness streams in {DIRS}"
    assert not bad, f"spec tokens plain greedy would not choose where plain was confident: {bad[:20]}"
