# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Whole model vs the transformers-5.17 goldens: 128-token prefill (per-layer residual PCC, top-1/top-5 vs golden top-100),
then one decode step at position 128. Needs the full weight set -> run on the 1x4 mesh (KIMI_MESH_SHAPE=1x4)."""
from __future__ import annotations

import os
import time

import pytest
import torch

from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.tests.utils import pcc
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.tt.model import KimiLinearModel


def _t(h):
    return (h if torch.is_tensor(h) else h[0]).float()


@pytest.fixture(scope="module")
def model(mesh_device, hf_config, checkpoint, cache_path):
    if mesh_device.get_num_devices() < 2:
        pytest.skip("the full model does not fit one chip (bfp8 experts ~50 GB)")
    t0 = time.time()
    m = KimiLinearModel(mesh_device, hf_config, checkpoint, max_batch_size=1, cache_path=cache_path)
    m.allocate_state(num_blocks=64)
    print(f"[model] built + state allocated in {time.time()-t0:.0f}s")
    return m


def test_prefill_and_decode_vs_goldens(model, goldens, hf_config):
    tokens = goldens["tokens"]  # [129] (128 prompt + 1 decode token)
    pre = goldens["runs"]["prefill128"]
    ref_top100 = pre["top100"]  # [128, 100] int32
    pt = torch.arange(64, dtype=torch.int32).reshape(1, 64)
    t0 = time.time()
    logits, hidden = model.prefill(
        tokens[:128], pt, slot=0, return_hidden=True
    )  # last-token logits [vocab], hidden [128, H]
    print(f"[prefill] 128 tokens in {time.time()-t0:.1f}s")
    # last-layer hidden vs golden final_norm INPUT (hook 'final_norm' in)
    ref_h = _t(pre["hooks"]["final_norm"]["in"])[0]  # [128, H]
    p = pcc(ref_h, hidden)
    print(f"[pcc] last-layer hidden (128 tokens): {p:.5f}")
    # last token top-k
    ref_last = ref_top100[-1].long()
    tt_top5 = logits.topk(5).indices
    print(f"[prefill logits] tt top-5 {tt_top5.tolist()} | ref top-5 {ref_last[:5].tolist()}")
    top1 = int(tt_top5[0] == ref_last[0])
    top5 = int(ref_last[0] in tt_top5.tolist())
    print(
        f"[prefill logits] top-1 match {top1} top-5 contains ref argmax {top5}; ref argmax rank in tt top-100 {(logits.topk(100).indices == ref_last[0]).nonzero().flatten().tolist()}"
    )
    # decode step at position 128
    dec = goldens["runs"]["decode128"]
    tok = tokens[128:129]
    t0 = time.time()
    dlog = model.decode(tok.reshape(1), torch.tensor([128], dtype=torch.int32), pt)[0]
    print(f"[decode] one step in {time.time()-t0:.2f}s (eager)")
    ref_d = dec["top100"].long()
    dt5 = dlog.topk(5).indices
    print(
        f"[decode logits] tt top-5 {dt5.tolist()} | ref top-5 {ref_d[:5].tolist()}; pcc(logits) {pcc(dec['logits_last'], dlog):.5f}"
    )
    assert p > 0.95, p
    assert top5 == 1
    assert ref_d[0] in dt5.tolist()


def test_teacher_forcing_refpt(model, hf_config):
    """Teacher-forced accuracy on the stage-02 AIME24 refpt (100 generated tokens): top-1 / top-5 vs the HF top-100."""
    import sys

    sys.path.insert(0, os.environ["RT"])
    from readiness_check.schema import load_reference

    ref = load_reference(os.environ["KIMI_REFPT"])
    e = ref.entries[0]
    prompt = e.prompt_tokens[0]
    gen = e.generated_tokens[0]
    topk = e.topk_tokens.long()  # [G, 100]
    pt = torch.arange(64, dtype=torch.int32).reshape(1, 64)
    t0 = time.time()
    logits = model.prefill(prompt, pt, slot=0)
    print(f"[tf] prefill {prompt.numel()} tokens: {time.time()-t0:.1f}s")
    hits1 = hits5 = hits100 = 0
    cur = logits
    times = []
    for i in range(gen.numel()):
        pred = cur.topk(5).indices.tolist()
        hits1 += int(pred[0] == int(topk[i, 0]))
        hits5 += int(int(topk[i, 0]) in pred)
        hits100 += int(pred[0] in topk[i].tolist())
        pos = prompt.numel() + i
        t1 = time.time()
        cur = model.decode(gen[i : i + 1], torch.tensor([pos], dtype=torch.int32), pt)[0]
        times.append(time.time() - t1)
    G = gen.numel()
    print(
        f"[tf] top-1 {hits1/G:.3f} top-5 {hits5/G:.3f} top-100 {hits100/G:.3f}; eager decode {1000*sum(times)/len(times):.0f} ms/token"
    )
    assert hits5 / G >= 0.9, hits5 / G
