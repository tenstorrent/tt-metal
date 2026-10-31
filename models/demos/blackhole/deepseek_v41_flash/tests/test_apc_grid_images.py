# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hardware-free tests: the APC M0' grid policy (tip / zone / divergence snapshots) plans image prompts on the image-aware key chain (DSV41_APC_IMAGES=1),
and text-only plans are identical with the image code present.

    pytest --noconftest models/demos/blackhole/deepseek_v41_flash/tests/test_apc_grid_images.py
"""

import torch

from models.demos.blackhole.deepseek_v41_flash.tests.test_apc_grid import C, make_cache, seg
from models.demos.blackhole.deepseek_v41_flash.tt import prefix_cache as PC
from models.demos.blackhole.deepseek_v41_flash.tt.vision_state import VisionSeq

IMG = 99999


def image_prompt(n, start, k):
    t = torch.as_tensor(seg(7, n)).long().clone()
    t[start : start + k] = IMG
    return t


def vseq(t, spans, keys):
    types = torch.full((t.numel(),), -1, dtype=torch.int64)
    rows = []
    for s, k in spans:
        types[s : s + k] = 0
        rows.append((s, torch.zeros(k, 4, dtype=torch.bfloat16)))
    return VisionSeq(types, rows, keys)


def test_text_only_plans_identical_with_the_image_code_present(monkeypatch):
    t = torch.as_tensor(seg(5, 9000)).long()
    off = make_cache(monkeypatch)
    monkeypatch.delenv("DSV41_APC_IMAGES", raising=False)
    off.new_prompt(0, t, t.numel(), 0)
    plan_off = dict(off.plans[0][1])
    on = make_cache(monkeypatch)
    on.image_token_id = IMG
    monkeypatch.setenv("DSV41_APC_IMAGES", "1")
    on.new_prompt(0, t, t.numel(), 0, vs=None)
    assert dict(on.plans[0][1]) == plan_off and plan_off
    # a zone plan (bounded replay) is identical too
    off.new_prompt(1, t, t.numel(), 4096)
    on.new_prompt(1, t, t.numel(), 4096, vs=None)
    assert dict(on.plans[1][1]) == dict(off.plans[1][1])


def test_image_prompt_gets_a_plan_only_with_the_switch(monkeypatch):
    t = image_prompt(9000, 600, 100)
    vs = vseq(t, [(600, 100)], ["aa" * 16])
    c = make_cache(monkeypatch)
    c.image_token_id = IMG
    monkeypatch.delenv("DSV41_APC_IMAGES", raising=False)
    c.new_prompt(0, t, t.numel(), 0, vs=vs)
    assert 0 not in c.plans  # switch off: image prompts bypass the cache (unchanged behaviour)
    monkeypatch.setenv("DSV41_APC_IMAGES", "1")
    c.new_prompt(0, t, t.numel(), 0, vs=vs)
    plan = c.plans[0][1]
    assert plan.get(9000 // C * C) == "tip"  # the tip of the image prompt is planned
    c.new_prompt(1, t, t.numel(), 0, vs=None)  # image placeholders no span explains: no plan, no crash
    assert 1 not in c.plans


def test_divergence_follows_the_image_key_chain(monkeypatch):
    monkeypatch.setenv("DSV41_APC_IMAGES", "1")
    c = make_cache(monkeypatch)
    c.image_token_id = IMG
    t = image_prompt(9000, 600, 100)
    vsA = vseq(t, [(600, 100)], ["aa" * 16])
    vsB = vseq(t, [(600, 100)], ["bb" * 16])
    c.new_prompt(0, t, t.numel(), 0, vs=vsA)
    # the same text with the same image: its chain was seen -> a divergence point below the tip
    t2 = t.clone()
    t2[8000:] = torch.as_tensor(seg(9, 1000)).long()
    c.new_prompt(1, t2, t2.numel(), 0, vs=vseq(t2, [(600, 100)], ["aa" * 16]))
    assert "div" in c.plans[1][1].values()
    # the same text with ANOTHER image shares nothing past the image start (chain differs): the divergence point is below the image or absent
    c.new_prompt(2, t, t.numel(), 0, vs=vsB)
    div = [B for B, k in c.plans[2][1].items() if k == "div"]
    assert all(B <= 512 for B in div)
    assert PC.key_tokens(t, PC.image_spans(vsA), IMG)[0].tolist() != PC.key_tokens(t, PC.image_spans(vsB), IMG)[0].tolist()
