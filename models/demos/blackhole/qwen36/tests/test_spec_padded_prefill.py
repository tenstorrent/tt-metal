# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DEVICE test of the spec engine's self-heal for a padded prefill (vLLM pad_spec_decode), driven directly (no vLLM).

vLLM pads a waiting request whose prefix-cache hit leaves ONE uncached token to 1 + K scheduled tokens (K placeholder
drafts); a runner that counts the placeholders as prompt tokens prefills the row K tokens past the prompt, at the vLLM
start_pos L - 1 (block aligned). The first verify then starts at the real prompt length L. The engine must re-prefill
the row at L instead of asserting, and the first verify must equal that of a clean L-token prefill.

Row 0: clean prefill of the L-token prompt. Row 1: the same prompt + K junk tokens, start_pos = L - 1, on page-table
blocks shared with row 0 for the cached prefix (like vLLM APC). Both rows join in ONE verify at start_pos L with the same
first token. Pass: no exception, the self-heal warning is logged for row 1 only, the engine bookkeeping holds L, and
the row-1 first-step logits equal the row-0 ones (PCC >= 0.999, equal argmax).
"""

import pytest
import torch
from loguru import logger

from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, BLOCK_SIZE, DEVICE_PARAMS
from models.demos.blackhole.qwen36.tests.spec_helpers import PAD
from models.demos.blackhole.qwen36.tests.test_spec_continuous import NB, B, K, _setup

PCC_MIN = 0.999
L = 5 * BLOCK_SIZE + 1  # 1 mod 64: the vLLM hit covers 5 full blocks, one uncached token


def _pcc(a, b):
    a, b = a.double().reshape(-1), b.double().reshape(-1)
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


@run_for_blackhole()
@pytest.mark.timeout(3000)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_padded_prefill_self_heal(mesh_device):
    assert BLOCK_SIZE == 64
    model, eng, tokenizer, prompts = _setup(mesh_device)
    prompt = (
        list(prompts[3])[:L] if len(prompts[3]) >= L else list(prompts[3]) + list(prompts[4])[: L - len(prompts[3])]
    )
    assert len(prompt) == L
    junk = [int(t) for t in prompts[6][:K]]
    cached = (L - 1) // BLOCK_SIZE  # 5 shared blocks
    blocks0 = list(range(1, 1 + 8))
    blocks1 = blocks0[:cached] + list(range(20, 20 + 8 - cached))
    pt0 = torch.tensor(blocks0 + [0] * (NB - len(blocks0)), dtype=torch.int32)
    pt1 = torch.tensor(blocks1 + [0] * (NB - len(blocks1)), dtype=torch.int32)

    warnings = []
    sink = logger.add(lambda m: warnings.append(str(m)), level="WARNING")
    try:
        eng.prefill(0, prompt, pt0[None])
        eng.prefill(1, prompt + junk, pt1[None], start_pos=L - 1)
        assert eng._fresh[eng._phys[0]] == L and eng._fresh[eng._phys[1]] == L + K

        tokens = torch.zeros((B, 1 + K), dtype=torch.int32)
        start = torch.full((B, 1 + K), PAD, dtype=torch.int32)
        pt = torch.zeros((B, NB), dtype=torch.int32)
        first = int(prompt[-1])
        for r, row in ((0, pt0), (1, pt1)):
            tokens[r] = PAD
            tokens[r, 0] = first
            start[r, 0] = L
            pt[r] = row
        res = eng.decode_forward(
            tokens,
            start,
            num_valid_drafts=torch.zeros(B, dtype=torch.int32),
            accepted_counts=torch.ones(B, dtype=torch.int32),
            page_table=pt,
            spec_mode="logits",
        )
    finally:
        logger.remove(sink)

    heal = [w for w in warnings if "re-prefilling the row" in w]
    logger.info(f"self-heal warnings: {len(heal)}: {heal[:2]}")
    assert len(heal) == 1, f"expected exactly one self-heal warning, got {len(heal)}"
    assert f"slot {eng._phys[1]}: prefilled {L + K} tokens" in heal[0], heal[0]
    assert eng._fresh == {} or all(v == L for v in eng._fresh.values())
    lg0, lg1 = res.logits[0][0].float(), res.logits[1][0].float()
    pcc = _pcc(lg0, lg1)
    a0, a1 = int(lg0.argmax()), int(lg1.argmax())
    logger.info(f"RESULT padded vs clean first verify: pcc={pcc:.6f} argmax {a1} vs {a0} equal={torch.equal(lg0, lg1)}")
    assert a0 == a1, f"first verify argmax differs: {a1} vs clean {a0}"
    assert pcc >= PCC_MIN, f"first verify logits pcc {pcc:.5f} < {PCC_MIN}"
