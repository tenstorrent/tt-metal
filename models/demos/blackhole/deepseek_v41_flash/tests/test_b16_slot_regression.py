# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Regression test of the B=16 / 262,144 wrong-answer bug (evals_gpqa_tau3/b16_isolation): the vLLM adapter built like the production server (B=16, max_ctx 262,144, one prefill slot per mesh row,
chunk 512, spec k=5, warm-up traces), driven with the plugin call protocol, must answer a 4,934-token tool-calling prompt (tests/data/probe_t22_k3_ids.json, airline system prompt + 14 tools) with
get_user_details(user_id=omar_rossi_1241) in EVERY user slot: all 16 slots at once, slots 0 / 1 / 5 / 15 alone, again after other requests ran, plain decode and speculative decode; and the logits of the
teacher-forced answer must be healthy (top-1 >= 38 and a margin > 10 at the checked steps; the broken state gave top-1 20-31, margins 0.2-3 at step 11).

Cause it guards (DSV41_KEY_EXPORT_FILL=0, the old default): the slice_write index-key export re-allocated the decode key slab that the captured decode traces read. Runs with that setting fail here.

    pytest models/demos/blackhole/deepseek_v41_flash/tests/test_b16_slot_regression.py   (needs the 32-chip galaxy, ~20 min build with warm caches)
Env of the production spec (DSV41_VLLM_SPEC_K=5, DSV41_VLLM_CHUNK=512, DSV41_SPEC_MAX_CTX=262144, DSV41_VISION=1, ...) is set by the test; DSV41_KEY_EXPORT_FILL=0 in the environment makes it fail (evidence in the README).
"""

import json
import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tests.test_b16_server_like import (
    GOOD,
    IDS,
    Sim,
    _hf_config,
    _stub_plugin,
)

# the answer of the healthy model (tokens of '\n\n<DSML calls>... get_user_details ... omar_rossi_1241'), teacher forced to read the logits step by step
GOOD_TOKENS = [
    271,
    30,
    128825,
    10699,
    1018,
    30,
    128825,
    34756,
    2329,
    1281,
    1133,
    26674,
    90983,
    3816,
    30,
    128825,
    10767,
    2329,
    1281,
    5265,
    5165,
    4,
    3418,
    1281,
]
LOGIT_STEPS = (3, 11)  # steps (0-based, fed token index) whose top-1 / margin are checked
PROD_ENV = {
    "DSV41_VLLM_SPEC_K": "5",
    "DSV41_VLLM_CHUNK": "512",
    "DSV41_SPEC_MAX_CTX": "262144",
    "DSV41_VLLM_STATS_EVERY": "100000",
    "DSV41_LAYERS": "0-39",
    "DSV41_ENGRAM_RAM": "1",
    "DSV41_POOL_DTYPE": "fp8",
    "DSV41_VISION": "1",
    "MOE_COMPUTE_FP32_ACC": "1",
    "MOE_COMPUTE_BFP8_WEIGHTS": "1",
}


@pytest.fixture
def adapter(mesh_device):
    from models.demos.blackhole.deepseek_v41_flash.tt.generator_vllm import DeepseekV41ForCausalLM

    for k, v in PROD_ENV.items():
        os.environ.setdefault(k, v)
    _stub_plugin()
    model = DeepseekV41ForCausalLM.initialize_vllm_model(_hf_config(), mesh_device, 16, 262144)
    yield model


def _logit_check(model, ids, slot):
    """Teacher-forced decode of the healthy answer through the ordinary decode step: top-1 / margin at LOGIT_STEPS."""
    S, W = Sim(model), 16
    model.release_request(slot)
    first = S.prefill({slot: ids})
    assert first[slot] == GOOD_TOKENS[0]
    pos, cur, seen = len(ids), first[slot], {}
    for i in range(max(LOGIT_STEPS) + 1):
        tokens = torch.zeros(W, 1, dtype=torch.int32)
        start = torch.full((W,), -1, dtype=torch.int64)
        tokens[slot, 0], start[slot] = cur, pos
        model._decode_forward_impl(tokens, start, sampling_params=S.sp(W))
        lg = model.m.read_logits().float()[model.slots.phys[slot]]
        v, ix = lg.topk(2)
        seen[i] = (int(ix[0]), float(v[0]), float(v[0] - v[1]))
        cur, pos = GOOD_TOKENS[i + 1], pos + 1
    for i in LOGIT_STEPS:
        tok, top1, margin = seen[i]
        assert tok == GOOD_TOKENS[i + 1], f"slot {slot} step {i}: argmax {tok} != {GOOD_TOKENS[i + 1]} ({seen[i]})"
        assert (
            top1 >= 38.0 and margin > 10.0
        ), f"slot {slot} step {i}: degraded logits top-1 {top1:.1f} margin {margin:.1f}"


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {"l1_small_size": 16384, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 800000000},
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(7200)
def test_b16_every_slot_answers(mesh_device, device_params, adapter):
    ids = json.load(open(IDS))["ids"]
    assert len(ids) == 4934
    S, tok = Sim(adapter), adapter.generator.tokenizer

    def check(name, com, phys):
        res = S.report(name, com, phys, tok)
        bad = [s for s, ok in res.items() if not ok]
        assert not bad, f"{name}: slots {bad} did not answer get_user_details({GOOD})"

    # single requests, spec decode (the production flow), first on a fresh server state
    for s in (0, 1, 5, 15):
        com, phys = S.run({s: ids}, ntok=80, mode="spec")
        check(f"spec single s{s}", com, phys)
    # plain (non-speculative) decode steps after the same prefill: isolates the decode key state from the drafter
    for s in (0, 5):
        com, phys = S.run({s: ids}, ntok=80, mode="plain")
        check(f"plain s{s}", com, phys)
    # all 16 users at once (4 per mesh row), after other requests have run
    com, phys = S.run({s: ids for s in range(16)}, ntok=80, mode="spec")
    check("spec all16", com, phys)
    com, phys = S.run({s: ids for s in range(16)}, ntok=80, mode="spec", together=False)
    check("spec all16 sequential prefills", com, phys)
    # and again single slots after the full batch (history dependence)
    for s in (15, 1):
        com, phys = S.run({s: ids}, ntok=80, mode="spec")
        check(f"spec single s{s} after the batch", com, phys)
    # logits of the teacher-forced answer: healthy band in two different slots
    for s in (0, 7):
        _logit_check(adapter, ids, s)
