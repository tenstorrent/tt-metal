# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end greedy generation parity: Mistral4Generator vs the HF reference greedy loop.

Validates the full serving loop the unit decode tests don't exercise: parallel prefill that
populates the KV cache, on-device argmax sampling, host re-embed, and the decode cache-append
across multiple steps. The TT free-running greedy token IDs are compared to the reference model's
own autoregressive greedy IDs (2-layer, fp8-dequantized) on the same prompt.
"""
import os

import pytest
import torch
from loguru import logger
from transformers import AutoConfig

import ttnn
from models.demos.mistral4.tests.m4_text_reference import load_m4_text_reference, load_m4_weights
from models.demos.mistral4.tt.mistral4_generator import Mistral4Generator
from models.demos.mistral4.tt.mistral4_text import TtMistral4TextModel

N_LAYERS = 2
PROMPT_LEN = 16
MAX_NEW = 8


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 30000000, "num_command_queues": 1}],
    indirect=True,
)
def test_m4_generate(mesh_device, reset_seeds):
    ckpt = os.environ["HF_MODEL"]
    cfg = AutoConfig.from_pretrained(ckpt).text_config

    # Reference: load the 2-layer causal LM, build a prompt, greedy-decode autoregressively, and
    # capture per-position RoPE cos/sin for the full horizon to feed the TT generator.
    ref, _, _ = load_m4_text_reference(ckpt, n_layers=N_LAYERS)
    torch.manual_seed(0)
    ids = torch.randint(0, cfg.vocab_size, (1, PROMPT_LEN))
    total = PROMPT_LEN + MAX_NEW
    pos = torch.arange(total)[None]
    cos_full, sin_full = ref.model.rotary_emb(torch.zeros(1, total, cfg.hidden_size), pos)
    cos_full, sin_full = cos_full.float(), sin_full.float()

    ref_ids, cur = [], ids.clone()
    with torch.no_grad():
        for _ in range(MAX_NEW):
            nxt = ref(cur).logits[:, -1].argmax(-1)  # [1]
            ref_ids.append(nxt)
            cur = torch.cat([cur, nxt[:, None]], dim=1)
    ref_ids = torch.stack(ref_ids, dim=1)  # [1, MAX_NEW]

    # TT: same prompt + cos/sin, free-running greedy with on-device argmax.
    tsd = load_m4_weights(ckpt, N_LAYERS)
    tt = TtMistral4TextModel(
        mesh_device, tsd, cfg, N_LAYERS, cfg.rms_norm_eps, shard_experts=True, expert_dtype=ttnn.bfloat8_b
    )
    gen = Mistral4Generator(tt, tsd["model.embed_tokens.weight"], mesh_device, max_seq=64)
    tt_ids = gen.greedy(ids, cos_full, sin_full, MAX_NEW)  # [1, MAX_NEW]

    match = (tt_ids == ref_ids).float().mean().item()
    logger.info(f"greedy IDs  ref={ref_ids.tolist()}  tt={tt_ids.tolist()}  match={match:.3f}")
    assert tt_ids[0, 0] == ref_ids[0, 0], f"first generated token differs: tt={tt_ids[0,0]} ref={ref_ids[0,0]}"
    assert match >= 0.75, f"greedy ID match {match:.3f} below 0.75"
