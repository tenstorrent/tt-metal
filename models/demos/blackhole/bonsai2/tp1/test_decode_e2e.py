# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Decode-only end-to-end check on one chip: teacher-forced top-1/top-5 against the HF reference, and traced
decode ms/token. Every reference token goes through traced decode from position 0; there is no prefill.

Teacher forcing from position 0 is a causal forward pass, so this scores the positions TokenAccuracy scores
(the second half of the reference) without the prefill path, which fails on a p300c at c75e2d5 for both GDN
modules (the default one: L1 clash in gated_delta_attn_seq at the 2048 bucket; TP=1: prefill sized for TP=4).
    HF_MODEL=<dense HF dir> pytest models/demos/blackhole/bonsai2/tp1/test_decode_e2e.py -s
Flags under test: QWEN36_TP1_GDN=1, QWEN36_1C_SHARDED_NORM=1, QWEN36_1C_UNPAD_LOGITS=1, QWEN36_1C_DOWN_PCFG=1. DECODE_E2E_RESULTS=<path.json>
writes the numbers.
"""
import json
import os
import statistics
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt.generator_interface import prime_decode_trace
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.tt_transformers.tt.generator import Generator

BLOCK_SIZE = 64
NUM_BLOCKS = 64  # text_demo's minimum budget (4096 positions)
WARMUP_STEPS = 32


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "num_command_queues": 2}], indirect=True)
def test_decode_e2e(mesh_device):
    mesh_device.enable_program_cache()
    t0 = time.time()
    model = Qwen36Model.from_pretrained(
        mesh_device, max_batch_size=1, max_seq_len=NUM_BLOCKS * BLOCK_SIZE, enable_mtp=False
    )
    args = model.args
    logger.info(f"Model load: {time.time() - t0:.1f}s")
    ref = torch.load(f"models/tt_transformers/tests/reference_outputs/{args.model_name}.refpt")
    tokens = ref["reference_tokens"][0]
    top5 = ref["top5_tokens"]  # top5[i]: the reference's top-5 for token i + 1 given tokens[: i + 1]
    n = top5.shape[0]
    scored = range(tokens.shape[0] // 2 - 1, n)  # TokenAccuracy's span: predictions of the second half

    assert n < NUM_BLOCKS * BLOCK_SIZE
    model.allocate_kv_caches([NUM_BLOCKS, args.n_kv_heads, BLOCK_SIZE, args.head_dim], ttnn.bfloat16, batch_size=1)
    model.reset_state()
    page_table = torch.arange(NUM_BLOCKS, dtype=torch.int32).unsqueeze(0)
    gen = Generator([model], [args], mesh_device)
    prime_decode_trace(gen, model, tokens[:1].view(1, 1), torch.tensor([0]), page_table)

    preds, step_s = [], []
    for pos in range(n):
        t = time.perf_counter()
        out = gen.decode_forward(
            tokens[pos].view(1, 1),
            torch.tensor([pos]),
            page_table=page_table,
            kv_cache=None,
            enable_trace=True,
            read_from_device=True,
            reload_inputs=True,
            reload_page_table=False,
            reload_sampling_params=False,
            reset_sampling_state=False,
        )
        logits = (out[0] if isinstance(out, tuple) else out).squeeze().float()
        preds.append(int(logits.argmax()))
        step_s.append(time.perf_counter() - t)
        assert not torch.isnan(logits).any(), f"NaN in decode at position {pos}"

    def acc(span):
        top1 = sum(preds[i] == top5[i, 0].item() for i in span) / len(span)
        top5_ = sum(preds[i] in top5[i].tolist() for i in span) / len(span)
        return 100 * top1, 100 * top5_

    results = {
        # Every QWEN36_* variable that is SET, so each JSON stands on its own: a result is only
        # interpretable next to the exact flags that produced it, and listing a fixed four silently
        # mislabelled earlier runs (FUSED_QKV / OPROJ_1D / ALL_BFP4 appeared only in the log header).
        "flags": {k: v for k, v in sorted(os.environ.items()) if k.startswith(("QWEN36_", "BONSAI_"))},
        "top1_top5_second_half": acc(scored),
        "top1_top5_all": acc(range(n)),
        "ms_per_token_median": 1e3 * statistics.median(step_s[WARMUP_STEPS:]),
        "ms_per_token_mean": 1e3 * statistics.fmean(step_s[WARMUP_STEPS:]),
        "positions": n,
    }
    logger.info(
        f"flags {results['flags']}: top-1/top-5 on the scored half {results['top1_top5_second_half'][0]:.2f}/"
        f"{results['top1_top5_second_half'][1]:.2f}%, all positions {results['top1_top5_all'][0]:.2f}/"
        f"{results['top1_top5_all'][1]:.2f}%; decode {results['ms_per_token_median']:.1f} ms/token median"
    )
    if os.environ.get("DECODE_E2E_RESULTS"):
        with open(os.environ["DECODE_E2E_RESULTS"], "w") as f:
            json.dump(results, f, indent=2)
