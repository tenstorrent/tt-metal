# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Wormhole LoudBox (T3K) prefill last-token logits PCC for GLM-4.7-Flash.

Wormhole counterpart to ``test_text_prefill_logits.py`` (which is Blackhole-only
because its bf16-expert correctness config OOMs WH DRAM).

Why a separate WH test + a 0.95 bar (not 0.97):
  * The 0.97 reference config needs **bf16 experts**, which do not fit WH T3K DRAM
    (~7 GB/chip for experts alone, even EP-sharded -> OOM). On WH the highest expert
    precision that fits is **bf8**.
  * Measured WH prefill-logits PCC vs HF at ISL-128 (TP=0, MOE_FP32_ACC=1):
        bf4 experts = 0.935 | bf8 experts = 0.954 | bf16 = OOM (Blackhole-only).
    So ~0.95 is the achievable WH ceiling; this test gates at 0.95.
  * Uses TP=0 (replicated) like the Blackhole correctness test. NOTE: the production
    TP=1 path currently scores only ~0.65 (a separate tensor-parallel accuracy bug);
    this reference test isolates precision from that bug by using TP=0.

Run:
  TT_METAL_GTEST_ETH_DISPATCH=1 GLM4_MOE_LITE_CCL_NUM_LINKS=1 \
  GLM4_MOE_LITE_CCL_TOPOLOGY=linear GLM4_MOE_LITE_PREFILL_MATMUL_TUNED=0 \
  python -m pytest -q -s \
    models/experimental/glm4_moe_lite/tests/pipeline_tests/test_text_prefill_logits_wh.py
"""

from __future__ import annotations

import bz2

import pytest
import torch
from loguru import logger

from models.common.utility_functions import comp_allclose, comp_pcc, run_for_wormhole_b0

from models.experimental.glm4_moe_lite.tests.pipeline_tests.test_utils import (
    PROMPT_FILE,
    alloc_kv_cache_and_page_table,
    apply_wh_correctness_env,
    compute_max_seq_len,
    create_runner,
    fabric_1d_trace_device_params,
    hf_prefill_last_token_logits,
    load_hf_causal_lm,
    load_tokenizer,
    mesh_shape_param,
    require_snapshot,
    scale_page_params,
)

# Achievable WH ceiling with bf8 experts (bf16 OOMs). See module docstring.
WH_PCC_REQUIRED = 0.95


@torch.no_grad()
@run_for_wormhole_b0()
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("seq_len", (128,), ids=["128"])
@pytest.mark.parametrize("batch_size", (1,))
@pytest.mark.parametrize("page_params", [{"page_block_size": 64, "page_max_num_blocks": 1024}])
@pytest.mark.parametrize("device_params", fabric_1d_trace_device_params(num_command_queues=2), indirect=True)
@pytest.mark.parametrize("mesh_device", [mesh_shape_param()], indirect=True)
def test_text_prefill_logits_wh(
    seq_len,
    batch_size,
    page_params,
    mesh_device,
    reset_seeds,
    is_ci_env,
    monkeypatch: pytest.MonkeyPatch,
):
    """WH prefill last-token logits PCC vs HF (bf8 experts, TP=0) -> gate at 0.95."""
    apply_wh_correctness_env(monkeypatch)

    pcc_required = WH_PCC_REQUIRED
    snap = require_snapshot()
    page_params = scale_page_params(page_params, seq_len, batch_size)
    logger.info(
        f"Paged KV cache: {page_params['page_max_num_blocks']} blocks x "
        f"{page_params['page_block_size']} (seq_len={seq_len})"
    )

    tok = load_tokenizer(snap)
    with bz2.open(PROMPT_FILE, "rt", encoding="utf-8") as f:
        prompt = f.read()
    encoded_prompt = tok(prompt, add_special_tokens=True)["input_ids"][:seq_len]
    if len(encoded_prompt) < seq_len:
        pytest.skip(f"Prompt corpus shorter than seq_len={seq_len}")
    logger.info(f"Prefill length: {seq_len} tokens")

    hf_model = load_hf_causal_lm(snap)
    vocab_size = int(hf_model.config.vocab_size)

    block_size = int(page_params["page_block_size"])
    total_tokens = seq_len + 32
    max_seq_len = compute_max_seq_len(total_tokens, block_size)
    runner = create_runner(
        mesh_device=mesh_device,
        snapshot_dir=snap,
        max_seq_len=max_seq_len,
        cache_subdir="pipeline_prefill_logits_wh",
    )
    kv_cache, page_table, _ = alloc_kv_cache_and_page_table(
        mesh_device=mesh_device,
        runner=runner,
        batch_size=batch_size,
        total_tokens=total_tokens,
        block_size=block_size,
    )

    prompt_ids = torch.tensor([encoded_prompt], dtype=torch.int32)
    logger.info("Running TT prefill...")
    tt_logits = runner.prefill(
        tokens=prompt_ids,
        prompt_lens=[seq_len],
        page_table=page_table,
        kv_cache=kv_cache,
        seq_pad_multiple=int(page_params["page_block_size"]),
    )

    logger.info("Running HF prefill reference...")
    hf_input = torch.tensor([encoded_prompt], dtype=torch.long)
    ref_logits = hf_prefill_last_token_logits(hf_model, hf_input, vocab_size)

    tt_logits_cmp = tt_logits[:, :1, :vocab_size].to(dtype=torch.float32)
    passing, pcc_val = comp_pcc(ref_logits, tt_logits_cmp, pcc_required)
    logger.info(comp_allclose(ref_logits, tt_logits_cmp))
    logger.info(f"[WH] Prefill last-token logits PCC (seq_len={seq_len}): {pcc_val}")

    assert passing, f"[WH] Prefill logits PCC {pcc_val} below {pcc_required} for seq_len={seq_len}."
