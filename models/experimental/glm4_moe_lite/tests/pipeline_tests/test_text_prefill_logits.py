# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Full-depth text prefill last-token logits PCC for GLM-4.7-Flash.

Modeled on ``models/experimental/mistral_24b/tests/pipeline_tests/test_text_prefill_logits.py``:
real tokens from *Tale of Two Cities*, all decoder layers + norm + lm_head, sweeping
``seq_len`` up to the model context window. HF reference uses ``AutoModelForCausalLM``.
"""

from __future__ import annotations

import bz2

import pytest
import torch
from loguru import logger

from models.common.utility_functions import comp_allclose, comp_pcc, run_for_blackhole

from models.experimental.glm4_moe_lite.tests.pipeline_tests.test_utils import (
    PROMPT_FILE,
    apply_correctness_env,
    alloc_kv_cache_and_page_table,
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


@torch.no_grad()
@run_for_blackhole()
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("seq_len", (128, 1024), ids=["128", "1k"])
@pytest.mark.parametrize("batch_size", (1,))
@pytest.mark.parametrize("page_params", [{"page_block_size": 64, "page_max_num_blocks": 1024}])
@pytest.mark.parametrize("device_params", fabric_1d_trace_device_params(num_command_queues=2), indirect=True)
@pytest.mark.parametrize("mesh_device", [mesh_shape_param()], indirect=True)
def test_text_prefill_logits(
    seq_len,
    batch_size,
    page_params,
    mesh_device,
    reset_seeds,
    is_ci_env,
    monkeypatch: pytest.MonkeyPatch,
):
    """Prefill last-token logits PCC vs HF for increasing sequence lengths (full model depth)."""
    if is_ci_env and seq_len != 128:
        pytest.skip("CI runs prefill logits at seq_len=128 only.")

    apply_correctness_env(monkeypatch)

    pcc_required = 0.97
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
        cache_subdir="pipeline_prefill_logits",
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
    logger.info(f"Prefill last-token logits PCC (seq_len={seq_len}): {pcc_val}")

    assert passing, f"Prefill logits PCC {pcc_val} below {pcc_required} for seq_len={seq_len}."
