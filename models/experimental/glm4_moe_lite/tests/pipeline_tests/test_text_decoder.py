# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Decode logits PCC test for GLM-4.7-Flash text decoder.

Full model (all layers + norm + lm_head) vs HuggingFace after a short prefill,
then ``generation_length`` decode steps with pseudo-random token inputs so TT and HF
KV caches stay aligned.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

from models.common.utility_functions import comp_allclose, comp_pcc, run_for_blackhole

from models.experimental.glm4_moe_lite.tests.pipeline_tests.test_utils import (
    apply_correctness_env,
    alloc_kv_cache_and_page_table,
    compute_max_seq_len,
    create_runner,
    fabric_1d_trace_device_params,
    hf_decode_logits,
    load_hf_causal_lm,
    load_tokenizer,
    mesh_shape_param,
    require_snapshot,
)


@torch.no_grad()
@run_for_blackhole()
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("generation_length", (32,))
@pytest.mark.parametrize("batch_size", (1,))
@pytest.mark.parametrize("prefill_len", (32,))
@pytest.mark.parametrize("page_params", [{"page_block_size": 64, "page_max_num_blocks": 1024}])
@pytest.mark.parametrize("device_params", fabric_1d_trace_device_params(num_command_queues=2), indirect=True)
@pytest.mark.parametrize("mesh_device", [mesh_shape_param()], indirect=True)
def test_text_decoder(
    prefill_len,
    batch_size,
    generation_length,
    page_params,
    mesh_device,
    reset_seeds,
    monkeypatch: pytest.MonkeyPatch,
):
    """Decode logits PCC: full model vs HF after prefill, for multiple decode steps."""
    apply_correctness_env(monkeypatch)

    pcc_required = 0.97
    snap = require_snapshot()
    tok = load_tokenizer(snap)
    enc = tok("Hello.", add_special_tokens=True)["input_ids"][:prefill_len]
    if len(enc) < prefill_len:
        pytest.skip(f"Tokenizer produced fewer than {prefill_len} tokens")

    hf_model = load_hf_causal_lm(snap)
    vocab_size = int(hf_model.config.vocab_size)

    block_size = int(page_params["page_block_size"])
    total_tokens = prefill_len + generation_length + 32
    max_seq_len = compute_max_seq_len(total_tokens, block_size)
    runner = create_runner(
        mesh_device=mesh_device,
        snapshot_dir=snap,
        max_seq_len=max_seq_len,
        cache_subdir="pipeline_text_decoder",
    )
    kv_cache, page_table, _ = alloc_kv_cache_and_page_table(
        mesh_device=mesh_device,
        runner=runner,
        batch_size=batch_size,
        total_tokens=total_tokens,
        block_size=block_size,
    )

    prompt_ids = torch.tensor([enc], dtype=torch.int32)
    _ = runner.prefill(
        tokens=prompt_ids,
        prompt_lens=[prefill_len],
        page_table=page_table,
        kv_cache=kv_cache,
        seq_pad_multiple=int(page_params["page_block_size"]),
    )

    hf_input = torch.tensor([enc], dtype=torch.long)
    hf_outputs = hf_model(hf_input, use_cache=True, return_dict=True)
    hf_past = hf_outputs.past_key_values

    torch.manual_seed(0)
    all_tests_pass = True
    for step in range(generation_length):
        token_id = int(torch.randint(0, vocab_size, (1,)).item())
        pos = prefill_len + step
        logger.info(f"[Text Decoder] step={step} pos={pos} token_id={token_id}")

        tt_logits = runner.decode(
            tokens=torch.tensor([[token_id]], dtype=torch.int32),
            start_pos=torch.tensor([pos], dtype=torch.int32),
            page_table=page_table,
            kv_cache=kv_cache,
        )
        tt_logits_cmp = tt_logits[:batch_size, :1, :vocab_size].to(dtype=torch.float32)

        ref_logits, hf_past = hf_decode_logits(
            hf_model,
            token_id=token_id,
            past_key_values=hf_past,
            vocab_size=vocab_size,
        )

        passing, pcc_message = comp_pcc(ref_logits, tt_logits_cmp, pcc_required)
        logger.info(comp_allclose(ref_logits, tt_logits_cmp))
        logger.info(pcc_message)
        if not passing:
            logger.warning(f"GLM text decoder logits failed at step={step} pos={pos}")
            all_tests_pass = False

    assert all_tests_pass, f"Logits PCC below {pcc_required} for one or more decode iterations. Check warnings!"
