# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Teacher-forced token accuracy of the TT Gemma 4 stack against the HF reference.

The whole-model gate the per-op PCC tests cannot give: the TT generator is
fed the reference passage token by token (prefill of the first ``PREFILL_LEN``
tokens, then one decode step per position with the TRUE next token), and at
every position its top-1 / top-5 are scored against the Hugging Face model's
predictions stored by ``generate_reference_tokens.py``:

    top1  = TT argmax == HF argmax
    top5  = HF argmax in TT top-5

Batch 8 and 32 feed the same passage on every row and additionally require
every row to agree with row 0 token for token (batch-dependent branches in
RoPE / SDPA splits showed up exactly there, tt-metal#59300).

    HF_MODEL=google/gemma-4-31B-it MESH_DEVICE=P150x4 pytest test_token_accuracy.py -k "1x4 and batch_1"
"""

import os

import pytest
import torch
from loguru import logger

from models.demos.gemma4.demo.text_demo import (
    _batch_page_params,
    _create_tt_page_table,
    _install_hybrid_page_tables,
    _resolve_bounded_sliding,
)
from models.demos.gemma4.tests.test_factory import parametrize_mesh_with_fabric, skip_if_config_only_checkpoint
from models.demos.gemma4.tt.generator import Gemma4Generator
from models.tt_transformers.tt.common import PagedAttentionConfig

PREFILL_LEN = 128
# Gates, set just under the measured values so a regression trips them;
# override for an investigation with GEMMA4_MIN_TOP1 / GEMMA4_MIN_TOP5.
# Measured 2026-10-07, 31B on the QB2 1x4 mesh (bfp8 weights + KV), batch 1,
# 385 scored positions: top-1 vs HF 0.8545, HF top-1 in TT top-5 0.9922,
# top-1 vs the passage 0.4338 where HF itself scores 0.5246. The 9-point gap
# against the passage is the precision loss to chase, not test noise.
MIN_TOP1 = float(os.environ.get("GEMMA4_MIN_TOP1", "0.82"))
MIN_TOP5 = float(os.environ.get("GEMMA4_MIN_TOP5", "0.98"))


def _reference_path(model_path):
    name = os.path.basename(os.path.normpath(model_path))
    return os.path.join(os.path.dirname(__file__), "reference_outputs", f"{name}.refpt")


def _logits_of(out):
    logits = out[0] if isinstance(out, tuple) else out
    return logits.reshape(logits.shape[0], -1).float()  # [B, vocab]


@pytest.fixture
def model_path():
    return os.environ.get("HF_MODEL", "google/gemma-4-31B-it")


@parametrize_mesh_with_fabric(device_params_extra={"trace_region_size": 256_000_000})
@pytest.mark.parametrize("batch_size", [1, 8, 32], ids=["batch_1", "batch_8", "batch_32"])
def test_token_accuracy(mesh_device, model_path, batch_size):
    skip_if_config_only_checkpoint()
    ref_path = _reference_path(model_path)
    if not os.path.exists(ref_path):
        pytest.skip(f"no HF reference at {ref_path}; run generate_reference_tokens.py")
    ref = torch.load(ref_path)
    ids = ref["reference_tokens"]  # [1, N + 1]
    hf_top1 = ref["top1_tokens"]  # [N]
    score_start = int(ref.get("score_start", 1))  # first passage token; earlier positions are the prompt
    n_positions = ids.shape[1] - 1
    assert n_positions > PREFILL_LEN

    max_new = n_positions - PREFILL_LEN + 1
    max_seq_len = 4096
    page_params = _batch_page_params(batch_size, PREFILL_LEN, max_new)
    paged_attention_config = PagedAttentionConfig(
        block_size=int(page_params["page_block_size"]),
        max_num_blocks=int(page_params["page_max_num_blocks"]),
    )
    page_table = _create_tt_page_table(batch_size, paged_attention_config)
    bounded_sliding = _resolve_bounded_sliding(max_seq_len, mesh_device, model_path)

    generator, tt_kv_cache, _tokenizer = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=batch_size,
        max_seq_len=max_seq_len,
        paged_attention_config=paged_attention_config,
        bounded_sliding_kv_cache=bounded_sliding,
    )
    model_args = generator.model_args[0]
    if bounded_sliding:
        _install_hybrid_page_tables(
            generator.model[0],
            model_args,
            batch_size=batch_size,
            block_size=int(page_params["page_block_size"]),
            max_seq_len=max_seq_len,
        )

    # Prefill the first PREFILL_LEN tokens; its logits predict position PREFILL_LEN - 1.
    prefill_tokens = ids[:, :PREFILL_LEN].repeat(batch_size, 1)
    prefill_out = generator.prefill_forward_text(
        prefill_tokens,
        page_table=page_table,
        kv_cache=tt_kv_cache,
        prompt_lens=[PREFILL_LEN] * batch_size,
        warmup_prefill=False,
        enable_trace=False,
    )
    scored_positions = [PREFILL_LEN - 1]
    tt_logits = [_logits_of(prefill_out)]

    # Teacher-forced decode: feed the TRUE token at ``pos``; the logits predict pos + 1.
    current_pos = torch.full((batch_size,), PREFILL_LEN, dtype=torch.int32)
    for pos in range(PREFILL_LEN, n_positions):
        tok = ids[:, pos].repeat(batch_size, 1)
        out = generator.decode_forward(
            tok,
            current_pos,
            page_table=page_table,
            kv_cache=tt_kv_cache,
            enable_trace=True,
            reload_inputs=True,
            reload_page_table=False,
            reload_sampling_params=False,
            reset_sampling_state=False,
        )
        tt_logits.append(_logits_of(out))
        scored_positions.append(pos)
        current_pos += 1

    logits = torch.stack(tt_logits, dim=1)  # [B, S, vocab]
    positions = torch.tensor(scored_positions)
    keep = positions >= score_start - 1  # logits at p predict p + 1
    logits, positions = logits[:, keep], positions[keep]
    tt_top5 = torch.topk(logits, k=5, dim=-1).indices  # [B, S, 5]
    tt_top1 = tt_top5[..., 0]
    hf1 = hf_top1[positions]  # [S]

    top1 = (tt_top1[0] == hf1).float().mean().item()
    top5 = (tt_top5[0] == hf1[:, None]).any(-1).float().mean().item()
    passage_next = ids[0, positions + 1]
    passage_top1 = (tt_top1[0] == passage_next).float().mean().item()
    logger.info(
        f"{os.path.basename(model_path)} batch={batch_size} scored={int(keep.sum())}: "
        f"top1 vs HF {top1:.4f}, HF-top1 in TT-top5 {top5:.4f}, top1 vs passage {passage_top1:.4f}"
    )

    if batch_size > 1:
        rows_match = (tt_top1 == tt_top1[0:1]).all(dim=-1)  # [B]
        bad = [int(b) for b in torch.nonzero(~rows_match).flatten()]
        assert not bad, f"rows {bad} disagree with row 0 on an identical teacher-forced passage"

    assert top1 >= MIN_TOP1, f"top-1 agreement with HF {top1:.4f} < {MIN_TOP1}"
    assert top5 >= MIN_TOP5, f"HF top-1 inside TT top-5 {top5:.4f} < {MIN_TOP5}"
