# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Perf harness for nomic_sdpa's key padding: per shape, NOMIC_SDPA_REPS calls (default 5) each of
  1. nomic_sdpa without key_lengths (the unpadded kernels),
  2. nomic_sdpa with a length-sorted right padding as embed() builds one (key_lengths),
  3. stock scaled_dot_product_attention with the model's additive mask, program and compute configs (what a padded
     batch ran before key padding),
in that order, for scripts/run_safe_pytest.sh --profile. Device kernel time is read from the profiler CSV (one row a
call). Correctness is test_ttnn_operators_attention.py's job; this file only checks the outputs are finite.
"""

import os

import pytest
import torch
import ttnn

from eval.golden_tests.nomic_sdpa.helpers import make_qkv
from models.experimental.nomic_embed_text_v2_moe.reference.configuration_nomic_moe import load_vendored_config
from models.experimental.nomic_embed_text_v2_moe.tt.attention import sdpa_program_config
from models.experimental.nomic_embed_text_v2_moe.tt.common import additive_attention_mask
from models.experimental.nomic_embed_text_v2_moe.tt.model_config import OpGroup, TtModelConfig
from ttnn.operations.nomic_sdpa import nomic_sdpa


def sorted_lengths(batch, seqlen):
    """A length-sorted right-padded batch: the first row full, the rest falling to a single token. NOMIC_SDPA_LENGTHS=
    minus1: every row one token short (the mask path with the least zeroing)."""
    if os.environ.get("NOMIC_SDPA_LENGTHS") == "minus1":
        return [seqlen - 1] * batch
    pattern = [seqlen, seqlen - 1, seqlen - 31, seqlen // 2 + 33, seqlen // 2, seqlen // 3, 33, 1]
    return sorted((pattern[b % len(pattern)] for b in range(batch)), reverse=True)


@pytest.mark.parametrize("shape", [(2, 512), (4, 512), (8, 512)], ids=lambda s: f"{s[0]}x{s[1]}")
def test_nomic_sdpa_key_padding_perf(device, shape):
    batch, seqlen = shape
    reps = int(os.environ.get("NOMIC_SDPA_REPS", "5"))
    mem = ttnn.L1_MEMORY_CONFIG
    q, k, v = make_qkv((batch, 12, seqlen, 64), "real", seed=0)
    tq, tk, tv = (
        ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
        for t in (q, k, v)
    )
    lengths = sorted_lengths(batch, seqlen)
    keep = (torch.arange(seqlen) < torch.tensor(lengths)[:, None]).long()

    def run(fn):
        for _ in range(reps):
            out = fn()
            assert torch.isfinite(ttnn.to_torch(out).float()).all()
            ttnn.deallocate(out)

    run(lambda: nomic_sdpa(tq, tk, tv, scale=1.0, memory_config=mem))
    run(lambda: nomic_sdpa(tq, tk, tv, scale=1.0, key_lengths=lengths, memory_config=mem))

    tt_config = TtModelConfig.from_device(device)
    config = load_vendored_config()
    compute = tt_config.compute_kernel_config(OpGroup.SDPA)
    mask = additive_attention_mask(keep, device, mask_dtype=tt_config.attention_mask_dtype)
    program = sdpa_program_config(
        batch, seqlen, config.num_attention_heads, tt_config.core_grid, compute.fp32_dest_acc_en, True
    )
    run(
        lambda: ttnn.transformer.scaled_dot_product_attention(
            tq,
            tk,
            tv,
            attn_mask=mask,
            is_causal=False,
            scale=1.0,
            memory_config=mem,
            output_concat_heads=True,
            program_config=program,
            compute_kernel_config=compute,
        )
    )
    for tensor in (tq, tk, tv, mask):
        ttnn.deallocate(tensor)
