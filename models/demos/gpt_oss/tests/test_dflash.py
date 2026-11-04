# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DFlash accuracy: the tokens of the traced DFlash generation (demo/dflash_demo.py) against the target's own greedy
decode, teacher-forced: the plain decode path is fed DFlash's tokens and its argmax at every position is compared with
DFlash's next token. Greedy DFlash keeps only tokens the 8-row verify pass chose, so the two agree up to BF16 rounding
differences between the verify pass's multi-row ops and the one-row decode (near-ties flip; a free-running comparison
diverges at the first flip, here after ~100-180 tokens). Also checks the mean accepted tokens per round, which falls
if the drafter's context or K / V cache is wrong.

    pytest models/demos/gpt_oss/tests/test_dflash.py
"""

import pytest
import torch
from loguru import logger

from models.demos.gpt_oss.demo.dflash_demo import (
    NEW_TOKENS,
    book_prompt,
    build_dflash,
    fetch_books,
    parametrize_dflash_mesh,
    run_dflash,
)

ISLS = {"1k": 1024, "16k": 16384}  # 16K: the long-context attention paths (block attention over many slices)
MIN_AGREEMENT = 0.95  # teacher-forced top-1 agreement (measured 0.973 at 1K, 0.984 at 16K)
MIN_ACCEPT = 3.0  # mean accepted tokens per round (measured 4.11 at 1K, 3.31 at 16K)


def plain_greedy_next(runner, token, pos):
    """The target's greedy next token after `token` at position pos (eager fused decode, one user)."""
    out = runner.generator.decode_forward(
        torch.tensor([token]),
        torch.tensor([pos]),
        enable_trace=False,
        page_table=runner.page_table,
        kv_cache=runner.kv_cache,
        sampling_params=None,
        reload_inputs=True,
        reload_page_table=False,
        reload_sampling_params=False,
        reset_sampling_state=False,
    )
    logits = out[0] if isinstance(out, tuple) else out
    return int(logits.reshape(-1)[: runner.V].float().argmax())


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("isl", list(ISLS.values()), ids=[f"isl_{k}" for k in ISLS])
@parametrize_dflash_mesh()
def test_dflash_accuracy(mesh_device, device_params, isl):
    fetch_books(isl)
    runner, tokenizer = build_dflash(mesh_device, max(4096, 1 << (isl + NEW_TOKENS + 15).bit_length()))
    prompt_ids = book_prompt(tokenizer, isl)
    try:
        toks, accept, _ = run_dflash(runner, prompt_ids, NEW_TOKENS)
    finally:
        runner.release_traces()
    # (the prompt's K / V are untouched: the passes write positions >= isl, which the decode below rewrites in order)
    plain = [plain_greedy_next(runner, toks[i], isl + i) for i in range(len(toks) - 1)]
    agree = sum(int(p == t) for p, t in zip(plain, toks[1:])) / len(plain)
    mean_accept = sum(accept) / len(accept)
    first_diff = next((i + 1 for i, (p, t) in enumerate(zip(plain, toks[1:])) if p != t), None)
    logger.info(
        f"DFlash vs plain greedy (teacher-forced): {agree:.3f} top-1 agreement over {len(plain)} tokens (first "
        f"difference at token {first_diff}); {mean_accept:.2f} tokens per round"
    )
    assert agree >= MIN_AGREEMENT, f"teacher-forced agreement {agree:.3f} < {MIN_AGREEMENT}"
    assert mean_accept >= MIN_ACCEPT, f"mean accepted tokens per round {mean_accept:.2f} < {MIN_ACCEPT}"
