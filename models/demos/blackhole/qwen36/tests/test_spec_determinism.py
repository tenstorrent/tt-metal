# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Spec decode must be deterministic: same prompt, same model, same tokens every time.

Greedy decoding has no sampling, so two runs of the same request MUST emit identical ids and the
same acceptance rate. This caught a real race: the speculative verify writes the K+1 candidates'
KV with one batched paged_update_cache, but all candidates belong to ONE sequence, so their
consecutive positions share a physical block and several cores read-modify-write the SAME 32-row
tile. Runs then differed in acceptance (2.82 vs 2.61) and diverged from plain greedy at a different
token each time.

Parametrized over a BLOCK_SIZE-aligned prompt (128) and an UNALIGNED one (130). The unaligned length
is the regression guard for the seed forward's KV placement: the eager seed ran through the
masked-bucket prefill's block-aligned paged_fill_cache, so at an anchor that is not a multiple of
BLOCK_SIZE the seed token's K/V landed (anchor % BLOCK_SIZE) slots early and clobbered real prompt KV
in that block. A 128-token prompt divides evenly and hides the bug entirely.

Run: MESH_DEVICE=P150x4 pytest models/demos/blackhole/qwen36/tests/test_spec_determinism.py -v -s
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, _MULTI, BLOCK_SIZE, DEVICE_PARAMS, _get_prompt
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

RUNS = 3
MAX_NEW = 48


def _prompt_of_len(target, tokenizer):
    """Exactly `target` token ids, deterministically. _get_prompt already repeats-and-clips the
    shared 128-token prompt for any length <= 256, but truncate defensively so an unaligned target
    (130) is guaranteed exact — that exactness is the whole point of the 130 case."""
    ids = _get_prompt(target, tokenizer)
    while ids.shape[1] < target:  # belt-and-braces: repeat, then clip
        ids = torch.cat([ids, ids], dim=1)
    ids = ids[:, :target]
    assert ids.shape[1] == target, f"prompt is {ids.shape[1]} tokens, wanted exactly {target}"
    return ids


# 128 is BLOCK_SIZE-aligned; 130 is not (130 % 64 == 2) and exercises the unaligned-anchor seed KV write.
@run_for_blackhole()
@pytest.mark.parametrize("prompt_len", [128, 130])
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_spec_decode_is_deterministic(mesh_device, prompt_len):
    if not _MULTI:
        pytest.skip("spec decode is the TP path; run with MESH_DEVICE=P150x4")
    from transformers import AutoTokenizer

    from models.demos.blackhole.qwen36.tt.spec_decode import SpeculativeDecoder

    device = mesh_device
    device.enable_program_cache()
    num_blocks = 64
    model = Qwen36Model.from_pretrained(device, max_batch_size=1, max_seq_len=num_blocks * BLOCK_SIZE)
    assert model.mtp is not None, "MTP head not built"
    tokenizer = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    token_ids = _prompt_of_len(prompt_len, tokenizer)
    prompt_ids = token_ids[0].tolist()
    kv_shape = [num_blocks, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim]
    pt = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)

    outs, accepts = [], []
    for r in range(RUNS):
        model.free_kv_caches()
        model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)
        dec = SpeculativeDecoder(model, pt)
        outs.append(dec.generate(prompt_ids, MAX_NEW))
        accepts.append(dec.accept_rate())
        logger.info(f"[determinism] prompt_len={prompt_len} run {r}: accept={accepts[-1]:.3f}/{dec.K}")
        model.free_kv_caches()

    for r in range(1, RUNS):
        div = next((i for i in range(MAX_NEW) if outs[r][i] != outs[0][i]), None)
        assert div is None, (
            f"run {r} diverged from run 0 at token {div} (prompt_len={prompt_len}) — "
            f"spec decode is nondeterministic.\n"
            f"run0={outs[0]}\nrun{r}={outs[r]}"
        )
    assert (
        len(set(f"{a:.6f}" for a in accepts)) == 1
    ), f"acceptance varied across runs (prompt_len={prompt_len}): {accepts}"
    logger.info(f"[determinism] prompt_len={prompt_len}: {RUNS} runs identical, accept={accepts[0]:.3f}")
