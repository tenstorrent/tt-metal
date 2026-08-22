# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Speculative decode vs plain greedy decode — single device, truncated model.

Spec-decode correctness does not depend on drafter quality (committed tokens
always come from the target verify rows), so an n_layers-truncated backbone is
a valid equivalence harness: spec output must match plain greedy decode up to
bf16 chunk-vs-decode numerics (the verify runs the masked chunk path, plain
decode the T=1 path — a near-tie top-2 logit gap can flip a token; the assert
follows gemma4's test_spec_decode_matches_greedy and requires exact match).

Needs a checkpoint with mtp.* weights (3.8 family):
    HF_MODEL=/path/to/qwen38-27b pytest models/demos/blackhole/qwen36/tests/test_spec_decode.py -v
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.blackhole.qwen36.tt.weight_mapping import checkpoint_has_mtp, load_qwen36_mtp_state_dict

DEVICE_PARAMS = [{"l1_small_size": 24576, "num_command_queues": 2}]
pytestmark = [run_for_blackhole(), pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)]

_N_LAYERS = 8  # truncated backbone: 6 GDN + 2 full-attn (layers 3, 7)
_BLOCK_SIZE = 64
_NUM_BLOCKS = 32  # 2048-token budget
_PROMPT_LEN = 100  # deliberately not block-aligned
_MAX_NEW = 32


def _plain_greedy(model, token_ids, page_table, max_new):
    logits = model.prefill_paged(token_ids, page_table)
    next_token = int(ttnn.to_torch(logits).squeeze().float().argmax())
    generated = [next_token]
    T = token_ids.shape[1]
    for i in range(max_new - 1):
        out = model.decode_paged(torch.tensor([[next_token]], dtype=torch.long), T + i, page_table)
        next_token = int(ttnn.to_torch(out).squeeze().float().argmax())
        generated.append(next_token)
    return generated


def test_spec_decode_matches_greedy(device, request):
    from models.demos.blackhole.qwen36.tt.mtp import Qwen36MTPHead
    from models.demos.blackhole.qwen36.tt.spec_decode import Qwen36SpeculativeDecoder

    model = Qwen36Model.from_pretrained(device, max_seq_len=_NUM_BLOCKS * _BLOCK_SIZE, n_layers=_N_LAYERS)
    if not checkpoint_has_mtp(model.args.CKPT_DIR):
        pytest.skip(f"checkpoint {model.args.CKPT_DIR} has no mtp.* weights")

    torch.manual_seed(7)
    token_ids = torch.randint(10, 1000, (1, _PROMPT_LEN), dtype=torch.long)
    page_table = torch.arange(_NUM_BLOCKS, dtype=torch.int32).unsqueeze(0)

    kv_shape = [_NUM_BLOCKS, model.args.n_kv_heads, _BLOCK_SIZE, model.args.head_dim]
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)
    plain = _plain_greedy(model, token_ids, page_table, _MAX_NEW)

    # Fresh caches for the spec run (spec.prefill re-zeros GDN itself).
    model.free_kv_caches()
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)

    mtp_sd = load_qwen36_mtp_state_dict(model.args.CKPT_DIR)
    mtp = Qwen36MTPHead(
        device,
        model.args,
        mtp_sd,
        embedding=model.embd,
        lm_head_weight=model.lm_head_weight,
        rope=model.rope,
    )
    mtp.allocate_kv_cache(_NUM_BLOCKS)
    spec = Qwen36SpeculativeDecoder(model, mtp, page_table, draft_len=3)
    spec.prefill(token_ids)
    generated, stats = spec.generate(_MAX_NEW)
    mtp.free_kv_cache()

    logger.info(f"plain: {plain}")
    logger.info(f"spec:  {generated} (stats: {stats})")
    n = min(len(plain), len(generated))
    assert generated[:n] == plain[:n], (
        f"spec decode diverged from plain greedy at token "
        f"{next(i for i in range(n) if generated[i] != plain[i])}: spec={generated[:n]} plain={plain[:n]}"
    )
    assert len(generated) >= 1
