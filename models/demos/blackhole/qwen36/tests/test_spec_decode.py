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


class _RecordingHead:
    """Delegating MTP-head wrapper that records every step's live loop inputs.

    The recording feeds the shadow-reference replay below: it separates "the
    drafter is fed wrong inputs" (a loop bug) from "the drafter's predictions
    are poor on these inputs" (model-side, e.g. truncated-backbone hiddens)."""

    def __init__(self, inner):
        self._inner = inner
        self.calls = []  # (token_id, hidden clone, position, tt_logits)

    def step(self, token_id, hidden_row, position):
        logits, hidden = self._inner.step(token_id, hidden_row, position)
        self.calls.append((int(token_id), hidden_row.detach().clone().float(), int(position), logits))
        return logits, hidden

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _load_checkpoint_tensor(ckpt_dir, key):
    import json

    from safetensors import safe_open

    with open(f"{ckpt_dir}/model.safetensors.index.json") as f:
        fname = json.load(f)["weight_map"][key]
    with safe_open(f"{ckpt_dir}/{fname}", framework="pt") as sf:
        return sf.get_tensor(key)


def _shadow_reference_replay(model, mtp_sd, calls):
    """Replay the recorded drafter steps through MTPTorchReference (real shared
    embedding + LM head) and return (per-step logits PCCs, greedy agreement)."""
    from models.demos.blackhole.qwen36.reference.mtp_torch import MTPTorchReference
    from models.demos.blackhole.qwen36.tests.test_factory import compute_pcc

    ckpt = model.args.CKPT_DIR
    ref = MTPTorchReference(
        mtp_sd,
        embed_weight=_load_checkpoint_tensor(ckpt, "model.language_model.embed_tokens.weight"),
        lm_head_weight=_load_checkpoint_tensor(ckpt, "lm_head.weight"),
        num_heads=model.args.n_heads,
        num_kv_heads=model.args.n_kv_heads,
        head_dim=model.args.head_dim,
        rope_head_dim=model.args.rope_head_dim,
        rope_theta=model.args.rope_theta,
        norm_eps=model.args.norm_eps,
        max_seq_len=model.args.max_seq_len,
    )
    pccs, agree = [], 0
    for tok, hid, pos, tt_logits in calls:
        ref_logits, _ = ref.step(tok, hid, pos)
        pccs.append(compute_pcc(ref_logits, tt_logits))
        agree += int(ref_logits.argmax()) == int(tt_logits.argmax())
    return pccs, agree / max(1, len(calls))


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
    recorder = _RecordingHead(mtp)
    spec = Qwen36SpeculativeDecoder(model, recorder, page_table, draft_len=3)
    spec.prefill(token_ids)
    generated, stats = spec.generate(_MAX_NEW)
    mtp.free_kv_cache()

    # Shadow-reference replay of every live drafter step: proves the TT head
    # computed its reference function on the loop's actual inputs, so a low
    # accept rate here is drafter quality (truncated-backbone hiddens are out
    # of distribution for the MTP head), not loop plumbing.
    pccs, agreement = _shadow_reference_replay(model, mtp_sd, recorder.calls)
    logger.info(
        f"shadow drafter replay: {len(pccs)} steps, PCC mean={sum(pccs)/len(pccs):.5f} "
        f"min={min(pccs):.5f}, greedy agreement={agreement:.2%}, accept_rate={stats['accept_rate']:.2f}"
    )

    logger.info(f"plain: {plain}")
    logger.info(f"spec:  {generated} (stats: {stats})")
    n = min(len(plain), len(generated))
    assert generated[:n] == plain[:n], (
        f"spec decode diverged from plain greedy at token "
        f"{next(i for i in range(n) if generated[i] != plain[i])}: spec={generated[:n]} plain={plain[:n]}"
    )
    assert len(generated) >= 1
    assert (
        sum(pccs) / len(pccs) > 0.97
    ), f"drafter deviates from torch reference on live inputs: mean PCC {sum(pccs)/len(pccs)}"
    assert min(pccs) > 0.90, f"drafter step deviates from torch reference: min PCC {min(pccs)}"
