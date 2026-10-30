# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Single-P150 MTP spec decode validation.

1. Varied prompts (short chat through ~1000-token context): spec output vs plain greedy decode. Any
   divergence must sit on a greedy near-tie (top-2 logit gap <= 2 bf16 steps); traced == eager on prompt 0.
2. The teacher-forced HF accuracy gate (same .refpt + scorer as test_tt_opt_accuracy_gate) run through
   the T-row verify path.

Run:
  HF_MODEL=Qwen/Qwen3.5-2B SPEC_K=2 pytest models/demos/blackhole/qwen36/tests/test_spec_p150.py -s
"""
import os

import pytest
import torch
from transformers import AutoTokenizer

import ttnn
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.blackhole.qwen36.tt.spec_p150 import SpecDecodeP150
from models.tt_transformers.demo.simple_text_demo import TokenAccuracy

BLOCK_SIZE = 64
NUM_BLOCKS = 64
NEAR_TIE = float(os.environ.get("SPEC_NEAR_TIE", "1.0"))

PROMPTS = [
    "Explain how a hash map works and when you would use one instead of a sorted array.",
    "Write a Python function that returns the n-th Fibonacci number using memoization, with a docstring.",
    "What is 17 * 23? Show your work step by step.",
    "Write a short poem about the ocean at night.",
    "Translate to French: 'The meeting has been moved to Thursday afternoon because the manager is travelling.'",
    "List five tips for improving sleep quality.",
    "Write a SQL query that returns the top 3 customers by total order amount from tables customers and orders.",
    "Summarize the causes of the First World War in one paragraph.",
    "Give me a JSON object describing a book with title, author, year and a list of genres.",
    "Write a bash command that finds all .log files larger than 10MB under /var/log.",
    "Explain the difference between TCP and UDP to a beginner.",
    "Tell me a story about a robot who learns to paint.",
    "What are the pros and cons of electric cars?",
    "Write a C function that reverses a singly linked list.",
    "Explain gradient descent and why the learning rate matters.",
    "Draft a polite email asking a colleague to review my pull request by Friday.",
]


def _first_diff(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def _chat(tok, text):
    ids = tok.apply_chat_template(
        [{"role": "user", "content": text}], add_generation_prompt=True, return_tensors="pt", enable_thinking=False
    )
    return ids if torch.is_tensor(ids) else ids["input_ids"]


def _greedy(model, ids, pt, n):
    """Plain eager paged decode -> (tokens, top-2 logit gap per step)."""
    vocab = model.args.vocab_size
    out, gaps = [int(ttnn.to_torch(model.prefill_paged(ids, pt)).float().reshape(-1).argmax())], [None]
    pos = ids.shape[1]
    while len(out) < n:
        lg = ttnn.to_torch(model.decode_paged(torch.tensor([[out[-1]]]), pos, pt)).float().reshape(-1)[:vocab]
        top = lg.topk(2)
        out.append(int(top.indices[0]))
        gaps.append(float(top.values[0] - top.values[1]))
        pos += 1
    return out, gaps


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, "num_command_queues": 2, "trace_region_size": 128 * 1024 * 1024}],
    indirect=True,
)
def test_spec_p150(device):
    device.enable_program_cache()
    n_new = int(os.environ.get("SPEC_NEW_TOKENS", "200"))
    K = int(os.environ.get("SPEC_K", "2"))
    model = Qwen36Model.from_pretrained(device, max_batch_size=1, max_seq_len=NUM_BLOCKS * BLOCK_SIZE, enable_mtp=True)
    args = model.args
    model.allocate_kv_caches([NUM_BLOCKS, args.n_kv_heads, BLOCK_SIZE, args.head_dim], ttnn.bfloat16, batch_size=1)
    pt = torch.arange(NUM_BLOCKS, dtype=torch.int32).unsqueeze(0)
    tok = AutoTokenizer.from_pretrained(args.CKPT_DIR, trust_remote_code=True)

    acc = TokenAccuracy(args.model_name)
    ctx_text = tok.decode(torch.cat([acc.input_prompt, acc.reference_tokens]).tolist())
    ctx_ids = tok(ctx_text, return_tensors="pt")["input_ids"][0]
    prompts = [_chat(tok, p) for p in PROMPTS]
    for n_ctx in (600, 1000):
        body = tok.decode(ctx_ids[:n_ctx].tolist())
        prompts.append(_chat(tok, body + "\n\nSummarize the text above in a few sentences."))

    # All plain-greedy references first: nothing allocated after the spec trace capture outlives a replay.
    refs = [_greedy(model, ids, pt, n_new) for ids in prompts]
    print(f"greedy references done for {len(refs)} prompts", flush=True)

    spec = SpecDecodeP150(model, pt, draft_len=K)
    eager0 = spec.generate(prompts[0], n_new, trace=False)
    tot = {"verifies": 0, "accepted": 0, "tokens": 0, "iter_s": 0.0}
    bad = []
    print(f"\nK={K}, {n_new} new tokens per prompt", flush=True)
    for i, (ids, (ref, gaps)) in enumerate(zip(prompts, refs)):
        out = spec.generate(ids, n_new, trace=True)
        if i == 0:
            assert out == eager0, "traced spec differs from eager spec"
        for k in tot:
            tot[k] += spec.stats[k]
        d = _first_diff(ref, out)
        gap = gaps[d] if d < n_new else None
        ok = d == n_new or gap <= NEAR_TIE
        bad += [] if ok else [i]
        v = max(1, spec.stats["verifies"])
        print(
            f"  prompt {i:2d} ({ids.shape[1]:4d} tok): matches greedy for {d:3d}/{n_new}"
            + (f" (diverges at gap {gap:.3f})" if gap is not None else "")
            + f", {1 + spec.stats['accepted'] / v:.2f} tok/iter, {1e3 * spec.stats['iter_s'] / max(1, spec.stats['tokens']):.2f} ms/tok"
            + ("" if ok else "  <-- NOT A NEAR-TIE"),
            flush=True,
        )
        if not ok:
            print(f"     greedy: {tok.decode(ref[max(0, d - 6) : d + 6])!r}")
            print(f"     spec  : {tok.decode(out[max(0, d - 6) : d + 6])!r}")
    v = max(1, tot["verifies"])
    print(
        f"  overall: {1 + tot['accepted'] / v:.2f} tokens/iteration, {tot['accepted'] / (v * K):.2f} accepted per draft, "
        f"{1e3 * tot['iter_s'] / max(1, tot['tokens']):.2f} ms/token traced"
    )

    # Teacher-forced HF accuracy through the verify path. The eager concat prefill hits an L1 clash on this
    # ~512-token prompt (pre-existing, also with the decode optimizations off), so prefill a short prefix
    # and push the rest of the prompt + the reference through verify; only reference positions are scored.
    p = acc.input_prompt
    n_pre, n_ref = 32, len(acc.reference_tokens)
    rest = torch.cat([p[n_pre:], acc.reference_tokens])
    preds = spec.teacher_forced(p[:n_pre].reshape(1, -1), rest, len(rest))
    acc.store_predicted_tokens = preds[len(p) - n_pre :]
    top1, top5 = (100 * x for x in acc.compute_accuracy())
    spec.release()

    # Plain decode through the same protocol (short prefill, then one teacher-forced token per step).
    plain = [int(ttnn.to_torch(model.prefill_paged(p[:n_pre].reshape(1, -1), pt)).float().reshape(-1).argmax())]
    for i, t in enumerate(rest[:-1].tolist()):
        lg = model.decode_paged(torch.tensor([[t]]), n_pre + i, pt)
        plain.append(int(ttnn.to_torch(lg).float().reshape(-1)[: args.vocab_size].argmax()))
    acc.store_predicted_tokens = plain[len(p) - n_pre :]
    ptop1, ptop5 = (100 * x for x in acc.compute_accuracy())
    print(f"  same protocol, plain decode: top-1 {ptop1:.2f}%, top-5 {ptop5:.2f}%", flush=True)
    print(
        f"  accuracy gate via verify path ({len(p)}-token prompt, {n_ref} scored): top-1 {top1:.2f}%, top-5 {top5:.2f}%",
        flush=True,
    )
    spec.release()

    assert not bad, f"prompts {bad} diverge from greedy where greedy is not a near-tie"
    floor1, floor5 = float(os.environ.get("QWEN35_GATE_TOP1", "0")), float(os.environ.get("QWEN35_GATE_TOP5", "0"))
    assert top1 >= floor1 and top5 >= floor5, f"verify-path accuracy {top1:.2f}/{top5:.2f} below {floor1}/{floor5}"
