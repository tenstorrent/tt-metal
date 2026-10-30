# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Qwen36P150MTPForCausalLM driven the way the vLLM TT plugin runner drives a block-output model, without a server.

initialize_vllm_model -> allocate_kv_cache -> warmup (eager, then traced) -> per request: prefill_forward, then
decode_forward steps of output_tokens_per_step tokens over a page table that grows like vLLM's -> release_request.
Output must match plain greedy up to the first bf16 near-tie. Covers short prompts (warm drafter) and one prompt
longer than a prefill chunk (chunk-trace prefill, cold drafter).

Run:
  HF_MODEL=Qwen/Qwen3.5-2B MESH_DEVICE=P150 pytest models/demos/blackhole/qwen36/tests/test_spec_p150_vllm.py -s
"""
import math
import os
import time

import pytest
import torch
from transformers import AutoConfig, AutoTokenizer

pytest.importorskip("vllm")

import ttnn  # noqa: E402
from models.demos.blackhole.qwen36.tests.test_spec_p150 import NEAR_TIE, PROMPTS, _chat, _first_diff  # noqa: E402
from models.demos.blackhole.qwen36.tt.qwen36_vllm import Qwen36P150MTPForCausalLM  # noqa: E402
from models.tt_transformers.demo.simple_text_demo import TokenAccuracy  # noqa: E402

BLOCK_SIZE = 64
MAX_SEQ_LEN = 4096
NUM_BLOCKS = MAX_SEQ_LEN // BLOCK_SIZE
LONG_PROMPT = 2300


def _table(blocks, last_pos, lookahead):
    """vLLM-style [1, NUM_BLOCKS] row: blocks up to last_pos + lookahead, zeros (the null block) after."""
    need = min(len(blocks), math.ceil((last_pos + 1 + lookahead) / BLOCK_SIZE))
    return torch.tensor([blocks[:need] + [0] * (NUM_BLOCKS - need)], dtype=torch.int32)


def _greedy(model, ids, pt, n):
    """Plain greedy on the serving prefill (masked bucket / chunk trace) + eager paged decode -> (tokens, top-2 gaps)."""
    vocab = model.args.vocab_size
    lg = ttnn.to_torch(model.prefill_traced_chunked(ids, pt, actual_len=ids.shape[1])).float().reshape(-1)[:vocab]
    out, gaps, pos = [int(lg.argmax())], [None], ids.shape[1]
    while len(out) < n:
        lg = ttnn.to_torch(model.decode_paged(torch.tensor([[out[-1]]]), pos, pt)).float().reshape(-1)[:vocab]
        top = lg.topk(2)
        out.append(int(top.indices[0]))
        gaps.append(float(top.values[0] - top.values[1]))
        pos += 1
    return out, gaps


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, "num_command_queues": 2, "trace_region_size": 256 * 1024 * 1024}],
    indirect=True,
)
def test_spec_p150_vllm(device):
    device.enable_program_cache()
    n_new = int(os.environ.get("SPEC_NEW_TOKENS", "200"))
    hf_config = AutoConfig.from_pretrained(os.environ["HF_MODEL"], trust_remote_code=True)
    gen = Qwen36P150MTPForCausalLM.initialize_vllm_model(hf_config, device, 1, MAX_SEQ_LEN)
    model, args = gen.model[0], gen.model[0].args
    N = gen.model_capabilities["output_tokens_per_step"]
    kv = gen.allocate_kv_cache([NUM_BLOCKS, args.n_kv_heads, BLOCK_SIZE, args.head_dim], ttnn.bfloat16, args.n_layers)

    tok = AutoTokenizer.from_pretrained(args.CKPT_DIR, trust_remote_code=True)
    prompts = [_chat(tok, p) for p in PROMPTS[:8]]
    acc = TokenAccuracy(args.model_name)
    ctx = torch.cat([acc.input_prompt, acc.reference_tokens]).tolist()
    long_ids = (ctx * math.ceil(LONG_PROMPT / len(ctx)))[:LONG_PROMPT]
    prompts.append(_chat(tok, tok.decode(long_ids) + "\n\nSummarize the text above in a few sentences."))

    # References before warmup: eager decode compiles programs, which must not happen once a trace is recorded.
    # The long prompt's plain prefill needs the chunk trace, so it is only checked for coherent output.
    ref_pt = torch.arange(NUM_BLOCKS, dtype=torch.int32).unsqueeze(0)
    refs = [_greedy(model, ids, ref_pt, n_new) for ids in prompts[:-1]] + [None]

    gen.warmup_model_prefill(kv, False)
    gen.warmup_model_prefill(kv, True)
    gen.warmup_model_decode(kv, True, 1, NUM_BLOCKS, True)

    bad, tot_tokens, tot_s = [], 0, 0.0
    for i, (ids, ref) in enumerate(zip(prompts, refs)):
        g = torch.Generator().manual_seed(i)
        blocks = (torch.randperm(NUM_BLOCKS - 1, generator=g) + 1).tolist()  # block 0 is vLLM's null block
        S = ids.shape[1]
        logits, _ = gen.prefill_forward(ids, _table(blocks, S - 1, 2 * N), kv, [S], empty_slots=[0])
        out, pos, steady = [int(logits.reshape(-1).argmax())], S, None
        while len(out) < n_new:
            if len(out) > 1 and steady is None:  # first decode step also captures the request's trace
                steady = (time.perf_counter(), len(out))
            row = gen.read_decode_output(
                gen.decode_forward(
                    tokens=torch.tensor([[out[-1]]]),
                    start_pos=torch.tensor([pos]),
                    page_table=_table(blocks, pos, 2 * N),
                    kv_cache=kv,
                    enable_trace=True,
                    read_from_device=True,
                )
            )
            assert tuple(row.shape) == (1, N), f"decode row {tuple(row.shape)} != (1, {N})"
            out.extend(row[0].tolist())
            pos += N
        dt, n_steady = time.perf_counter() - steady[0], len(out) - steady[1]
        gen.release_request(0)
        out = out[:n_new]
        tot_tokens, tot_s = tot_tokens + n_steady, tot_s + dt
        if ref is None:
            print(f"  prompt {i} ({S:4d} tok): {n_steady / dt:.1f} tok/s", flush=True)
            print(f"     served: {tok.decode(out[:80])!r}", flush=True)
            continue
        ref, gaps = ref
        d = _first_diff(ref, out)
        ok = d == n_new or gaps[d] <= NEAR_TIE
        bad += [] if ok else [i]
        print(
            f"  prompt {i} ({S:4d} tok): matches greedy for {d:3d}/{n_new}"
            + (f" (diverges at gap {gaps[d]:.3f})" if d < n_new else "")
            + f", {n_steady / dt:.1f} tok/s"
            + ("" if ok else "  <-- NOT A NEAR-TIE"),
            flush=True,
        )
    gen.release_persistent_capture()
    print(f"  overall decode: {tot_tokens / tot_s:.1f} tok/s ({N} tokens per vLLM step)", flush=True)
    assert not bad, f"prompts {bad} diverge from greedy where greedy is not a near-tie"
