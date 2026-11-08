"""Debug reproduction for BUG 2 / BUG 3 (untracked scratch test)."""
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tests.test_vllm_batched_mm import (
    BLOCK,
    BPU,
    DEVICE_PARAMS,
    MAX_SEQ,
    NUM_BLOCKS,
    WIDTH,
    Driver,
    Req,
    mk_rows,
)
from models.demos.blackhole.qwen36.tt.qwen36_vllm import Qwen36ForCausalLM

PROMPTS = [
    "Explain photosynthesis in detail, step by step.",
    "Write a short story about a robot who learns to paint.",
    "What are the main causes of the French Revolution?",
    "Give me a Python function that computes the Fibonacci sequence and explain it.",
    "Describe the water cycle for a ten year old.",
    "List five tips for improving sleep quality and justify each.",
    "Summarize the plot of Romeo and Juliet in a few paragraphs.",
    "How does a transformer neural network work? Be thorough.",
]
STEPS1, STEPS2 = int(os.environ.get("DBG_S1", 64)), int(os.environ.get("DBG_S2", 32))


def first_div(a, b):
    return next(
        (i for i, (x, y) in enumerate(zip(a, b)) if x != y),
        min(len(a), len(b)) if len(a) == len(b) else min(len(a), len(b)),
    )


@pytest.mark.timeout(3500)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_dbg(mesh_device, reset_seeds, ensure_gc):
    from transformers import AutoConfig, AutoTokenizer

    hf = os.environ["HF_MODEL"]
    tok = AutoTokenizer.from_pretrained(hf)
    reqs = []
    for i, p in enumerate(PROMPTS):
        t = tok.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        ids = tok(t, return_tensors="pt")["input_ids"].to(torch.int32)
        reqs.append(Req(f"p{i}", ids))
    gen = Qwen36ForCausalLM.initialize_vllm_model(
        AutoConfig.from_pretrained(hf), mesh_device, max_batch_size=WIDTH, max_seq_len=MAX_SEQ
    )
    model = gen.model[0]
    kv = gen.allocate_kv_cache(
        (NUM_BLOCKS, model.args.n_local_kv_heads, BLOCK, model.args.head_dim), ttnn.bfloat16, len(model.layers)
    )
    dkw = dict(kv_cache=kv, max_batch_size=WIDTH, num_blocks=BPU, can_sample_on_device=True)
    gen.warmup_model_prefill(kv_cache=kv, enable_trace=False)
    gen.warmup_model_decode(enable_trace=False, **dkw)
    gen.already_warmed_up_prefill = False
    gen.warmup_model_prefill(kv_cache=kv, enable_trace=True)
    gen.warmup_model_decode(enable_trace=True, **dkw)
    logger.info(f"prefix cache: {model._prefix_cache is not None}")
    d = Driver(gen, kv, tok)
    d.blocks = lambda L: _blocks(d, L)

    def alone(i, n):
        lg, bl = d.prefill([reqs[i]], [0])
        rows = mk_rows([reqs[i]], lg, bl)
        for _ in range(n - 1):
            d.decode_step(rows)
        return rows[0]["out"]

    # R1
    P = 0
    runs = [alone(P, STEPS1) for _ in range(4)]
    for k in range(1, 4):
        print(f"R1 run{k} vs run0: first_div={first_div(runs[0], runs[k])}/{STEPS1} identical={runs[0]==runs[k]}")
    print("R1 text:", repr(tok.decode(runs[0], skip_special_tokens=True))[:200])
    # R2 batch
    lg, bl = d.prefill(reqs, list(range(8)))
    rows = mk_rows(reqs, lg, bl)
    for _ in range(STEPS1 - 1):
        d.decode_step(rows)
    keep = [0, 2, 4, 6, 7]
    drop = [1, 3, 5]
    remap = keep + drop + list(range(8, WIDTH))
    surv = [rows[i] for i in keep]
    d.decode_step(surv, remap=remap)
    for _ in range(STEPS2 - 1):
        d.decode_step(surv)
    # pre-condense outputs of dropped (just first STEPS1)
    refs = {i: alone(i, STEPS1 + STEPS2) for i in range(8)}
    refs2 = {i: alone(i, STEPS1 + STEPS2) for i in (0, 4)}
    for i in range(8):
        out = rows[i]["out"]
        r = refs[i]
        n = min(len(out), len(r))
        fd = first_div(out[:n], r[:n])
        print(
            f"R2 user{i} ({'surv' if i in keep else 'dropped'}) len={len(out)} first_div_vs_alone={fd}/{n} :: {repr(tok.decode(out, skip_special_tokens=True))[:90]}"
        )
    for i in (0, 4):
        print(
            f"R2 alone repeat user{i}: identical_to_first_alone={refs2[i]==refs[i]} first_div={first_div(refs2[i], refs[i])}"
        )


def _blocks(d, L):
    n = (L + 200 + BLOCK) // BLOCK + 1
    ids = list(range(d.next_block, d.next_block + n))
    d.next_block += n
    assert d.next_block <= NUM_BLOCKS
    return ids
