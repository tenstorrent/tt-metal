# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DEVICE test of vLLM chunked prefill (supports_chunked_prefill) driven like the vLLM TT plugin.

The plugin splits a prompt longer than the per-step token budget across engine steps: every chunk's row holds the
prompt from position 0, prompt_lens = the chunk END, start_pos = the tokens already prefilled; non-final chunk ends are
128-aligned, the final chunk may be a single token, and the decode slot may change between chunks. Each chunked
prefill is compared with a REFERENCE single-shot prefill of the same tokens (fresh blocks): last-token logits PCC, top-1,
then greedy decode tokens side by side. One scenario runs decode steps of another request between the chunks.
Chunked prefill is opt-in; the test turns it on itself (QWEN36_CHUNKED_PREFILL=1).

    pytest models/demos/blackhole/qwen36/tests/test_chunked_prefill.py -svq
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.sampling.sampling_params import SamplingParams
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.tests.test_prefix_cache import (
    BLOCK,
    BPU,
    DEVICE_PARAMS,
    MAX_SEQ,
    NUM_BLOCKS,
    WIDTH,
    Harness,
    _corpus,
    pcc,
)
from models.demos.blackhole.qwen36.tt.qwen36_vllm import Qwen36ForCausalLM, _build_model_capabilities

PCC_MIN = 0.99


def prefill_chunks(h, toks, blocks, bounds, slots):
    """Prefill toks over the chunk ends `bounds` (last == len(toks)); chunk i goes to decode slot slots[i]. Returns the
    final chunk's logits and the total wall time."""
    pt = torch.zeros(1, BPU, dtype=torch.int32)
    pt[0, : len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
    start, logits, t = 0, None, 0.0
    for end, slot in zip(bounds, slots):
        row = torch.tensor(toks[:end], dtype=torch.int32).reshape(1, end)
        t0 = time.perf_counter()
        out, _ = h.gen.prefill_forward(row, pt, h.kv, [end], start_pos=[start], empty_slots=[slot], enable_trace=True)
        t += time.perf_counter() - t0
        logits, start = out.reshape(-1)[: h.vocab].float(), end
    return logits, t


def decode(h, rows, steps):
    """Greedy decode; rows[i] (row i == decode slot i, front-packed) = dict(blocks, pos, last), advanced in place."""
    outs = [[] for _ in rows]
    sp = SamplingParams(temperature=[0.0] * WIDTH, top_k=[1] * WIDTH, top_p=[1.0] * WIDTH, seed=[None] * WIDTH)
    for _ in range(steps):
        tokens = torch.zeros(WIDTH, 1, dtype=torch.int32)
        start = torch.full((WIDTH,), -1, dtype=torch.int32)
        pt = torch.zeros(WIDTH, BPU, dtype=torch.int32)
        for i, r in enumerate(rows):
            tokens[i, 0], start[i] = r["last"], r["pos"]
            pt[i, : len(r["blocks"])] = torch.tensor(r["blocks"], dtype=torch.int32)
        out = h.gen.decode_forward(
            tokens=tokens,
            start_pos=start,
            page_table=pt,
            kv_cache=h.kv,
            sampling_params=sp,
            reload_inputs=True,
            reload_page_table=False,
            reload_sampling_params=True,
            reset_sampling_state=True,
            enable_trace=True,
            read_from_device=True,
        )
        toks = torch.as_tensor(out[0] if isinstance(out, tuple) else out).reshape(-1)
        for i, r in enumerate(rows):
            r["last"], r["pos"] = int(toks[i]), r["pos"] + 1
            outs[i].append(r["last"])
    return outs


@run_for_blackhole()
@pytest.mark.timeout(3500)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_chunked_prefill_serving(mesh_device, reset_seeds, ensure_gc, monkeypatch):
    from transformers import AutoConfig, AutoTokenizer

    monkeypatch.setenv("QWEN36_CHUNKED_PREFILL", "1")  # opt-in; read by initialize_vllm_model / warmup

    hf = os.environ["HF_MODEL"]
    tok = AutoTokenizer.from_pretrained(hf)
    hf_config = AutoConfig.from_pretrained(hf)
    gen = Qwen36ForCausalLM.initialize_vllm_model(hf_config, mesh_device, max_batch_size=WIDTH, max_seq_len=MAX_SEQ)
    model = gen.model[0]
    kv_shape = (NUM_BLOCKS, model.args.n_local_kv_heads, BLOCK, model.args.head_dim)
    kv = gen.allocate_kv_cache(kv_shape, ttnn.bfloat16, len(model.layers))
    dkw = dict(kv_cache=kv, max_batch_size=WIDTH, num_blocks=BPU, can_sample_on_device=True)
    gen.warmup_model_prefill(kv_cache=kv, enable_trace=False)
    gen.warmup_model_decode(enable_trace=False, **dkw)
    gen.already_warmed_up_prefill = False
    gen.warmup_model_prefill(kv_cache=kv, enable_trace=True)
    gen.warmup_model_decode(enable_trace=True, **dkw)
    store = model._chunk_state_cache
    assert store is not None, "the chunked-prefill continuation store was not built by warmup_model_prefill"
    assert _build_model_capabilities()["supports_chunked_prefill"]

    h = Harness(gen, kv, _corpus(tok))
    failures, results = [], []

    def compare(tag, lc, lr, tc, tr):
        p = pcc(lc, lr)
        t1 = int(lc.argmax()) == int(lr.argmax())
        results.append((tag, round(p, 6), t1, round(tc, 3), round(tr, 3)))
        logger.info(f"RESULT {tag}: pcc={p:.6f} top1={t1} t_chunked={tc:.3f}s t_ref={tr:.3f}s")
        if not (p >= PCC_MIN and t1):
            failures.append(f"{tag}: pcc={p:.6f} top1_equal={t1}")

    def same_tokens(tag, a, b):
        results.append((tag, a, b))
        logger.info(f"RESULT {tag}: chunked={a} ref={b}")
        if a != b:
            failures.append(f"{tag}: decode tokens differ chunked={a} ref={b}")

    # ---- C1: three aligned chunks (slot 0) vs a single-shot reference (slot 1), then decode both side by side
    P = h.text(10000)
    pb, rb = h.blocks_for(len(P)), h.blocks_for(len(P))
    hits0 = store.stats["hits"]
    lc, tc = prefill_chunks(h, P, pb, [4096, 8192, len(P)], [0, 0, 0])
    lr, tr = prefill_chunks(h, P, rb, [len(P)], [1])
    compare("C1 3 chunks L=10000", lc, lr, tc, tr)
    if store.stats["hits"] - hits0 != 2:
        failures.append(f"C1: expected 2 continuation hits, got {store.stats['hits'] - hits0}")
    outs = decode(
        h, [dict(blocks=pb, pos=len(P), last=int(lc.argmax())), dict(blocks=rb, pos=len(P), last=int(lr.argmax()))], 8
    )
    same_tokens("C1 decode8", outs[0], outs[1])

    # ---- C2: APC snapshot cache detached: the continuation store alone carries the GDN state
    apc = model._prefix_cache
    model._prefix_cache = None
    try:
        P2 = h.text(7000)
        hits0 = store.stats["hits"]
        lc, tc = prefill_chunks(h, P2, h.blocks_for(len(P2)), [2688, len(P2)], [2, 2])
        lr, tr = prefill_chunks(h, P2, h.blocks_for(len(P2)), [len(P2)], [2])
        compare("C2 no-APC 2 chunks L=7000", lc, lr, tc, tr)
        if store.stats["hits"] - hits0 != 1:
            failures.append(f"C2: expected 1 continuation hit, got {store.stats['hits'] - hits0}")
    finally:
        model._prefix_cache = apc

    # ---- C3: decode of another request between the chunks; odd 128-multiple chunks, a 1-token final chunk, and the
    # decode slot changing for the final chunk (5 -> 1)
    Q = h.text(600)
    qa, qb = h.blocks_for(len(Q)), h.blocks_for(len(Q))
    lq, _ = prefill_chunks(h, Q, qa, [len(Q)], [0])
    q_ref = decode(h, [dict(blocks=qa, pos=len(Q), last=int(lq.argmax()))], 12)[0]  # uninterrupted reference
    lq2, _ = prefill_chunks(h, Q, qb, [len(Q)], [0])
    q_row = dict(blocks=qb, pos=len(Q), last=int(lq2.argmax()))
    R = h.text(5249)
    rbk = h.blocks_for(len(R))
    bounds, slots = [1280, 3456, 5248, 5249], [5, 5, 5, 1]
    pt = torch.zeros(1, BPU, dtype=torch.int32)
    pt[0, : len(rbk)] = torch.tensor(rbk, dtype=torch.int32)
    q_out, start, tc = [], 0, 0.0
    for i, (end, slot) in enumerate(zip(bounds, slots)):
        row = torch.tensor(R[:end], dtype=torch.int32).reshape(1, end)
        t0 = time.perf_counter()
        out, _ = gen.prefill_forward(row, pt, kv, [end], start_pos=[start], empty_slots=[slot], enable_trace=True)
        tc += time.perf_counter() - t0
        start = end
        if i < len(bounds) - 1:
            q_out += decode(h, [q_row], 4)[0]  # the plugin's interleaved decode step(s)
    lc = out.reshape(-1)[: h.vocab].float()
    same_tokens("C3 interleaved decode12", q_out, q_ref)
    rr = h.blocks_for(len(R))
    lr, tr = prefill_chunks(h, R, rr, [len(R)], [2])
    compare("C3 4 chunks L=5249 (1-token final, slot 5->1, decode between)", lc, lr, tc, tr)
    outs = decode(
        h,
        [
            q_row,
            dict(blocks=rbk, pos=len(R), last=int(lc.argmax())),
            dict(blocks=rr, pos=len(R), last=int(lr.argmax())),
        ],
        8,
    )
    same_tokens("C3 decode8", outs[1], outs[2])

    # ---- C4: with a known per-step budget only a full step snapshots (a short aligned prompt does not)
    budget0 = model._chunk_budget
    model._chunk_budget = (4096, 0, 128)
    try:
        saves0 = store.stats["saves"]
        S = h.text(1024)
        prefill_chunks(h, S, h.blocks_for(len(S)), [len(S)], [2])
        if store.stats["saves"] != saves0:
            failures.append("C4: a 1024-token prompt in a 4096-token step must not snapshot")
        P4 = h.text(6000)
        hits0, saves0 = store.stats["hits"], store.stats["saves"]
        lc, tc = prefill_chunks(h, P4, h.blocks_for(len(P4)), [4096, len(P4)], [2, 2])
        lr, tr = prefill_chunks(h, P4, h.blocks_for(len(P4)), [len(P4)], [2])
        compare("C4 budget=4096 2 chunks L=6000", lc, lr, tc, tr)
        if (store.stats["saves"] - saves0, store.stats["hits"] - hits0) != (1, 1):
            failures.append(
                f"C4: expected 1 save + 1 hit, got {store.stats['saves'] - saves0} + {store.stats['hits'] - hits0}"
            )
    finally:
        model._chunk_budget = budget0

    print("\n==== chunked-prefill summary ====")
    for r in results:
        print(r)
    print("continuation store stats", store.stats)
    assert not failures, "; ".join(failures)
