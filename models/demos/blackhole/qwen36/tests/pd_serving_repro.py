# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""vLLM-like serving sequence in-process: interleaved prompts of different lengths, zero-padded page-table rows
with block ids starting at 1 (block 0 = vLLM null block), padded decode rows (token 0, position -1, zero page
table), traced masked-bucket prefill. Each prompt is prefilled twice at different points of the sequence; the
two logits hashes must agree (history independence) and the first token must match HF's.

Run: MESH_DEVICE=P150x4 TT_VISIBLE_DEVICES=... QWEN36_DET_TRACED=1 pytest .../pd_serving_repro.py -x -s
"""
import hashlib
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, _MULTI, BLOCK_SIZE, DEVICE_PARAMS
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

BMAX = int(os.environ.get("QWEN36_SRV_BMAX", "8"))
NUM_BLOCKS = int(os.environ.get("QWEN36_SRV_BLOCKS", "256"))
PT_WIDTH = 64  # zero-padded page-table row width (vLLM: max_num_blocks_per_req)
DECODE_STEPS = int(os.environ.get("QWEN36_SRV_DECODE_STEPS", "6"))
PROMPTS = [
    "Write a haiku about autumn rain.",
    "Translate to French: The quick brown fox jumps over the lazy dog. Then count the words.",
    "List the planets of the solar system in order from the sun, one line each.",
    "What is the capital of Australia and when was it founded?",
    "Explain the difference between TCP and UDP in three short paragraphs, then give one example protocol built on each and say why it fits.",
]
EXPECTED_FIRST = {
    "Write a haiku about autumn rain.": "Gray",
    "Translate to French: The quick brown fox jumps over the lazy dog. Then count the words.": "Le",
}


class BlockAlloc:
    def __init__(self, n):
        self.free = list(range(n - 2, 0, -1))  # never 0 (null) and never the last block (pad block)

    def take(self, k):
        return [self.free.pop() for _ in range(k)]

    def give(self, ids):
        self.free.extend(sorted(ids, reverse=True))


def _row(blocks):
    r = torch.zeros(PT_WIDTH, dtype=torch.int32)
    r[: len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
    return r


@run_for_blackhole()
@pytest.mark.timeout(3000)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_serving_sequence_is_history_independent(mesh_device):
    if not _MULTI:
        pytest.skip("TP path only")
    from transformers import AutoTokenizer

    device = mesh_device
    device.enable_program_cache()
    model = Qwen36Model.from_pretrained(device, max_batch_size=BMAX, max_seq_len=PT_WIDTH * BLOCK_SIZE)
    tok = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    kv_shape = [NUM_BLOCKS, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim]
    model.free_kv_caches()
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=BMAX)  # pad block = NUM_BLOCKS-1
    if os.environ.get("QWEN36_DET_TRACED", "1") == "1":
        pt_full = torch.arange(NUM_BLOCKS, dtype=torch.int32).reshape(1, -1)
        prev = model._bind_gdn_prefill_scratch()
        try:
            model.capture_prefill_trace_chunked(device, pt_full, chunk_size=2048, capture_chunk_trace=True)
        finally:
            model._unbind_gdn_prefill_scratch(prev)
    alloc = BlockAlloc(NUM_BLOCKS)
    encoded = {}
    for p in PROMPTS:
        ids = tok.apply_chat_template(
            [{"role": "user", "content": p}], add_generation_prompt=True, tokenize=True, enable_thinking=False
        )
        encoded[p] = [int(x) for x in (ids["input_ids"] if hasattr(ids, "keys") else ids)]
    seen = {}
    bad = []
    seq = PROMPTS + PROMPTS[::-1] + PROMPTS  # each prompt 3x, interleaved with the others
    try:
        for it, p in enumerate(seq):
            ids = encoded[p]
            T = len(ids)
            n_blocks = -(-(T + DECODE_STEPS + 1) // BLOCK_SIZE)
            blocks = alloc.take(n_blocks)
            slot = it % BMAX
            pt = _row(blocks).reshape(1, -1)
            lg = model.prefill_paged_slots([torch.tensor([ids], dtype=torch.int32)], pt, [slot], valid_lens=[T])[0]
            lg = lg.reshape(-1)[: model.vocab_size].float()
            h = hashlib.sha1(lg.numpy().tobytes()).hexdigest()[:10]
            first = int(lg.argmax())
            first_s = tok.decode([first])
            # vLLM-like decode steps: only `slot` row is live; other rows padded (token 0, pos -1, zero page table)
            tokens = torch.zeros((BMAX, 1), dtype=torch.int32)
            positions = torch.full((BMAX,), -1, dtype=torch.int32)
            page_tables = torch.zeros((BMAX, PT_WIDTH), dtype=torch.int32)
            page_tables[slot] = _row(blocks)
            tokens[slot, 0], positions[slot] = first, T
            gen = [first]
            for s in range(DECODE_STEPS):
                dev = model.prepare_inputs_decode(tokens, positions, page_tables)
                out, _ = model.ttnn_decode_forward(dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3])
                lgd = model.process_output_decode(out, BMAX)[:, 0, : model.vocab_size].float()
                nxt = int(lgd[slot].argmax())
                gen.append(nxt)
                tokens[slot, 0], positions[slot] = nxt, T + s + 1
            text = tok.decode(gen)
            ref = seen.setdefault(p, (h, first_s, text))
            status = "SAME" if ref[0] == h else f"DIFF (first was {ref[1]!r} hash {ref[0]})"
            exp = EXPECTED_FIRST.get(p)
            hf = "" if exp is None else (" HF-OK" if first_s == exp else f" HF-MISMATCH(expected {exp!r})")
            logger.info(
                f"[srv] it {it:2d} slot {slot} T={T:3d} blocks={blocks[:3]}..: first={first_s!r} hash={h} {status}{hf} :: {text[:50]!r}"
            )
            if ref[0] != h or (exp is not None and first_s != exp):
                bad.append((it, p[:30], status, hf))
            alloc.give(blocks)
    finally:
        model.free_kv_caches()
    assert not bad, f"history-dependent or wrong prefills: {bad}"
