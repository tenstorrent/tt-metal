# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""In-process reproduction of the vLLM plain-serving path at a bucketed decode width.

vLLM (max_num_seqs = 8, decode bucketing on) prefills a request into its decode slot with prefill_paged_slots and then
decodes at the smallest power-of-two width >= the number of live requests: ONE request decodes at width 1 while the GDN
state buffers stay [8, ...]. This test does exactly that on the model API and compares the greedy continuation of slot 0
decoded at width 1 against the same continuation decoded at the full width 8 (rows 1..7 carry dummy users). The two must
agree (a bucketed step is defined as "the same rows of the full-width step"); the decoded text is printed so a garbage
trajectory is visible even where the two disagree only late.

Run: MESH_DEVICE=P150x4 pytest models/demos/blackhole/qwen36/tests/plain_bucket_repro.py -x -s
Env: QWEN36_REPRO_STEPS (24), QWEN36_REPRO_WIDTHS ("1,2,4"), QWEN36_REPRO_PROMPT_TOKENS (0 = a chat prompt).
"""
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, _MULTI, BLOCK_SIZE, DEVICE_PARAMS
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

BMAX = 8
STEPS = int(os.environ.get("QWEN36_REPRO_STEPS", "24"))
WIDTHS = [int(w) for w in os.environ.get("QWEN36_REPRO_WIDTHS", "1,2,4").split(",")]
NUM_BLOCKS_PER_USER = 8  # 512 positions per user: plenty for a chat prompt + STEPS


def _prefill_slot0(model, ids, page_tables):
    """The vLLM batched prefill of one request into decode slot 0; returns the greedy first token."""
    T = len(ids)
    logits = model.prefill_paged_slots([torch.tensor([ids], dtype=torch.int32)], page_tables[0:1], [0], valid_lens=[T])
    lg = logits[0].reshape(-1)[: model.vocab_size].float()
    return int(lg.argmax())


def _decode_row0(model, width, first, T, page_tables, steps):
    """Greedy-decode slot 0 for ``steps`` steps at decode width ``width`` (rows 1..width-1 = dummy users that keep
    feeding a fixed token at a fixed position; their state rows are whatever they are -- don't care)."""
    out_ids, tok = [], first
    for s in range(steps):
        tokens = torch.zeros((width, 1), dtype=torch.int32)
        positions = torch.zeros((width,), dtype=torch.int32)
        tokens[0, 0], positions[0] = tok, T + s
        for u in range(1, width):
            tokens[u, 0], positions[u] = 100 + u, 64  # dummy rows: some token at some position inside their own blocks
        dev = model.prepare_inputs_decode(tokens, positions, page_tables[:width])
        out, _ = model.ttnn_decode_forward(dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3])
        lg = model.process_output_decode(out, width)[:, 0, : model.vocab_size].float()
        tok = int(lg[0].argmax())
        out_ids.append(tok)
    return out_ids


@run_for_blackhole()
@pytest.mark.timeout(2400)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_bucketed_plain_decode_matches_full_width_after_slot_prefill(mesh_device):
    if not _MULTI:
        pytest.skip("TP path only")
    from transformers import AutoTokenizer

    device = mesh_device
    device.enable_program_cache()
    model = Qwen36Model.from_pretrained(device, max_batch_size=BMAX, max_seq_len=NUM_BLOCKS_PER_USER * BLOCK_SIZE * 2)
    tok = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    n_tok = int(os.environ.get("QWEN36_REPRO_PROMPT_TOKENS", "0"))
    if n_tok:
        ids = [int(x) for x in torch.randint(1000, 50000, (n_tok,))]
    else:
        text = (
            os.environ.get("QWEN36_REPRO_PROMPT_TEXT")
            or "List the planets of the solar system in order from the sun, one line each."
        )
        msgs = [{"role": "user", "content": text}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, enable_thinking=False)
        if isinstance(ids, dict) or hasattr(ids, "keys"):  # newer transformers return a BatchEncoding
            ids = list(ids["input_ids"])
        ids = [int(x) for x in ids]
    T = len(ids)
    bpu = NUM_BLOCKS_PER_USER
    page_tables = torch.stack([torch.arange(u * bpu, (u + 1) * bpu, dtype=torch.int32) for u in range(BMAX)])
    kv_shape = [BMAX * bpu, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim]
    model.free_kv_caches()
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=BMAX)
    try:
        # Full-width reference first (this is what a server with decode bucketing OFF runs).
        first = _prefill_slot0(model, ids, page_tables)
        ref = _decode_row0(model, BMAX, first, T, page_tables, STEPS)
        logger.info(f"[repro] width-{BMAX}: {tok.decode([first] + ref)!r}")
        results = {}
        for w in WIDTHS:
            f2 = _prefill_slot0(model, ids, page_tables)  # restore slot 0's state + KV, then decode narrow
            assert f2 == first, f"prefill is not deterministic: first token {f2} vs {first}"
            got = _decode_row0(model, w, first, T, page_tables, STEPS)
            same = sum(1 for a, b in zip(got, ref) if a == b)
            results[w] = (got, same)
            logger.info(f"[repro] width-{w}: {tok.decode([first] + got)!r} -- {same}/{STEPS} tokens match width-{BMAX}")
        out = os.environ.get("QWEN36_REPRO_OUT")
        if out:
            import json

            json.dump(
                {
                    "first": first,
                    "ref": ref,
                    "widths": {w: g for w, (g, s) in results.items()},
                    "text_ref": tok.decode([first] + ref),
                },
                open(out, "w"),
            )
        bad = {w: s for w, (g, s) in results.items() if s < STEPS - 2}  # allow a late bf16 near-tie flip or two
        assert not bad, f"bucketed decode diverges from the full-width decode after a slot prefill: matches {bad}"
    finally:
        model.free_kv_caches()
