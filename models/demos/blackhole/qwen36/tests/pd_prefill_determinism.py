# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Is the served prefill path (prefill_paged_slots -> traced masked bucket) bit-deterministic? Run it N times on one
prompt and compare the host logits. Run: MESH_DEVICE=P150x4 TT_VISIBLE_DEVICES=... pytest .../pd_prefill_determinism.py -s"""
import hashlib
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, _MULTI, BLOCK_SIZE, DEVICE_PARAMS
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

BMAX, BPU, N = 8, 8, int(os.environ.get("QWEN36_DET_ITERS", "12"))


@run_for_blackhole()
@pytest.mark.timeout(2400)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_prefill_determinism(mesh_device):
    if not _MULTI:
        pytest.skip("TP path only")
    from transformers import AutoTokenizer

    device = mesh_device
    device.enable_program_cache()
    model = Qwen36Model.from_pretrained(device, max_batch_size=BMAX, max_seq_len=BPU * BLOCK_SIZE * 2)
    tok = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    prompts = [
        "Write a haiku about autumn rain.",
        "Translate to French: The quick brown fox jumps over the lazy dog. Then count the words.",
    ]
    page_tables = torch.stack([torch.arange(u * BPU, (u + 1) * BPU, dtype=torch.int32) for u in range(BMAX)])
    kv_shape = [BMAX * BPU, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim]
    model.free_kv_caches()
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=BMAX)
    if os.environ.get("QWEN36_DET_TRACED", "0") == "1":
        # Mirror Qwen36ForCausalLM.warmup_model_prefill: capture the chunk trace + masked-bucket traces against
        # the persistent B=1 scratch, so prefill_paged_slots replays traces exactly as the served path does.
        pt_full = torch.arange(BMAX * BPU, dtype=torch.int32).reshape(1, -1)
        prev = model._bind_gdn_prefill_scratch()
        try:
            model.capture_prefill_trace_chunked(device, pt_full, chunk_size=2048, capture_chunk_trace=True)
        finally:
            model._unbind_gdn_prefill_scratch(prev)
        logger.info(f"[det] traced prefill: mb traces={sorted(model._mb_traces.keys()) if model._mb_traces else None}")
    bad = 0
    try:
        for text in prompts:
            ids = tok.apply_chat_template(
                [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True, enable_thinking=False
            )
            ids = [int(x) for x in (ids["input_ids"] if hasattr(ids, "keys") else ids)]
            ref = None
            for it in range(N):
                slot = it % BMAX
                lg = model.prefill_paged_slots(
                    [torch.tensor([ids], dtype=torch.int32)],
                    page_tables[slot : slot + 1],
                    [slot],
                    valid_lens=[len(ids)],
                )[0]
                lg = lg.reshape(-1)[: model.vocab_size].float()
                h = hashlib.sha1(lg.numpy().tobytes()).hexdigest()[:10]
                top = int(lg.argmax())
                if ref is None:
                    ref = (h, lg.clone())
                diff = float((lg - ref[1]).abs().max())
                same = h == ref[0]
                bad += not same
                logger.info(
                    f"[det] {text[:24]!r} iter {it} slot {slot}: top={tok.decode([top])!r} hash={h} {'SAME' if same else f'DIFF maxabs={diff:.4g}'}"
                )
    finally:
        model.free_kv_caches()
    assert bad == 0, f"{bad} non-deterministic prefills"
