# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The vllm-tt-plugin contract of ``DeepseekV41ForCausalLM`` on the Galaxy, driven as the plugin's model runner
would, without vLLM: build (``initialize_vllm_model``), ``allocate_kv_cache``, the two-phase warm-up, a two-request
prefill batch, decode steps on the batch, then a third request that takes a slot a finished request held (the
captures made on the slot serve the new request, an odd prompt whose compressor carry must reach the odd capture).

Request 0 is the text the model knows by heart (A Tale of Two Cities), teacher-forced on its next tokens and judged
by ``book_top1`` as the decode test; request 1 is a different stretch of the same text, free-running, present so the
batch has two rows. ``V41_GEN_STEPS`` decode steps (default 16), ``V41_GEN_PROMPT`` tokens (255), ``V41_GEN_LAYERS``
(``all``; the generator reads it as ``TT_V41_LAYERS``). Results go to the log as ``V41_GENERATOR_RESULT {json}`` and to ``V41_GEN_OUT``.
"""

import json
import os
import time

import pytest
import torch
from loguru import logger

from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import oracle as orc
from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig as C
from models.demos.deepseek_v3_d_p.tests.v41.galaxy_meshes import galaxy_meshes
from models.demos.deepseek_v3_d_p.tests.v41.test_v41_decode import TRACE_REGION
from models.demos.deepseek_v3_d_p.tests.v41.test_v41_depth import _dram_bytes, parse_layers
from models.demos.deepseek_v3_d_p.tt.v41.generator_vllm import DeepseekV41ForCausalLM
from models.demos.deepseek_v3_d_p.tt.v41.weights import resolve_checkpoint


def _page_table(first_blocks: list[int], width: int = 8) -> torch.Tensor:
    table = torch.zeros(len(first_blocks), width, dtype=torch.int32)
    for i, b in enumerate(first_blocks):
        table[i, 0] = b
    return table


@pytest.mark.timeout(int(os.environ.get("V41_GEN_TIMEOUT", 2 * 3600)))
@pytest.mark.parametrize("mesh_device, device_params", galaxy_meshes(trace_region_size=TRACE_REGION), indirect=True)
def test_v41_generator_vllm(mesh_device, device_params, monkeypatch):
    layers = parse_layers(os.environ.get("V41_GEN_LAYERS", "all"))
    prompt = int(os.environ.get("V41_GEN_PROMPT", 255))
    steps = int(os.environ.get("V41_GEN_STEPS", 16))
    chunk = int(os.environ.get("TT_V41_CHUNK", 1024))
    monkeypatch.setenv("TT_V41_CHUNK", str(chunk))
    monkeypatch.setenv("TT_V41_LAYERS", ",".join(str(l) for l in layers))
    if resolve_checkpoint() is None:
        pytest.skip("V4.1 checkpoint shards not downloaded")
    max_seq_len = prompt + steps + 64
    result = {
        "layers": layers,
        "n_layers": len(layers),
        "prompt": prompt,
        "steps": steps,
        "chunk": chunk,
        "max_model_len": max_seq_len,
        "mesh": list(mesh_device.shape),
    }

    t0 = time.time()
    model = DeepseekV41ForCausalLM.initialize_vllm_model(None, mesh_device, max_batch_size=2, max_seq_len=max_seq_len)
    result["build_s"] = round(time.time() - t0, 1)
    kv = model.allocate_kv_cache((16, 1, 64, 512), torch.bfloat16, len(layers))
    assert len(kv) == len(layers)
    t0 = time.time()
    model.warmup_model_prefill(enable_trace=False, kv_cache=kv, can_sample_on_device=False)
    model.warmup_model_decode(
        enable_trace=False, kv_cache=kv, max_batch_size=2, num_blocks=16, can_sample_on_device=False
    )
    model.warmup_model_prefill(enable_trace=True, kv_cache=kv, can_sample_on_device=False)
    model.warmup_model_decode(
        enable_trace=True, kv_cache=kv, max_batch_size=2, num_blocks=16, can_sample_on_device=False
    )
    result["warmup_s"] = round(time.time() - t0, 1)
    result["dram_after_warmup_bytes"] = _dram_bytes(mesh_device)
    logger.info(f"generator: built in {result['build_s']} s, warmed up in {result['warmup_s']} s")

    text = orc.text_tokens(2 * prompt + 2 * steps + 2)[0]
    book_a, other = text[: prompt + steps + 1], text[prompt + steps + 1 :]
    tokenizer = model.tokenizer

    def prefill(rows: list[tuple[int, torch.Tensor]]) -> torch.Tensor:
        width = max(int(t.numel()) for _, t in rows)
        tokens = torch.zeros(len(rows), width, dtype=torch.int32)
        for i, (_, t) in enumerate(rows):
            tokens[i, : t.numel()] = t
        t0 = time.time()
        out = model.prefill_forward(
            tokens=tokens,
            prompt_lens=torch.tensor([int(t.numel()) for _, t in rows]),
            page_table=_page_table([b for b, _ in rows]),
            start_pos=torch.zeros(len(rows), dtype=torch.int32),
            kv_cache=kv,
            enable_trace=True,
        )
        result.setdefault("prefill_s", []).append(round(time.time() - t0, 3))
        assert out.shape == (len(rows), 1, C.VOCAB_SIZE) and not torch.isnan(out).any()
        return out

    def decode(rows: list[tuple[int, int, int]]) -> torch.Tensor:
        """rows: (first block, token, position); block 0 rows are idle."""
        t0 = time.time()
        out = model.decode_forward(
            tokens=torch.tensor([[t] for _, t, _ in rows], dtype=torch.int32),
            start_pos=torch.tensor([p for _, _, p in rows], dtype=torch.int32),
            page_table=_page_table([b for b, _, _ in rows]),
            kv_cache=kv,
            enable_trace=True,
            read_from_device=True,
        )
        result.setdefault("decode_s", []).append(round(time.time() - t0, 3))
        assert out.shape == (len(rows), 1, C.VOCAB_SIZE) and not torch.isnan(out).any()
        return out

    # --- two requests: the book (block 7, teacher-forced) and another stretch (block 9, free-running) ---
    logits = prefill([(7, book_a[:prompt]), (9, other[:prompt])])
    fed_a, tok_b = int(book_a[prompt]), int(logits[1, 0].argmax())
    hits, preds_a, text_b = 0, [], []
    for s in range(steps):
        p = prompt + s
        out = decode([(7, fed_a, p), (9, tok_b, p)])
        pred_a = int(out[0, 0].argmax())
        preds_a.append(pred_a)
        hits += pred_a == int(book_a[p + 1])
        fed_a = int(book_a[p + 1])
        tok_b = int(out[1, 0].argmax())
        text_b.append(tok_b)
        logger.info(f"generator: step {s} at {p}: book {pred_a} vs {int(book_a[p + 1])}, other {tok_b}")
    result["book_top1"] = round(hits / steps, 4)
    result["text"] = tokenizer.decode(preds_a)
    result["book_text"] = tokenizer.decode(book_a[prompt + 1 : prompt + steps + 1].tolist())
    result["other_text"] = tokenizer.decode(text_b)

    # --- a third request takes the least recently used slot: the same book prompt, from the start ---
    logits = prefill([(11, book_a[:prompt])])
    fed, hits2, preds = int(book_a[prompt]), 0, []
    for s in range(min(steps, 8)):
        p = prompt + s
        out = decode([(11, fed, p), (0, 0, 0)])
        pred = int(out[0, 0].argmax())
        preds.append(pred)
        hits2 += pred == int(book_a[p + 1])
        fed = int(book_a[p + 1])
    result["book_top1_reused_slot"] = round(hits2 / min(steps, 8), 4)
    result["text_reused_slot"] = tokenizer.decode(preds)
    result["slots"] = sorted(model.slots.keys())
    result["dram_after_requests_bytes"] = _dram_bytes(mesh_device)
    logger.info("V41_GENERATOR_RESULT " + json.dumps(result))
    out = os.environ.get("V41_GEN_OUT")
    if out:
        with open(out, "w") as f:
            json.dump(result, f, indent=1)
    model.decoder.release()
