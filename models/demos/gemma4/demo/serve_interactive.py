# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Minimal interactive HTTP server for Gemma4, for local chat-with-the-model use.

This is NOT vLLM: single request at a time (an asyncio.Lock serializes access to the
one device-resident model), and every request re-prefills the *entire* conversation
from position 0 rather than reusing KV across turns — the page table is a fixed
identity mapping reused every call, so a new prefill simply overwrites the previous
turn's pages. That is the simplest correct thing given a client that resends full
chat history each turn (as ~/scripts/client_demo.sh does), and it is fine at
interactive/chat latency; it does not attempt vLLM's incremental-prefix reuse or
continuous batching.

Reuses the exact model-loading and prefill/decode calls from text_demo_v2.py's
test_demo_text — only the fixed-prompts loop is replaced with an HTTP loop.

Endpoints match what ~/scripts/client_demo.sh expects, unmodified:
  GET  /tt-liveness           -> {"model_ready": true} once warmup is done
  GET  /v1/models             -> {"data": [{"id": <model_path>}]}
  POST /v1/chat/completions   -> OpenAI-shaped, streaming (SSE) or non-streaming
  POST /v1/completions        -> same, for raw-text (non-chat) use

Usage (defaults below target this QB2 box's pre-cached 12B weights — override any of them
to point elsewhere, e.g. a different HF_MODEL/mesh):
    pytest -sq models/demos/gemma4/demo/serve_interactive.py::test_serve

Then, in another shell:
    ~/scripts/client_demo.sh 128
"""

import asyncio
import json
import math
import os

import pytest
import torch
from loguru import logger

from models.demos.gemma4.demo.text_demo_v2 import _device_params, _host_sample, _model_path, create_tt_page_table
from models.demos.gemma4.tt.generator import Gemma4Generator
from models.tt_transformers.tt.common import PagedAttentionConfig, preprocess_inputs_prefill

# Defaults validated on this QB2 (4x p150, 1x4 mesh) box on 2026-08-13 — every var below is
# still overridable from the environment, this just removes the need to set them every time.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HOME", "/mnt/models/huggingface")
os.environ.setdefault("HF_MODEL", "google/gemma-4-12B-it")
os.environ.setdefault(
    "TT_CACHE_PATH",
    os.path.join(os.environ["HF_HOME"], "tt_cache", os.environ["HF_MODEL"].replace("/", "--")),
)

MAX_SEQ_LEN = int(os.environ.get("GEMMA4_MAX_SEQ_LEN", 1024))
PREFILL_TRACE_MAX = int(os.environ.get("GEMMA4_PREFILL_TRACE_MAX_SEQ", 4096))
PORT = int(os.environ.get("PORT", 8000))
API_KEY = os.environ.get("API_KEY", "your-secret-key")
BLOCK_SIZE = 32


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {
            "N150": (1, 1),
            "N300": (1, 2),
            "P150": (1, 1),
            "P300": (1, 2),
            "P150x4": (1, 4),
            "P300x2": (1, 4),
            "P150x8": (1, 8),
            "T3K": (1, 8),
        }.get(os.environ.get("MESH_DEVICE"), (1, 4))
    ],
    indirect=True,
)
def test_serve(mesh_device):
    import uvicorn
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse, StreamingResponse

    model_path = _model_path()
    batch_size = 1
    max_num_blocks = batch_size * math.ceil(MAX_SEQ_LEN / BLOCK_SIZE)
    paged_attention_config = PagedAttentionConfig(block_size=BLOCK_SIZE, max_num_blocks=max_num_blocks)

    logger.info(f"Loading Gemma4 from {model_path} (max_seq_len={MAX_SEQ_LEN})...")
    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=batch_size,
        max_seq_len=MAX_SEQ_LEN,
        num_layers=None,
        paged_attention_config=paged_attention_config,
        bounded_sliding_kv_cache=False,
    )
    model_args_list = generator.model_args
    page_table = create_tt_page_table(batch_size, paged_attention_config)

    # Gemma4Generator.from_pretrained only defaults stop_tokens to the tokenizer's
    # single eos_token_id. generation_config.json carries the model's *actual*
    # stop set (e.g. a turn/channel-boundary token beyond the base EOS) — without
    # merging it in, decode runs past the real answer and starts repeating.
    try:
        from transformers import GenerationConfig

        extra_eos = GenerationConfig.from_pretrained(model_path).eos_token_id
        extra_eos = [extra_eos] if isinstance(extra_eos, int) else list(extra_eos or [])
        tokenizer.stop_tokens = sorted(set(tokenizer.stop_tokens) | set(extra_eos))
        logger.info(f"stop_tokens: {tokenizer.stop_tokens}")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Could not merge generation_config.json eos tokens: {e}")

    prefill_enable_trace = MAX_SEQ_LEN < PREFILL_TRACE_MAX
    logger.info("Warming up prefill...")
    generator.warmup_model_prefill(
        kv_cache=tt_kv_cache, enable_trace=prefill_enable_trace, can_sample_on_device=False, greedy_only=True
    )
    model_ready = {"value": True}
    logger.info(f"Warmup complete — serving on :{PORT}. First real decode still pays a one-time compile.")

    lock = asyncio.Lock()

    def _run_generation(prompt_text: str, instruct: bool, max_tokens: int, temperature: float, top_p: float):
        """Blocking generator: yields ('token', text) then a final ('done', usage)."""
        input_tokens_prefill_pt, encoded_prompts, decoding_pos, prefill_lens = preprocess_inputs_prefill(
            [prompt_text], tokenizer, model_args_list, instruct, max_tokens, max_prefill_len=MAX_SEQ_LEN
        )
        input_tokens_prefill_pt = torch.stack(input_tokens_prefill_pt).view(batch_size, -1)

        prefill_logits = generator.prefill_forward_text(
            input_tokens_prefill_pt,
            page_table=page_table,
            kv_cache=tt_kv_cache,
            prompt_lens=decoding_pos,
            warmup_prefill=False,
            enable_trace=prefill_enable_trace,
        )
        out_tok = _host_sample(prefill_logits, temperature, top_p)
        current_pos = torch.tensor([decoding_pos[0]])

        output_ids = [int(out_tok[0, 0].item())]
        prev_text = ""
        completion_tokens = 0
        if output_ids[-1] not in tokenizer.stop_tokens:
            completion_tokens = 1
            text = tokenizer.decode(output_ids)
            delta, prev_text = text[len(prev_text) :], text
            if delta:
                yield ("token", delta)

        iteration = 1
        while completion_tokens < max_tokens and output_ids[-1] not in tokenizer.stop_tokens:
            logits, _ = generator.decode_forward(
                out_tok,
                current_pos,
                enable_trace=True,
                page_table=page_table,
                kv_cache=tt_kv_cache,
                sampling_params=None,
            )
            out_tok = _host_sample(logits, temperature, top_p)
            current_pos += 1
            tok = int(out_tok[0, 0].item())
            if tok in tokenizer.stop_tokens:
                break
            output_ids.append(tok)
            completion_tokens += 1
            text = tokenizer.decode(output_ids)
            delta, prev_text = text[len(prev_text) :], text
            if delta:
                yield ("token", delta)
            iteration += 1

        yield ("done", {"prompt_tokens": prefill_lens[0], "completion_tokens": completion_tokens})

    app = FastAPI()

    @app.get("/tt-liveness")
    async def liveness():
        return {"model_ready": model_ready["value"]}

    @app.get("/v1/models")
    async def models():
        return {"data": [{"id": model_path}]}

    async def _handle(prompt_text, instruct, max_tokens, temperature, top_p, stream, is_chat):
        if stream:

            async def gen():
                # The lock must wrap the actual generation, not just this coroutine's setup:
                # `async with lock:` around only the `return StreamingResponse(...)` construction
                # (the previous bug here) releases the lock before gen() has executed a single
                # line, since building a StreamingResponse only creates this generator object —
                # it doesn't drive it. All requests share ONE physical KV-cache slot
                # (batch_size=1), so without the lock actually held for the full duration,
                # concurrent requests interleave token-by-token against that same slot and
                # corrupt each other's context (confirmed: 4 concurrent different-topic prompts
                # produced answers cross-contaminated with each other's content). Holding the
                # lock here means concurrent requests queue and run one at a time — no
                # corruption, but also no throughput gain from concurrency (there is only one
                # batch slot to give out).
                async with lock:
                    for kind, payload in _run_generation(prompt_text, instruct, max_tokens, temperature, top_p):
                        if kind == "token":
                            delta = {"content": payload} if is_chat else {}
                            choice = (
                                {"index": 0, "delta": delta, "finish_reason": None}
                                if is_chat
                                else {"index": 0, "text": payload, "finish_reason": None}
                            )
                            yield _sse({"choices": [choice]})
                            # _run_generation is a plain sync generator doing blocking device
                            # work between yields, so `await send(...)` inside StreamingResponse
                            # never contains a genuine suspension point — the whole response
                            # would otherwise get generated and only flushed to the socket in
                            # one burst once the generator is fully exhausted. This forces a
                            # real event-loop tick so uvicorn actually writes each chunk now.
                            await asyncio.sleep(0)
                        else:
                            choice = {"index": 0, "finish_reason": "stop"}
                            if is_chat:
                                choice["delta"] = {}
                            else:
                                choice["text"] = ""
                            yield _sse({"choices": [choice], "usage": payload})
                    yield "data: [DONE]\n\n"

            return StreamingResponse(gen(), media_type="text/event-stream")

        async with lock:
            text = ""
            usage = {}
            for kind, payload in _run_generation(prompt_text, instruct, max_tokens, temperature, top_p):
                if kind == "token":
                    text += payload
                else:
                    usage = payload
        if is_chat:
            message = {"role": "assistant", "content": text}
            return JSONResponse(
                {"choices": [{"index": 0, "message": message, "finish_reason": "stop"}], "usage": usage}
            )
        return JSONResponse({"choices": [{"index": 0, "text": text, "finish_reason": "stop"}], "usage": usage})

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        messages = body.get("messages")
        if not messages:
            raise HTTPException(400, "messages required")
        # NOTE: no multi-turn memory. Gemma4Generator patches model_args.encode_prompt
        # (generator.py's _patch_model_args) to a closure that only ever wraps a single
        # plain string as one user turn — it has no path for a multi-message history, and
        # feeding it a pre-rendered multi-turn string double-applies the chat template
        # (the literal turn/channel markers become part of the *input* text, which this
        # preview checkpoint doesn't reliably terminate from — it degenerates into
        # repeating a "<|channel>thought" preamble instead of stopping). So each request
        # is treated as a fresh single-turn generation from just the latest user message,
        # matching exactly the call path text_demo_v2.py's passing demo test uses.
        last_user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), None)
        if not last_user:
            raise HTTPException(400, "no user message found")
        return await _handle(
            last_user,
            instruct=True,
            max_tokens=int(body.get("max_tokens", 128)),
            temperature=float(body.get("temperature") or 0.0),
            top_p=float(body.get("top_p") or 1.0),
            stream=bool(body.get("stream", False)),
            is_chat=True,
        )

    @app.post("/v1/completions")
    async def completions(request: Request):
        body = await request.json()
        prompt_text = body.get("prompt")
        if not prompt_text:
            raise HTTPException(400, "prompt required")
        return await _handle(
            prompt_text,
            instruct=False,
            max_tokens=int(body.get("max_tokens", 128)),
            temperature=float(body.get("temperature") or 0.0),
            top_p=float(body.get("top_p") or 1.0),
            stream=bool(body.get("stream", False)),
            is_chat=False,
        )

    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
