# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exercise actual lm-eval HTTP transport against a synthetic localhost API."""

import argparse
import asyncio
import inspect
import json
from pathlib import Path

from aiohttp import ClientSession, web
from lm_eval.models.openai_completions import LocalChatCompletion

MODEL = "google/gemma-4-26B-A4B-it"
REVISION = "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"
GENERATION = {
    "max_gen_toks": 4096,
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 64,
    "do_sample": True,
    "until": [],
    "chat_template_kwargs": {"enable_thinking": False},
    "logprobs": True,
    "top_logprobs": 0,
}


async def check():
    from importlib.metadata import version

    import benchmark_stage

    expected = Path(
        "/home/mvasiljevic/.codex-personal-gemma4/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.14/runtime/benchmark_stage"
    )
    assert Path(benchmark_stage.__file__).resolve().parent == expected
    messages = [
        {"role": "user", "content": "Synthetic example: what is one plus one?"},
        {"role": "assistant", "content": "Two."},
        {"role": "user", "content": "Synthetic probe: what is two plus two?"},
    ]
    captured = []

    async def receive(request):
        payload = await request.json()
        captured.append(payload)
        return web.json_response(
            {
                "id": "synthetic-transport-probe",
                "object": "chat.completion",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "Four."}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", receive)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    backend = LocalChatCompletion(
        model=MODEL,
        base_url=f"http://127.0.0.1:{port}/v1/chat/completions",
        num_concurrent=32,
        max_retries=0,
        tokenized_requests=False,
        tokenizer_backend=None,
        max_gen_toks=2048,
    )
    try:
        serialized = backend.apply_chat_template(messages)
        assert json.loads(serialized.prompt) == messages
        async with ClientSession() as session:
            answers = await backend.amodel_call(session, asyncio.Semaphore(32), [serialized], gen_kwargs=GENERATION)
        assert answers == ["Four."]
        assert len(captured) == 1
        payload = captured[0]
        assert payload["messages"] == messages
        for key in ("temperature", "top_p", "top_k", "chat_template_kwargs", "logprobs", "top_logprobs"):
            assert payload[key] == GENERATION[key], key
        assert payload["max_tokens"] == 4096 and payload["stop"] == []
        assert "ignore_eos" not in payload
        return {
            "scope": "Synthetic local HTTP only; no model inference, live server, benchmark score, or measured token counts.",
            "benchmark_module": benchmark_stage.__file__,
            "lm_eval_version": version("lm_eval"),
            "transformers_version": version("transformers"),
            "backend_source": inspect.getfile(LocalChatCompletion),
            "generation": GENERATION,
            "captured_payload": payload,
            "synthetic_response_parsed": answers,
            "client_template": "structured JSON messages only; no native template rendered client-side",
            "native_render_scope": "Server rendering is separately checked by benchmark_tokenizer_check.py in the compatible server environment.",
            "assertions_passed": True,
        }
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(asyncio.run(check()), indent=2) + "\n")
    print(f"Transport checks passed: {args.output}")
