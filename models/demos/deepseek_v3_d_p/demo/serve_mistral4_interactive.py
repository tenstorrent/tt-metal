# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Minimal interactive chat server for the Mistral Small 4 119B *prefill* bring-up.

WHAT THIS IS
------------
`deepseek_v3_d_p` is the **disaggregated-prefill** stack: it has no decode path at all
(there is not one `*decode*` file in the folder). So this server does not decode. It
generates by **re-running the full prefill for every single token**:

    prefill(prompt[0:n])       -> token n        (logits are read at row actual_isl-1)
    prefill(prompt[0:n+1])     -> token n+1
    ...

That is O(n) prefills for n tokens instead of one prefill plus n cheap decode steps, so it
is quadratic and slow in absolute terms. It is, however, *exactly* the code path that the
bring-up validated -- same weights, same mesh, same `TtPrefillTransformer.forward` -- which
makes it a real end-to-end demo and a useful qualitative numerics probe, with no new kernels.

Why not fake a decode with a 1-token chunked prefill? The chunked path shards the sequence
over `sp_factor` devices and each shard is tile-aligned, and the MoE tightens that further
(see SPEED), so the smallest expressible chunk is hundreds of tokens, not 1. Single-token
decode needs the real decode kernels in `models/demos/deepseek_v3/` -- a separate bring-up.

CORRECTNESS OF THE RE-PREFILL TRICK
-----------------------------------
The token buffer is a fixed `isl_total` window; only the first `actual_isl` entries are real
and the tail holds pad. Attention is causal, so row `actual_isl-1` -- the only row whose
logits we read -- can never attend to the pad tail. Stale KV left in the cache beyond
`actual_isl` from the previous step is therefore unobservable.

SPEED
-----
Cost is set by the *padded* window `isl_total`, not by `actual_isl`: every step computes the
whole window. Measured at window 512 over a 17-token reply: **~1.05 s/token, TTFT 979 ms**.

How it scales with the window is **not settled**. Do not repeat the mistake of averaging a
short reply: the first token of a process pays program compilation, so a 2-token average once
reported 2.80 s/token when the steady state was ~1.05. Every token's latency is logged
individually for exactly this reason -- read those lines, discard the first few, and generate
30+ tokens before comparing two windows.

The compile cost does not recur across processes: tt-metal caches JIT artifacts on disk, so a
second server on the same configuration answers its first request in under a second.

`PREFILL_SERVE_SEQ_LEN` must be a multiple of `64 * sp_factor`, so **512 is the minimum**.
Tile alignment alone would permit 256, but the MoE's `masked_bincount` runs on a 64-core grid
and needs the per-chip token count divisible by 64 -- a 256 window dies mid-request with
"Token count (32) must be divisible by the 64-core grid used by masked_bincount".

Measured tok/s is logged per request and reported in the final SSE usage payload.

API
---
OpenAI-shaped and SSE-streaming, so `~/scripts/client_demo.sh` and any OpenAI client work
unmodified: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/tt-liveness`.

Unlike the Gemma4 server, multi-turn works properly here: every request re-prefills the whole
conversation from position 0 anyway, so applying the chat template to the full message list is
the natural thing rather than a special case.

RUN
---
    cd /data/kmabee/tt-metal
    export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
    export LD_LIBRARY_PATH=$PWD/build_Release/lib:$LD_LIBRARY_PATH
    export MISTRAL4_HF_MODEL=/data/kmabee/models/Mistral-Small-4-119B-2603
    export TT_MISTRAL4_PREFILL_TTNN_CACHE=/data/kmabee/mistral4_caches/ttnn_cache_8x4
    export TT_MISTRAL4_PREFILL_HOST_REF_CACHE=/data/kmabee/mistral4_caches/ref_cache
    export PREFILL_SERVE_SEQ_LEN=1024 PORT=8000
    ./python_env/bin/pytest models/demos/deepseek_v3_d_p/demo/serve_mistral4_interactive.py \
        -k "serve" -s

Then talk to it (streams tokens as they are produced):

    curl -N http://localhost:8000/v1/chat/completions \
      -H 'Content-Type: application/json' -H 'Authorization: Bearer dummy' \
      -d '{"model":"mistral-small-4","stream":true,"max_tokens":32,
           "messages":[{"role":"user","content":"Name three French cities."}]}'
"""

# NOTE: deliberately NO `from __future__ import annotations`. FastAPI resolves handler annotations
# with typing.get_type_hints against the MODULE globals, and `Request` is imported inside
# _build_app() (so that this module stays importable without fastapi). With postponed evaluation the
# annotation stays the string "Request", is looked up in module globals, and startup dies with
# `NameError: name 'Request' is not defined` wrapped in a pydantic PydanticUndefinedAnnotation.
# Without it, annotations are evaluated where they are written -- inside _build_app, where Request
# is in scope -- and FastAPI gets the real class.
import asyncio
import json
import os
import time
from datetime import date as _date
from typing import Any

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.mistral_small4_config import MistralSmall4Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode

# The generated window and the port are the two knobs a demo actually needs.
SERVE_SEQ_LEN = int(os.environ.get("PREFILL_SERVE_SEQ_LEN", 1024))
PORT = int(os.environ.get("PORT", 8000))
MODEL_NAME = os.environ.get("PREFILL_SERVE_MODEL_NAME", "mistral-small-4-119b")
# Hard ceiling per request so a runaway generation cannot hold the mesh forever.
MAX_TOKENS_CAP = int(os.environ.get("PREFILL_SERVE_MAX_TOKENS", 128))
# Mistral's chat template injects a ~545-token default system prompt (tool-use instructions, a
# knowledge-cutoff blurb, Le Chat branding) when the caller sends no system message. Since every
# token of prompt sits inside the fixed window and the window sets per-token latency, that default
# alone would force a 1024-token window for a one-line question. A short system prompt costs 31
# tokens for the same question -- an 18x reduction -- which is what keeps the minimum 512-token
# window usable for real conversations. Set to "" to use Mistral's real default.
DEFAULT_SYSTEM_PROMPT = os.environ.get("PREFILL_SERVE_SYSTEM_PROMPT", "You are a helpful assistant. Answer briefly.")


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


class PrefillTokenGenerator:
    """Wraps a built `TtPrefillTransformer` as a one-token-at-a-time generator.

    Owns the host-side token window and the per-step device upload. Everything it needs is
    handed in by `run_model`, so the served model is bit-for-bit the configuration the
    bring-up tests validated.
    """

    def __init__(
        self,
        *,
        transformer,
        mesh_device,
        kvpe_cache,
        index_kv_cache,
        tokenizer,
        config,
        isl_total: int,
        sp_factor: int,
        isl_per_chip: int,
        chunk_order,
        padding_side: str,
    ):
        # Right padding is load-bearing: the LM head reads row `actual_isl - 1`. With left
        # padding it reads `seq_len - 1` instead, which is the wrong row for a partial window.
        assert padding_side == "right", (
            f"serve requires right padding (LM head reads row actual_isl-1); got '{padding_side}'. "
            "Run the serve entry with the right_pad tokenizer."
        )
        self.transformer = transformer
        self.mesh_device = mesh_device
        self.kvpe_cache = kvpe_cache
        self.index_kv_cache = index_kv_cache
        self.tokenizer = tokenizer
        self.config = config
        self.isl_total = isl_total
        self.sp_factor = sp_factor
        self.isl_per_chip = isl_per_chip
        self.chunk_order = chunk_order
        # Trace state: None until enable_trace() captures. _temperature is set per request
        # because the eagerly-run tail samples outside the trace.
        self._controller = None
        self._trace_input = None
        self._trace_hidden = None
        self._temperature = 0.0

        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 11
        # Gemma4 lesson: seed the stop set from the model's generation_config, not just the
        # tokenizer's single eos_token_id. Mistral's set happens to be just {2}, but read it
        # rather than assume it.
        self.stop_ids = {int(tokenizer.eos_token_id)} if tokenizer.eos_token_id is not None else set()
        gen_cfg = os.path.join(str(os.environ.get("MISTRAL4_HF_MODEL", "")), "generation_config.json")
        if os.path.isfile(gen_cfg):
            try:
                eos = json.load(open(gen_cfg)).get("eos_token_id")
                if isinstance(eos, int):
                    self.stop_ids.add(eos)
                elif isinstance(eos, list):
                    self.stop_ids.update(int(e) for e in eos)
            except Exception as e:  # a malformed generation_config must not take the server down
                logger.warning(f"could not read stop tokens from {gen_cfg}: {e}")
        logger.info(f"serve: stop token ids = {sorted(self.stop_ids)}, pad id = {self.pad_id}")

    def _upload(self, host_ids: torch.Tensor) -> ttnn.Tensor:
        """Shard the [1, isl_total] host token window onto the mesh exactly as run_model does."""
        ids = host_ids
        if self.chunk_order is not None:
            from models.demos.deepseek_v3_d_p.tt.mla.utils import reorder_tensor_chunks

            ids = reorder_tensor_chunks(ids.unsqueeze(1).unsqueeze(-1), self.chunk_order, seq_dim=2)
            ids = ids.squeeze(1).squeeze(-1)
        return ttnn.from_torch(
            ids.reshape(self.sp_factor, 1, self.isl_per_chip),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device, mesh_shape=tuple(self.mesh_device.shape), dims=(0, None)
            ),
        )

    def enable_trace(self):
        """Capture the block stack once so each token is a trace replay instead of 2316 dispatches.

        Measured on this model: 1715 ms/token eager -> 80.1 ms/token traced, a 21x speedup, with the
        traced path sampling the identical token. The forward is ~95% host-bound (80 ms of device
        kernel time under a 1760 ms wall clock), so removing per-op dispatch is the whole win --
        80.1 ms is essentially the device kernel time, i.e. we now pay for silicon.

        Capture is done ONCE at startup with ``actual_isl = isl_total`` -- the entire window marked
        real -- which makes it independent of any request's prompt length, so one capture serves
        every conversation. That is safe for the same causality reason the whole re-prefill trick is
        safe: the LM head reads row ``n-1`` and nothing attends past it, so positions after the real
        tokens cannot influence the logits we read, whether they hold pad or garbage. The tail
        (norm/LM-head/sample) stays eager -- it ends in a blocking D2H that cannot be traced -- which
        is also what lets each step select the CORRECT row ``n-1`` while the captured block stack
        stays invariant.
        """
        from models.demos.deepseek_v3_d_p.utils.sub_device_trace import SubDeviceTraceController

        window = torch.full((1, self.isl_total), self.pad_id, dtype=torch.int64)
        self._trace_input = self._upload(window)

        # Warm-compile before capturing: a capture records dispatch, not compilation.
        self.transformer(
            self._trace_input,
            self.kvpe_cache,
            actual_isl=self.isl_total,
            return_intermediates=False,
            read_profiler=False,
            temperature=0.0,
            index_kv_cache=self.index_kv_cache,
            stop_after_blocks=True,
        )
        ttnn.synchronize_device(self.mesh_device)

        self._controller = SubDeviceTraceController(self.mesh_device)
        self.transformer.set_trace_controller(self._controller)
        self._controller.begin_capture()
        self._trace_hidden = self.transformer(
            self._trace_input,
            self.kvpe_cache,
            actual_isl=self.isl_total,
            return_intermediates=False,
            read_profiler=False,
            temperature=0.0,
            index_kv_cache=self.index_kv_cache,
            stop_after_blocks=True,
        )
        self._controller.end_capture()
        logger.success(
            f"serve: trace captured ({self._controller.num_segments} segments) — "
            f"expect ~20x faster tokens than the eager path"
        )

    def release_trace(self):
        """Release traces + MoE sub-device managers. Leaving either registered segfaults
        close_mesh_device at teardown, so this must run even on a failed startup."""
        if self._controller is not None:
            try:
                self._controller.release()
            finally:
                self._controller = None
                self.transformer.set_trace_controller(None)
                self.transformer.release_sub_device_managers()

    def _forward_token(self, window: torch.Tensor, n: int) -> int:
        """One token: traced replay when captured, otherwise the plain eager forward."""
        if self._controller is None:
            tt_tokens = self._upload(window)
            token_id, _prob, _ = self.transformer(
                tt_tokens,
                self.kvpe_cache,
                actual_isl=n,
                return_intermediates=False,
                read_profiler=False,
                temperature=self._temperature,
                index_kv_cache=self.index_kv_cache,
            )
            ttnn.synchronize_device(self.mesh_device)
            ttnn.deallocate(tt_tokens)
            return int(token_id)

        # Traced: write this step's tokens into the tensor the capture recorded -- a fresh
        # from_torch would land elsewhere and the replay would keep reading the old address.
        host = ttnn.from_torch(
            window.reshape(self.sp_factor, 1, self.isl_per_chip),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device, mesh_shape=tuple(self.mesh_device.shape), dims=(0, None)
            ),
        )
        ttnn.copy_host_to_device_tensor(host, self._trace_input)
        self._controller.replay()
        # Tail runs eagerly on the replay's output, which stays at its captured address. `n` -- the
        # true token count -- selects the row here, so a single capture serves every step.
        h = self.transformer.norm(self._trace_hidden)
        _logits, first_token_logits = self.transformer._lm_head_and_extract(h, n)
        token_id, _prob, _sweep = self.transformer._sample(first_token_logits, n, self._temperature)
        return int(token_id)

    def generate(self, prompt_ids: list[int], max_tokens: int, temperature: float):
        """Yield (token_id, text_piece, seconds_for_this_token) until stop/limit/window-full."""
        n = len(prompt_ids)
        if n >= self.isl_total:
            raise ValueError(
                f"prompt is {n} tokens but the served window is {self.isl_total}. "
                f"Raise PREFILL_SERVE_SEQ_LEN (multiple of {32 * self.sp_factor}) or shorten the prompt."
            )

        window = torch.full((1, self.isl_total), self.pad_id, dtype=torch.int64)
        window[0, :n] = torch.tensor(prompt_ids, dtype=torch.int64)

        for _ in range(max_tokens):
            if n >= self.isl_total:
                logger.warning(f"serve: hit the {self.isl_total}-token window, stopping generation")
                break
            t0 = time.time()
            self._temperature = temperature
            token_id = self._forward_token(window, n)
            dt = time.time() - t0
            # Log EVERY token's latency, not just the request average. The first token of the first
            # request in a process pays program compilation and can be seconds slower than the rest,
            # so an average over a short reply is not a steady-state number -- averaging 2 tokens
            # once produced a "2.80 s/token" figure when the true steady state was ~1.05 s/token.
            # Per-token lines make that visible instead of hiding it in the mean.
            logger.info(f"serve: token {n - len(prompt_ids) + 1} in {dt:.2f}s (actual_isl={n})")

            if int(token_id) in self.stop_ids:
                logger.info(f"serve: stop token {int(token_id)} after {n - len(prompt_ids)} generated tokens")
                break

            window[0, n] = int(token_id)
            n += 1
            piece = self.tokenizer.decode([int(token_id)], skip_special_tokens=False)
            yield int(token_id), piece, dt


def _build_app(gen: PrefillTokenGenerator):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse

    app = FastAPI(title="Mistral Small 4 prefill-only chat")
    # batch_size 1 and a single shared KV cache: concurrent generations would interleave
    # token-by-token on the same slot and corrupt each other. The Gemma4 server originally
    # released the lock before generating a single token; here the lock is acquired INSIDE
    # the streaming body and held across the whole loop, so requests queue instead.
    lock = asyncio.Lock()

    def _prompt_ids_from_messages(messages: list[dict]) -> list[int]:
        """Apply the checkpoint's chat template. Every request re-prefills from position 0,
        so the full multi-turn history is just part of the prompt -- no special casing."""
        if DEFAULT_SYSTEM_PROMPT and not any(m.get("role") == "system" for m in messages):
            messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}] + list(messages)
        try:
            # Template to TEXT first so the literal `{today}` placeholder in this checkpoint's
            # chat_template.jinja can be filled. The template never substitutes it (Mistral's own
            # tooling does), so tokenizing directly would feed the model the characters "{today}".
            text = gen.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            text = text.replace("{today}", _date.today().strftime("%A, %B %-d, %Y"))
            ids = gen.tokenizer(text, add_special_tokens=False)["input_ids"]
        except Exception as e:
            # No/!broken template -> fall back to a plain concatenation rather than 500.
            logger.warning(f"apply_chat_template failed ({e}); falling back to plain concatenation")
            text = "\n".join(f"{m.get('role')}: {m.get('content')}" for m in messages) + "\nassistant:"
            ids = gen.tokenizer(text, add_special_tokens=True)["input_ids"]
        return list(ids[0]) if isinstance(ids[0], list) else list(ids)

    @app.get("/tt-liveness")
    async def liveness():
        # `model_ready` is load-bearing: ~/scripts/client_demo.sh polls this endpoint and only
        # proceeds when that key is truthy, so omitting it leaves the client spinning forever.
        # By the time this app exists the model is already built on device, so it is always True.
        return {
            "status": "ok",
            "model_ready": True,
            "model": MODEL_NAME,
            "window": gen.isl_total,
            "mode": "prefill-only (re-prefills the whole window per token)",
        }

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "tenstorrent"}]}

    async def _handle(prompt_ids: list[int], max_tokens: int, temperature: float, stream: bool, is_chat: bool):
        max_tokens = max(1, min(int(max_tokens), MAX_TOKENS_CAP))
        created = int(time.time())
        rid = f"chatcmpl-{created}"
        obj = "chat.completion.chunk" if is_chat else "text_completion"

        if stream:

            async def body():
                # Lock is taken here, around the actual generation, and held to the last token.
                async with lock:
                    t_start = time.time()
                    ntok = 0
                    try:
                        for _tid, piece, _dt in gen.generate(prompt_ids, max_tokens, temperature):
                            ntok += 1
                            delta = {"content": piece} if is_chat else None
                            choice = (
                                {"index": 0, "delta": delta, "finish_reason": None}
                                if is_chat
                                else {"index": 0, "text": piece, "finish_reason": None}
                            )
                            yield _sse(
                                {"id": rid, "object": obj, "created": created, "model": MODEL_NAME, "choices": [choice]}
                            )
                            # Without an actual event-loop tick uvicorn buffers the whole
                            # response until the generator exhausts, which defeats streaming.
                            await asyncio.sleep(0)
                    except Exception as e:
                        logger.exception("generation failed")
                        # client_demo.sh's contract for errors: finish_reason == "error" with the
                        # message in the normal text field. A bare {"error": ...} chunk has no
                        # "choices" key and makes the client raise KeyError instead of showing it.
                        msg = f"{type(e).__name__}: {e}"
                        errc = (
                            {"index": 0, "delta": {"content": msg}, "finish_reason": "error"}
                            if is_chat
                            else {"index": 0, "text": msg, "finish_reason": "error"}
                        )
                        yield _sse(
                            {"id": rid, "object": obj, "created": created, "model": MODEL_NAME, "choices": [errc]}
                        )
                        yield "data: [DONE]\n\n"
                        return
                    elapsed = time.time() - t_start
                    tps = ntok / elapsed if elapsed > 0 else 0.0
                    logger.info(
                        f"serve: {ntok} tokens in {elapsed:.1f}s = {tps:.2f} tok/s " f"({elapsed / ntok:.2f}s/token)"
                        if ntok
                        else "serve: 0 tokens"
                    )
                    fin = (
                        {"index": 0, "delta": {}, "finish_reason": "stop"}
                        if is_chat
                        else {"index": 0, "text": "", "finish_reason": "stop"}
                    )
                    yield _sse(
                        {
                            "id": rid,
                            "object": obj,
                            "created": created,
                            "model": MODEL_NAME,
                            "choices": [fin],
                            "usage": {
                                "prompt_tokens": len(prompt_ids),
                                "completion_tokens": ntok,
                                "total_tokens": len(prompt_ids) + ntok,
                                "tt_seconds": round(elapsed, 2),
                                "tt_tokens_per_second": round(tps, 3),
                            },
                        }
                    )
                    yield "data: [DONE]\n\n"

            return StreamingResponse(body(), media_type="text/event-stream")

        async with lock:
            t_start = time.time()
            pieces = []
            try:
                for _tid, piece, _dt in gen.generate(prompt_ids, max_tokens, temperature):
                    pieces.append(piece)
            except Exception as e:
                logger.exception("generation failed")
                return JSONResponse(status_code=400, content={"error": {"message": str(e)}})
            elapsed = time.time() - t_start
            text = "".join(pieces)
            choice = (
                {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
                if is_chat
                else {"index": 0, "text": text, "finish_reason": "stop"}
            )
            return {
                "id": rid,
                "object": "chat.completion" if is_chat else "text_completion",
                "created": created,
                "model": MODEL_NAME,
                "choices": [choice],
                "usage": {
                    "prompt_tokens": len(prompt_ids),
                    "completion_tokens": len(pieces),
                    "total_tokens": len(prompt_ids) + len(pieces),
                    "tt_seconds": round(elapsed, 2),
                    "tt_tokens_per_second": round(len(pieces) / elapsed, 3) if elapsed > 0 else 0.0,
                },
            }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        b = await request.json()
        ids = _prompt_ids_from_messages(b.get("messages", []))
        return await _handle(
            ids, b.get("max_tokens", 32), float(b.get("temperature", 0.0)), bool(b.get("stream", False)), True
        )

    @app.post("/v1/completions")
    async def completions(request: Request):
        b = await request.json()
        ids = list(gen.tokenizer(b.get("prompt", ""), add_special_tokens=True)["input_ids"])
        return await _handle(
            ids, b.get("max_tokens", 32), float(b.get("temperature", 0.0)), bool(b.get("stream", False)), False
        )

    return app


def serve_hook(**ctx: Any) -> None:
    """Called by `run_model` once the validated model is built on device. Blocks in uvicorn."""
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
    except ModuleNotFoundError as e:
        # create_venv.sh does not install these; they are demo-only deps. Failing here rather
        # than after a multi-minute weight load would be the kinder default, but the model is
        # already up by this point, so say exactly what to run.
        raise ModuleNotFoundError(
            f"{e.name} is not installed in python_env. This server needs it:\n"
            f"    ./python_env/bin/pip install fastapi uvicorn"
        ) from e
    import uvicorn

    gen = PrefillTokenGenerator(**ctx)

    # Capture the trace once at startup: ~21x faster tokens (1715 ms -> 80 ms measured), same
    # sampled token. Falls back to the eager path rather than failing the demo, since eager is
    # merely slow while no server at all is useless.
    if os.environ.get("PREFILL_SERVE_TRACE", "1") not in ("0", "false", "no"):
        try:
            gen.enable_trace()
        except Exception as e:
            logger.exception(f"serve: trace capture failed ({e}); falling back to the eager path")
            gen.release_trace()
    else:
        logger.warning("serve: PREFILL_SERVE_TRACE disabled — using the ~21x slower eager path")

    app = _build_app(gen)
    logger.success(
        f"Mistral Small 4 prefill-only chat server on :{PORT} | window={gen.isl_total} tokens "
        f"| mesh={tuple(gen.mesh_device.shape)} | re-prefills the whole window per token"
    )
    try:
        uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
    finally:
        # Ctrl-C lands here. Releasing before the mesh_device fixture tears down avoids the
        # close_mesh_device segfault that registered sub-device managers cause.
        gen.release_trace()
        logger.info("serve: trace + sub-device managers released")


# --- pytest entry: reuses the bring-up's fixtures so the served model == the tested model ---


# Deliberately the SAME mesh/fabric/gate/expert parametrization as
# test_mistral4_prefill_transformer's `smoke-json_prompts-pretrained` row. The point of the demo
# is that it serves the configuration the bring-up actually validated, so anything that would
# change numerics (mesh 8x4, num_links 2, GPT_DEVICE gate, 128 experts, 36 layers, real weights)
# is copied from there rather than chosen here. Only isl_total is free: it sets the window and
# therefore the per-token latency.
@pytest.mark.skipif(not is_blackhole(), reason="Mistral Small 4 bring-up targets Blackhole")
@pytest.mark.parametrize("tokenizer", ["right"], indirect=True, ids=["right_pad"])
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    [
        pytest.param(
            (8, 4),
            # Upstream retired FABRIC_1D for the 8x4 prefill rows on this branch (it now reports
            # "unfeasible on the given hardware" and SKIPs); torus_xy is the 8x4 ring/ring profile
            # that replaced it -- see test_prefill_transformer.py's own mesh-8x4 row.
            torus_xy_device_params(fabric_payload_size=MistralSmall4Config.FABRIC_PAYLOAD_SIZE),
            2,
            ttnn.Topology.Linear,
            # The tests' `requires_mesh_topology` mark is registered by tests/conftest.py and is not
            # visible here, where it would only emit PytestUnknownMarkWarning and gate nothing.
            # Dropped: this demo simply requires the 8x4 galaxy, and mesh_device fails plainly
            # without it.
            id="mesh-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["mistral_small4"], indirect=True, ids=["mistral4"])
@pytest.mark.timeout(0)  # a server runs until killed
def test_serve(
    variant,
    config_only,
    mesh_device,
    device_params,
    num_links,
    topology,
    weight_cache_path,
    is_ci_env,
    is_ci_v2_env,
    tokenizer,
    request,
):
    """Serve the 36-layer Mistral Small 4 model with real weights for interactive chat."""
    from models.demos.deepseek_v3_d_p.tests.test_prefill_transformer import run_model

    # Fail on missing demo-only deps NOW, not after several minutes of weight loading.
    for mod in ("fastapi", "uvicorn"):
        pytest.importorskip(mod, reason=f"{mod} not in python_env: ./python_env/bin/pip install fastapi uvicorn")

    sp_factor = 8
    # 64, not 32, tokens per chip: tile alignment alone would allow a 256 window, but the MoE's
    # masked_bincount runs on a 64-core grid and requires the per-chip token count to be divisible
    # by 64. A 256 window passes a 32-based check and then dies mid-request on device with
    # "Token count (32) must be divisible by the 64-core grid used by masked_bincount", so the
    # stricter bound belongs here where it is a clear startup message. Measured, not inferred.
    min_multiple = 64 * sp_factor
    if SERVE_SEQ_LEN % min_multiple != 0:
        pytest.fail(
            f"PREFILL_SERVE_SEQ_LEN={SERVE_SEQ_LEN} must be a multiple of {min_multiple} "
            f"(64 tokens/chip for the MoE masked_bincount grid x sp={sp_factor}); 512 is the minimum"
        )

    run_model(
        variant,
        config_only,
        mesh_device,
        device_params,
        False,  # is_balanced
        SERVE_SEQ_LEN,  # isl_total -- the served window
        8,  # dispatch_buffer_capacity_factor
        36,  # num_layers
        MistralSmall4Config.NUM_ROUTED_EXPERTS,
        GateComputeMode.GPT_DEVICE,
        num_links,
        topology,
        False,  # pcc_validation
        False,  # determinism_check
        1,  # num_iterations
        "json_prompts",  # input_source: only used to build the throwaway startup token window
        True,  # use_pretrained -- real checkpoint
        False,  # return_kv_cache
        0.0,  # temperature (per-request override)
        weight_cache_path,
        is_ci_env,
        is_ci_v2_env,
        tokenizer,
        request,
        serve_hook=serve_hook,
    )
