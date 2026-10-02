# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible chat server for MiMo-V2.6-Flash on a TT mesh (stdlib http.server, one request at a time).

    POST /v1/chat/completions   (stream true/false, tools -> tool_calls, <think> -> reasoning_content)
    GET  /v1/models

Generation = chunked prefill + greedy/sampled tokens by re-prefilling the chunk that holds the newest token (earlier
chunks stay in the KV cache). Prefix caching across requests: the token list in the KV cache is kept; a new request
re-prefills only from the chunk holding the first differing token (chunk-aligned starts, as the sliding-window layers
need). The HTTP threads hand jobs to the thread that owns the device (``serve_forever`` below, the pytest main thread).

Env: MIMO_SERVE_PORT (8000), MIMO_SERVE_HOST (0.0.0.0), MIMO_SERVE_MAX_CTX (65536 tokens: prompt + completion),
MIMO_SERVE_CHUNK (1024), MIMO_SERVE_LAYERS (48), MIMO_SERVE_THINKING (1: default enable_thinking),
MIMO_SERVE_MAX_TOKENS (8192: default completion cap), MIMO_SERVE_TEMPERATURE / MIMO_SERVE_TOP_P (defaults when the
request has none: 0.6 / 0.95), MIMO_SERVE_MODEL_ID (mimo-v2.6-flash), MIMO_SERVE_WARMUP (0: tokens of a dummy prompt
prefilled at startup, to JIT the chunk positions), MIMO_SERVE_PID (pid file). The MiMoRuntimeOptions MIMO_* env apply.
"""

import json
import math
import os
import queue
import re
import signal
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from loguru import logger

from models.demos.mimo_v2_d_p.reference.config import MiMoTextConfig
from models.demos.mimo_v2_d_p.reference.remote_st import LOCAL
from models.demos.mimo_v2_d_p.reference.weights import global_state, layer_state
from models.demos.mimo_v2_d_p.tt.model import PAD_TOKEN_ID, TtMiMoModel
from models.demos.mimo_v2_d_p.tt.options import MiMoRuntimeOptions

ENV = os.environ.get
PORT = int(ENV("MIMO_SERVE_PORT", "8000"))
HOST = ENV("MIMO_SERVE_HOST", "0.0.0.0")
MAX_CTX = int(ENV("MIMO_SERVE_MAX_CTX", "65536"))
CHUNK = int(ENV("MIMO_SERVE_CHUNK", "1024"))
N_LAYERS = int(ENV("MIMO_SERVE_LAYERS", "48"))
THINKING = ENV("MIMO_SERVE_THINKING", "1") != "0"
MAX_TOKENS = int(ENV("MIMO_SERVE_MAX_TOKENS", "8192"))
TEMPERATURE = float(ENV("MIMO_SERVE_TEMPERATURE", "0.6"))
TOP_P = float(ENV("MIMO_SERVE_TOP_P", "0.95"))
MODEL_ID = ENV("MIMO_SERVE_MODEL_ID", "mimo-v2.6-flash")
WARMUP = int(ENV("MIMO_SERVE_WARMUP", "0"))

THINK_OPEN, THINK_CLOSE, TOOL_OPEN = "<think>", "</think>", "<tool_call>"


# ------------------------------------------------------------------ prompt
def normalize_messages(messages):
    """OpenAI messages -> what the MiMo chat template expects (tool-call arguments as dicts, text parts joined)."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("role") == "developer":
            m["role"] = "system"
        c = m.get("content")
        if isinstance(c, list):
            m["content"] = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
        elif c is None:
            m["content"] = ""
        if m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                tc = dict(tc)
                fn = dict(tc.get("function") or {})
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args.strip() else {}
                    except json.JSONDecodeError:
                        args = {"arguments": args}
                fn["arguments"] = args if isinstance(args, dict) else {"arguments": args}
                tc["function"] = fn
                calls.append(tc)
            m["tool_calls"] = calls
        if "reasoning" in m and "reasoning_content" not in m:  # some clients echo the field under this name
            m["reasoning_content"] = m.pop("reasoning")
        out.append(m)
    return out


# ------------------------------------------------------------------ output parsing
def _hold(s, tag):
    """Length of the longest suffix of ``s`` that is a proper prefix of ``tag`` (held back while streaming)."""
    for k in range(min(len(tag) - 1, len(s)), 0, -1):
        if s.endswith(tag[:k]):
            return k
    return 0


def split_output(text, final):
    """Generated text -> (reasoning, content, tool_text | None). Not ``final``: trailing partial tags held back."""
    reasoning, rest = "", text
    t = text.lstrip()
    if t.startswith(THINK_OPEN) or (not final and THINK_OPEN.startswith(t)):
        body = t[len(THINK_OPEN) :] if t.startswith(THINK_OPEN) else ""
        if THINK_CLOSE in body:
            reasoning, rest = body.split(THINK_CLOSE, 1)
        else:
            reasoning = body if final else body[: len(body) - _hold(body, THINK_CLOSE)]
            return reasoning.strip("\n") if final else reasoning.lstrip("\n"), "", None
    tool = None
    if TOOL_OPEN in rest:
        rest, tool = rest.split(TOOL_OPEN, 1)
        tool = TOOL_OPEN + tool
    elif not final:
        rest = rest[: len(rest) - _hold(rest, TOOL_OPEN)]
    content = rest.lstrip()
    if final or tool is not None:
        content = content.rstrip()
    return (reasoning.strip("\n") if final else reasoning.lstrip("\n")), content, tool


def _schema_type(tool_schemas, fn, arg):
    props = ((tool_schemas.get(fn) or {}).get("properties")) or {}
    s = props.get(arg) or {}
    t = s.get("type")
    if t is None:
        for alt in s.get("anyOf", []) + s.get("oneOf", []):
            if alt.get("type") and alt.get("type") != "null":
                t = alt["type"]
                break
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), "string")
    return t


def convert_param(value, typ):
    if value.startswith("\n"):
        value = value[1:]
    if value.endswith("\n"):
        value = value[:-1]
    if typ in (None, "string"):
        return value
    v = value.strip()
    try:
        if typ == "integer":
            return int(v)
        if typ == "number":
            f = float(v)
            return int(f) if f.is_integer() and "." not in v and "e" not in v.lower() else f
        if typ == "boolean":
            if v.lower() in ("true", "false"):
                return v.lower() == "true"
            return json.loads(v)
        return json.loads(v)  # object / array / null
    except (ValueError, json.JSONDecodeError):
        return value


_FN = re.compile(r"<function=([^>\n]+)>(.*?)(?:</function>|$)", re.S)
_PARAM = re.compile(r"<parameter=([^>\n]+)>(.*?)(?:</parameter>|(?=<parameter=)|$)", re.S)


def parse_tool_calls(tool_text, tool_schemas):
    calls = []
    for block in re.split(r"<tool_call>", tool_text)[1:]:
        block = block.split("</tool_call>", 1)[0]
        m = _FN.search(block)
        if not m:
            continue
        name, body = m.group(1).strip(), m.group(2)
        args = {}
        for pm in _PARAM.finditer(body):
            arg = pm.group(1).strip()
            args[arg] = convert_param(pm.group(2), _schema_type(tool_schemas, name, arg))
        if not args and body.strip().startswith("{"):  # a model that wrote JSON arguments
            try:
                args = json.loads(body.strip())
            except json.JSONDecodeError:
                pass
        calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
            }
        )
    return calls


# ------------------------------------------------------------------ sampling
def sample(logits, temperature, top_p, gen):
    if temperature is None or temperature <= 0:
        return int(logits.argmax())
    probs = torch.softmax(logits.float() / temperature, -1)
    if top_p is not None and 0 < top_p < 1:
        p, i = probs.sort(descending=True)
        keep = (p.cumsum(-1) - p) < top_p
        p = p * keep
        return int(i[torch.multinomial(p / p.sum(), 1, generator=gen)])
    return int(torch.multinomial(probs, 1, generator=gen))


# ------------------------------------------------------------------ engine
class BadRequest(Exception):
    pass


class Engine:
    """The model + the KV cache's token list. Runs only on the device thread."""

    def __init__(self, mesh_device, device_params):
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(str(LOCAL), trust_remote_code=True)
        self.cfg = MiMoTextConfig.from_json()
        self.C = CHUNK
        self.max_ctx = MAX_CTX
        max_seq = (math.ceil(MAX_CTX / CHUNK) + 1) * CHUNK
        t0 = time.perf_counter()
        self.model = TtMiMoModel(
            mesh_device,
            self.cfg,
            lambda i: layer_state(i, self.cfg),
            fabric_config=device_params["fabric_config"],
            max_seq_len=max_seq,
            chunk_size=CHUNK,
            layers=list(range(N_LAYERS)),
            global_state=global_state,
            lm_head=True,
            options=MiMoRuntimeOptions.from_env(),
        )
        logger.info(f"model built in {time.perf_counter() - t0:.0f} s (max_seq {max_seq}, chunk {CHUNK})")
        self.cached = []  # tokens whose KV is in the cache, positions 0..len-1
        self.stop_ids = {self.tok.eos_token_id}
        for t in ("<|im_end|>", "<|endoftext|>"):
            i = self.tok.convert_tokens_to_ids(t)
            if isinstance(i, int) and i >= 0 and i != self.tok.unk_token_id:
                self.stop_ids.add(i)

    def render(self, messages, tools, thinking):
        s = self.tok.apply_chat_template(
            normalize_messages(messages),
            tools=tools or None,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
        return self.tok(s, add_special_tokens=False)["input_ids"]

    def _chunk(self, ids, c):
        C = self.C
        end = min(len(ids), (c + 1) * C)
        chunk = torch.full((C,), PAD_TOKEN_ID, dtype=torch.long)
        chunk[: end - c * C] = torch.tensor(ids[c * C : end])
        x = self.model.prefill_chunk(chunk, c * C, valid_end=end)
        logits = self.model.next_token_logits(x, c * C, end - 1)
        x.deallocate(True)
        self.cached = list(ids[:end])
        return logits

    def prefill(self, ids):
        """-> (logits after ids[-1], tokens reused from the cache, chunks run)."""
        common = 0
        for a, b in zip(self.cached, ids):
            if a != b:
                break
            common += 1
        last = (len(ids) - 1) // self.C
        first = min(common // self.C, last)
        self.cached = self.cached[: first * self.C]
        for c in range(first, last):
            self._chunk(ids, c)
        logits = self._chunk(ids, last)
        return logits, first * self.C, last - first + 1

    def warmup(self, n):
        ids = self.tok("hello " * n, add_special_tokens=False)["input_ids"][:n]
        t0 = time.perf_counter()
        self.prefill(ids)
        logger.info(f"warmup: prefilled {len(ids)} tokens in {time.perf_counter() - t0:.1f} s")
        self.cached = []

    def run(self, job):
        """Generate for ``job``; pushes ('delta', reasoning, content) / ('done', response dict) / ('error', ...)."""
        r = job.req
        t_start = time.perf_counter()
        thinking = THINKING
        ctk = r.get("chat_template_kwargs") or {}
        if "enable_thinking" in ctk:
            thinking = bool(ctk["enable_thinking"])
        elif isinstance(r.get("enable_thinking"), bool):  # Qwen-style top-level flag
            thinking = r["enable_thinking"]
        if r.get("reasoning_effort") in ("none", "minimal"):
            thinking = False
        tools = r.get("tools") or None
        if r.get("tool_choice") == "none":
            tools = None
        ids = self.render(r.get("messages") or [], tools, thinking)
        if len(ids) >= self.max_ctx:
            raise BadRequest(f"prompt is {len(ids)} tokens; the server's max context is {self.max_ctx}")
        max_new = r.get("max_completion_tokens") or r.get("max_tokens") or MAX_TOKENS
        max_new = max(1, min(int(max_new), self.max_ctx - len(ids)))
        temperature = r.get("temperature", TEMPERATURE)
        top_p = r.get("top_p", TOP_P)
        gen = torch.Generator()
        gen.manual_seed(int(r["seed"]) if r.get("seed") is not None else int(time.time_ns() % (2**31)))
        stops = r.get("stop") or []
        stops = [stops] if isinstance(stops, str) else list(stops)
        tool_schemas = {
            (t.get("function") or {}).get("name"): (t.get("function") or {}).get("parameters") or {}
            for t in (tools or [])
        }

        if r.get("mimo_reset_cache"):  # non-standard: forget the prefix cache (to measure a cold prefill)
            self.cached = []
        logits, reused, n_chunks = self.prefill(ids)
        t_first = time.perf_counter()
        out, finish = [], None
        sent_r, sent_c = 0, 0
        text = ""
        # The thinking-off prompt already ends with <think></think>; the model's text is content.
        while True:
            t = sample(logits, temperature, top_p, gen)
            if t in self.stop_ids:
                finish = "stop"
                break
            out.append(t)
            text = self.tok.decode(out, skip_special_tokens=False)
            if stops and any(s in text for s in stops):
                text = text[: min(text.index(s) for s in stops if s in text)]
                finish = "stop"
                break
            if job.stream:
                safe = text[:-1] if text.endswith("�") else text
                rs, cs, tool = split_output(safe, final=False)
                if len(rs) > sent_r or len(cs) > sent_c:
                    job.events.put(("delta", rs[sent_r:], cs[sent_c:]))
                    sent_r, sent_c = max(sent_r, len(rs)), max(sent_c, len(cs))
            if job.cancelled.is_set():
                finish = "cancelled"
                break
            if len(out) >= max_new:
                finish = "length"
                break
            seq = ids + out
            logits = self._chunk(seq, (len(seq) - 1) // self.C)
        t_end = time.perf_counter()

        reasoning, content, tool_text = split_output(text, final=True)
        calls = parse_tool_calls(tool_text, tool_schemas) if tool_text else []
        if calls and finish == "stop":
            finish = "tool_calls"
        if job.stream:  # flush what the hold-back kept
            if len(reasoning) > sent_r or len(content) > sent_c:
                job.events.put(("delta", reasoning[sent_r:], content[sent_c:]))
        gen_s = t_end - t_first
        timings = {
            "prompt_tokens": len(ids),
            "cached_tokens": reused,
            "prefill_chunks": n_chunks,
            "ttft_s": round(t_first - t_start, 3),
            "completion_tokens": len(out),
            "decode_s": round(gen_s, 3),
            "decode_tok_s": round((len(out) - 1) / gen_s, 2) if len(out) > 1 and gen_s > 0 else None,
            "thinking": thinking,
        }
        logger.info(f"request {job.id}: {timings} finish={finish}")
        job.events.put(
            (
                "done",
                {
                    "reasoning": reasoning,
                    "content": content,
                    "tool_calls": calls,
                    "finish": "stop" if finish == "cancelled" else finish,
                    "usage": {
                        "prompt_tokens": len(ids),
                        "completion_tokens": len(out),
                        "total_tokens": len(ids) + len(out),
                        "prompt_tokens_details": {"cached_tokens": reused},
                    },
                    "timings": timings,
                },
            )
        )


class Job:
    def __init__(self, req):
        self.req = req
        self.stream = bool(req.get("stream"))
        self.id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        self.created = int(time.time())
        self.events = queue.Queue()
        self.cancelled = threading.Event()


JOBS = queue.Queue()


# ------------------------------------------------------------------ HTTP
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        logger.debug("http " + fmt % args)

    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, code, msg, typ="invalid_request_error"):
        self._json(code, {"error": {"message": msg, "type": typ, "code": code}})

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")
        if path in ("/v1/models", "/models"):
            self._json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": MODEL_ID,
                            "object": "model",
                            "created": 0,
                            "owned_by": "tenstorrent",
                            "max_model_len": MAX_CTX,
                        }
                    ],
                },
            )
        elif path in ("/health", "/v1/health"):
            self._json(200, {"status": "ok"})
        else:
            self._error(404, f"no route {self.path}")

    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")
        if path not in ("/v1/chat/completions", "/chat/completions"):
            return self._error(404, f"no route {self.path}")
        try:
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, json.JSONDecodeError) as e:
            return self._error(400, f"bad JSON: {e}")
        if not req.get("messages"):
            return self._error(400, "messages is required")
        job = Job(req)
        JOBS.put(job)
        if job.stream:
            self._stream(job)
        else:
            ev = job.events.get()
            if ev[0] == "error":
                return self._error(ev[1], ev[2])
            d = ev[1]
            msg = {"role": "assistant", "content": d["content"] or (None if d["tool_calls"] else "")}
            if d["reasoning"]:
                msg["reasoning_content"] = d["reasoning"]
            if d["tool_calls"]:
                msg["tool_calls"] = d["tool_calls"]
            self._json(
                200,
                {
                    "id": job.id,
                    "object": "chat.completion",
                    "created": job.created,
                    "model": req.get("model") or MODEL_ID,
                    "choices": [{"index": 0, "message": msg, "finish_reason": d["finish"]}],
                    "usage": d["usage"],
                    "timings": d["timings"],
                },
            )

    def _stream(self, job):
        model = job.req.get("model") or MODEL_ID
        include_usage = bool((job.req.get("stream_options") or {}).get("include_usage"))
        started = False

        def chunk(delta, finish=None, **extra):
            return {
                "id": job.id,
                "object": "chat.completion.chunk",
                "created": job.created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                **extra,
            }

        def send(obj):
            self.wfile.write(f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode())
            self.wfile.flush()

        try:
            while True:
                ev = job.events.get()
                if ev[0] == "error":
                    if not started:
                        return self._error(ev[1], ev[2])
                    send({"error": {"message": ev[2], "code": ev[1]}})
                    break
                if not started:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    send(chunk({"role": "assistant", "content": ""}))
                    started = True
                if ev[0] == "delta":
                    _, r, c = ev
                    d = {}
                    if r:
                        d["reasoning_content"] = r
                    if c:
                        d["content"] = c
                    if d:
                        send(chunk(d))
                elif ev[0] == "done":
                    d = ev[1]
                    for i, tc in enumerate(d["tool_calls"]):
                        send(chunk({"tool_calls": [{"index": i, **tc}]}))
                    send(chunk({}, d["finish"], timings=d["timings"]))
                    if include_usage:
                        send({**chunk({}), "choices": [], "usage": d["usage"]})
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                    break
        except (BrokenPipeError, ConnectionResetError):
            job.cancelled.set()
            logger.info(f"request {job.id}: client went away, cancelling")
            while True:  # drain until the engine finishes this job
                if job.events.get()[0] in ("done", "error"):
                    break
        self.close_connection = True


def serve_forever(mesh_device, device_params):
    """Build the model, start the HTTP thread, run jobs on this (the device) thread until SIGTERM / SIGINT."""
    engine = Engine(mesh_device, device_params)
    if WARMUP:
        engine.warmup(min(WARMUP, MAX_CTX - 1))
    stop = threading.Event()
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    pid_file = Path(ENV("MIMO_SERVE_PID", Path(__file__).parents[4] / "generated" / "mimo_serve" / "server.pid"))
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()))
    for s in (signal.SIGTERM, signal.SIGINT):
        signal.signal(s, lambda *_: stop.set())
    logger.info(
        f"MIMO_SERVE: listening on http://{HOST}:{PORT}/v1 (model {MODEL_ID}, max ctx {MAX_CTX}) pid {os.getpid()}"
    )
    try:
        while not stop.is_set():
            try:
                job = JOBS.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                engine.run(job)
            except BadRequest as e:
                job.events.put(("error", 400, str(e)))
            except Exception as e:  # keep serving; the device may need a restart if this was a device error
                logger.exception(f"request {job.id} failed")
                job.events.put(("error", 500, f"{type(e).__name__}: {e}"))
    finally:
        httpd.shutdown()
        pid_file.unlink(missing_ok=True)
        logger.info("MIMO_SERVE: stopped")
