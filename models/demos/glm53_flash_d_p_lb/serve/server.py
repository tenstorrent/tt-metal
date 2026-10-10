# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""OpenAI-style chat server for GLM-5.3-Flash on the LoudBox (2x4, all 45 layers), loaded once, for talking to the
model (long context, needle-in-a-haystack, degradation checks). stdlib http.server, one request at a time.

The model is the prefill engine's runtime (get_adapter("glm53_flash_d_p_lb"), the object tt-metal's prefill runner
drives), every token a prefill: decode via prefill re-runs the chunk holding the newest token from its boundary, the
KDA carries restored from the slot's boundary snapshot (tt/runners/adapter.py: _position). Prefix cache: the slot's
token list is kept; a request restarts at the latest chunk boundary the runtime holds state for (0, the snapshot,
or where the slot left off) at or before its first differing token.

    POST /v1/chat/completions  {"messages": [...], "max_tokens", "temperature", "top_p", "seed",
                                "reasoning_effort" (or chat_template_kwargs.reasoning_effort), "reset_cache"}
                               -> message.reasoning_content (before </think>) and message.content
    POST /v1/completions       {"prompt": str | [token ids], "max_tokens", ...}   (no template)
    POST /v1/tokenize          {"text": str} -> {"ids", "n"}
    GET  /health, /v1/models

Responses carry a non-standard "timings" (prompt tokens, reused, prefill s, s / token) and "token_ids".

Env: GLM_SERVE_PORT (8000), GLM_SERVE_HOST (0.0.0.0), GLM_SERVE_CHUNK (2048), GLM_SERVE_MAX_CTX (65536),
GLM_SERVE_EFFORT (low: the template's reasoning_effort, low / medium / high / max), GLM_SERVE_MAX_TOKENS (1024), GLM_SERVE_PID (pid file)."""

import json
import os
import queue
import signal
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from loguru import logger

ENV = os.environ.get
PORT = int(ENV("GLM_SERVE_PORT", "8000"))
HOST = ENV("GLM_SERVE_HOST", "0.0.0.0")
CHUNK = int(ENV("GLM_SERVE_CHUNK", "2048"))
MAX_CTX = int(ENV("GLM_SERVE_MAX_CTX", "65536"))
EFFORT = ENV("GLM_SERVE_EFFORT", "low")  # the template's reasoning_effort (the model always opens <think>)
MAX_TOKENS = int(ENV("GLM_SERVE_MAX_TOKENS", "1024"))
MODEL_ID = "glm-5.3-flash-tt"
ROOT = Path(__file__).resolve().parents[4]


class BadRequest(Exception):
    pass


class Engine:
    """The runtime on the device thread: prefix-cached prefill and greedy / sampled decode via prefill."""

    def __init__(self, mesh):
        from models.demos.common.bringup.testing.harness import spec
        from models.demos.common.bringup.testing.smoke import tokenizer
        from models.demos.common.prefill.adapter import PrefillRunParams, get_adapter
        from models.demos.glm53_flash_d_p.reference.weights import WeightLoader
        from models.demos.glm53_flash_d_p.tt.runners.adapter import resolve_model_path

        t0 = time.time()
        self.S = S = spec()
        os.environ.setdefault("PREFILL_SP", str(S.mesh[0]))
        os.environ.setdefault("PREFILL_TP", str(S.mesh[1]))
        adapter = get_adapter(S.get("contract.adapter", S.model))
        hf = adapter.load_hf_config()
        params = PrefillRunParams(
            mesh_shape=tuple(S.mesh),
            num_layers=hf.num_hidden_layers,
            first_layer_idx=0,
            is_first_rank=True,
            is_last_rank=True,
            max_seq_len=MAX_CTX,
            chunk_size=CHUNK,
            num_users=1,
            capacity_factor=1,
            num_links=int(S.get("contract.num_links") or 1),
            gate_mode_name=adapter.default_gate_mode,
            kv_only_last_layer=False,
            weight_cache_path=adapter.weight_cache_path(tuple(S.mesh)),
        )
        self.kv = adapter.allocate_kv_cache(mesh_device=mesh, hf_config=hf, params=params)
        t1 = time.time()
        self.rt = adapter.build_runtime(mesh_device=mesh, hf_config=hf, params=params)
        t2 = time.time()
        self.lm_head = WeightLoader(resolve_model_path()).get("lm_head.weight").float()
        self.tok = tokenizer(S)
        self.rt.hidden_sink = self._sink
        self.last = None
        self.cached = []  # the slot's tokens (all prefilled)
        stops = {self.tok.eos_token_id}
        for t in ("<|user|>", "<|observation|>", "<|endoftext|>"):
            i = self.tok.convert_tokens_to_ids(t)
            if isinstance(i, int) and i != self.tok.unk_token_id:
                stops.add(i)
        self.stops = {s for s in stops if s is not None}
        logger.info(
            f"GLM_SERVE: model up in {time.time() - t0:.0f} s (runtime {t2 - t1:.0f} s, kv {t1 - t0:.0f} s); chunk "
            f"{CHUNK}, max ctx {MAX_CTX}, stops {sorted(self.stops)}"
        )

    # -- device
    def _sink(self, h, slot, start, end):
        import ttnn
        from models.demos.glm53_flash_d_p.tt.common import replicated_to_host, split_to_host

        hidden = self.rt.model.final_norm(h)
        t = (split_to_host(hidden) if self.rt.model.layout == "split" else replicated_to_host(hidden)).float()
        ttnn.deallocate(hidden)
        self.last = torch.nn.functional.linear(t.reshape(-1, t.shape[-1])[end - 1 - start], self.lm_head)

    def _chunk(self, seq, start, end):
        import ttnn

        inp = self.rt.make_chunk_input(seq[start:end])
        self.rt.prefill_chunk(inp, self.kv, slot_id=0, actual_start=start, actual_end=end)
        ttnn.deallocate(inp)
        return self.last

    def prefill(self, ids):
        """-> (logits after ids[-1], tokens reused, chunks run). Restarts at the latest boundary the runtime holds
        KDA state for (0, its snapshot, or where the slot left off) at or before the first differing token."""
        if not ids or len(ids) >= MAX_CTX:
            raise BadRequest(f"prompt of {len(ids)} tokens: must be 1..{MAX_CTX - 1}")
        p = next((i for i, (a, b) in enumerate(zip(ids, self.cached)) if a != b), min(len(ids), len(self.cached)))
        p = min(p, len(ids) - 1)  # always run the chunk holding the last token (its logits)
        m = self.kv.marks[0]
        cands = [0, m["snap_at"]] + ([m["pos"]] if m["pos"] % CHUNK == 0 else [])
        start = max(c for c in cands if c <= p and c <= len(self.cached))
        logits, n = None, 0
        for a in range(start, len(ids), CHUNK):
            logits = self._chunk(ids, a, min(a + CHUNK, len(ids)))
            n += 1
        self.cached = list(ids)
        return logits, start, n

    def step(self, seq):
        """seq's last token is new: re-run the chunk holding it -> logits after it."""
        logits = self._chunk(seq, (len(seq) - 1) // CHUNK * CHUNK, len(seq))
        self.cached = list(seq)
        return logits

    # -- requests
    def render(self, messages, effort):
        text = self.tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, reasoning_effort=effort
        )
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def generate(self, ids, max_tokens, temperature=0.0, top_p=1.0, seed=None, reset=False):
        if reset:
            self.cached = []
        gen = torch.Generator().manual_seed(seed) if seed is not None else None
        t0 = time.time()
        logits, reused, n_chunks = self.prefill(ids)
        t_prefill = time.time() - t0
        out, finish, t1 = [], "length", time.time()
        while len(out) < max_tokens:
            t = sample(logits, temperature, top_p, gen)
            if t in self.stops:
                finish = "stop"
                break
            out.append(t)
            if len(ids) + len(out) >= MAX_CTX:
                break
            logits = self.step(ids + out)
        dt = time.time() - t1
        return (
            out,
            finish,
            {
                "prompt_tokens": len(ids),
                "reused_tokens": reused,
                "prefill_chunks": n_chunks,
                "prefill_s": round(t_prefill, 3),
                "completion_tokens": len(out),
                "decode_s": round(dt, 3),
                "s_per_token": round(dt / max(1, len(out)), 3),
            },
        )


def sample(logits, temperature, top_p, gen):
    if not temperature:
        return int(logits.argmax())
    probs = torch.softmax(logits.float() / temperature, -1)
    if top_p < 1.0:
        p, i = probs.sort(descending=True)
        keep = p.cumsum(-1) - p < top_p
        probs = torch.zeros_like(probs).scatter(0, i[keep], p[keep])
    return int(torch.multinomial(probs / probs.sum(), 1, generator=gen))


class Job:
    def __init__(self, kind, req):
        self.kind, self.req, self.id = kind, req, uuid.uuid4().hex[:12]
        self.done = threading.Event()
        self.result = None
        self.error = None


JOBS: "queue.Queue[Job]" = queue.Queue()
ENGINE: Engine = None


def run_job(job):
    r, e = job.req, ENGINE
    max_tokens = int(r.get("max_tokens") or r.get("max_completion_tokens") or MAX_TOKENS)
    kw = dict(
        temperature=float(r.get("temperature") or 0.0),
        top_p=float(r.get("top_p") or 1.0),
        seed=r.get("seed"),
        reset=bool(r.get("reset_cache")),
    )
    if job.kind == "tokenize":
        ids = e.tok(r["text"], add_special_tokens=False)["input_ids"]
        job.result = {"ids": ids, "n": len(ids)}
        return
    if job.kind == "chat":
        effort = r.get("reasoning_effort") or (r.get("chat_template_kwargs") or {}).get("reasoning_effort") or EFFORT
        ids = e.render(r["messages"], effort)
    else:
        p = r["prompt"]
        ids = list(p) if isinstance(p, list) else e.tok(p, add_special_tokens=False)["input_ids"]
    out, finish, timings = e.generate(ids, max_tokens, **kw)
    text = e.tok.decode(out)
    logger.info(f"GLM_SERVE: {job.kind} {job.id}: {timings}")
    if job.kind == "chat":
        think, sep, answer = text.partition("</think>")
        msg = {"role": "assistant", "content": answer.strip() if sep else "", "reasoning_content": think.strip()}
        choice = {"index": 0, "message": msg, "finish_reason": finish}
    else:
        choice = {"index": 0, "text": text, "finish_reason": finish}
    job.result = {
        "id": job.id,
        "object": "chat.completion" if job.kind == "chat" else "text_completion",
        "model": MODEL_ID,
        "choices": [choice],
        "usage": {
            "prompt_tokens": timings["prompt_tokens"],
            "completion_tokens": len(out),
            "total_tokens": timings["prompt_tokens"] + len(out),
        },
        "timings": timings,
        "token_ids": out,
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")
        if path in ("/health", "/v1/health"):
            self._json(200, {"status": "ok", "queued": JOBS.qsize()})
        elif path in ("/v1/models", "/models"):
            self._json(200, {"object": "list", "data": [{"id": MODEL_ID, "max_model_len": MAX_CTX}]})
        else:
            self._json(404, {"error": {"message": f"no route {path}"}})

    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")
        kind = {"/v1/chat/completions": "chat", "/v1/completions": "completion", "/v1/tokenize": "tokenize"}.get(path)
        if kind is None:
            return self._json(404, {"error": {"message": f"no route {path}"}})
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
        except json.JSONDecodeError as err:
            return self._json(400, {"error": {"message": f"bad JSON: {err}"}})
        if req.get("stream"):
            return self._json(400, {"error": {"message": "stream is not supported"}})
        job = Job(kind, req)
        JOBS.put(job)
        job.done.wait()
        if job.error:
            return self._json(job.error[0], {"error": {"message": job.error[1]}})
        self._json(200, job.result)


def serve_forever(mesh_device):
    global ENGINE
    ENGINE = Engine(mesh_device)
    stop = threading.Event()
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    pid_file = Path(ENV("GLM_SERVE_PID", ROOT / "generated" / "glm_serve" / "server.pid"))
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()))
    for s in (signal.SIGTERM, signal.SIGINT):
        signal.signal(s, lambda *_: stop.set())
    logger.info(f"GLM_SERVE: listening on http://{HOST}:{PORT}/v1 (max ctx {MAX_CTX}, chunk {CHUNK}) pid {os.getpid()}")
    try:
        while not stop.is_set():
            try:
                job = JOBS.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                run_job(job)
            except BadRequest as err:
                job.error = (400, str(err))
            except Exception as err:  # keep serving; a device error may need a restart
                logger.exception(f"request {job.id} failed")
                job.error = (500, f"{type(err).__name__}: {err}")
                ENGINE.cached = []
            finally:
                job.done.set()
    finally:
        httpd.shutdown()
        pid_file.unlink(missing_ok=True)
        logger.info("GLM_SERVE: stopped")
