# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible chat server for Xing4.0-29B-A4B whose every token goes through tt-d-gen's prefill engine
("decode via prefill"; serve/README.md). Slow by design: it hammers the engine's prefill path for correctness and
stability, it is not a decode server.

Three processes (this one supervises the other two):
  front end  (this process, python_env): HTTP, chat template, tokenizer, sampling. A request's generation loop:
             admit(prompt + tokens so far) to the engine daemon -> its PREFILL_DONE (slot, resident, position) ->
             the logits file the runner wrote for that slot's last chunk -> sample -> append -> admit again. The
             engine remounts the slot's resident prefix every step (block 64), so each token re-prefills
             [align_down((n - 1) // 64 * 64, 32), n) padded to one 5120-token chunk.
  runner     (serve/runner.py, python_env): tt-metal's prefill_runner on the 4x2 mesh (FABRIC_2D, D2H layer acks),
             plus the LM head on the chunk's last real row.
  engine     (serve/engine_daemon.py, tt-d-gen's Python): te.BackendRuntime, PREFILL role, for the server's life.

Every step is checked (counters under GET /stats, failures logged with "CHECK FAIL"): PREFILL_DONE at the prompt
length; resident a multiple of 32 and at most the reusable prefix cap ((n - 1) // 64 * 64); the logits file of the
chunk the server plan says is last (server_rules.chunk_plan(n, resident)); finite logits. At startup the spec's smoke
prompt must answer "Paris" (greedy) or the server does not open. While idle, a heartbeat prompt every
XING_SERVE_HEARTBEAT_S keeps the runner's dispatch timeout (180 s; it counts the wait for the next chunk) from
firing, and its greedy token must stay the same (counter heartbeat_mismatch).

    GET  /                      the chat page (chat.html): streaming chat + tt-d-gen telemetry
    POST /v1/chat/completions   stream / non-stream; tools -> tool_calls; <think> -> reasoning_content
    POST /v1/completions        raw prompt (no chat template)
    GET  /v1/models, /health, /stats (/stats: our checks, the pools, and every scalar of tt-d-gen's TelemetrySnapshot)
Every response's `timings.dgen` is what tt-d-gen reported per step (ADMITTED reused_tokens / resident prefix / slot);
streamed chunks carry `xing_progress` (tokens so far, the last step's slot and reuse). Requests carry an
`X-Xing-Pool` header (web, the default, or hammer) that caps each pool's concurrent generations (XING_SERVE_POOLS).

Env: XING_SERVE_PORT (8000), XING_SERVE_HOST (0.0.0.0), XING_SERVE_SLOTS (4: engine max_slots = PREFILL_NUM_USERS),
XING_SERVE_MAX_TOKENS (1024: default completion cap), XING_SERVE_TEMPERATURE / XING_SERVE_TOP_P (0 / 1: greedy when
the request sets none), XING_SERVE_THINKING (0: default enable_thinking), XING_SERVE_HEARTBEAT_S (60),
XING_SERVE_DIR (generated/xing_serve: logs, pid, status), XING_SERVE_MODEL_ID (xing4.0-29b-a4b),
XING_SERVE_READY_S (3600: runner load + compile + engine attach), XING_SERVE_STEP_S (600: one step's bound),
XING_SERVE_POOLS (web:8,hammer:8).
The bring-up env applies: BRINGUP_SPEC, BRINGUP_HF (checkpoint), BRINGUP_SERVER_REPO (tt-d-gen checkout).
"""

from __future__ import annotations

import glob
import json
import os
import queue
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

ENV = os.environ.get
ROOT = Path(__file__).resolve().parents[4]
PORT = int(ENV("XING_SERVE_PORT", "8000"))
HOST = ENV("XING_SERVE_HOST", "0.0.0.0")
SLOTS = int(ENV("XING_SERVE_SLOTS", "4"))
MAX_TOKENS = int(ENV("XING_SERVE_MAX_TOKENS", "1024"))
TEMPERATURE = float(ENV("XING_SERVE_TEMPERATURE", "0"))
TOP_P = float(ENV("XING_SERVE_TOP_P", "1"))
THINKING = ENV("XING_SERVE_THINKING", "0") != "0"
HEARTBEAT_S = float(ENV("XING_SERVE_HEARTBEAT_S", "60"))
SERVE_DIR = Path(ENV("XING_SERVE_DIR", str(ROOT / "generated" / "xing_serve")))
MODEL_ID = ENV("XING_SERVE_MODEL_ID", "xing4.0-29b-a4b")
READY_S = float(ENV("XING_SERVE_READY_S", "3600"))
STEP_S = float(ENV("XING_SERVE_STEP_S", "600"))
# Concurrent generations per pool (X-Xing-Pool header; untagged requests are "web"). The engine picks slots itself,
# so this caps what each pool has in flight; cached prefixes still share the engine's eviction over all slots.
POOLS = {k: int(v) for k, v in (p.split(":") for p in ENV("XING_SERVE_POOLS", "web:8,hammer:8").split(","))}
CHAT_HTML = Path(__file__).with_name("chat.html")

THINK_OPEN, THINK_CLOSE, TOOL_OPEN = "<think>", "</think>", "<tool_call>"
HEARTBEAT_TEXT = "Reply with the single word: ok."


def log(msg: str) -> None:
    print(f"[xing serve] {time.strftime('%H:%M:%S')} {msg}", flush=True)


class BadRequest(Exception):
    pass


class StackDown(Exception):
    pass


# ------------------------------------------------------------------ prompt / output (Xing chat template)
def normalize_messages(messages):
    """OpenAI messages -> what the Xing chat template expects (tool-call arguments as dicts, developer -> system)."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("role") == "developer":
            m["role"] = "system"
        if m.get("content") is None:
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
        out.append(m)
    return out


def _hold(s, tag):
    """Length of the longest suffix of ``s`` that is a proper prefix of ``tag`` (held back while streaming)."""
    for k in range(min(len(tag) - 1, len(s)), 0, -1):
        if s.endswith(tag[:k]):
            return k
    return 0


def split_output(text, thinking, final):
    """Generated text -> (reasoning, content, tool_text | None). With thinking on, the prompt ends in "<think>\\n",
    so the text starts inside the reasoning. Not ``final``: trailing partial tags held back."""
    reasoning, rest = "", text
    if thinking:
        if THINK_CLOSE in text:
            reasoning, rest = text.split(THINK_CLOSE, 1)
        else:
            reasoning = text if final else text[: len(text) - _hold(text, THINK_CLOSE)]
            return (reasoning.strip("\n") if final else reasoning.lstrip("\n")), "", None
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


_PARAM = re.compile(r"<param_key>(.*?)</param_key>\s*<param_value>(.*?)(?:</param_value>|$)", re.S)


def parse_tool_calls(tool_text, tool_schemas):
    """Xing's format: <tool_call>NAME<param_key>k</param_key><param_value>v</param_value>...</tool_call>."""
    calls = []
    for block in tool_text.split(TOOL_OPEN)[1:]:
        block = block.split("</tool_call>", 1)[0]
        name = block.split("<param_key>", 1)[0].strip()
        if not name:
            continue
        props = ((tool_schemas.get(name) or {}).get("properties")) or {}
        args = {}
        for k, v in _PARAM.findall(block):
            k = k.strip()
            typ = (props.get(k) or {}).get("type")
            if typ not in (None, "string"):
                try:
                    v = json.loads(v)
                except json.JSONDecodeError:
                    pass
            args[k] = v
        calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
            }
        )
    return calls


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


# ------------------------------------------------------------------ the stack
class Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.c = {
            "steps": 0,
            "requests": 0,
            "tokens": 0,
            "remounts": 0,
            "cold": 0,
            "check_fail": 0,
            "engine_errors": 0,
            "heartbeats": 0,
            "heartbeat_mismatch": 0,
        }
        self.step_s = []  # last 1000 step latencies
        self.fails = []  # last 50 check failures

    def add(self, **kw):
        with self.lock:
            for k, v in kw.items():
                self.c[k] = self.c.get(k, 0) + v

    def fail(self, what):
        log(f"CHECK FAIL: {what}")
        with self.lock:
            self.c["check_fail"] += 1
            self.fails = (self.fails + [f"{time.strftime('%H:%M:%S')} {what}"])[-50:]

    def step(self, s):
        with self.lock:
            self.step_s = (self.step_s + [s])[-1000:]

    def snapshot(self):
        with self.lock:
            st = sorted(self.step_s)
            pct = (lambda q: round(st[min(len(st) - 1, int(q * len(st)))], 3)) if st else (lambda q: None)
            return {**self.c, "step_s_p50": pct(0.5), "step_s_p99": pct(0.99), "recent_failures": list(self.fails)}


class Stack:
    """Starts and watches the runner (which starts the engine daemon); admits through the daemon's socket."""

    def __init__(self):
        from models.demos.common.bringup.testing import dgen_engine as D
        from models.demos.xing40_a4b_d_p.tests.bringup.contract import server_rules as R
        from models.demos.xing40_a4b_d_p.tests.bringup.contract.test_runner_contract import runner_env

        self.R = R
        self.dir = SERVE_DIR
        self.dir.mkdir(parents=True, exist_ok=True)
        build, why = D.find_build()
        if build is None:
            raise RuntimeError(f"no tt-d-gen engine build ({why}); set BRINGUP_SERVER_REPO (agents/dgen-build.md)")
        self.build = build
        sid = f"xing_serve_{os.getpid()}"
        self.sock = f"/dev/shm/{sid}.sock"  # AF_UNIX paths are capped at 108 bytes
        self.logits_dir = Path(f"/dev/shm/{sid}_logits")
        self.logits_dir.mkdir(parents=True, exist_ok=True)
        cfg = {
            "socket": self.sock,
            "service_id": sid,
            "ack_shm_name": f"/tt_prefill_layer_acks_{sid}",  # prefill_runner.py _serve_request
            "chunk_size": R.CHUNK,
            "layers_per_chunk": R.NUM_LAYERS,
            "sp_factor": R.SP,
            "max_slots": SLOTS,
            "max_seq_len": R.MAX_SEQ,
            "kv_block_size": R.KV_BLOCK,
            "connect_timeout_ms": 600000,
            "stall_s": STEP_S,
            "status": str(self.dir / "engine_status.json"),
            "pidfile": str(self.dir / "engine.pid"),
        }
        self.cfg_path = self.dir / "engine_config.json"
        self.cfg_path.write_text(json.dumps(cfg, indent=1))
        daemon = Path(__file__).with_name("engine_daemon.py")
        drv = " ".join(_q(a) for a in build.argv()[:-1] + [str(daemon), str(self.cfg_path)])
        snd = " ".join(
            _q(a) for a in (sys.executable, "-m", "models.demos.common.bringup.testing.dgen_engine", "shutdown", sid)
        )
        rc = _q(str(self.dir / "engine.rc"))
        daemon_cmd = ["/bin/sh", "-c", f"{drv}; rc=$?; {snd}; echo $rc > {rc}"]
        trace = self.dir / "trace"
        trace.mkdir(exist_ok=True)
        (trace / "metadata.json").write_text(json.dumps({"token_ids": [1]}))
        env = runner_env(self.dir, trace)
        env.update(
            PREFILL_NUM_USERS=str(SLOTS),
            PREFILL_H2D_SERVICE_ID=sid,
            XING_SERVE_DAEMON_CMD=json.dumps(daemon_cmd),
            XING_SERVE_LOGITS_DIR=str(self.logits_dir),
        )
        for f in ("engine.rc", "engine.log", "engine_status.json", "engine.pid"):
            (self.dir / f).unlink(missing_ok=True)
        if os.path.exists(self.sock):
            os.unlink(self.sock)
        self.runner_log = self.dir / "runner.log"
        log(f"feeder: {build.describe()}; slots {SLOTS}; logs {self.runner_log}, {self.dir / 'engine.log'}")
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "models.demos.xing40_a4b_d_p.serve.runner", str(self.dir)],
            env=env,
            stdout=self.runner_log.open("w"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.down = None  # why the stack is down
        self.last_admit = time.monotonic()
        self.inflight = 0
        self.lock = threading.Lock()

    # -- lifecycle
    def wait_ready(self):
        t0 = time.monotonic()
        while not os.path.exists(self.sock):
            self.check_alive()
            if time.monotonic() - t0 > READY_S:
                raise StackDown(f"engine not ready after {READY_S:.0f} s (see {self.runner_log})")
            time.sleep(2.0)
        log(f"engine daemon ready after {time.monotonic() - t0:.0f} s")

    def check_alive(self):
        if self.down:
            raise StackDown(self.down)
        rcf = self.dir / "engine.rc"
        if self.proc.poll() is not None:
            self.down = f"runner exited {self.proc.returncode} (see {self.runner_log})"
        elif rcf.exists() and rcf.read_text().strip():
            self.down = f"engine daemon exited {rcf.read_text().strip()} (see {self.dir / 'engine.log'})"
        if self.down:
            log(f"STACK DOWN: {self.down}")
            raise StackDown(self.down)

    def stop(self):
        """SIGTERM the daemon (rt.stop(), then the sh wrapper sends the runner's shutdown sentinel), wait for the
        runner to leave its loop and close the mesh; kill the group if it does not."""
        if self.proc.poll() is None:
            pidf = self.dir / "engine.pid"
            try:
                os.kill(int(pidf.read_text()), signal.SIGTERM)
            except (FileNotFoundError, ValueError, ProcessLookupError):
                log("no engine daemon to stop; killing the runner's process group")
                os.killpg(self.proc.pid, signal.SIGKILL)
            try:
                self.proc.wait(timeout=300)
            except subprocess.TimeoutExpired:
                log("runner did not exit in 300 s; killing its process group")
                for sig in (signal.SIGTERM, signal.SIGKILL):
                    try:
                        os.killpg(self.proc.pid, sig)
                        self.proc.wait(timeout=30)
                        break
                    except (ProcessLookupError, subprocess.TimeoutExpired):
                        continue
        for p in self.logits_dir.glob("*"):
            p.unlink(missing_ok=True)
        self.logits_dir.rmdir()
        log(f"stack stopped (runner exit {self.proc.returncode})")

    # -- one step
    def _call(self, req: dict) -> dict:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(STEP_S)
        try:
            s.connect(self.sock)
            s.sendall((json.dumps(req) + "\n").encode())
            buf = b""
            while not buf.endswith(b"\n"):
                part = s.recv(65536)
                if not part:
                    break
                buf += part
        except OSError as e:
            self.check_alive()
            raise StackDown(f"engine socket: {type(e).__name__}: {e}")
        finally:
            s.close()
        if not buf:
            self.check_alive()
            raise StackDown("engine daemon closed the connection")
        return json.loads(buf)

    def stats(self) -> dict:
        try:
            return self._call({"op": "stats"})
        except StackDown as e:
            return {"ok": False, "error": str(e)}

    def step(self, ids: list[int], stats: Stats) -> tuple[torch.Tensor, dict]:
        """Prefill ``ids`` through the engine; -> (logits after ids[-1], engine record)."""
        R = self.R
        self.check_alive()
        n = len(ids)
        with self.lock:
            self.inflight += 1
            self.last_admit = time.monotonic()
        t_admit = time.time_ns()
        t0 = time.monotonic()
        try:
            r = self._call({"op": "admit", "token_ids": ids})
        finally:
            with self.lock:
                self.inflight -= 1
                self.last_admit = time.monotonic()
        if not r.get("ok"):
            stats.add(engine_errors=1)
            self.check_alive()
            raise RuntimeError(f"engine: {r.get('error')}")
        slot, res = r["slot"], r.get("resident", 0)
        stats.add(steps=1, remounts=int(res > 0), cold=int(res == 0))
        cap = R.reusable_prefix_cap(R.KV_BLOCK, n)
        if res % R.TILE or res > cap:
            stats.fail(f"slot {slot}: resident {res} for a {n}-token prompt (tile-aligned, <= cap {cap})")
        last_start = R.chunk_plan(n, res)[-1][0]
        logits, head = self._take_logits(slot, last_start, n, t_admit)
        if not head["finite"]:
            stats.fail(f"slot {slot} [{last_start}, {n}): non-finite logits")
        step_s = time.monotonic() - t0
        stats.step(step_s)
        engine_s = r["done_s"] - r["admit_s"] if "done_s" in r and "admit_s" in r else None
        return logits, {
            **r,
            "last_chunk": [last_start, n],
            "argmax": head["argmax"],
            "step_s": step_s,
            "engine_s": engine_s,
        }

    def _take_logits(self, slot, start, end, t_admit) -> tuple[torch.Tensor, dict]:
        """The runner's logits file for this request's last chunk: (slot, start, end), written after the admit; the
        oldest such, claimed by an atomic rename (two requests never take one file)."""
        deadline = time.monotonic() + STEP_S
        pat = str(self.logits_dir / f"s{slot}_e{end}_*.bin")
        while True:
            cands = []
            for p in glob.glob(pat):
                try:
                    with open(p, "rb") as f:
                        head = json.loads(f.readline())
                except (FileNotFoundError, json.JSONDecodeError):
                    continue
                if head["t_ns"] >= t_admit and head["start"] == start:
                    cands.append((head["seq"], p, head))
            for _, p, head in sorted(cands):
                mine = f"{p}.{os.getpid()}.{threading.get_ident()}"
                try:
                    os.rename(p, mine)
                except FileNotFoundError:
                    continue  # another request claimed it
                with open(mine, "rb") as f:
                    f.readline()
                    data = f.read()
                os.unlink(mine)
                return torch.frombuffer(bytearray(data), dtype=torch.float32), head
            if time.monotonic() > deadline:
                raise StackDown(f"no logits for slot {slot} chunk [{start}, {end}) after {STEP_S:.0f} s")
            self.check_alive()
            time.sleep(0.005)


def _q(a: str) -> str:
    import shlex

    return shlex.quote(a)


# ------------------------------------------------------------------ generation
class Engine:
    """Tokenizer + the stack; generation loops run on the HTTP threads (concurrent requests interleave in the
    engine)."""

    def __init__(self, stack: Stack):
        from models.demos.common.bringup.testing.harness import spec
        from models.demos.common.bringup.testing.smoke import tokenizer

        self.spec = spec()
        self.tok = tokenizer(self.spec)
        self.stack = stack
        self.stats = Stats()
        self.max_seq = stack.R.MAX_SEQ
        self.stop_ids = {t for t in (self.tok.eos_token_id,) if t is not None}
        for t in ("<_end>",):
            i = self.tok.convert_tokens_to_ids(t)
            if isinstance(i, int) and i >= 0 and i != self.tok.unk_token_id:
                self.stop_ids.add(i)
        self.hb_token = None

    def render(self, messages, tools, thinking):
        s = self.tok.apply_chat_template(
            normalize_messages(messages),
            tools=tools or None,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
        return self.tok(s, add_special_tokens=False)["input_ids"]

    def generate(self, ids, max_new, temperature, top_p, gen, stops=(), on_text=None, cancelled=None):
        """-> (out ids, text, finish, per-step records)."""
        if len(ids) >= self.max_seq:
            raise BadRequest(f"prompt is {len(ids)} tokens; the server's max context is {self.max_seq}")
        max_new = max(1, min(int(max_new), self.max_seq - len(ids)))
        out, steps, text, finish = [], [], "", None
        while True:
            logits, rec = self.stack.step(ids + out, self.stats)
            steps.append(
                {
                    k: rec.get(k)
                    for k in ("slot", "resident", "reused", "prompt_len", "last_chunk", "step_s", "engine_s")
                }
            )
            t = sample(logits, temperature, top_p, gen)
            if t in self.stop_ids:
                finish = "stop"
                break
            out.append(t)
            self.stats.add(tokens=1)
            text = self.tok.decode(out, skip_special_tokens=False)
            if stops and any(s in text for s in stops):
                text = text[: min(text.index(s) for s in stops if s in text)]
                finish = "stop"
                break
            if on_text:
                on_text(text, steps)
            if cancelled is not None and cancelled.is_set():
                finish = "cancelled"
                break
            if len(out) >= max_new:
                finish = "length"
                break
        return out, text, finish, steps

    def self_check(self):
        """The spec's smoke prompt, greedy, must answer as the bring-up expects ("Paris")."""
        from models.demos.common.bringup.testing.smoke import prompt_ids

        smoke = self.spec.data["intake"]["smoke"]
        t0 = time.time()
        out, text, finish, steps = self.generate(prompt_ids(self.spec, self.tok), 8, 0, None, None)
        ok = smoke["expect"].lower() in text.lower()
        log(f"self-check {smoke['prompt']!r} -> {text!r} ({finish}, {len(steps)} steps, {time.time() - t0:.0f} s)")
        if not ok:
            raise StackDown(f"self-check failed: {text!r} does not contain {smoke['expect']!r}")

    def heartbeat(self):
        ids = self.render([{"role": "user", "content": HEARTBEAT_TEXT}], None, False)
        logits, rec = self.stack.step(ids, self.stats)
        t = int(logits.argmax())
        self.stats.add(heartbeats=1)
        if self.hb_token is None:
            self.hb_token = t
        elif t != self.hb_token:
            self.stats.add(heartbeat_mismatch=1)
            self.stats.fail(
                f"heartbeat greedy token {t} != first {self.hb_token} (slot {rec['slot']}, "
                f"resident {rec.get('resident')})"
            )

    def run(self, job):
        r = job.req
        t_start = time.perf_counter()
        thinking = THINKING
        ctk = r.get("chat_template_kwargs") or {}
        if "enable_thinking" in ctk:
            thinking = bool(ctk["enable_thinking"])
        elif isinstance(r.get("enable_thinking"), bool):
            thinking = r["enable_thinking"]
        if r.get("reasoning_effort") in ("none", "minimal"):
            thinking = False
        tools = r.get("tools") or None
        if r.get("tool_choice") == "none":
            tools = None
        if job.raw:
            if not isinstance(r.get("prompt"), str):
                raise BadRequest("prompt (a string) is required")
            ids = self.tok(r["prompt"], add_special_tokens=False)["input_ids"]
            thinking = False
        else:
            ids = self.render(r.get("messages") or [], tools, thinking)
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
        sent = {"r": 0, "c": 0}

        def on_text(text, steps):
            if not job.stream:
                return
            st = steps[-1]
            progress = {
                "tokens": len(steps),
                "elapsed_s": round(time.perf_counter() - t_start, 3),
                "step_s": round(st["step_s"], 3),
                "slot": st["slot"],
                "reused": st["reused"],
                "prompt_len": st["prompt_len"],
            }
            safe = text[:-1] if text.endswith("�") else text
            if job.raw:
                rs, cs = "", safe
            else:
                rs, cs, _ = split_output(safe, thinking, final=False)
            job.events.put(("delta", rs[sent["r"] :], cs[sent["c"] :], progress))
            sent["r"], sent["c"] = max(sent["r"], len(rs)), max(sent["c"], len(cs))

        self.stats.add(requests=1)
        max_new = r.get("max_completion_tokens") or r.get("max_tokens") or MAX_TOKENS
        out, text, finish, steps = self.generate(ids, max_new, temperature, top_p, gen, stops, on_text, job.cancelled)
        t_end = time.perf_counter()
        if job.raw:
            reasoning, content, calls = "", text, []
        else:
            reasoning, content, tool_text = split_output(text, thinking, final=True)
            calls = parse_tool_calls(tool_text, tool_schemas) if tool_text else []
        if calls and finish == "stop":
            finish = "tool_calls"
        if job.stream and (len(reasoning) > sent["r"] or len(content) > sent["c"]):
            job.events.put(("delta", reasoning[sent["r"] :], content[sent["c"] :], None))
        timings = {
            "prompt_tokens": len(ids),
            "completion_tokens": len(out),
            "steps": len(steps),
            "first_step": steps[0] if steps else None,
            "total_s": round(t_end - t_start, 3),
            "s_per_token": round((t_end - t_start) / max(1, len(steps)), 3),
            "slots": sorted({s["slot"] for s in steps}),
            "thinking": thinking,
            "pool": job.pool,
            "dgen": dgen_summary(steps, t_end - t_start),
        }
        log(
            f"request {job.id}: {len(ids)} + {len(out)} tokens in {timings['total_s']} s, finish {finish}, "
            f"slots {timings['slots']}"
        )
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
                        "prompt_tokens_details": {"cached_tokens": steps[0]["resident"] if steps else 0},
                    },
                    "timings": timings,
                    "token_ids": out,
                },
            )
        )


def dgen_summary(steps, total_s):
    """What tt-d-gen reported for one request, from its ADMITTED events (reused_tokens: the prompt tokens it served
    from a cached prefix; position_id: the resident prefix it remounted). Step 0 is the prompt the user sent; every
    later step is one generated token re-admitted as prompt + tokens so far. tok/s is ours (the prefill role has no
    token counter)."""
    if not steps:
        return None
    p, dec = steps[0], steps[1:]
    return {
        "prompt": {
            "tokens": p["prompt_len"],
            "reused": p["reused"],
            "resident": p["resident"],
            "hit": bool(p["reused"]),
            "computed": p["prompt_len"] - p["resident"],
            "slot": p["slot"],
            "step_s": round(p["step_s"], 3),
        },
        "decode": {
            "steps": len(dec),
            "hits": sum(1 for s in dec if s["reused"]),
            "reused": sum(s["reused"] for s in dec),
            "prompt_tokens": sum(s["prompt_len"] for s in dec),
            "slots": sorted({s["slot"] for s in dec}),
        },
        "tok_s": round(len(steps) / total_s, 3) if total_s > 0 else None,
        "step_reused": [s["reused"] for s in steps[:4096]],
        "step_s": [round(s["step_s"], 3) for s in steps[:4096]],
    }


class Pool:
    def __init__(self, n):
        self.sem = threading.Semaphore(n)
        self.size, self.active, self.waiting = n, 0, 0
        self.lock = threading.Lock()


POOL = {k: Pool(v) for k, v in POOLS.items()}


class Job:
    def __init__(self, req, raw=False, pool="web"):
        self.req = req
        self.raw = raw
        self.pool = pool
        self.stream = bool(req.get("stream"))
        self.id = f"{'cmpl' if raw else 'chatcmpl'}-{uuid.uuid4().hex[:24]}"
        self.created = int(time.time())
        self.events = queue.Queue()
        self.cancelled = threading.Event()


ENGINE: Engine | None = None


def run_job(job):
    pool = POOL[job.pool]
    with pool.lock:
        pool.waiting += 1
    pool.sem.acquire()
    with pool.lock:
        pool.waiting -= 1
        pool.active += 1
    try:
        ENGINE.run(job)
    except BadRequest as e:
        job.events.put(("error", 400, str(e)))
    except StackDown as e:
        job.events.put(("error", 503, f"stack down: {e}"))
    except Exception as e:
        import traceback

        log(f"request {job.id} failed: {traceback.format_exc()}")
        job.events.put(("error", 500, f"{type(e).__name__}: {e}"))
    finally:
        with pool.lock:
            pool.active -= 1
        pool.sem.release()


# ------------------------------------------------------------------ HTTP
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

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
        if path in ("", "/chat", "/index.html"):
            body = CHAT_HTML.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)
        elif path in ("/v1/models", "/models"):
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
                            "max_model_len": ENGINE.max_seq,
                        }
                    ],
                },
            )
        elif path in ("/health", "/v1/health"):
            down = ENGINE.stack.down
            self._json(503 if down else 200, {"status": "down" if down else "ok", "reason": down})
        elif path in ("/stats", "/v1/stats"):
            self._json(
                200,
                {
                    "server": ENGINE.stats.snapshot(),
                    "engine": ENGINE.stack.stats(),
                    "feeder": ENGINE.stack.build.describe(),
                    "down": ENGINE.stack.down,
                    "pools": {k: {"size": p.size, "active": p.active, "waiting": p.waiting} for k, p in POOL.items()},
                    "slots": SLOTS,
                },
            )
        else:
            self._error(404, f"no route {self.path}")

    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")
        if path in ("/v1/chat/completions", "/chat/completions"):
            raw = False
        elif path in ("/v1/completions", "/completions"):
            raw = True
        else:
            return self._error(404, f"no route {self.path}")
        try:
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, json.JSONDecodeError) as e:
            return self._error(400, f"bad JSON: {e}")
        if not raw and not req.get("messages"):
            return self._error(400, "messages is required")
        pool = (self.headers.get("X-Xing-Pool") or "web").strip()
        if pool not in POOL:
            return self._error(400, f"unknown pool {pool!r} (pools: {sorted(POOL)})")
        job = Job(req, raw, pool)
        threading.Thread(target=run_job, args=(job,), daemon=True).start()
        if job.stream:
            return self._stream(job)
        ev = job.events.get()
        if ev[0] == "error":
            return self._error(ev[1], ev[2])
        d = ev[1]
        if raw:
            choice = {"index": 0, "text": d["content"], "finish_reason": d["finish"], "token_ids": d["token_ids"]}
            obj = "text_completion"
        else:
            msg = {"role": "assistant", "content": d["content"] or (None if d["tool_calls"] else "")}
            if d["reasoning"]:
                msg["reasoning_content"] = d["reasoning"]
            if d["tool_calls"]:
                msg["tool_calls"] = d["tool_calls"]
            choice = {"index": 0, "message": msg, "finish_reason": d["finish"], "token_ids": d["token_ids"]}
            obj = "chat.completion"
        self._json(
            200,
            {
                "id": job.id,
                "object": obj,
                "created": job.created,
                "model": req.get("model") or MODEL_ID,
                "choices": [choice],
                "usage": d["usage"],
                "timings": d["timings"],
            },
        )

    def _stream(self, job):
        model = job.req.get("model") or MODEL_ID
        include_usage = bool((job.req.get("stream_options") or {}).get("include_usage"))
        started = False

        def chunk(delta, finish=None, **extra):
            if job.raw:
                return {
                    "id": job.id,
                    "object": "text_completion",
                    "created": job.created,
                    "model": model,
                    "choices": [{"index": 0, "text": delta.get("content", ""), "finish_reason": finish}],
                    **extra,
                }
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
                    if not job.raw:
                        send(chunk({"role": "assistant", "content": ""}))
                    started = True
                if ev[0] == "delta":
                    _, r, c, progress = ev
                    d = {}
                    if r:
                        d["reasoning_content"] = r
                    if c:
                        d["content"] = c
                    if d or progress:
                        send(chunk(d, **({"xing_progress": progress} if progress else {})))
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
            log(f"request {job.id}: client went away, cancelling")
        self.close_connection = True


def serve_forever():
    """Start the stack, self-check, serve HTTP until SIGTERM / SIGINT (serve/stop.sh) or the stack goes down."""
    global ENGINE
    stop = threading.Event()
    for s in (signal.SIGTERM, signal.SIGINT):
        signal.signal(s, lambda *_: stop.set())
    SERVE_DIR.mkdir(parents=True, exist_ok=True)
    pid_file = SERVE_DIR / "server.pid"
    pid_file.write_text(str(os.getpid()))
    stack = Stack()
    httpd = None
    try:
        ENGINE = Engine(stack)
        stack.wait_ready()
        ENGINE.self_check()
        httpd = ThreadingHTTPServer((HOST, PORT), Handler)
        httpd.daemon_threads = True
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        log(
            f"XING_SERVE: listening on http://{HOST}:{PORT}/v1 (model {MODEL_ID}, max ctx {ENGINE.max_seq}, "
            f"slots {SLOTS}) pid {os.getpid()}; feeder {stack.build.describe()}"
        )
        last_status = 0.0
        while not stop.is_set():
            stop.wait(1.0)
            stack.check_alive()
            now = time.monotonic()
            with stack.lock:
                idle = stack.inflight == 0 and now - stack.last_admit > HEARTBEAT_S
            if idle:
                ENGINE.heartbeat()
            if now - last_status > 30:
                last_status = now
                (SERVE_DIR / "stats.json").write_text(json.dumps(ENGINE.stats.snapshot(), indent=1))
    except StackDown as e:
        log(f"XING_SERVE: stopping, stack down: {e}")
        raise
    finally:
        if httpd is not None:
            httpd.shutdown()
        if ENGINE is not None:
            (SERVE_DIR / "stats.json").write_text(json.dumps(ENGINE.stats.snapshot(), indent=1))
            log(f"final stats: {json.dumps(ENGINE.stats.snapshot())}")
        stack.stop()
        pid_file.unlink(missing_ok=True)
        log("XING_SERVE: stopped")
