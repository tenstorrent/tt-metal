"""CPU tests for emb_serve_fastapi.py (no chip): batching logic, tokenizer thread safety, trace-shape closure, and the
server's start-up / worker-death / device-hang / shutdown handling with a stub worker (EMB_STUB_WORKER).

Run (source env_emb.sh first, for the tokenizer cache): cd scripts && python3 -m pytest -q test_emb_serve.py
"""

import os
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import requests

from emb_batcher import collect_batch, right_pad

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- batching (pattern 3)
def _getter(q):
    def get(timeout):
        if timeout is None:
            return q.get()
        if timeout == 0:
            return q.get_nowait()
        return q.get(timeout=timeout)

    return get


def test_collect_takes_only_queued_without_wait():
    q = queue.Queue()
    for i in range(5):
        q.put((i, [1, 2]))
    tasks, stop = collect_batch(_getter(q), 0.0, 32)
    assert [t[0] for t in tasks] == [0, 1, 2, 3, 4] and not stop


def test_collect_caps_at_max_texts():
    q = queue.Queue()
    for i in range(10):
        q.put((i, [1]))
    tasks, _ = collect_batch(_getter(q), 0.0, 4)
    assert len(tasks) == 4 and q.qsize() == 6


def test_collect_stop_sentinel_after_tasks():
    q = queue.Queue()
    q.put((0, [1]))
    q.put(None)
    tasks, stop = collect_batch(_getter(q), 0.0, 32)
    assert len(tasks) == 1 and stop


def test_collect_stop_first():
    q = queue.Queue()
    q.put(None)
    assert collect_batch(_getter(q), 0.0, 32) == ([], True)


def test_collect_waits_for_late_tasks():
    q = queue.Queue()
    q.put((0, [1]))
    threading.Timer(0.05, q.put, args=((1, [1]),)).start()
    tasks, _ = collect_batch(_getter(q), 0.5, 2)
    assert len(tasks) == 2


def test_right_pad():
    ids, mask = right_pad([[5, 6, 7], [8]])
    assert ids == [[5, 6, 7], [8, 0, 0]] and mask == [[1, 1, 1], [1, 0, 0]]


# --------------------------------------------------------------------------- tokenizer threads (pattern 4)
def test_backend_encode_shared_across_threads_is_safe():
    """The template's path (backend_tokenizer.encode on one shared tokenizer) under 16 threads gives the same ids as
    a single thread and never raises; the server now uses one tokenizer per thread anyway."""
    from emb_common import load_tokenizer, make_texts

    from emb_serve_fastapi import MAX_SEQ_LEN, _fast_encode

    tok = load_tokenizer()
    texts = [t for n in (32, 128, 512) for t in make_texts(tok, n, 8)]
    ref = [_fast_encode(tok, t, MAX_SEQ_LEN) for t in texts]
    errors, out = [], {}

    def run(k):
        try:
            for rep in range(40):
                for i, t in enumerate(texts):
                    if _fast_encode(tok, t, MAX_SEQ_LEN) != ref[i]:
                        errors.append(f"mismatch t{k} i{i}")
        except Exception as e:  # "Already borrowed" would land here
            errors.append(repr(e))

    threads = [threading.Thread(target=run, args=(k,)) for k in range(16)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors, errors[:3]


# --------------------------------------------------------------------------- trace shapes (pattern 5)
def test_every_request_shape_is_a_captured_trace():
    """forward() pads every (rows, length) to a shape captured at start-up: no compile or capture while serving."""
    gv = pytest.importorskip("models.demos.embedding_gemma2.demo.generator_vllm")
    from models.demos.embedding_gemma2.tt.config import SEQ_BUCKETS, bucket_for

    max_seq_len, max_tokens = 8192, gv.DEFAULT_SERVE_MAX_TOKENS
    ladder = gv.batch_ladder(32)
    served = set(gv.serving_shapes(ladder, SEQ_BUCKETS, max_seq_len, max_tokens))
    for rows in range(1, 33):
        for length in sorted({1, 31, 32, 33, 127, 128, 129, 511, 512, 513, 2047, 2049, 4097, 8191, 8192}):
            bucket = bucket_for(length, max_seq_len)
            for _, _, padded in gv.plan_chunks(rows, bucket, ladder, max_tokens):
                assert (padded, bucket) in served, (rows, length, padded, bucket)


# --------------------------------------------------------------------------- server with a stub worker (1, 2, 6, a, b)
def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    def __init__(self, stub, *extra):
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        env = {**os.environ, "EMB_STUB_WORKER": stub}
        cmd = [sys.executable, str(HERE / "emb_serve_fastapi.py"), "--port", str(self.port), "--server-phys", "0", *extra]
        self.log = open(f"/tmp/test_emb_serve_{self.port}.log", "w")
        self.proc = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=self.log, stderr=subprocess.STDOUT)

    def wait_listening(self, timeout=60):
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                return requests.get(self.url + "/health", timeout=2)
            except requests.ConnectionError:
                time.sleep(0.2)
        raise AssertionError("server never accepted connections")

    def wait_status(self, status, timeout=120):
        t0 = time.time()
        while time.time() - t0 < timeout:
            r = requests.get(self.url + "/health", timeout=10)  # start-up imports may briefly hold the GIL
            if r.json()["status"] == status:
                return r
            time.sleep(0.2)
        raise AssertionError(f"health never became {status}: {r.json()}")

    def embed(self, inp, timeout=30, **kw):
        return requests.post(self.url + "/v1/embeddings", json={"input": inp, **kw}, timeout=timeout)

    def stop(self, timeout=60):
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                raise AssertionError("server did not exit on SIGTERM")
        self.log.close()


@pytest.fixture
def server_factory():
    made = []

    def make(stub, *extra):
        s = Server(stub, *extra)
        made.append(s)
        return s

    yield make
    for s in made:
        try:
            s.stop()
        except AssertionError:
            pass


def test_starting_then_ok_and_ordering(server_factory):
    s = server_factory("slow_start:4", "--max-batch", "8")
    r = s.wait_listening()
    assert r.status_code == 503 and r.json()["status"] == "starting"
    assert s.embed("hello").status_code == 503
    s.wait_status("ok")
    texts = ["a", "a longer piece of text", "mid size text here"]
    r = s.embed(texts)
    assert r.status_code == 200
    counts = [d["embedding"][0] for d in r.json()["data"]]
    assert counts[0] < counts[2] < counts[1]  # element 0 is the token count: row i belongs to text i
    assert s.embed("").status_code == 422
    assert s.embed(["ok", ""]).status_code == 422
    assert s.embed("x", encoding_format="bogus").status_code == 422
    m = requests.get(s.url + "/metrics", timeout=5).json()
    assert m["texts_per_forward"], m


def test_worker_death_fails_pending_and_health(server_factory):
    s = server_factory("die_after:3", "--max-batch", "8")
    s.wait_listening()
    s.wait_status("ok")
    for _ in range(3):
        assert s.embed("fine").status_code == 200
    t0 = time.time()
    r = s.embed("this one kills the worker")
    assert r.status_code == 503 and time.time() - t0 < 5
    h = requests.get(s.url + "/health", timeout=10)
    assert h.status_code == 503 and h.json()["status"] == "failed"
    assert s.embed("after").status_code == 503


def test_device_hang_hits_deadline_and_releases_worker(server_factory):
    s = server_factory("hang_after:2", "--max-batch", "8", "--device-timeout-s", "3")
    s.wait_listening()
    s.wait_status("ok")
    for _ in range(2):
        assert s.embed("fine").status_code == 200
    t0 = time.time()
    r = s.embed("this one hangs", timeout=30)
    assert r.status_code == 503 and "device timeout" in r.json()["detail"] and time.time() - t0 < 8
    h = requests.get(s.url + "/health", timeout=10)
    assert h.status_code == 503 and "device timeout" in h.json()["error"]


def test_shutdown_is_prompt(server_factory):
    s = server_factory("ok", "--max-batch", "8")
    s.wait_listening()
    s.wait_status("ok")
    assert s.embed("hi").status_code == 200
    t0 = time.time()
    s.stop(timeout=45)
    assert time.time() - t0 < 45
