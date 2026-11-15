# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""FIX.35 (OWNER 2026-11-15): a MEASURED whole prefill 0 -> 1,024,000 in the realistic setting, host vs device per
chunk. No model change.

Setting: DS41_DECODER_SKIP=decode (the gate's env), every other switch its shipped default (DS41_ENGRAM_TABLE set to
its shipped default when unset: DS41-245). One process, the model loaded once (the adapter's load path, FIX.29).

Driven as the serving engine drives it: the adapter's runtime (tt/runners/adapter.DS41PrefillRuntime, built as
prefill_runner.main builds it: build_runtime, allocate_kv_cache for 1 slot at deploy.seq, compile), then ONE request
in slot 0 of N_CHUNKS (200) chunks of 5120 real tokens, ``runtime.prefill_chunk(inp, kv, slot_id=0, actual_start=s,
actual_end=s + 5120)`` back to back. The runtime itself marks every chunk not-last under DECODER_SKIP=decode
(begin_chunk(.., last=False)). The engine inputs (the server's [sp, 1, chunk / sp] uint32 payload, server_rules.
server_payload) are put on the device before the clock starts, as the engine's H2D socket has them on the device
before the chunk runs. No layer-completion sink and no D2H ack service (the in-process host sink syncs per layer;
the engine's D2H acks are one device op per layer on the same queue, not run here). No sync between chunks but the
runtime's own: prefill_chunk reads the engine input back once per chunk (adapter._host_ids: the engram step's host
token ids), a blocking read behind the previous chunk's work, so each chunk waits there for the device to finish the
chunk before. The test times that read (``rb_ms``) by wrapping the runtime's _host_ids (timing only).

Tokens (real text, no chunk repeated: asserted): the canonical prompt (reference/prompt.py: the chat-wrapped "Recite
A Tale of Two Cities" request and the whole book, model_turn; its first 56320 ids are the goldens' prompt.json,
asserted), then the articles of WikiText-2 (raw, train split; FULL_PREFILL_DOCS, default
/mnt/models/agentic-prefill-goldens/wikitext-2-raw-v1-train.txt) in file order, each tokenized on its own and
joined by "\\n\\n", until 1,024,000 + the warm-up's tokens. The warm-up request uses the tokens after the timed ones.

Order: runtime.compile (the engine's cold chunk), a warm-up request of WARM_CHUNKS (3) chunks at 0, then
  timed requests (no profiler) until one builds no program (at most MAX_REQUESTS): the short warm-up does not compile
  every position's programs (FIX.35's first run: the first full request built 788 programs at later chunks, DS41-281).
  The first request is reported as first_*; the warm one is THE measurement (e2e_*). Per request: cumulative wall from
  the first host call (prefill_chunk of chunk 0) to the sync after the chunk ending at each of 51200, 128000, 256000,
  512000, 768000, 1024000: a ttnn.synchronize_device at those 6 boundaries only (its time: sync_ms_p<p>; the next
  chunk's read-back would wait for the same work, so it adds only the call). Per chunk: call_ms (prefill_chunk call
  to return), rb_ms (the runtime's read-back: the wait for the previous chunk's device work plus the read), host_ms =
  call_ms - rb_ms (first host call after the wait to the last enqueue returning), new programs.
  Then the same request under the program real-time profiler (F155; its own callback, every program's start / end
  per chip, kept in arrays, no sync): per chunk and chip, busy = sum of its programs' time, first start, last end.
  device_ms = the slowest chip's busy (as FIX.34); span = first start to last end on the run's critical chip (most
  busy time); gap = idle on that chip between the chunk before's last end and this chunk's first start (chip clocks
  are not compared across chips). Its walls are recorded as prof_* (what the profiler adds).

Verdict: chunk k is host-bound when the device was (almost) done with it before the host came back: the next
chunk's read-back waited < 1 ms beyond an idle read-back (rb_idle_ms: the read-back right after an explicit sync); the
chunk before a boundary uses its sync. Also host_ge_device_chunks (host_ms >= device_ms), wall vs sum(device_ms) and
vs the critical chip's timeline. Compared with FIX.34 (results/FIX.34.json: one chunk timed alone at p): the chunk
starting at p here (the last chunk for p = 1024000).

Metrics: e2e_wall_s_p<p>, e2e_tok_s_p<p>, sync_ms_p<p>, host_ms_mean / max, rb_ms_mean, call_ms_mean,
device_ms_mean, span_ms_mean, gap_ms_mean / max, host_bound_chunks, host_ge_device_chunks, device_ms_sum_s,
device_idle_between_s / inside_s, requests_until_warm, per_chunk_* lists, full_prefill_table (markdown, also printed).

    PYTHONPATH=$PWD DS41_DECODER_SKIP=decode scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/test_full_prefill.py
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from array import array
from pathlib import Path

import numpy as np
import pytest

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec
from models.demos.common.bringup.testing.profile import cache_entries
from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

S = spec()
BATCH_SHARED = True  # FIX.29: takes the model of a bring-up batch process (core/batch.shareable)
POINTS = (51200, 128000, 256000, 512000, 768000, 1024000)
N_CHUNKS = 200  # 200 x 5120 = 1,024,000 on galaxy_8x4
WARM_CHUNKS = 3
MAX_REQUESTS = 4  # timed requests until one builds no program
DOCS = "/mnt/models/agentic-prefill-goldens/wikitext-2-raw-v1-train.txt"
RT_SPAN = 1 << 16  # the real-time profiler's 16-bit runtime ids (testing/profiler.RT_ID_SPAN, F262)


# ---------------------------------------------------------------- tokens
def _canonical_full(tok) -> list[int]:
    """The canonical prompt, whole (reference/prompt.build_ids without the cut)."""
    import bz2

    from models.demos.common.bringup.core.freeze import sha256_file
    from models.demos.common.bringup.core.spec import CODE_ROOT
    from models.demos.common.bringup.reference import prompt

    src = Path(S.get("text.source", prompt.DEFAULT_SOURCE))
    rec = prompt.load(S)
    want = rec.get("source_sha256") if rec else None
    # (DS41-280: the spec's text.source moved out of models/tt_transformers; the same file, by its sha in prompt.json)
    cands = [src if src.is_absolute() else CODE_ROOT / src] + [
        CODE_ROOT / d / src.name for d in ("models/ttt_compat/tests", "models/demos/llama3_70b_galaxy/tests")
    ]
    path = next((c for c in cands if c.exists() and (want is None or sha256_file(c) == want)), None)
    assert path is not None, f"the canonical text {src} (sha {want}) is not in {[str(c) for c in cands]}"
    with (bz2.open if path.suffix == ".bz2" else open)(path, "rt", encoding="utf-8") as f:
        text = f.read()
    assert prompt.wrap_mode(S) == "model_turn", prompt.wrap_mode(S)
    pre = tok.apply_chat_template(
        [{"role": "user", "content": S.get("text.request", prompt.DEFAULT_REQUEST)}],
        add_generation_prompt=True,
        tokenize=False,
        **prompt.template_kwargs(S),
    )
    return tok(pre, add_special_tokens=False)["input_ids"] + tok(text, add_special_tokens=False)["input_ids"]


def _articles(path: str):
    """WikiText-2 raw articles in file order: a top-level heading line ' = Title = ' starts one."""
    head = re.compile(r"^ = [^=].* = $")
    cur = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if head.match(line.rstrip("\n")) and cur:
                yield "".join(cur).strip()
                cur = []
            cur.append(line)
    if cur:
        yield "".join(cur).strip()


def build_tokens(tok, n: int) -> tuple[list[int], dict]:
    """>= n ids of real text: the canonical prompt, then WikiText-2 articles."""
    from models.demos.common.bringup.reference import prompt

    ids = _canonical_full(tok)
    info = {"canonical_tokens": len(ids)}
    rec = prompt.load(S)
    if rec is not None:
        k = len(rec["token_ids"])
        assert ids[:k] == rec["token_ids"], "the canonical prompt built here differs from prompt.json"
        info["canonical_prompt_json_prefix"] = k
    path = os.environ.get("FULL_PREFILL_DOCS", DOCS)
    sep = tok("\n\n", add_special_tokens=False)["input_ids"]
    n_doc = 0
    batch_txt = []
    for art in _articles(path):
        if not art:
            continue
        batch_txt.append(art)
        if len(batch_txt) < 64:
            continue
        for a in tok(batch_txt, add_special_tokens=False)["input_ids"]:
            ids += sep + a
            n_doc += 1
        batch_txt = []
        if len(ids) >= n:
            break
    if len(ids) < n and batch_txt:
        for a in tok(batch_txt, add_special_tokens=False)["input_ids"]:
            ids += sep + a
            n_doc += 1
    info.update(docs_source=path, docs_articles=n_doc, total_tokens=len(ids))
    if len(ids) < n:
        raise ValueError(f"real text gives {len(ids)} tokens < {n} (canonical prompt + {n_doc} articles of {path})")
    return ids, info


# ---------------------------------------------------------------- the engine side
def _build_runtime(mesh_device, max_seq: int, chunk: int):
    """adapter -> runtime -> engine-owned KV cache (1 slot) -> compile, as prefill_runner.main (contract/caches.build
    with this test's max_seq)."""
    from models.demos.common.prefill.adapter import PrefillRunParams, get_adapter
    from models.demos.deepseek_v41_flash_d_p.tests.bringup.contract import caches as C

    sp, tp = tuple(mesh_device.shape)
    adapter = get_adapter(C.adapter_name())
    hf = adapter.load_hf_config()
    hf.max_seq_len = max_seq
    params = PrefillRunParams(
        mesh_shape=(sp, tp),
        num_layers=S.num_layers,
        first_layer_idx=0,
        is_first_rank=True,
        is_last_rank=True,
        max_seq_len=max_seq,
        chunk_size=chunk,
        num_users=1,
        capacity_factor=8,
        num_links=2,
        gate_mode_name=adapter.default_gate_mode,
        kv_only_last_layer=True,
        weight_cache_path=adapter.weight_cache_path((sp, tp)),
    )
    t0 = time.time()
    runtime = adapter.build_runtime(mesh_device=mesh_device, hf_config=hf, params=params)
    print(f"runtime ready in {time.time() - t0:.1f} s", flush=True)
    kv = adapter.allocate_kv_cache(mesh_device=mesh_device, hf_config=hf, params=params)
    t0 = time.time()
    runtime.compile(kv)
    print(f"runtime.compile (the engine's cold chunk) {time.time() - t0:.1f} s", flush=True)
    return runtime, kv


def _inputs(mesh_device, ids, starts, chunk):
    from models.demos.deepseek_v41_flash_d_p.tests.bringup.contract import caches as C
    from models.demos.deepseek_v41_flash_d_p.tests.bringup.contract import server_rules as R

    return [C.payload_tensor(mesh_device, R.server_payload(ids[s : s + chunk], s, s + chunk)) for s in starts]


class _RtRecorder:
    """Every program's (runtime id, chip, start ns, end ns) from the real-time profiler, kept in arrays (no per-record
    Python objects), with the host's device-operation id when each batch arrived (to unwrap the 16-bit ids)."""

    def __init__(self, mesh):
        import ttnn
        from models.demos.common.bringup.testing import profiler

        profiler.require_rt()
        self.ttnn = ttnn
        self.chip = {int(d): c for c, d in enumerate(mesh.get_device_ids())}
        self.rid, self.dev, self.host = array("q"), array("q"), array("q")
        self.st, self.en = array("d"), array("d")
        self.dropped = 0
        self.handle = None

    def start(self):
        rid, dev, host, st, en = self.rid, self.dev, self.host, self.st, self.en
        cur = self.ttnn._ttnn.get_device_operation_id

        def on_batch(batch):
            self.dropped += int(getattr(batch, "dropped", 0) or 0)
            h = int(cur())
            for r in batch.records:
                f = float(r.frequency)
                if f <= 0 or r.end_timestamp <= r.start_timestamp:
                    continue
                rid.append(int(r.runtime_id))
                dev.append(int(r.chip_id))
                host.append(h)
                st.append(r.start_timestamp / f)
                en.append(r.end_timestamp / f)

        self.handle = self.ttnn.device.RegisterProgramRealtimeProfilerCallback(on_batch)

    def stop(self, settle_s: float = 2.0):
        time.sleep(settle_s)
        if self.handle is not None:
            self.ttnn.device.UnregisterProgramRealtimeProfilerCallback(self.handle)
            self.handle = None

    def per_chunk(self, chunk_ids: list[int], end_id: int) -> dict:
        """chunk k = device-operation ids [chunk_ids[k], chunk_ids[k + 1]) (end_id after the last): per chunk and chip,
        busy ns, first start, last end, programs."""
        rid = np.frombuffer(self.rid, dtype=np.int64)
        host = np.frombuffer(self.host, dtype=np.int64)
        full = host - ((host - rid) % RT_SPAN)  # the newest id <= the host's id at arrival with these low 16 bits
        dev = np.frombuffer(self.dev, dtype=np.int64)
        st, en = np.frombuffer(self.st, dtype=np.float64), np.frombuffer(self.en, dtype=np.float64)
        bounds = np.asarray(list(chunk_ids) + [end_id], dtype=np.int64)
        k = np.searchsorted(bounds, full, side="right") - 1
        keep = (k >= 0) & (k < len(chunk_ids))
        chips = sorted(set(self.chip.values()))
        n, nc = len(chunk_ids), len(chips)
        busy = np.zeros((n, nc))
        first = np.full((n, nc), np.inf)
        last = np.full((n, nc), -np.inf)
        progs = np.zeros((n, nc), dtype=np.int64)
        ci = np.asarray([self.chip.get(int(d), -1) for d in dev], dtype=np.int64) if dev.size else dev
        keep &= ci >= 0
        k, ci, st, en = k[keep], ci[keep], st[keep], en[keep]
        np.add.at(busy, (k, ci), en - st)
        np.minimum.at(first, (k, ci), st)
        np.maximum.at(last, (k, ci), en)
        np.add.at(progs, (k, ci), 1)
        return {"busy": busy, "first": first, "last": last, "programs": progs, "records": int(keep.sum())}


# ---------------------------------------------------------------- the test
@mesh_parametrize(timeout_s=3 * 3600)
def test_full_prefill(mesh_device, monkeypatch):
    if mesh_device is None:
        pytest.skip("device test")
    import ttnn

    assert settings.get("DECODER_SKIP") == "decode", (
        "FIX.35 measures DECODER_SKIP=decode: run with DS41_DECODER_SKIP=decode "
        f"(got {settings.get('DECODER_SKIP')})"
    )
    if os.environ.get(settings.env_name("ENGRAM_TABLE")) is None:  # (the shipped default, not F222's test value)
        monkeypatch.setenv(settings.env_name("ENGRAM_TABLE"), str(settings.table["ENGRAM_TABLE"].default))
    print(f"ENGRAM_TABLE={settings.get('ENGRAM_TABLE')}")
    chunk = int(S.get("target.chunk"))
    total = N_CHUNKS * chunk
    assert total == POINTS[-1], (N_CHUNKS, chunk)
    deploy = int(S.get("deploy.seq") or 0)
    assert deploy >= total and deploy % chunk == 0, (deploy, total, chunk)
    assert list(S.layers()) == list(range(S.num_layers)), "the whole model is served"
    # the load's position tables cover the request (DS41-279: the spec's target.seq is the ladder's)
    monkeypatch.setitem(S.data["target"], "seq", deploy)

    from models.demos.common.bringup.reference.golden import tokenizer

    t0 = time.time()
    ids, tinfo = build_tokens(tokenizer(S), total + WARM_CHUNKS * chunk)
    timed = ids[:total]
    hashes = {
        hashlib.sha256(np.asarray(timed[s : s + chunk], dtype=np.int64).tobytes()).hexdigest()
        for s in range(0, total, chunk)
    }
    assert len(hashes) == N_CHUNKS, f"{N_CHUNKS - len(hashes)} chunk(s) repeat"
    warm = ids[total : total + WARM_CHUNKS * chunk]
    print(f"tokens ({time.time() - t0:.0f} s): {tinfo}", flush=True)
    for k, v in tinfo.items():
        metrics.record(f"text_{k}", v)

    runtime, kv = _build_runtime(mesh_device, deploy, chunk)
    model = runtime.model
    assert model.skip_mode == "decode", model.skip_mode  # (the env's: the runtime's load reads it)
    starts = list(range(0, total, chunk))
    t0 = time.time()
    inp = _inputs(mesh_device, timed, starts, chunk)
    inp_w = _inputs(mesh_device, warm, list(range(0, WARM_CHUNKS * chunk, chunk)), chunk)
    ttnn.synchronize_device(mesh_device)
    print(f"engine inputs on the device ({len(inp)} + {len(inp_w)}) in {time.time() - t0:.1f} s", flush=True)

    rb = []
    orig = runtime._host_ids

    def timed_host_ids(*a, **k):  # (timing only: the runtime's own read-back)
        t = time.perf_counter()
        out = orig(*a, **k)
        rb.append(time.perf_counter() - t)
        return out

    runtime._host_ids = timed_host_ids

    def request(inputs, at_points=True, ids_out=None):
        """One request in slot 0: per-chunk call / read-back / host ms, new programs, and the cumulative walls."""
        rb.clear()
        call, cum, sync, newp = [], {}, {}, []
        pts = set(POINTS) if at_points else set()
        ttnn.synchronize_device(mesh_device)
        n_prev = cache_entries(mesh_device)
        t_first = time.perf_counter()
        for k, x in enumerate(inputs):
            s = k * chunk
            if ids_out is not None:
                ids_out.append(ttnn._ttnn.get_device_operation_id())
            t = time.perf_counter()
            runtime.prefill_chunk(x, kv, slot_id=0, actual_start=s, actual_end=s + chunk, request_id=k)
            call.append(time.perf_counter() - t)
            n = cache_entries(mesh_device)
            newp.append(-1 if n is None or n_prev is None else n - n_prev)
            n_prev = n
            if s + chunk in pts or k == len(inputs) - 1:
                t = time.perf_counter()
                ttnn.synchronize_device(mesh_device)
                t2 = time.perf_counter()
                sync[s + chunk], cum[s + chunk] = (t2 - t) * 1e3, t2 - t_first
        assert len(rb) == len(inputs), (len(rb), len(inputs))
        return {
            "call": [c * 1e3 for c in call],
            "rb": [r * 1e3 for r in rb],
            "host": [(c - r) * 1e3 for c, r in zip(call, rb)],
            "new": newp,
            "cum": cum,
            "sync": sync,
        }

    runs = []
    try:
        t0 = time.time()
        request(inp_w, at_points=False)
        print(f"warm-up request ({WARM_CHUNKS} chunks) {time.time() - t0:.1f} s", flush=True)
        # the measured request: the first one that builds no program (the short warm-up does not compile every
        # position's programs: the first full request built 788 in FIX.35's first run); every earlier one reported
        for i in range(MAX_REQUESTS):
            R = request(inp)
            runs.append(R)
            new = sum(R["new"])
            where = [k for k, v in enumerate(R["new"]) if v]
            print(
                f"request {i}: {R['cum'][total]:.2f} s, new programs {new} at chunks {where[:40]}"
                f"{' ...' if len(where) > 40 else ''}",
                flush=True,
            )
            if new == 0:
                break
        M = runs[-1]
        assert sum(M["new"]) == 0, f"no request of {MAX_REQUESTS} was warm: {[sum(r['new']) for r in runs]}"

        rec = _RtRecorder(mesh_device)
        chunk_ids = []
        rec.start()
        try:
            P = request(inp, ids_out=chunk_ids)
            end_id = ttnn._ttnn.get_device_operation_id()
        finally:
            rec.stop()
        dev = rec.per_chunk(chunk_ids, end_id)
        print(
            f"profiled request: {P['cum'][total]:.2f} s, {dev['records']} records, dropped {rec.dropped}, new programs "
            f"{sum(P['new'])}",
            flush=True,
        )
        metrics.record("prof_rt_dropped", rec.dropped)
    finally:
        runtime._host_ids = orig
        for x in inp + inp_w:
            ttnn.deallocate(x)

    _report(runs, P, dev, chunk, total)


def _fix34() -> dict:
    for d in (metrics.results_dir(), getattr(S, "ledger_dir", None) and Path(S.ledger_dir) / "results"):
        if d and (Path(d) / "FIX.34.json").exists():
            m = json.loads((Path(d) / "FIX.34.json").read_text())["metrics"]
            return {k: v["value"] for k, v in m.items()}
    return {}


def _report(runs, P, dev, chunk, total):
    M = runs[-1]
    busy = dev["busy"] / 1e6  # [chunk, chip] ms
    n = busy.shape[0]
    device_ms = busy.max(axis=1)
    crit = int(busy.sum(axis=0).argmax())
    first, last = dev["first"][:, crit], dev["last"][:, crit]
    span = (last - first) / 1e6
    gap = np.concatenate([[0.0], (first[1:] - last[:-1]) / 1e6])
    host = np.asarray(M["host"])
    rbm = np.asarray(M["rb"])
    call = np.asarray(M["call"])
    # the read-back with nothing to wait for: the chunks right after an explicit sync (chunk 0, after each boundary)
    after_sync = [0] + [p // chunk for p in POINTS[:-1]]
    rb_idle = float(np.median(rbm[after_sync]))
    # chunk k host-bound: when the next chunk's read-back came, the device had (almost) nothing left of chunk k
    # (waited < 1 ms beyond an idle read-back); the last chunk: its final sync
    wait_next = np.concatenate([rbm[1:] - rb_idle, [M["sync"][total]]])
    wait_next[[k - 1 for k in after_sync[1:]]] = [M["sync"][p] for p in POINTS[:-1]]
    hb = wait_next < 1.0
    idle_between = gap[1:].sum()
    idle_inside = (span - busy[:, crit]).sum()
    wall = M["cum"][total]

    def rec(name, v):
        metrics.record(name, v if isinstance(v, (list, str, int)) else round(float(v), 3))

    for tag, R in [("", M), ("prof_", P)] + ([("first_", runs[0])] if len(runs) > 1 else []):
        for p in POINTS:
            rec(f"{tag}e2e_wall_s_p{p}", R["cum"][p])
            rec(f"{tag}e2e_tok_s_p{p}", p / R["cum"][p])
            rec(f"{tag}sync_ms_p{p}", R["sync"][p])
        rec(f"{tag}host_ms_mean", np.mean(R["host"]))
        rec(f"{tag}host_ms_max", np.max(R["host"]))
        rec(f"{tag}rb_ms_mean", np.mean(R["rb"]))
        rec(f"{tag}call_ms_mean", np.mean(R["call"]))
        rec(f"{tag}new_programs", int(sum(R["new"])))
    rec("requests_until_warm", len(runs))
    rec("host_ms_max_chunk", int(host.argmax()))
    rec("rb_idle_ms", rb_idle)
    rec("device_ms_mean", device_ms.mean())
    rec("device_ms_max", device_ms.max())
    rec("device_ms_sum_s", device_ms.sum() / 1e3)
    rec("span_ms_mean", span.mean())
    rec("span_ms_sum_s", span.sum() / 1e3)
    rec("gap_ms_mean", gap[1:].mean())
    rec("gap_ms_max", gap[1:].max())
    rec("gap_ms_max_chunk", int(gap.argmax()))
    rec("device_idle_between_s", idle_between / 1e3)
    rec("device_idle_inside_s", idle_inside / 1e3)
    rec("critical_chip", crit)
    rec("host_bound_chunks", int(hb.sum()))
    rec("host_ge_device_chunks", int((host >= device_ms).sum()))
    rec("wall_minus_device_s", wall - device_ms.sum() / 1e3)
    for name, v in (
        ("host_ms", host),
        ("rb_ms", rbm),
        ("call_ms", call),
        ("wait_next_ms", wait_next),
        ("device_ms", device_ms),
        ("span_ms", span),
        ("gap_ms", gap),
        ("prof_host_ms", np.asarray(P["host"])),
        ("first_new_programs", np.asarray(runs[0]["new"])),
    ):
        rec(f"per_chunk_{name}", [round(float(x), 2) for x in v])
    rec("per_chunk_programs", [int(x) for x in dev["programs"][:, crit]])

    lines = [
        "| boundary | cumulative wall s | tok/s | sync ms | sum device ms (s) | sum span (s) | sum host ms (s) | "
        "sum gap (s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for p in POINTS:
        k = p // chunk
        lines.append(
            f"| {p} | {M['cum'][p]:.2f} | {p / M['cum'][p]:.0f} | {M['sync'][p]:.1f} | "
            f"{device_ms[:k].sum() / 1e3:.2f} | {span[:k].sum() / 1e3:.2f} | {host[:k].sum() / 1e3:.2f} | "
            f"{gap[1:k].sum() / 1e3:.2f} |"
        )
    lines += [
        "",
        "| chunks | host ms mean (max) | read-back ms mean | device ms mean (max) | span ms mean | gap ms mean (max) | "
        "host-bound |",
        "|---|---|---|---|---|---|---|",
    ]
    for a, b in ((0, 1), (1, 10), (10, 25), (25, 50), (50, 100), (100, 150), (150, n)):
        sl = slice(a, b)
        g = gap[max(a, 1) : b]
        lines.append(
            f"| {a}-{b - 1} | {host[sl].mean():.1f} ({host[sl].max():.1f}) | {rbm[sl].mean():.1f} | "
            f"{device_ms[sl].mean():.1f} ({device_ms[sl].max():.1f}) | {span[sl].mean():.1f} | "
            f"{(g.mean() if g.size else 0):.2f} ({(g.max() if g.size else 0):.2f}) | {int(hb[sl].sum())} |"
        )
    f34 = _fix34()
    if f34:
        lines += [
            "",
            "| FIX.34 point | FIX.34 wall / device / host ms | chunk start here | call ms | host ms | read-back ms | "
            "device ms | span ms | gap ms |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for p in POINTS:
            k = min(p // chunk, n - 1)
            w, d, h = (f34.get(f"steady_{x}_ms_p{p}") for x in ("wall", "device", "host"))
            lines.append(
                f"| {p} | {w} / {d} / {h} | {k * chunk} | {call[k]:.1f} | {host[k]:.1f} | {rbm[k]:.1f} | "
                f"{device_ms[k]:.1f} | {span[k]:.1f} | {gap[k]:.2f} |"
            )
    verdict = (
        f"warm wall {wall:.2f} s ({total / wall:.0f} tok/s) vs sum of device ms {device_ms.sum() / 1e3:.2f} s "
        f"(slowest chip's program time per chunk) and the critical chip {crit}'s timeline {span.sum() / 1e3:.2f} s "
        f"+ {idle_between / 1e3:.2f} s between chunks; host-bound (the device done before the host came back) on "
        f"{int(hb.sum())} of {n} chunks; host enqueue >= device ms on {int((host >= device_ms).sum())}; requests "
        f"until warm {len(runs)} (new programs {[int(sum(r['new'])) for r in runs]})"
    )
    lines += ["", "Verdict: " + verdict]
    table = "\n".join(lines)
    rec("full_prefill_table", table)
    rec("verdict", verdict)
    print("\n" + table, flush=True)
