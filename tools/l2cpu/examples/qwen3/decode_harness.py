#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
#
# SPDX-License-Identifier: Apache-2.0
"""x280 decode harness: Qwen3 on one chip, host-mediated decode loop ("Plan A shape") with a
pluggable sampler that works on the ROW-MAJOR bf16 logits tensor in DRAM.

    python tools/l2cpu/examples/qwen3/decode_harness.py --model Qwen/Qwen3-8B --batch 1 --prompts 5 \
        --max-new-tokens 256 --sampler compare:hostref,torchargmax --temperature 0 --out result.json

Per decode step:
  1. host writes tokens / current_pos / rope idxs into the persistent trace inputs
  2. trace "fwd": model.ttnn_decode_forward(..., on_device_logits=True) -> TILE bfp8 logits (L1),
     plus the on-device plus_one of current_pos / rope idxs (harmless here, the host rewrites them)
  3. trace "cvt": ttnn.untilize(logits, DRAM) -> ROW_MAJOR bf16 [1,1,32,Vpad] DRAM interleaved
     (one-shot; --convert chunked = split/untilize/concat fallback)
  4. sampler.sample(step) -> B tokens (x280s step index = decode step + 1; step 0 = prefill token)
--one-trace captures 2+3 as one trace (no sync between them, so no fwd/cvt split in the timings).
Prefill runs eagerly through the stock Generator (no prefill traces; prefill is not measured).
The model code under models/ is not modified.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import deps  # noqa: E402,F401  (bindings to the other l2cpu components, see deps.py)

PROMPTS_FILE = "models/tt_transformers/demo/sample_prompts/input_data_questions_prefill_128.json"


def log(*a):
    print("[harness %s]" % time.strftime("%H:%M:%S"), *a, flush=True)


def check_env():
    home = os.environ.get("TT_METAL_HOME")
    assert home, "set TT_METAL_HOME (and PYTHONPATH) to the tt-metal tree first"
    import ttnn

    assert os.path.realpath(ttnn.__file__).startswith(os.path.realpath(home)), (ttnn.__file__, home)
    log("ttnn from", ttnn.__file__)
    return home


def load_prompts(home, n):
    with open(os.path.join(home, PROMPTS_FILE)) as f:
        data = json.load(f)
    prompts = [d["prompt"] for d in data]
    while len(prompts) < n:
        prompts = prompts + prompts
    return prompts[:n]


def encode(model_args, tokenizer, prompt, input_tokens=None):
    """Chat-templated exactly like the demo (encode_prompt_hf, add_generation_prompt=True).
    input_tokens: make the templated prompt exactly that many tokens by repeating the user text
    (if short) and then left-trimming the user text tokens (if long)."""
    ids = model_args.encode_prompt(prompt, instruct=True)
    if not input_tokens:
        return ids, prompt
    text = prompt
    while len(model_args.encode_prompt(text, instruct=True)) < input_tokens:
        text = prompt + "\n\n" + text
    raw = tokenizer.encode(text, add_special_tokens=False)
    overhead = len(model_args.encode_prompt(text, instruct=True)) - len(raw)
    keep = input_tokens - overhead
    for _ in range(8):
        cand = tokenizer.decode(raw[-keep:])
        ids = model_args.encode_prompt(cand, instruct=True)
        if len(ids) == input_tokens:
            return ids, cand
        keep += input_tokens - len(ids)
    # tokenizer drift: token-level surgery on the templated ids, keeping the template head
    full = model_args.encode_prompt(text, instruct=True)
    x, y = model_args.encode_prompt("a", instruct=True), model_args.encode_prompt("b", instruct=True)
    head = next(i for i in range(len(x)) if x[i] != y[i])
    ids = full[:head] + full[len(full) - input_tokens + head :]
    return ids, tokenizer.decode(ids)


def page_table_for(batch, max_num_blocks, seed=0):
    import torch

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(max_num_blocks, generator=g)
    return torch.argsort(perm).reshape(batch, max_num_blocks // batch)


class DecodeHarness:
    def __init__(self, a):
        self.a = a
        import torch
        import ttnn

        self.ttnn = ttnn
        self.torch = torch
        from models.tt_transformers.tt.common import Mode, PagedAttentionConfig, create_tt_model
        from models.tt_transformers.tt.generator import Generator
        from models.tt_transformers.tt.model_config import DecodersPrecision

        self.Mode = Mode
        t0 = time.time()
        mesh_kwargs = dict(
            mesh_shape=ttnn.MeshShape(1, 1), num_command_queues=1, dispatch_core_config=ttnn.DispatchCoreConfig()
        )
        if a.trace_region_size:
            mesh_kwargs["trace_region_size"] = a.trace_region_size
        self.mesh = ttnn.open_mesh_device(**mesh_kwargs)
        log("device open %.1fs" % (time.time() - t0))
        if a.x280:  # arena + firmware BEFORE any trace capture (fresh chip reset first)
            from l2cpu_sampler import l2cpu_bootstrap

            l2cpu_bootstrap(self.mesh, log=log)
        self.page_params = {"page_block_size": 32, "page_max_num_blocks_per_dp": 1024}
        pac = PagedAttentionConfig(block_size=32, max_num_blocks=1024)
        t0 = time.time()
        opt = (
            (lambda ma: DecodersPrecision.performance(ma.n_layers, ma.model_name))
            if a.precision == "performance"
            else (lambda ma: DecodersPrecision.accuracy(ma.n_layers, ma.model_name))
        )
        self.model_args, self.model, self.kv_cache, _ = create_tt_model(
            self.mesh,
            instruct=True,
            max_batch_size=a.batch,
            optimizations=opt,
            max_seq_len=a.max_seq_len,
            paged_attention_config=pac,
            dtype=ttnn.bfloat8_b,
        )
        if a.prefill_mode == "sequential":
            self.model_args.disable_batched_prefill = True
        self.load_s = time.time() - t0
        log(
            "model load %.1fs, sampling module: %s"
            % (self.load_s, type(getattr(self.model, "sampling", None)).__name__)
        )
        self.tokenizer = self.model_args.tokenizer
        self.generator = Generator([self.model], [self.model_args], self.mesh, tokenizer=self.tokenizer)
        self.page_table = page_table_for(a.batch, 1024)
        self.vocab = self.model_args.vocab_size
        self.inputs = None
        self.traces = None
        self.convert_used = None

    # ---------------- decode graph pieces ----------------
    def _fwd(self):
        tok, pos, rot, pt = self.inputs[:4]
        return self.model.ttnn_decode_forward(tok, pos, rot, pt, kv_cache=self.kv_cache, on_device_logits=True)

    def _cvt(self, tile_logits, mode):
        ttnn = self.ttnn
        if mode == "oneshot":
            return ttnn.untilize(tile_logits, use_multicore=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        W = tile_logits.shape[-1]
        n = 4
        w = ((W + n - 1) // n + 31) // 32 * 32
        chunks = ttnn.split(tile_logits, w, dim=3)
        outs = []
        for c in chunks:
            outs.append(ttnn.untilize(c, use_multicore=True))
            c.deallocate()
        out = ttnn.concat(outs, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for o in outs:
            ttnn.deallocate(o)
        return out

    def write_inputs(self, tokens, pos, full=False):
        host = self.model.prepare_decode_inputs_host(tokens, pos, page_table=self.page_table)
        if self.inputs is None:
            from models.tt_transformers.tt.common import copy_host_to_device

            self.inputs = copy_host_to_device(host, mesh_device=self.mesh)
            return
        n = 4 if full else 3
        for i in range(n):
            if host[i] is not None:
                self.ttnn.copy_host_to_device_tensor(host[i], self.inputs[i])

    def compile_and_capture(self):
        """Eager compile run (with the real step-0 inputs: its KV write is the correct one), then capture."""
        ttnn = self.ttnn
        self.model.switch_mode(self.Mode.DECODE)
        t0 = time.time()
        tile = self._fwd()
        modes = [self.a.convert] if self.a.convert != "auto" else ["oneshot", "chunked"]
        err = None
        for m in modes:
            try:
                rm = self._cvt(tile, m)
                ttnn.synchronize_device(self.mesh)
                self.convert_used = m
                break
            except Exception as e:  # noqa: BLE001
                err = e
                log("convert mode %s FAILED at compile: %s" % (m, str(e).splitlines()[0] if str(e) else repr(e)))
                self.convert_fail = {m: str(e)[:2000]}
        else:
            raise RuntimeError("no convert mode works: %s" % err)
        self.eager_meta = {"tile": tensor_meta(tile), "rm": tensor_meta(rm)}
        ttnn.deallocate(rm)
        ttnn.deallocate(tile)
        log("compile run done (%.1fs), convert=%s" % (time.time() - t0, self.convert_used))
        t0 = time.time()
        if self.a.one_trace:
            tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            tile = self._fwd()
            rm = self._cvt(tile, self.convert_used)
            ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
            self.traces = [tid]
        else:
            tid_f = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            tile = self._fwd()
            ttnn.end_trace_capture(self.mesh, tid_f, cq_id=0)
            tid_c = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            rm = self._cvt(tile, self.convert_used)
            ttnn.end_trace_capture(self.mesh, tid_c, cq_id=0)
            self.traces = [tid_f, tid_c]
        self.tile_logits, self.logits = tile, rm
        log("trace capture done (%.1fs)" % (time.time() - t0))

    def run_traces(self, timings):
        ttnn = self.ttnn
        if len(self.traces) == 1:
            t0 = time.perf_counter()
            ttnn.execute_trace(self.mesh, self.traces[0], cq_id=0, blocking=False)
            ttnn.synchronize_device(self.mesh)
            timings["fwd_cvt"] = time.perf_counter() - t0
            timings["fwd"] = timings["cvt"] = None
        else:
            t0 = time.perf_counter()
            ttnn.execute_trace(self.mesh, self.traces[0], cq_id=0, blocking=False)
            ttnn.synchronize_device(self.mesh)
            t1 = time.perf_counter()
            ttnn.execute_trace(self.mesh, self.traces[1], cq_id=0, blocking=False)
            ttnn.synchronize_device(self.mesh)
            t2 = time.perf_counter()
            timings["fwd"], timings["cvt"], timings["fwd_cvt"] = t1 - t0, t2 - t1, t2 - t0

    # ---------------- one group of B users ----------------
    def prefill(self, prompts):
        torch = self.torch
        enc = [encode(self.model_args, self.tokenizer, p, self.a.input_tokens) for p in prompts]
        ids = [e[0] for e in enc]
        lens = [len(x) for x in ids]
        L = max(lens)
        toks = torch.zeros((len(ids), L), dtype=torch.int32)
        for i, x in enumerate(ids):
            toks[i, : len(x)] = torch.tensor(x, dtype=torch.int32)
        if self.a.clear_kv:
            for layer in self.model.layers:
                k, v = layer.attention.layer_past
                self.ttnn.mul(k, 0, output_tensor=k)
                self.ttnn.mul(v, 0, output_tensor=v)
        t0 = time.time()
        logits = self.generator.prefill_forward_text(
            toks,
            page_table=self.page_table,
            kv_cache=[self.kv_cache],
            prompt_lens=lens,
            sampling_params=None,
            warmup_prefill=False,
            enable_trace=False,
        )
        dt = time.time() - t0
        return ids, lens, [e[1] for e in enc], logits, dt

    def run_group(self, gi, prompts, sampler, params, seeds, recorder):
        torch = self.torch
        from x280s_ref import X280S, bf16_bits

        from dh_samplers import argmax_lowest, row_hash

        B = self.a.batch
        ids, lens, used_text, plog, pre_s = self.prefill(prompts)
        log("group %d prefill %.2fs, prompt lens %s" % (gi, pre_s, lens if B <= 8 else (min(lens), max(lens))))
        # prefill token = x280s step 0 on the bf16-rounded prefill logits (host reference library)
        plog = plog.reshape(B, -1)[:, : self.vocab].float().numpy()
        prow = bf16_bits(plog)
        lib = X280S()
        first = []
        for b in range(B):
            p = params[b]
            t, _ = lib.sample(
                prow[b], p["temperature"], p["top_k"], p["top_p"], seeds[b], user=b, step=0, vocab=self.vocab
            )
            if p["temperature"] <= 0:
                assert t == argmax_lowest(prow[b], self.vocab), "prefill argmax disagreement"
            first.append(int(t))
        out = [[t] for t in first]
        hashes = [[] for _ in range(B)]
        pos = torch.tensor(lens, dtype=torch.int64)
        tokens = torch.tensor(first, dtype=torch.int64)
        steps = []
        stop = set(getattr(self.tokenizer, "stop_tokens", None) or [self.tokenizer.eos_token_id])
        for s in range(self.a.max_new_tokens):
            tm = {}
            t0 = t_wall = time.perf_counter()
            self.write_inputs(tokens, pos, full=(s == 0))
            tm["write"] = time.perf_counter() - t0
            if self.traces is None:
                self.compile_and_capture()
                tm["capture"] = True
                self.write_inputs(tokens, pos, full=True)  # the eager compile run advanced pos/rot on device
            if s == 0:
                self.bind(sampler, params, seeds)
            self.run_traces(tm)
            if s == 0:
                self.logits_addr_seen.add(self.logits.buffer_address())
            t0 = time.perf_counter()
            toks = sampler.sample(s + 1)
            tm["sampler"] = time.perf_counter() - t0
            tm["read"] = sampler.last.get("read", 0.0)
            tm["sample"] = sampler.last.get("sample", 0.0)
            reader = getattr(sampler, "reader", None)
            if reader is not None and self.a.hash_rows:
                rows = reader.rows(s + 1)
                for b in range(B):
                    hashes[b].append(row_hash(rows[b, : self.vocab]))
                if recorder is not None:
                    recorder(gi, s, rows)
            tm["total"] = tm["write"] + tm["fwd_cvt"] + tm["sampler"]
            steps.append(tm)
            for b in range(B):
                out[b].append(int(toks[b]))
            tokens = torch.tensor(toks, dtype=torch.int64)
            pos = pos + 1
            tm["wall"] = time.perf_counter() - t_wall  # whole iteration incl. bookkeeping (and hashing if on)
        users = []
        for b in range(B):
            gen = out[b]
            eos_at = next((i for i, t in enumerate(gen) if t in stop), None)
            users.append(
                {
                    "user": b,
                    "prompt_tokens": lens[b],
                    "prompt_head": used_text[b][:120],
                    "tokens": gen,
                    "eos_at": eos_at,
                    "text": self.tokenizer.decode(gen[:eos_at] if eos_at is not None else gen),
                    "row_hashes": hashes[b],
                }
            )
        return {"group": gi, "prefill_s": pre_s, "users": users, "steps": steps}

    def bind(self, sampler, params, seeds):
        """Bind once (the logits tensor is the same for the whole session); per group only drop the
        reader's per-step cache, since step numbers restart."""
        if getattr(self, "_bound", None) is sampler:
            if getattr(sampler, "reader", None) is not None:
                sampler.reader.invalidate()
            return
        sampler.bind(self.mesh, self.logits, self.a.batch, self.vocab, params, seeds)
        self._bound = sampler

    logits_addr_seen = set()


def tensor_meta(t):
    d = {
        "shape": list(t.shape),
        "padded_shape": list(t.padded_shape),
        "dtype": str(t.dtype),
        "layout": str(t.layout),
        "memory_config": str(t.memory_config()),
    }
    try:
        d["buffer_address"] = t.buffer_address()
        d["buffer_address_hex"] = hex(t.buffer_address())
        d["page_size"] = t.buffer_page_size()
        d["aligned_page_size"] = t.buffer_aligned_page_size()
    except Exception as e:  # noqa: BLE001
        d["buffer_error"] = str(e)[:200]
    return d


def summarize(groups, skip_first=True):
    keys = ["write", "fwd", "cvt", "fwd_cvt", "read", "sample", "sampler", "total", "wall"]
    rows = []
    for g in groups:
        rows += g["steps"][1:] if skip_first else g["steps"]
    out = {"steps_counted": len(rows)}
    for k in keys:
        v = [r[k] for r in rows if r.get(k) is not None]
        if v:
            out[k + "_ms_median"] = 1e3 * float(np.median(v))
            out[k + "_ms_mean"] = 1e3 * float(np.mean(v))
    if "total_ms_mean" in out:
        out["ms_per_token"] = out["total_ms_mean"]
        out["ms_per_token_median"] = out["total_ms_median"]
        out["tokens_per_s_per_user"] = 1e3 / out["total_ms_mean"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--cache", help="TT_CACHE_PATH (default: deps.weights_cache, see README)")
    ap.add_argument("--batch", type=int, default=1, choices=[1, 32])
    ap.add_argument("--prompts", type=int, default=None, help="groups of B users (default 5 at batch 1, 1 at batch 32)")
    ap.add_argument("--max-new-tokens", type=int, default=256, help="decode steps per group")
    ap.add_argument("--sampler", default="hostref", help="hostref | torchargmax | x280 | compare:A,B")
    ap.add_argument(
        "--reader",
        default="full",
        choices=["rows", "full"],
        help="host readback: needed pages via NoC, or the whole tensor",
    )
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=lambda s: int(s, 0), default=1234)
    ap.add_argument("--record-rows", type=int, default=0)
    ap.add_argument("--rows-out")
    ap.add_argument("--input-tokens", type=int, default=None)
    ap.add_argument("--convert", default="auto", choices=["auto", "oneshot", "chunked"])
    ap.add_argument("--one-trace", action="store_true")
    ap.add_argument("--prefill-mode", default="batched", choices=["batched", "sequential"])
    ap.add_argument("--clear-kv", type=int, default=1)
    ap.add_argument("--hash-rows", type=int, default=1)
    ap.add_argument("--precision", default="performance", choices=["performance", "accuracy"])
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--trace-region-size", type=int, default=0)
    ap.add_argument("--readback-bench", type=int, default=20, help="after the run: time N full vs rows reads")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-user-mix", action="store_true", help="per-user settings from sampling_mix.user_params")
    ap.add_argument("--x280", action="store_true", help="boot the x280 firmware after device open (sampler x280)")
    a = ap.parse_args()

    home = check_env()
    os.chdir(home)
    os.environ["HF_MODEL"] = a.model
    if a.cache:
        os.environ["TT_CACHE_PATH"] = a.cache
    elif "TT_CACHE_PATH" not in os.environ:
        os.environ["TT_CACHE_PATH"] = deps.weights_cache(a.model)
    log("HF_MODEL", a.model, "TT_CACHE_PATH", os.environ["TT_CACHE_PATH"])
    import torch

    torch.manual_seed(213919)
    from dh_samplers import CompareSampler, make_sampler

    if a.prompts is None:
        a.prompts = 5 if a.batch == 1 else 1
    B = a.batch
    params = [{"temperature": a.temperature, "top_k": a.top_k, "top_p": a.top_p}] * B
    seeds = [a.seed] * B
    if a.per_user_mix:  # sampling_mix.py: user u -> setting u % 4, seed 1234 + u
        from sampling_mix import user_params

        up = user_params(B)
        params = [{"temperature": t, "top_k": k, "top_p": p} for (t, k, p, _) in up]
        seeds = [sd for (*_, sd) in up]
    all_prompts = load_prompts(home, a.prompts * B if B == 1 else 32)

    t_start = time.time()
    h = DecodeHarness(a)
    sampler = make_sampler(a.sampler, a.reader)

    rec = None
    rec_rows = []
    if a.record_rows:
        per_group = -(-a.record_rows // (a.prompts * B))

        def rec(gi, s, rows):
            if s < per_group and len(rec_rows) < a.record_rows:
                for b in range(B):
                    if len(rec_rows) < a.record_rows:
                        rec_rows.append(rows[b].copy())

    groups = []
    for gi in range(a.prompts):
        prompts = all_prompts[gi * B : (gi + 1) * B] if B == 1 else [all_prompts[(j + gi) % 32] for j in range(32)]
        if isinstance(sampler, CompareSampler):
            sampler.context = {"group": gi}
        g = h.run_group(gi, prompts, sampler, params, seeds, rec)
        groups.append(g)
        s = summarize([g])
        for u in g["users"][: min(B, 3)]:
            log("group %d user %d eos_at=%s: %r" % (gi, u["user"], u["eos_at"], u["text"][:200]))
        log(
            "group %d: %.2f ms/token (median %.2f), fwd %.2f cvt %.2f fwd+cvt %.2f read %.2f sample %.2f write %.2f"
            % (
                gi,
                s.get("ms_per_token", 0),
                s.get("ms_per_token_median", 0),
                s.get("fwd_ms_median", 0) or 0,
                s.get("cvt_ms_median", 0) or 0,
                s.get("fwd_cvt_ms_median", 0),
                s.get("read_ms_median", 0),
                s.get("sample_ms_median", 0),
                s.get("write_ms_median", 0),
            )
        )

    meta = {
        "logits_rm": tensor_meta(h.logits),
        "logits_tile": tensor_meta(h.tile_logits),
        "eager_compile": h.eager_meta,
        "convert": h.convert_used,
        "convert_fail": getattr(h, "convert_fail", None),
        "logits_addr_seen_at_step0_per_group": sorted(h.logits_addr_seen),
        "inputs": [tensor_meta(t) for t in h.inputs if t is not None],
    }
    try:
        meta["dram_bank_table"] = h.ttnn.cluster.get_dram_bank_table(0)
        reader = getattr(sampler, "reader", None)
        if reader is not None and reader.banks is not None:
            meta["row_pages"] = [
                dict(zip(("noc_x", "noc_y", "addr"), reader.page_location(b))) for b in range(min(B, 8))
            ]
    except Exception as e:  # noqa: BLE001
        meta["bank_table_error"] = str(e)[:200]

    # readback microbenchmark + cross-check of the two read paths on the final step's logits
    bench = {}
    reader = getattr(sampler, "reader", None)
    if reader is not None and a.readback_bench:
        from dh_samplers import LogitsReader

        rd_full = LogitsReader(h.mesh, h.logits, B, h.vocab, mode="full")
        rd_rows = LogitsReader(h.mesh, h.logits, B, h.vocab, mode="rows")
        tf, tr = [], []
        for i in range(a.readback_bench):
            x = rd_full.rows(("f", i))
            tf.append(rd_full.last_read_s)
            y = rd_rows.rows(("r", i))
            tr.append(rd_rows.last_read_s)
        bench = {
            "full_ms_median": 1e3 * float(np.median(tf)),
            "full_bytes": rd_full.bytes_read,
            "rows_ms_median": 1e3 * float(np.median(tr)),
            "rows_bytes": rd_rows.bytes_read,
            "full_equals_rows": bool(np.array_equal(x, y)),
            "n": a.readback_bench,
        }
        log("readback bench:", bench)

    res = {
        "config": vars(a),
        "model_load_s": h.load_s,
        "wall_s": time.time() - t_start,
        "sampler": sampler.name,
        "params_per_user": params,
        "seeds": seeds,
        "logits_meta": meta,
        "readback_bench": bench,
        "summary": summarize(groups),
        "per_group_summary": [summarize([g]) for g in groups],
        "groups": groups,
    }
    for smp in (sampler, getattr(sampler, "primary", None)):
        if getattr(smp, "name", "") == "x280":
            res["x280"] = smp.summary()
    if isinstance(sampler, CompareSampler):
        res["compare"] = sampler.summary()
        if hasattr(sampler.reference, "ties"):
            res["compare"]["torchargmax_ties"] = sampler.reference.ties
        if hasattr(sampler.primary, "ties"):
            res["compare"]["torchargmax_ties"] = sampler.primary.ties
        log("COMPARE:", {k: v for k, v in res["compare"].items() if k != "first_mismatch"})
    if a.record_rows:
        arr = np.stack(rec_rows).astype(np.uint16)
        np.save(a.rows_out, arr)
        res["rows_out"] = {"path": a.rows_out, "shape": list(arr.shape), "dtype": str(arr.dtype)}
        log("wrote rows", a.rows_out, arr.shape)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, default=str)
    log("summary:", res["summary"])
    log("wrote", a.out)
    ok = res.get("compare", {}).get("identical", True)
    h.ttnn.close_mesh_device(h.mesh)
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
