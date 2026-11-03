# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Where the pipeline-parallel prefill's time goes.

ttft:   the served single-user path (generator_vllm.Qwen36ResidentForCausalLM.prefill_forward: the decode model's
        counter reset, pp.reset, pp.run of all but the last token, the handoff, the last token's step and its logits)
        per prompt length: end to end without synchronizing, then by part with a synchronize after each: resets,
        every tick's host control writes and device time, handoff parts, step, logits.
chunks: pp.run of one prompt at every chunk size (pick_chunk's TICK_MS table was measured on QB2).
queue:  N back-to-back prompts of T tokens through the served path (the rows engine runs the same reset + run +
        handoff + first step per request): prefill tok/s of a queue.
ops:    one eager tick per (chunk, context) with every chip active, between start_/stop_ signposts, for the device
        profiler (python -m tracy -r -v -m qwen36_resident.tests.pp_profile ops ...). Builds no decode model.

usage: python -m qwen36_resident.tests.pp_profile ttft --T 128,2048,8192 --reps 3 --out D/ttft.json
       python -m qwen36_resident.tests.pp_profile ops --ops 1024:0,1024:65536 --max-len 131072
"""
import argparse
import faulthandler
import json
import math
import sys
import time

import torch

import ttnn
from qwen36_resident.tests.long_ctx import Clock, close_mesh, log, open_mesh

CORPUS = "/home/ttuser/experiments/qwen36_27b/profiles/resident/256k/corpus_ids.pt"
LONG = 65536  # prompts from this length run last, after --long-cool seconds (chip 1 throttles under sustained load)


class Served:
    """the plain single-user build of generator_vllm (decode model + PP prefill, traces captured last)"""

    def __init__(self, mesh, max_seq_len, chunks, clock, decode=True):
        from qwen36_resident import weights as Q
        from qwen36_resident.model import Dims, ResidentModel, State
        from qwen36_resident.prefill.pp_prefill import PPPrefill

        ck = Q.Checkpoint(Q.checkpoint_dir())
        cfg = ck.config
        layers, interval = cfg["num_hidden_layers"], cfg["full_attention_interval"]
        self.mesh, self.vocab = mesh, cfg["vocab_size"]
        C = max(chunks)
        max_len = -(-max_seq_len // C) * C
        self.pp = PPPrefill(mesh, ck, layers, interval, chunk=list(chunks), max_len=max_len)
        clock(f"prefill model (max_len {self.pp.max_len})")
        self.model = None
        if not decode:
            return
        self.d = d = Dims(mesh.get_num_devices(), mesh.dram_grid_size().x, vocab=self.vocab)
        self.emb = Q.embedding(ck)
        n_attn = sum((l + 1) % interval == 0 for l in range(layers))
        st = State(d, layers - n_attn, n_attn, 0, max_pos=max_len + 64, zero=True, kv=False)
        w = Q.load(ck, d, layers, interval)
        self.model = ResidentModel(mesh, d, w, st, layers, interval, lm_head=True, embed=self.pp.embed)
        del w, st
        m = self.model
        m.step(m.token_state(self.emb[0].float(), 0))
        m.reset()
        m.capture_trace()
        self.pp.capture()
        self.pp.handoff(m, 1)
        ttnn.synchronize_device(mesh)
        clock("decode model, traces, handoff compiled")

    def sync(self):
        ttnn.synchronize_device(self.mesh)

    def first_step(self, ids):
        T = len(ids)
        m = self.model
        m.step(m.token_state(self.emb[int(ids[T - 1])].float(), T - 1))
        self.sync()

    def served(self, ids, chunk=None):
        """generator_vllm prefill_forward: the last token's logits"""
        T = len(ids)
        self.model.reset(state=T <= 1)
        self.pp.reset()
        if T > 1:
            self.pp.run(ids[: T - 1], chunk=chunk)
            self.pp.handoff(self.model, T - 1)
        self.first_step(ids)
        return self.model.logits()

    def parts(self, ids, chunk=None):
        """the served path with a synchronize after each part: seconds per part, per-tick (host, device) seconds"""
        T = len(ids)
        now = time.perf_counter
        out = {}
        self.sync()
        t = now()

        def lap(name):
            nonlocal t
            self.sync()
            t1 = now()
            out[name] = t1 - t
            t = t1

        self.model.reset(state=False)
        lap("model_reset")
        self.pp.reset()
        lap("pp_reset")
        ticks = []
        self.pp.run(ids[: T - 1], timings=ticks, chunk=chunk)
        lap("ticks")
        ho = {}
        self.pp.handoff(self.model, T - 1, ho)
        lap("handoff")
        self.first_step(ids)
        lap("step")
        self.model.logits()
        lap("logits")
        out["handoff_parts"] = ho
        out["tick_host"] = [h for h, _ in ticks]
        out["tick_dev"] = [d for _, d in ticks]
        out["chunk"] = self.pp.C
        return out


def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2]


def cool(s):
    if s > 0:
        time.sleep(s)


def run_ttft(r, a, corpus, Ts, logits=None):
    res = []
    for T in Ts:
        ids = corpus[:T]
        if logits is not None:  # (--save-logits: the served path's first logits per prompt length)
            logits[T] = r.served(ids).float().clone()
        # warm (a long prompt finds every program compiled by the shorter ones; it would only heat the chips)
        if T < LONG:
            r.served(ids)
        e2e = []
        for _ in range(a.reps):
            cool(a.cool)
            r.sync()
            t0 = time.perf_counter()
            r.served(ids)
            e2e.append(time.perf_counter() - t0)
        cool(a.cool)
        p = r.parts(ids)
        n, C = r.pp.n, p["chunk"]
        chunks = -(-(T - 1) // C)
        dev, host = p["tick_dev"], p["tick_host"]
        row = dict(
            T=T,
            chunk=C,
            chunks=chunks,
            ticks=len(dev),
            ttft_ms=1e3 * median(e2e),
            ttft_all_ms=[1e3 * x for x in e2e],
            model_reset_ms=1e3 * p["model_reset"],
            pp_reset_ms=1e3 * p["pp_reset"],
            ticks_sync_ms=1e3 * p["ticks"],
            tick_dev_sum_ms=1e3 * sum(dev),
            tick_host_sum_ms=1e3 * sum(host),
            tick_dev_first_ms=1e3 * dev[0],
            tick_dev_last_ms=1e3 * dev[-1],
            tick_dev_mean_ms=1e3 * sum(dev) / len(dev),
            fill_drain_ms=1e3 * (sum(dev[: n - 1]) + sum(dev[chunks:])) if chunks >= n - 1 else None,
            fill_drain_share=(n - 1) / (chunks + n - 1),
            handoff_ms=1e3 * p["handoff"],
            handoff_parts_ms={k: 1e3 * v for k, v in p["handoff_parts"].items()},
            step_ms=1e3 * p["step"],
            logits_ms=1e3 * p["logits"],
            tick_dev_ms=[round(1e3 * x, 3) for x in dev],
            tick_host_ms=[round(1e3 * x, 3) for x in host],
        )
        res.append(row)
        log(
            f"T={T} C={C} ticks {len(dev)}: TTFT {row['ttft_ms']:.1f} ms (e2e {[round(x, 1) for x in row['ttft_all_ms']]}); "
            f"parts: model_reset {row['model_reset_ms']:.1f}, pp_reset {row['pp_reset_ms']:.1f}, ticks(sync) "
            f"{row['ticks_sync_ms']:.1f} [dev {row['tick_dev_sum_ms']:.1f}, host ctl {row['tick_host_sum_ms']:.1f}, "
            f"mean tick {row['tick_dev_mean_ms']:.2f}, first {row['tick_dev_first_ms']:.2f}, last {row['tick_dev_last_ms']:.2f}], "
            f"handoff {row['handoff_ms']:.1f} {json.dumps({k: round(v, 1) for k, v in row['handoff_parts_ms'].items()})}, "
            f"step {row['step_ms']:.1f}, logits {row['logits_ms']:.1f}"
        )
    return res


def run_chunks(r, a, corpus):
    res = []
    for T in a.chunk_T:
        ids = corpus[:T]
        row = dict(T=T, pick=r.pp.pick_chunk(T - 1), ms={})
        for C in r.pp.chunks:
            r.served(ids, chunk=C)
            ts = []
            for _ in range(a.reps):
                r.sync()
                t0 = time.perf_counter()
                r.served(ids, chunk=C)
                ts.append(time.perf_counter() - t0)
            row["ms"][C] = 1e3 * median(ts)
        best = min(row["ms"], key=row["ms"].get)
        row["best"] = best
        res.append(row)
        log(
            f"T={T}: TTFT by chunk {json.dumps({c: round(v, 1) for c, v in row['ms'].items()})}; pick {row['pick']}, best {best}"
        )
    return res


def run_queue(r, a, corpus):
    res = []
    for T in a.queue_T:
        prompts = [corpus[k * T : (k + 1) * T] for k in range(a.queue_n)]
        r.served(prompts[0])
        cool(a.cool)
        r.sync()
        t0 = time.perf_counter()
        for p in prompts:
            r.served(p)
        dt = time.perf_counter() - t0
        row = dict(T=T, n=a.queue_n, s=dt, per_req_ms=1e3 * dt / a.queue_n, tok_s=a.queue_n * T / dt)
        res.append(row)
        log(f"queue T={T} x{a.queue_n}: {dt:.2f} s, {row['per_req_ms']:.1f} ms/request, {row['tok_s']:.0f} tok/s")
    return res


def run_handoff(r, a, corpus):
    """the handoff's device copy alone (after a real prefill of T tokens), and its GDN-state and KV jobs apart"""
    res = []
    pp, m = r.pp, r.model
    for T in a.handoff_T:
        r.served(corpus[:T])
        row = dict(T=T)
        for name, keep in (("all", None), ("gdn", True), ("kv", False)):
            jobs, pks = [], []
            for p in range(pp.n):
                jb, pk = pp._ho_jobs(m, p, T - 1)
                if keep is not None:
                    sel = (pp._ho["tile_of"][p][pp._ho["tile_of"][p] < -(-(T - 1) // 32)] < 0) == keep
                    jb, pk = jb[sel], pk[sel]
                jobs.append(jb)
                pks.append(pk)
            lanes = pp._ho_lanes(jobs, pks)
            jobs_ts = [pp._ho_upload(lj) for lj, _ in lanes]
            ts = []
            for _ in range(a.reps):
                r.sync()
                t0 = time.perf_counter()
                pp._ho_copy(m, jobs_ts, lanes, None)
                r.sync()
                ts.append(time.perf_counter() - t0)
            for t in jobs_ts:
                ttnn.deallocate(t)
            mb = sum(int(j[:, 7].mul(j[:, 3]).sum()) for j in jobs) / pp.n / 1e6
            row[name] = dict(ms=1e3 * median(ts), mb_per_chip=mb, gb_s=mb / median(ts) / 1e3)
        res.append(row)
        log(
            f"handoff copy T={T}: "
            + ", ".join(
                f"{k} {v['ms']:.2f} ms ({v['mb_per_chip']:.1f} MB/chip, {v['gb_s']:.2f} GB/s)"
                for k, v in row.items()
                if k != "T"
            )
        )
    return res


def run_ops(pp, a):
    from tracy import signpost

    shard = ttnn.ShardTensorToMesh(pp.mesh, dim=0)

    def read_profiler():  # the device profiler's buffers hold < 1000 ops between reads
        ttnn.synchronize_device(pp.mesh)
        try:
            ttnn.ReadDeviceProfiler(pp.mesh)
        except Exception as e:  # (not profiling)
            log(f"ReadDeviceProfiler: {e}")

    read_profiler()
    for spec in a.ops.split(","):
        C, ctx = (int(x) for x in spec.split(":"))
        pp._use(C)
        t = max(ctx // C, pp.n - 1)  # chip 0 runs chunk t (context t * C), chip p chunk t - p: all active
        n_tokens = (t + 1) * C
        assert n_tokens <= pp.max_len, f"ops {spec}: needs max_len >= {n_tokens}"
        ids = torch.randint(0, 1000, (pp.n, 1, C), dtype=torch.int32)
        writes = pp._control(t, n_tokens) + [
            (ttnn.from_torch(ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=shard), pp.ids)
        ]
        for host, dev in writes:
            ttnn.copy_host_to_device_tensor(host, dev)
        for rep in range(a.op_reps + 1):
            tag = f"C{C}_x{t * C}_r{rep}"
            ttnn.synchronize_device(pp.mesh)
            signpost(f"start_{tag}")
            t0 = time.perf_counter()
            pp._body()
            ttnn.synchronize_device(pp.mesh)
            signpost(f"stop_{tag}")
            log(f"ops {tag}: eager tick {1e3 * (time.perf_counter() - t0):.1f} ms (host dispatch included)")
            read_profiler()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["ttft", "ops", "all"])
    ap.add_argument("--T", default="128,2048,8192,32768")
    ap.add_argument("--chunk-T", default="")
    ap.add_argument("--queue-T", default="")
    ap.add_argument("--queue-n", type=int, default=4)
    ap.add_argument("--handoff-T", default="", help="(trees with the split handoff: _ho_jobs / _ho_copy)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--cool", type=float, default=0.0, help="seconds to sleep before each measured run")
    ap.add_argument("--long-cool", type=float, default=60.0)
    ap.add_argument("--max-len", type=int, default=0, help="default: the largest T + 1024")
    ap.add_argument("--chunks", default="128,256,512,1024")
    ap.add_argument("--ops", default="1024:0,512:0,256:0,128:0")
    ap.add_argument("--op-reps", type=int, default=1)
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--out", default="")
    ap.add_argument("--save-logits", default="", help="ttft: save the first logits per prompt length")
    a = ap.parse_args()
    ints = lambda s: [int(x) for x in s.split(",") if x]
    a.T, a.chunk_T, a.queue_T, a.handoff_T = ints(a.T), ints(a.chunk_T), ints(a.queue_T), ints(a.handoff_T)
    chunks = ints(a.chunks)
    faulthandler.dump_traceback_later(1800, repeat=True, file=sys.stderr)
    mesh = open_mesh()
    clock = Clock(mesh, log)
    out = dict(args=vars(a))
    try:
        if a.mode == "ops":
            longest = max((int(s.split(":")[1]) + 4 * int(s.split(":")[0]) for s in a.ops.split(",")), default=4096)
            r = Served(mesh, max(a.max_len, longest), chunks, clock, decode=False)
            run_ops(r.pp, a)
            return
        corpus = torch.load(a.corpus)["ids"]
        longest = max(a.T + a.chunk_T + a.queue_T + a.handoff_T + [1])
        r = Served(mesh, a.max_len or longest + 1024, chunks, clock)
        logits = {} if a.save_logits else None
        out["ttft"] = run_ttft(r, a, corpus, [T for T in a.T if T < LONG], logits)
        if a.chunk_T:
            out["chunks"] = run_chunks(r, a, corpus)
        if a.queue_T:
            out["queue"] = run_queue(r, a, corpus)
        if a.handoff_T:
            out["handoff"] = run_handoff(r, a, corpus)
        if any(T >= LONG for T in a.T):
            cool(a.long_cool)
            out["ttft"] += run_ttft(r, a, corpus, [T for T in a.T if T >= LONG], logits)
        if a.save_logits:
            torch.save(logits, a.save_logits)
    finally:
        if a.out:
            with open(a.out, "w") as f:
                json.dump(out, f, indent=1)
        close_mesh(mesh)


if __name__ == "__main__":
    main()
