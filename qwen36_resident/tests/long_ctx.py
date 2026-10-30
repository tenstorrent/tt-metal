# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Long-context checks of the resident Qwen3.8 model (up to max_model_len 262144) on the 1 x 4 mesh.

corpus:   natural text (the tech_reports markdown of this tree, files in sorted order) tokenized once:
          ids [N] int64 and the file list, for prompts that are not one repeated sentence.
ttnn:     the long-context reference: our ttnn TP model (models/demos/blackhole/qwen36), paged KV, chunked
          prefill (prefill_for_spec) of the first P corpus tokens, then `steps` paged decodes teacher-forced
          on the corpus; saves the 1 + steps logit rows per prompt length.
resident: builds the resident decode + pipeline-parallel prefill for max_len (as generator_vllm does) and
          reports host RSS and per-chip DRAM per build phase; then optionally
            --ref:     teacher-forced top-1 against the Hugging Face reference (every argmax recorded),
            --teacher: the ttnn run's teacher-forced rows at the same prompt lengths (logits saved, and
                       compared with a ttnn file of the same length when one exists),
            --speed:   TTFT (prefill of all but the last prompt token, handoff, the last token's step and its
                       logits) and the traced hub-fed greedy TPOT per prompt length, with the first greedy
                       tokens recorded.
compare:  two saved logit files (ttnn vs resident, or resident before vs after a change).

usage: python -m qwen36_resident.tests.long_ctx corpus --out ids.pt
       python -m qwen36_resident.tests.long_ctx ttnn --corpus ids.pt --prompts 65536,131072 --out-dir D
       python -m qwen36_resident.tests.long_ctx resident --corpus ids.pt --max-len 262144 --teacher 65536
           --speed 8192,32768 --out-dir D --tag res
"""
import argparse
import faulthandler
import glob
import inspect
import json
import os
import resource
import sys
import time

import torch

import ttnn

TT_BYTES = {
    ttnn.bfloat16: 2.0,
    ttnn.float32: 4.0,
    ttnn.uint32: 4.0,
    ttnn.int32: 4.0,
    ttnn.bfloat8_b: 1088 / 1024,
    ttnn.bfloat4_b: 576 / 1024,
}


def host_mem():
    """(current, peak) resident set of this process in GB"""
    cur = peak = 0.0
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS:"):
            cur = int(line.split()[1]) / 2**20
        elif line.startswith("VmHWM:"):
            peak = int(line.split()[1]) / 2**20
    return cur, max(peak, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)


def dram(mesh):
    """(allocated, total) DRAM per chip in GB (the mesh allocates every buffer on all chips in lockstep)"""
    v = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
    return v.total_bytes_allocated_per_bank * v.num_banks / 1e9, v.total_bytes_per_bank * v.num_banks / 1e9


def chip_bytes(t):
    """DRAM / L1 bytes of a device tensor on one chip"""
    if t is None or not ttnn.is_tensor_storage_on_device(t):
        return 0
    d = ttnn.get_device_tensors(t)[0]
    vol = 1
    for s in d.padded_shape:
        vol *= s
    return vol * TT_BYTES.get(d.dtype, 4.0)


class Clock:
    def __init__(self, mesh, log):
        self.mesh, self.log, self.t0, self.rows = mesh, log, time.time(), []

    def __call__(self, name):
        cur, peak = host_mem()
        alloc, total = dram(self.mesh) if self.mesh is not None else (0, 0)
        row = dict(phase=name, s=time.time() - self.t0, rss_gb=cur, rss_peak_gb=peak, dram_gb=alloc)
        self.rows.append(row)
        self.log(
            f"[{row['s']:7.1f} s] {name}: host RSS {cur:.1f} GB (peak {peak:.1f}), DRAM/chip {alloc:.2f} of {total:.2f} GB"
        )


def log(msg):
    print(msg, flush=True)


def open_mesh(trace_region=134217728, **kw):
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    return ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=trace_region, **kw)


def close_mesh(mesh):
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


# ---------------------------------------------------------------------------------------------- corpus
def _prose(path):
    lines = [l for l in open(path, encoding="utf-8", errors="replace").read().splitlines() if l.strip()]
    return sum(l.lstrip().startswith("|") for l in lines) < 0.3 * max(1, len(lines))


def build_corpus(out, tokens):
    from transformers import AutoTokenizer

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    tok = AutoTokenizer.from_pretrained(os.environ["HF_MODEL"])
    files = sorted(glob.glob(f"{root}/tech_reports/**/*.md", recursive=True))
    files += sorted(glob.glob(f"{root}/docs/source/**/*.rst", recursive=True))
    # mostly prose: a file that is mainly a table (GEMM_FLOPS: ~30K tokens of benchmark rows) is left out
    files = [f for f in files if _prose(f)]
    ids, used = [], []
    for f in files:
        text = open(f, encoding="utf-8", errors="replace").read()
        ids += tok(f"\n\n<!-- {os.path.relpath(f, root)} -->\n\n" + text, add_special_tokens=False)["input_ids"]
        used.append(os.path.relpath(f, root))
        if len(ids) >= tokens:
            break
    assert len(ids) >= tokens, f"corpus has only {len(ids)} tokens"
    torch.save(dict(ids=torch.tensor(ids[:tokens], dtype=torch.int64), files=used), out)
    log(f"corpus: {tokens} tokens from {len(used)} files -> {out}")


# ---------------------------------------------------------------------------------------------- compare
def compare(ref, dev, ids=None):
    """ref, dev: [N, vocab] logits of the same positions; ids: the true next tokens [N] (teacher accuracy)"""
    lr, ld = torch.log_softmax(ref.double(), -1), torch.log_softmax(dev.double(), -1)
    r1, d1 = lr.argmax(-1), ld.argmax(-1)
    agree = r1 == d1
    i = torch.arange(len(r1))
    gap_ref = (lr[i, r1] - lr[i, d1])[~agree]  # how much the reference prefers its pick over the device's
    gap_dev = (ld[i, d1] - ld[i, r1])[~agree]
    top10 = lr.topk(10, -1).indices
    out = dict(
        positions=len(r1),
        top1_agree=float(agree.double().mean()),
        disagreements=int((~agree).sum()),
        max_gap_ref=float(gap_ref.max()) if len(gap_ref) else 0.0,
        max_gap_dev=float(gap_dev.max()) if len(gap_dev) else 0.0,
        gaps_ref=[round(float(g), 4) for g in gap_ref],
        max_abs_lp_top10=float((lr.gather(-1, top10) - ld.gather(-1, top10)).abs().max()),
        mean_kl=float((lr.exp() * (lr - ld)).sum(-1).mean()),
    )
    if ids is not None:
        out["ref_acc"] = float((r1 == ids).double().mean())
        out["dev_acc"] = float((d1 == ids).double().mean())
    return out


# ---------------------------------------------------------------------------------------------- ttnn
@torch.no_grad()
def run_ttnn(a):
    from models.demos.blackhole.qwen36.tt.model import Qwen36Model

    corpus = torch.load(a.corpus)["ids"]
    prompts = [int(p) for p in a.prompts.split(",")]
    mesh = open_mesh(l1_small_size=24576, num_command_queues=2)
    clock = Clock(mesh, log)
    try:
        model = Qwen36Model.from_pretrained(mesh, max_batch_size=1, max_seq_len=a.max_seq_len)
        clock("ttnn model")
        block = 64
        nb = a.max_seq_len // block
        kv_shape = [nb + 1, model.args.n_local_kv_heads, block, model.args.head_dim]
        pt = torch.arange(nb, dtype=torch.int32).reshape(1, nb)
        vocab = model.vocab_size
        for P in prompts:
            assert P + a.steps <= a.max_seq_len
            model.free_kv_caches()
            model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)
            clock(f"P={P}: KV caches")
            prompt = corpus[:P].to(torch.int32).reshape(1, P)
            t0 = time.time()
            lg = model.prefill_for_spec(prompt, pt, P, lambda hidden, chunk_start, valid_len: None)
            lt = ttnn.to_torch(lg, mesh_composer=ttnn.ConcatMeshToTensor(model.mesh_device, dim=0))
            ttnn.deallocate(lg)
            ttft = time.time() - t0
            rows = [lt.reshape(-1)[:vocab].float()]
            clock(f"P={P}: prefill {ttft:.1f} s")
            t0 = time.time()
            for k in range(a.steps):
                row, hidden = model.decode_step_paged(int(corpus[P + k]), P + k, pt)
                ttnn.deallocate(hidden)
                rows.append(row[:vocab].float())
            dt = (time.time() - t0) / a.steps
            logits = torch.stack(rows)
            acc = float((logits.argmax(-1) == corpus[P : P + a.steps + 1]).double().mean())
            path = f"{a.out_dir}/ttnn_P{P}.pt"
            torch.save(dict(P=P, steps=a.steps, logits=logits, ttft_s=ttft, decode_s=dt, teacher_acc=acc), path)
            clock(f"P={P}: {a.steps} decodes ({1e3 * dt:.1f} ms each), teacher acc {acc:.4f} -> {path}")
        model.free_kv_caches()
    finally:
        close_mesh(mesh)


# ---------------------------------------------------------------------------------------------- resident
def check_zero_dram(mesh):
    """common.zero_dram zeroes exactly its tensors, one or many per call: runs of neighbouring buffers of ones of four
    kinds (interleaved bf8 tiles, height-sharded bf8 tiles as the decode KV caches, interleaved uint32 rows, one 64-byte
    page that leaves three of every bank's four cores without a range), every other buffer zeroed. One tensor per call
    first (as device_zeros, which seeds the program cache), then 28 per call twice (as PPPrefill.reset; the second hits
    the cache at other addresses), then one per call again: a cache hit must not reuse a program built for shorter
    per-core runtime args (the 0d5fc31baa6 reset crash)"""
    from qwen36_resident.common import zero_dram

    banks = mesh.dram_grid_size().x
    sharded = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))}),
            [3 * 32, 256],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    kinds = [
        ([1, 16], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, ttnn.DRAM_MEMORY_CONFIG),
        ([5, 4, 64, 256], ttnn.bfloat8_b, ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG),
        ([banks * 3 * 32, 256], ttnn.bfloat8_b, ttnn.TILE_LAYOUT, sharded),
        ([77, 32], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, ttnn.DRAM_MEMORY_CONFIG),
    ]
    rep = ttnn.ReplicateTensorToMesh(mesh)
    comp = ttnn.ConcatMeshToTensor(mesh, dim=0)

    def run(per_kind, batched):
        """2 per_kind + 1 neighbours of each kind, the odd ones zeroed (all in one call when batched); returns them"""
        runs = []
        for shape, dt, layout, mc in kinds:
            ones = torch.ones(shape, dtype=torch.int32 if dt == ttnn.uint32 else torch.float32)
            runs.append(
                [
                    ttnn.from_torch(ones, dtype=dt, layout=layout, device=mesh, memory_config=mc, mesh_mapper=rep)
                    for _ in range(2 * per_kind + 1)
                ]
            )
        zeroed = [t for ts in runs for t in ts[1::2]]
        for group in [zeroed] if batched else [[t] for t in zeroed]:
            zero_dram(mesh, group)
        for (shape, dt, _, _), ts in zip(kinds, runs):
            for i, t in enumerate(ts):
                got = ttnn.to_torch(t, mesh_composer=comp).float()
                if i % 2:
                    assert got.abs().max() == 0, ("not zeroed", shape, dt, len(zeroed) if batched else 1)
                else:
                    assert bool((got == 1).all()), ("a neighbour changed", shape, dt, len(zeroed) if batched else 1)
        return [t for ts in runs for t in ts]

    live = run(1, False) + run(7, True) + run(7, True) + run(1, False)
    for t in live:
        ttnn.deallocate(t)
    log("zero_dram: zeroes its tensors and leaves the neighbours alone (1 per call, 28 per call twice, 1 per call)")


class Resident:
    """the resident decode + PP prefill, as generator_vllm builds them"""

    def __init__(self, mesh, max_seq_len, clock):
        from qwen36_resident import weights as Q
        from qwen36_resident.model import Dims, ResidentModel, State, is_attn
        from qwen36_resident.prefill.pp_prefill import PPPrefill

        chunks = (128, 256, 512, 1024)
        self.mesh = mesh
        ck = Q.Checkpoint(Q.checkpoint_dir())
        cfg = ck.config
        layers, interval, self.vocab = cfg["num_hidden_layers"], cfg["full_attention_interval"], cfg["vocab_size"]
        self.d = d = Dims(mesh.get_num_devices(), mesh.dram_grid_size().x, vocab=self.vocab)
        self.emb = Q.embedding(ck)
        C = max(chunks)
        self.max_len = max_len = -(-max_seq_len // C) * C
        n_attn = sum(is_attn(l, interval) for l in range(layers))
        kw = dict(kv=False) if "kv" in inspect.signature(State.__init__).parameters else {}
        st = State(d, layers - n_attn, n_attn, 0, max_pos=max_len + 64, zero=True, **kw)
        clock(f"state (kv images: {not kw})")
        self.pp = PPPrefill(mesh, ck, layers, interval, chunk=list(chunks), max_len=max_len)
        clock("prefill model")
        w = Q.load(ck, d, layers, interval)
        clock("decode weights (host)")
        self.model = ResidentModel(mesh, d, w, st, layers, interval, lm_head=True, embed=self.pp.embed)
        del w, st
        clock("decode model")
        m = self.model
        m.step(m.token_state(self.emb[0].float(), 0))
        ttnn.synchronize_device(mesh)
        m.reset()
        m.capture_trace()
        clock("decode trace")
        self.pp.capture()
        clock("prefill trace")
        self.pp.handoff(m, 1)
        ttnn.synchronize_device(mesh)
        clock("handoff compiled")

    def budget(self):
        """per-chip GB of the device tensors by kind"""
        pp, m = self.pp, self.model
        gb = lambda ts: sum(chip_bytes(t) for t in ts) / 1e9
        smm = [e[k] for e in pp.w for k in ("G_s", "U_s", "W_s") if k in e]
        pp_w = [v for e in pp.w for k, v in e.items() if k not in ("G_s", "U_s", "W_s")]
        geo = [v for g in pp._geo.values() for k, v in g.items() if isinstance(v, ttnn.Tensor)]
        geo += [v for g in pp._geo.values() if g.get("smm_out") for v in g["smm_out"].values()]
        out = dict(
            decode_weights=gb(m._keep),
            embedding=gb([pp.embed]),
            prefill_weights=gb(pp_w),
            prefill_smm_copies=gb(smm),
            prefill_kv=gb(list(pp.K.values()) + list(pp.V.values())),
            prefill_state_and_buffers=gb(list(pp.S.values()) + list(pp.carry.values()) + geo),
            decode_kv=gb([m.K_t, m.V_t]),
            decode_gdn_state=gb([m.state_t]),
            decode_conv_hist=gb([m.hist_t]),
            rope_table=gb([getattr(m, "rope_t", None)]),
            token_state=gb([m.tok_t, m.x_t]),
        )
        out["sum"] = sum(out.values())
        return out

    def logits_at(self, token, pos):
        m = self.model
        m.step(m.token_state(self.emb[int(token)].float(), int(pos)))
        ttnn.synchronize_device(self.mesh)
        return m.logits()[:, : self.d.vocab_chip].reshape(-1)[: self.vocab].float()

    def prefill(self, ids, timings=None):
        """the prompt's state and its last token's logits (generator_vllm.prefill_forward); timings: the
        seconds of the prefill ticks and of each handoff part (synchronizing between them)"""
        m, pp = self.model, self.pp
        m.reset()
        pp.reset()
        T = len(ids)
        if T > 1:
            t0 = time.time()
            pp.run(ids[: T - 1])
            if timings is not None:
                ttnn.synchronize_device(self.mesh)
                timings["prefill_ticks"] = time.time() - t0
            pp.handoff(m, T - 1, timings)
        return self.logits_at(ids[T - 1], T - 1)


def core_map(mesh, model):
    """the decode's core roles on the grid (as ResidentModel places them), one character per core"""
    from qwen36_resident.common import streamer_cores
    from qwen36_resident.model import WORKERS, worker_rect

    groups, used = streamer_cores(mesh)
    g = mesh.compute_with_storage_grid_size()
    role = {xy: "s" for xy in used}
    cores = [c for grp in groups for c in grp]
    xs, ys = [c.x for c in cores], [c.y for c in cores]
    hub = next((x, y) for x in range(min(xs), max(xs) + 1) for y in range(min(ys), max(ys) + 1) if (x, y) not in used)
    used.add(hub)
    role[hub] = "H"
    workers = worker_rect(mesh, used, WORKERS)
    for c in workers:
        used.add((c.x, c.y))
        role[(c.x, c.y)] = "w"
    free = [(x, y) for y in range(g.y) for x in range(g.x) if (x, y) not in used]
    role[free[0]] = "L"
    for xy in free[1 : 1 + model.d.nv * model.head_groups]:
        role[xy] = "g"
    rows = ["".join(role.get((x, y), ".") for x in range(g.x)) for y in range(g.y)]
    log("cores (s streamer, H hub, w attention worker, L leader, g GDN head, . free):\n  " + "\n  ".join(rows))
    return rows


def hf_teacher(r, ref_path):
    ref = torch.load(ref_path)
    P, N = ref["prefix"], ref["scored"]
    out = dict(argmax=[], top1=0.0)
    hits = total = 0
    for s in ref["seqs"]:
        ids = s["ids"]
        lg = r.prefill(ids[:P])
        picks = []
        for k in range(N):
            p = P - 1 + k
            if k:
                lg = r.logits_at(ids[p], p)
            picks.append(int(lg.argmax()))
            hits += picks[-1] == int(s["top_i"][p][0])
        total += N
        out["argmax"].append(picks)
    out["top1"] = hits / total
    log(f"HF teacher-forced: top-1 {out['top1']:.4f} over {total} positions")
    return out


def resident_teacher(r, corpus, P, steps, out_dir, tag):
    ttnn.synchronize_device(r.mesh)
    t0 = time.time()
    parts = {}
    rows = [r.prefill(corpus[:P], parts)]
    ttft = time.time() - t0
    t0 = time.time()
    for k in range(steps):
        rows.append(r.logits_at(corpus[P + k], P + k))
    dt = (time.time() - t0) / steps
    logits = torch.stack(rows)
    nxt = corpus[P : P + steps + 1]
    path = f"{out_dir}/{tag}_P{P}.pt"
    torch.save(dict(P=P, steps=steps, logits=logits, ttft_s=ttft), path)
    res = dict(P=P, ttft_s=ttft, ttft_parts_s=parts, step_ms_untraced=1e3 * dt)
    res["teacher_acc"] = float((logits.argmax(-1) == nxt).double().mean())
    ref = f"{out_dir}/ttnn_P{P}.pt"
    if os.path.exists(ref):
        res["vs_ttnn"] = compare(torch.load(ref)["logits"], logits, nxt)
    log(f"teacher P={P}: {json.dumps(res)}")
    return res


def speed(r, corpus, T, steps, greedy):
    m, mesh = r.model, r.mesh
    prompt = corpus[:T]
    ttnn.synchronize_device(mesh)
    t0 = time.time()
    lg = r.prefill(prompt)
    ttft = time.time() - t0
    first = int(lg.argmax())
    m.set_sampling(0.0, 1, 1.0, 0)
    m.step(m.token_state(r.emb[first].float(), T))
    toks = [first]
    for _ in range(greedy):
        ttnn.synchronize_device(mesh)
        toks.append(m.fed_token()[0])
        m.step(None)
    for _ in range(8):
        m.step(None)
    ttnn.synchronize_device(mesh)
    t0 = time.time()
    for _ in range(steps):
        m.step(None)
    ttnn.synchronize_device(mesh)
    tpot = (time.time() - t0) / steps * 1e3
    log(f"T={T}: TTFT {ttft:.3f} s, TPOT {tpot:.2f} ms ({1e3 / tpot:.1f} tok/s), greedy {toks[:12]}...")
    return dict(T=T, ttft_s=ttft, tpot_ms=tpot, greedy=toks)


def run_resident(a):
    corpus = torch.load(a.corpus)["ids"] if a.corpus else None
    mesh = open_mesh()
    clock = Clock(mesh, log)
    out = dict(max_len=a.max_len, tag=a.tag)
    try:
        clock("mesh open")
        try:
            check_zero_dram(mesh)
        except ImportError:  # (a tree without zero_dram)
            pass
        r = Resident(mesh, a.max_len, clock)
        out["build"] = clock.rows
        out["budget_gb_per_chip"] = r.budget()
        out["cores"] = core_map(mesh, r.model)
        out["dram_view_gb_per_chip"] = dram(mesh)
        log(f"budget per chip (GB): {json.dumps({k: round(v, 3) for k, v in out['budget_gb_per_chip'].items()})}")
        if a.speed_first:
            out["speed"] = [speed(r, corpus, int(T), a.tpot_steps, a.greedy) for T in _ints(a.speed)]
        if a.ref:
            out["hf"] = hf_teacher(r, a.ref)
        out["teacher"] = [resident_teacher(r, corpus, int(P), a.steps, a.out_dir, a.tag) for P in _ints(a.teacher)]
        if not a.speed_first:
            out["speed"] = [speed(r, corpus, int(T), a.tpot_steps, a.greedy) for T in _ints(a.speed)]
        out["end"] = dict(zip(("rss_gb", "rss_peak_gb"), host_mem()))
    finally:
        if a.out_dir:
            json.dump(out, open(f"{a.out_dir}/{a.tag}.json", "w"), indent=1)
        close_mesh(mesh)


def run_tpot(a):
    """decode-only TPOT at positions: the resident decode without the prefill (its KV caches zero-filled: a step's
    attention cost depends on its position only), hub-fed greedy steps from each position"""
    from qwen36_resident import weights as Q
    from qwen36_resident.model import ATTN_2NOC, ATTN_RR, WORKERS, Dims, ResidentModel, State, is_attn

    mesh = open_mesh()
    clock = Clock(mesh, log)
    out = dict(max_len=a.max_len, tag=a.tag, workers=WORKERS, round_robin=ATTN_RR, two_nocs=ATTN_2NOC, tpot={})
    try:
        ck = Q.Checkpoint(Q.checkpoint_dir())
        cfg = ck.config
        layers, interval = cfg["num_hidden_layers"], cfg["full_attention_interval"]
        d = Dims(mesh.get_num_devices(), mesh.dram_grid_size().x, vocab=cfg["vocab_size"])
        emb = Q.embedding(ck)
        embed = ttnn.from_torch(
            emb.bfloat16(),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        n_attn = sum(is_attn(l, interval) for l in range(layers))
        st = State(d, layers - n_attn, n_attn, 0, max_pos=a.max_len + 64, zero=True, kv=False)
        m = ResidentModel(mesh, d, Q.load(ck, d, layers, interval), st, layers, interval, lm_head=True, embed=embed)
        clock("decode model")
        out["cores"] = core_map(mesh, m)
        m.step(m.token_state(emb[0].float(), 0))
        ttnn.synchronize_device(mesh)
        m.capture_trace()
        m.set_sampling(0.0, 1, 1.0, 0)
        for T in _ints(a.speed):
            m.reset()
            m.step(m.token_state(emb[0].float(), T))
            for _ in range(8):
                m.step(None)
            ttnn.synchronize_device(mesh)
            t0 = time.time()
            for _ in range(a.tpot_steps):
                m.step(None)
            ttnn.synchronize_device(mesh)
            out["tpot"][T] = (time.time() - t0) / a.tpot_steps * 1e3
            log(f"decode-only T={T}: TPOT {out['tpot'][T]:.2f} ms")
    finally:
        if a.out_dir:
            json.dump(out, open(f"{a.out_dir}/{a.tag}.json", "w"), indent=1)
        close_mesh(mesh)


def _ints(s):
    return [int(v) for v in s.split(",")] if s else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["corpus", "ttnn", "resident", "tpot", "compare"])
    ap.add_argument("--out")
    ap.add_argument("--tokens", type=int, default=262400)
    ap.add_argument("--corpus")
    ap.add_argument("--prompts", default="65536,131072,261888")
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--max-seq-len", type=int, default=262144)
    ap.add_argument("--out-dir")
    ap.add_argument("--max-len", type=int, default=262144)
    ap.add_argument("--tag", default="res")
    ap.add_argument("--ref")
    ap.add_argument("--teacher", default="")
    ap.add_argument("--speed", default="")
    ap.add_argument("--tpot-steps", type=int, default=128)
    ap.add_argument("--greedy", type=int, default=32)
    ap.add_argument("--speed-first", action="store_true", help="the speed runs before the teacher-forced ones")
    ap.add_argument("files", nargs="*")
    a = ap.parse_args()
    faulthandler.dump_traceback_later(300, repeat=True, file=sys.stderr)
    if a.mode == "corpus":
        build_corpus(a.out, a.tokens)
    elif a.mode == "ttnn":
        run_ttnn(a)
    elif a.mode == "resident":
        run_resident(a)
    elif a.mode == "tpot":
        run_tpot(a)
    else:
        x, y = (torch.load(f) for f in a.files)
        log(json.dumps(compare(x["logits"], y["logits"]), indent=1))
        log(f"bit-identical: {torch.equal(x['logits'], y['logits'])}")


if __name__ == "__main__":
    main()
