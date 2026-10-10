# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DEVICE test of model-side prefix caching (GDN snapshot cache) driven like the vLLM TT plugin.

A (1, 8) mesh is opened like a vLLM DP rank (FABRIC_1D_RING, 1 GiB trace region); Qwen36ForCausalLM is initialised,
warmed up (eager then traced, as the plugin does) and prefill_forward is called directly with hand-built page tables
that mimic vLLM APC (block size 64, block 0 = null block, cached prefix blocks SHARED between requests, start_pos =
cached tokens).  Every cached-resume result is compared with a REFERENCE full prefill (start_pos=0, fresh blocks) of the
identical tokens: last-token logits PCC, top-1 equality, top-5 overlap, wall time.

    pytest models/demos/blackhole/qwen36/tests/test_prefix_cache.py -svq     (QWEN36_PREFIX_CACHE_DEBUG=1 for the log)
"""

import glob
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.sampling.sampling_params import SamplingParams
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.tt.qwen36_vllm import Qwen36ForCausalLM

BLOCK = 64
MAX_SEQ = 16384
BPU = MAX_SEQ // BLOCK  # page-table width (max_model_len / block_size)
NUM_BLOCKS = 1024  # 65536 KV tokens
WIDTH = 32  # max_num_seqs
PCC_MIN = 0.99
PCC_S4_MIN = 0.9999

DEVICE_PARAMS = [
    {
        "l1_small_size": 24576,
        "num_command_queues": 2,
        "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
        "trace_region_size": 1024 * 1024 * 1024,
    }
]

RESULTS = []  # (scenario, name, hit/notes, pcc, top1, top5, t_cached, t_ref)


def pcc(a, b):
    a, b = a.double().reshape(-1), b.double().reshape(-1)
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


class Harness:
    def __init__(self, gen, kv, corpus):
        self.gen, self.kv, self.corpus = gen, kv, corpus
        self.vocab = gen.model[0].args.vocab_size
        self.next_block = 1  # block 0 is vLLM's null block, never assigned
        self.cursor = 0

    def text(self, n):
        """n fresh corpus tokens (every call returns a different, non-overlapping slice)."""
        out = self.corpus[self.cursor : self.cursor + n]
        self.cursor += n
        assert len(out) == n, "corpus too short"
        return out

    def alloc(self, n):
        ids = list(range(self.next_block, self.next_block + n))
        self.next_block += n
        assert self.next_block <= NUM_BLOCKS, "out of KV blocks"
        return ids

    def blocks_for(self, L, shared=()):
        """Page-table blocks for a prompt of L tokens (+1 spare block for decode); `shared` are the cached prefix blocks."""
        need = (L + BLOCK) // BLOCK + 1
        return list(shared) + self.alloc(need - len(shared))

    def prefill(self, toks, blocks, start_pos, slot):
        L = len(toks)
        pt = torch.zeros(1, BPU, dtype=torch.int32)
        pt[0, : len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
        t = torch.tensor(toks, dtype=torch.int32).reshape(1, L)
        t0 = time.perf_counter()
        logits, _ = self.gen.prefill_forward(
            t, pt, self.kv, [L], start_pos=[start_pos], empty_slots=[slot], enable_trace=True
        )
        dt = time.perf_counter() - t0
        return logits.reshape(-1)[: self.vocab].float(), dt

    def compare(self, scen, name, notes, lc, tc, lr, tr):
        p = pcc(lc, lr)
        top1 = int(lc.argmax()) == int(lr.argmax())
        top5 = len(set(lc.topk(5).indices.tolist()) & set(lr.topk(5).indices.tolist()))
        RESULTS.append((scen, name, notes, p, top1, top5, tc, tr))
        logger.info(
            f"RESULT {scen} {name}: {notes} pcc={p:.6f} top1={top1} top5={top5}/5 t_cached={tc:.3f}s t_ref={tr:.3f}s"
        )
        return p, top1

    def cached_vs_ref(self, scen, name, toks, shared_blocks, start_pos, slot=2, ref_slot=2):
        blocks = self.blocks_for(len(toks), shared_blocks)
        lc, tc = self.prefill(toks, blocks, start_pos, slot)
        rblocks = self.blocks_for(len(toks))
        lr, tr = self.prefill(toks, rblocks, 0, ref_slot)
        p, t1 = self.compare(scen, name, f"L={len(toks)} start={start_pos}", lc, tc, lr, tr)
        return blocks, rblocks, lc, lr, p, t1

    def greedy_decode(self, reqs, steps):
        """reqs: list of (blocks, prompt_len, first_token) in slot order (row i == slot i). Greedy device sampling,
        full authoritative reload every step (host_reload mode of the plugin contract)."""
        n = len(reqs)
        outs = [[] for _ in reqs]
        sp = SamplingParams(temperature=[0.0] * WIDTH, top_k=[1] * WIDTH, top_p=[1.0] * WIDTH, seed=[None] * WIDTH)
        for s in range(steps):
            tokens = torch.zeros(WIDTH, 1, dtype=torch.int32)
            start = torch.full((WIDTH,), -1, dtype=torch.int32)
            pt = torch.zeros(WIDTH, BPU, dtype=torch.int32)
            for i, (blocks, L, first) in enumerate(reqs):
                tokens[i, 0] = outs[i][-1] if outs[i] else first
                start[i] = L + s
                pt[i, : len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
            out = self.gen.decode_forward(
                tokens=tokens,
                start_pos=start,
                page_table=pt,
                kv_cache=self.kv,
                sampling_params=sp,
                reload_inputs=True,
                reload_page_table=False,
                reload_sampling_params=True,
                reset_sampling_state=True,
                enable_trace=True,
                read_from_device=True,
            )
            toks = torch.as_tensor(out[0] if isinstance(out, tuple) else out).reshape(-1)
            for i in range(n):
                outs[i].append(int(toks[i]))
        return outs


def _corpus(tokenizer):
    files = sorted(glob.glob("tech_reports/**/*.md", recursive=True), key=os.path.getsize, reverse=True)[:40]
    text = "\n\n".join(open(f, errors="ignore").read() for f in files)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    assert len(ids) > 40000, len(ids)
    return ids


import json
import random
import statistics

LENS = [int(x) for x in os.environ.get("PO_LENS", "128,1024,2048,2049,4096,8192").split(",")]
REPS = 3
BREAKDOWN = os.environ.get("PO_BREAKDOWN", "0") == "1"
OUT = os.environ.get("PO_OUT", "/tmp/po.jsonl")
ACC = {}
COUNTS = {}
DEV = [None]


def _wrap(obj, name, key):
    orig = getattr(obj, name)

    def w(*a, **k):
        ttnn.synchronize_device(DEV[0])
        t0 = time.perf_counter()
        r = orig(*a, **k)
        ttnn.synchronize_device(DEV[0])
        ACC[key] = ACC.get(key, 0.0) + (time.perf_counter() - t0) * 1000
        ACC[key + "#"] = ACC.get(key + "#", 0) + 1
        return r

    setattr(obj, name, w)


def _count(mod, name):
    orig = getattr(mod, name)

    def w(*a, **k):
        COUNTS[name] = COUNTS.get(name, 0) + 1
        return orig(*a, **k)

    setattr(mod, name, w)


@run_for_blackhole()
@pytest.mark.timeout(3500)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_prefill_overhead(mesh_device, reset_seeds, ensure_gc):
    from transformers import AutoConfig, AutoTokenizer

    DEV[0] = mesh_device
    hf = os.environ["HF_MODEL"]
    tok = AutoTokenizer.from_pretrained(hf)
    hf_config = AutoConfig.from_pretrained(hf)
    gen = Qwen36ForCausalLM.initialize_vllm_model(hf_config, mesh_device, max_batch_size=WIDTH, max_seq_len=MAX_SEQ)
    model = gen.model[0]
    kv_shape = (NUM_BLOCKS, model.args.n_local_kv_heads, BLOCK, model.args.head_dim)
    kv = gen.allocate_kv_cache(kv_shape, ttnn.bfloat16, len(model.layers))
    dkw = dict(kv_cache=kv, max_batch_size=WIDTH, num_blocks=BPU, can_sample_on_device=True)
    gen.warmup_model_prefill(kv_cache=kv, enable_trace=False)
    gen.warmup_model_decode(enable_trace=False, **dkw)
    gen.already_warmed_up_prefill = False
    gen.warmup_model_prefill(kv_cache=kv, enable_trace=True)
    gen.warmup_model_decode(enable_trace=True, **dkw)
    cache = model._prefix_cache
    logger.info(
        f"PO config PC={os.environ.get('QWEN36_PREFIX_CACHE')} OP={os.environ.get('QWEN36_GDN_DECODE_STEP_OP')} "
        f"cache={cache is not None}"
    )

    h = Harness(gen, kv, _corpus(tok))
    rng = random.Random(1234)

    if BREAKDOWN:
        _wrap(model, "prefill_paged_slots", "prefill_paged_slots")
        _wrap(model, "_prefill_with_prefix_cache", "_prefill_with_prefix_cache")
        _wrap(model, "prefill_traced_chunked", "legacy_prefill_traced_chunked")
        _wrap(model, "_prepare_short_trace_gdn_init", "_prepare_short_trace_gdn_init")
        _wrap(model, "_write_gdn_slot_from_scratch", "_write_gdn_slot_from_scratch")
        _wrap(model, "_prefill_range", "_prefill_range")
        _wrap(model, "prepare_gdn_decode_width", "prepare_gdn_decode_width")
        if cache is not None:
            _wrap(cache, "save", "cache.save")
            _wrap(cache, "restore", "cache.restore")
    for n in ("copy", "embedding", "to_torch", "from_torch", "execute_trace"):
        _count(ttnn, n)

    sp = SamplingParams(temperature=[0.0] * WIDTH, top_k=[1] * WIDTH, top_p=[1.0] * WIDTH, seed=[None] * WIDTH)

    def decode_once(blocks, pos, tokval):
        tokens = torch.zeros(WIDTH, 1, dtype=torch.int32)
        tokens[0, 0] = tokval
        start = torch.full((WIDTH,), -1, dtype=torch.int32)
        start[0] = pos
        pt = torch.zeros(WIDTH, BPU, dtype=torch.int32)
        pt[0, : len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
        t0 = time.perf_counter()
        out = gen.decode_forward(
            tokens=tokens,
            start_pos=start,
            page_table=pt,
            kv_cache=kv,
            sampling_params=sp,
            reload_inputs=True,
            reload_page_table=False,
            reload_sampling_params=True,
            reset_sampling_state=True,
            enable_trace=True,
            read_from_device=True,
        )
        ttnn.synchronize_device(mesh_device)
        dt = (time.perf_counter() - t0) * 1000
        toks = torch.as_tensor(out[0] if isinstance(out, tuple) else out).reshape(-1)
        return int(toks[0]), dt

    for L in LENS:
        res = []
        for rep in range(REPS + 1):  # rep 0 = discarded extra warm
            off = rng.randrange(0, len(h.corpus) - L - 1)
            toks = h.corpus[off : off + L]
            h.next_block = 1
            blocks = h.blocks_for(L)
            ACC.clear()
            COUNTS.clear()
            ttnn.synchronize_device(mesh_device)
            pt = torch.zeros(1, BPU, dtype=torch.int32)
            pt[0, : len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
            t = torch.tensor(toks, dtype=torch.int32).reshape(1, L)
            t0 = time.perf_counter()
            logits, _ = gen.prefill_forward(t, pt, kv, [L], start_pos=[0], empty_slots=[0], enable_trace=True)
            ttnn.synchronize_device(mesh_device)
            dt = (time.perf_counter() - t0) * 1000
            pacc, pcnt = dict(ACC), dict(COUNTS)
            nxt = int(logits.reshape(-1)[: h.vocab].float().argmax())
            ACC.clear()
            times = []
            dfirst = {}
            for s in range(6):
                nxt, d = decode_once(blocks, L + s, nxt)
                times.append(d)
                if s == 0:
                    dfirst = dict(ACC)
            if rep > 0:
                res.append(
                    dict(
                        L=L,
                        rep=rep,
                        prefill_ms=dt,
                        acc=pacc,
                        counts=pcnt,
                        dec_first=times[0],
                        dec_steady=statistics.median(times[2:]),
                        dec_acc_first=dfirst,
                    )
                )
            logger.info(
                f"PO L={L} rep={rep} prefill={dt:.1f}ms dec_first={times[0]:.1f} "
                f"steady={statistics.median(times[2:]):.1f} acc={ {k: round(v, 1) for k, v in pacc.items()} } "
                f"counts={pcnt} dec_acc_first={dfirst}"
            )
        keys = set().union(*[r["acc"] for r in res])
        with open(OUT, "a") as f:
            f.write(
                json.dumps(
                    dict(
                        L=L,
                        med_prefill=statistics.median(r["prefill_ms"] for r in res),
                        med_dec_first=statistics.median(r["dec_first"] for r in res),
                        med_dec_steady=statistics.median(r["dec_steady"] for r in res),
                        med_acc={k: statistics.median(r["acc"].get(k, 0) for r in res) for k in keys},
                        counts=res[-1]["counts"],
                        runs=res,
                        cfg=dict(
                            PC=os.environ.get("QWEN36_PREFIX_CACHE"),
                            OP=os.environ.get("QWEN36_GDN_DECODE_STEP_OP"),
                            BD=BREAKDOWN,
                        ),
                    )
                )
                + "\n"
            )
