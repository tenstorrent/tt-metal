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


@run_for_blackhole()
@pytest.mark.timeout(3500)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_prefix_cache_serving(mesh_device, reset_seeds, ensure_gc):
    from transformers import AutoConfig, AutoTokenizer

    hf = os.environ["HF_MODEL"]
    tok = AutoTokenizer.from_pretrained(hf)
    hf_config = AutoConfig.from_pretrained(hf)
    gen = Qwen36ForCausalLM.initialize_vllm_model(hf_config, mesh_device, max_batch_size=WIDTH, max_seq_len=MAX_SEQ)
    model = gen.model[0]
    kv_shape = (NUM_BLOCKS, model.args.n_local_kv_heads, BLOCK, model.args.head_dim)
    kv = gen.allocate_kv_cache(kv_shape, ttnn.bfloat16, len(model.layers))
    # plugin order: eager warmup, reset flag, traced prefill + decode warmup
    dkw = dict(kv_cache=kv, max_batch_size=WIDTH, num_blocks=BPU, can_sample_on_device=True)
    gen.warmup_model_prefill(kv_cache=kv, enable_trace=False)
    gen.warmup_model_decode(enable_trace=False, **dkw)
    gen.already_warmed_up_prefill = False
    gen.warmup_model_prefill(kv_cache=kv, enable_trace=True)
    gen.warmup_model_decode(enable_trace=True, **dkw)
    assert model._prefix_cache is not None, "GdnPrefixStateCache was not built by warmup_model_prefill"
    cache = model._prefix_cache

    h = Harness(gen, kv, _corpus(tok))
    failures = []

    def check(p, t1, tag, pmin=PCC_MIN):
        if not (p >= pmin and t1):
            failures.append(f"{tag}: pcc={p:.6f} top1_equal={t1}")

    # ---- S4 baseline: short prompt BEFORE any prefix-cache activity
    t300 = h.text(300)
    b = h.blocks_for(300)
    base_logits, _ = h.prefill(t300, b, 0, 2)
    assert cache.stats["saves"] == 0

    # ---- S1 long shared prefix
    shared = h.text(5000)
    sA, sB, sC = h.text(700), h.text(900), h.text(500)
    A = shared + sA
    a_blocks = h.blocks_for(len(A))
    lA, tA = h.prefill(A, a_blocks, 0, 2)
    logger.info(f"S1 A prefill {tA:.3f}s, snapshots after A: {sorted(cache._positions)}")
    nshare = 5000 // BLOCK  # 78
    B = shared + sB
    # B in slot 0 (cached), reference of B in slot 1 -> both decode afterwards
    b_blocks = h.blocks_for(len(B), a_blocks[:nshare])
    assert b_blocks[:nshare] == a_blocks[:nshare]
    lB, tB = h.prefill(B, b_blocks, nshare * BLOCK, 0)
    snaps_after_B = sorted(cache._positions)
    rb_blocks = h.blocks_for(len(B))
    lBr, tBr = h.prefill(B, rb_blocks, 0, 1)
    p, t1 = h.compare("S1", "B", f"L={len(B)} start={nshare*BLOCK} snaps_after_B={snaps_after_B}", lB, tB, lBr, tBr)
    check(p, t1, "S1 B")
    # decode B (slot 0) and its reference (slot 1) side by side
    firstB, firstBr = int(lB.argmax()), int(lBr.argmax())
    try:
        outs = h.greedy_decode([(b_blocks, len(B), firstB), (rb_blocks, len(B), firstBr)], 8)
        logger.info(f"S1 decode cached={outs[0]} ref={outs[1]}")
        RESULTS.append(("S1", "decode8", f"cached={outs[0]} ref={outs[1]}", float("nan"), outs[0] == outs[1], 0, 0, 0))
        if outs[0] != outs[1]:
            failures.append(f"S1 decode tokens differ: cached={outs[0]} ref={outs[1]}")
    except Exception as e:  # decode is secondary; report it
        logger.exception("S1 decode failed")
        failures.append(f"S1 decode raised {type(e).__name__}: {e}")
    C = shared + sC
    _, _, _, _, p, t1 = h.cached_vs_ref("S1", "C", C, a_blocks[:nshare], nshare * BLOCK)
    check(p, t1, "S1 C")

    # ---- S2 short shared prefix (<2048)
    sh2 = h.text(1000)
    D = sh2 + h.text(200)
    d_blocks = h.blocks_for(len(D))
    h.prefill(D, d_blocks, 0, 2)
    n2 = 1000 // BLOCK  # 15
    E = sh2 + h.text(300)
    _, _, _, _, p, t1 = h.cached_vs_ref("S2", "E", E, d_blocks[:n2], n2 * BLOCK)
    check(p, t1, "S2 E")
    F = sh2 + h.text(250)
    _, _, _, _, p, t1 = h.cached_vs_ref("S2", "F", F, d_blocks[:n2], n2 * BLOCK)
    check(p, t1, "S2 F")

    # ---- S3 exact repeat
    G = h.text(3000)
    g_blocks = h.blocks_for(len(G))
    lG1, tG1 = h.prefill(G, g_blocks, 0, 2)
    st = ((3000 - 1) // BLOCK) * BLOCK  # 2944
    lG2, tG2 = h.prefill(G, g_blocks, st, 2)  # same blocks: vLLM re-uses the cached prefix blocks, new tail blocks
    p, t1 = h.compare("S3", "G repeat", f"L=3000 start={st}", lG2, tG2, lG1, tG1)
    check(p, t1, "S3 G")

    # ---- S4 regression: same 300-token prompt after prefix-cache activity
    b2 = h.blocks_for(300)
    l4, t4 = h.prefill(t300, b2, 0, 2)
    p4 = pcc(l4, base_logits)
    same = torch.equal(l4, base_logits)
    RESULTS.append(("S4", "300tok", f"bitwise={same}", p4, int(l4.argmax()) == int(base_logits.argmax()), 0, t4, 0))
    check(p4, int(l4.argmax()) == int(base_logits.argmax()), "S4", PCC_S4_MIN)

    logger.info(f"cache stats: {cache.stats}")
    print("\n==== prefix-cache summary ====")
    for r in RESULTS:
        print(r)
    print("stats", cache.stats)
    assert not failures, "; ".join(failures)
