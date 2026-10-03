# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""END-TO-END accuracy harness on real GSM8K prompts: DEVICE prefill -> DEVICE decode state -> decode with the device writing its own KV.

For every batch of 16 questions (rows [16 b, 16 b + 16) of the reference generation in DSV41_E2E_REF, see reference/ref_e2e_gen.py):
  prefill   device prefill of the 16 prompts (or DSV41_BOOT=1: state seeded from the CPU reference prefill, batch 0 only)
  cl        closed loop: the device-sampled token is fed back (needs no reference); saves the device token stream
  tf        teacher forced along the reference's own greedy continuation (needs the finished reference); saves per-step device
            top-50 / logsumexp (all steps) and full bf16 logits (first 64 steps) + the device KV (DSV41_E2E_KV_BATCHES) for offline metrics
Between modes the decode state is restored from device-side clones taken right after the prefill (so cl / tf start from the same state).
Env: DSV41_E2E_REF, DSV41_E2E_OUT (dir), DSV41_E2E_BATCHES ("0,1,2,3"), DSV41_E2E_MODES ("cl,tf"), DSV41_CL_STEPS (200), DSV41_TF_STEPS (128),
     DSV41_E2E_KV_BATCHES ("0"), DSV41_BOOT, DSV41_ENGRAM_RAM (1).
Analysis: reference/e2e_analyze.py.   Horizon limit: positions < 256 (ratio-0 layers' linear cache / max_comp 256).
"""
import gc
import os
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference import ref_layer as R
from models.demos.blackhole.deepseek_v41_flash.tt.decoder import DSV41Decoder
from models.demos.blackhole.deepseek_v41_flash.tt.device_head import DSV41DeviceEmbedding, DSV41DeviceHead
from models.demos.blackhole.deepseek_v41_flash.tt.engram import DSV41DeviceEngram, HostEngramRows
from models.demos.blackhole.deepseek_v41_flash.tt.loader import load_layer
from models.demos.blackhole.deepseek_v41_flash.tt.model import DSV41DecodeChain
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import _Shards
from models.demos.blackhole.deepseek_v41_flash.tt.prefill_attention import DSV41PrefillAttention
from models.demos.blackhole.deepseek_v41_flash.tt.prefill_layer import DSV41PrefillLayer, DSV41PrefillMoE
from models.demos.blackhole.deepseek_v41_flash.tt.prefill_model import DSV41PrefillModel, T
from models.demos.blackhole.deepseek_v41_flash.tt.step_state import DSV41StepState

REF = os.environ.get("DSV41_E2E_REF", "/mnt/tt-data/ssinghal/dsv4-e2e-ref")
OUT = os.environ.get("DSV41_E2E_OUT", "/mnt/tt-data/ssinghal/dsv4-e2e-dev/base")
BATCHES = [int(x) for x in os.environ.get("DSV41_E2E_BATCHES", "0,1,2,3").split(",")]
MODES = os.environ.get("DSV41_E2E_MODES", "cl,tf").split(",")
CL_STEPS = int(os.environ.get("DSV41_CL_STEPS", "200"))
TF_STEPS = int(os.environ.get("DSV41_TF_STEPS", "128"))
KV_BATCHES = [int(x) for x in os.environ.get("DSV41_E2E_KV_BATCHES", "0").split(",") if x != ""]
BOOT = os.environ.get("DSV41_BOOT") == "1"
U = 4
NB = 4 * U


def state_tensors(attn):
    ts = [attn.cache]
    for n in ("prev_cs",):
        if getattr(attn, n, None) is not None:
            ts.append(getattr(attn, n))
    ts += [t for t in getattr(attn, "cs_state", []) if t is not None]
    return ts


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 700_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(21600)
@torch.no_grad()
def test_e2e_gsm8k(mesh_device):
    md = mesh_device
    rows, cols = tuple(md.shape)
    os.makedirs(OUT, exist_ok=True)
    log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
    meta = torch.load(os.path.join(REF, "meta.pt"))
    prompt_all, S = meta["prompt"], meta["S"]
    refres = torch.load(os.path.join(REF, "results.pt")) if "tf" in MODES else None
    _a, _, _b = os.environ.get("DSV41_E2E_LAYERS", "0-39").partition("-")
    layer_ids = list(range(int(_a), int(_b or _a) + 1))  # a short range = smoke test of the harness
    MAX_ROPE = max(256, S + 64)
    chain = DSV41DecodeChain(md, users_per_row=U, max_comp=256, log=log)
    B = chain.B
    sh = _Shards()
    pool = ThreadPoolExecutor(max_workers=2)
    futs = {}
    submit = lambda L: futs.setdefault(L, pool.submit(load_layer, L, True, MAX_ROPE)) if L in layer_ids else None
    for L in layer_ids[:2]:
        submit(L)
    pls, built, groups, first_moe = [], [], {}, None
    t0 = time.time()
    seed_dir = os.path.join(REF, "state")
    for L in layer_ids:
        refL = torch.load(os.path.join(seed_dir, f"layer_{L}.pt"), mmap=True)
        meta_L = {"state": {k: v[:NB] for k, v in refL["state"].items()}, "S": S, "gate_cutoff": refL["gate_cutoff"]}
        submit(L + 1), submit(L + 2)
        w = futs.pop(L).result()
        layer, attn = chain.build_layer(L, meta_L, w)
        del w
        if not BOOT:  # the prefill must write the state
            ttnn.copy(ttnn.zeros_like(attn.cache), attn.cache)
            if getattr(attn, "prev_cs", None) is not None:
                ttnn.copy(ttnn.zeros_like(attn.prev_cs), attn.prev_cs)
        attn.prefill = DSV41PrefillAttention(attn, sh.get(f"layers.{L}.attn.attn_sink").float())
        pmoe = DSV41PrefillMoE(layer.moe, T=T, buffers=None if first_moe is None else first_moe.decode.buffers)
        first_moe = first_moe or pmoe
        pls.append((L, DSV41PrefillLayer(layer, attn.prefill, pmoe, T=T)))
        key = getattr(attn, "ratio", 0)
        if key not in groups:
            groups[key] = DSV41StepState(attn)
        built.append((L, layer, key))
        gc.collect()
    log(f"built {len(layer_ids)} layers in {time.time() - t0:.0f}s")

    engram_ids = [l for l in (1, 14) if l in layer_ids]
    host_rows = HostEngramRows(
        tuple(engram_ids), max_batch_size=B, max_seq_len=max(256, S + max(CL_STEPS, TF_STEPS) + 32)
    )
    if os.environ.get("DSV41_ENGRAM_RAM", "1") == "1":
        t1 = time.time()
        host_rows.load_ram()
        log(f"Engram tables in RAM {time.time() - t1:.0f}s")
    dev_engram = {l: DSV41DeviceEngram(md, l, sh, mesh_config=chain.mesh_config, ccl=chain.ccl) for l in engram_ids}
    embedding = DSV41DeviceEmbedding(md, sh.get("embed.weight"), users_per_row=U)
    head = DSV41DeviceHead(md, sh.get("norm.weight").float(), sh.get("head.weight"), norm_eps=R.model_args().norm_eps)
    model = DSV41PrefillModel(md, pls, embedding, head, dev_engram, host_rows, users_per_row=U)
    dec = DSV41Decoder(md, built, embedding, head, dev_engram, step_states=groups)
    dec.enable_sampling(chain.mesh_config, chain.ccl)
    pos_at = lambda i: torch.full((B,), S + i)
    EOS = 1
    snap = {}  # layer -> list of device clones of the decode state (taken after every prefill)
    state = {"tid": None, "dl": None}

    cur_b = [0]

    def snapshot(b):
        for L, layer, _ in built:
            snap[(b, L)] = [ttnn.clone(t) for t in state_tensors(layer.attention)]

    def restore():
        for L, layer, _ in built:
            for c, t in zip(snap[(cur_b[0], L)], state_tensors(layer.attention)):
                ttnn.copy(c, t)

    def step(tokens, i):
        """one traced decode step at position S + i for the 16 users; returns the (device) logits tensor"""
        hs = host_rows.hashes(tokens[:, None], S + i)
        dec.set_packed_inputs(tokens, host_rows.rows_all(hs, engram_ids), pos_at(i))
        if state["tid"] is None:  # compile pass + trace capture on the very first step (state restored around it)
            dec.forward()
            ttnn.synchronize_device(md)
            restore()
            state["tid"] = ttnn.begin_trace_capture(md, cq_id=0)
            state["dl"] = dec.forward()
            ttnn.end_trace_capture(md, state["tid"], cq_id=0)
            ttnn.synchronize_device(md)
            restore()
            host_rows.hashes(prompt[:, :], 0)  # hash history back to the prompt (this step's hash was consumed above)
            hs = host_rows.hashes(tokens[:, None], S + i)
            dec.set_packed_inputs(tokens, host_rows.rows_all(hs, engram_ids), pos_at(i))
        ttnn.execute_trace(md, state["tid"], cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        return state["dl"]

    def dump_kv(path):
        out = {}
        for L, layer, _ in built:
            t = layer.attention.cache
            out[L] = torch.cat([ttnn.to_torch(ttnn.get_device_tensors(t)[r * cols]) for r in range(rows)])[:B, 0].to(
                torch.bfloat16
            )
        torch.save(out, path)

    # phase 1: device prefill of every batch (eager) BEFORE the decode trace exists; the decode state of each batch is cloned on the device
    results = {}
    for b in BATCHES:
        sl = slice(NB * b, NB * b + NB)
        prompt = prompt_all[sl]
        tag = f"b{b}"
        t1 = time.time()
        if BOOT:
            first_tok = refres["stream"][sl, S] if refres is not None else torch.zeros(B, dtype=torch.long)
            host_rows.hashes(prompt, 0)
            logits0 = None
        else:
            logits0 = model.run(prompt)
            first_tok = logits0.argmax(-1)
            log(f"{tag}: device prefill {time.time() - t1:.1f}s; first tokens {first_tok.tolist()}")
        snapshot(b)
        res = {"S": S, "prompt": prompt, "first_tok": first_tok, "logits0_topk": None}
        if logits0 is not None:
            v, i_ = logits0.float().topk(50, -1)
            res["logits0_topk"] = (v, i_, torch.logsumexp(logits0.float(), -1))
            res["logits0_full"] = logits0.to(torch.bfloat16)
        results[b] = res
        torch.save(res, os.path.join(OUT, f"res_{tag}.pt"))
    # phase 2: decode (trace) from each batch's saved state
    for b in BATCHES:
        sl = slice(NB * b, NB * b + NB)
        prompt = prompt_all[sl]
        tag = f"b{b}"
        cur_b[0] = b
        res, first_tok = results[b], results[b]["first_tok"]
        for mode in MODES:
            restore()
            host_rows.hashes(prompt, 0)  # reset the rolling hash state to "prompt seen"
            if mode == "cl":
                tok, toks_out, done = first_tok.clone(), [first_tok.clone()], torch.zeros(B, dtype=torch.bool)
                t2 = time.time()
                for i in range(CL_STEPS):
                    step(tok, i)
                    tok = head.combine(dec.sampled)[:B].clone()
                    toks_out.append(tok)
                    done |= tok == EOS
                    if bool(done.all()):
                        break
                res["cl_tokens"] = torch.stack(toks_out, 1)  # [B, n+1]: first token (prefill) + one per decode step
                log(
                    f"{tag}: closed loop {len(toks_out) - 1} steps in {time.time() - t2:.1f}s, done users {int(done.sum())}/{B}"
                )
            elif mode == "tf":
                stream = refres["stream"][sl]  # [B, S + n]
                N = min(TF_STEPS, stream.shape[1] - S - 1)
                tv, ti, lse, full = [], [], [], []
                t2 = time.time()
                for i in range(N):
                    dl = step(stream[:, S + i], i)
                    lg = head.gather_logits(dl)[:B].float()
                    v, i_ = lg.topk(50, -1)
                    tv.append(v), ti.append(i_), lse.append(torch.logsumexp(lg, -1))
                    if i < 64:
                        full.append(lg.to(torch.bfloat16))
                res.update(
                    tf_topk_val=torch.stack(tv),
                    tf_topk_idx=torch.stack(ti),
                    tf_lse=torch.stack(lse),
                    tf_full=torch.stack(full),
                )
                log(f"{tag}: teacher forced {N} steps in {time.time() - t2:.1f}s")
                if b in KV_BATCHES:
                    dump_kv(os.path.join(OUT, f"kv_{tag}.pt"))  # device KV after the TF run (positions < S + N written)
                    res["kv_n_written"] = S + N
        torch.save(res, os.path.join(OUT, f"res_{tag}.pt"))
    log("DONE")
