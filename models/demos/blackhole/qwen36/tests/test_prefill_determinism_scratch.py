# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""In-process repro of the TP=1-mode prefill nondeterminism at T = 63/64 (Phase 2 block-edge drill).

RESOLVED (p1b): the prefill itself is deterministic; the nondeterminism needs the page-table row vLLM hands the
model, whose STALE tail entry (BlockTable.add_row never clears past the new request's blocks) usually ALIASES the
request's own block ([2, 2, 0, ...] for a 1-block prompt). The traced masked bucket's fixed-width fill table then
named that block twice (masked_bucket_trace.fill_pt_row trusted any non-zero tail entry), so ONE paged_fill_cache
wrote the bucket's pad rows over the real K/V rows -- a multi-core write race (test_paged_fill_alias_scratch.py).
P1B_STALE=1 reproduces that row shape; QWEN36_PREFILL_TRUST_PT_TAIL=1 restores the old rule (6/6 distinct logits at
T=1,2,63,64; fixed: identical). Lengths whose real blocks fill the bucket's fill width (65..128) were never affected.

ONE process, ONE (1,1) mesh, the real 27B loaded once (TP=1 mode, bf8 paged KV). For each prompt length T the SAME
T-token prompt is prefilled N times through the path vLLM serving uses and the first-step logits, the GDN
recurrent/conv state and the request's K/V blocks are compared bit-for-bit across repeats.

Env knobs:
  P1B_B        max_batch_size: 8 -> prefill_paged_slots (decode-node shape, max_num_seqs=8); 1 -> prefill_traced_chunked
               on the B=1 buffers (prefill-node shape, max_num_seqs=1). Default 8.
  P1B_LENS     comma list of prompt lengths (default 1,2,31,32,33,62,63,64,65,66,127,128,129)
  P1B_REPEATS  repeats per length (default 8)
  P1B_CHURN    1 -> between repeats run a different prefill (random 300 tokens into slot 1) + 3 decode steps (memory churn,
               like the decode steps / exports vLLM runs between requests)
  P1B_PT_ZEROPAD 1 -> vLLM-style page-table row: only ceil(T/64) real blocks, the rest 0 (default 1)
  P1B_EXPORT   1 -> after every prefill run the KV-transfer export (P-node shape: prefill -> export_request_state)
  P1B_HEAPCHURN 1 -> background thread allocating/freeing small torch tensors during the run (host heap churn)
  P1B_PT_WIDTH page-table row width (default 1024 = vLLM max_num_blocks_per_req at 64k ctx)
  P1B_STALE    1 -> vLLM-style STALE page-table row: entries past the request's blocks hold another (finished)
               request's block ids instead of zeros (vLLM's BlockTable.add_row never clears them)
  P1B_VARY_BLOCKS 1 -> rotate the request's blocks across repeats (vLLM's free-block queue hands out different blocks)
  QWEN36_PREFILL_BUCKET_TRACE  1 (serving default) -> traced masked bucket; 0 -> eager masked bucket

  P1B_RAWIDS   token-id fixture (default: p1b_rawids.json next to this file)
  P1B_OUT      JSON result path (default: ./p1b_determinism_B<B>.json)

Run (one die, TP=1 mode):
  export TT_VISIBLE_DEVICES=<chip> MESH_DEVICE=P150 QWEN36_FORCE_TP_PATH=1 QWEN36_SKIP_VISION=1 QWEN_SDPA_BF8=1 \
      QWEN36_PREFILL_BUCKET_TRACE=1 QWEN36_GDN_DECODE_FUSED=0 HF_MODEL=<Qwen3.6/3.8-27B weights>
  P1B_STALE=1 pytest -svq models/demos/blackhole/qwen36/tests/test_prefill_determinism_scratch.py
"""
import json
import os
import threading
import time

import numpy as np

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

BLOCK = 64
BPU = 80  # blocks per slot region (4095 tokens + headroom); 8 x 80 = 640 (multiple of 32)
RAWIDS = os.environ.get("P1B_RAWIDS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "p1b_rawids.json"))


def _prompt(T):
    d = json.load(open(RAWIDS))
    ids65, ids128 = d["rawids_65"], d["rawids_128"]
    base = ids65 + ids128[65:]  # 128 tokens; [:64] and [:65] and [:66] are exactly the Phase 2 T64/T65/T66 prompts
    if T <= 128:
        return base[:T]
    if T == 129:
        return ids128 + [ids65[3]]
    if T <= 2048:
        return d["rawids_2048"][:T]
    if T == 2049:
        return d["rawids_2049"]
    return (d["rawids_2049"] * ((T // 2049) + 1))[:T]


def _region(u):
    return list(range(u * BPU, (u + 1) * BPU))


def _gdn_layers(model):
    return [l.attention for l in model.layers if not l.is_full_attention]


def _attn_layers(model):
    return [l.attention for l in model.layers if l.is_full_attention]


def _beq(a, b):
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def _state_rows(model, slot):
    out = []
    for dn in _gdn_layers(model):
        rec = ttnn.to_torch(dn.rec_state)[slot : slot + 1].clone()
        taps = torch.stack([ttnn.to_torch(dn.conv_states[m])[0, slot].clone() for m in range(dn.K)])
        out.append((rec, taps))
    return out


def _kv_blocks(model, blocks):
    out = []
    for li, at in enumerate(_attn_layers(model)):
        for name, t in (("k", at.paged_k), ("v", at.paged_v)):
            nkv, bs, hd = int(t.shape[1]), int(t.shape[2]), int(t.shape[3])
            for b in blocks:
                s = ttnn.slice(t, (b, 0, 0, 0), (b + 1, nkv, bs, hd))
                out.append(ttnn.to_torch(s).clone())
                ttnn.deallocate(s)
    return out


def _decode_step(model, vocab, width, active):
    tokens = torch.zeros(width, 1, dtype=torch.int32)
    pos = torch.full((width,), -1, dtype=torch.int32)
    pt = torch.full((width, BPU), BPU * 8, dtype=torch.int32)
    for slot, (tok, p, blocks) in active.items():
        tokens[slot, 0] = int(tok)
        pos[slot] = int(p)
        pt[slot] = torch.tensor(blocks, dtype=torch.int32)
    dev = model.prepare_inputs_decode(tokens, pos, pt)
    out, _ = model.ttnn_decode_forward(dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3])
    logits = model.process_output_decode(out, width)[:, 0, :vocab].float().clone()
    ttnn.deallocate(out)
    return logits


@torch.no_grad()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "trace_region_size": 536870912}], indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param((1, 1), id="1x1")], indirect=True)
def test_prefill_determinism(mesh_device, reset_seeds, ensure_gc):
    B = int(os.environ.get("P1B_B", "8"))
    lens = [int(x) for x in os.environ.get("P1B_LENS", "1,2,31,32,33,62,63,64,65,66,127,128,129").split(",")]
    reps = int(os.environ.get("P1B_REPEATS", "8"))
    out_path = os.environ.get("P1B_OUT", os.path.join(os.getcwd(), f"p1b_determinism_B{B}.json"))
    num_blocks = 8 * BPU
    t0 = time.perf_counter()
    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=B, max_seq_len=8192)
    assert model.use_tp
    args = model.args
    vocab = args.vocab_size
    model.allocate_kv_caches((num_blocks + 1, args.n_local_kv_heads, BLOCK, args.head_dim), ttnn.bfloat8_b, batch_size=B)
    logger.info(f"[p1b] model loaded in {time.perf_counter()-t0:.0f}s; B={B} bucket_trace={model._mb_trace_buckets}")

    batched = B > 1
    if batched:
        for w in (1, 2, 4, 8):
            _decode_step(model, vocab, w, {0: (1, 0, _region(0))})
        for dn in _gdn_layers(model):
            dn.reset_state_inplace()
            dn._hist_packed_valid = False
        model.sync_gdn_decode_state()
        ttnn.synchronize_device(mesh_device)
    # prefill warmup exactly as qwen36_vllm.warmup_model_prefill
    t0 = time.perf_counter()
    warmup_pt = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)
    prev = model._bind_gdn_prefill_scratch() if batched else None
    try:
        model.capture_prefill_trace_chunked(mesh_device, warmup_pt, chunk_size=2048, capture_chunk_trace=True)
    finally:
        if prev is not None:
            model._unbind_gdn_prefill_scratch(prev)
    if batched:
        model.warmup_gdn_slot_write()
    ttnn.synchronize_device(mesh_device)
    logger.info(f"[p1b] prefill warmup {time.perf_counter()-t0:.0f}s; programs={mesh_device.num_program_cache_entries()}")
    n_pc = mesh_device.num_program_cache_entries()

    churn = os.environ.get("P1B_CHURN", "0") == "1"
    do_export = os.environ.get("P1B_EXPORT", "0") == "1"
    heap = os.environ.get("P1B_HEAPCHURN", "0") == "1"
    pt_width = int(os.environ.get("P1B_PT_WIDTH", "1024"))
    stale = os.environ.get("P1B_STALE", "0") == "1"
    hook = None
    if do_export:
        from models.demos.blackhole.qwen36.tests.test_kv_transfer_hook import make_sinks
        from models.demos.blackhole.qwen36.tt.kv_transfer import Qwen36KVTransfer
        hook = Qwen36KVTransfer(model)
        hook.warmup_kv_transfer(role="kv_producer", mode="dumpfile", chunk_tokens=2048, slots=range(B))
    stop = threading.Event()
    def _heap():
        import random
        keep = []
        while not stop.is_set():
            keep.append(torch.zeros(random.choice([1, 2, 4, 8, 16, 33, 128, 131, 512]), dtype=torch.int32))
            keep.append(np.zeros(random.choice([2, 4, 8, 64, 1024]), dtype=np.int64))
            if len(keep) > 64:
                keep = keep[32:]
    th = threading.Thread(target=_heap, daemon=True) if heap else None
    if th:
        th.start()
    zeropad = os.environ.get("P1B_PT_ZEROPAD", "1") == "1"
    vary = os.environ.get("P1B_VARY_BLOCKS", "0") == "1"
    torch.manual_seed(4321)
    churn_prompt = torch.randint(0, vocab, (1, 300), dtype=torch.long)
    slot = 0
    results = {}
    def _pt_row(blocks, nblk):
        row = torch.zeros(1, max(pt_width, BPU), dtype=torch.int32)
        if stale:
            # vLLM shape seen in the chip-1 server log: the row's stale tail entry ALIASES the request's own block
            # (previous request [1,2] freed, new request gets block 2 -> row [2, 2, 0, ...]).
            row[0, :nblk] = torch.tensor(blocks[:nblk], dtype=torch.int32)
            row[0, nblk] = int(blocks[nblk - 1])
        elif zeropad:
            row[0, :nblk] = torch.tensor(blocks[:nblk], dtype=torch.int32)
        else:
            row[0, :BPU] = torch.tensor(blocks, dtype=torch.int32)
        return row
    for T in lens:
        ids = _prompt(T)
        assert len(ids) == T
        toks = torch.tensor([ids], dtype=torch.long)
        nblk = -(-T // BLOCK)
        logits_all, states_all, kv_all = [], [], []
        for r in range(reps):
            # stale mode needs NON-ZERO block ids (a zero tail entry is legitimately treated as padding)
            blocks = _region(1 + (r % 7)) if vary else (_region(1) if stale else _region(0))
            if churn:
                if batched:
                    model.prefill_paged_slots([churn_prompt], torch.tensor([_region(7)], dtype=torch.int32), [1], valid_lens=[300])
                    for step in range(3):
                        _decode_step(model, vocab, 2, {1: (7, 300 + step, _region(7))})
                else:
                    lgt = model.prefill_traced_chunked(churn_prompt, torch.tensor([_region(7)], dtype=torch.int32), actual_len=300)
                    ttnn.deallocate(lgt)
            if batched:
                pt = _pt_row(blocks, nblk)
                hl = model.prefill_paged_slots([toks], pt, [slot], valid_lens=[T])
                lg = hl[0].reshape(-1)[:vocab].float().clone()
                st = _state_rows(model, slot)
            else:
                pt = _pt_row(blocks, nblk)
                lgt = model.prefill_traced_chunked(toks, pt, actual_len=T)
                ttnn.synchronize_device(mesh_device)
                lg = ttnn.to_torch(lgt).reshape(-1, vocab)[0].float().clone()
                ttnn.deallocate(lgt)
                st = _state_rows(model, 0)
            kv = _kv_blocks(model, blocks[:nblk])
            if do_export:
                manifest = hook.describe_request_state(T, blocks[:nblk])
                sinks = make_sinks(manifest)
                hook.export_request_state(blocks[:nblk], T, slot, sinks)
            logits_all.append(lg)
            states_all.append(st)
            kv_all.append(kv)
        ref = logits_all[0]
        distinct = []
        for lg in logits_all:
            if not any(torch.equal(lg, d) for d in distinct):
                distinct.append(lg)
        maxdiff = max((lg - ref).abs().max().item() for lg in logits_all)
        argmax = sorted({int(torch.argmax(lg)) for lg in logits_all})
        top2 = []
        for lg in logits_all:
            v, i = torch.topk(lg, 2)
            top2.append([int(i[0]), int(i[1]), round(float(v[0] - v[1]), 4)])
        bad_layers = set()
        for st in states_all[1:]:
            for j, ((ra, ta), (rb, tb)) in enumerate(zip(states_all[0], st)):
                if not _beq(ra, rb):
                    bad_layers.add(f"L{j}.rec")
                if not _beq(ta, tb):
                    bad_layers.add(f"L{j}.taps")
        kv_bad = 0
        for kv in kv_all[1:]:
            kv_bad += sum(not _beq(a, b) for a, b in zip(kv_all[0], kv))
        rec = {
            "T": T,
            "repeats": reps,
            "distinct_logits": len(distinct),
            "max_logit_diff": maxdiff,
            "argmax_set": argmax,
            "top2_per_run": top2,
            "gdn_state_mismatch_layers": sorted(bad_layers),
            "kv_block_mismatches": kv_bad,
            "new_programs": mesh_device.num_program_cache_entries() - n_pc,
        }
        results[str(T)] = rec
        logger.info(
            f"[p1b] T={T:5d} distinct={len(distinct)}/{reps} maxdiff={maxdiff:.4g} argmax={argmax} "
            f"gdn_bad={sorted(bad_layers)[:6]} kv_bad={kv_bad} new_programs={rec['new_programs']}"
        )
        json.dump({"B": B, "bucket_trace": os.environ.get("QWEN36_PREFILL_BUCKET_TRACE"), "churn": churn, "zeropad": zeropad,
                   "export": do_export, "heapchurn": heap, "pt_width": pt_width, "stale": stale,
                   "vary_blocks": vary, "results": results},
                  open(out_path, "w"), indent=1)
    stop.set()
    nondet = [T for T, r in results.items() if r["distinct_logits"] > 1]
    logger.info(f"[p1b] NONDETERMINISTIC lengths: {nondet}")
    assert not nondet, f"nondeterministic prefill at T={nondet}"
