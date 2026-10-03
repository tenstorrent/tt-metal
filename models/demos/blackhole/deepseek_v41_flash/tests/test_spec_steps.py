# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""M1: speculative VERIFY step over the full model: blocks of n = 1 + k teacher tokens per user in ONE traced step (paged spec attention), vs the
reference chain logits (dsv4-chain-m: 16 users, S=9, 6 teacher-forced decode steps). Prints per block-index logits PCC / argmax agreement and the
steady-state wall time per round (host rows + uploads + replay + readback).
Env: DSV41_K (default 1), DSV41_LAYERS (default 0-39), DSV41_CHAIN, DSV41_REPLAYS (timing replays, default 10)."""

import os
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference import ref_layer as R
from models.demos.blackhole.deepseek_v41_flash.tt.device_head import DSV41DeviceEmbedding, DSV41DeviceHead
from models.demos.blackhole.deepseek_v41_flash.tt.engram import DSV41DeviceEngram, HostEngramRows
from models.demos.blackhole.deepseek_v41_flash.tt.loader import load_layer
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import _Shards
from models.demos.blackhole.deepseek_v41_flash.tt.spec_chain import SpecChain
from models.demos.blackhole.deepseek_v41_flash.tt.spec_decoder import SpecVerifier
from models.demos.blackhole.deepseek_v41_flash.tt.spec_state import SpecStepState

CHAIN = os.environ.get("DSV41_CHAIN", "/mnt/tt-data/ssinghal/dsv4-chain-m")


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 900_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(14400)
@torch.no_grad()
def test_spec_steps(mesh_device):
    md = mesh_device
    a, _, b = os.environ.get("DSV41_LAYERS", "0-39").partition("-")
    layer_ids = list(range(int(a), int(b or a) + 1))
    k = int(os.environ.get("DSV41_K", "1"))
    n = 1 + k
    U = int(os.environ.get("DSV41_USERS_PER_ROW", "4"))
    toks = torch.load(os.path.join(CHAIN, "tokens.pt"))
    S, dec_tok = toks["prefill_tokens"].shape[1], toks["decode_tokens"]
    log = lambda m: print(m, flush=True)
    chain = SpecChain(md, users_per_row=U, n=n, log=log)
    B, Tn = chain.n_users, chain.n_users * n
    reps = B // 16
    assert B % 16 == 0
    if reps > 1:
        toks = {kk: v.repeat(reps, 1) for kk, v in toks.items()}
        dec_tok = toks["decode_tokens"]
    nblocks = dec_tok.shape[1] // n
    fin = torch.load(os.path.join(CHAIN, "final.pt")) if layer_ids[0] == 0 and layer_ids[-1] == 39 else None
    sh = _Shards()
    pool = ThreadPoolExecutor(max_workers=2)
    futs = {}
    submit = lambda L: futs.setdefault(L, pool.submit(load_layer, L)) if L in layer_ids else None
    for L in layer_ids[:2]:
        submit(L)
    built, groups, t0 = [], {}, time.time()
    for L in layer_ids:
        ref = torch.load(os.path.join(CHAIN, f"layer_{L}.pt"))
        ref["S"] = S
        if reps > 1:
            ref["state"] = {
                kk: (v.repeat(reps, *([1] * (v.dim() - 1))) if torch.is_tensor(v) else v)
                for kk, v in ref["state"].items()
            }
        submit(L + 1), submit(L + 2)
        layer, attn = chain.build_layer(L, ref, futs.pop(L).result())
        key = getattr(attn, "ratio", 0)
        if key not in groups:
            groups[key] = SpecStepState(attn)
        built.append((L, layer, key))
    log(f"built {len(layer_ids)} layers in {time.time() - t0:.0f}s")
    engram_ids = [l for l in (1, 14) if l in layer_ids]
    host_rows = HostEngramRows(tuple(engram_ids), max_batch_size=B) if engram_ids else None
    dev_engram = {l: DSV41DeviceEngram(md, l, sh, mesh_config=chain.mesh_config, ccl=chain.ccl) for l in engram_ids}
    embedding = DSV41DeviceEmbedding(md, sh.get("embed.weight"), users_per_row=chain.T)
    head = DSV41DeviceHead(md, sh.get("norm.weight").float(), sh.get("head.weight"), norm_eps=R.model_args().norm_eps)
    dec = SpecVerifier(md, built, embedding, head, dev_engram, step_states=groups)
    dec.enable_sampling(chain.mesh_config, chain.ccl)
    if host_rows is not None:
        t0 = time.time()
        host_rows.load_ram()
        log(f"Engram tables in RAM {time.time() - t0:.0f}s")
        host_rows.hashes(toks["prefill_tokens"], 0)
    block_tok = lambda i: dec_tok[:, i * n : (i + 1) * n]  # [B, n]
    hashes = [host_rows.hashes(block_tok(i), S + i * n) for i in range(nblocks)] if host_rows is not None else []

    def feed(i):
        rows = host_rows.rows_all(hashes[i], engram_ids) if engram_ids else {}
        rows = {l: r.reshape(Tn, 1, -1) for l, r in rows.items()}  # [B, n, Kin] -> user-major token rows
        pos = (torch.full((B, 1), S + i * n) + torch.arange(n).reshape(1, n)).reshape(-1)
        dec.set_packed_inputs(block_tok(i).reshape(-1), rows, pos)

    snaps = dec.snapshot_states()
    feed(0)
    dec.forward()
    ttnn.synchronize_device(md)
    dec.restore_states(snaps)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    logits = dec.forward()
    ttnn.end_trace_capture(md, tid, cq_id=0)
    ttnn.synchronize_device(md)
    dec.restore_states(snaps)
    log("trace captured")
    wall = []
    for i in range(nblocks):
        t = time.perf_counter()
        feed(i)
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        am = head.combine(dec.sampled)[:Tn].reshape(B, n)
        wall.append((time.perf_counter() - t) * 1e3)
        if fin is not None:
            got = head.gather_logits(logits)[:Tn].reshape(B, n, -1)
            for j in range(n):
                s = i * n + j
                want = fin["logits_steps"][s].repeat(reps, 1)
                log(
                    f"BLOCK {i} idx {j} (step {s}): logits PCC {R.pcc(got[:, j], want):.5f}  tokens match {int((am[:, j] == fin['argmax_steps'][s].repeat(reps)).sum())}/{B}"
                )
        log(f"round {i}: wall {wall[-1]:.1f} ms")
    # steady-state timing: replay the last round
    nrep = int(os.environ.get("DSV41_REPLAYS", "10"))
    ts = []
    for _ in range(nrep):
        t = time.perf_counter()
        feed(nblocks - 1)
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        head.combine(dec.sampled)
        ts.append((time.perf_counter() - t) * 1e3)
    t = time.perf_counter()
    for _ in range(nrep):
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(md)
    dev = (time.perf_counter() - t) / nrep * 1e3
    log(
        f"SPEC_TIMING k={k} n={n} users={B} rows/device={chain.T}: round wall (host rows+upload+replay+readback) mean {sum(ts) / len(ts):.1f} ms, "
        f"min {min(ts):.1f}; pure device replay {dev:.1f} ms"
    )
