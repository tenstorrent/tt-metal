# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PAGED variant (DSV41_PAGED=1 forced here): paged KV pool + per-user page tables in the decoder. Multi-token device decode with ONE captured trace replayed per token: tokens -> embedding -> layers (+Engram) -> head.
Teacher-forced tokens from the reference chain (reference/ref_chain.py --steps N); per step the device logits are compared
with the reference (PCC, argmax agreement) and the wall time of the whole step (host inputs + replay + readback) is timed.
Env: DSV41_CHAIN (default dsv4-chain-m), DSV41_LAYERS (default 0-39), DSV41_STEPS (default all saved)."""

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

os.environ["DSV41_PAGED"] = "1"
CHAIN = os.environ.get("DSV41_CHAIN", "/mnt/tt-data/ssinghal/dsv4-chain-m")
LAYERS = os.environ.get("DSV41_LAYERS", "0-39")


def synthetic_state(ref, ctx, B, with_index):
    """Random filled caches of context ``ctx`` for B users (timing runs): window ring, compressed latents / compressor state (kv sources), index keys."""
    ratio, g = ref["ratio"], torch.Generator().manual_seed(11)
    rn = lambda *shape: torch.randn(*shape, generator=g).to(torch.bfloat16)
    st = {"window": rn(B, 128, 512)}
    if ratio and ref["is_kv_source"]:
        st["comp"] = rn(B, ctx // ratio, 512)
        if ratio > 1:
            st["kv_state"], st["score_state"] = rn(B, ratio, 512), rn(B, ratio, 512)
        if with_index:
            st["index_k"] = (rn(B, ctx // ratio, 128) * 0.7).to(torch.bfloat16)
    return st


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
@pytest.mark.timeout(14400)
@torch.no_grad()
def test_paged_decode_steps(mesh_device):
    md = mesh_device
    a, _, b = LAYERS.partition("-")
    layer_ids = list(range(int(a), int(b or a) + 1))
    toks = torch.load(os.path.join(CHAIN, "tokens.pt"))
    S, dec_tok = toks["prefill_tokens"].shape[1], toks["decode_tokens"]  # [B, N]
    ndec = dec_tok.shape[1]
    SYN = int(
        os.environ.get("DSV41_SYN_CTX", "0")
    )  # >0: SYNTHETIC filled caches of this context length (timing; no reference), positions SYN..
    if SYN:
        S = SYN
    N = (
        int(os.environ["DSV41_STEPS"]) if "DSV41_STEPS" in os.environ else (6 if SYN else ndec)
    )  # steps beyond the dump cycle its tokens (timing only)
    log = lambda m: print(m, flush=True)
    T_users = int(
        os.environ.get("DSV41_USERS_PER_ROW", "4")
    )  # 4 -> batch 16, 8 -> batch 32, 16 -> batch 64 (mHC kernels need <= 16)
    chain = DSV41DecodeChain(md, users_per_row=T_users, log=log)
    B = chain.B
    ref_users = toks["prefill_tokens"].shape[0]
    reps = (
        B // ref_users
    )  # the reference chain has ref_users users (16, or 1 for the real 2048-token dump): user u of the batch runs the state/tokens of reference user u % ref_users
    assert B % ref_users == 0
    if reps > 1:
        toks = {k: v.repeat(reps, 1) for k, v in toks.items()}
        dec_tok = toks["decode_tokens"]
        log(f"batch {B}: tiling the {ref_users}-user reference state x{reps}")
    sh = _Shards()
    full = layer_ids[0] == 0 and layer_ids[-1] == 39
    fin = (
        torch.load(os.path.join(CHAIN, "final.pt"))
        if full and os.path.exists(os.path.join(CHAIN, "final.pt"))
        else None
    )

    pool = ThreadPoolExecutor(max_workers=2)
    futs = {}
    submit = lambda L: (
        futs.setdefault(L, pool.submit(load_layer, L, True, chain.max_ctx + 64, chain.use_indexer))
        if L in layer_ids
        else None
    )
    for L in layer_ids[:2]:
        submit(L)
    built, groups, t0 = [], {}, time.time()
    for L in layer_ids:
        ref = torch.load(os.path.join(CHAIN, f"layer_{L}.pt"))
        ref["S"] = S
        if SYN:
            ref["state"] = synthetic_state(ref, SYN, B, chain.use_indexer)
        elif reps > 1:
            ref["state"] = {
                k: (v.repeat(reps, *([1] * (v.dim() - 1))) if torch.is_tensor(v) else v)
                for k, v in ref["state"].items()
            }
        submit(L + 1), submit(L + 2)
        w = futs.pop(L).result()
        layer, attn = chain.build_layer(L, ref, w)
        del w
        key = getattr(attn, "ratio", 0)  # layers of one kind share the RoPE tables / masks / positions
        if key not in groups:
            groups[key] = chain.step_state(attn)  # tables of every position, built once in device DRAM
        built.append((L, layer, key))
    chain.finalize()
    ttnn.synchronize_device(md)
    mv = ttnn.get_memory_view(md, ttnn.BufferType.DRAM)
    MiB = 1 << 20
    free_bank, pool_rows = mv.total_bytes_free_per_bank, chain.kvpool.total_rows
    row_b = chain.kvpool.row_bytes
    page_bank = 320 * row_b / 8  # one 128-token page (320 rows) per bank
    log(
        f"PAGED_MEM batch {B} (users/row {T_users}) pool {chain.kvpool.num_pages} pages/row ({pool_rows} rows = {pool_rows / 1024:.0f} MiB per chip): "
        f"DRAM per bank allocated {mv.total_bytes_allocated_per_bank / MiB:.0f} MiB free {free_bank / MiB:.0f} MiB (= {free_bank * 8 / (1 << 30):.2f} GiB/chip free after the pool); "
        f"extra pages that still fit {int(free_bank // page_bank)} = {int(free_bank // page_bank) * 128 // T_users} more tokens per user (row bytes {row_b})"
    )
    log(
        f"built {len(layer_ids)} layers in {time.time() - t0:.0f}s; free pages per mesh row {chain.kvpool.free_pages()}"
    )

    engram_ids = [l for l in (1, 14) if l in layer_ids]
    host_rows = HostEngramRows(tuple(engram_ids), max_batch_size=B) if engram_ids and not SYN else None
    dev_engram = {
        l: DSV41DeviceEngram(
            md,
            l,
            sh,
            mesh_config=chain.mesh_config if os.environ.get("DSV41_ENGRAM_TP", "1") == "1" else None,
            ccl=chain.ccl,
        )
        for l in engram_ids
    }
    embedding = DSV41DeviceEmbedding(md, sh.get("embed.weight"), users_per_row=T_users)
    head = DSV41DeviceHead(md, sh.get("norm.weight").float(), sh.get("head.weight"), norm_eps=R.model_args().norm_eps)
    dec = DSV41Decoder(md, built, embedding, head, dev_engram, step_states=groups)
    packed = os.environ.get("DSV41_PACKED", "1") == "1"
    if packed:
        dec.enable_sampling(chain.mesh_config, chain.ccl)
    set_in = (
        (lambda i, rows: dec.set_packed_inputs(dec_tok[:, i % ndec], rows, pos_at(i)))
        if packed
        else (lambda i, rows: dec.set_inputs(dec_tok[:, i % ndec], rows, pos_at(i)))
    )

    hashes = []
    if host_rows is not None and os.environ.get("DSV41_ENGRAM_RAM", "1") == "1":
        t0 = time.time()
        host_rows.load_ram()
        log(f"Engram tables loaded into process memory in {time.time() - t0:.0f}s")
    if host_rows is not None:
        host_rows.hashes(toks["prefill_tokens"], 0)
        hashes = [
            host_rows.hashes(dec_tok[:, (i % ndec) : (i % ndec) + 1], S + i) for i in range(N)
        ]  # sequential: the hash state rolls
    if SYN:  # timing only: random Engram rows
        g = torch.Generator().manual_seed(7)
        syn_rows = {
            l: (torch.randn(B, 1, dev_engram[l].kin, generator=g) * 0.02).to(torch.bfloat16) for l in engram_ids
        }
        rows_at = lambda i: syn_rows
    else:
        rows_at = lambda i: host_rows.rows_all(hashes[i], engram_ids) if engram_ids else {}
    pos_at = lambda i: torch.full((B,), S + i)

    # compile pass at step 0 (restores the step-carried compressor state afterwards), then capture the trace
    snaps = dec.snapshot_states()
    chain.kvpool.ensure(pos_at(0), 128)
    set_in(0, rows_at(0))
    dec.forward()
    ttnn.synchronize_device(md)
    dec.restore_states(snaps)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    logits = dec.forward()
    ttnn.end_trace_capture(md, tid, cq_id=0)
    ttnn.synchronize_device(md)
    dec.restore_states(snaps)
    log("trace captured")

    pccs, match, wall = [], [], []
    for i in range(N):
        t = time.perf_counter()
        chain.kvpool.ensure(pos_at(i), 128)
        rows_i = rows_at(i)
        t_rows = time.perf_counter()
        set_in(i, rows_i)
        t_set = time.perf_counter()
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        t_launch = time.perf_counter()
        ttnn.synchronize_device(md)
        t_dev = time.perf_counter()
        am = (head.combine(dec.sampled) if packed else head.argmax(logits))[:B]
        wall.append((time.perf_counter() - t) * 1e3)
        log(
            f"HOST step {i}: engram rows {1e3 * (t_rows - t):.1f} | set_inputs {1e3 * (t_set - t_rows):.1f} | launch {1e3 * (t_launch - t_set):.1f} "
            f"| device (sync) {1e3 * (t_dev - t_launch):.1f} | argmax readback {1e3 * (time.perf_counter() - t_dev):.1f} ms"
        )
        if fin is not None and "logits_steps" in fin and not SYN and i < ndec:
            got = head.gather_logits(logits)[:B]
            pccs.append(R.pcc(got, fin["logits_steps"][i].repeat(reps, 1)))
            match.append(int((am == fin["argmax_steps"][i].repeat(reps)).sum()))
            log(
                f"STEP {i} pos {S + i}: logits PCC {pccs[-1]:.5f}  tokens match {match[-1]}/{B}  wall {wall[-1]:.1f} ms"
            )
        else:
            log(f"STEP {i} pos {S + i}: wall {wall[-1]:.1f} ms  tokens {am.tolist()}")
    steady = (
        wall[1:] if N > 1 else wall
    )  # step 0 carries one-off first-call costs (sampling / readback ops): report it apart
    log(
        f"DECODE_STEPS {N} steps: step 0 {wall[0]:.1f} ms (first-call costs); steady state (steps 1..{N - 1}) mean wall "
        f"{sum(steady) / len(steady):.1f} ms/token incl. host inputs + readback -> {1e3 / (sum(steady) / len(steady)):.1f} tok/s/user "
        f"(all-steps mean {sum(wall) / N:.1f} ms)"
    )
