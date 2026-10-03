# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Multi-token device decode with ONE captured trace replayed per token: tokens -> embedding -> layers (+Engram) -> head.
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
from models.demos.blackhole.deepseek_v41_flash.tt.step_state import DSV41StepState

CHAIN = os.environ.get("DSV41_CHAIN", "/mnt/tt-data/ssinghal/dsv4-chain-m")
LAYERS = os.environ.get("DSV41_LAYERS", "0-39")


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
def test_decode_steps(mesh_device):
    md = mesh_device
    a, _, b = LAYERS.partition("-")
    layer_ids = list(range(int(a), int(b or a) + 1))
    toks = torch.load(os.path.join(CHAIN, "tokens.pt"))
    S, dec_tok = toks["prefill_tokens"].shape[1], toks["decode_tokens"]  # [B, N]
    N = min(dec_tok.shape[1], int(os.environ.get("DSV41_STEPS", "99")))
    log = lambda m: print(m, flush=True)
    T_users = int(
        os.environ.get("DSV41_USERS_PER_ROW", "4")
    )  # 4 -> batch 16, 8 -> batch 32, 16 -> batch 64 (mHC kernels need <= 16)
    chain = DSV41DecodeChain(md, users_per_row=T_users, log=log)
    B = chain.B
    reps = (
        B // 16
    )  # the reference chain has 16 users: user u of a bigger batch runs the state/tokens of reference user u % 16
    assert B % 16 == 0
    if reps > 1:
        toks = {k: v.repeat(reps, 1) for k, v in toks.items()}
        dec_tok = toks["decode_tokens"]
        log(f"batch {B}: tiling the 16-user reference state x{reps}")
    sh = _Shards()
    full = layer_ids[0] == 0 and layer_ids[-1] == 39
    fin = (
        torch.load(os.path.join(CHAIN, "final.pt"))
        if full and os.path.exists(os.path.join(CHAIN, "final.pt"))
        else None
    )

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
                k: (v.repeat(reps, *([1] * (v.dim() - 1))) if torch.is_tensor(v) else v)
                for k, v in ref["state"].items()
            }
        submit(L + 1), submit(L + 2)
        w = futs.pop(L).result()
        layer, attn = chain.build_layer(L, ref, w)
        del w
        key = getattr(attn, "ratio", 0)  # layers of one kind share the RoPE tables / masks / positions
        if key not in groups:
            groups[key] = DSV41StepState(attn)  # tables of every position, built once in device DRAM
        built.append((L, layer, key))
    log(f"built {len(layer_ids)} layers in {time.time() - t0:.0f}s")

    engram_ids = [l for l in (1, 14) if l in layer_ids]
    host_rows = HostEngramRows(tuple(engram_ids), max_batch_size=B) if engram_ids else None
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
        (lambda i, rows: dec.set_packed_inputs(dec_tok[:, i], rows, pos_at(i)))
        if packed
        else (lambda i, rows: dec.set_inputs(dec_tok[:, i], rows, pos_at(i)))
    )

    hashes = []
    if host_rows is not None and os.environ.get("DSV41_ENGRAM_RAM", "1") == "1":
        t0 = time.time()
        host_rows.load_ram()
        log(f"Engram tables loaded into process memory in {time.time() - t0:.0f}s")
    if host_rows is not None:
        host_rows.hashes(toks["prefill_tokens"], 0)
        hashes = [host_rows.hashes(dec_tok[:, i : i + 1], S + i) for i in range(N)]  # sequential: the hash state rolls
    rows_at = lambda i: host_rows.rows_all(hashes[i], engram_ids) if engram_ids else {}
    pos_at = lambda i: torch.full((B,), S + i)

    # compile pass at step 0 (restores the step-carried compressor state afterwards), then capture the trace
    snaps = dec.snapshot_states()
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
        if fin is not None and "logits_steps" in fin:
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
