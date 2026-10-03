# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CLOSED-LOOP autoregressive decode (no teacher forcing): the token sampled on the device at step i is the input of step i+1, so the
Engram rows of the NEW token (host hash + RAM table gather + upload) are on the critical path. Starts from the prefilled dsv4-chain-m
state, N steps (DSV41_STEPS, default 32). Variants (DSV41_MODES, comma list, run in one process):
  naive : packed upload (ctrl + rows replicated over the 8 mesh columns), explicit sync, 4-device sampled readback
  opt   : rows uploaded column-sharded (1/8 of the bytes, all-gathered inside the trace), no explicit sync, the global argmax token
          all-gathered over the mesh rows in the trace and read from ONE device
  dl    : opt + the token / position fed back on the device (no ctrl upload): only the Engram rows of the new token are uploaded
Prints the token ids of every step (for comparison with the reference argmax)."""

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
def test_decode_closed_loop(mesh_device):
    md = mesh_device
    a, _, b = LAYERS.partition("-")
    layer_ids = list(range(int(a), int(b or a) + 1))
    toks = torch.load(os.path.join(CHAIN, "tokens.pt"))
    S, dec_tok = toks["prefill_tokens"].shape[1], toks["decode_tokens"]  # [B, N]
    N = int(os.environ.get("DSV41_STEPS", "32"))
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
    mc, ccl = chain.mesh_config, chain.ccl
    ids = engram_ids
    dec.enable_sampling(mc, ccl)
    hr = host_rows
    t0 = time.time()
    log("loading Engram tables into process memory ...")
    hr.load_ram()
    log(f"Engram tables loaded into process memory in {time.time() - t0:.0f}s")
    rows_n, cols_n = tuple(md.shape)
    Tu = B // rows_n
    pos_at = lambda i: torch.full((B,), S + i)
    tok0 = dec_tok[:, 0].clone()
    snaps = dec.snapshot_states()
    orig_sample, orig_sg = head.sample, head.sample_global
    pc = time.perf_counter

    def gsample(logits):
        """global argmax (as sample_global) -> (uint32 RM [T,1], fp32 [1,1,1,rows*32] gathered over the mesh rows: every device holds all users)."""
        cols = head.cols
        shard = 129280 // cols
        mx = ttnn.max(logits, dim=-1, keepdim=True)
        idx = ttnn.argmax(ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT), dim=-1, keepdim=True)
        idxf = ttnn.typecast(ttnn.to_layout(idx, ttnn.TILE_LAYOUT), ttnn.float32)
        mx_all = mc.allgather(mx, ccl, axis=1, dim=3)
        ix_all = mc.allgather(idxf, ccl, axis=1, dim=3)
        best = ttnn.argmax(ttnn.to_layout(mx_all, ttnn.ROW_MAJOR_LAYOUT), dim=-1, keepdim=True)
        bestf = ttnn.typecast(ttnn.to_layout(best, ttnn.TILE_LAYOUT), ttnn.float32)
        if getattr(head, "_col_ids", None) is None:
            head._col_ids = ttnn.from_torch(
                torch.arange(cols, dtype=torch.float32).reshape(1, 1, 1, cols),
                device=md,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(md),
            )
        sel = ttnn.sum(ttnn.multiply(ttnn.eq(bestf, head._col_ids), ix_all), dim=-1, keepdim=True)
        tok = ttnn.add(ttnn.multiply(bestf, float(shard)), sel)  # [1,1,T,1] fp32
        T = tok.shape[2]
        nxt = ttnn.reshape(ttnn.to_layout(ttnn.typecast(tok, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT), [T, 1])
        g = ttnn.transpose(tok, 2, 3)  # [1,1,1,T]
        if T < 32:
            g = ttnn.pad(g, [(0, 0), (0, 0), (0, 0), (0, 32 - T)], 0.0)
        g = mc.allgather(g, ccl, axis=0, dim=3)  # [1,1,1,rows*32]
        return nxt, g

    read1 = (
        lambda g: ttnn.to_torch(ttnn.get_device_tensors(g)[0])
        .float()
        .reshape(rows_n, 32)[:, :Tu]
        .reshape(-1)
        .round()
        .long()
    )

    rpool = ThreadPoolExecutor(max_workers=max(1, len(ids)))

    def rows_par(h):
        """Engram rows of all layers, one worker thread per layer (the gather / dequant ops release the GIL)."""
        fs = {l: rpool.submit(hr.rows, l, h) for l in ids}
        return {l: f.result() for l, f in fs.items()}

    def run(full_mode):
        mode, shard_rows = (
            (full_mode[:-1], True) if full_mode.endswith("S") else (full_mode, False)
        )  # suffix S: column-sharded rows upload
        dec.ctrl = dec.rows_cat = None
        dec.device_loop = False
        dec.rows_shard = shard_rows
        get_rows = (lambda h: hr.rows_all(h, ids)) if mode == "naive" else rows_par
        head.sample, head.sample_global = orig_sample, orig_sg
        if mode == "opt":

            def s_opt(logits, mc_, ccl_):
                head._gtok = gsample(logits)[1]
                return head._gtok

            head.sample = s_opt
        elif mode == "dl":

            def s_dl(logits, mc_, ccl_):
                nxt, head._gtok = gsample(logits)
                return nxt

            head.sample_global = s_dl
        hr.hashes(toks["prefill_tokens"], 0)  # reset the rolling hash state
        rows = get_rows(hr.hashes(tok0[:, None], S))
        prep = dec.prepare_packed_inputs(tok0, rows, pos_at(0), with_ctrl=mode != "dl")
        dev_mp = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(rows_n, cols_n))
        if mode == "dl":
            dec.enable_device_loop(mc, ccl, tok0, pos_at(0))
            dec.upload_rows_only(prep)
            host_tp = (
                ttnn.from_torch(
                    tok0.reshape(-1, 1).to(torch.int32),
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=dev_mp,
                ),
                ttnn.from_torch(
                    pos_at(0).reshape(-1).to(torch.int32),
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=dev_mp,
                ),
            )
            reset_dl = lambda: (
                ttnn.copy_host_to_device_tensor(host_tp[0], dec.tok_dev),
                ttnn.copy_host_to_device_tensor(host_tp[1], dec.pos_dev),
            )
        else:
            dec.upload_packed_inputs(prep)
            reset_dl = lambda: None
        dec.restore_states(snaps)
        dec.forward()  # compile pass
        ttnn.synchronize_device(md)
        dec.restore_states(snaps), reset_dl()
        tid = ttnn.begin_trace_capture(md, cq_id=0)
        logits = dec.forward()
        ttnn.end_trace_capture(md, tid, cq_id=0)
        ttnn.synchronize_device(md)
        dec.restore_states(snaps), reset_dl()
        readsrc = {"naive": lambda: dec.sampled, "opt": lambda: head._gtok, "dl": lambda: head._gtok}[mode]()
        log(f"[{full_mode}] trace captured")
        hr.hashes(toks["prefill_tokens"], 0)
        tok = tok0
        rows = get_rows(hr.hashes(tok[:, None], S))
        out, tms = [], []
        for i in range(N):
            ta = pc()
            prep = dec.prepare_packed_inputs(tok, rows, pos_at(i), with_ctrl=mode != "dl")
            tb = pc()
            (dec.upload_rows_only if mode == "dl" else dec.upload_packed_inputs)(prep)
            tc = pc()
            ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
            td = pc()
            if mode == "naive":
                ttnn.synchronize_device(md)
                te = pc()
                tok = head.combine(readsrc)[:B]
            else:
                te = pc()
                tok = read1(readsrc)[:B]
            tf = pc()
            out.append(tok.clone())
            if i + 1 < N:
                h = hr.hashes(tok[:, None], S + i + 1)
                tg = pc()
                rows = get_rows(h)
            else:
                tg = th = pc()
            th = pc()
            tms.append(
                [1e3 * (x - y) for x, y in ((tb, ta), (tc, tb), (td, tc), (tf, td), (tg, tf), (th, tg), (th, ta))]
            )
            log(
                f"[{full_mode}] step {i}: prep {tms[-1][0]:.2f} | upload {tms[-1][1]:.2f} | launch {tms[-1][2]:.2f} | device+readback {tms[-1][3]:.2f} | hash {tms[-1][4]:.2f} | rows {tms[-1][5]:.2f} | LOOP {tms[-1][6]:.2f} ms"
            )
        ttnn.release_trace(md, tid)
        st = torch.tensor(
            tms[1 : N - 1]
        )  # steps 1..N-2 (step 0 has first-call costs, the last has no next-token lookup)
        m = st.mean(0).tolist()
        log(
            f"CLOSED_LOOP[{full_mode}] B={B} steps={N}: steady {m[6]:.1f} ms/token -> {1e3 / m[6]:.2f} tok/s/user ({1e3 * B / m[6]:.0f} tok/s total) | "
            f"prep {m[0]:.2f} upload {m[1]:.2f} launch {m[2]:.2f} device+readback {m[3]:.2f} hash {m[4]:.2f} rows {m[5]:.2f} | step0 {tms[0][6]:.1f} ms"
        )
        T = torch.stack(out)  # [N, B]
        log(f"TOKENS[{full_mode}] step0 {T[0].tolist()}")
        if fin is not None:
            log(
                f"[{full_mode}] step0 match vs reference argmax_steps[0]: {int((T[0] == fin['argmax_steps'][0].repeat(reps)).sum())}/{B}"
            )
        nm = min(N - 1, dec_tok.shape[1] - 1)
        log(
            f"[{full_mode}] match vs reference teacher tokens (step i+1): "
            + " ".join(f"{int((T[i] == dec_tok[:, i + 1]).sum())}" for i in range(nm))
            + f" of {B}"
        )
        for i in range(N):
            log(f"TOKENS[{full_mode}] {i}: {T[i].tolist()}")
        return T

    modes = os.environ.get("DSV41_MODES", "naive,opt,dl,optS,dlS").split(",")
    res = {}
    for md_name in modes:
        res[md_name] = run(md_name)
    ref = res[modes[0]]
    for k, v in res.items():
        log(f"TOKEN_AGREE {k} vs {modes[0]}: {int((v == ref).sum())}/{v.numel()}")
