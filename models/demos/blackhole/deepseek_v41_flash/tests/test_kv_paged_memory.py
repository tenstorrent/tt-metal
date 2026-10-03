# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Measure the real free DRAM per chip with the full model resident (all 40 layers + embedding + head + Engram weights,
one captured decode trace). Prints DRAMMEM lines (per bank, MiB; BH has 8 DRAM banks per chip).
Env: DSV41_LAYERS (default 0-39), DSV41_USERS_PER_ROW (default 4), DSV41_CHAIN."""

import os
import time

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
MiB = 1 << 20


def mem(md, tag):
    ttnn.synchronize_device(md)
    mv = ttnn.get_memory_view(md, ttnn.BufferType.DRAM)
    print(
        f"DRAMMEM {tag:38s} total {mv.total_bytes_per_bank / MiB:8.1f}  allocated {mv.total_bytes_allocated_per_bank / MiB:8.1f}  "
        f"free {mv.total_bytes_free_per_bank / MiB:8.1f}  largest_free_block {mv.largest_contiguous_bytes_free_per_bank / MiB:8.1f}  (MiB/bank)",
        flush=True,
    )


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
def test_kv_paged_memory(mesh_device):
    md = mesh_device
    a, _, b = LAYERS.partition("-")
    layer_ids = list(range(int(a), int(b or a) + 1))
    toks = torch.load(os.path.join(CHAIN, "tokens.pt"))
    S, dec_tok = toks["prefill_tokens"].shape[1], toks["decode_tokens"]
    log = lambda m: print(m, flush=True)
    T_users = int(os.environ.get("DSV41_USERS_PER_ROW", "4"))
    mem(md, "device open (trace region carved)")
    chain = DSV41DecodeChain(md, users_per_row=T_users, log=log)
    B = chain.B
    reps = B // 16
    if reps > 1:
        toks = {k: v.repeat(reps, 1) for k, v in toks.items()}
        dec_tok = toks["decode_tokens"]
    sh = _Shards()
    built, groups, t0 = [], {}, time.time()
    for L in layer_ids:
        ref = torch.load(os.path.join(CHAIN, f"layer_{L}.pt"))
        ref["S"] = S
        if reps > 1:
            ref["state"] = {
                k: (v.repeat(reps, *([1] * (v.dim() - 1))) if torch.is_tensor(v) else v)
                for k, v in ref["state"].items()
            }
        w = load_layer(L)
        layer, attn = chain.build_layer(L, ref, w)
        del w
        key = getattr(attn, "ratio", 0)
        if key not in groups:
            groups[key] = DSV41StepState(attn)
        built.append((L, layer, key))
        if L in (0, 1, 2, 3, 19, 20) or L % 10 == 9:
            mem(md, f"after layer {L}")
    log(f"built {len(layer_ids)} layers in {time.time() - t0:.0f}s")
    mem(md, "all layers (incl step states)")
    engram_ids = [l for l in (1, 14) if l in layer_ids]
    host_rows = HostEngramRows(tuple(engram_ids), max_batch_size=B) if engram_ids else None
    dev_engram = {l: DSV41DeviceEngram(md, l, sh, mesh_config=chain.mesh_config, ccl=chain.ccl) for l in engram_ids}
    mem(md, "after device Engram weights")
    embedding = DSV41DeviceEmbedding(md, sh.get("embed.weight"), users_per_row=T_users)
    mem(md, "after embedding")
    head = DSV41DeviceHead(md, sh.get("norm.weight").float(), sh.get("head.weight"), norm_eps=R.model_args().norm_eps)
    mem(md, "after head")
    dec = DSV41Decoder(md, built, embedding, head, dev_engram, step_states=groups)
    dec.enable_sampling(chain.mesh_config, chain.ccl)
    pos = torch.full((B,), S)
    if host_rows is not None:
        host_rows.hashes(toks["prefill_tokens"], 0)
        h0 = host_rows.hashes(dec_tok[:, 0:1], S)
        rows0 = host_rows.rows_all(h0, engram_ids)
    else:
        rows0 = {}
    dec.set_packed_inputs(dec_tok[:, 0], rows0, pos)
    mem(md, "after step inputs")
    snaps = dec.snapshot_states()
    dec.forward()
    mem(md, "after eager step (compile pass)")
    dec.restore_states(snaps)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    dec.forward()
    ttnn.end_trace_capture(md, tid, cq_id=0)
    mem(md, "after trace capture")
    ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
    mem(md, "after trace replay")
    log("DRAMMEM done")
