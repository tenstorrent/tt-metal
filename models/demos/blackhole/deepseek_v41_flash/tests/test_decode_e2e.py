# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Token ids -> embedding -> layers (+Engram) -> LM head -> argmax, all on the device, one decode token for 16 users, as ONE
trace. Compares the logits / tokens with the reference chain (final.pt) and reports the traced ms/token.
Env: DSV41_LAYERS (default "0-39"), DSV41_CHAIN (reference chain dir with the prefilled cache state), DSV41_ENGRAM=0 to skip."""

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

CHAIN = os.environ.get("DSV41_CHAIN", "/mnt/tt-data/ssinghal/dsv4-chain-e")
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
def test_decode_e2e(mesh_device):
    md = mesh_device
    a, _, b = LAYERS.partition("-")
    layer_ids = list(range(int(a), int(b or a) + 1))
    toks = torch.load(os.path.join(CHAIN, "tokens.pt"))
    S = toks["prefill_tokens"].shape[1]
    dec_tokens = toks["decode_tokens"].reshape(-1)
    log = lambda m: print(m, flush=True)
    chain = DSV41DecodeChain(md, log=log)
    sh = _Shards()

    pool = ThreadPoolExecutor(max_workers=2)
    futs = {}
    submit = lambda L: futs.setdefault(L, pool.submit(load_layer, L)) if L in layer_ids else None
    for L in layer_ids[:2]:
        submit(L)
    t0 = time.time()
    built = []
    for L in layer_ids:
        ref = torch.load(os.path.join(CHAIN, f"layer_{L}.pt"))
        ref["S"] = S
        submit(L + 1), submit(L + 2)
        w = futs.pop(L).result()
        layer, attn = chain.build_layer(L, ref, w)
        del w
        built.append((L, layer, attn.step_inputs(torch.full((chain.B,), S))))
        if L % 5 == 0:
            log(f"built layer {L} ({time.time() - t0:.0f}s)")

    use_engram = os.environ.get("DSV41_ENGRAM", "1") == "1"
    engram_ids = [l for l in (1, 14) if l in layer_ids] if use_engram else []
    host_rows = HostEngramRows(tuple(engram_ids)) if engram_ids else None
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
    embedding = DSV41DeviceEmbedding(md, sh.get("embed.weight"))
    head = DSV41DeviceHead(md, sh.get("norm.weight").float(), sh.get("head.weight"), norm_eps=R.model_args().norm_eps)
    dec = DSV41Decoder(md, built, embedding, head, dev_engram)

    def host_inputs():
        rows = {}
        if host_rows is not None:
            host_rows.hashes(toks["prefill_tokens"], 0)  # advances the hash state like the reference
            hashes = host_rows.hashes(toks["decode_tokens"], S)
            rows = {l: host_rows.rows(l, hashes) for l in engram_ids}
        return rows

    th = time.perf_counter()
    rows = host_inputs()
    log(f"host Engram hashes+rows: {(time.perf_counter() - th) * 1e3:.1f} ms")
    dec.set_inputs(dec_tokens, rows)

    logits = dec.forward()  # eager: compiles, and gives the result to check
    ttnn.synchronize_device(md)
    got = head.gather_logits(logits)[: chain.B]
    am = head.argmax(logits)[: chain.B]
    if layer_ids[-1] == 39 and layer_ids[0] == 0:
        fin = torch.load(os.path.join(CHAIN, "final.pt"))
        log(
            f"RESULT e2e logits PCC {R.pcc(got, fin['logits']):.5f}; tokens match {int((am == fin['argmax']).sum())}/{chain.B}"
        )
        log(f"device tokens: {am.tolist()}\nref    tokens: {fin['argmax'].tolist()}")

    t = time.perf_counter()
    dec.forward()
    ttnn.synchronize_device(md)
    log(f"eager second pass {(time.perf_counter() - t) * 1e3:.1f} ms")
    if os.environ.get("DSV41_L1_DIAG") == "1":  # the diagnostics synchronise, which a trace capture forbids
        return
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    dec.forward()
    ttnn.end_trace_capture(md, tid, cq_id=0)
    ttnn.synchronize_device(md)
    for _ in range(3):
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(md)
    n = 20
    t = time.perf_counter()
    for _ in range(n):
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(md)
    ms = (time.perf_counter() - t) / n * 1e3
    log(
        f"E2E_TRACED {len(layer_ids)} layers + embed + engram + head: {ms:.2f} ms/token -> {1e3 / ms * len(layer_ids) / 40:.1f} tok/s/user (scaled to 40 layers)"
    )
