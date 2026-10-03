# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-layer accuracy study over several decode steps (layer-major, step-minor; the device carries its own KV / compressor state across steps).

Chained mode (DSV41_TEACHER=0): layer L step s is fed the DEVICE output of layer L-1 at step s (Engram applied on the host at layers 1/14).
Teacher mode (DSV41_TEACHER=1): every layer step is fed the REFERENCE input (dec_in_steps; already contains the Engram write).
Saves DSV41_OUT (.pt): {"x": {(L, s): [B,4,D] bf16}, "pre": {(L,s): [B,4]}, "idx": {(L,s): [B,6]}, "wt": ...} for offline analysis
(PCC vs bf16 / fp32 reference, error excluding massive channels, router flips, per-user spread, logits via HostHead).
Env: DSV41_CHAIN (default dsv4-chain-m), DSV41_NSTEPS (6), DSV41_LAYERS (0-39), DSV41_OUT.
"""
import os
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference import ref_layer as R
from models.demos.blackhole.deepseek_v41_flash.tt.loader import load_layer
from models.demos.blackhole.deepseek_v41_flash.tt.model import DSV41DecodeChain

CHAIN = os.environ.get("DSV41_CHAIN", "/mnt/tt-data/ssinghal/dsv4-chain-m")
LAYERS = os.environ.get("DSV41_LAYERS", "0-39")
NSTEPS = int(os.environ.get("DSV41_NSTEPS", "6"))
TEACHER = os.environ.get("DSV41_TEACHER") == "1"
OUT = os.environ.get("DSV41_OUT", "/mnt/tt-data/ssinghal/dsv4-logs/h45a_chain_steps.pt")


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [pytest.param({"l1_small_size": 16384, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}, id="ring")],
    indirect=True,
)
@pytest.mark.timeout(14400)
@torch.no_grad()
def test_chain_steps(mesh_device):
    a, _, b = LAYERS.partition("-")
    layers = list(range(int(a), int(b or a) + 1))
    toks = torch.load(os.path.join(CHAIN, "tokens.pt"))
    S = toks["prefill_tokens"].shape[1]
    log = lambda m: print(m, flush=True)
    chain = DSV41DecodeChain(mesh_device, max_comp=int(os.environ.get("DSV41_MAX_COMP", "128")), log=log)
    B = chain.B
    engram, hashes = None, []
    if not TEACHER:
        from models.demos.blackhole.deepseek_v41_flash.tt.host_model import HostEngram

        engram = HostEngram()
        engram.hashes(toks["prefill_tokens"], 0)
        hashes = [engram.hashes(toks["decode_tokens"][:, i : i + 1], S + i) for i in range(NSTEPS)]
    pool = ThreadPoolExecutor(max_workers=2)
    futs = {}
    submit = lambda L: futs.setdefault(L, pool.submit(load_layer, L)) if L in layers else None
    for L in layers[:2]:
        submit(L)
    first = torch.load(os.path.join(CHAIN, f"layer_{layers[0]}.pt"))
    xs = [s_[0] for s_ in first["dec_in_steps"]][:NSTEPS]
    pres = [s_[1] for s_ in first["dec_in_steps"]][:NSTEPS]
    res = {
        "x": {},
        "pre": {},
        "idx": {},
        "wt": {},
        "S": S,
        "teacher": TEACHER,
        "env": {k: v for k, v in os.environ.items() if k.startswith(("DSV41", "MOE_COMPUTE"))},
    }
    for L in layers:
        ref = torch.load(os.path.join(CHAIN, f"layer_{L}.pt"))
        ref["S"] = S
        submit(L + 1), submit(L + 2)
        w = futs.pop(L).result()
        layer, attn = chain.build_layer(L, ref, w)
        del w
        for s in range(NSTEPS):
            if TEACHER:
                x, pre = ref["dec_in_steps"][s]
            else:
                x, pre = xs[s], pres[s]
                if engram is not None and L in engram.mods:
                    x = engram.apply(L, x, hashes[s])
            st = attn.step_inputs(torch.full((B,), S + s))
            tx, tp = chain.to_dev(x.reshape(B, -1), 4 * 5120), chain.to_dev(pre.reshape(B, -1), 4)
            out, nxt = layer.forward(tx, tp, st)
            ttnn.synchronize_device(mesh_device)
            xo = chain.to_host(out, 4 * 5120).reshape(B, 1, 4, 5120)
            po = chain.to_host(nxt, 4).reshape(B, 1, 4)
            try:
                sc, ix = layer.moe.last_routing
                cols = chain.cols
                rd = lambda t: torch.cat(
                    [
                        ttnn.to_torch(ttnn.get_device_tensors(t)[r * cols]).reshape(chain.T, -1)
                        for r in range(chain.rows)
                    ]
                )
                res["idx"][(L, s)] = rd(ix).to(torch.int32)[:B].clone()
                res["wt"][(L, s)] = rd(sc).float()[:B].clone()
            except Exception as e:  # routing readback is best effort
                log(f"routing readback failed: {e}")
            res["x"][(L, s)], res["pre"][(L, s)] = xo.reshape(B, 4, 5120).to(torch.bfloat16), po.reshape(B, 4)
            xs[s], pres[s] = xo, po
            ro, rp = ref["dec_out_steps"][s]
            log(
                f"RESULT layer {L:2d} step {s}: streams PCC {R.pcc(xo, ro.float()):.5f}  pre PCC {R.pcc(po, rp.float()):.5f}"
            )
        del layer, attn
        torch.save(res, OUT)
    log("DONE")
