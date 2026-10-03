# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One decode token through DeepSeek-V4.1-Flash layers on the Blackhole Galaxy vs the CPU reference chain.

Needs the reference chain files (reference/ref_chain.py) for the selected layers. Env:
  DSV41_LAYERS  e.g. "0-39" (default "0-0"),  DSV41_CHAIN  chain dir,  DSV41_ENGRAM=1,  DSV41_HEAD=1
Per layer it reports the PCC of the device output (chained: input = previous DEVICE output) against the reference
output; with the head enabled it also compares the logits and the argmax token of every user.
"""

import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference import ref_layer as R
from models.demos.blackhole.deepseek_v41_flash.tt.model import DSV41DecodeChain

CHAIN = os.environ.get("DSV41_CHAIN", "/mnt/tt-data/ssinghal/dsv4-chain")
LAYERS = os.environ.get("DSV41_LAYERS", "0-0")


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [pytest.param({"l1_small_size": 16384, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}, id="ring")],
    indirect=True,
)
@pytest.mark.timeout(14400)
@torch.no_grad()
def test_decode_chain(mesh_device):
    layers = []
    for part in LAYERS.split(","):  # "0-39" or "20,39" or "20-21,39"
        a, _, b = part.partition("-")
        layers += list(range(int(a), int(b or a) + 1))
    toks = torch.load(os.path.join(CHAIN, "tokens.pt"))
    S = toks["prefill_tokens"].shape[1]
    chain = DSV41DecodeChain(mesh_device, log=lambda m: print(m, flush=True))

    engram = hashes = None
    if os.environ.get("DSV41_ENGRAM") == "1":
        from models.demos.blackhole.deepseek_v41_flash.tt.host_model import HostEngram

        engram = HostEngram()
        engram.hashes(toks["prefill_tokens"], 0)
        hashes = engram.hashes(toks["decode_tokens"], S)

    from concurrent.futures import ThreadPoolExecutor

    from models.demos.blackhole.deepseek_v41_flash.tt.loader import load_layer

    prefetch = ThreadPoolExecutor(max_workers=2)  # host weight loading (I/O + dequant) overlaps device work
    futures = {}

    def submit(L):
        if L in layers and L not in futures:
            futures[L] = prefetch.submit(load_layer, L)

    for L in layers[:2]:
        submit(L)
    first = torch.load(os.path.join(CHAIN, f"layer_{layers[0]}.pt"))
    x, pre = first["dec_in"], first["pre_in"]
    worst = 1.0
    for L in layers:
        ref = torch.load(os.path.join(CHAIN, f"layer_{L}.pt"))
        ref["S"] = S
        if (
            os.environ.get("DSV41_TEACHER") == "1"
        ):  # isolate each layer: feed it the reference input, not the device chain's
            x, pre = ref["dec_in"], ref["pre_in"]
        submit(L + 1)
        submit(L + 2)
        w = futures.pop(L).result()
        x, pre = chain.run_layer(L, ref, x, pre, S, engram=engram, engram_hashes=hashes, w=w)
        del w
        p_x, p_pre = R.pcc(x, ref["dec_out"].float()), R.pcc(pre, ref["pre_out"].float())
        worst = min(worst, p_x)
        print(f"RESULT layer {L:2d}: streams PCC {p_x:.5f}  pre PCC {p_pre:.5f}", flush=True)
        if os.environ.get("DSV41_DIAG") == "1":
            r, g = ref["dec_out"].float().reshape(x.shape[0], 4, 5120), x.float().reshape(x.shape[0], 4, 5120)
            err = g - r
            print("DIAG per-user stream PCC:", [round(R.pcc(g[b], r[b]), 3) for b in range(g.shape[0])], flush=True)
            print(
                "DIAG per-user |err|/|ref|:",
                [round(float(err[b].norm() / r[b].norm()), 3) for b in range(g.shape[0])],
                flush=True,
            )
            ch = err.pow(2).sum((0, 1))  # error energy per hidden channel
            top = torch.topk(ch, 5)
            print(
                "DIAG top error channels:",
                [(int(i), round(float(v / ch.sum()), 3)) for v, i in zip(top.values, top.indices)],
                flush=True,
            )
            refch = r.pow(2).sum((0, 1))
            print(
                "DIAG ref energy share of those channels:",
                [round(float(refch[i] / refch.sum()), 3) for i in top.indices],
                flush=True,
            )
            print(
                "DIAG |ref| mean %.3f max %.1f ; |dev| mean %.3f max %.1f"
                % (r.abs().mean(), r.abs().max(), g.abs().mean(), g.abs().max()),
                flush=True,
            )

    if os.environ.get("DSV41_HEAD") == "1":
        from models.demos.blackhole.deepseek_v41_flash.tt.host_model import HostHead

        logits = HostHead()(x, pre)
        fin = torch.load(os.path.join(CHAIN, "final.pt"))
        match = (logits.argmax(-1) == fin["argmax"]).float().mean().item()
        print(
            f"RESULT logits PCC {R.pcc(logits, fin['logits']):.5f}; argmax token match {match * 100:.0f}% of {logits.shape[0]} users",
            flush=True,
        )
        print("device tokens:", logits.argmax(-1).tolist(), flush=True)
        print("ref    tokens:", fin["argmax"].tolist(), flush=True)
    assert worst > 0.9
