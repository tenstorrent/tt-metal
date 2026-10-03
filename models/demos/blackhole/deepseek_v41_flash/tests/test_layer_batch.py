# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One decode layer at users_per_row = 4 / 8 / 16 with the 16-user reference state and inputs tiled: per-user PCC against the
reference output and a NaN/inf scan of each stage (attention out, FFN input, routed, shared). Env DSV41_BATCH_LAYERS ("2,20")."""

import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference import ref_layer as R
from models.demos.blackhole.deepseek_v41_flash.tt.loader import load_layer
from models.demos.blackhole.deepseek_v41_flash.tt.model import DSV41DecodeChain

CHAIN = os.environ.get("DSV41_CHAIN", "/mnt/tt-data/ssinghal/dsv4-chain-m")
LAYERS = [int(x) for x in os.environ.get("DSV41_BATCH_LAYERS", "0,2").split(",")]
UPR = [int(x) for x in os.environ.get("DSV41_BATCH_UPR", "4,8,16").split(",")]


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [pytest.param({"l1_small_size": 16384, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}, id="ring")],
    indirect=True,
)
@torch.no_grad()
def test_layer_batch(mesh_device):
    md = mesh_device
    rows, cols = tuple(md.shape)
    S = torch.load(os.path.join(CHAIN, "tokens.pt"))["prefill_tokens"].shape[1]
    for upr in UPR:
        for L in LAYERS:
            chain = DSV41DecodeChain(md, users_per_row=upr, log=print)
            B, reps = chain.B, chain.B // 16
            ref = torch.load(os.path.join(CHAIN, f"layer_{L}.pt"))
            ref["S"] = S
            ref["state"] = {
                k: (v.repeat(reps, *([1] * (v.dim() - 1))) if torch.is_tensor(v) else v)
                for k, v in ref["state"].items()
            }
            layer, attn = chain.build_layer(L, ref, load_layer(L))
            layer.debug = {}
            st = attn.step_inputs(torch.full((B,), S))
            x_in = ref["dec_in"].repeat(reps, 1, 1, 1)
            pre_in = ref["pre_in"].repeat(reps, 1, 1)
            tx, tp = chain.to_dev(x_in.reshape(B, -1), 4 * 5120), chain.to_dev(pre_in.reshape(B, -1), 4)
            out, nxt = layer.forward(tx, tp, st)
            ttnn.synchronize_device(md)
            got = chain.to_host(out, 4 * 5120).reshape(B, 1, 4, 5120)
            want = ref["dec_out"].float().repeat(reps, 1, 1, 1)
            per_user = [R.pcc(got[u], want[u]) for u in range(B)]
            bad = [u for u, p in enumerate(per_user) if not (p > 0.99)]
            print(
                f"BAT upr {upr} (batch {B}) layer {L}: all-user PCC {R.pcc(got, want):.5f}  min per-user {min(per_user):.5f}  users below 0.99: {bad}",
                flush=True,
            )
            for k, t in layer.debug.items():
                try:
                    h = torch.cat(
                        [
                            ttnn.to_torch(ttnn.get_device_tensors(t)[r * cols]).float().reshape(upr, -1)
                            for r in range(rows)
                        ]
                    )
                    print(
                        f"BAT    {k:10s} shape {tuple(h.shape)}  nan {int(torch.isnan(h).sum())}  inf {int(torch.isinf(h).sum())}  absmax {float(h.abs().nan_to_num().max()):.2f}",
                        flush=True,
                    )
                except Exception as e:
                    print(f"BAT    {k}: could not read ({str(e)[:60]})", flush=True)
            del layer, attn, chain
