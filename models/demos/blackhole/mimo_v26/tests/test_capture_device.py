# SPDX-License-Identifier: Apache-2.0
"""Capture the DEVICE per-layer tensors of the free-running 48-layer prefill (S=128, B=16, tokens of the CPU dump mimo-prefill-s128b16) so the CPU oracle can
be run teacher-forced per layer from the device h_in. One file per layer: {"prefill": {h_in, attn_in, attn_out, ffn_in, ffn_out, h_out [16,128,4096] bf16,
routing_idx [2048,8] int64, routing_w [2048,8] float32}, "meta": {...}} (same field names as the CPU dump; no route_gap/state/decode) + logits.pt (last-token
logits [16, V] float32, plus 'norm_in' = last-token hidden [16,4096] bf16).
Env: CAP_OUT (dir), CAP_LAYERS (48), MIMO_EXPERT_OP, MIMO_CKPT."""
import os

import pytest
import torch

import ttnn
from models.demos.blackhole.mimo_v26.tt.model import MiMoModel

E = os.environ.get
OUT = E("CAP_OUT", "/mnt/tt-data/ssinghal/mimo-dumps/mimo-device-s128b16")
NL = int(E("CAP_LAYERS", "48"))
DUMP = "/mnt/tt-data/ssinghal/mimo-dumps/mimo-prefill-s128b16"
ROWS, UPR, S, H = 4, 4, 128, 4096


def users(t):  # [rows, U*S, H] -> [16,S,H]
    return t.reshape(ROWS * UPR, S, -1)


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "trace_region_size": 64 * 1024 * 1024,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(14400)
@torch.no_grad()
def test_capture_device(mesh_device):
    os.makedirs(OUT, exist_ok=True)
    toks = torch.load(f"{DUMP}/tokens.pt")
    model = MiMoModel(
        mesh_device, n_layers=NL, max_pos=384, users=32, ring_blocks=S // 64 + 8, log=lambda s: print(s, flush=True)
    )
    h = model.embed_tokens(toks["prefill_tokens"].reshape(ROWS, UPR * S))
    bf = lambda t: users(model.rows_to_host(t)).bfloat16()
    for L in range(NL):
        lyr = model.layers[L]
        hin = bf(h)
        taps = {}
        h2 = lyr.prefill(h, S, UPR, taps=taps)
        pre = dict(h_in=hin, h_out=bf(h2))
        for k in ("attn_in", "attn_out", "ffn_in", "ffn_out"):
            pre[k] = bf(taps[k])
        if L > 0:
            w, ix = lyr.last_route
            dts_i, dts_w = ttnn.get_device_tensors(ix), ttnn.get_device_tensors(w)
            pre["routing_idx"] = torch.stack(
                [ttnn.to_torch(dts_i[r * 8]).reshape(UPR * S, 8).long() for r in range(ROWS)]
            ).reshape(-1, 8)
            pre["routing_w"] = torch.stack(
                [ttnn.to_torch(dts_w[r * 8]).reshape(UPR * S, 8).float() for r in range(ROWS)]
            ).reshape(-1, 8)
        torch.save(
            {"prefill": pre, "meta": {"layer": L, "S": S, "B": 16, "source": "device", "dtype": "bf16"}},
            f"{OUT}/layer_{L}.pt",
        )
        print(f"CAPTURE layer {L} saved", flush=True)
        for t in taps.values():
            ttnn.deallocate(t)
        ttnn.deallocate(h)
        h = h2
    hh = model.rows_to_host(h)
    last = torch.zeros(ROWS, 32, H)
    for r in range(ROWS):
        for u in range(UPR):
            last[r, u] = hh[r, u * S + S - 1]
    lg = model.final_logits(model.upload_rows(last))[:, :UPR].reshape(16, -1)
    torch.save(
        {"logits": lg, "norm_in": last[:, :UPR].reshape(16, H).bfloat16(), "prefill_tokens": toks["prefill_tokens"]},
        f"{OUT}/logits.pt",
    )
    print("CAPTURE done", flush=True)
