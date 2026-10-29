# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Up=2 vs Up=1 numerics bisect: B=8 on layers UPV_LAYERS (e.g. 0,1,2), the same 8 real-text prompts (own code each, length UPV_LEN) in one prefill call, last-token logits saved to
UPV_SAVE/<name>.pt and compared with UPV_REF/<name>.pt (PCC, max |d|, top-1) when it exists. Run once per Up (DSV41_PREFILL_UP=1 / 2) with the same name.

    DSV41_PREFILL_UP=2 DSV41_PF_PAGED=read DSV41_PF_UMASK=1 DSV41_PF_SINGLE=1 UPV_LAYERS=0,1,2 UPV_NAME=l3 UPV_SAVE=... pytest tests/test_up_layer_bisect.py -s
"""

import json
import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.common import create_tt_model, default_page_params
from models.tt_transformers.tt.common import PagedAttentionConfig

LAYERS = [int(x) for x in os.environ.get("UPV_LAYERS", "0,1,2").split(",")]
LEN = int(os.environ.get("UPV_LEN", "4096"))
NAME = os.environ.get("UPV_NAME", "x")
B = 8


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": int(os.environ.get("DSV41_TRACE_REGION", "650000000")),
            },
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(7200)
@torch.no_grad()
def test_up_layer_bisect(mesh_device):
    md = mesh_device
    log = lambda s: print(s, flush=True)
    max_ctx = 65536
    pp = default_page_params(max_ctx, B // 4)
    args, m, _, _ = create_tt_model(
        md,
        B,
        max_ctx,
        PagedAttentionConfig(block_size=pp["page_block_size"], max_num_blocks=pp["page_max_num_blocks_per_dp"]),
        layer_ids=LAYERS,
        log=log,
    )
    from transformers import AutoTokenizer

    tk = AutoTokenizer.from_pretrained(os.environ.get("DSV41_CKPT", "/mnt/tt-data/ssinghal/deepseek-v41-flash-1017"))
    body = json.load(open("/mnt/tt-data/ssinghal/dsv4-logs/ttopt/needle_64k.json"))[0]["prompt"]
    fill = tk(body.split("\n\n", 1)[1].rsplit("\n\n", 1)[0], add_special_tokens=False)["input_ids"]
    codes = [4817, 2953, 6071, 3306, 7742, 1589, 9024, 5368]
    ids = []
    for b in range(B):
        h = tk(f"The secret code is {codes[b]}. Remember it.\n\n", add_special_tokens=False)["input_ids"]
        t = tk("\n\nThe secret code mentioned at the very start of the text above is", add_special_tokens=False)[
            "input_ids"
        ]
        n = LEN - len(h) - len(t) - 1
        ids.append(torch.tensor([tk.bos_token_id] + h + (fill * (n // len(fill) + 1))[:n] + t, dtype=torch.long))
    for b in range(B):
        m.release_user(b)
    res = m.prefill_interleaved(
        [(b, ids[b], 0, LEN) for b in range(B)],
        int(os.environ.get("UPV_CHUNK", "512")),
        s_pad_max=None,
        want_logits=True,
        final={b: True for b in range(B)},
        totals={b: LEN for b in range(B)},
    )
    lg = [res[b][1].float().clone() for b in range(B)]
    log(f"UPV {NAME} Up={m.Up} first tokens {[int(res[b][0]) for b in range(B)]}")
    sd, rd = os.environ.get("UPV_SAVE"), os.environ.get("UPV_REF")
    if sd:
        os.makedirs(sd, exist_ok=True)
        torch.save(lg, f"{sd}/{NAME}.pt")
    if rd and os.path.exists(f"{rd}/{NAME}.pt"):
        ref = torch.load(f"{rd}/{NAME}.pt")
        pcs = []
        for b in range(B):
            a, r = lg[b] - lg[b].mean(), ref[b] - ref[b].mean()
            pcs.append(float(a @ r / (a.norm() * r.norm() + 1e-12)))
        log(
            f"UPV {NAME} vs ref: pcc per user {[round(p, 5) for p in pcs]} max|d| {max(float((x - y).abs().max()) for x, y in zip(lg, ref)):.4f}"
        )
