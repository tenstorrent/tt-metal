# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Probe: does a plain ttnn per-expert FFN with bf16 activations (bfp8 weights, HiFi2, fp32 acc) reach PCC 0.999 on layer L real tokens,
where the unified/fused ops (bfp8 activations/intermediates hard-wired in their programs) give 0.993 / 0.997 at L5? 1x1 mesh, first MIMO_PROBE_T tokens
of the track-0 dump, dump routing, real experts. Env: MIMO_LAYER (5), MIMO_PROBE_T (128), MIMO_PROBE_ACT (bf16|bfp8)."""
import os

import pytest
import torch

import ttnn
from models.demos.blackhole.mimo_v26.tests.test_moe_prefill_4x8 import pcc
from models.demos.blackhole.mimo_v26.tt.moe_weights import load_expert

DUMP = "/mnt/tt-data/ssinghal/mimo-dumps/mimo-prefill-s128b16"
L = int(os.environ.get("MIMO_LAYER", "5"))
T = int(os.environ.get("MIMO_PROBE_T", "128"))


@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@torch.no_grad()
def test_probe(mesh_device, mesh_shape):
    a = torch.load(f"{DUMP}/layer_{L}.pt", mmap=True)["prefill"]
    x = a["ffn_in"].reshape(-1, 4096)[:T]
    idx, w = a["routing_idx"][:T], a["routing_w"][:T]
    ref = a["ffn_out"].reshape(-1, 4096)[:T].float()
    ckc = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    act_dt = ttnn.bfloat16 if os.environ.get("MIMO_PROBE_ACT", "bf16") == "bf16" else ttnn.bfloat8_b
    out = torch.zeros(T, 4096)
    for e in sorted(set(idx.flatten().tolist())):
        tok, slot = (idx == e).nonzero(as_tuple=True)
        g, u, d = load_expert(L, e)
        td = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh_device)
        tg, tu, tdn = td(g), td(u), td(d)
        rows = x[tok]
        xd = ttnn.from_torch(rows, dtype=act_dt, layout=ttnn.TILE_LAYOUT, device=mesh_device)
        gg = ttnn.matmul(xd, tg, compute_kernel_config=ckc, dtype=ttnn.bfloat16)
        uu = ttnn.matmul(xd, tu, compute_kernel_config=ckc, dtype=ttnn.bfloat16)
        h = ttnn.multiply(ttnn.silu(gg), uu, dtype=ttnn.bfloat16)
        o = ttnn.matmul(h, tdn, compute_kernel_config=ckc, dtype=ttnn.bfloat16)
        out.index_add_(0, tok, ttnn.to_torch(o).float()[: len(tok)] * w[tok, slot].unsqueeze(1))
        for t in (tg, tu, tdn, xd, gg, uu, h, o):
            ttnn.deallocate(t)
    print(
        f"RESULT probe L{L} T={T} act={act_dt}: plain ttnn per-expert FFN (bfp8 weights, HiFi2 fp32acc) PCC vs bf16 dump ffn_out = {pcc(out, ref):.5f}",
        flush=True,
    )
