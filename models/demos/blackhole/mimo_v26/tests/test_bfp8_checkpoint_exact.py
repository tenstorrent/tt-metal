# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Is bfp8 lossless for the checkpoint's fp4 (e2m1, low nibble first, e8m0 scale per 32 along K) routed experts?
Reads real experts with a minimal safetensors reader (MIMO_CKPT, default /mnt/tt-data/MiMo-V2.6-Flash-RL), dequantises to bf16
(DSV4 ref_kernels.dequant_fp4_weight nibble order; e8m0 decoded as 2^(s-127): both UNVERIFIED vs MiMo, sanity-checked below), transposes to the
device layout [K,N] (bfp8 blocks of 16 then run along N, as in the device weights), converts to bfp8 both on host (from_torch) and on device
(typecast bf16 -> bfloat8_b), reads back and reports max abs diff / PCC / number of changed values. Env: MIMO_LAYERS (1,20,46), MIMO_EXPERTS (0,137,255).
"""
import json
import os

import pytest
import torch

import ttnn

CKPT = os.environ.get("MIMO_CKPT", "/mnt/tt-data/MiMo-V2.6-Flash-RL")
LAYERS = [int(v) for v in os.environ.get("MIMO_LAYERS", "1,20,46").split(",")]
EXPERTS = [int(v) for v in os.environ.get("MIMO_EXPERTS", "0,137,255").split(",")]
FP4_TABLE = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
_IDX = {}


def dequant_fp4(w_u8, s_u8, block=32):
    b = w_u8.view(torch.uint8)
    pair = torch.stack([FP4_TABLE[(b & 0x0F).long()], FP4_TABLE[(b >> 4).long()]], dim=-1)
    n = b.size(0)
    wf = pair.reshape(n, -1)
    scale = torch.pow(2.0, s_u8.view(torch.uint8).float() - 127.0)
    return (wf.view(n, -1, block) * scale.unsqueeze(-1)).reshape(n, -1), b, s_u8


def get(name):
    from safetensors import safe_open

    if not _IDX:
        _IDX.update(json.load(open(os.path.join(CKPT, "model.safetensors.index.json")))["weight_map"])
    with safe_open(os.path.join(CKPT, _IDX[name]), "pt") as f:
        return f.get_tensor(name)


def pcc(a, b):
    return float(torch.corrcoef(torch.stack([a.flatten().double(), b.flatten().double()]))[0, 1])


@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("layer", LAYERS)
def test_bfp8_reproduces_fp4_checkpoint(mesh_device, mesh_shape, layer):
    if not os.path.exists(os.path.join(CKPT, "model.safetensors.index.json")):
        pytest.skip("checkpoint missing")
    nbad = 0
    for e in EXPERTS:
        for proj in ("gate_proj", "up_proj", "down_proj"):
            base = f"model.layers.{layer}.mlp.experts.{e}.{proj}"
            w, packed, s = dequant_fp4(get(base + ".weight"), get(base + ".weight_scale"))  # [out, in]
            codes = torch.bincount(torch.cat([(packed & 15).flatten(), (packed >> 4).flatten()]).long(), minlength=16)
            wk = w.t().contiguous().to(torch.bfloat16)  # device layout [K, N]
            exact_bf16 = bool((wk.float() == w.t()).all())
            host = ttnn.to_torch(ttnn.from_torch(wk, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)).float()
            dev_bf16 = ttnn.from_torch(wk, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
            dev = ttnn.to_torch(ttnn.typecast(dev_bf16, ttnn.bfloat8_b)).float()
            ref = w.t()
            print(
                f"RESULT layer={layer} e={e} {proj} KxN={tuple(ref.shape)} absmax={ref.abs().max():.4e} scale_e8m0=[{int(s.min())},{int(s.max())}] "
                f"codes_used={int((codes > 0).sum())}/16 bf16_exact={exact_bf16} "
                f"host_bfp8: maxdiff={(host - ref).abs().max():.3e} changed={(host != ref).sum().item()}/{ref.numel()} pcc={pcc(host, ref):.8f} | "
                f"device_typecast_bfp8: maxdiff={(dev - ref).abs().max():.3e} changed={(dev != ref).sum().item()}/{ref.numel()} pcc={pcc(dev, ref):.8f}",
                flush=True,
            )
            nbad += int((host != ref).any()) + int((dev != ref).any())
    assert (
        nbad == 0
    ), f"{nbad} tensors not reproduced exactly by bfp8 (see RESULT lines): precision decision for the coordinator"
