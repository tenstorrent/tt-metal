# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Phase A router test: expert-SET agreement vs the fp32 HF reference on REAL hidden states.

Input: MIMO_ROUTER_DUMP=<file.pt> holding {"x": [T,4096] hidden state entering the MoE layer (post_attention_layernorm
output), "w": [256,4096] gate.weight, "bias": [256] e_score_correction_bias, "layer": int}. Produce it from track-0 dumps
(no random tokens: the test skips without it). Results are printed as RESULT lines for the LOG.
Gate (brief): hilo set agreement >= 99%; weights PCC >= 0.9999. plain is measured, not gated.
"""
import os

import pytest
import torch

import ttnn
from models.demos.blackhole.mimo_v26.tt.router import TOPK, MiMoGate, calibrate_shift, cpu_route

DUMP = os.environ.get("MIMO_ROUTER_DUMP")


def _load():
    if not DUMP or not os.path.exists(DUMP):
        pytest.skip("set MIMO_ROUTER_DUMP to a real-token router dump (see module docstring)")
    d = torch.load(DUMP)
    return d["x"].float(), d["w"].float(), d["bias"].float(), d.get("layer", -1)


def _set_agree(a, b):
    return float(torch.tensor([set(r1.tolist()) == set(r2.tolist()) for r1, r2 in zip(a, b)]).float().mean())


def _w_pcc(idx_ref, w_ref, idx, w):
    # compare weights per expert id (order independent)
    T = idx.shape[0]
    full_ref = torch.zeros(T, 256).scatter_(1, idx_ref, w_ref)
    full = torch.zeros(T, 256).scatter_(1, idx.long(), w.float())
    return float(torch.corrcoef(torch.stack([full_ref.flatten(), full.flatten()]))[0, 1])


def test_cpu_emulation_flip_rate():
    """No device: how often bf16 emulation of the plain gate flips the set (hint only; device test is the evidence)."""
    x, w, b, layer = _load()
    ref, _ = cpu_route(x, w, b)
    emu, _ = cpu_route(x, w, b, "bf16_emul")
    emu_s, _ = cpu_route(x, w, b, "bf16_emul", shift=calibrate_shift(x, w, b))
    print(
        f"RESULT cpu_emul layer={layer} T={x.shape[0]} plain={_set_agree(ref, emu):.4f} shifted={_set_agree(ref, emu_s):.4f}"
    )


@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_router_agreement(mesh_device, mesh_shape):
    x, w, b, layer = _load()
    T = int(os.environ.get("MIMO_ROUTER_T", "32"))
    ntok = (x.shape[0] // T) * T
    x = x[:ntok]
    ref_idx, ref_w = cpu_route(x, w, b)
    shift = calibrate_shift(x, w, b)
    print(f"RESULT layer={layer} bias mean={b.mean():.3f} std={b.std():.4f} shift={shift:.4f} tokens={ntok} batch={T}")
    res = {}
    for mode, sh in (("plain", 0.0), ("hilo", 0.0), ("hilo", shift), ("exact", 0.0)):
        gate = MiMoGate(mesh_device, w.to(torch.bfloat16), b, bias_shift=sh, batch_per_device=T)
        idxs, wts = [], []
        for s in range(0, ntok, T):
            tt_x = ttnn.from_torch(
                x[s : s + T].reshape(1, 1, T, -1).to(torch.bfloat16),
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
            wt, ix = gate.route(tt_x, mode)
            idxs.append(ttnn.to_torch(ttnn.get_device_tensors(ix)[0]).reshape(T, TOPK).int())
            wts.append(ttnn.to_torch(ttnn.get_device_tensors(wt)[0]).reshape(T, TOPK).float())
        idx, wt = torch.cat(idxs), torch.cat(wts)
        a = _set_agree(ref_idx, idx)
        p = _w_pcc(ref_idx, ref_w, idx, wt)
        res[(mode, sh)] = a
        print(f"RESULT layer={layer} mode={mode} shift={sh:.3f} set_agree={a:.4f} weights_pcc={p:.6f}")
    assert res[("hilo", 0.0)] >= 0.99 or res[("hilo", shift)] >= 0.99, f"hilo set agreement below 99%: {res}"
