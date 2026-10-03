# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolate moe_compute's arithmetic error. Real layer weights and FFN inputs (from dump_ffn_inputs.py), routing fixed
by us (gate bypassed). Device output is compared with CPU oracles built on bit-exact emulated bfp4 weights:

  exact          : checkpoint fp4 weights, exact math
  bfp4 oracle    : emulated bfloat4_b weights, exact math
  bfp4 + bf16    : emulated bfloat4_b weights, bf16-rounded intermediates
residual (device vs bfp4 oracle) = arithmetic / plumbing of the kernel, with weight quantisation removed.
"""

import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference import ref_layer as R
from models.demos.blackhole.deepseek_v41_flash.reference.bfp_emulation import bfp_roundtrip
from models.demos.blackhole.deepseek_v41_flash.tt.moe_block import DSV41MoEBlock
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import load_moe_layer

CHAIN = os.environ.get("DSV41_CHAIN", "/mnt/tt-data/ssinghal/dsv4-chain-e")
bf = lambda t: t.to(torch.bfloat16).float()
rel = lambda a, b: float((a - b).norm() / b.norm())


def expert(x, w0, w1, w2, bf16_act=False):
    g, u = x @ w0, x @ w1
    if bf16_act:
        g, u = bf(g), bf(u)
        return bf(bf(bf(torch.nn.functional.silu(g)) * u) @ w2)
    return (torch.nn.functional.silu(g) * u) @ w2


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [pytest.param({"l1_small_size": 16384, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}, id="ring")],
    indirect=True,
)
@pytest.mark.parametrize("layer_id", [int(x) for x in os.environ.get("DSV41_ISO_LAYERS", "1").split(",")])
@pytest.mark.timeout(3000)
@torch.no_grad()
def test_isolated(mesh_device, layer_id):
    rows, cols = tuple(mesh_device.shape)
    d = torch.load(os.path.join(CHAIN, f"ffn_inputs_{layer_id}.pt"))
    x, idx, wt = d["x"], d["idx"], d["wt"]  # [16,5120] bf16, [16,6], [16,6]
    B = x.shape[0]
    w = load_moe_layer(layer_id)
    moe = DSV41MoEBlock(mesh_device, w, gate_bias_shift=0.0)
    moe.warmup()
    shard = ttnn.ShardTensor2dMesh(mesh_device, dims=(0, None), mesh_shape=(rows, cols))
    up = lambda t, dt, lay: ttnn.from_torch(
        t, device=mesh_device, dtype=dt, layout=lay, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=shard
    )
    tt_x = up(x.reshape(B, 1, 1, 5120), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    tt_i = up(idx.reshape(B, 1, 1, 6).to(torch.int32), ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT)

    captured = {}
    orig_tilize = ttnn.tilize_with_val_padding

    def spy_tilize(t, *a, **k):  # t = moe_compute's combine output, before scores / reduction
        captured["combine"] = [ttnn.to_torch(dt).float() for dt in ttnn.get_device_tensors(t)]
        return orig_tilize(t, *a, **k)

    ttnn.tilize_with_val_padding = spy_tilize

    def run(scores):
        tt_s = up(scores.reshape(B, 1, 1, 6).to(torch.bfloat16), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        out = moe.decode.forward(tt_x=tt_x, tt_scores=tt_s, tt_indices=tt_i, layer_id=0)
        ttnn.synchronize_device(mesh_device)
        comp = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 3), mesh_shape=(rows, cols))
        return (
            ttnn.to_torch(ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG), mesh_composer=comp)
            .reshape(B, -1)
            .float()
        )

    xf = x.float()
    cache = {}

    def weights(e):
        if e not in cache:
            a, b, c = (w["w0"][0, e].float(), w["w1"][0, e].float(), w["w2"][0, e].float())
            cache[e] = (a, b, c, bfp_roundtrip(a, 3), bfp_roundtrip(b, 3), bfp_roundtrip(c, 3))
        return cache[e]

    def oracle(scores):
        outs = {k: torch.zeros(B, 5120) for k in ("exact", "bfp4", "bfp4+bf16")}
        for t in range(B):
            for j in range(6):
                s = float(scores[t, j])
                if s == 0.0:
                    continue
                a, b, c, qa, qb, qc = weights(int(idx[t, j]))
                outs["exact"][t] += s * expert(xf[t : t + 1], a, b, c)[0]
                outs["bfp4"][t] += s * expert(xf[t : t + 1], qa, qb, qc)[0]
                outs["bfp4+bf16"][t] += s * expert(xf[t : t + 1], qa, qb, qc, True)[0]
        return outs

    def alpha_stats(dev, orc):
        a = (dev * orc).sum(1) / (orc * orc).sum(1)
        return float(a.mean()), float(a.std())

    scales = [float(v) for v in os.environ.get("DSV41_ISO_SCALES", "1.0").split(",")]
    for sc in scales:
        # scale the INPUT tokens: a pure linear gain stays constant, an activation-dependent gain changes
        xs = (x.float() * sc).to(torch.bfloat16)
        xf = xs.float()
        tt_x = up(xs.reshape(B, 1, 1, 5120), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        for name, scores in (("single expert", torch.tensor([1.0, 0, 0, 0, 0, 0]).repeat(B, 1)), ("full routed", wt)):
            dev = run(scores)
            o = oracle(scores)
            if name == "single expert" and sc == 1.0:
                cmb = captured["combine"]
                print(f"ISO combine-output tensors: {len(cmb)} devices, shape of device 0: {tuple(cmb[0].shape)}")
                # tokens of mesh row r live on devices r*cols..; slot 0 of the combine output is the expert output
                for r in range(rows):
                    t = cmb[r * cols]
                    flat = t.reshape(t.shape[0], -1, t.shape[-1])  # [k, tokens, hidden]
                    for tok in range(flat.shape[1]):
                        g = r * flat.shape[1] + tok
                        a = flat[0, tok, :5120]
                        ref = oracle_tok0 = None
                    break
                t0 = cmb[0].reshape(cmb[0].shape[0], -1, cmb[0].shape[-1])  # [k, tokens, hidden] on device (0,0)
                for tok in range(t0.shape[1]):
                    exps = [int(e) for e in idx[tok]]
                    orc = []
                    for e in exps:
                        a_, b_, c_, qa, qb, qc = weights(e)
                        orc.append(expert(xf[tok : tok + 1], qa, qb, qc)[0])
                    line = []
                    for k in range(t0.shape[0]):
                        slot = t0[k, tok, :5120]
                        if float(slot.norm()) == 0.0:
                            line.append(f"slot{k}: zero")
                            continue
                        pccs = [R.pcc(slot, o_) for o_ in orc]
                        j = max(range(6), key=lambda i: pccs[i])
                        gain = float((slot * orc[j]).sum() / (orc[j] * orc[j]).sum())
                        line.append(f"slot{k}->expert#{j} PCC {pccs[j]:.4f} gain {gain:.3f}")
                    print(f"ISO combine token {tok}: " + " | ".join(line))
            am, asd = alpha_stats(dev, o["bfp4"])
            print(
                f"ISO L{layer_id} x*{sc:<5} [{name:13s}] dev/exact rel {rel(dev, o['exact']):.4f} | dev vs bfp4-oracle rel {rel(dev, o['bfp4']):.4f} "
                f"PCC {R.pcc(dev, o['bfp4']):.5f} | |dev|/|oracle| {float(dev.norm() / o['bfp4'].norm()):.4f} | per-token gain mean {am:.4f} std {asd:.4f} "
                f"| oracle vs exact {rel(o['bfp4'], o['exact']):.4f}"
            )
