# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness gate for MiMo MoE decode changes at the serving shape: one layer on Galaxy row 0, 8 decode slots,
BFP4 experts (moe_decode_common.verify_steps: one / two requests of 4 consecutive tokens, 4 / 8 distinct tokens).

1. Decode: every active slot's output must match the base implementation's saved output on the same inputs, PCC >=
   0.9998 overall and >= 0.9995 on every (step, slot) row, so one dropped or mis-weighted expert of one token fails
   it; every row of the [32, hidden] output (idle and padding rows too) must be finite.
2. Decode against the fp32 reference evaluated with the device's own top-8 routing: PCC >= 0.998 overall and >=
   0.99 per row (BFP4 GPTQ experts, LoFi; the base: 0.99899 and 0.9935), so the base itself cannot drift.
3. Prefill (TtMiMoMoE.forward) on 96 real tokens must match the base prefill output, PCC >= 0.9995 overall and
   >= 0.999 per token: decode may change the expert layout, and prefill must keep working.

``--write-base`` records the base outputs (outside the repository). Prints MOE_VERIFY_CORRECTNESS_PASS after every
check passed.
"""

import argparse
import os

import torch

import ttnn
from models.experimental.mimo_v2_6_flash.reference import model as ref
from models.experimental.mimo_v2_6_flash.tests.perf import moe_decode_common as common

BASE = os.environ.get("MIMO_MOE_VERIFY_BASE", os.path.join(common.GOLDEN_DIR, "moe_verify_layer1_bfp4_base.pt"))
STEPS = 6  # x 4 shapes
USERS = 8
PREFILL_TOKENS = 96


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def row_pcc(a, b):
    return min(pcc(x, y) for x, y in zip(a, b))


def run_decode(row, moe, inputs, steps):
    """Per step: the active slots' outputs [n, hidden], the device's routing weights [n, E] and whether all 32
    output rows are finite."""
    outs, dense_rows, finite = [], [], True
    for _, idx in steps:
        x, mask = common.step_inputs(row, inputs, idx)
        dense = moe.route(x, mask)
        out = moe.forward_decode(x, mask)
        full = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()[0, 0]
        d = ttnn.to_torch(ttnn.get_device_tensors(dense)[0]).float()[0, 0]
        finite &= bool(torch.isfinite(full).all())
        live = [s for s, i in enumerate(idx) if i >= 0]
        outs.append(full[live])
        dense_rows.append(d[live])
        for t in (x, mask, dense, out):
            t.deallocate(True)
    return outs, dense_rows, finite


def run_prefill(row, moe, inputs):
    rep = ttnn.ReplicateTensorToMesh(row)
    x = ttnn.from_torch(
        inputs[:PREFILL_TOKENS].reshape(1, 1, PREFILL_TOKENS, -1).bfloat16(),
        device=row,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=rep,
    )
    out = moe.forward(x)
    got = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()[0, 0, :PREFILL_TOKENS]
    x.deallocate(True)
    out.deallocate(True)
    return got


def reference(inputs, steps, dense_rows, experts):
    """fp32 MoE of each step's active tokens with the device's routing (weights as the device used them)."""
    outs = []
    with torch.no_grad():
        for (_, idx), d in zip(steps, dense_rows):
            x = torch.stack([inputs[i] for i in idx if i >= 0]).float()
            y = torch.zeros_like(x)
            for t in range(x.shape[0]):
                for e in torch.nonzero(d[t]).reshape(-1).tolist():
                    gate = x[t] @ experts["gate_proj"][e].float().T
                    up = x[t] @ experts["up_proj"][e].float().T
                    y[t] += d[t, e] * ((torch.nn.functional.silu(gate) * up) @ experts["down_proj"][e].float().T)
            outs.append(y)
    return outs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    parser.add_argument("--write-base", action="store_true", help="record this code's outputs as the base")
    args = parser.parse_args()

    cfg = common.config()
    ckpt = common.checkpoint(cfg)
    inputs = common.golden_moe_inputs(cfg, ckpt)
    steps = common.verify_steps(cfg, STEPS)
    mesh, row = common.open_row()
    ok = True
    try:
        moe = common.build_moe(row, cfg, "model_bfp4experts", ttnn.bfloat4_b, users=USERS)
        outs, dense_rows, finite = run_decode(row, moe, inputs, steps)
        pre = run_prefill(row, moe, inputs)
    finally:
        common.close(mesh, row)
    if args.write_base:
        torch.save({"decode": outs, "prefill": pre}, BASE)
        print(f"wrote base outputs {BASE}", flush=True)
        return
    base = torch.load(BASE)
    got, want = torch.cat(outs), torch.cat(base["decode"])
    p, r = pcc(got, want), row_pcc(got, want)
    print(f"decode vs base: PCC {p:.6f} (worst row {r:.6f}), finite {finite}", flush=True)
    ok &= p >= 0.9998 and r >= 0.9995 and finite
    for name in ("one_request", "two_requests", "plain4", "plain8"):
        sel = [k for k, (n, _) in enumerate(steps) if n == name]
        g, w = torch.cat([outs[k] for k in sel]), torch.cat([base["decode"][k] for k in sel])
        print(f"  {name}: PCC {pcc(g, w):.6f} (worst row {row_pcc(g, w):.6f})", flush=True)
    experts = ref.LazyExperts(ckpt, common.LAYER, torch.bfloat16)
    sel = list(range(0, len(steps), 3))  # a third of the steps, every shape
    ref_out = torch.cat(reference(inputs, [steps[k] for k in sel], [dense_rows[k] for k in sel], experts))
    dev_out = torch.cat([outs[k] for k in sel])
    p, r = pcc(dev_out, ref_out), row_pcc(dev_out, ref_out)
    print(f"decode vs fp32 reference with device routing: PCC {p:.6f} (worst row {r:.6f})", flush=True)
    ok &= p >= 0.998 and r >= 0.99
    pp, rp = pcc(pre, base["prefill"]), row_pcc(pre, base["prefill"])
    print(f"prefill vs base: PCC {pp:.6f} (worst token {rp:.6f})", flush=True)
    ok &= pp >= 0.9995 and rp >= 0.999
    if ok:
        print("MOE_VERIFY_CORRECTNESS_PASS", flush=True)
    else:
        raise RuntimeError("MoE verify correctness check failed")


if __name__ == "__main__":
    main()
