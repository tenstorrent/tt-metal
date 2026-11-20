# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Accuracy of the fused row matmul at each pass schedule on the checkpoint's weights, one core, next to ttnn.matmul
on the same operands.

    projections (layers 0-5, the share of chip $HY4_FID_CHIP of fused/projections/layout.py, BFP8):
        q_a        y [8, 6144] (BF16) x the chip's 64 columns
        gate       the same y x the chip's 512 gate columns (the sums before the sigmoid)
        o_proj     the gated head outputs of the chip's 2 heads [8, 512] (BF16) x o_proj's 512 rows -> [8, 6144]
    one routed expert of the MoE (layers 1-5, chip $HY4_FID_CHIP's 64 intermediate channels of a TP 32 split, BFP4):
        gate|up    x [8, 6144] (BF16) x [6144, 64 | 64]
        down       the expert's activation [8, 64] (BF16) x [64, 6144]
        chain      gate|up on the device, SwiGLU on the host, down on the device

Every form gets the same operands (the inputs of fused/projections/tests and fused/moe/tests) and is compared with
float64 arithmetic on the weights as stored (ttnn's BFP rounding), FP32 sums on both sides: relative rms error,
gain (the least-squares factor of the output on the reference) and PCC. "format" is the same reference against the
checkpoint's weights before rounding: what the storage format alone costs.

One HY4_MK9 line per (layer, matmul, form); all of them also go to $HY4_PROBE_DIR/accuracy.json.
Needs $HY4_HF_MODEL. On the mock Galaxy the programs are compiled only."""

import json
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.hy4_preview.fused.fidelity import lab as LAB
from models.experimental.hy4_preview.tests.common import pcc
from models.experimental.hy4_preview.tt.common import compute_config

pytestmark = pytest.mark.use_module_device

OUT_DIR = Path(os.environ.get("HY4_PROBE_DIR", "generated/hy4_preview/logs/mk9_probe"))
CHIP = int(os.environ.get("HY4_FID_CHIP", "13"))
LAYERS = [int(v) for v in os.environ.get("HY4_FID_LAYERS", "0,1,2,3,4,5").split(",")]
# form: (pass schedule, DEST layout of lab.LAYOUTS, the tree's custom_mm). "custom_mm" is what the fused pieces run
# today; the pass schedules run on the split accumulators with the FP32 merge (hifi_mm_merge).
FORMS = {
    "custom_mm": ("lofi", "dense_split", True),
    "lofi": ("lofi", "dense_split_exact", False),
    "hifi2": ("hifi2", "dense_split_exact", False),
    "hifi3": ("hifi3", "dense_split_exact", False),
    "hifi4": ("hifi4", "dense_split_exact", False),
    "p02": ("p02", "dense_split_exact", False),
    "hifi4 plain": ("hifi4", "dense", False),
    "hifi4 finalize": ("hifi4", "dense_split", False),
}
FIDELITIES = [f for f in os.environ.get("HY4_FID_FORMS", ",".join(FORMS)).split(",") if f]
TTNN_FIDELITIES = {
    "LoFi": ttnn.MathFidelity.LoFi,
    "HiFi2": ttnn.MathFidelity.HiFi2,
    "HiFi4": ttnn.MathFidelity.HiFi4,
}
_results = []


def bf16(t):
    return t.to(torch.bfloat16).float()


def block_k(kt, limit=32):
    return max(k for k in range(1, min(kt, limit) + 1) if kt % k == 0)


def lab_matmul(device, x, w, fmt, form, ct):
    """x [8, K] @ w [K, N] through the fused row matmul, N in slices of ct tile columns (one core's L1 holds a
    slice)."""
    kt, n = w.shape[0] // LAB.TILE, w.shape[1] // LAB.TILE
    assert n % ct == 0, (n, ct)
    fidelity, layout, ref = FORMS[form]
    passes, step = LAB.FIDELITIES[fidelity]
    op = LAB.Lab(device, kt, ct, fmt=fmt, passes=passes, step=step, kb=block_k(kt), layout=layout, ref=ref)
    outs = []
    try:
        op.set_x(x)
        for c in range(0, n, ct):
            op.set_w(w[:, LAB.TILE * c : LAB.TILE * (c + ct)])
            op()
            if LAB.MOCK:
                return None
            outs.append(op.read())
    finally:
        op.close()
    return torch.cat(outs, dim=1)


def ttnn_matmul(device, x, w, fmt, fidelity):
    """The same operands through ttnn.matmul (FP32 accumulation, FP32 output)."""
    ta = ttnn.from_torch(x.reshape(1, 1, *x.shape).float(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tw = ttnn.from_torch(
        w.reshape(1, 1, *w.shape).float().contiguous(), dtype=LAB.DTYPES[fmt], layout=ttnn.TILE_LAYOUT, device=device
    )
    ckc = compute_config(TTNN_FIDELITIES[fidelity], fp32_acc=True, packer_l1_acc=True)
    t = ttnn.matmul(ta, tw, compute_kernel_config=ckc, dtype=ttnn.float32)
    out = ttnn.to_torch(t).float().reshape(x.shape[0], w.shape[1])
    for v in (ta, tw, t):
        ttnn.deallocate(v)
    return out


def report(layer, name, form, ref, dev):
    ref, dev = ref.double(), dev.double()
    rel = ((ref - dev).pow(2).sum().sqrt() / ref.pow(2).sum().sqrt()).item()
    gain = ((ref * dev).sum() / ref.pow(2).sum()).item()
    score = pcc(ref.float(), dev.float())
    worst = (ref - dev).abs().max().item()
    _results.append(dict(layer=layer, matmul=name, form=form, rel_rms=rel, gain=gain, pcc=score, max_abs=worst))
    logger.info(
        f"HY4_MK9 accuracy L{layer} {name} [{form}]: rel_rms={rel:.3e} gain={gain:.5f} pcc={score:.7f} "
        f"max_abs={worst:.3e} ref_rms={ref.pow(2).mean().sqrt().item():.3e}"
    )


def save():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "accuracy.json"
    old = json.loads(path.read_text()) if path.exists() else []
    keys = {(r["layer"], r["matmul"], r["form"]) for r in _results}
    old = [r for r in old if (r["layer"], r["matmul"], r["form"]) not in keys]
    path.write_text(json.dumps(old + _results, indent=1))


def compare(device, layer, name, x, w, fmt, ct):
    """All forms of one matmul against exact arithmetic on the stored weights. Returns the reference."""
    stored = LAB.stored(w, fmt)
    exact = x.double() @ stored.double()
    if not LAB.MOCK:
        report(layer, name, f"format {fmt}", x.double() @ w.double(), exact)
    for fidelity in FIDELITIES:
        out = lab_matmul(device, x, w, fmt, fidelity, ct)
        if out is not None:
            report(layer, name, fidelity if fidelity == "custom_mm" else f"hifi_mm {fidelity}", exact, out)
    if not LAB.MOCK:
        for fidelity in TTNN_FIDELITIES:
            report(layer, name, f"ttnn.matmul {fidelity}", exact, ttnn_matmul(device, x, w, fmt, fidelity))
    return exact


@pytest.mark.parametrize("layer", LAYERS)
def test_accuracy_projections(device, layer):
    if not os.environ.get("HY4_HF_MODEL"):
        pytest.skip("HY4_HF_MODEL is not set")
    from models.experimental.hy4_preview.fused.projections import layout as L
    from models.experimental.hy4_preview.fused.projections import prepare
    from models.experimental.hy4_preview.fused.projections.tests.harness import random_input
    from models.experimental.hy4_preview.fused.projections.tests.test_post import post_inputs

    w = L.chip_weights(prepare.layer_weights(layer), CHIP)
    y, _ = random_input(seed=10 + layer)
    ctx, gate = post_inputs(w, seed=10 + layer)
    _, gated = L.post_golden(ctx, gate, w)
    compare(device, layer, "q_a", y, w["q_a"].contiguous(), "bfp8", 2)
    compare(device, layer, "gate", y, w["gate"].contiguous(), "bfp8", 2)
    compare(device, layer, "o_proj", bf16(gated), w["o"].contiguous(), "bfp8", 8)
    save()


def expert_slice(layer, expert, chip):
    """(gate|up [6144, 128], down [64, 6144]) of one routed expert: the chip's 64 intermediate channels, fp32."""
    import safetensors

    root = Path(os.environ["HY4_HF_MODEL"])
    weight_map = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"model.layers.{layer}.mlp.experts."
    out = []
    for name in ("gate_up_proj", "down_proj"):
        with safetensors.safe_open(str(root / weight_map[prefix + name]), framework="pt") as f:
            out.append(f.get_slice(prefix + name)[expert].float())
    gate_up, down = out  # [4096, 6144], [6144, 2048]
    c0, c1, I = 64 * chip, 64 * chip + 64, 2048
    w_gu = torch.cat((gate_up[c0:c1].T, gate_up[I + c0 : I + c1].T), dim=1).contiguous()
    return w_gu, down[:, c0:c1].T.contiguous()


@pytest.mark.parametrize("layer", [v for v in LAYERS if v >= 1])
def test_accuracy_expert(device, layer):
    if not os.environ.get("HY4_HF_MODEL"):
        pytest.skip("HY4_HF_MODEL is not set")
    from models.experimental.hy4_preview.fused.moe import layout as M
    from models.experimental.hy4_preview.fused.moe import prepare as moe_prepare
    from models.experimental.hy4_preview.fused.moe.tests.test_real import tokens

    x = bf16(tokens()[:8])
    prefix = f"model.layers.{layer}.mlp.gate."
    router_w, router_b = moe_prepare.checkpoint_tensors([prefix + "weight", prefix + "e_score_correction_bias"])
    _, picks, _ = M.route_golden(x, router_w, router_b)
    expert = int(picks[0, 0])
    w_gu, w_dn = expert_slice(layer, expert, CHIP)
    name = f"expert {expert}"
    gu = compare(device, layer, f"{name} gate|up", x, w_gu, "bfp4", 4)
    act = M.swiglu(gu[:, :64], gu[:, 64:])
    compare(device, layer, f"{name} down", bf16(act), w_dn, "bfp4", 8)

    # the chain: gate|up on the device, SwiGLU in fp32 on the host (BF16 activation), down on the device
    chain_ref = act @ LAB.stored(w_dn, "bfp4").double()
    for fidelity in FIDELITIES:
        dev_gu = lab_matmul(device, x, w_gu, "bfp4", fidelity, 4)
        if dev_gu is None:
            continue
        dev_act = bf16(M.swiglu(dev_gu[:, :64], dev_gu[:, 64:]))
        form = fidelity if fidelity == "custom_mm" else f"hifi_mm {fidelity}"
        report(layer, f"{name} chain", form, chain_ref, lab_matmul(device, dev_act, w_dn, "bfp4", fidelity, 8))
    if not LAB.MOCK:
        for fidelity in TTNN_FIDELITIES:
            dev_gu = ttnn_matmul(device, x, w_gu, "bfp4", fidelity)
            dev_act = bf16(M.swiglu(dev_gu[:, :64], dev_gu[:, 64:]))
            report(
                layer,
                f"{name} chain",
                f"ttnn.matmul {fidelity}",
                chain_ref,
                ttnn_matmul(device, dev_act, w_dn, "bfp4", fidelity),
            )
    save()
