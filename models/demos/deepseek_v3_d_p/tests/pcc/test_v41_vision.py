# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash ViT + aligner on ONE device vs the checkpoint's own ``inference/vision.py`` on the CPU, on a real image
through the checkpoint's own ``image_processor.load_image`` (tt-blaze vit_plan.md V1).

The reference of record is vision.py in **fp32**; the bf16 run (what model.py builds, generate.py's default dtype) is the
noise floor. MEASURED on T1 (2026-10-12, DS41F-0045): bf16 vs fp32 reaches per-row PCC 0.83 on the aligner rows. From
block 12 a few tokens carry massive activations (channels 308 / 53, +-1520); ordinary rows must cancel to ~0 there, and
bf16 rounding breaks that. A bf16 golden would therefore grade the device against noise.

Graded: every block's output, the final-norm output and the aligner rows, against fp32 (aggregate, per-row minimum and
mean) with the bf16 floor alongside. The device aligner alone on the fp32 reference's ViT output separates the aligner's
own error from the tower's.

    V41_VIT_IMAGE=<path> (default the T1 dog)   V41_VIT_LAYERS=32   V41_VIT_MLP_FP32=0   V41_VIT_PCC=0.75 (aligner per-row min)   V41_VIT_MEAN_GAP=0.002
    V41_VIT_DUMP=<file.pt>
"""

from __future__ import annotations

import json
import os
import sys
import time
import types

import pytest
import torch
from loguru import logger

from models.demos.deepseek_v3_d_p.tt.v41.config import DEFAULT_MODEL_DIR
from models.demos.deepseek_v3_d_p.tt.v41.vision import V41Vision, load_vision_state
from models.demos.deepseek_v3_d_p.tt.v41.weights import checkpoint

IMAGE = os.environ.get("V41_VIT_IMAGE", "/mnt/tt-data/sdawle/dsv41_vit/images/T1_dog.jpg")
N_LAYERS = int(os.environ.get("V41_VIT_LAYERS", "32"))
# MEASURED floors (DS41F-0045, flash k_chunk 512): T1 aligner rows per-row min 0.97 / mean 0.9990; T4 (8649 patches) 0.82 / 0.9980
# against bf16 model.py's 0.90 / 0.9993. The flash SDPA's bf16 running accumulators cost the large grids ~0.001 of mean PCC.
MIN_PCC = float(os.environ.get("V41_VIT_PCC", "0.75"))
MEAN_GAP = float(os.environ.get("V41_VIT_MEAN_GAP", "0.002"))
MLP_FP32 = os.environ.get("V41_VIT_MLP_FP32", "0") == "1"


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def _row_pcc(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a, b = a.double(), b.double()
    a, b = a - a.mean(-1, keepdim=True), b - b.mean(-1, keepdim=True)
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-30)


def _reference(model_dir: str, state: dict, n_layers: int, dtype):
    """The checkpoint's ViT + Aligner modules in ``dtype`` (bf16 = generate.py's default dtype before building)."""
    sys.path.insert(0, os.path.join(model_dir, "inference"))
    import vision as vis  # noqa: E402

    cfg = json.load(open(os.path.join(model_dir, "inference", "config.json")))
    args = types.SimpleNamespace(**cfg)
    args.vision_n_layers = n_layers
    saved = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        vit, al = vis.ViT(args), vis.Aligner(args)
    finally:
        torch.set_default_dtype(saved)
    vit.load_state_dict(
        {
            k[len("vision.") :]: v
            for k, v in state.items()
            if k.startswith("vision.") and (not k.startswith("vision.blocks.") or int(k.split(".")[2]) < n_layers)
        },
        strict=True,
    )
    al.load_state_dict({k[len("aligner.") :]: v for k, v in state.items() if k.startswith("aligner.")}, strict=True)
    return vit.eval(), al.eval(), args


def _run_ref(vit, al, patches, n_h, n_w, dtype):
    blocks = []
    hooks = [b.register_forward_hook(lambda m, a, out: blocks.append(out.detach().float().clone())) for b in vit.blocks]
    with torch.inference_mode():
        y = vit(patches.to(dtype), n_h, n_w)
        rows = al(y, n_h, n_w)
    for h in hooks:
        h.remove()
    return blocks, y.float(), rows.float()


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
def test_v41_vision(mesh_device):
    model_dir = DEFAULT_MODEL_DIR
    state = load_vision_state(checkpoint(model_dir))
    sys.path.insert(0, os.path.join(model_dir, "inference"))
    import image_processor as ip  # noqa: E402

    vit32, al32, args = _reference(model_dir, state, N_LAYERS, torch.float32)
    vit16, al16, _ = _reference(model_dir, state, N_LAYERS, torch.bfloat16)
    patches, n_h, n_w, n_llm_h, n_llm_w = ip.load_image({"url": IMAGE}, args)
    logger.info(
        f"[v41-vit] {IMAGE}: ViT grid {n_h} x {n_w} = {patches.shape[0]} patches, LLM grid {n_llm_h} x {n_llm_w} "
        f"({ip.num_image_tokens(n_llm_h, n_llm_w)} span tokens); device MLP fp32 {MLP_FP32}"
    )
    t0 = time.perf_counter()
    b32, y32, rows32 = _run_ref(vit32, al32, patches, n_h, n_w, torch.float32)
    t_ref = time.perf_counter() - t0
    b16, y16, rows16 = _run_ref(vit16, al16, patches, n_h, n_w, torch.bfloat16)

    t0 = time.perf_counter()
    dev = V41Vision(mesh_device, state, n_layers=N_LAYERS, mlp_fp32=MLP_FP32)
    logger.info(f"[v41-vit] device weights uploaded in {time.perf_counter() - t0:.1f} s")
    dev.aligner(dev.vit(patches, n_h, n_w), n_h, n_w)  # warm: program compile
    cap = []
    dev_vit = dev.vit(patches, n_h, n_w, capture=cap)
    t0 = time.perf_counter()
    dev_vit_nc = dev.vit(patches, n_h, n_w)
    t_vit = time.perf_counter() - t0
    t0 = time.perf_counter()
    dev_rows = dev.aligner(dev_vit_nc, n_h, n_w)
    t_al = time.perf_counter() - t0
    al_on_ref = dev.aligner(y32.to(torch.bfloat16), n_h, n_w)
    assert torch.equal(dev_vit_nc, dev_vit), "the device ViT is not deterministic run to run"

    for i, (d, r, f) in enumerate(zip(cap, b32, b16)):
        rp, rf = _row_pcc(d, r), _row_pcc(f, r)
        logger.info(
            f"[v41-vit] block {i:2d} vs fp32: device PCC {_pcc(d, r):.6f} per-row min {float(rp.min()):.4f} "
            f"(row {int(rp.argmin())}) | bf16 ref per-row min {float(rf.min()):.4f} (row {int(rf.argmin())})"
        )
    summary = {}
    for name, d, r, f in (("final norm", dev_vit, y32, y16), ("aligner rows", dev_rows, rows32, rows16)):
        rp, rf = _row_pcc(d, r), _row_pcc(f, r)
        summary[name] = (rp, rf)
        logger.info(
            f"[v41-vit] {name} {tuple(r.shape)} vs fp32: device PCC {_pcc(d, r):.6f}, per-row min {float(rp.min()):.4f} "
            f"mean {float(rp.mean()):.5f} | bf16 ref PCC {_pcc(f, r):.6f}, per-row min {float(rf.min()):.4f} "
            f"mean {float(rf.mean()):.5f}"
        )
    rp_al = _row_pcc(al_on_ref, rows32)
    logger.info(
        f"[v41-vit] device aligner alone (on the fp32 ViT output): per-row min {float(rp_al.min()):.5f}, mean "
        f"{float(rp_al.mean()):.6f}"
    )
    logger.info(
        f"[v41-vit] wall: CPU fp32 reference {t_ref:.2f} s; device ViT {t_vit * 1e3:.1f} ms, aligner {t_al * 1e3:.1f} ms "
        f"(host unfold incl.)"
    )
    if os.environ.get("V41_VIT_DUMP"):
        torch.save(
            dict(
                image=IMAGE,
                grid=(n_h, n_w),
                mlp_fp32=MLP_FP32,
                ref32_vit=y32,
                ref16_vit=y16,
                dev_vit=dev_vit,
                ref32_rows=rows32,
                ref16_rows=rows16,
                dev_rows=dev_rows,
                ref32_blocks=b32,
                ref16_blocks=b16,
                dev_blocks=cap,
            ),
            os.environ["V41_VIT_DUMP"],
        )
    if os.environ.get("V41_VIT_TF", "0") == "1":
        # teacher forcing: device block i on the fp32 reference's block i-1 output; its attention / MLP sublayers against
        # fp32 on the same input, with bf16 model.py on the same (bf16-cast) input as the floor. Localises WHICH sublayer
        # adds the error, block by block, without the accumulated drift.
        with torch.inference_mode():
            x_prev32 = vit32.patch_embed(patches.float())
            for i in range(N_LAYERS):
                b32_, b16_ = vit32.blocks[i], vit16.blocks[i]
                cos, sin = __import__("vision").get_vision_cos_sin(n_h, n_w, vit32.rope_dim, vit32.rope_theta)
                a32 = b32_.attn(b32_.norm1(x_prev32), cos, sin)
                x1_32 = x_prev32 + a32
                m32 = b32_.mlp(b32_.norm2(x1_32))
                x16 = x_prev32.to(torch.bfloat16)
                a16 = b16_.attn(b16_.norm1(x16), cos, sin).float()
                x1_16 = x16 + a16.to(torch.bfloat16)
                m16 = b16_.mlp(b16_.norm2(x1_16)).float()
                d = dev.block_forced(x_prev32, i, n_h, n_w)
                ra, rfa = _row_pcc(d["attn"], a32), _row_pcc(a16, a32)
                rm, rfm = _row_pcc(d["mlp"], m32), _row_pcc(m16, m32)
                logger.info(
                    f"[v41-vit-tf] block {i:2d}: attn dev min {float(ra.min()):.4f} mean {float(ra.mean()):.5f} | bf16 min "
                    f"{float(rfa.min()):.4f} mean {float(rfa.mean()):.5f} || mlp dev min {float(rm.min()):.4f} mean "
                    f"{float(rm.mean()):.5f} | bf16 min {float(rfm.min()):.4f} mean {float(rfm.mean()):.5f}"
                )
                x_prev32 = b32[i]
    rp, rf = summary["aligner rows"]
    assert float(rp_al.min()) >= 0.99, f"device aligner alone: per-row min PCC {float(rp_al.min()):.5f} < 0.99"
    assert float(rp.min()) >= MIN_PCC, f"aligner rows vs fp32: per-row min PCC {float(rp.min()):.5f} < {MIN_PCC}"
    assert (
        float(rp.mean()) >= float(rf.mean()) - MEAN_GAP
    ), f"aligner rows vs fp32: device per-row mean {float(rp.mean()):.5f} is below the bf16 reference's {float(rf.mean()):.5f}"
