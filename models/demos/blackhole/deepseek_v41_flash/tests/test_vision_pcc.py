# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1-Flash image encoder (tt/vision.py: ViT + aligner, tensor parallel over the mesh columns) vs the checkpoint's reference
``inference/vision.py`` on the CPU (bf16, same weights). Images: synthetic arrays of a few sizes (-> different patch grids, single window and padded tail).

  VIS_SIZES   comma list of WxH pixel sizes (default 280x196,588x420)
  VIS_LAYERS  run only the first N blocks on both sides (default all 32)
"""

import importlib.util
import os

import numpy as np
import pytest
import torch
from PIL import Image

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference import ref_layer as R
from models.demos.blackhole.deepseek_v41_flash.tt import vision_prompt as VP
from models.demos.blackhole.deepseek_v41_flash.tt.model_args import CKPT
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import _Shards
from models.demos.blackhole.deepseek_v41_flash.tt.vision import DSV41Vision, VisionArgs
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager


def _ref_modules(va, n_layers, sh):
    spec = importlib.util.spec_from_file_location("dsv41_ckpt_vision", os.path.join(CKPT, "inference", "vision.py"))
    V = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(V)
    from types import SimpleNamespace

    a = SimpleNamespace(
        vision_patch_size=va.patch,
        vision_dim=va.vdim,
        vision_n_heads=va.n_heads,
        vision_inter_dim=va.inter,
        vision_rope_theta=va.theta,
        vision_n_layers=n_layers,
        vision_downsample_ratio=va.ratio,
        dim=va.dim,
    )
    vit, al = V.ViT(a).to(torch.bfloat16), V.Aligner(a).to(torch.bfloat16)
    sd = {
        k[len("vision.") :]: sh.get(k)
        for k in sh.index
        if k.startswith("vision.") and (not k.startswith("vision.blocks.") or int(k.split(".")[2]) < n_layers)
    }
    vit.load_state_dict(
        {k: v.to(torch.bfloat16) if v.dtype != torch.float32 else v for k, v in sd.items()}, strict=True
    )
    al.load_state_dict(
        {k[len("aligner.") :]: sh.get(k).to(torch.bfloat16) for k in sh.index if k.startswith("aligner.")}, strict=True
    )
    return vit.eval(), al.eval()


def _image(w, h, seed=0):
    rng = np.random.RandomState(seed)
    base = rng.randint(0, 255, (h // 14 + 1, w // 14 + 1, 3)).astype(np.uint8)
    return Image.fromarray(base).resize((w, h), Image.BICUBIC)


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [pytest.param({"l1_small_size": 16384, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}, id="ring")],
    indirect=True,
)
@torch.no_grad()
def test_vision_vs_reference(mesh_device):
    md = mesh_device
    va = VisionArgs()
    n_layers = int(os.environ.get("VIS_LAYERS", va.n_layers))
    va.n_layers = n_layers
    sh = _Shards()
    vit, al = _ref_modules(va, n_layers, sh)
    vision = DSV41Vision(md, mesh_4x8(), CCLManager(md, num_links=2, topology=ttnn.Topology.Ring), args=va, shards=sh)
    sizes = [tuple(int(x) for x in s.split("x")) for s in os.environ.get("VIS_SIZES", "280x196,588x420").split(",")]
    for w, h in sizes:
        (img,) = VP.image_inputs_from_pil([{"data": _to_bytes(_image(w, h))}], va)
        n_h, n_w = img.n_vit_h, img.n_vit_w
        ref_f = vit(img.patches, n_h, n_w)
        ref = al(ref_f, n_h, n_w).float()
        import time

        vision.encode(img.patches, n_h, n_w)  # warm (compile)
        t0 = time.perf_counter()
        got = vision.encode(img.patches, n_h, n_w).float()
        dt = time.perf_counter() - t0
        assert got.shape == ref.shape, (got.shape, ref.shape)
        pcc = R.pcc(got, ref)
        print(
            f"VISION {w}x{h}: grid {n_h}x{n_w} -> {tuple(ref.shape)} tokens, PCC {pcc:.5f}, device+host {dt*1e3:.0f} ms, max|diff| {(got-ref).abs().max():.3f}, ref absmax {ref.abs().max():.2f}",
            flush=True,
        )
        assert pcc > 0.99, pcc


def _to_bytes(im):
    import io

    b = io.BytesIO()
    im.save(b, format="PNG")
    return b.getvalue()
