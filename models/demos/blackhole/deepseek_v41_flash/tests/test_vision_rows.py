# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Row-batched image encoder (tt/vision.py ``encode_many``: the 4 mesh rows encode 4 different images at once, the 3x3 unfold is an on-device row gather):
  (a) every image of a batch of mixed sizes vs the CPU reference (checkpoint ``inference/vision.py``): PCC,
  (b) an image encoded alone == the same image inside a batch whose images fall in the same padded bucket: bit-identical,
  (c) time of 1, 2, 4 and 8 images (8 = two waves), eager and (VIS_TRACE=1) traced.

    VIS_LAYERS (all 32), VIS_TRACE (0/1), VIS_REPEAT (5)
"""

import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference import ref_layer as R
from models.demos.blackhole.deepseek_v41_flash.tests.test_vision_pcc import _image, _ref_modules
from models.demos.blackhole.deepseek_v41_flash.tt import vision_prompt as VP
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import _Shards
from models.demos.blackhole.deepseek_v41_flash.tt.vision import DSV41Vision, VisionArgs
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager

SIZES = [(640, 480), (1100, 700), (300, 900), (800, 800), (1400, 1000), (600, 400), (700, 700), (500, 1200)]


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {"l1_small_size": 16384, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 400000000},
            id="ring",
        )
    ],
    indirect=True,
)
@torch.no_grad()
def test_vision_rows(mesh_device):
    import traceback

    try:
        _body(mesh_device)
    except BaseException:
        traceback.print_exc()
        raise


def _body(md):
    va = VisionArgs()
    n_layers = int(os.environ.get("VIS_LAYERS", va.n_layers))
    va.n_layers = n_layers
    sh = _Shards()
    vit, al = _ref_modules(va, n_layers, sh)
    vision = DSV41Vision(md, mesh_4x8(), CCLManager(md, num_links=2, topology=ttnn.Topology.Ring), args=va, shards=sh)
    t0 = time.perf_counter()
    vision.warmup()
    print(f"warmup {time.perf_counter() - t0:.1f} s", flush=True)
    if os.environ.get("VIS_TRACE", "0") == "1":
        t0 = time.perf_counter()
        vision.capture_traces()
        print(f"capture_traces {time.perf_counter() - t0:.1f} s, {len(vision.traces)} traces", flush=True)
    imgs = []
    for i, (w, h) in enumerate(SIZES):
        patches, n_h, n_w, _, _ = VP.patchify_pil(_image(w, h, i), va)
        imgs.append((patches, n_h, n_w))
    refs = [al(vit(p, h, w), h, w).float() for p, h, w in imgs]

    # (a) mixed sizes, 8 images = two waves
    got = vision.encode_many(imgs)
    for i, (g, r) in enumerate(zip(got, refs)):
        pcc = R.pcc(g.float(), r)
        print(f"image {i} {SIZES[i]} grid {imgs[i][1]}x{imgs[i][2]}: PCC {pcc:.5f}", flush=True)
        assert g.shape == r.shape and pcc > 0.985, (i, pcc)
    for i in (7, 3):  # each alone (a different padded bucket than in the wave)
        pcc = R.pcc(vision.encode(*imgs[i]).float(), refs[i])
        print(f"image {i} alone: PCC {pcc:.5f}", flush=True)

    old_path = os.environ.get("VIS_OLD")
    if old_path:  # the first (host unfold) tower on the same image
        import importlib.util

        spec = importlib.util.spec_from_file_location("dsv41_vision_old", old_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        old = mod.DSV41Vision(
            md, mesh_4x8(), CCLManager(md, num_links=2, topology=ttnn.Topology.Ring), args=va, shards=sh
        )
        for i in (7, 3):
            print(f"OLD tower image {i}: PCC {R.pcc(old.encode(*imgs[i]).float(), refs[i]):.5f}", flush=True)
    # (b) alone == in a batch of the same padded bucket (bit-identical); in a batch with a larger bucket: same up to rounding
    same = [imgs[0], imgs[5], imgs[0], imgs[5]]  # 1610 / 1536 patches: bucket 2048 alone and in the batch
    alone = [vision.encode_many([x])[0] for x in same[:2]]
    batch = vision.encode_many(same)
    for i, (a_, b_) in enumerate(zip(alone, batch[:2])):
        print(f"alone vs batch (same bucket) image {i}: bit-equal {torch.equal(a_, b_)}", flush=True)
        assert torch.equal(a_, b_)
    mixed = vision.encode_many([imgs[0], imgs[3], imgs[6], imgs[5]])  # n up to 3364: bucket 4096
    for i, j in ((0, 0), (3, 1), (6, 2), (5, 3)):
        p_ = R.pcc(vision.encode_many([imgs[i]])[0].float(), mixed[j].float())
        print(f"alone vs batch (larger bucket) image {i}: PCC {p_:.5f}", flush=True)
        assert p_ > 0.999

    # (c) timing: k copies of one image (the padded bucket stays the same: pure scaling over the mesh rows), then a mix of sizes
    rep = int(os.environ.get("VIS_REPEAT", "5"))
    for name, pool in (
        ("1610-patch image x k", [imgs[0]] * 8),
        ("mixed sizes", [imgs[i] for i in (0, 3, 6, 5, 4, 1, 2, 7)]),
        ("7200-patch image x k", [imgs[4]] * 8),
    ):
        for k in (1, 2, 4, 8):
            batch = pool[:k]
            vision.encode_many(batch)
            t0 = time.perf_counter()
            for _ in range(rep):
                vision.encode_many(batch)
            dt = (time.perf_counter() - t0) / rep
            print(
                f"TIMING {name}: {k} image(s): {dt * 1e3:.0f} ms per call, {dt * 1e3 / k:.0f} ms per image", flush=True
            )
