# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""``DeepSeekV41Vision`` (ViT + aligner, TP4 on one 1x4 stage) against the checkpoint's ``inference/vision.py``.

Synthetic images of a few sizes go through the checkpoint's image processor: grids from 33x46 to 93x93 patches,
wide and tall, most not a multiple of the 3x3 downsample (the aligner pads them). An image is a random colour
per cell, upscaled bicubically, as metal_2's test does. Per-pixel noise is not a usable input: the ViT turns a
few tokens into high-norm outliers around block 12, and on noise which tokens they are flips on bf16-level
differences, so the device and the reference part ways although every block agrees on the same input.

The reference runs on the CPU in fp32. Each image also logs where its error lies: the ViT features, the device
aligner on the reference features, and the tokens whose 3x3 window lies inside the image against those at its
edge. The first image also checks the delimiter rows of ``image_rows``.

``test_vision_traced`` checks the traced tower and row write against the eager ones.

    pytest models/experimental/deepseek_v41_flash/tests/test_vision.py
"""

import importlib.util
import io
import os

import numpy as np
import pytest
import torch
from loguru import logger
from PIL import Image

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.deepseek_v41_flash.tt.config import (
    CACHE_DIR,
    DEFAULT_MODEL_DIR,
    VISION_STAGE,
    dequantized,
    load_vision_config,
)
from models.experimental.deepseek_v41_flash.tt.vision import (
    IMAGE,
    SDPA_CHUNK,
    VIT_BUCKETS,
    DeepSeekV41Vision,
    EmbeddingRowWriter,
    _bucket,
    image_processor,
)
from models.experimental.deepseek_v4_flash.demo.chat_cli import open_mesh_device
from models.experimental.deepseek_v4_flash.tt.weight_cache import WeightCache
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader, resolve_snapshot_dir

# metal_2's bar for the same tower (TP8, against a bf16 reference).
PCC = 0.99
TP_SIZE = 4
# (width, height) in pixels -> ViT grids 33x46, 43x58, 29x115, 93x29, 93x93, 81x108.
IMAGE_SIZES = ((588, 420), (800, 600), (1600, 400), (400, 1300), (1302, 1302), (2000, 1500))


def _snapshot():
    try:
        return resolve_snapshot_dir(DEFAULT_MODEL_DIR)
    except FileNotFoundError:
        return None


def _reference(snapshot, loader, vision_config):
    """fp32 ``ViT`` and ``Aligner`` of ``inference/vision.py`` with the checkpoint's weights."""
    spec = importlib.util.spec_from_file_location("deepseek_v41_vision", snapshot / "inference" / "vision.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    v = vision_config
    args = type(
        "Args",
        (),
        dict(
            vision_dim=v.hidden_size,
            vision_n_heads=v.num_attention_heads,
            vision_inter_dim=v.intermediate_size,
            vision_patch_size=v.patch_size,
            vision_rope_theta=float(v.rope_theta),
            vision_n_layers=v.num_hidden_layers,
            vision_downsample_ratio=v.downsample_ratio,
            dim=v.text_hidden_size,
        ),
    )
    vit, aligner = module.ViT(args), module.Aligner(args)
    for prefix, net in (("vision.", vit), ("aligner.", aligner)):
        net.load_state_dict({k: dequantized(loader, prefix + k)() for k in net.state_dict()})
        net.eval()
    return vit, aligner


def _image_png(width: int, height: int, seed: int) -> bytes:
    """A random colour per 14 x 14 cell, upscaled bicubically to ``width`` x ``height``, as PNG bytes."""
    cells = np.random.RandomState(seed).randint(0, 255, (height // 14 + 1, width // 14 + 1, 3)).astype(np.uint8)
    png = io.BytesIO()
    Image.fromarray(cells).resize((width, height), Image.BICUBIC).save(png, format="PNG")
    return png.getvalue()


@pytest.mark.skipif(_snapshot() is None, reason=f"V4.1-Flash checkpoint not found under {DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(3600)
@torch.no_grad()
def test_vision():
    snapshot = _snapshot()
    vision_config = load_vision_config(snapshot)
    loader = DeepseekV4WeightLoader(snapshot)
    vit, aligner = _reference(snapshot, loader, vision_config)
    # The whole system, opened as the server opens it (rows x TP4, the system profile's fabric), then the vision stage.
    chips = ttnn._ttnn.multi_device.SystemMeshDescriptor().shape().mesh_size()
    rows = chips // TP_SIZE
    if VISION_STAGE >= rows:
        pytest.skip(f"vision stage {VISION_STAGE} is past the {rows} mesh rows of a {chips}-chip system")
    with open_mesh_device(None, mesh_shape=(rows, TP_SIZE)) as mesh_device:
        device = mesh_device.create_submesh(ttnn.MeshShape(1, TP_SIZE), ttnn.MeshCoordinate(VISION_STAGE, 0))
        tower = DeepSeekV41Vision(vision_config, loader, device, cache=WeightCache(CACHE_DIR) if CACHE_DIR else None)
        _check(tower, vit, aligner, snapshot, vision_config)


def _diagnose(tower, patches, n_h, n_w, want_features, want, got):
    """Log where an encode loses accuracy: the ViT features (and whether the padded rows the unfold
    reads are zero), the device aligner on the reference features, and the tokens whose 3x3 window
    lies inside the image against those that reach past its edge."""
    n, r = n_h * n_w, tower.config.downsample_ratio
    n_pad = _bucket(n + 1, VIT_BUCKETS, SDPA_CHUNK)
    features_dev = tower.vit(patches, n_h, n_w, n_pad)
    features = ttnn.to_torch(ttnn.get_device_tensors(features_dev)[0]).float().reshape(n_pad, -1)
    ttnn.deallocate(features_dev)
    _, vit_pcc = comp_pcc(want_features, features[:n])
    tail = features[n:].abs().max().item()

    padded = torch.zeros(n_pad, want_features.shape[-1])
    padded[:n] = want_features
    given = tower._upload(padded, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    aligned_dev = tower.aligner(given, n_h, n_w)
    ttnn.deallocate(given)
    aligned = ttnn.to_torch(ttnn.get_device_tensors(aligned_dev)[0]).float()[0, 0, : want.shape[0]]
    ttnn.deallocate(aligned_dev)
    _, aligner_pcc = comp_pcc(want, aligned)

    th, tw = -(-n_h // r), -(-n_w // r)
    edge = torch.zeros(th, tw, dtype=torch.bool)
    if n_h % r:
        edge[-1] = True
    if n_w % r:
        edge[:, -1] = True
    edge = edge.flatten()
    parts = {"interior": ~edge, "edge": edge}
    token_pcc = {k: comp_pcc(want[m], got[m])[1] for k, m in parts.items() if m.any()}
    logger.info(
        f"  {n_h}x{n_w}: ViT features PCC {vit_pcc}, padded rows max |x| {tail}, "
        f"aligner on the reference features PCC {aligner_pcc}, tokens {token_pcc}"
    )


def _check(tower, vit, aligner, snapshot, vision_config):
    """Encode the synthetic images on ``tower`` against the CPU reference."""
    processor = image_processor(snapshot)
    proc_args = type(
        "Args",
        (),
        dict(
            vision_patch_size=vision_config.patch_size,
            vision_downsample_ratio=vision_config.downsample_ratio,
            vision_max_n_token=vision_config.max_image_tokens,
            vision_min_pixels=vision_config.min_pixels,
            vision_max_wh_ratio=vision_config.max_wh_ratio,
        ),
    )
    failures = []
    for i, (width, height) in enumerate(IMAGE_SIZES):
        patches, n_h, n_w, n_llm_h, n_llm_w = processor.load_image({"data": _image_png(width, height, i)}, proc_args)
        types = processor.image_token_types(n_llm_h, n_llm_w)
        want_features = vit(patches.float(), n_h, n_w)
        want = aligner(want_features, n_h, n_w)
        rows = tower.image_rows(patches, n_h, n_w, types).float()
        got = rows[types == IMAGE]
        _, pcc = comp_pcc(want, got)
        logger.info(f"image {width}x{height} -> {n_h}x{n_w} patches, span of {types.numel()} rows: PCC {pcc}")
        _diagnose(tower, patches, n_h, n_w, want_features, want, got)
        if got.shape != want.shape or pcc < PCC:
            failures.append(
                f"{width}x{height} ({n_h}x{n_w}): shape {tuple(got.shape)} vs {tuple(want.shape)}, PCC {pcc}"
            )
        if i == 0:
            for kind, row in tower.delimiters.items():
                if not torch.equal(rows[types == kind], row.float().expand(int((types == kind).sum()), -1)):
                    failures.append(f"delimiter rows of type {kind} differ from the learned vector")
    assert not failures, "; ".join(failures)


@pytest.mark.skipif(_snapshot() is None, reason=f"V4.1-Flash checkpoint not found under {DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(3600)
@torch.no_grad()
def test_vision_traced():
    """The traced tower (:meth:`DeepSeekV41Vision.capture`) and row write (:class:`EmbeddingRowWriter`) against the
    eager ones, replaying with nothing allocated on the device. Two images share the 3072 bucket and two the 9216
    one, so a replay that kept the previous image's inputs shows. ``DEEPSEEK_V4_TRACE_REGION_SIZE`` sizes the trace
    region (default 256 MiB)."""
    snapshot = _snapshot()
    vision_config = load_vision_config(snapshot)
    loader = DeepseekV4WeightLoader(snapshot)
    processor = image_processor(snapshot)
    proc_args = type(
        "Args",
        (),
        dict(
            vision_patch_size=vision_config.patch_size,
            vision_downsample_ratio=vision_config.downsample_ratio,
            vision_max_n_token=vision_config.max_image_tokens,
            vision_min_pixels=vision_config.min_pixels,
            vision_max_wh_ratio=vision_config.max_wh_ratio,
        ),
    )
    images = [
        processor.load_image({"data": _image_png(width, height, i)}, proc_args)[:3]
        for i, (width, height) in enumerate(IMAGE_SIZES)
    ]
    chips = ttnn._ttnn.multi_device.SystemMeshDescriptor().shape().mesh_size()
    rows = chips // TP_SIZE
    if VISION_STAGE >= rows:
        pytest.skip(f"vision stage {VISION_STAGE} is past the {rows} mesh rows of a {chips}-chip system")
    trace_region = int(os.environ.get("DEEPSEEK_V4_TRACE_REGION_SIZE", 256 << 20))
    with open_mesh_device(trace_region, mesh_shape=(rows, TP_SIZE)) as mesh_device:
        device = mesh_device.create_submesh(ttnn.MeshShape(1, TP_SIZE), ttnn.MeshCoordinate(VISION_STAGE, 0))
        tower = DeepSeekV41Vision(vision_config, loader, device, cache=WeightCache(CACHE_DIR) if CACHE_DIR else None)
        tower.warmup()
        eager = [tower.encode(patches, n_h, n_w).float() for patches, n_h, n_w in images]

        vocab, hidden = 64, vision_config.text_hidden_size
        table = ttnn.from_torch(
            torch.zeros(vocab + 2048, hidden),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )
        tower.capture()
        writer = EmbeddingRowWriter(table, vocab)
        allocated = len(ttnn._ttnn.reports.get_buffers(device))

        failures = []
        for (patches, n_h, n_w), want in zip(images, eager):
            got = tower.encode(patches, n_h, n_w).float()
            _, pcc = comp_pcc(want, got)
            diff = (want - got).abs().max().item()
            logger.info(f"{n_h}x{n_w} patches: traced vs eager PCC {pcc}, max |diff| {diff}")
            if got.shape != want.shape or pcc < 0.9999:
                failures.append(f"{n_h}x{n_w}: traced vs eager shape {tuple(got.shape)}, PCC {pcc}")
        rows_in = eager[0][:100]
        ids = vocab + torch.randperm(2048)[:100]
        writer.write(rows_in, ids)
        written = ttnn.to_torch(ttnn.get_device_tensors(table)[0]).float().reshape(-1, hidden)
        if not torch.equal(written[ids], rows_in.bfloat16().float()):
            failures.append("the traced row write put other rows at the ids")
        others = torch.ones(written.shape[0], dtype=torch.bool)
        others[ids] = False
        if written[others].abs().max().item() != 0:
            failures.append("the traced row write touched rows past its ids")
        after = len(ttnn._ttnn.reports.get_buffers(device))
        if after != allocated:
            failures.append(f"{after - allocated} device buffers allocated by the traced replays")
    assert not failures, "; ".join(failures)
