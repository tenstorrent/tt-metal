# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The TTNN ViT (tt/vision) on one 1x8 row vs the fp32 reference (reference/vision.py), real weights.

Inputs come from vLLM's processor (reference/vision_inputs.py) on the repo's sample images: a 448x448 image, a
640x360 image, a 1024x1024 image, two images in one call, and a 32-frame 640x360 video (also split into several
passes); bf16 and bfloat8_b weights. Checks the PCC of every block's output (row order) and of the final embeddings,
that padding a pass to its row bucket leaves the real rows alone, that ``warm()`` compiles every program a call runs,
and the encoder backend on a busy device.

Run on one Galaxy row, as a row engine does:
    TT_VISIBLE_DEVICES=<the row's 8 chips> pytest models/experimental/mimo_v2_6_flash/tests/test_vision_tp8.py
The converted weights go to ``MIMO_VISION_CACHE`` (default: a ``vision`` cache next to the LLM's, per checkpoint).
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.mimo_v2_6_flash.reference import vision as ref
from models.experimental.mimo_v2_6_flash.reference import vision_inputs as vin
from models.experimental.mimo_v2_6_flash.tests.common import GALAXY_DEVICE_PARAMS, model_path, pcc
from models.experimental.mimo_v2_6_flash.tt.vision.encoder import MiMoVisionEncoder
from models.experimental.mimo_v2_6_flash.tt.vision.vit import (
    NUMERICS,
    ROW_BUCKETS,
    PassGeometry,
    TtMiMoVisionTransformer,
    plan_passes,
)

BLOCK_PCC = 0.99
FINAL_PCC = 0.99


def vision_cache_dir():
    root = os.environ.get("MIMO_VISION_CACHE", "/home/zni/mimo/mm/vision/tt-cache")
    return os.path.join(root, os.path.basename(os.path.normpath(model_path())))


def make_case(path, case):
    """(pixel_values, grid_thw, num_tokens) from the vLLM processor."""
    if case == "image_448":
        out = vin.process_images(path, [vin.sample_image(size=(448, 448))])
    elif case == "image_640x480":
        out = vin.process_images(path, [vin.sample_image(size=(640, 480))])
    elif case == "tiny_image":
        out = vin.process_images(path, [vin.sample_image(size=(64, 48))])
    elif case == "image_1024":
        out = vin.process_images(path, [vin.sample_image("demo.jpeg", size=(1024, 1024))])
    elif case == "image_640x360":
        out = vin.process_images(path, [vin.sample_image("house_in_field_1080p.jpg", size=(640, 360))])
    elif case == "two_images":
        images = [vin.sample_image(size=(448, 448)), vin.sample_image("demo.jpeg", size=(512, 384))]
        out = vin.process_images(path, images)
    elif case.startswith("video_32f"):
        out = vin.process_video(path, vin.sample_video(num_frames=32, size=(640, 360)))
        return out["pixel_values_videos"], out["video_grid_thw"], out["num_tokens"]
    else:
        raise ValueError(case)
    return out["pixel_values"], out["image_grid_thw"], out["num_tokens"]


@pytest.fixture(scope="module")
def reference():
    torch.set_num_threads(min(16, torch.get_num_threads()))
    path = model_path()
    return path, ref.MiMoVisionReference.from_model_path(path)


@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "case, max_rows, weights",
    [
        ("image_448", None, "bf16"),
        ("image_640x360", None, "bf16"),
        ("image_1024", None, "bf16"),
        ("two_images", None, "bf16"),
        ("video_32f", None, "bf16"),
        ("video_32f_passes", 4096, "bf16"),
        ("image_448", None, "bfp8"),
        ("image_1024", None, "bfp8"),
        ("two_images", None, "bfp8"),
        ("video_32f", None, "bfp8"),
    ],
)
def test_vision_tp8(mesh_device, reference, case, max_rows, weights):
    path, model_ref = reference
    pixel_values, grid, num_tokens = make_case(path, case)
    cfg = model_ref.cfg
    n_tokens = ref.num_merged_tokens(grid, cfg.spatial_merge_size)
    assert n_tokens == sum(num_tokens), (n_tokens, num_tokens)

    t0 = time.time()
    with torch.no_grad():
        expected, expected_hidden = model_ref(pixel_values, grid, return_hidden=True)
    logger.info(
        f"{case}: grid {ref.as_grid_list(grid)}, {pixel_values.shape[0]} patches, reference {time.time() - t0:.1f} s"
    )

    t0 = time.time()
    kwargs = {} if max_rows is None else {"max_rows": max_rows}
    kwargs["weight_dtype"] = ttnn.bfloat8_b if weights == "bfp8" else ttnn.bfloat16
    model = TtMiMoVisionTransformer.from_checkpoint(mesh_device, path, cache_dir=vision_cache_dir(), **kwargs)
    logger.info(f"weights loaded in {time.time() - t0:.1f} s")

    hidden = []
    out = model(pixel_values, grid, hidden=hidden if max_rows is None else None)
    assert out.shape == (n_tokens, cfg.out_hidden_size) and out.dtype == torch.bfloat16, out.shape
    final = pcc(out.float(), expected)
    if hidden:
        assert len(hidden) == cfg.depth
        scores = [pcc(h, e) for h, e in zip(hidden, expected_hidden)]
        table = " ".join(f"{i}:{s:.4f}" for i, s in enumerate(scores))
        logger.info(f"{case} ({weights}): per-block PCC {table}")
        worst = min(range(cfg.depth), key=lambda i: scores[i])
        logger.info(f"{case} ({weights}): min block PCC {scores[worst]:.5f} (block {worst}), final PCC {final:.5f}")
        assert scores[worst] >= BLOCK_PCC, f"block {worst} PCC {scores[worst]:.5f}"
    else:
        logger.info(f"{case}: final PCC {final:.5f} ({max_rows} rows per pass)")
    assert final >= FINAL_PCC, f"final PCC {final:.5f}"
    ttnn.synchronize_device(mesh_device)


def dram_allocated(mesh_device):
    mv = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
    return mv.total_bytes_allocated_per_bank * mv.num_banks


@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("l1_kb_per_core", [0, 384])
def test_vision_encoder_busy_device(mesh_device, reference, l1_kb_per_core):
    """The encoder backend (MiMoVisionEncoder) as the LLM calls it between its steps: L1 partly held by
    long-lived buffers (decode keeps some), images and a video back to back, nothing left allocated after a call
    (the first call of a shape keeps only its programs' binaries, so each case runs once before the check)."""
    path, model_ref = reference
    grid_size = mesh_device.compute_with_storage_grid_size()
    cores = grid_size.x * grid_size.y
    held = None
    if l1_kb_per_core:
        tiles_per_core = l1_kb_per_core * 1024 // (32 * 32 * 2)
        held = ttnn.from_torch(
            torch.zeros(1, 1, 32 * tiles_per_core, 32 * cores),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
    encoder = MiMoVisionEncoder(mesh_device, path, cache_dir=vision_cache_dir())
    for case in ("image_1024", "video_32f", "two_images"):
        pixel_values, grid, _ = make_case(path, case)
        embed = encoder.embed_video if case.startswith("video") else encoder.embed_image
        embed(pixel_values, grid)
        before = dram_allocated(mesh_device)
        out = embed(pixel_values, grid)
        assert dram_allocated(mesh_device) == before, "the call left device tensors allocated"
        with torch.no_grad():
            expected = model_ref(pixel_values, grid)
        score = pcc(out.float(), expected)
        logger.info(f"{case} with {l1_kb_per_core} KB of L1 held per core: final PCC {score:.5f}")
        assert score >= FINAL_PCC
    if held is not None:
        held.deallocate(True)


@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("case", ["image_448", "image_640x480", "image_1024", "two_images", "video_32f"])
def test_vision_bucket_padding_is_isolated(mesh_device, reference, case):
    """Padding a pass up to its row bucket is a masked window of its own, dropped from the output:

    * isolation: garbage instead of zeros in the padding rows leaves the real embeddings bit for bit;
    * vs no buckets (each pass padded only to a multiple of 128): the real embeddings agree to rounding (other row
      counts pick other op configurations, whose rounding block 27's massive activations amplify), and both are as
      close to the fp32 reference."""
    path, model_ref = reference
    pixel_values, grid, _ = make_case(path, case)
    cache = vision_cache_dir()
    bucketed = TtMiMoVisionTransformer.from_checkpoint(mesh_device, path, cache_dir=cache)
    passes = plan_passes(grid, bucketed.max_rows)
    out_bucket = bucketed(pixel_values, grid).float()

    g = torch.Generator().manual_seed(1)
    garbage_out = []
    for start, end, grids in passes:
        geo = PassGeometry.build(grids, bucketed.buckets)
        junk = torch.randn(geo.rows - geo.n, pixel_values.shape[-1], generator=g) * 3
        rows = torch.cat([pixel_values[start:end].float(), junk])
        garbage_out.append(bucketed.forward_pass(rows, geo).float())
    garbage_out = torch.cat(garbage_out)
    assert torch.equal(garbage_out, out_bucket), "padding rows leak into the real rows"

    exact = TtMiMoVisionTransformer.from_checkpoint(mesh_device, path, cache_dir=cache, buckets=None)
    out_exact = exact(pixel_values, grid).float()
    with torch.no_grad():
        expected = model_ref(pixel_values, grid)
    rows_exact = [PassGeometry.build(g, None).rows for _, _, g in passes]
    rows_bucket = [PassGeometry.build(g, ROW_BUCKETS).rows for _, _, g in passes]
    agree = pcc(out_bucket, out_exact)
    logger.info(
        f"{case}: rows {rows_exact} -> buckets {rows_bucket}: garbage padding bit-identical; vs no buckets: "
        f"identical {torch.equal(out_bucket, out_exact)}, PCC {agree:.6f}; vs fp32: no buckets "
        f"{pcc(out_exact, expected):.5f}, buckets {pcc(out_bucket, expected):.5f}"
    )
    assert agree >= 0.999 and pcc(out_bucket, expected) >= FINAL_PCC


@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
def test_vision_warm_bounds_programs(mesh_device, reference):
    """After ``warm()``, inputs of any size compile no program and leave no DRAM allocated."""
    path, _ = reference
    encoder = MiMoVisionEncoder(mesh_device, path, cache_dir=vision_cache_dir())
    programs, dram = mesh_device.num_program_cache_entries(), dram_allocated(mesh_device)
    t0 = time.time()
    per_bucket = encoder.warm()
    warm_s = time.time() - t0
    warm_programs = mesh_device.num_program_cache_entries() - programs
    warm_dram = dram_allocated(mesh_device) - dram
    logger.info(
        f"warm(): {warm_s:.1f} s for buckets {list(per_bucket)} "
        f"({', '.join(f'{r}: {t:.2f} s' for r, t in per_bucket.items())}); {warm_programs} programs, "
        f"{warm_dram / 1e6:.2f} MB DRAM per chip"
    )
    programs, dram = mesh_device.num_program_cache_entries(), dram_allocated(mesh_device)
    for case in ("tiny_image", "image_448", "image_640x480", "image_1024", "two_images", "video_32f"):
        pixel_values, grid, _ = make_case(path, case)
        embed = encoder.embed_video if case.startswith("video") else encoder.embed_image
        embed(pixel_values, grid)
        new = mesh_device.num_program_cache_entries() - programs
        assert new == 0, f"{case} compiled {new} programs after warm()"
        assert dram_allocated(mesh_device) == dram, f"{case} left DRAM allocated"


@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("numerics", ["balanced", "accurate"])
def test_vision_deterministic_per_bucket(mesh_device, reference, numerics):
    """Every row bucket (the only pass sizes a call runs) gives bit-identical outputs run to run: guards against
    races between the all-reduces and the ops after them (see VisionNumerics)."""
    path, model_ref = reference
    cfg = model_ref.cfg
    model = TtMiMoVisionTransformer.from_checkpoint(
        mesh_device, path, cache_dir=vision_cache_dir(), numerics=NUMERICS[numerics]
    )
    g = torch.Generator().manual_seed(0)
    for rows in model.buckets:
        grids = [(1, 32, rows // 32)]
        pixels = torch.randn(rows, cfg.patch_dim, generator=g).bfloat16()
        outs = [model(pixels, grids) for _ in range(3)]
        same = [torch.equal(outs[0], o) for o in outs[1:]]
        logger.info(f"{numerics} {rows} rows: runs identical {same}")
        assert all(same), f"{numerics}: {rows}-row bucket differs run to run"
