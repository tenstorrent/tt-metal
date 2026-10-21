# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PCC of TtVisionTower (ViT + projector + patch merge on the prefill mesh) vs the fp32 torch reference, real
checkpoint weights.

Opens the prefill runner's mesh (PREFILL_SP x PREFILL_TP, PREFILL_FABRIC_MODE), so launch it the way the runner is
launched, e.g. on the 16p16d Galaxy's prefill half:

    python3 ttnn/ttnn/distributed/ttrun.py --tcp-interface lo --rank-binding <prefill binding.yaml> \
        --mpi-args "--host localhost:1 --bind-to none --tag-output -x PATH -x LD_LIBRARY_PATH" \
        -- python3 -m pytest models/demos/minimax_m3/tests/unit/test_vision_tower_vs_ref.py -s

HF_MODEL (or PREFILL_HF_MODEL) names the checkpoint; M3_VISION_CACHE optionally caches the device weights.
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.common.prefill.runners.runner_utils import open_mesh_device
from models.demos.minimax_m3.config import MeshConfig
from models.demos.minimax_m3.reference.vision import MiniMaxM3VisionReference, VisionArgs, load_vision_state_dict
from models.demos.minimax_m3.tt.ccl import CCLManager
from models.demos.minimax_m3.tt.runners.adapters.minimax_m3 import MiniMaxM3Config
from models.demos.minimax_m3.tt.vision import TtVisionTower
from models.demos.minimax_m3.utils.general_utils import get_default_num_links

CKPT = os.getenv("PREFILL_HF_MODEL") or os.getenv("HF_MODEL")
SP = int(os.getenv("PREFILL_SP", "0"))
TP = int(os.getenv("PREFILL_TP", "0"))
pytestmark = pytest.mark.skipif(not (CKPT and SP and TP), reason="needs HF_MODEL and the prefill mesh env")


@pytest.fixture(scope="module")
def mesh():
    mesh_device = open_mesh_device((SP, TP), MiniMaxM3Config)
    yield mesh_device
    ttnn.close_mesh_device(mesh_device)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.fixture(scope="module")
def models(mesh):
    args = VisionArgs.from_checkpoint(CKPT)
    state = load_vision_state_dict(CKPT)
    tower = TtVisionTower(
        mesh,
        MeshConfig((SP, TP), tp=TP),
        CCLManager(mesh, num_links=get_default_num_links(mesh), topology=ttnn.Topology.Linear),
        state,
        args,
        cache_dir=os.getenv("M3_VISION_CACHE"),
    )
    return tower, MiniMaxM3VisionReference(state, args), args


def _image_patches(path: str, max_long_side: int | None):
    """Processor patches of a real image: the checkpoint's area bound, or a MiniMax long-side bound."""
    from PIL import Image
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(CKPT, trust_remote_code=True).image_processor
    img = Image.open(path).convert("RGB")
    if max_long_side:
        w, h = img.size
        s = max_long_side / max(w, h)
        img = img.resize((max(28, round(w * s / 28) * 28), max(28, round(h * s / 28) * 28)), Image.BICUBIC)
        out = proc(images=[img], return_tensors="pt", max_pixels=max_long_side**2)
    else:
        out = proc(images=[img], return_tensors="pt")
    return out["pixel_values"], [tuple(g) for g in out["image_grid_thw"].tolist()]


def _gather_rows(tt_rows, mesh, rows: int, tp: int):
    """Device 0's row of the mesh holds every row, its TP slice of hidden; stitch the slices."""
    dts = ttnn.get_device_tensors(tt_rows)
    cols = mesh.shape[1]
    return torch.cat([ttnn.to_torch(dts[j]).float().reshape(-1, dts[j].shape[-1]) for j in range(cols)], dim=-1)[:rows]


SAMPLES = os.path.join(os.path.dirname(__file__), "../../../../sample_data")


# the fp32 host reference of a default-tier image takes minutes
@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "case",
    ["cat_low", "house_default", "batch_with_video"],
)
def test_vision_tower(models, mesh, case):
    tower, ref, args = models
    torch.manual_seed(0)
    if case == "cat_low":
        pv, grid = _image_patches(os.path.join(SAMPLES, "huggingface_cat_image.jpg"), None)
    elif case == "house_default":
        pv, grid = _image_patches(os.path.join(SAMPLES, "house_in_field_1080p.jpg"), 2016)
    else:
        pv1, g1 = _image_patches(os.path.join(SAMPLES, "huggingface_cat_image.jpg"), None)
        g_video = (args.segment_max_frames + 2, 12, 16)
        pv_video = torch.randn(g_video[0] * g_video[1] * g_video[2], args.patch_elems) * 0.8
        pv = torch.cat([pv1, pv_video, pv1[: 8 * 8]])
        grid = g1 + [g_video, (1, 8, 8)]

    batch, rows = tower(pv, grid)
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    rows.deallocate(True)
    batch, rows = tower(pv, grid)
    ttnn.synchronize_device(mesh)
    ms = (time.perf_counter() - t0) * 1000
    got = _gather_rows(rows, mesh, batch.rows, TP)
    expected = ref(pv.float(), grid)
    logger.info(f"{case}: {batch.patches} patches ({batch.local}/device), {batch.rows} rows, {ms:.1f} ms warm")
    passing, pcc = comp_pcc(expected, got, 0.99)
    logger.info(f"{case}: PCC {pcc}")
    assert passing, pcc


def _write_item(root, name: str, pv: torch.Tensor, grid):
    import struct

    header = struct.pack("<4s8I", b"TMM1", 1, 0, pv.shape[0], pv.shape[1], 1, *grid).ljust(64, b"\0")
    with open(os.path.join(root, f"{name}.px"), "wb") as f:
        f.write(header + pv.to(torch.bfloat16).view(torch.uint16).numpy().tobytes())


def test_media_rows_at_positions(models, mesh, tmp_path, monkeypatch):
    """Each chunk's embedding gets exactly the tower's rows at the manifest's positions, and nothing elsewhere,
    including an item split by a chunk boundary and a chunk that starts mid-slab."""
    import json

    from models.demos.common.prefill.chunk_layout import rotated_chunk_positions
    from models.demos.minimax_m3.tt.media import MediaRows

    tower, _ref, args = models
    monkeypatch.setenv("TT_M3_MM_DIR", str(tmp_path))
    chunk = 5120
    chunk_local = chunk // SP
    grids = {"a" * 16: (1, 20, 24), "b" * 16: (1, 16, 16)}
    pvs = {h: torch.randn(t * hh * w, args.patch_elems) for h, (t, hh, w) in grids.items()}
    for h in grids:
        _write_item(str(tmp_path), h, pvs[h], grids[h])
    # item a: 120 rows starting 40 rows before the first chunk boundary; item b: 64 rows in the second chunk
    runs = {"a" * 16: [[chunk - 40, 0, 120]], "b" * 16: [[chunk + 1000, 0, 64]]}
    manifest = {
        "version": 1,
        "items": [{"hash": h, "kind": "image", "grid": list(grids[h]), "runs": runs[h]} for h in grids],
    }
    tag = 0x1234
    with open(tmp_path / f"req_{tag:016x}.json", "w") as f:
        json.dump(manifest, f)
    expected_rows = {}
    for h in grids:
        batch, rows = tower(pvs[h], [grids[h]])
        expected_rows[h] = _gather_rows(rows, mesh, batch.rows, TP)
        rows.deallocate(True)

    media = MediaRows(tower, mesh, 0, chunk)
    for start, end in ((0, chunk), (chunk, 2 * chunk), (chunk + 2048, chunk + 4096)):
        zeros = torch.zeros(SP, 1, chunk_local, args.text_hidden, dtype=torch.bfloat16)
        x = ttnn.from_torch(
            zeros.reshape(SP, 1, chunk_local, args.text_hidden),
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=mesh.shape, dims=(0, 3)),
        )
        x = media.add(x, tag, start, end)
        dts = ttnn.get_device_tensors(x)
        got = torch.stack(
            [
                torch.cat([ttnn.to_torch(dts[r * TP + j]).float().reshape(chunk_local, -1) for j in range(TP)], -1)
                for r in range(SP)
            ]
        )
        want = torch.zeros_like(got)
        for c, chip_pos in enumerate(rotated_chunk_positions(start, SP, chunk_local)):
            for r, p in enumerate(chip_pos):
                for h, rr in runs.items():
                    pos, off, cnt = rr[0]
                    if pos <= p < pos + cnt and start <= p < end:
                        want[c, r] = expected_rows[h][off + p - pos]
        hit = want.abs().sum(-1) != 0
        assert torch.count_nonzero(got[~hit]) == 0, f"chunk [{start}, {end}): rows outside the media positions"
        if hit.any():
            # one tower call for both items here, one each for `expected_rows`: same math, other accumulation order
            passing, pcc = comp_pcc(want[hit], got[hit], 0.999)
            assert passing, f"chunk [{start}, {end}): {pcc}"
    media.release()

    # 32 x 32 = 1024 patches alone fill a bucket exactly: the encoded rows need no padding slice.
    h, grid = "c" * 16, (1, 32, 32)
    pv = torch.randn(32 * 32, args.patch_elems)
    _write_item(str(tmp_path), h, pv, grid)
    with open(tmp_path / "req_0000000000005678.json", "w") as f:
        json.dump(
            {"version": 1, "items": [{"hash": h, "kind": "image", "grid": list(grid), "runs": [[300, 0, 256]]}]}, f
        )
    x = ttnn.from_torch(
        torch.zeros(SP, 1, chunk_local, args.text_hidden, dtype=torch.bfloat16),
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=mesh.shape, dims=(0, 3)),
    )
    x = media.add(x, 0x5678, 0, 1000)
    assert media.batches[0]["zero_row"] == 256
    media.release()


def test_media_rows_split_video(models, mesh, tmp_path, monkeypatch):
    """A video larger than one tower call is encoded in pieces of whole attention segments, rows unchanged."""
    import json

    from models.demos.common.prefill.chunk_layout import rotated_chunk_positions
    from models.demos.minimax_m3.tt.media import MediaRows

    tower, _ref, args = models
    monkeypatch.setenv("TT_M3_MM_DIR", str(tmp_path))
    chunk = 5120
    chunk_local = chunk // SP
    h, grid = "d" * 16, (2 * args.segment_max_frames, 16, 16)
    pv = torch.randn(grid[0] * 256, args.patch_elems)
    _write_item(str(tmp_path), h, pv, grid)
    m = 16 * 16 // 4
    runs = [[100 + k * (m + 10), k * m, m] for k in range(grid[0])]
    with open(tmp_path / "req_0000000000009abc.json", "w") as f:
        json.dump({"version": 1, "items": [{"hash": h, "kind": "video", "grid": list(grid), "runs": runs}]}, f)
    batch, rows = tower(pv, [grid])
    expected = _gather_rows(rows, mesh, batch.rows, TP)
    rows.deallocate(True)

    media = MediaRows(tower, mesh, 0, chunk)
    media.max_patches = args.segment_max_frames * 256
    x = ttnn.from_torch(
        torch.zeros(SP, 1, chunk_local, args.text_hidden, dtype=torch.bfloat16),
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=mesh.shape, dims=(0, 3)),
    )
    x = media.add(x, 0x9ABC, 0, chunk)
    assert len(media.batches) == 2, len(media.batches)
    dts = ttnn.get_device_tensors(x)
    got = torch.stack(
        [
            torch.cat([ttnn.to_torch(dts[r * TP + j]).float().reshape(chunk_local, -1) for j in range(TP)], -1)
            for r in range(SP)
        ]
    )
    want = torch.zeros_like(got)
    for c, chip_pos in enumerate(rotated_chunk_positions(0, SP, chunk_local)):
        for r, p in enumerate(chip_pos):
            for pos, off, cnt in runs:
                if pos <= p < pos + cnt:
                    want[c, r] = expected[off + p - pos]
    hit = want.abs().sum(-1) != 0
    assert torch.count_nonzero(got[~hit]) == 0
    passing, pcc = comp_pcc(want[hit], got[hit], 0.999)
    assert passing, pcc
    media.release()
