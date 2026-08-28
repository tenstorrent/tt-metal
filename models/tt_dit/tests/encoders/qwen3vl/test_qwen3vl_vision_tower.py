# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Qwen3-VL vision tower end to end on device against the HF reference: patch
# embed, position embeddings, every block, the output and deepstack mergers, at
# the released depth and only production grid shapes.

import time

import pytest
import torch
import transformers

import ttnn

from ....encoders.qwen3vl.vision_qwen3vl import Qwen3VlVisionModel, pad_patches_for_sp, vision_cu_seqlens
from ....utils import tensor
from ....utils.check import assert_quality
from .common import (
    FABRIC,
    HIDDEN_ACT,
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    NORM_EPS,
    NUM_HEADS,
    NUM_POSITION_EMBEDDINGS,
    OUT_HIDDEN_SIZE,
    SPATIAL_MERGE_SIZE,
    VISION_PARAMS,
    resolve_parallel,
    skip_if_sp_misaligned,
    sp_shard,
    vision_config,
)

DEPTH = 27  # released tower: 27 blocks, deepstack taps at 8/16/24
DEEPSTACK_INDEXES = [8, 16, 24]

# Grids measured through the checkpoint's own image processor; orientation pairs catch h-vs-w swaps.
# The 4032-patch canvases are 31.5 x 128 rows, so SP=4 skips there (model geometry, not a port bug).
_CANVAS = {
    "canvas_1to1": [[1, 48, 48]],
    "canvas_4to3": [[1, 48, 64]],
    "canvas_3to4": [[1, 64, 48]],
    "canvas_16to9": [[1, 48, 84]],
    "canvas_9to16": [[1, 84, 48]],
    "canvas_4to1": [[1, 32, 126]],  # area cap binds first, short edge forced to 32
    "canvas_1to4": [[1, 126, 32]],
}
_REFERENCE = {
    "ref_1to1": [[1, 128, 128]],
    "ref_4to3": [[1, 128, 170]],
    "ref_3to4": [[1, 170, 128]],
    "ref_16to9": [[1, 128, 228]],
    "ref_9to16": [[1, 228, 128]],
    "ref_4to1": [[1, 128, 512]],
    "ref_1to4": [[1, 512, 128]],
}
# an image is one block, a video one block PER FRAME; a tower ignoring boundaries passes single-block grids
_MULTI = {
    "two_refs": [[1, 128, 128], [1, 128, 170]],
    "video_3_frames": [[3, 48, 48]],
    "image_and_video": [[1, 128, 128], [3, 48, 48]],
    "max_load": [[1, 128, 128]] * 9 + [[3, 48, 48]] * 3,  # documented ceiling: 9 ref images + 3 ref videos
}

GRIDS = {**_CANVAS, **_REFERENCE, **_MULTI}


@pytest.fixture(scope="module")
def reference():
    torch.manual_seed(0)
    config = vision_config(depth=DEPTH, deepstack_visual_indexes=DEEPSTACK_INDEXES)
    return transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLVisionModel._from_config(config).eval()


def _tower(reference, submesh, parallel_config=None, ccl_manager=None):
    tower = Qwen3VlVisionModel(
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        depth=DEPTH,
        intermediate_size=INTERMEDIATE_SIZE,
        spatial_merge_size=SPATIAL_MERGE_SIZE,
        num_position_embeddings=NUM_POSITION_EMBEDDINGS,
        out_hidden_size=OUT_HIDDEN_SIZE,
        hidden_act=HIDDEN_ACT,
        norm_eps=NORM_EPS,
        deepstack_visual_indexes=DEEPSTACK_INDEXES,
        mesh_device=submesh,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
    )
    tower.load_torch_state_dict(reference.state_dict())
    return tower


# perf mode skips the quadratic CPU golden (shapes/finiteness only) and says nothing about accuracy
_PERF_GRIDS = ("max_load", "ref_4to1", "ref_1to4", "image_and_video")
assert all(name in GRIDS for name in _PERF_GRIDS)
# The largest grids' check-mode CPU golden exceeds the default 300 s budget (measured at tp8_sp4:
# ~492 s for the 65k single-block references, ~778 s for max_load's 168k rows). Per-grid timeout
# headroom on the check cases only -- perf mode skips the golden. Sizes from 43682d4.
_TIMEOUTS = {"ref_4to1": 900, "ref_1to4": 900, "max_load": 1800}
_CASES = [
    pytest.param(
        name, True, id=f"check-{name}", marks=[pytest.mark.timeout(_TIMEOUTS[name])] if name in _TIMEOUTS else []
    )
    for name in GRIDS
] + [pytest.param(name, False, id=f"perf-{name}") for name in _PERF_GRIDS]


@VISION_PARAMS
@pytest.mark.parametrize(("name", "check_pcc"), _CASES)
def test_tower_on_device(reference, mesh_device, submesh_shape, tp_axis, sp_axis, num_links, name, check_pcc):
    """The full tower through both outputs: merged tokens and one deepstack feature per tap."""
    submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    grid = torch.tensor(GRIDS[name], dtype=torch.long)
    total = sum(t * h * w for t, h, w in GRIDS[name])
    skip_if_sp_misaligned(total, submesh, sp_axis)
    patch_dim = 3 * 2 * 16 * 16

    torch.manual_seed(0)
    patches = torch.randn(total, patch_dim)
    golden_tokens, golden_deepstack = None, None
    if check_pcc:
        with torch.no_grad():
            ref_out = reference(patches, grid_thw=grid, return_dict=True)
        golden_tokens = ref_out.pooler_output.float()
        golden_deepstack = [f.float() for f in ref_out.deepstack_features]
        assert len(golden_deepstack) == len(DEEPSTACK_INDEXES), "reference did not emit one feature per index"

    cu_seqlens = vision_cu_seqlens(grid)
    assert len(cu_seqlens) - 1 == int(grid[:, 0].sum()), f"expected one block per frame, got {cu_seqlens}"
    assert cu_seqlens[0] == 0 and cu_seqlens[-1] == total, f"cu_seqlens must span [0, {total}]: {cu_seqlens}"

    tower = _tower(reference, submesh, *resolve_parallel(submesh, tp_axis, sp_axis, num_links))
    # `perf` runs the whole iteration twice: the first pass compiles/caches kernels, the second is the
    # measured steady-state run. `check` runs once -- the golden PCC needs no warmup. Idiom:
    # test_transformer_qwenimage.py. Each iteration is timed in two parts, split by a device sync:
    #   prep = per-request host build (rope + position tables) + the host->device upload of every input
    #   op   = the tower forward itself
    # `patches` stays outside (it is the fixed test input feeding the golden, i.e. "pixels already in
    # host memory"); everything else a real request rebuilds per call lives inside the timed region.
    n_iters = 1 if check_pcc else 2
    for iteration in range(n_iters):
        print(f"Tower Forward ({iteration + 1}/{n_iters})")
        ttnn.synchronize_device(submesh)  # device idle before timing
        t_start = time.time()

        # --- prep: host build + host->device upload ---
        cos, sin = tower.prepare_rope(grid)
        pos = tower.prepare_pos_embeds(grid)
        tt_patches = sp_shard(patches, submesh, sp_axis)
        tt_pos = sp_shard(pos, submesh, sp_axis)
        tt_cos, tt_sin = sp_shard(cos, submesh, sp_axis), sp_shard(sin, submesh, sp_axis)
        ttnn.synchronize_device(submesh)  # uploads landed on device
        t_prep_done = time.time()

        # --- op: the tower forward ---
        tokens, deepstack = tower.forward(
            tt_patches,
            pos_embeds=tt_pos,
            rope=(tt_cos, tt_sin),
            cu_seqlens=cu_seqlens,
        )
        ttnn.synchronize_device(submesh)  # forward complete
        t_end = time.time()

        print(
            f"iter {iteration + 1}/{n_iters}: "
            f"prep {(t_prep_done - t_start) * 1000:8.1f} ms (host build + H2D) | "
            f"op {(t_end - t_prep_done) * 1000:8.1f} ms (tower.forward) | "
            f"e2e {(t_end - t_start) * 1000:8.1f} ms"
        )

    merged = total // SPATIAL_MERGE_SIZE**2
    actual_tokens = tensor.to_torch(tokens, mesh_axes=[None, None])
    assert actual_tokens.shape[-2:] == (merged, OUT_HIDDEN_SIZE), f"{tuple(actual_tokens.shape)}"
    assert len(deepstack) == len(DEEPSTACK_INDEXES), f"expected {len(DEEPSTACK_INDEXES)} features"
    features = [tensor.to_torch(f, mesh_axes=[None, None]) for f in deepstack]
    for i, feature in enumerate(features):
        assert feature.shape[-2:] == (merged, OUT_HIDDEN_SIZE), f"deepstack {i}: {tuple(feature.shape)}"

    if check_pcc:
        assert_quality(golden_tokens, actual_tokens, pcc=0.99)
        for golden_feature, feature in zip(golden_deepstack, features):
            assert_quality(golden_feature, feature, pcc=0.99)
    else:
        assert torch.isfinite(actual_tokens).all(), "merged tokens contain NaN or Inf"
        for i, feature in enumerate(features):
            assert torch.isfinite(feature).all(), f"deepstack {i} contains NaN or Inf"

    # Routing by list index can yield the right count and shapes while tapping one layer twice.
    for i in range(len(features) - 1):
        assert not torch.allclose(
            features[i], features[i + 1], atol=1e-2
        ), f"deepstack features {i} and {i + 1} are identical; layer routing is wrong"


# 26,944 patches: 64 short of every SP alignment on this galaxy (%128 == %256 == 64), so both configs
# below exercise pad_patches_for_sp rather than skipping. h/w stay even for the 2x2 merge.
_MISALIGNED_GRID = [[1, 128, 128], [1, 110, 96]]


@pytest.mark.parametrize(
    ("mesh_device", "submesh_shape", "tp_axis", "sp_axis", "num_links", "device_params"),
    [
        pytest.param((8, 4), (8, 4), 0, 1, 2, FABRIC, id="tp8_sp4"),
        pytest.param((4, 8), (4, 8), 0, 1, 2, FABRIC, id="tp4_sp8"),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_tower_sp_padding(reference, mesh_device, submesh_shape, tp_axis, sp_axis, num_links):
    """A patch count that does NOT divide `sp * 32` runs via [`pad_patches_for_sp`]: the pad rows ride
    in a phantom attention window and the merged garbage is trimmed after the SP gather, so the output
    matches the reference on the REAL patches exactly as if nothing was padded.

    This is the multi-galaxy (e.g. 4x32, alignment 1024) enablement, tested on one galaxy: alignment
    is a property of `total % (sp * 32)` alone, so a grid misaligned at sp=4/8 exercises the identical
    path -- phantom window, isolated garbage rows, post-gather trim -- that two_refs (38,144 % 1024
    != 0) hits at SP=32.
    """
    submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    grid = torch.tensor(_MISALIGNED_GRID, dtype=torch.long)
    total = sum(t * h * w for t, h, w in _MISALIGNED_GRID)
    sp_factor = tuple(submesh.shape)[sp_axis]
    assert total % (sp_factor * 32) != 0, "grid is aligned; this test would not exercise the pad path"

    torch.manual_seed(0)
    patch_dim = 3 * 2 * 16 * 16
    patches = torch.randn(total, patch_dim)
    with torch.no_grad():
        ref_out = reference(patches, grid_thw=grid, return_dict=True)
    golden_tokens = ref_out.pooler_output.float()
    golden_deepstack = [f.float() for f in ref_out.deepstack_features]

    tower = _tower(reference, submesh, *resolve_parallel(submesh, tp_axis, sp_axis, num_links))
    cos, sin = tower.prepare_rope(grid)
    pos = tower.prepare_pos_embeds(grid)
    p_patches, p_pos, (p_cos, p_sin), p_cu, logical = pad_patches_for_sp(
        patches, pos, (cos, sin), vision_cu_seqlens(grid), sp_factor=sp_factor
    )
    assert logical == total and p_patches.shape[0] % (sp_factor * 32) == 0
    assert len(p_cu) == len(vision_cu_seqlens(grid)) + 1, "expected one phantom window for the pad"

    tokens, deepstack = tower.forward(
        sp_shard(p_patches, submesh, sp_axis),
        pos_embeds=sp_shard(p_pos, submesh, sp_axis),
        rope=(sp_shard(p_cos, submesh, sp_axis), sp_shard(p_sin, submesh, sp_axis)),
        cu_seqlens=p_cu,
        logical_patches=logical,
    )

    merged = total // SPATIAL_MERGE_SIZE**2
    actual_tokens = tensor.to_torch(tokens, mesh_axes=[None, None])
    assert actual_tokens.shape[-2:] == (merged, OUT_HIDDEN_SIZE), f"{tuple(actual_tokens.shape)}"
    assert_quality(golden_tokens, actual_tokens, pcc=0.99)
    for golden_feature, feature_tt in zip(golden_deepstack, deepstack):
        feature = tensor.to_torch(feature_tt, mesh_axes=[None, None])
        assert feature.shape[-2:] == (merged, OUT_HIDDEN_SIZE), f"{tuple(feature.shape)}"
        assert_quality(golden_feature, feature, pcc=0.99)
