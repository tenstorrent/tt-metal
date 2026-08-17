# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Qwen3-VL vision tower end to end on device against the HF reference: patch
# embed, position embeddings, every block, the output and deepstack mergers, at
# the released depth and only production grid shapes.

import pytest
import torch
import transformers

import ttnn

from ....encoders.qwen3vl.vision_qwen3vl import Qwen3VlVisionModel, vision_cu_seqlens
from ....utils import tensor
from ....utils.check import assert_quality
from .common import (
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
_CASES = [pytest.param(name, True, id=f"check-{name}") for name in GRIDS] + [
    pytest.param(name, False, id=f"perf-{name}") for name in _PERF_GRIDS
]


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
    cos, sin = tower.prepare_rope(grid)
    pos = tower.prepare_pos_embeds(grid)
    tokens, deepstack = tower.forward(
        sp_shard(patches, submesh, sp_axis),
        pos_embeds=sp_shard(pos, submesh, sp_axis),
        rope=(sp_shard(cos, submesh, sp_axis), sp_shard(sin, submesh, sp_axis)),
        cu_seqlens=cu_seqlens,
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
