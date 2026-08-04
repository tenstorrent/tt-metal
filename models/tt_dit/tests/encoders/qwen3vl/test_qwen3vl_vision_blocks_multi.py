# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# Qwen3-VL vision tower with several references: the `ref2va` path.
#
# `ref2va` packs up to nine images and three videos, and attention must not cross
# from one to another. `cu_seqlens = repeat_interleave(h*w, t).cumsum()` names the
# boundaries: one image is one block, and a video is one block *per frame*.
#
# Handled by attending within each block in turn rather than with a
# block-diagonal mask. An `s x s` mask is 17 GiB for a full request of nine
# images and three five-frame videos -- more than half a Blackhole chip, for a
# mask -- while a couple of dozen smaller attentions cost nothing extra. The
# reference's non-flash path splits the same way.
#
# The tests that matter here are the ones that would pass if the blocking were
# absent: with a single block, blocked and unblocked attention must agree
# exactly; with several, they must not.
# =============================================================================

import pytest
import torch
import transformers

import ttnn

from ....encoders.qwen3vl.vision_qwen3vl import (
    Qwen3VlVisionAttention,
    Qwen3VlVisionModel,
    vision_cu_seqlens,
    vision_rope_tensors,
)
from ....utils import tensor
from ....utils.check import assert_quality
from ....utils.tensor import bf16_tensor

VIS_HIDDEN = 192
VIS_HEADS = 4
VIS_HEAD_DIM = VIS_HIDDEN // VIS_HEADS  # 48 -- not tile-aligned, pads to 64
VIS_DEPTH = 3
VIS_INTERMEDIATE = 256
SPATIAL_MERGE_SIZE = 2
NUM_POSITION_EMBEDDINGS = 64
OUT_HIDDEN = 128
DEEPSTACK_INDEXES = [1]
NORM_EPS = 1e-6
PATCH_DIM = 3 * 2 * 16 * 16

GRIDS = {
    "one_image": [[1, 4, 6]],
    "two_images": [[1, 4, 6], [1, 4, 4]],
    "video_3_frames": [[3, 4, 4]],
    "images_and_video": [[1, 4, 6], [2, 4, 4], [1, 2, 2]],
}


@pytest.fixture(scope="module")
def reference():
    config = transformers.Qwen3VLVisionConfig(
        depth=VIS_DEPTH,
        hidden_size=VIS_HIDDEN,
        num_heads=VIS_HEADS,
        intermediate_size=VIS_INTERMEDIATE,
        in_channels=3,
        patch_size=16,
        temporal_patch_size=2,
        spatial_merge_size=SPATIAL_MERGE_SIZE,
        num_position_embeddings=NUM_POSITION_EMBEDDINGS,
        out_hidden_size=OUT_HIDDEN,
        hidden_act="gelu_pytorch_tanh",
        deepstack_visual_indexes=DEEPSTACK_INDEXES,
    )
    torch.manual_seed(0)
    return transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLVisionModel._from_config(config).eval()


# --------------------------------------------------------------------------- host


@pytest.mark.parametrize("name", list(GRIDS))
def test_cu_seqlens_matches_reference(name):
    """Block boundaries match `transformers.vision_utils.get_vision_cu_seqlens`."""
    from transformers.vision_utils import get_vision_cu_seqlens

    grid = torch.tensor(GRIDS[name], dtype=torch.long)
    expected = get_vision_cu_seqlens(grid).tolist()
    assert list(vision_cu_seqlens(grid)) == expected, f"{name}"


def test_a_video_is_one_block_per_frame():
    """A `t`-frame video contributes `t` blocks, not one -- frames never attend to each other."""
    assert vision_cu_seqlens(torch.tensor([[3, 4, 4]])) == (0, 16, 32, 48)
    assert vision_cu_seqlens(torch.tensor([[1, 4, 4]])) == (0, 16)


def test_one_image_is_a_single_block():
    """The `fl2va` case: a single block, which is why it needed no blocking at all."""
    assert vision_cu_seqlens(torch.tensor([[1, 48, 84]])) == (0, 4032)


# ------------------------------------------------------------------------- device

_MESH = [pytest.param((1, 1), (1, 1), id="single")]


def _rope(grid, submesh):
    cos, sin = vision_rope_tensors(grid, head_dim=VIS_HEAD_DIM, spatial_merge_size=SPATIAL_MERGE_SIZE)
    return bf16_tensor(cos, device=submesh), bf16_tensor(sin, device=submesh)


@pytest.mark.parametrize(("mesh_device", "submesh_shape"), _MESH, indirect=["mesh_device"])
def test_single_block_is_unaffected_by_blocking(reference, mesh_device, submesh_shape):
    """With one block, passing `cu_seqlens` must change nothing.

    This is the `fl2va` regression guard: the blocked path has to reduce exactly to the plain one, or
    adding `ref2va` support would have silently perturbed the already-verified single-image result.
    """
    submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    grid = torch.tensor(GRIDS["one_image"], dtype=torch.long)
    total = 4 * 6

    torch.manual_seed(0)
    x = bf16_tensor(torch.randn(total, VIS_HIDDEN), device=submesh)
    attn = Qwen3VlVisionAttention(hidden_size=VIS_HIDDEN, num_heads=VIS_HEADS, mesh_device=submesh)
    attn.load_torch_state_dict(reference.blocks[0].attn.state_dict())
    pe = _rope(grid, submesh)

    plain = tensor.to_torch(attn.forward(x, pos_embeds=pe), mesh_axes=[None, None])
    blocked = tensor.to_torch(
        attn.forward(x, pos_embeds=pe, cu_seqlens=vision_cu_seqlens(grid)), mesh_axes=[None, None]
    )
    assert torch.equal(plain, blocked), "blocking changed the single-block result"


@pytest.mark.parametrize(("mesh_device", "submesh_shape"), _MESH, indirect=["mesh_device"])
@pytest.mark.parametrize("name", ["two_images", "video_3_frames", "images_and_video"])
def test_blocking_changes_multi_block_attention(reference, mesh_device, submesh_shape, name):
    """With several blocks, blocking must change the result -- otherwise it is not being applied.

    The failure this guards against is the quiet one: omitting `cu_seqlens` lets every reference attend
    to every other, which produces correct shapes and wrong conditioning.
    """
    submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    grid = torch.tensor(GRIDS[name], dtype=torch.long)
    cu = vision_cu_seqlens(grid)
    assert len(cu) > 2, f"{name} should be multi-block"

    torch.manual_seed(0)
    x = bf16_tensor(torch.randn(cu[-1], VIS_HIDDEN), device=submesh)
    attn = Qwen3VlVisionAttention(hidden_size=VIS_HIDDEN, num_heads=VIS_HEADS, mesh_device=submesh)
    attn.load_torch_state_dict(reference.blocks[0].attn.state_dict())
    pe = _rope(grid, submesh)

    unblocked = tensor.to_torch(attn.forward(x, pos_embeds=pe), mesh_axes=[None, None])
    blocked = tensor.to_torch(attn.forward(x, pos_embeds=pe, cu_seqlens=cu), mesh_axes=[None, None])
    assert not torch.allclose(unblocked, blocked, atol=1e-2), "blocking had no effect; references bleed together"


@pytest.mark.parametrize(("mesh_device", "submesh_shape"), _MESH, indirect=["mesh_device"])
def test_cu_seqlens_must_span_the_sequence(reference, mesh_device, submesh_shape, expect_error):
    """Boundaries that do not cover the input are a caller error, not a partial attention."""
    submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    grid = torch.tensor(GRIDS["two_images"], dtype=torch.long)
    attn = Qwen3VlVisionAttention(hidden_size=VIS_HIDDEN, num_heads=VIS_HEADS, mesh_device=submesh)
    attn.load_torch_state_dict(reference.blocks[0].attn.state_dict())
    x = bf16_tensor(torch.zeros(40, VIS_HIDDEN), device=submesh)
    with expect_error(ValueError, "must span"):
        attn.forward(x, pos_embeds=_rope(grid, submesh), cu_seqlens=(0, 16, 30))


@pytest.mark.parametrize(("mesh_device", "submesh_shape"), _MESH, indirect=["mesh_device"])
@pytest.mark.parametrize("name", list(GRIDS))
def test_tower_matches_reference_with_multiple_references(reference, mesh_device, submesh_shape, name):
    """The whole tower against `Qwen3VLVisionModel` for multi-image and video inputs.

    The reference splits by `cu_seqlens` itself on its non-flash path, so this is a direct check that
    our per-block attention reproduces it -- including that the mergers group patches within a
    reference rather than across the boundary.
    """
    submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    grid = torch.tensor(GRIDS[name], dtype=torch.long)
    total = sum(t * h * w for t, h, w in GRIDS[name])

    torch.manual_seed(0)
    patches = torch.randn(total, PATCH_DIM)
    with torch.no_grad():
        ref_out = reference(patches, grid_thw=grid, return_dict=True)

    tower = Qwen3VlVisionModel(
        hidden_size=VIS_HIDDEN,
        num_heads=VIS_HEADS,
        depth=VIS_DEPTH,
        intermediate_size=VIS_INTERMEDIATE,
        spatial_merge_size=SPATIAL_MERGE_SIZE,
        num_position_embeddings=NUM_POSITION_EMBEDDINGS,
        out_hidden_size=OUT_HIDDEN,
        hidden_act="gelu_pytorch_tanh",
        norm_eps=NORM_EPS,
        deepstack_visual_indexes=DEEPSTACK_INDEXES,
        mesh_device=submesh,
    )
    tower.load_torch_state_dict(reference.state_dict())
    cos, sin = tower.prepare_rope(grid)
    tokens, deepstack = tower.forward(
        bf16_tensor(patches, device=submesh),
        pos_embeds=bf16_tensor(tower.prepare_pos_embeds(grid), device=submesh),
        rope=(bf16_tensor(cos, device=submesh), bf16_tensor(sin, device=submesh)),
        cu_seqlens=vision_cu_seqlens(grid),
    )

    merged = total // SPATIAL_MERGE_SIZE**2
    actual = tensor.to_torch(tokens, mesh_axes=[None, None])
    assert actual.shape[-2:] == (merged, OUT_HIDDEN), f"{tuple(actual.shape)}"
    assert_quality(ref_out.pooler_output.float(), actual, pcc=0.99)

    assert len(deepstack) == len(DEEPSTACK_INDEXES)
    for feature, golden in zip(deepstack, ref_out.deepstack_features):
        assert_quality(golden.float(), tensor.to_torch(feature, mesh_axes=[None, None]), pcc=0.99)
