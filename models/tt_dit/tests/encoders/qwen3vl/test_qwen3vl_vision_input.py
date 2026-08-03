# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# Qwen3-VL vision tower, input stage: patch embedding and position embedding.
#
# The position embedding is a bilinear interpolation of a fixed 48x48 table onto
# the actual patch grid. That is the common path, not an edge case:
# `num_position_embeddings` is 2304 = 48**2 while a 768x1344 keyframe -- the
# largest canvas MiniMax-H3's `resolve_canvas_size` produces -- is 48x84 = 4032
# patches. It is a pure function of `grid_thw` and the table, so it is computed on
# the host and uploaded, the way the decoder's rotary tensors already are.
#
# The patch embedding is a `Conv3d` in the reference, but its kernel equals its
# stride over already-flattened patches, so it reduces to a matmul.
#
# The host half runs anywhere; the device half needs a mesh.
# =============================================================================

import pytest
import torch
import transformers

import ttnn

from ....encoders.qwen3vl.vision_qwen3vl import (
    Qwen3VlVisionPatchEmbed,
    vision_bilinear_indices_and_weights,
    vision_pos_embeds,
)
from ....utils import tensor
from ....utils.check import assert_quality
from ....utils.tensor import bf16_tensor

# MiniMax-H3's conditioner vision config.
IN_CHANNELS = 3
HIDDEN_SIZE = 1152
PATCH_SIZE = 16
TEMPORAL_PATCH_SIZE = 2
SPATIAL_MERGE_SIZE = 2
NUM_POSITION_EMBEDDINGS = 2304
NUM_GRID_PER_SIDE = 48  # int(2304 ** 0.5)
PATCH_DIM = IN_CHANNELS * TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE  # 1536

# The canvases `resolve_canvas_size` produces, plus a video and a multi-image case.
GRIDS = {
    "keyframe_16_9": [[1, 48, 84]],  # 768x1344, 4032 patches -- the max-area canvas
    "keyframe_4_3": [[1, 48, 64]],  # 768x1024
    "keyframe_1_1": [[1, 48, 48]],  # 768x768, exactly the table -- no interpolation needed
    "small": [[1, 4, 6]],
    "video_2f": [[2, 4, 6]],
    "two_images": [[1, 4, 6], [1, 6, 2]],
}


@pytest.fixture(scope="module")
def reference():
    """A `Qwen3VLVisionModel` with the real geometry but a shallow stack.

    `depth` is cut to 1: the input stage under test reads none of the blocks, and the position table
    and patch-embed kernel are full size.
    """
    config = transformers.Qwen3VLVisionConfig(
        depth=1,
        hidden_size=HIDDEN_SIZE,
        num_heads=16,
        intermediate_size=4304,
        in_channels=IN_CHANNELS,
        patch_size=PATCH_SIZE,
        temporal_patch_size=TEMPORAL_PATCH_SIZE,
        spatial_merge_size=SPATIAL_MERGE_SIZE,
        num_position_embeddings=NUM_POSITION_EMBEDDINGS,
        out_hidden_size=5120,
        deepstack_visual_indexes=[0],
    )
    torch.manual_seed(0)
    return transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLVisionModel._from_config(config).eval()


# --------------------------------------------------------------------------- host


@pytest.mark.parametrize("name", list(GRIDS))
def test_bilinear_indices_and_weights_match_reference(name):
    """Corner indices and weights match `transformers.vision_utils`."""
    from transformers.vision_utils import get_vision_bilinear_indices_and_weights

    grid = torch.tensor(GRIDS[name], dtype=torch.long)
    expected_idx, expected_w = get_vision_bilinear_indices_and_weights(
        grid, num_grid_per_side=NUM_GRID_PER_SIDE, spatial_merge_size=SPATIAL_MERGE_SIZE
    )
    idx, weights = vision_bilinear_indices_and_weights(
        grid, num_grid_per_side=NUM_GRID_PER_SIDE, spatial_merge_size=SPATIAL_MERGE_SIZE
    )

    assert torch.equal(idx, expected_idx), f"{name}: corner indices differ"
    assert torch.equal(weights, expected_w), f"{name}: bilinear weights differ"
    # every patch's four weights form a partition of unity, or the interpolation is not bilinear
    assert torch.allclose(weights.sum(0), torch.ones(weights.shape[1]), atol=1e-6)


@pytest.mark.parametrize("name", list(GRIDS))
def test_pos_embeds_match_reference(reference, name):
    """The interpolated position embedding matches the reference's own computation."""
    from transformers.vision_utils import get_vision_bilinear_indices_and_weights

    grid = torch.tensor(GRIDS[name], dtype=torch.long)
    idx, weights = get_vision_bilinear_indices_and_weights(
        grid, num_grid_per_side=NUM_GRID_PER_SIDE, spatial_merge_size=SPATIAL_MERGE_SIZE
    )
    with torch.no_grad():
        expected = (reference.pos_embed(idx) * weights[:, :, None]).sum(0)

    actual = vision_pos_embeds(
        reference.pos_embed.weight.detach(),
        grid,
        num_grid_per_side=NUM_GRID_PER_SIDE,
        spatial_merge_size=SPATIAL_MERGE_SIZE,
    )

    total = sum(t * h * w for t, h, w in GRIDS[name])
    assert actual.shape == (total, HIDDEN_SIZE), f"{tuple(actual.shape)} != {(total, HIDDEN_SIZE)}"
    assert torch.allclose(actual, expected, atol=1e-6), f"{name}: max diff {(actual - expected).abs().max():.2e}"


def test_square_canvas_needs_no_interpolation():
    """A 48x48 grid lands exactly on the table, so each patch is one table row at weight 1.

    Pins the degenerate case: if this ever interpolated, the 1:1 canvas would be subtly wrong while
    the other aspect ratios looked fine.
    """
    grid = torch.tensor([[1, NUM_GRID_PER_SIDE, NUM_GRID_PER_SIDE]], dtype=torch.long)
    idx, weights = vision_bilinear_indices_and_weights(
        grid, num_grid_per_side=NUM_GRID_PER_SIDE, spatial_merge_size=SPATIAL_MERGE_SIZE
    )
    # all the mass is on the first corner
    assert torch.equal(weights[0], torch.ones_like(weights[0]))
    assert torch.equal(weights[1:], torch.zeros_like(weights[1:]))
    assert torch.equal(idx[0].sort().values, torch.arange(NUM_POSITION_EMBEDDINGS))


def test_patch_embed_is_a_matmul(reference):
    """The reference `Conv3d` and a flattened-weight matmul agree to fp32 accumulation order.

    This equivalence is why the port uses a `Linear` instead of a convolution.
    """
    torch.manual_seed(0)
    patches = torch.randn(37, PATCH_DIM)
    with torch.no_grad():
        expected = reference.patch_embed(patches)
    weight = reference.patch_embed.proj.weight.detach().reshape(HIDDEN_SIZE, PATCH_DIM)
    actual = torch.nn.functional.linear(patches, weight, reference.patch_embed.proj.bias.detach())
    assert (expected - actual).abs().max() < 1e-4


# ------------------------------------------------------------------------- device


@pytest.mark.parametrize(
    ("mesh_device", "submesh_shape"),
    [pytest.param((1, 1), (1, 1), id="single"), pytest.param((4, 8), (1, 4), id="mesh1x4")],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("name", ["small", "keyframe_16_9"])
def test_patch_embed_on_device(reference, mesh_device, submesh_shape, name):
    """The ported patch embed reproduces the reference `Conv3d`, weights replicated."""
    submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    total = sum(t * h * w for t, h, w in GRIDS[name])

    torch.manual_seed(0)
    patches = torch.randn(total, PATCH_DIM)
    with torch.no_grad():
        golden = reference.patch_embed(patches).float()

    embed = Qwen3VlVisionPatchEmbed(
        in_channels=IN_CHANNELS,
        hidden_size=HIDDEN_SIZE,
        patch_size=PATCH_SIZE,
        temporal_patch_size=TEMPORAL_PATCH_SIZE,
        mesh_device=submesh,
    )
    embed.load_torch_state_dict(reference.patch_embed.state_dict())

    out = embed.forward(bf16_tensor(patches, device=submesh))
    actual = tensor.to_torch(out, mesh_axes=[None, None])

    assert actual.shape[-2:] == (total, HIDDEN_SIZE), f"{tuple(actual.shape)}"
    assert_quality(golden, actual, pcc=0.999)
