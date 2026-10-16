# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The vision tower's CPU contracts without a device: the pinned configuration and tensor inventory, the checkpoint's
vision load domain, the synthetic fixtures' pins, the preprocessing against a Conv3d, the position machinery against
``F.interpolate``, and every device-layout equivalence of ``ttnn/vision_layout.py`` (interleaved rotary, padded heads,
padded MLP width, the budget's byte model) against ``vision_reference.py``.  No TTNN import."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from models.demos.blackhole.qwen38_flash_next.tests.fixtures import vision_fixture
from models.demos.blackhole.qwen38_flash_next.tools.checkpoint_budget import vision_resident_layout
from models.demos.blackhole.qwen38_flash_next.ttnn import vision_layout
from models.demos.blackhole.qwen38_flash_next.vision_reference import (
    VISION_TENSOR_BYTES,
    VISION_TENSOR_COUNT,
    VisionTowerConfig,
    apply_rotary,
    attention_segments,
    interpolated_pos_embed,
    merged_token_count,
    patchify,
    pos_embed_interpolation,
    rotary_cos_sin,
    vision_position_ids,
    vision_tensor_names,
    vision_tensor_shapes,
)

CHECKPOINT = Path(os.environ.get("QWEN38_CHECKPOINT", "/nonexistent/Qwen3.8-Flash-Next"))
CONFIG = VisionTowerConfig()


def test_pinned_configuration_and_inventory():
    assert (CONFIG.head_dim, CONFIG.patch_dim, CONFIG.grid_side, CONFIG.merged_hidden_size, CONFIG.resize_factor) == (
        72,
        1536,
        48,
        4608,
        32,
    )
    names = vision_tensor_names(CONFIG)
    shapes = vision_tensor_shapes(CONFIG)
    assert len(names) == VISION_TENSOR_COUNT == len(shapes)
    assert all(name.startswith("model.visual.") for name in names)
    assert {name[len("model.visual.") :] for name in names} == set(shapes)
    assert sum(2 * torch.Size(shape).numel() for shape in shapes.values()) == VISION_TENSOR_BYTES


@pytest.mark.skipif(not CHECKPOINT.is_dir(), reason="set QWEN38_CHECKPOINT to the pinned checkpoint directory")
def test_checkpoint_vision_domain_reads_the_tower_from_shard_one():
    from models.demos.blackhole.qwen38_flash_next.checkpoint import VISION_SHARD, Qwen38Checkpoint

    checkpoint = Qwen38Checkpoint(CHECKPOINT)
    names = checkpoint.vision_tensor_names()
    assert names == vision_tensor_names(CONFIG)
    assert all(checkpoint.metadata(name).shard == VISION_SHARD for name in names)
    assert VisionTowerConfig.from_checkpoint(CHECKPOINT) == CONFIG
    state = checkpoint.vision_state_dict()
    expected = vision_tensor_shapes(CONFIG)
    assert set(state) == set(expected)
    assert all(
        tuple(state[name].shape) == shape and state[name].dtype == torch.bfloat16 for name, shape in expected.items()
    )
    assert sum(tensor.numel() * 2 for tensor in state.values()) == VISION_TENSOR_BYTES


@pytest.mark.parametrize("spec", vision_fixture.PINNED_FIXTURES, ids=lambda spec: spec.name)
def test_synthetic_fixtures_are_pinned(spec):
    image = vision_fixture.fixture_image(spec)  # verifies the sha256 pin
    assert image.dtype == torch.uint8 and tuple(image.shape) == (3, spec.height, spec.width)
    if not spec.reference:  # the time / memory fixtures: the pin is the test, the patches are the device test's
        return
    patches, grid_thw = vision_fixture.pixel_patches(spec, CONFIG)
    assert tuple(patches.shape) == (spec.patches, CONFIG.patch_dim)
    assert grid_thw.tolist() == [[1, *spec.grid]]
    assert spec.resized == (spec.height, spec.width), "a fixture must not be resized by the processor's pixel bounds"
    assert float(patches.min()) >= -1.0 and float(patches.max()) <= 1.0
    assert merged_token_count(grid_thw, CONFIG) == spec.merged_tokens
    assert attention_segments(grid_thw) == ((0, spec.patches),)
    assert grid_thw[0, 1] % 2 == 0 and grid_thw[0, 2] % 2 == 0


def test_patchify_order_is_the_conv3d_unfold():
    generator = torch.Generator().manual_seed(7)
    image = torch.randn(3, 64, 96, generator=generator)
    patches, grid = patchify(image, CONFIG)
    assert grid == (1, 4, 6) and tuple(patches.shape) == (24, CONFIG.patch_dim)
    weight = torch.randn(CONFIG.hidden_size, 3, 2, 16, 16, generator=generator)
    # The reference's patch embedding is one linear over the (channel, temporal, row, col) flattening ...
    linear = patches @ weight.reshape(CONFIG.hidden_size, -1).T
    # ... which is the Conv3d with kernel == stride the checkpoint defines, patch by patch.
    conv = F.conv3d(patches.reshape(-1, 3, 2, 16, 16), weight, stride=(2, 16, 16)).reshape(-1, CONFIG.hidden_size)
    torch.testing.assert_close(linear, conv, rtol=1e-4, atol=1e-3)
    # Patch 0 is the top-left 16x16 of the image, patch 1 its right neighbour (merge block order), patch 2 below it.
    torch.testing.assert_close(patches[0].reshape(3, 2, 16, 16)[:, 0], image[:, :16, :16])
    torch.testing.assert_close(patches[1].reshape(3, 2, 16, 16)[:, 1], image[:, :16, 16:32])
    torch.testing.assert_close(patches[2].reshape(3, 2, 16, 16)[:, 0], image[:, 16:32, :16])
    torch.testing.assert_close(patches[4].reshape(3, 2, 16, 16)[:, 0], image[:, :16, 32:48])


def test_position_ids_are_block_major_over_merge_blocks():
    positions = vision_position_ids(torch.tensor([[1, 4, 6]]), CONFIG)
    assert positions.tolist()[:8] == [[0, 0], [0, 1], [1, 0], [1, 1], [0, 2], [0, 3], [1, 2], [1, 3]]
    assert positions.tolist()[-1] == [3, 5]
    video = vision_position_ids(torch.tensor([[2, 4, 6]]), CONFIG)
    assert video.shape == (48, 2) and torch.equal(video[:24], video[24:])
    assert attention_segments(torch.tensor([[2, 4, 6], [1, 2, 2]])) == ((0, 24), (24, 48), (48, 52))


def test_position_table_resampling_is_bilinear_align_corners():
    generator = torch.Generator().manual_seed(11)
    table = torch.randn(CONFIG.num_position_embeddings, 8, generator=generator)
    for grid in ([1, 6, 10], [1, 48, 48], [1, 2, 2], [1, 64, 16]):
        grid_thw = torch.tensor([grid])
        ours = interpolated_pos_embed(table, grid_thw, CONFIG)
        square = table.reshape(1, CONFIG.grid_side, CONFIG.grid_side, 8).permute(0, 3, 1, 2)
        resampled = F.interpolate(square, size=(grid[1], grid[2]), mode="bilinear", align_corners=True)[0]
        raster = resampled.permute(1, 2, 0).reshape(grid[1] * grid[2], 8)
        positions = vision_position_ids(grid_thw, CONFIG)
        expected = raster[positions[:, 0] * grid[2] + positions[:, 1]]
        torch.testing.assert_close(ours, expected, rtol=1e-4, atol=1e-4)  # FP32 weight products in a different order
        indices, weights = pos_embed_interpolation(grid_thw, CONFIG)
        torch.testing.assert_close(weights.sum(1), torch.ones(len(weights)))
        assert int(indices.min()) >= 0 and int(indices.max()) < CONFIG.num_position_embeddings


def _rotary_inputs(generator, n=40):
    grid_thw = torch.tensor([[1, 4, 10]])
    cos, sin = rotary_cos_sin(vision_position_ids(grid_thw, CONFIG), CONFIG)
    q = torch.randn(n, CONFIG.num_heads, CONFIG.head_dim, generator=generator)
    k = torch.randn(n, CONFIG.num_heads, CONFIG.head_dim, generator=generator)
    return cos, sin, q, k


def test_interleaved_rotary_equals_rotate_half():
    generator = torch.Generator().manual_seed(23)
    cos, sin, q, k = _rotary_inputs(generator)
    n = q.shape[0]
    rows = vision_layout.padded_rows(n)
    perm = vision_layout.interleave_permutation(CONFIG.head_dim)
    inverse = torch.argsort(perm)
    cos_dev, sin_dev = vision_layout.device_cos_sin(cos, sin, rows=rows)
    assert tuple(cos_dev.shape) == (1, 1, rows, vision_layout.PADDED_HEAD_DIM)
    assert torch.all(cos_dev[0, 0, n:] == 1) and torch.all(sin_dev[0, 0, n:] == 0)
    assert torch.all(cos_dev[0, 0, :, CONFIG.head_dim :] == 1) and torch.all(sin_dev[0, 0, :, CONFIG.head_dim :] == 0)

    def device_side(x):
        padded = torch.zeros(rows, CONFIG.num_heads, vision_layout.PADDED_HEAD_DIM)
        padded[:n, :, : CONFIG.head_dim] = x[..., perm]
        rotated = vision_layout.rotate_interleaved(
            padded.transpose(0, 1), cos_dev[0, 0].unsqueeze(0), sin_dev[0, 0].unsqueeze(0)
        ).transpose(0, 1)
        assert torch.all(rotated[:, :, CONFIG.head_dim :] == 0) and torch.all(rotated[n:] == 0)
        return rotated

    q_dev, k_dev = device_side(q), device_side(k)
    torch.testing.assert_close(
        q_dev[:n, :, : CONFIG.head_dim][..., inverse], apply_rotary(q, cos, sin), rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(
        k_dev[:n, :, : CONFIG.head_dim][..., inverse], apply_rotary(k, cos, sin), rtol=1e-5, atol=1e-5
    )
    scores_reference = torch.einsum("nhd,mhd->hnm", apply_rotary(q, cos, sin), apply_rotary(k, cos, sin))
    scores_device = torch.einsum("nhd,mhd->hnm", q_dev[:n], k_dev[:n])
    torch.testing.assert_close(scores_device, scores_reference, rtol=1e-4, atol=1e-4)


def test_fused_qkv_and_output_projection_layouts():
    generator = torch.Generator().manual_seed(29)
    hidden, heads, head_dim = CONFIG.hidden_size, CONFIG.num_heads, CONFIG.head_dim
    padded_head = vision_layout.PADDED_HEAD_DIM
    x = torch.randn(40, hidden, generator=generator)
    weight = torch.randn(3 * hidden, hidden, generator=generator) * 0.02
    bias = torch.randn(3 * hidden, generator=generator)
    q, k, v = (x @ weight.T + bias).reshape(40, 3, heads, head_dim).unbind(1)
    weight_dev, bias_dev = vision_layout.fused_qkv_weight(weight, bias, heads=heads, head_dim=head_dim)
    assert tuple(weight_dev.shape) == (hidden, 3 * heads * padded_head) and tuple(bias_dev.shape) == (
        3 * heads * padded_head,
    )
    q_dev, k_dev, v_dev = (x @ weight_dev + bias_dev).reshape(40, 3, heads, padded_head).unbind(1)
    inverse = torch.argsort(vision_layout.interleave_permutation(head_dim))
    for dev in (q_dev, k_dev, v_dev):
        assert torch.all(dev[..., head_dim:] == 0)
    torch.testing.assert_close(q_dev[..., :head_dim][..., inverse], q, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(k_dev[..., :head_dim][..., inverse], k, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(v_dev[..., :head_dim], v, rtol=1e-4, atol=1e-4)

    proj = torch.randn(hidden, hidden, generator=generator) * 0.02
    proj_dev = vision_layout.attention_output_weight(proj, heads=heads, head_dim=head_dim)
    assert tuple(proj_dev.shape) == (heads * padded_head, hidden)
    attended = torch.randn(40, heads, head_dim, generator=generator)
    padded = torch.zeros(40, heads, padded_head)
    padded[..., :head_dim] = attended
    torch.testing.assert_close(
        padded.reshape(40, -1) @ proj_dev, attended.reshape(40, hidden) @ proj.T, rtol=1e-4, atol=1e-4
    )


def test_padded_mlp_width_is_exact():
    generator = torch.Generator().manual_seed(31)
    hidden, inter, padded = CONFIG.hidden_size, CONFIG.intermediate_size, vision_layout.PADDED_INTERMEDIATE
    x = torch.randn(40, hidden, generator=generator)
    w1, b1 = torch.randn(inter, hidden, generator=generator) * 0.02, torch.randn(inter, generator=generator)
    w2, b2 = torch.randn(hidden, inter, generator=generator) * 0.02, torch.randn(hidden, generator=generator)
    w1_dev, b1_dev = vision_layout.padded_linear(w1, b1, out_features=padded)
    w2_dev, b2_dev = vision_layout.padded_linear(w2, b2, in_features=padded)
    inner_dev = F.gelu(x @ w1_dev + b1_dev, approximate="tanh")
    assert tuple(inner_dev.shape) == (40, padded) and torch.all(inner_dev[:, inter:] == 0)
    reference = F.gelu(x @ w1.T + b1, approximate="tanh") @ w2.T + b2
    torch.testing.assert_close(inner_dev @ w2_dev + b2_dev, reference, rtol=1e-4, atol=1e-4)


def test_budget_layout_counts_the_device_tower_bytes():
    generator = torch.Generator().manual_seed(37)
    state = {
        name: torch.randn(*shape, generator=generator) * 0.02 for name, shape in vision_tensor_shapes(CONFIG).items()
    }
    layout = vision_layout.tower_layout(state, CONFIG)

    def tile_bytes(tensor: torch.Tensor) -> int:
        shape = tuple(tensor.shape)
        rows, cols = (32, shape[-1]) if len(shape) == 1 else shape[-2:]
        return (-(-rows // 32) * 32) * (-(-cols // 32) * 32) * 2

    total = (
        tile_bytes(layout["patch_weight"])
        + tile_bytes(layout["patch_bias"])
        + tile_bytes(layout["rotary_transformation"])
    )
    total += sum(tile_bytes(tensor) for block in layout["blocks"] for tensor in block.values())
    for name in (
        "merger_norm_weight",
        "merger_norm_bias",
        "merger_fc1_weight",
        "merger_fc1_bias",
        "merger_fc2_weight",
        "merger_fc2_bias",
    ):
        total += tile_bytes(layout[name])
    assert total == vision_resident_layout(mesh_size=1)["device_total"]
    four = vision_resident_layout(mesh_size=4)
    assert (
        four["device_total"]
        == total - (tile_bytes(layout["merger_fc2_weight"]) + tile_bytes(layout["merger_fc2_bias"])) * 3 // 4
    )
    assert 950_000_000 < four["device_total"] < 1_000_000_000  # about 0.95 GB per die replicated (BF16, padded)
