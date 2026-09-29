# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The host MRoPE builder against the pinned Transformers ``Qwen4Exp`` implementation (CPU, no weights, no device).

Set ``QWEN38_TRANSFORMERS_SRC`` to the directory holding the pinned ``transformers`` package (the same gate as
``test_transformers_oracle.py``).
"""

import functools
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.mrope import (
    IMAGE_TOKEN_ID,
    ROPE_DIM,
    ROPE_THETA,
    VISION_END_TOKEN_ID,
    VISION_START_TOKEN_ID,
    Qwen38ImageGrid,
    mrope_positions,
    mrope_rows,
    rope_inverse_frequency,
    rope_table,
)

TRANSFORMERS_SRC = os.environ.get("QWEN38_TRANSFORMERS_SRC")
if not TRANSFORMERS_SRC:
    pytest.skip("set QWEN38_TRANSFORMERS_SRC to the pinned Transformers src directory", allow_module_level=True)
sys.path.insert(0, str(Path(TRANSFORMERS_SRC)))

from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig  # noqa: E402
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpModel, Qwen4ExpTextRotaryEmbedding  # noqa: E402

HEAD_DIM = 256
ROPE_PARAMETERS = {
    "mrope_interleaved": True,
    "mrope_section": [11, 11, 10],
    "partial_rotary_factor": 0.25,
    "rope_theta": ROPE_THETA,
    "rope_type": "default",
}


def _reference_rope_index(ids: list[int], grids: list[Qwen38ImageGrid]):
    """``Qwen4ExpModel.get_rope_index`` on a stub self (no weights): the 3D ids ``[3, 1, L]`` and the delta."""

    stub = SimpleNamespace(config=SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2)))
    stub.get_vision_position_ids = functools.partial(Qwen4ExpModel.get_vision_position_ids, stub)
    input_ids = torch.tensor([ids], dtype=torch.long)
    token_types = (input_ids == IMAGE_TOKEN_ID).to(torch.int32)
    grid_thw = torch.tensor([grid.as_tuple() for grid in grids], dtype=torch.long) if grids else None
    return Qwen4ExpModel.get_rope_index(stub, input_ids, token_types, image_grid_thw=grid_thw)


def _random_prompt(generator: random.Random, images: int):
    grids = []
    ids = [generator.randrange(1000, 200_000) for _ in range(generator.randrange(1, 12))]
    for _ in range(images):
        t = 1 if generator.random() < 0.8 else generator.randrange(1, 4)
        grid = Qwen38ImageGrid(t, 2 * generator.randrange(1, 24), 2 * generator.randrange(1, 24))
        grids.append(grid)
        ids += [VISION_START_TOKEN_ID] + [IMAGE_TOKEN_ID] * grid.merged_tokens + [VISION_END_TOKEN_ID]
        ids += [generator.randrange(1000, 200_000) for _ in range(generator.randrange(0, 9))]
    return ids, grids


@pytest.mark.parametrize("seed", range(12))
def test_positions_and_delta_match_get_rope_index(seed):
    generator = random.Random(seed)
    ids, grids = _random_prompt(generator, images=generator.randrange(0, 4))
    positions = mrope_positions(ids, grids)
    reference_ids, reference_delta = _reference_rope_index(ids, grids)
    assert reference_ids.shape == (3, 1, len(ids))
    assert torch.equal(positions.axes, reference_ids[:, 0, :].to(torch.int64))
    assert positions.delta == int(reference_delta.reshape(-1)[0].item())
    assert positions.text_only == (not grids)


def test_temporal_grids_continue_at_the_spatial_span():
    """A grid with t > 1 (a video frame pair as an image grid): its t positions run p0 .. p0 + t - 1, the text after it
    continues at p0 + max(h, w) // 2 whatever t is (the reference's current_pos rule), and the delta follows."""

    for grid in (Qwen38ImageGrid(3, 2, 2), Qwen38ImageGrid(2, 4, 8), Qwen38ImageGrid(4, 6, 2)):
        ids = [7, VISION_START_TOKEN_ID] + [IMAGE_TOKEN_ID] * grid.merged_tokens + [VISION_END_TOKEN_ID, 8, 9]
        positions = mrope_positions(ids, [grid])
        reference_ids, reference_delta = _reference_rope_index(ids, [grid])
        assert torch.equal(positions.axes, reference_ids[:, 0, :].to(torch.int64)), grid
        assert positions.delta == int(reference_delta.item()), grid


def test_adjacent_images_are_separated_by_the_markers():
    grids = [Qwen38ImageGrid(1, 4, 4), Qwen38ImageGrid(1, 6, 2)]
    ids = [5]
    for grid in grids:
        ids += [VISION_START_TOKEN_ID] + [IMAGE_TOKEN_ID] * grid.merged_tokens + [VISION_END_TOKEN_ID]
    positions = mrope_positions(ids, grids)
    reference_ids, reference_delta = _reference_rope_index(ids, grids)
    assert torch.equal(positions.axes, reference_ids[:, 0, :].to(torch.int64))
    assert positions.delta == int(reference_delta.item())


def _reference_rotary():
    config = Qwen4ExpTextConfig(
        head_dim=HEAD_DIM, rope_parameters=dict(ROPE_PARAMETERS), max_position_embeddings=262_144
    )
    rotary = Qwen4ExpTextRotaryEmbedding(config)
    assert rotary.inv_freq.shape == (ROPE_DIM // 2,)
    assert torch.equal(rotary.inv_freq, rope_inverse_frequency())
    return rotary


def test_rows_match_the_interleaved_rotary_embedding_in_fp32():
    rotary = _reference_rotary()
    generator = random.Random(7)
    length = 300
    axes = torch.tensor([[generator.randrange(0, length) for _ in range(64)] for _ in range(3)], dtype=torch.int64)
    reference_cos, reference_sin = rotary(torch.zeros((1, 64, HEAD_DIM), dtype=torch.float32), axes[:, None, :])
    assert reference_cos.shape == (1, 64, ROPE_DIM)
    # The fp32 rows built the way the port builds its BF16 tables (per-position products, cos in fp32).
    inverse_frequency = rope_inverse_frequency()
    products = torch.outer(torch.arange(length, dtype=torch.float32), inverse_frequency)
    table = torch.cat((products, products), dim=-1)
    cos, sin = mrope_rows(axes, table.cos(), table.sin())
    torch.testing.assert_close(cos, reference_cos[0], rtol=0, atol=2e-6)
    torch.testing.assert_close(sin, reference_sin[0], rtol=0, atol=2e-6)
    # Distinct axes really are selected: swapping the h axis for the t axis changes the h columns only.
    swapped = axes.clone()
    swapped[1] = axes[0]
    cos_swapped, _ = mrope_rows(swapped, table.cos(), table.sin())
    changed = (cos_swapped != cos).any(dim=0)
    h_columns = torch.tensor([column % 32 % 3 == 1 for column in range(ROPE_DIM)])
    assert not changed[~h_columns].any()
    assert changed[h_columns].any()


def test_bf16_rows_match_the_reference_for_text_positions_and_image_positions():
    rotary = _reference_rotary()
    inverse_frequency = rope_inverse_frequency()
    cos_table, sin_table = rope_table(64, inverse_frequency)
    text = torch.arange(7, dtype=torch.int64).reshape(1, -1).expand(3, -1)
    image = torch.tensor([[9, 9, 9, 9], [9, 9, 10, 10], [9, 10, 9, 10]], dtype=torch.int64)
    for axes in (text, image):
        cos, sin = mrope_rows(axes, cos_table, sin_table)
        reference_cos, reference_sin = rotary(
            torch.zeros((1, axes.shape[1], HEAD_DIM), dtype=torch.bfloat16), axes[:, None, :]
        )
        # The reference casts fp32 rows to BF16 as the port does; the per-position product path and the batched
        # matmul path agree to within one BF16 unit in the last place.
        torch.testing.assert_close(cos.float(), reference_cos[0].float(), rtol=2**-7, atol=2**-9)
        torch.testing.assert_close(sin.float(), reference_sin[0].float(), rtol=2**-7, atol=2**-9)
        exact = (cos == reference_cos[0]).float().mean().item()
        assert exact > 0.95, f"only {exact:.3f} of the BF16 cos entries are bitwise the reference's"
