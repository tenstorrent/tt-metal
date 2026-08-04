# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host-only gate for the MiniMax-H3 AdaLN precompute.

Precomputing lets the 13B AdaLN parameters -- 40% of the checkpoint -- stay off
the device entirely, which the model card explicitly licenses. The risk it buys
is numerical: the table has to be what the reference would have computed at that
step, or every block is modulated slightly wrong at every step in the same
direction.

Two batch-size effects are asserted here because they pull in opposite directions
and the design depends on both:

- ``adaln_proj`` **is** batch-stable, so a step's rows can be projected together.
- ``time_embedder`` is **not**, so ``temb`` must be computed per step. Batching it
  over all 98 distinct levels shifts it ~5e-6 and the bf16 projection amplifies
  that to about one ULP on ~0.7% of values.

The full 26 GB build against the real checkpoint lives in
``build_adaln_table.py``; it reported 0 mismatches over 196 (step, layer)
projections and all 49 final-layer rows. These tests use a small synthetic
checkpoint so they run without it.
"""

import pytest
import torch
from safetensors.torch import save_file

from ....pipelines.minimax_h3 import adaln_precompute as ap
from ....pipelines.minimax_h3.packing import MINIMAX_H3_KEYFRAME_NOISE_AUG
from ....pipelines.minimax_h3.scheduler import MiniMaxH3Scheduler

HIDDEN = 64
TIME_EMBED_DIM = 32
FREQ_DIM = 16
NUM_LAYERS = 3
STEPS = 6


def _schedules(steps=STEPS):
    video = MiniMaxH3Scheduler(12.0)
    audio = MiniMaxH3Scheduler(3.0)
    video.set_timesteps(steps)
    audio.set_timesteps(steps)
    return video, audio


def _step_timesteps(steps=STEPS):
    video, audio = _schedules(steps)
    return ap.request_step_timesteps(video.sigmas, audio.sigmas, MINIMAX_H3_KEYFRAME_NOISE_AUG)


@pytest.fixture
def synthetic_checkpoint(tmp_path):
    """A shape-faithful miniature of the FL2VA transformer's AdaLN surface."""
    torch.manual_seed(0)
    tensors = {
        "time_embedder.proj_in.weight": torch.randn(HIDDEN, FREQ_DIM),
        "time_embedder.proj_in.bias": torch.randn(HIDDEN),
        "time_embedder.proj_out.weight": torch.randn(TIME_EMBED_DIM, HIDDEN),
        "time_embedder.proj_out.bias": torch.randn(TIME_EMBED_DIM),
        "final_layer.adaln_proj.linear.weight": torch.randn(2 * HIDDEN, TIME_EMBED_DIM).bfloat16(),
        "final_layer.adaln_proj.linear.bias": torch.randn(2 * HIDDEN).bfloat16(),
    }
    for layer in range(NUM_LAYERS):
        out = ap.MINIMAX_H3_ADALN_PARAMS * HIDDEN * ap.MINIMAX_H3_MODALITY_NUM
        tensors[f"blocks.{layer}.adaln_proj.linear.weight"] = torch.randn(out, TIME_EMBED_DIM).bfloat16()
        tensors[f"blocks.{layer}.adaln_proj.linear.bias"] = torch.randn(out).bfloat16()
    # Two shards, so the multi-shard key index is exercised.
    keys = sorted(tensors)
    save_file({k: tensors[k] for k in keys[: len(keys) // 2]}, tmp_path / "model-00001-of-00002.safetensors")
    save_file({k: tensors[k] for k in keys[len(keys) // 2 :]}, tmp_path / "model-00002-of-00002.safetensors")
    return tmp_path, tensors


def _table(checkpoint, steps=STEPS):
    return ap.precompute_adaln_table(
        checkpoint,
        _step_timesteps(steps),
        num_layers=NUM_LAYERS,
        hidden_size=HIDDEN,
        freq_dim=FREQ_DIM,
    )


def test_frequency_embedding_is_cosine_then_sine():
    """Matches ``Timesteps(flip_sin_to_cos=True, downscale_freq_shift=0)``."""
    diffusers_embeddings = pytest.importorskip("diffusers.models.embeddings")
    timesteps = torch.tensor([0.0, 0.02, 0.5, 0.999, 1.0], dtype=torch.float32)
    ours = ap.timestep_frequency_embedding(timesteps, 256)
    theirs = diffusers_embeddings.get_timestep_embedding(timesteps, 256, flip_sin_to_cos=True, downscale_freq_shift=0)
    assert torch.equal(ours, theirs)


def test_silu_must_run_before_the_bfloat16_cast():
    """Activating in bf16 instead of fp32 shifts modulation ~7.8e-3.

    That bias is identical for every block at every step, so it accumulates
    coherently down the trajectory instead of averaging out.
    """
    torch.manual_seed(1)
    weight = torch.randn(6 * HIDDEN * 3, TIME_EMBED_DIM).bfloat16()
    bias = torch.randn(6 * HIDDEN * 3).bfloat16()
    temb = torch.randn(4, TIME_EMBED_DIM, dtype=torch.float32)

    correct = ap.project_block_adaln(temb, weight, bias, HIDDEN)
    hoisted = torch.nn.functional.linear(torch.nn.functional.silu(temb.bfloat16()), weight, bias)
    hoisted = torch.stack(hoisted.view(-1, 6 * HIDDEN).chunk(6, dim=-1), dim=1)

    assert not torch.equal(correct, hoisted)
    assert (correct.float() - hoisted.float()).abs().max() > 1e-3


def test_adaln_projection_is_batch_stable():
    """Projecting a row alone or alongside others must be bitwise identical.

    This is what allows a step's two or three levels to be projected together.
    """
    torch.manual_seed(2)
    weight = torch.randn(6 * HIDDEN * 3, TIME_EMBED_DIM).bfloat16()
    bias = torch.randn(6 * HIDDEN * 3).bfloat16()
    temb = torch.randn(16, TIME_EMBED_DIM, dtype=torch.float32)

    batched = ap.project_block_adaln(temb, weight, bias, HIDDEN)
    for row in (0, 5, 15):
        alone = ap.project_block_adaln(temb[row : row + 1], weight, bias, HIDDEN)
        assert torch.equal(batched[row * 3 : (row + 1) * 3], alone)


def test_time_embedding_is_not_batch_stable(synthetic_checkpoint):
    """The reason ``temb`` is computed per step rather than once.

    If a torch or BLAS update ever made this stable, the table could be built over
    the global distinct set and shrink by ~a third -- so this asserts the current
    behaviour rather than silently depending on it.
    """
    _, tensors = synthetic_checkpoint
    args = (
        tensors["time_embedder.proj_in.weight"],
        tensors["time_embedder.proj_in.bias"],
        tensors["time_embedder.proj_out.weight"],
        tensors["time_embedder.proj_out.bias"],
    )
    steps = _step_timesteps()
    everything = torch.unique(torch.cat(steps), sorted=True)

    batched = ap.time_embedding(everything, *args, freq_dim=FREQ_DIM)
    per_step = ap.time_embedding(steps[0], *args, freq_dim=FREQ_DIM)
    rows = [int((everything == value).nonzero()[0, 0]) for value in steps[0]]

    # Same values, same weights; only the batch shape differs.
    assert torch.allclose(batched[rows], per_step, atol=1e-4)


def test_step_timesteps_carry_two_or_three_levels():
    steps = _step_timesteps()
    assert len(steps) == _schedules()[0].num_inference_steps
    for levels in steps:
        assert levels.numel() in (2, 3)
        # Sorted and distinct, as torch.unique guarantees and the layout assumes.
        assert bool((levels[1:] > levels[:-1]).all())


def test_step_timesteps_pin_conditioning_at_the_noise_aug_floor():
    video, audio = _schedules()
    steps = ap.request_step_timesteps(video.sigmas, audio.sigmas, MINIMAX_H3_KEYFRAME_NOISE_AUG)
    for index, levels in enumerate(steps):
        expected = max(float(video.timesteps[index]), MINIMAX_H3_KEYFRAME_NOISE_AUG)
        assert pytest.approx(float(levels.max())) == expected


def test_table_shape_and_offsets(synthetic_checkpoint):
    checkpoint, _ = synthetic_checkpoint
    steps = _step_timesteps()
    table = _table(checkpoint)

    total_rows = sum(int(levels.numel()) for levels in steps)
    assert table.num_layers == NUM_LAYERS
    assert table.hidden_size == HIDDEN
    assert table.num_steps == len(steps)
    assert tuple(table.block_params.shape) == (
        NUM_LAYERS,
        total_rows * ap.MINIMAX_H3_MODALITY_NUM,
        ap.MINIMAX_H3_ADALN_PARAMS,
        HIDDEN,
    )
    assert table.block_params.dtype == torch.bfloat16
    assert tuple(table.final_shift.shape) == (total_rows, HIDDEN)

    assert int(table.step_offsets[0]) == 0
    assert int(table.step_offsets[-1]) == total_rows
    for index, levels in enumerate(steps):
        assert torch.equal(table.step_timesteps(index), levels)


def test_table_matches_per_step_recompute(synthetic_checkpoint):
    """The gate: every table row equals what that step would have computed."""
    checkpoint, tensors = synthetic_checkpoint
    steps = _step_timesteps()
    table = _table(checkpoint)
    embed_args = (
        tensors["time_embedder.proj_in.weight"],
        tensors["time_embedder.proj_in.bias"],
        tensors["time_embedder.proj_out.weight"],
        tensors["time_embedder.proj_out.bias"],
    )

    for step, levels in enumerate(steps):
        temb = ap.time_embedding(levels, *embed_args, freq_dim=FREQ_DIM)
        rows = table.step_rows(step, torch.arange(levels.numel()))
        low = int(rows[0]) * ap.MINIMAX_H3_MODALITY_NUM
        high = (int(rows[-1]) + 1) * ap.MINIMAX_H3_MODALITY_NUM

        for layer in range(NUM_LAYERS):
            expected = ap.project_block_adaln(
                temb,
                tensors[f"blocks.{layer}.adaln_proj.linear.weight"],
                tensors[f"blocks.{layer}.adaln_proj.linear.bias"],
                HIDDEN,
            )
            assert torch.equal(table.block_params[layer, low:high], expected)

        shift, scale = ap.project_final_adaln(
            temb,
            tensors["final_layer.adaln_proj.linear.weight"],
            tensors["final_layer.adaln_proj.linear.bias"],
        )
        assert torch.equal(table.final_shift[rows[0] : rows[-1] + 1], shift)
        assert torch.equal(table.final_scale[rows[0] : rows[-1] + 1], scale)


def test_adaln_indices_address_timestep_and_modality(synthetic_checkpoint):
    checkpoint, _ = synthetic_checkpoint
    table = _table(checkpoint)
    step = 3
    offset = int(table.step_offsets[step])

    timestep_indices = torch.tensor([0, 0, 1, 1])
    token_tags = torch.tensor([0, 2, 1, -1])
    indices = table.adaln_indices(step, timestep_indices, token_tags)

    expected = torch.tensor(
        [
            (offset + 0) * 3 + 0,
            (offset + 0) * 3 + 2,
            (offset + 1) * 3 + 1,
            # Padding rows carry tag -1 and clamp to the video slot; their output
            # is discarded either way.
            (offset + 1) * 3 + 0,
        ]
    )
    assert torch.equal(indices, expected)


def test_step_rows_rejects_out_of_range_index(synthetic_checkpoint, expect_error):
    checkpoint, _ = synthetic_checkpoint
    table = _table(checkpoint)
    span = int(table.step_offsets[1]) - int(table.step_offsets[0])
    with expect_error(ValueError, "exceeds step"):
        table.step_rows(0, torch.tensor([span]))


def test_precompute_rejects_missing_checkpoint(tmp_path, expect_error):
    with expect_error(FileNotFoundError, "no model-.*safetensors"):
        ap.precompute_adaln_table(tmp_path, _step_timesteps(), num_layers=1, hidden_size=HIDDEN)


def test_mismatched_schedule_lengths_rejected(expect_error):
    video, audio = _schedules()
    with expect_error(ValueError, "equal length"):
        ap.request_step_timesteps(video.sigmas, audio.sigmas[:-2], MINIMAX_H3_KEYFRAME_NOISE_AUG)


def test_table_is_far_smaller_than_the_weights_it_replaces(synthetic_checkpoint):
    """The whole justification: the table must not approach the weight size.

    At the real shapes this is 1.416 GB against 26.0 GB of ``adaln_proj``
    parameters, i.e. 0.354 vs 6.50 GB per device at TP=4.
    """
    checkpoint, _ = synthetic_checkpoint
    table = _table(checkpoint)
    weight_bytes = NUM_LAYERS * (ap.MINIMAX_H3_ADALN_PARAMS * HIDDEN * ap.MINIMAX_H3_MODALITY_NUM) * TIME_EMBED_DIM * 2
    assert table.nbytes() < weight_bytes
