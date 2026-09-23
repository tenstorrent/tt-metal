# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host-only gate for the MiniMax-H3 AdaLN precompute, on a small synthetic checkpoint.
The design rests on ``adaln_proj`` being batch-stable while ``time_embedder`` is not. The full
26 GB build was verified once against the real checkpoint (git history: tools/build_adaln_table.py)."""

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
    diffusers_embeddings = pytest.importorskip("diffusers.models.embeddings")
    timesteps = torch.tensor([0.0, 0.02, 0.5, 0.999, 1.0], dtype=torch.float32)
    ours = ap.timestep_frequency_embedding(timesteps, 256)
    theirs = diffusers_embeddings.get_timestep_embedding(timesteps, 256, flip_sin_to_cos=True, downscale_freq_shift=0)
    assert torch.equal(ours, theirs)


def test_silu_must_run_before_the_bfloat16_cast():
    """bf16 activation shifts modulation ~7.8e-3, coherently across every block and step."""
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
    torch.manual_seed(2)
    weight = torch.randn(6 * HIDDEN * 3, TIME_EMBED_DIM).bfloat16()
    bias = torch.randn(6 * HIDDEN * 3).bfloat16()
    temb = torch.randn(16, TIME_EMBED_DIM, dtype=torch.float32)

    batched = ap.project_block_adaln(temb, weight, bias, HIDDEN)
    for row in (0, 5, 15):
        alone = ap.project_block_adaln(temb[row : row + 1], weight, bias, HIDDEN)
        assert torch.equal(batched[row * 3 : (row + 1) * 3], alone)


def test_time_embedding_is_not_batch_stable(synthetic_checkpoint):
    """Why ``temb`` is computed per step, asserted rather than silently depended on."""
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

    assert torch.allclose(batched[rows], per_step, atol=1e-4)


def test_step_timesteps_pin_conditioning_at_the_noise_aug_floor():
    video, audio = _schedules()
    steps = ap.request_step_timesteps(video.sigmas, audio.sigmas, MINIMAX_H3_KEYFRAME_NOISE_AUG)
    for index, levels in enumerate(steps):
        expected = max(float(video.timesteps[index]), MINIMAX_H3_KEYFRAME_NOISE_AUG)
        assert pytest.approx(float(levels.max())) == expected


def test_table_matches_per_step_recompute(synthetic_checkpoint):
    checkpoint, tensors = synthetic_checkpoint
    steps = _step_timesteps()
    table = _table(checkpoint)

    total_rows = sum(int(levels.numel()) for levels in steps)
    assert len(steps) == _schedules()[0].num_inference_steps
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

    embed_args = (
        tensors["time_embedder.proj_in.weight"],
        tensors["time_embedder.proj_in.bias"],
        tensors["time_embedder.proj_out.weight"],
        tensors["time_embedder.proj_out.bias"],
    )

    for step, levels in enumerate(steps):
        assert levels.numel() in (2, 3)
        assert bool((levels[1:] > levels[:-1]).all())
        assert torch.equal(table.step_timesteps(step), levels)

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
            # tag -1 padding rows clamp to the video slot; their output is discarded
            (offset + 1) * 3 + 0,
        ]
    )
    assert torch.equal(indices, expected)
