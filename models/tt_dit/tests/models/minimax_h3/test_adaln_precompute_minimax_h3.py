# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""The host-precomputed AdaLN table must equal what the device would have projected.

This is what makes MiniMax-H3 fit one Blackhole chip: `time_embedder`, all 50 `adaln_proj` and
`norm_out.linear` -- ~26 GB, ~40% of the checkpoint -- stay on disk and a table addressed by the
same row indices replaces them. If the table is even slightly off it is a silent accuracy loss:
every block reads the same modulation, so a difference biases all 50 identically at every step and
accumulates along the trajectory rather than averaging out. No end-to-end metric would name it.

The host leg checks the table against an independent reference (diffusers' own
`get_timestep_embedding` and plain `torch.nn.functional.linear`), on both checkpoint key layouts.
The device leg checks `MiniMaxH3AdalnCache` against the on-device projection a block would have
done -- the `1 +` fold on the scales, the TP column split and the row order all at once.
"""

from __future__ import annotations

import pytest
import torch
from diffusers.models.embeddings import get_timestep_embedding
from safetensors.torch import save_file

import ttnn

from ....models.transformers.minimax_h3.adaln_cache_minimax_h3 import MiniMaxH3AdalnCache
from ....models.transformers.minimax_h3.transformer_block_minimax_h3 import (
    _SCALE_MLP,
    _SCALE_MSA,
    MODALITY_NUM,
    NUM_MODULATION_PARAMS,
    MiniMaxH3TransformerBlock,
)
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....pipelines.minimax_h3 import adaln_precompute as ap
from ....utils.tensor import from_torch
from .common import SMALL_LINE_PARALLEL

# Agreement bound between the host table and the on-device projection. Both legs are bf16 GEMMs
# over the same weights, so the gap is rounding; measured 0.003 on the 1x1 mesh.
MAX_TABLE_REL_RMSE = 0.02
# ... and the discriminating power of that bound: the correct rows must be this many times closer
# than an adjacent step's, which is what a wrong `step_offset` would return.
WRONG_STEP_MARGIN = 5.0

HIDDEN = 64
TIME_EMBED_DIM = 32
FREQ_DIM = 16
NUM_LAYERS = 3
STEPS = 4
SLOTS = 3

# Names differ between the original MiniMax release and the diffusers conversion the pipeline
# loads; the builder resolves either, and getting that wrong would be a KeyError on a 144 GB
# snapshot rather than on a 1 MB fixture.
_LAYOUTS = {
    "minimax": {
        "proj_in": "time_embedder.proj_in",
        "proj_out": "time_embedder.proj_out",
        "block": "blocks.{layer}.adaln_proj.linear",
        "final": "final_layer.adaln_proj.linear",
    },
    "diffusers": {
        "proj_in": "time_embedder.linear_1",
        "proj_out": "time_embedder.linear_2",
        "block": "transformer_blocks.{layer}.adaln_proj.linear",
        "final": "norm_out.linear",
    },
}


def _step_levels(steps: int = STEPS, slots: int = SLOTS) -> list[torch.Tensor]:
    """Slot levels shaped like `packing.slot_levels` output: fixed width, no dedup.

    Modelled on a real schedule rather than random noise -- video descending over the trajectory,
    audio on its own faster schedule, the keyframe slot pinned at its noise-aug floor -- so
    consecutive steps are as far apart as they are in a request, which is what makes the
    wrong-step comparison below a real discriminator.
    """
    video = torch.linspace(1.0, 0.05, steps, dtype=torch.float32)
    return [torch.tensor([float(v), float(v) ** 2, max(float(v), 0.5)][:slots], dtype=torch.float32) for v in video]


def _weights() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    block_out = NUM_MODULATION_PARAMS * HIDDEN * MODALITY_NUM
    return {
        "proj_in.weight": torch.randn(HIDDEN, FREQ_DIM),
        "proj_in.bias": torch.randn(HIDDEN),
        "proj_out.weight": torch.randn(TIME_EMBED_DIM, HIDDEN),
        "proj_out.bias": torch.randn(TIME_EMBED_DIM),
        "final.weight": torch.randn(2 * HIDDEN, TIME_EMBED_DIM).bfloat16(),
        "final.bias": torch.randn(2 * HIDDEN).bfloat16(),
        **{f"block{layer}.weight": torch.randn(block_out, TIME_EMBED_DIM).bfloat16() for layer in range(NUM_LAYERS)},
        **{f"block{layer}.bias": torch.randn(block_out).bfloat16() for layer in range(NUM_LAYERS)},
    }


def _write_checkpoint(directory, weights: dict[str, torch.Tensor], layout: str) -> None:
    names = _LAYOUTS[layout]
    tensors = {}
    for role in ("proj_in", "proj_out"):
        for suffix in ("weight", "bias"):
            tensors[f"{names[role]}.{suffix}"] = weights[f"{role}.{suffix}"]
    for suffix in ("weight", "bias"):
        tensors[f"{names['final']}.{suffix}"] = weights[f"final.{suffix}"]
    for layer in range(NUM_LAYERS):
        prefix = names["block"].format(layer=layer)
        for suffix in ("weight", "bias"):
            tensors[f"{prefix}.{suffix}"] = weights[f"block{layer}.{suffix}"]
    # Two shards, so the cross-shard key index is exercised rather than a single-file fast path.
    keys = sorted(tensors)
    stem = "model" if layout == "minimax" else "diffusion_pytorch_model"
    save_file({k: tensors[k] for k in keys[: len(keys) // 2]}, str(directory / f"{stem}-00001-of-00002.safetensors"))
    save_file({k: tensors[k] for k in keys[len(keys) // 2 :]}, str(directory / f"{stem}-00002-of-00002.safetensors"))


def _reference_temb(levels: torch.Tensor, weights: dict[str, torch.Tensor]) -> torch.Tensor:
    """`temb` from diffusers' own sinusoidal embedding, in fp32.

    `flip_sin_to_cos=True, downscale_freq_shift=0` is the checkpoint's `Timesteps` configuration --
    cosine before sine. Independent of `adaln_precompute`'s own implementation, which is the point.
    """
    embedding = get_timestep_embedding(levels.to(torch.float32), FREQ_DIM, flip_sin_to_cos=True, downscale_freq_shift=0)
    hidden = torch.nn.functional.linear(embedding, weights["proj_in.weight"], weights["proj_in.bias"])
    hidden = torch.nn.functional.silu(hidden)
    return torch.nn.functional.linear(hidden, weights["proj_out.weight"], weights["proj_out.bias"])


@pytest.mark.parametrize("layout", sorted(_LAYOUTS))
def test_adaln_table_matches_a_plain_torch_projection(tmp_path, layout):
    weights = _weights()
    _write_checkpoint(tmp_path, weights, layout)
    step_levels = _step_levels()

    table = ap.precompute_adaln_table(
        tmp_path, step_levels, num_layers=NUM_LAYERS, hidden_size=HIDDEN, freq_dim=FREQ_DIM
    )

    assert table.num_steps == STEPS
    assert table.num_slots == SLOTS
    assert table.num_layers == NUM_LAYERS
    assert table.hidden_size == HIDDEN
    assert table.block_params.shape == (NUM_LAYERS, STEPS * SLOTS * MODALITY_NUM, NUM_MODULATION_PARAMS, HIDDEN)
    assert table.final_shift.shape == (STEPS * SLOTS, HIDDEN)

    for step, levels in enumerate(step_levels):
        temb = _reference_temb(levels, weights)
        # SiLU at temb's fp32 precision, only the result cast to the projection's bf16 -- the
        # ordering the reference uses and the one the builder must reproduce.
        activated = torch.nn.functional.silu(temb).to(torch.bfloat16)

        for layer in range(NUM_LAYERS):
            expected = torch.nn.functional.linear(
                activated, weights[f"block{layer}.weight"], weights[f"block{layer}.bias"]
            )
            # Reference layout of the output dim: [modality][param][hidden].
            expected = expected.view(SLOTS, MODALITY_NUM, NUM_MODULATION_PARAMS, HIDDEN)
            for slot in range(SLOTS):
                for tag in range(MODALITY_NUM):
                    row = table.step_offset(step) * MODALITY_NUM + slot * MODALITY_NUM + tag
                    torch.testing.assert_close(
                        table.block_params[layer, row].to(torch.float32),
                        expected[slot, tag].to(torch.float32),
                        rtol=0,
                        atol=0,
                        msg=f"{layout} layer {layer} step {step} slot {slot} tag {tag}",
                    )

        final = torch.nn.functional.linear(activated, weights["final.weight"], weights["final.bias"])
        shift, scale = final.chunk(2, dim=-1)
        rows = slice(table.step_offset(step), table.step_offset(step) + SLOTS)
        torch.testing.assert_close(table.final_shift[rows].to(torch.float32), shift.to(torch.float32), rtol=0, atol=0)
        torch.testing.assert_close(table.final_scale[rows].to(torch.float32), scale.to(torch.float32), rtol=0, atol=0)


def test_step_offset_is_the_slot_stride(tmp_path, expect_error):
    weights = _weights()
    _write_checkpoint(tmp_path, weights, "diffusers")
    table = ap.precompute_adaln_table(
        tmp_path, _step_levels(), num_layers=NUM_LAYERS, hidden_size=HIDDEN, freq_dim=FREQ_DIM
    )
    assert [table.step_offset(step) for step in range(STEPS)] == [step * SLOTS for step in range(STEPS)]
    with expect_error(ValueError, "outside the table"):
        table.step_offset(STEPS)


def test_ragged_step_levels_are_rejected(tmp_path, expect_error):
    """The role tuple is pinned per request, so a step with a different slot count is a bug in the
    caller, not a table this module should silently build a broken addressing scheme for."""
    weights = _weights()
    _write_checkpoint(tmp_path, weights, "diffusers")
    levels = _step_levels()
    levels[1] = levels[1][:-1]
    with expect_error(ValueError, "same number of slots"):
        ap.precompute_adaln_table(tmp_path, levels, num_layers=NUM_LAYERS, hidden_size=HIDDEN, freq_dim=FREQ_DIM)


def test_is_adaln_key_selects_the_whole_branch_and_nothing_else():
    """The drop predicate decides what never reaches host memory. Too wide silently removes weights
    the model still needs (a load-time failure); too narrow leaves the 26 GB in place."""
    dropped = [
        "time_embedder.linear_1.weight",
        "time_embedder.linear_2.bias",
        "transformer_blocks.0.adaln_proj.linear.weight",
        "transformer_blocks.49.adaln_proj.linear.bias",
        "norm_out.linear.weight",
        "norm_out.linear.bias",
    ]
    kept = [
        "transformer_blocks.0.attn.to_qkv.weight",
        "transformer_blocks.0.ff.net.0.proj.weight",
        "transformer_blocks.0.norm1.weight",
        "norm_out.norm.weight",
        "proj_out.weight",
        "audio_proj_out.bias",
        "proj_in.weight",
        "context_embedder.weight",
        "token_refiner.blocks.0.attn.to_qkv.weight",
    ]
    assert [key for key in dropped if not ap.is_adaln_key(key)] == []
    assert [key for key in kept if ap.is_adaln_key(key)] == []


@SMALL_LINE_PARALLEL
def test_adaln_cache_matches_the_on_device_projection(
    mesh_device, sp_axis, tp_axis, num_links, device_params, topology, is_fsdp, tmp_path
):
    """The cached tables must be what the block would have computed on device.

    Same weights, same levels: one leg projects `temb` through a real `adaln_proj` on the mesh, the
    other reads the host-built table back off the device. Agreement covers the `1 +` fold on the two
    scales, the TP column split and the `slot * MODALITY_NUM + tag` row order in one comparison.
    """
    weights = _weights()
    _write_checkpoint(tmp_path, weights, "diffusers")
    step_levels = _step_levels()

    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tuple(mesh_device.shape)[tp_axis]),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=tuple(mesh_device.shape)[sp_axis]),
        cfg_parallel=None,
    )
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)

    block = MiniMaxH3TransformerBlock(
        hidden_size=HIDDEN,
        num_heads=1,
        head_dim=HIDDEN,
        ffn_dim=2 * HIDDEN,
        time_embed_dim=TIME_EMBED_DIM,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
    )
    # Only the projection is loaded: the rest of the block is irrelevant here and a partial
    # `load_torch_state_dict` on the whole block trips the attention's rotary relayout. The block's
    # own `_prepare_torch_state` still does the TP reorder, so the device tensor is exactly what a
    # real load would have produced.
    state = {
        "adaln_proj.linear.weight": weights["block0.weight"],
        "adaln_proj.linear.bias": weights["block0.bias"],
    }
    block._prepare_torch_state(state)  # noqa: SLF001
    block.adaln_proj.load_torch_state_dict({"weight": state["adaln_proj.weight"], "bias": state["adaln_proj.bias"]})

    table = ap.precompute_adaln_table(
        tmp_path, step_levels, num_layers=NUM_LAYERS, hidden_size=HIDDEN, freq_dim=FREQ_DIM
    )
    cache = MiniMaxH3AdalnCache(
        table,
        mesh_device=mesh_device,
        parallel_config=parallel_config,
        num_layers=NUM_LAYERS,
        hidden_size=HIDDEN,
    )
    cached = cache.block_tables(0)

    def rel_rmse(reference: torch.Tensor, test: torch.Tensor) -> float:
        reference, test = reference.flatten().to(torch.float64), test.flatten().to(torch.float64)
        return float(torch.linalg.vector_norm(test - reference) / torch.linalg.vector_norm(reference))

    rows_per_step = SLOTS * MODALITY_NUM
    for step, levels in enumerate(step_levels):
        temb = from_torch(
            _reference_temb(levels, weights).reshape(1, 1, SLOTS, TIME_EMBED_DIM),
            device=mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
        )
        projected = block._modulation_tables(temb)  # noqa: SLF001
        base = cache.step_offset(step) * MODALITY_NUM
        for param, (device_table, cached_table) in enumerate(zip(projected, cached, strict=True)):
            got = ttnn.to_torch(ttnn.get_device_tensors(device_table)[0]).to(torch.float32)
            full = ttnn.to_torch(ttnn.get_device_tensors(cached_table)[0]).to(torch.float32)
            want = full[base : base + rows_per_step]
            # Both legs are bf16 GEMMs, but the device runs a tiled HiFi2 matmul and the host a
            # plain one, so this is the bf16 rounding floor, not bit equality. Relative RMSE over
            # the whole table rather than an elementwise bound: the modulation values straddle zero
            # and a near-zero entry makes any elementwise relative tolerance meaningless.
            assert (
                rel_rmse(want, got) <= MAX_TABLE_REL_RMSE
            ), f"param {param} at step {step}: rel RMSE {rel_rmse(want, got):.4f}"
            # The bound above only means something if it can fail. The same param at a *different*
            # step is the nearest wrong answer a `step_offset` bug would produce; the correct rows
            # must be an order of magnitude closer than that. Stated as a ratio rather than an
            # absolute floor because how far apart two steps are is a property of the schedule.
            other = (step + 1) % STEPS
            wrong = full[other * rows_per_step : (other + 1) * rows_per_step]
            assert rel_rmse(wrong, got) > WRONG_STEP_MARGIN * max(rel_rmse(want, got), 1e-6), (
                f"param {param}: step {step} matches to {rel_rmse(want, got):.4f} but step {other} "
                f"to {rel_rmse(wrong, got):.4f} -- too close to detect a row-offset bug"
            )
            if param in (_SCALE_MSA, _SCALE_MLP):
                # Both legs fold the `1 +` into the scales. Dropping it on one side alone would
                # show up as a rel RMSE around the table's own scale, which the bound above catches.
                assert float(want.mean()) != 0.0
