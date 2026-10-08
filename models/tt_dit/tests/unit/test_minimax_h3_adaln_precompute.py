# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for the precomputed AdaLN table on a tiny fake checkpoint: the table against a torch
reference of the on-device path, the adapter folds, row matching, and the HyperFlow pipeline's table
build, disk cache and weight filtering. A wrong table still produces video, so none of this fails on
device."""

import json
import math

import pytest
import torch
from safetensors.torch import save_file

import ttnn
from models.tt_dit.pipelines.minimax_h3 import pipeline_minimax_h3_turbo as turbo
from models.tt_dit.pipelines.minimax_h3.adaln_precompute import (
    MINIMAX_H3_ADALN_PARAMS,
    MINIMAX_H3_MODALITY_NUM,
    AdalnTwoTime,
    MiniMaxH3AdalnLoraFold,
    precompute_adaln_table,
    request_step_levels,
    slot_table_rows,
)
from models.tt_dit.pipelines.minimax_h3.hyperflow_minimax_h3 import MARKER_KEY, MiniMaxH3HyperFlow
from models.tt_dit.pipelines.minimax_h3.packing import MINIMAX_H3_KEYFRAME_NOISE_AUG
from models.tt_dit.pipelines.minimax_h3.pipeline_minimax_h3 import (
    AUDIO_SHIFT,
    MINIMAX_H3_AUDIO_CONDITION_TIMESTEP,
    VIDEO_SHIFT,
)
from models.tt_dit.pipelines.minimax_h3.pipeline_minimax_h3_turbo import MiniMaxH3TurboPipeline

FREQ, TEMB_HIDDEN, TEMB, HIDDEN, LAYERS, RANK = 8, 12, 6, 4, 2, 2
GATE = 0.25
GRID = (1.0, 0.931506, 0.839236, 0.703462, 0.5, 0.296538, 0.160764, 0.068494, 0.0)
METADATA = {
    MARKER_KEY: "true",
    "hyperflow_version": "1.0",
    "hyperflow_gate": str(GATE),
    "hyperflow_sigmas": json.dumps(list(GRID)),
    "hyperflow_video_shift": str(VIDEO_SHIFT),
    "hyperflow_audio_shift": str(AUDIO_SHIFT),
    "lora_alpha": str(RANK),
}


def _checkpoint(generator) -> dict[str, torch.Tensor]:
    """Diffusers spelling, fp32 time embedder and bf16 projections, as the published snapshot."""
    state = {
        "time_embedder.linear_1.weight": torch.randn(TEMB_HIDDEN, FREQ, generator=generator),
        "time_embedder.linear_1.bias": torch.randn(TEMB_HIDDEN, generator=generator),
        "time_embedder.linear_2.weight": torch.randn(TEMB, TEMB_HIDDEN, generator=generator),
        "time_embedder.linear_2.bias": torch.randn(TEMB, generator=generator),
        "norm_out.linear.weight": torch.randn(2 * HIDDEN, TEMB, generator=generator).bfloat16(),
        "norm_out.linear.bias": torch.randn(2 * HIDDEN, generator=generator).bfloat16(),
        "proj_in.weight": torch.randn(4, 4, generator=generator),
    }
    for layer in range(LAYERS):
        out = MINIMAX_H3_MODALITY_NUM * MINIMAX_H3_ADALN_PARAMS * HIDDEN
        state[f"transformer_blocks.{layer}.adaln_proj.linear.weight"] = torch.randn(
            out, TEMB, generator=generator
        ).bfloat16()
        state[f"transformer_blocks.{layer}.adaln_proj.linear.bias"] = torch.randn(out, generator=generator).bfloat16()
        state[f"transformer_blocks.{layer}.attn.to_out.0.weight"] = torch.randn(HIDDEN, HIDDEN, generator=generator)
    return state


@pytest.fixture
def checkpoint(tmp_path):
    state = _checkpoint(torch.Generator().manual_seed(0))
    directory = tmp_path / "MiniMax-H3" / "transformer"
    directory.mkdir(parents=True)
    save_file(state, str(directory / "diffusion_pytorch_model.safetensors"))
    return directory, state


# ---------------------------------------------------------- torch reference of the device path


def _reference_temb(t: torch.Tensor, state: dict[str, torch.Tensor], prefix: str = "time_embedder") -> torch.Tensor:
    """`time_proj` (cos before sin) then the fp32 embedder, as the device runs it."""
    half = FREQ // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    sinusoid = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    hidden = torch.nn.functional.silu(
        sinusoid @ state[f"{prefix}.linear_1.weight"].float().T + state[f"{prefix}.linear_1.bias"].float()
    )
    return hidden @ state[f"{prefix}.linear_2.weight"].float().T + state[f"{prefix}.linear_2.bias"].float()


def _reference_rows(levels, state, endpoint_state=None, gate=0.0):
    """Per level: the blended temb, SiLU in fp32, cast to bf16, then each bf16 projection."""
    temb = _reference_temb(levels[:, 0], state)
    if gate:
        temb = temb + gate * (_reference_temb(levels[:, 1], endpoint_state) - temb)
    activated = torch.nn.functional.silu(temb).bfloat16()
    blocks = []
    for layer in range(LAYERS):
        prefix = f"transformer_blocks.{layer}.adaln_proj.linear"
        projected = torch.nn.functional.linear(activated, state[f"{prefix}.weight"], state[f"{prefix}.bias"])
        # [levels, modality * 6 * hidden] -> [levels * modality, 6, hidden]
        blocks.append(projected.view(-1, MINIMAX_H3_ADALN_PARAMS, HIDDEN))
    final = torch.nn.functional.linear(activated, state["norm_out.linear.weight"], state["norm_out.linear.bias"])
    return torch.stack(blocks), final[:, :HIDDEN], final[:, HIDDEN:]


def _levels():
    sigmas = torch.tensor([1.0, 0.8, 0.3, 0.0])
    endpoints = 1.0 - sigmas[1:]
    return request_step_levels(
        sigmas,
        sigmas,
        MINIMAX_H3_KEYFRAME_NOISE_AUG,
        audio_condition_timestep=MINIMAX_H3_AUDIO_CONDITION_TIMESTEP,
        video_endpoints=endpoints,
        audio_endpoints=endpoints,
    )


def _assert_matches_reference(table, step_levels, state, endpoint_state=None, gate=0.0):
    for step, levels in enumerate(step_levels):
        lo, hi = int(table.step_offsets[step]), int(table.step_offsets[step + 1])
        assert torch.equal(table.step_levels(step), levels)
        blocks, shift, scale = _reference_rows(levels, state, endpoint_state, gate)
        rows = slice(lo * MINIMAX_H3_MODALITY_NUM, hi * MINIMAX_H3_MODALITY_NUM)
        torch.testing.assert_close(table.block_params[:, rows], blocks, rtol=0, atol=0.02)
        torch.testing.assert_close(table.final_shift[lo:hi], shift, rtol=0, atol=0.02)
        torch.testing.assert_close(table.final_scale[lo:hi], scale, rtol=0, atol=0.02)


def _build(directory, step_levels, **kwargs):
    return precompute_adaln_table(
        directory, step_levels, num_layers=LAYERS, hidden_size=HIDDEN, freq_dim=FREQ, **kwargs
    )


def test_the_table_matches_the_device_path(checkpoint):
    directory, state = checkpoint
    step_levels = _levels()
    table = _build(directory, step_levels)
    assert table.num_steps == len(step_levels)
    assert table.block_params.dtype == torch.bfloat16
    assert table.block_params.shape[1] == sum(int(l.shape[0]) for l in step_levels) * MINIMAX_H3_MODALITY_NUM
    _assert_matches_reference(table, step_levels, state)


def test_the_two_time_blend_matches_the_device_path(checkpoint):
    directory, state = checkpoint
    step_levels = _levels()
    endpoint_state = {k: v.clone() for k, v in state.items()}
    endpoint_state["time_embedder.linear_1.weight"] += 0.5
    delta = {"endpoint_time_embedder.linear_1.weight": torch.full((TEMB_HIDDEN, FREQ), 0.5)}
    two_time = AdalnTwoTime(gate=GATE, weight_hook=MiniMaxH3AdalnLoraFold.endpoint(delta))
    table = _build(directory, step_levels, two_time=two_time)
    assert two_time.weight_hook.unapplied() == []
    _assert_matches_reference(table, step_levels, state, endpoint_state, GATE)


def test_gate_zero_is_the_single_time_table_bit_for_bit(checkpoint):
    directory, _ = checkpoint
    step_levels = _levels()
    # The endpoints differ from t, so only the gate can make the blend vanish.
    assert any(bool((levels[:, 0] != levels[:, 1]).any()) for levels in step_levels)
    single = _build(directory, step_levels)
    gated = _build(directory, step_levels, two_time=AdalnTwoTime(gate=0.0))
    for name in ("block_params", "final_shift", "final_scale"):
        assert torch.equal(getattr(single, name), getattr(gated, name)), name


def test_a_folded_delta_reaches_the_table(checkpoint):
    directory, state = checkpoint
    step_levels = _levels()
    deltas = {
        "time_embedder.linear_2.weight": torch.full((TEMB, TEMB_HIDDEN), 0.1),
        "transformer_blocks.1.adaln_proj.linear.weight": torch.full(
            state["transformer_blocks.1.adaln_proj.linear.weight"].shape, 0.25
        ),
        "norm_out.linear.weight": torch.full((2 * HIDDEN, TEMB), -0.125),
    }
    fold = MiniMaxH3AdalnLoraFold(deltas)
    table = _build(directory, step_levels, weight_hook=fold)
    assert fold.unapplied() == []
    folded = dict(state)
    for key, delta in deltas.items():
        fused = state[key].float() + delta
        folded[key] = fused if key.startswith("time_embedder.") else fused.to(state[key].dtype)
    _assert_matches_reference(table, step_levels, folded)
    unfolded = _build(directory, step_levels)
    assert not torch.equal(table.block_params[1], unfolded.block_params[1])


# ---------------------------------------------------------------------------- the folds


def test_a_fold_answers_to_both_checkpoint_spellings():
    delta = torch.ones(3, 2)
    fold = MiniMaxH3AdalnLoraFold({"transformer_blocks.7.adaln_proj.linear.weight": delta})
    original = fold("blocks.7.adaln_proj.linear.weight", torch.zeros(3, 2, dtype=torch.bfloat16))
    assert original.dtype == torch.bfloat16
    assert torch.equal(original.float(), delta)
    assert fold.unapplied() == []
    assert fold.targets() == ["transformer_blocks.7.adaln_proj.linear.weight"]


def test_an_unmatched_entry_is_reported_unapplied():
    fold = MiniMaxH3AdalnLoraFold(
        {
            "time_embedder.linear_1.weight": torch.ones(2, 2),
            "norm_out.linear.weight": torch.ones(2, 2),
        }
    )
    # The original-release spelling of the time embedder; it stays float32.
    out = fold("time_embedder.proj_in.weight", torch.zeros(2, 2, dtype=torch.bfloat16))
    assert out.dtype == torch.float32
    assert fold("proj_in.weight", torch.zeros(2, 2)).abs().sum() == 0
    assert fold.unapplied() == ["norm_out.linear.weight"]


def test_a_shape_mismatch_is_refused(expect_error):
    fold = MiniMaxH3AdalnLoraFold({"norm_out.linear.weight": torch.ones(2, 3)})
    with expect_error(ValueError, "does not match checkpoint"):
        fold("norm_out.linear.weight", torch.zeros(3, 2))


def test_the_endpoint_fold_renames_onto_the_base_embedder_and_drops_the_rest(expect_error):
    endpoint_delta = torch.full((2, 2), 3.0)
    deltas = {
        "endpoint_time_embedder.linear_1.weight": endpoint_delta,
        "time_embedder.linear_1.weight": torch.ones(2, 2),
        "transformer_blocks.0.adaln_proj.linear.weight": torch.ones(2, 2),
    }
    fold = MiniMaxH3AdalnLoraFold.endpoint(deltas)
    assert fold.targets() == ["time_embedder.linear_1.weight"]
    assert torch.equal(fold("time_embedder.linear_1.weight", torch.zeros(2, 2)), endpoint_delta)
    with expect_error(ValueError, "MiniMaxH3AdalnLoraFold.endpoint"):
        MiniMaxH3AdalnLoraFold(deltas)


# ----------------------------------------------------------------------- row matching


def test_slot_rows_match_their_level_by_value():
    sigmas = torch.tensor([1.0, 0.8, 0.3, 0.0])
    audio_sigmas = torch.tensor([1.0, 0.6, 0.2, 0.0])
    video_r, audio_r = 1.0 - sigmas[1:], 1.0 - audio_sigmas[1:]
    step_levels = request_step_levels(
        sigmas,
        audio_sigmas,
        MINIMAX_H3_KEYFRAME_NOISE_AUG,
        audio_condition_timestep=MINIMAX_H3_AUDIO_CONDITION_TIMESTEP,
        video_endpoints=video_r,
        audio_endpoints=audio_r,
    )
    table = _fake_table(step_levels)
    roles = ("video", "audio", "condition_video", "condition_audio")
    rows = slot_table_rows(
        table,
        roles,
        sigmas,
        audio_sigmas,
        MINIMAX_H3_KEYFRAME_NOISE_AUG,
        MINIMAX_H3_AUDIO_CONDITION_TIMESTEP,
        video_endpoints=video_r,
        audio_endpoints=audio_r,
    )
    assert len(rows) == 3
    for step, step_rows in enumerate(rows):
        t_video, t_audio = 1.0 - float(sigmas[step]), 1.0 - float(audio_sigmas[step])
        pinned = max(t_video, MINIMAX_H3_KEYFRAME_NOISE_AUG)
        expected = [
            (t_video, float(video_r[step])),
            (t_audio, float(audio_r[step])),
            (pinned, pinned),
            (MINIMAX_H3_AUDIO_CONDITION_TIMESTEP,) * 2,
        ]
        for row, pair in zip(step_rows.tolist(), expected):
            assert table.levels[row].tolist() == pytest.approx(list(pair))
            assert int(table.step_offsets[step]) <= row < int(table.step_offsets[step + 1])


def test_a_level_the_table_lacks_raises(expect_error):
    sigmas = torch.tensor([1.0, 0.5, 0.0])
    # Built without the ref2va audio-conditioning level.
    table = _fake_table(request_step_levels(sigmas, sigmas, MINIMAX_H3_KEYFRAME_NOISE_AUG))
    with expect_error(IndexError, "condition_audio"):
        slot_table_rows(
            table,
            ("video", "condition_audio"),
            sigmas,
            sigmas,
            MINIMAX_H3_KEYFRAME_NOISE_AUG,
            MINIMAX_H3_AUDIO_CONDITION_TIMESTEP,
        )


def _fake_table(step_levels):
    from models.tt_dit.pipelines.minimax_h3.adaln_precompute import MiniMaxH3AdalnTable

    counts = torch.tensor([int(levels.shape[0]) for levels in step_levels])
    return MiniMaxH3AdalnTable(
        levels=torch.cat(step_levels),
        step_offsets=torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)]),
        block_params=torch.zeros(1, int(counts.sum()) * MINIMAX_H3_MODALITY_NUM, MINIMAX_H3_ADALN_PARAMS, 1),
        final_shift=torch.zeros(int(counts.sum()), 1),
        final_scale=torch.zeros(int(counts.sum()), 1),
    )


# ------------------------------------------------------------------ the HyperFlow pipeline


def _adapter(path, generator) -> None:
    adapter = {}
    for embedder in ("time_embedder", "endpoint_time_embedder"):
        for linear, (out_features, in_features) in (
            ("linear_1", (TEMB_HIDDEN, FREQ)),
            ("linear_2", (TEMB, TEMB_HIDDEN)),
        ):
            adapter[f"transformer.{embedder}.{linear}.lora_A.weight"] = torch.randn(
                RANK, in_features, generator=generator
            )
            adapter[f"transformer.{embedder}.{linear}.lora_B.weight"] = torch.randn(
                out_features, RANK, generator=generator
            )
    save_file(adapter, str(path), metadata=METADATA)


class _FakeAdalnCache:
    """Stands in for the device upload; keeps the table it was handed."""

    def __init__(self, table, *, mesh_device, parallel_config, num_layers, hidden_size):
        assert (num_layers, hidden_size) == (LAYERS, HIDDEN)
        self.table = table
        self.num_steps = table.num_steps

    def assert_covers(self, num_inference_steps):
        assert num_inference_steps == self.num_steps


def _pipeline(weights_dir, adapter_path, *, task="t2va", strength=1.0, hyperflow=True) -> MiniMaxH3TurboPipeline:
    """The AdaLN hooks read these attributes; skip the mesh-bound constructor."""
    pipeline = object.__new__(MiniMaxH3TurboPipeline)
    pipeline.hyperflow = (
        MiniMaxH3HyperFlow.from_adapter_metadata(METADATA, video_shift=VIDEO_SHIFT, audio_shift=AUDIO_SHIFT)
        if hyperflow
        else None
    )
    pipeline.video_shift, pipeline.audio_shift = VIDEO_SHIFT, AUDIO_SHIFT
    pipeline.task = task
    pipeline.weights_dir = weights_dir
    pipeline.transformer_subfolder = "transformer"
    pipeline.transformer_config = {"num_layers": LAYERS, "hidden_size": HIDDEN, "freq_dim": FREQ}
    pipeline.lora_path = adapter_path
    pipeline.lora_strength = strength
    pipeline.mesh_device = pipeline.dit_parallel_config = None
    pipeline.dit_fsdp = False
    pipeline._adaln_cache = pipeline._adaln_table = None
    pipeline._adaln_slot_rows = {}
    pipeline._lora_digest = None
    return pipeline


@pytest.fixture
def hyperflow(checkpoint, tmp_path, monkeypatch):
    directory, state = checkpoint
    adapter_path = tmp_path / "hyperflow.safetensors"
    _adapter(adapter_path, torch.Generator().manual_seed(1))
    monkeypatch.setenv("TT_DIT_CACHE_DIR", str(tmp_path / "cache"))
    # Asking ttnn opens the cluster; these runs are single-process by construction.
    monkeypatch.setattr(ttnn, "using_distributed_env", lambda: False)
    monkeypatch.setattr(turbo, "MiniMaxH3AdalnCache", _FakeAdalnCache)
    return directory.parent, adapter_path, state


@pytest.mark.parametrize("task", ["t2va", "ref2va"])
def test_the_pipeline_builds_the_contract_table_and_caches_it(hyperflow, tmp_path, monkeypatch, task):
    weights_dir, adapter_path, state = hyperflow
    roles = ("video", "audio", "condition_video") + (("condition_audio",) if task == "ref2va" else ())
    pipeline = _pipeline(weights_dir, adapter_path, task=task)
    assert pipeline._precomputed_adaln()
    cache, rows = pipeline._prepare_adaln(roles)

    contract = pipeline.hyperflow
    assert cache.num_steps == len(rows) == contract.num_forwards
    assert all(step_rows.numel() == len(roles) for step_rows in rows)
    # Memoised per pipeline: the same upload serves every call, so traces stay bound to it.
    again, rows_again = pipeline._prepare_adaln(roles)
    assert again is cache and rows_again is rows

    # The table against the reference, with both host folds applied.
    deltas = pipeline._hyperflow_host_deltas()
    folded, endpoint = dict(state), dict(state)
    for key in turbo._TIME_EMBEDDER_KEYS:
        if key.endswith("weight"):
            folded[f"time_embedder.{key}"] = state[f"time_embedder.{key}"] + deltas[f"time_embedder.{key}"]
            endpoint[f"time_embedder.{key}"] = state[f"time_embedder.{key}"] + deltas[f"endpoint_time_embedder.{key}"]
    video, audio = pipeline._contract_schedulers()
    step_levels = request_step_levels(
        video.sigmas,
        audio.sigmas,
        MINIMAX_H3_KEYFRAME_NOISE_AUG,
        audio_condition_timestep=MINIMAX_H3_AUDIO_CONDITION_TIMESTEP if task == "ref2va" else None,
        video_endpoints=contract.endpoints(video.sigmas),
        audio_endpoints=contract.endpoints(audio.sigmas),
    )
    _assert_matches_reference(cache.table, step_levels, folded, endpoint, contract.gate)

    path = pipeline._adaln_cache_path(video, audio)
    assert path.is_file() and path.parent == tmp_path / "cache" / "minimax-h3-adaln"
    assert not list(path.parent.glob("*.tmp"))

    # A second pipeline reads the file instead of the checkpoint.
    def no_build(*args, **kwargs):
        raise AssertionError("rebuilt a cached table")

    monkeypatch.setattr(turbo, "precompute_adaln_table", no_build)
    cached, cached_rows = _pipeline(weights_dir, adapter_path, task=task)._prepare_adaln(roles)
    assert torch.equal(cached.table.block_params, cache.table.block_params)
    assert all(torch.equal(a, b) for a, b in zip(cached_rows, rows))


def test_the_cache_key_separates_what_changes_the_rows(hyperflow, tmp_path):
    weights_dir, adapter_path, _ = hyperflow
    base = _pipeline(weights_dir, adapter_path)
    schedulers = base._contract_schedulers()
    path = base._adaln_cache_path(*schedulers)
    assert path.name.endswith(".adaln.pt")
    assert _pipeline(weights_dir, adapter_path)._adaln_cache_path(*schedulers) == path
    assert _pipeline(weights_dir, adapter_path, strength=0.5)._adaln_cache_path(*schedulers) != path
    assert _pipeline(weights_dir, adapter_path, task="ref2va")._adaln_cache_path(*schedulers) != path
    other = tmp_path / "other.safetensors"
    _adapter(other, torch.Generator().manual_seed(2))
    assert _pipeline(weights_dir, other)._adaln_cache_path(*schedulers) != path
    gated = _pipeline(weights_dir, adapter_path)
    gated.hyperflow = MiniMaxH3HyperFlow.from_adapter_metadata(
        {**METADATA, "hyperflow_gate": "0.5"}, video_shift=VIDEO_SHIFT, audio_shift=AUDIO_SHIFT
    )
    assert gated._adaln_cache_path(*schedulers) != path


def test_an_adapter_without_an_endpoint_half_is_refused(hyperflow, tmp_path, expect_error):
    weights_dir, adapter_path, _ = hyperflow
    from safetensors.torch import load_file

    partial = {k: v for k, v in load_file(str(adapter_path)).items() if "endpoint_time_embedder" not in k}
    partial_path = tmp_path / "partial.safetensors"
    save_file(partial, str(partial_path), metadata=METADATA)
    with expect_error(RuntimeError, "endpoint_time_embedder"):
        _pipeline(weights_dir, partial_path)._prepare_adaln(("video",))


def test_a_turbo_adapter_keeps_the_device_projections(hyperflow):
    weights_dir, adapter_path, _ = hyperflow
    pipeline = _pipeline(weights_dir, adapter_path, hyperflow=False)
    assert not pipeline._precomputed_adaln()
    assert pipeline._dit_weight_mode() == "resident_adaln"


@pytest.mark.parametrize("precomputed", [False, True])
def test_read_safetensors_drops_only_what_the_table_replaces(hyperflow, precomputed):
    weights_dir, adapter_path, state = hyperflow
    pipeline = _pipeline(weights_dir, adapter_path, hyperflow=precomputed)
    read = pipeline._read_safetensors("transformer")
    replaced = {
        key
        for key in state
        if ".adaln_proj." in key or key.startswith("time_embedder.") or key.startswith("norm_out.linear")
    }
    assert set(read) == (set(state) - replaced if precomputed else set(state))
    assert pipeline._dit_weight_mode() == ("precomputed_adaln" if precomputed else "resident_adaln")
    pipeline.dit_fsdp = True
    assert pipeline._dit_weight_mode().endswith("_fsdp")


def test_read_safetensors_keeps_other_partitions_whole(hyperflow):
    weights_dir, adapter_path, _ = hyperflow
    vae = {"time_embedder.linear_1.weight": torch.ones(1), "norm_out.linear.weight": torch.ones(1)}
    (weights_dir / "vae").mkdir()
    save_file(vae, str(weights_dir / "vae" / "diffusion_pytorch_model.safetensors"))
    assert set(_pipeline(weights_dir, adapter_path)._read_safetensors("vae")) == set(vae)
