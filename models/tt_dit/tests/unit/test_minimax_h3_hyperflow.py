# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for HyperFlow on the Turbo pipeline: the sampling contract read from the adapter
header, the schedule and endpoints the denoise loop is handed, and the float32 fusion of both time
embedders. Every failure here is silent on device -- a wrong grid, endpoint or embedder delta still
produces video."""

import json

import pytest
import torch
from safetensors.torch import save_file

from models.tt_dit.experimental.lora.h3_adapter_loader import h3_host_deltas
from models.tt_dit.pipelines.minimax_h3.hyperflow_minimax_h3 import MARKER_KEY, MiniMaxH3HyperFlow, validate_sigmas
from models.tt_dit.pipelines.minimax_h3.pipeline_minimax_h3 import AUDIO_SHIFT, VIDEO_SHIFT
from models.tt_dit.pipelines.minimax_h3.pipeline_minimax_h3_turbo import HYPERFLOW_HOST_PREFIXES, MiniMaxH3TurboPipeline
from models.tt_dit.pipelines.minimax_h3.scheduler import shift_sigmas

# The header of minimax_h3_hyperflow_8step_v1.0.safetensors as published.
GRID_8STEP = (1.0, 0.931506, 0.839236, 0.703462, 0.5, 0.296538, 0.160764, 0.068494, 0.0)
PUBLISHED = {
    MARKER_KEY: "true",
    "hyperflow_version": "1.0",
    "hyperflow_gate": "0.25",
    "hyperflow_sigmas": json.dumps(list(GRID_8STEP)),
    "hyperflow_video_shift": "12.0",
    "hyperflow_audio_shift": "3.0",
    "tasks": json.dumps(["t2va", "fl2va", "ref2va"]),
    "compatible_subfolders": json.dumps(["transformer", "transformer_ref"]),
    "lora_alpha": "256",
    "lora_rank": "256",
}


def _metadata(**overrides) -> dict[str, str]:
    metadata = dict(PUBLISHED)
    for key, value in overrides.items():
        if value is None:
            metadata.pop(key, None)
        else:
            metadata[key] = value
    return metadata


def _contract(**overrides) -> MiniMaxH3HyperFlow:
    contract = MiniMaxH3HyperFlow.from_adapter_metadata(
        _metadata(**overrides), video_shift=VIDEO_SHIFT, audio_shift=AUDIO_SHIFT
    )
    assert contract is not None
    return contract


# ---------------------------------------------------------------- the contract


def test_the_published_header_parses():
    contract = _contract()
    assert (contract.num_grid_points, contract.num_forwards) == (9, 8)
    assert contract.gate == 0.25
    assert contract.sigmas == GRID_8STEP


def test_a_turbo_adapter_publishes_no_contract():
    assert MiniMaxH3HyperFlow.from_adapter_metadata({"alpha": "8"}, video_shift=6.0, audio_shift=3.0) is None
    assert MiniMaxH3HyperFlow.from_adapter_metadata(None, video_shift=12.0, audio_shift=3.0) is None


@pytest.mark.parametrize("dropped", ["hyperflow_version", "hyperflow_gate", "hyperflow_sigmas"])
def test_a_half_written_contract_raises(dropped, expect_error):
    with expect_error(ValueError, "missing"):
        _contract(**{dropped: None})


def test_a_pipeline_at_other_shifts_is_refused(expect_error):
    # The 768p Turbo shift: a deployment left at MINIMAX_H3_VIDEO_SHIFT=6 must not serve this file.
    with expect_error(ValueError, "cannot be reproduced"):
        MiniMaxH3HyperFlow.from_adapter_metadata(_metadata(), video_shift=6.0, audio_shift=AUDIO_SHIFT)


def test_only_the_adapters_own_step_count_is_honoured(expect_error):
    contract = _contract()
    contract.assert_forwards(None)
    contract.assert_forwards(9)
    with expect_error(ValueError, "cannot be honoured"):
        contract.assert_forwards(50)


def test_task_and_partition_gates(expect_error):
    contract = _contract()
    for task in ("t2va", "fl2va", "ref2va"):
        contract.assert_supports_task(task)
    contract.assert_supports_subfolder("transformer_ref")
    with expect_error(ValueError, "transformer_ref"):
        _contract(compatible_subfolders=json.dumps(["transformer"])).assert_supports_subfolder("transformer_ref")


@pytest.mark.parametrize("shift", [VIDEO_SHIFT, AUDIO_SHIFT])
def test_the_grid_is_shifted_with_the_schedulers_own_formula(shift):
    sigmas = _contract().modality_sigmas(shift)
    assert torch.equal(sigmas, shift_sigmas(validate_sigmas(GRID_8STEP), shift))
    validate_sigmas(sigmas)


# ----------------------------------------------------- the pipeline's schedule hooks


def _pipeline(*, warming: bool, contract: MiniMaxH3HyperFlow | None) -> MiniMaxH3TurboPipeline:
    """The schedule hooks read four attributes; skip the mesh-bound constructor."""
    pipeline = object.__new__(MiniMaxH3TurboPipeline)
    pipeline.hyperflow = contract
    pipeline._warming = warming
    pipeline.video_shift = VIDEO_SHIFT
    pipeline.audio_shift = AUDIO_SHIFT
    return pipeline


def test_a_request_runs_the_adapters_grid_on_both_modalities():
    contract = _contract()
    video, audio = _pipeline(warming=False, contract=contract)._build_schedulers(9)
    assert torch.equal(video.sigmas, contract.modality_sigmas(VIDEO_SHIFT))
    assert torch.equal(audio.sigmas, contract.modality_sigmas(AUDIO_SHIFT))
    assert video.num_inference_steps == audio.num_inference_steps == 8


def test_a_request_at_another_step_count_is_refused(expect_error):
    with expect_error(ValueError, "cannot be honoured"):
        _pipeline(warming=False, contract=_contract())._build_schedulers(50)


def test_warmup_keeps_its_short_schedule():
    video, _ = _pipeline(warming=True, contract=_contract())._build_schedulers(3)
    assert video.num_inference_steps == 2


@pytest.mark.parametrize("warming", [False, True])
def test_the_endpoint_of_a_step_is_the_next_steps_timestep(warming):
    """`r_i = 1 - sigma_{i+1}`, so the last step aims at a clean sample. Warmup gets endpoints too,
    so it compiles the same forward a request runs."""
    pipeline = _pipeline(warming=warming, contract=_contract())
    video, audio = pipeline._build_schedulers(3 if warming else 9)
    endpoints = pipeline._step_endpoints(video, audio)
    assert endpoints is not None
    for scheduler, endpoint in zip((video, audio), endpoints):
        assert endpoint.numel() == scheduler.num_inference_steps
        assert torch.equal(endpoint[:-1], scheduler.timesteps[1:])
        assert float(endpoint[-1]) == 1.0


def test_a_turbo_adapter_stays_single_time():
    pipeline = _pipeline(warming=False, contract=None)
    video, audio = pipeline._build_schedulers(5)
    assert video.num_inference_steps == 4
    assert pipeline._step_endpoints(video, audio) is None


# -------------------------------------------------- float32 time-embedder fusion

FREQ, HIDDEN, OUT, RANK = 8, 12, 6, 4


def _low_rank(generator, out_features, in_features):
    return {
        "lora_A.weight": torch.randn(RANK, in_features, generator=generator),
        "lora_B.weight": torch.randn(out_features, RANK, generator=generator),
    }


@pytest.fixture
def hyperflow_files(tmp_path):
    generator = torch.Generator().manual_seed(0)
    adapter = {}
    for embedder in ("time_embedder", "endpoint_time_embedder"):
        for linear, (out_features, in_features) in (("linear_1", (HIDDEN, FREQ)), ("linear_2", (OUT, HIDDEN))):
            for slot, tensor in _low_rank(generator, out_features, in_features).items():
                adapter[f"transformer.{embedder}.{linear}.{slot}"] = tensor
    # A block target in bfloat16, which the host fold must leave to the device loader.
    adapter["transformer.transformer_blocks.0.attn.to_out.0.lora_A.weight"] = torch.randn(RANK, 16).bfloat16()
    adapter["transformer.transformer_blocks.0.attn.to_out.0.lora_B.weight"] = torch.randn(16, RANK).bfloat16()
    adapter_path = tmp_path / "hyperflow.safetensors"
    save_file(adapter, str(adapter_path), metadata=_metadata(lora_alpha="2", lora_rank=str(RANK)))

    checkpoint = {
        "time_embedder.linear_1.weight": torch.randn(HIDDEN, FREQ, generator=generator),
        "time_embedder.linear_1.bias": torch.randn(HIDDEN, generator=generator),
        "time_embedder.linear_2.weight": torch.randn(OUT, HIDDEN, generator=generator),
        "time_embedder.linear_2.bias": torch.randn(OUT, generator=generator),
        "proj_in.weight": torch.randn(4, 4, generator=generator),
    }
    weights_dir = tmp_path / "MiniMax-H3"
    (weights_dir / "transformer").mkdir(parents=True)
    save_file(checkpoint, str(weights_dir / "transformer" / "diffusion_pytorch_model.safetensors"))
    return adapter_path, adapter, weights_dir, checkpoint


def test_host_deltas_cover_only_the_time_embedders_at_alpha_over_rank(hyperflow_files):
    adapter_path, adapter, _, _ = hyperflow_files
    deltas = h3_host_deltas(str(adapter_path), HYPERFLOW_HOST_PREFIXES, scale=0.5)
    assert sorted(deltas) == sorted(
        f"{embedder}.{linear}.weight"
        for embedder in ("time_embedder", "endpoint_time_embedder")
        for linear in ("linear_1", "linear_2")
    )
    key = "transformer.endpoint_time_embedder.linear_2"
    expected = 0.5 * (2 / RANK) * (adapter[f"{key}.lora_B.weight"] @ adapter[f"{key}.lora_A.weight"])
    delta = deltas["endpoint_time_embedder.linear_2.weight"]
    assert delta.dtype == torch.float32
    assert torch.equal(delta, expected)


def test_both_embedders_start_from_the_base_weights_with_their_own_delta(hyperflow_files):
    adapter_path, _, weights_dir, checkpoint = hyperflow_files
    pipeline = object.__new__(MiniMaxH3TurboPipeline)
    pipeline.lora_path = adapter_path
    pipeline.lora_strength = 1.0
    pipeline.weights_dir = weights_dir
    pipeline.transformer_subfolder = "transformer"

    states = pipeline._fused_time_embedder_states()
    deltas = h3_host_deltas(str(adapter_path), HYPERFLOW_HOST_PREFIXES)
    assert sorted(states) == ["endpoint_time_embedder", "time_embedder"]
    for embedder, state in states.items():
        for key, value in state.items():
            base = checkpoint[f"time_embedder.{key}"]
            expected = base + deltas[f"{embedder}.{key}"] if key.endswith("weight") else base
            assert value.dtype == torch.float32
            assert torch.equal(value, expected), f"{embedder}.{key}"
    # Seeded from one base, adapted apart: the endpoint embedder is not the base embedder.
    assert not torch.equal(
        states["time_embedder"]["linear_1.weight"], states["endpoint_time_embedder"]["linear_1.weight"]
    )
    # The fold must not write back into the base it read.
    assert torch.equal(
        pipeline._read_checkpoint_tensors(["time_embedder.linear_1.weight"])["time_embedder.linear_1.weight"],
        checkpoint["time_embedder.linear_1.weight"],
    )


def test_an_adapter_missing_an_embedder_target_is_refused(hyperflow_files, tmp_path, expect_error):
    adapter_path, adapter, weights_dir, _ = hyperflow_files
    partial = {k: v for k, v in adapter.items() if "endpoint_time_embedder.linear_2" not in k}
    partial_path = tmp_path / "partial.safetensors"
    save_file(partial, str(partial_path), metadata=_metadata())
    pipeline = object.__new__(MiniMaxH3TurboPipeline)
    pipeline.lora_path = partial_path
    pipeline.lora_strength = 1.0
    pipeline.weights_dir = weights_dir
    pipeline.transformer_subfolder = "transformer"
    with expect_error(RuntimeError, "endpoint_time_embedder.linear_2.weight"):
        pipeline._fused_time_embedder_states()
