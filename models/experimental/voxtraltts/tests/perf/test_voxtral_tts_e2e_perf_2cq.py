# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Voxtral TTS 2CQ E2E perf on P150: non-traced ``forward`` and traced generation loop (+ PCC gates)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import comp_pcc, profiler, run_for_blackhole
from models.demos.utils.common_demo_utils import get_mesh_mappers
from models.experimental.voxtraltts.reference.cpu_reference import VoxtralCPUReference
from models.experimental.voxtraltts.reference.voxtral_request import compose_speech_request
from models.experimental.voxtraltts.tests.common import resolve_voxtral_model_name_or_skip
from models.experimental.voxtraltts.tests.pcc.test_ttnn_voxtral_tts_e2e import (
    ACOUSTIC_MATCH_FRAC,
    TEXT_DECODE_STEP_PCC,
    WAVEFORM_PCC,
    _align_to_ref_shape,
    _cpu_text_decode_step,
)
from models.experimental.voxtraltts.tt.voxtral_tts import ACOUSTIC_CFG_ALPHA_DEFAULT, VoxtralTTSPipeline
from models.perf.perf_utils import prep_perf_report
from models.tt_cnn.tt.pipeline import (
    PipelineConfig,
    create_pipeline_from_config,
    get_memory_config_for_persistent_dram_tensor,
)

_DEMO_TEXT = "Hello from the Voxtral Tenstorrent demo."
_DEMO_VOICE = "casual_male"
_NUM_MEASUREMENT_ITERS = 4
_DUMMY_WIDTH = 32
_GENERATE_MAX_TOKENS = 8
_TEXT_MAX_SEQ_LEN = 512

_EXPECTED_E2E_FULL_THROUGHPUT_FPS = 0.25
_EXPECTED_E2E_TRACE_GEN_LOOP_FPS = 2.0

_DEVICE_PARAMS_2CQ = {"num_command_queues": 2}
_DEVICE_PARAMS_2CQ_TRACE = {
    "l1_small_size": 32768,
    "trace_region_size": 500_000_000,
    "num_command_queues": 2,
}


def _determine_num_cores_for_even_sharding(shard_dim: int, max_cores: int) -> int:
    number_of_cores = max_cores
    while shard_dim % number_of_cores != 0:
        assert number_of_cores > 0, "Unable to find core grid"
        number_of_cores -= 1
    return number_of_cores


def _dummy_height_for_tile_height_sharded_io(device: ttnn.Device) -> int:
    dram_max = int(device.dram_grid_size().x)
    for h in range(32, 262_144 + 1, 32):
        dram_n = _determine_num_cores_for_even_sharding(h, dram_max)
        dram_shard_h = h // dram_n
        if dram_shard_h % 32 != 0:
            continue
        if h % 8 == 0:
            l1_shard_h = h // 8
        else:
            l1_shard_h = h // 4
        if l1_shard_h % 32 != 0:
            continue
        return h
    raise RuntimeError(f"No TILE-safe dummy height found for dram_grid_x={dram_max}")


def _get_l1_input_memory_config(host_input: ttnn.Tensor) -> ttnn.MemoryConfig:
    height, width = int(host_input.shape[-2]), int(host_input.shape[-1])
    core_grid = ttnn.CoreGrid(x=8, y=1)
    if height % int(core_grid.num_cores) != 0:
        core_grid = ttnn.CoreGrid(x=4, y=1)
    return ttnn.create_sharded_memory_config(
        shape=(height // int(core_grid.num_cores), width),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _setup_sharded_pipeline_input(device: ttnn.Device) -> tuple[ttnn.Tensor, ttnn.MemoryConfig, ttnn.MemoryConfig]:
    inputs_mesh_mapper, _, _ = get_mesh_mappers(device)
    h = _dummy_height_for_tile_height_sharded_io(device)
    torch_z = torch.zeros((1, h, _DUMMY_WIDTH), dtype=torch.bfloat16)
    dummy_host = ttnn.from_torch(
        torch_z,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=inputs_mesh_mapper,
    )
    dram_config = get_memory_config_for_persistent_dram_tensor(
        dummy_host.shape, ttnn.TensorMemoryLayout.HEIGHT_SHARDED, device.dram_grid_size()
    )
    l1_config = _get_l1_input_memory_config(dummy_host)
    return dummy_host, dram_config, l1_config


@dataclass
class _TextDecodeTraceStep:
    x_embed_tt: ttnn.Tensor
    current_pos_tt: ttnn.Tensor
    rot_mats_global: object
    rot_mats_local: object | None
    page_table: ttnn.Tensor | None


@dataclass
class _VoxtralTraceGenStep:
    llm_hidden_tt: ttnn.Tensor
    noise_tt: ttnn.Tensor
    text_step: _TextDecodeTraceStep


@dataclass
class _E2EGoldenRef:
    """Expected non-traced E2E output: TT autoregressive codes + CPU-ref waveform decode."""

    codes_b37t: torch.Tensor
    waveform: torch.Tensor


def _stacked_codes_to_b37t(stacked: torch.Tensor, end_audio_id: int) -> torch.Tensor:
    eoa = (stacked[:, 0] == end_audio_id).nonzero(as_tuple=False)
    cut = int(eoa[0].item()) if len(eoa) else stacked.shape[0]
    shifted = stacked[:cut]
    audio_tokens = shifted - 2
    return audio_tokens.T.unsqueeze(0).long()


@dataclass
class _VoxtralTraceE2EBundle:
    steps: list[_VoxtralTraceGenStep]
    ref_last_hidden: torch.Tensor
    ref_last_trace_codes: torch.Tensor


def _upload_hidden_tt(pipe: VoxtralTTSPipeline, hidden: torch.Tensor) -> ttnn.Tensor:
    h_b1d = hidden.reshape(1, 1, -1).to(torch.bfloat16)
    return ttnn.from_torch(
        h_b1d,
        device=pipe.mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(pipe.mesh_device),
    )


def _upload_noise_tt(pipe: VoxtralTTSPipeline, noise: torch.Tensor) -> ttnn.Tensor:
    return ttnn.from_torch(
        noise.to(torch.bfloat16).unsqueeze(1),
        device=pipe.mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(pipe.mesh_device),
    )


def _upload_embed_tt(pipe: VoxtralTTSPipeline, x_embed: torch.Tensor) -> ttnn.Tensor:
    dim = pipe.text.inner.args.dim
    x_4d = x_embed.reshape(1, 1, 1, dim).to(dtype=torch.bfloat16).contiguous()
    return ttnn.from_torch(
        x_4d,
        device=pipe.mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(pipe.mesh_device),
    )


def _build_text_decode_trace_step(
    pipe: VoxtralTTSPipeline, x_embed: torch.Tensor, pos_idx: int
) -> _TextDecodeTraceStep:
    x_embed_tt = _upload_embed_tt(pipe, x_embed)
    dummy_token = torch.zeros(1, dtype=torch.int64)
    current_pos_t = torch.tensor([pos_idx], dtype=torch.int64)
    _, current_pos_tt, rope_idxs, page_table = pipe.text.prepare_inputs_decode(dummy_token, current_pos_t)
    rot_mats_global = pipe.text.inner.rope_setup.get_rot_mats(rope_idxs)
    rot_mats_local = (
        pipe.text.inner.rope_local_setup.get_rot_mats(rope_idxs)
        if hasattr(pipe.text.inner, "rope_local_setup")
        else None
    )
    return _TextDecodeTraceStep(
        x_embed_tt=x_embed_tt,
        current_pos_tt=current_pos_tt,
        rot_mats_global=rot_mats_global,
        rot_mats_local=rot_mats_local,
        page_table=page_table,
    )


def _materialize_voxtral_trace_e2e_bundle(
    pipe: VoxtralTTSPipeline,
    device: ttnn.Device,
    *,
    num_steps: int,
) -> _VoxtralTraceE2EBundle:
    """Prefill on device; per-step FM noise + decode inputs from CPU/TT acoustic (no device decode here)."""
    speech_request = compose_speech_request(_DEMO_TEXT, pipe.model_name_or_path, voice=_DEMO_VOICE, ref_audio=None)
    prompt_ids = speech_request["prompt_token_ids"]
    prompt_len = len(prompt_ids)

    embeds = pipe._build_voice_injected_embeds(prompt_ids, _DEMO_VOICE)
    tt_hidden = pipe.text.prefill_from_embeds(embeds, start_pos=0)

    cpu = _load_cpu_reference_or_skip(pipe.model_name_or_path)

    _, cpu_embeds = cpu._prompt_embeddings(prompt_ids, _DEMO_VOICE)
    cpu_prefill = cpu.text_model(
        inputs_embeds=cpu_embeds.unsqueeze(0),
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )
    cpu_hidden = cpu_prefill.hidden_states[-1][:, -1, :].squeeze(0)
    cpu_pkv = cpu_prefill.past_key_values

    cfg_alpha = torch.tensor(ACOUSTIC_CFG_ALPHA_DEFAULT, dtype=torch.bfloat16)
    n_acoustic = pipe.acoustic.n_acoustic_out
    steps: list[_VoxtralTraceGenStep] = []

    for step in range(num_steps):
        llm_hidden_tt = _upload_hidden_tt(pipe, tt_hidden)
        torch.manual_seed(10_000 + step)
        noise_tt = _upload_noise_tt(
            pipe,
            torch.randn(1, n_acoustic, dtype=torch.bfloat16),
        )
        tt_codes = pipe.acoustic_codes_forward(tt_hidden.unsqueeze(0), cfg_alpha).long()
        mm_embed = pipe._audio_codes_to_mm_embed(tt_codes)
        text_step = _build_text_decode_trace_step(pipe, mm_embed, prompt_len + step)
        steps.append(
            _VoxtralTraceGenStep(llm_hidden_tt=llm_hidden_tt, noise_tt=noise_tt, text_step=text_step),
        )

        cpu_hidden, cpu_pkv = _cpu_text_decode_step(
            cpu,
            audio_codes_b37=tt_codes,
            past_key_values=cpu_pkv,
        )
        tt_hidden = cpu_hidden.to(torch.bfloat16)

    last_step = steps[-1]
    llm_tile = pipe._acoustic_hidden_tile_copy(last_step.llm_hidden_tt)
    codes_tt = pipe.acoustic.forward_acoustic_trace_codes(
        llm_tile,
        last_step.noise_tt,
        float(ACOUSTIC_CFG_ALPHA_DEFAULT),
    )
    if llm_tile.is_allocated():
        ttnn.deallocate(llm_tile)
    ref_last_trace_codes = ttnn.to_torch(codes_tt).long().reshape(1, -1)
    if codes_tt.is_allocated():
        ttnn.deallocate(codes_tt)

    ttnn.synchronize_device(device)
    return _VoxtralTraceE2EBundle(
        steps=steps,
        ref_last_hidden=cpu_hidden.float(),
        ref_last_trace_codes=ref_last_trace_codes,
    )


def _free_trace_e2e_bundle(bundle: _VoxtralTraceE2EBundle) -> None:
    for step in bundle.steps:
        if step.llm_hidden_tt.is_allocated():
            ttnn.deallocate(step.llm_hidden_tt)
        if step.noise_tt.is_allocated():
            ttnn.deallocate(step.noise_tt)
        for t in (
            step.text_step.x_embed_tt,
            step.text_step.current_pos_tt,
            step.text_step.page_table,
        ):
            if t is not None and t.is_allocated():
                ttnn.deallocate(t)


def _load_cpu_reference_or_skip(model_name: str) -> VoxtralCPUReference:
    try:
        return VoxtralCPUReference(model_name_or_path=model_name, dtype="bfloat16", device="cpu")
    except Exception as exc:
        pytest.skip(f"CPU reference load failed: {exc}")


def _capture_e2e_golden(pipe: VoxtralTTSPipeline, *, num_steps: int) -> _E2EGoldenRef:
    """One clean ``forward_device_resident`` on a fresh pipeline; CPU-ref waveform decode for PCC baseline."""
    from models.experimental.voxtraltts.reference.audio_tokenizer_ops import audio_tokenizer_decode_reference

    _load_cpu_reference_or_skip(pipe.model_name_or_path)

    out = pipe.forward_device_resident(
        text=_DEMO_TEXT,
        voice=_DEMO_VOICE,
        max_tokens=num_steps,
        seed=0,
        fixed_step_count=True,
        include_waveform_decode=True,
    )
    assert out.codes_b37t.shape[2] > 0, "golden capture produced no acoustic frames"
    assert torch.isfinite(out.waveform).all(), "golden capture produced non-finite waveform"

    ref_wav = audio_tokenizer_decode_reference(
        out.codes_b37t,
        pipe.audio_tokenizer_sd,
        pipe.config.audio_tokenizer_args,
    ).reshape(1, 1, -1)
    ok, msg = comp_pcc(ref_wav.float(), out.waveform.float(), pcc=WAVEFORM_PCC)
    logger.info(f"golden capture waveform PCC (CPU ref decode vs TT vocoder): {msg}")
    assert ok, f"golden capture waveform PCC < {WAVEFORM_PCC}: {msg}"

    return _E2EGoldenRef(codes_b37t=out.codes_b37t.clone(), waveform=ref_wav.float())


def _assert_non_trace_e2e_pcc(device: ttnn.Device, golden: _E2EGoldenRef) -> None:
    """Post-pipeline correctness gate on a fresh pipeline (KV not polluted by perf capture/loop)."""
    pipe = _load_pipeline_or_skip(device)
    try:
        tt_out = pipe.forward_device_resident(
            text=_DEMO_TEXT,
            voice=_DEMO_VOICE,
            max_tokens=_GENERATE_MAX_TOKENS,
            seed=0,
            fixed_step_count=True,
            include_waveform_decode=True,
        )
    finally:
        pipe.cleanup_all()

    assert tt_out.codes_b37t.shape[2] > 0, "pipeline forward produced no acoustic frames"
    assert torch.isfinite(tt_out.waveform).all(), "pipeline forward produced non-finite waveform"

    ref_codes = golden.codes_b37t
    tt_codes = tt_out.codes_b37t.long()
    assert (
        ref_codes.shape == tt_codes.shape
    ), f"golden codes shape {tuple(ref_codes.shape)} != post-perf {tuple(tt_codes.shape)}"
    t_frames = ref_codes.shape[2]
    ref_t = ref_codes[:, :, :t_frames]
    tt_t = _align_to_ref_shape(ref_t, tt_codes[:, :, :t_frames])

    assert torch.equal(ref_t[:, :1, :], tt_t[:, :1, :]), "semantic token mismatch (non-traced E2E)"
    acoustic_matches = int((ref_t[:, 1:, :] == tt_t[:, 1:, :]).sum().item())
    acoustic_total = int(ref_t[:, 1:, :].numel())
    match_frac = acoustic_matches / acoustic_total if acoustic_total else 1.0
    logger.info(
        f"non-traced E2E acoustic agreement: {match_frac:.4f} "
        f"({acoustic_matches}/{acoustic_total}) target>={ACOUSTIC_MATCH_FRAC}"
    )
    assert (
        match_frac >= ACOUSTIC_MATCH_FRAC
    ), f"non-traced E2E acoustic agreement {match_frac:.4f} < {ACOUSTIC_MATCH_FRAC}"

    ref_wav = golden.waveform
    tt_wav = tt_out.waveform.float()
    n_samples = min(ref_wav.shape[-1], tt_wav.shape[-1])
    ok, msg = comp_pcc(ref_wav[..., :n_samples], tt_wav[..., :n_samples], pcc=WAVEFORM_PCC)
    logger.info(f"non-traced E2E waveform PCC: {msg}")
    assert ok, f"non-traced E2E waveform PCC < {WAVEFORM_PCC}: {msg}"


def _assert_trace_acoustic_codes(
    codes_tt: ttnn.Tensor,
    ref_codes_137: torch.Tensor,
) -> None:
    """Last-step acoustic codes from traced pipeline output vs pre-capture device golden."""
    tt_codes = ttnn.to_torch(ttnn.from_device(codes_tt)).long().reshape(1, -1)
    ref_codes = ref_codes_137.reshape(1, -1)
    n_cols = min(tt_codes.shape[1], ref_codes.shape[1])
    tt_c = tt_codes[:, :n_cols]
    ref_c = ref_codes[:, :n_cols]

    assert torch.equal(ref_c[:, :1], tt_c[:, :1]), "trace semantic token mismatch"
    acoustic_matches = int((ref_c[:, 1:] == tt_c[:, 1:]).sum().item())
    acoustic_total = int(ref_c[:, 1:].numel())
    match_frac = acoustic_matches / acoustic_total if acoustic_total else 1.0
    logger.info(
        f"trace last-step acoustic agreement: {match_frac:.4f} "
        f"({acoustic_matches}/{acoustic_total}) target>={ACOUSTIC_MATCH_FRAC}"
    )
    assert match_frac >= ACOUSTIC_MATCH_FRAC, f"trace acoustic agreement {match_frac:.4f} < {ACOUSTIC_MATCH_FRAC}"


def _assert_trace_final_hidden(
    pipe: VoxtralTTSPipeline,
    bundle: _VoxtralTraceE2EBundle,
) -> None:
    """Final text hidden after trace replay vs CPU reference (post-pipeline, not timed)."""
    last = bundle.steps[-1]
    hidden_tt = pipe.text.decode_step_from_embeds_tt(
        last.text_step.x_embed_tt,
        last.text_step.current_pos_tt,
        last.text_step.rot_mats_global,
        last.text_step.rot_mats_local,
        last.text_step.page_table,
    )
    try:
        _assert_trace_hidden_pcc(pipe, hidden_tt, bundle.ref_last_hidden)
    finally:
        if hidden_tt.is_allocated():
            ttnn.deallocate(hidden_tt)


def _assert_trace_hidden_pcc(
    pipe: VoxtralTTSPipeline,
    hidden_tt: ttnn.Tensor,
    ref_hidden: torch.Tensor,
) -> None:
    dim = pipe.text.inner.args.dim
    flat = ttnn.to_torch(ttnn.from_device(hidden_tt)).to(torch.bfloat16).contiguous().reshape(-1)
    tt_h = flat[:dim].float().cpu()
    ref_h = ref_hidden.reshape(-1)[:dim].float().cpu()
    ok, msg = comp_pcc(ref_h, tt_h, pcc=TEXT_DECODE_STEP_PCC)
    logger.info(f"trace last hidden PCC: {msg}")
    assert ok, f"trace last hidden PCC < {TEXT_DECODE_STEP_PCC}: {msg}"


def _make_voxtral_trace_forward_fn(
    pipe: VoxtralTTSPipeline,
    bundle: _VoxtralTraceE2EBundle,
) -> Callable[[ttnn.Tensor], ttnn.Tensor]:
    def forward(_device_input: ttnn.Tensor) -> ttnn.Tensor:
        del _device_input
        return pipe.forward_tts_generation_trace(
            bundle.steps,
            cfg_scalar=ACOUSTIC_CFG_ALPHA_DEFAULT,
        )

    return forward


def _make_voxtral_forward_fn(
    pipe: VoxtralTTSPipeline,
    *,
    fixed_step_count: bool,
    include_waveform_decode: bool,
) -> Callable[[ttnn.Tensor], ttnn.Tensor]:
    def forward(_device_input: ttnn.Tensor) -> ttnn.Tensor:
        del _device_input
        out = pipe.forward_device_resident(
            text=_DEMO_TEXT,
            voice=_DEMO_VOICE,
            max_tokens=_GENERATE_MAX_TOKENS,
            seed=0,
            fixed_step_count=fixed_step_count,
            include_waveform_decode=include_waveform_decode,
        )
        assert out.codes_b37t.shape[2] > 0, "pipeline forward_device_resident produced no acoustic frames"
        if include_waveform_decode:
            assert torch.isfinite(out.waveform).all(), "pipeline forward_device_resident produced non-finite waveform"
        dim = pipe.text.inner.args.dim
        return ttnn.from_torch(
            torch.zeros((1, 1, 1, dim), dtype=torch.bfloat16),
            device=pipe.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(pipe.mesh_device),
        )

    return forward


def _run_pipeline(
    device: ttnn.Device,
    forward_fn: Callable[[ttnn.Tensor], ttnn.Tensor],
    dummy_host: ttnn.Tensor,
    dram_config: ttnn.MemoryConfig,
    l1_config: ttnn.MemoryConfig,
    num_iters: int,
    *,
    use_trace: bool,
) -> list:
    pipeline = create_pipeline_from_config(
        config=PipelineConfig(
            use_trace=use_trace,
            num_command_queues=2,
            all_transfers_on_separate_command_queue=False,
        ),
        model=forward_fn,
        device=device,
        dram_input_memory_config=dram_config,
        l1_input_memory_config=l1_config,
    )
    ttnn.synchronize_device(device)
    try:
        profiler.start("compile")
        pipeline.compile(dummy_host)
        profiler.end("compile")
        host_inputs = [dummy_host] * num_iters
        profiler.start("run_model_pipeline_2cqs")
        outputs = pipeline.enqueue(host_inputs).pop_all()
        profiler.end("run_model_pipeline_2cqs")
        return outputs
    finally:
        pipeline.cleanup()


def _load_pipeline_or_skip(device: ttnn.Device) -> VoxtralTTSPipeline:
    name = resolve_voxtral_model_name_or_skip()
    try:
        from models.experimental.voxtraltts.tt.voxtral_tt_args import voxtral_text_default_optimizations

        return VoxtralTTSPipeline.from_model_name(
            device,
            model_name_or_path=name,
            text_max_seq_len=_TEXT_MAX_SEQ_LEN,
            text_optimizations=voxtral_text_default_optimizations,
        )
    except Exception as exc:
        pytest.skip(f"TT pipeline load failed: {exc}")


def _run_e2e_perf_test(
    device: ttnn.Device,
    *,
    use_trace: bool,
    expected_throughput_fps: float,
    include_waveform_decode: bool,
) -> None:
    pipe = _load_pipeline_or_skip(device)
    trace_bundle = None
    e2e_golden = None
    try:
        if use_trace:
            trace_bundle = _materialize_voxtral_trace_e2e_bundle(
                pipe,
                device,
                num_steps=_GENERATE_MAX_TOKENS,
            )
            forward_fn = _make_voxtral_trace_forward_fn(pipe, trace_bundle)
        else:
            pipe_golden = _load_pipeline_or_skip(device)
            try:
                e2e_golden = _capture_e2e_golden(pipe_golden, num_steps=_GENERATE_MAX_TOKENS)
            finally:
                pipe_golden.cleanup_all()
            forward_fn = _make_voxtral_forward_fn(
                pipe,
                fixed_step_count=True,
                include_waveform_decode=include_waveform_decode,
            )
        dummy_host, dram_config, l1_config = _setup_sharded_pipeline_input(device)
        assert dummy_host.storage_type() == ttnn.StorageType.HOST

        profiler.clear()
        num_devices = device.get_num_devices()
        batch_size = num_devices

        outputs = _run_pipeline(
            device,
            forward_fn,
            dummy_host,
            dram_config,
            l1_config,
            _NUM_MEASUREMENT_ITERS,
            use_trace=use_trace,
        )
        assert len(outputs) == _NUM_MEASUREMENT_ITERS

        if use_trace:
            _assert_trace_acoustic_codes(outputs[-1], trace_bundle.ref_last_trace_codes)
            _assert_trace_final_hidden(pipe, trace_bundle)
        else:
            assert e2e_golden is not None
            _assert_non_trace_e2e_pcc(device, e2e_golden)

        compile_time = profiler.get("compile")
        inference_time_avg = profiler.get("run_model_pipeline_2cqs") / _NUM_MEASUREMENT_ITERS
        expected_inference_time = batch_size / expected_throughput_fps
        trace_tag = "trace" if use_trace else "no_trace"
        decode_tag = (
            f"trace_gen{_GENERATE_MAX_TOKENS}" if use_trace else ("full" if include_waveform_decode else "gen_only")
        )
        prep_perf_report(
            model_name=f"ttnn_voxtral_tts_2cqs_{trace_tag}_{decode_tag}_batch_size{batch_size}",
            batch_size=batch_size,
            inference_and_compile_time=compile_time,
            inference_time=inference_time_avg,
            expected_compile_time=900,
            expected_inference_time=expected_inference_time,
            comments=(
                f"voxtral_tts_e2e_2cq_{trace_tag}_{decode_tag}_max_tokens{_GENERATE_MAX_TOKENS}"
                f"_batchsize{batch_size}"
            ),
            inference_time_cpu=0.0,
        )
        logger.info(
            f"Voxtral TTS E2E 2CQ ({trace_tag}, {decode_tag}) batch_size={batch_size} "
            f"max_tokens={_GENERATE_MAX_TOKENS} compile={compile_time:.2f}s "
            f"inference_avg={inference_time_avg:.4f}s FPS={batch_size / inference_time_avg:.3f}"
        )
    finally:
        if trace_bundle is not None:
            _free_trace_e2e_bundle(trace_bundle)
        pipe.cleanup_all()


@run_for_blackhole("Voxtral TTS E2E perf is targeted for P150/Blackhole")
@pytest.mark.models_performance_bare_metal
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [_DEVICE_PARAMS_2CQ], indirect=True)
@pytest.mark.parametrize("batch_size_per_device", (1,))
def test_voxtral_tts_e2e_perf_2cq(device, batch_size_per_device):
    """2CQ overlapped pipeline: full ``forward`` (prefill + gen loop + waveform) per iteration."""
    del batch_size_per_device
    _run_e2e_perf_test(
        device,
        use_trace=False,
        expected_throughput_fps=_EXPECTED_E2E_FULL_THROUGHPUT_FPS,
        include_waveform_decode=True,
    )


@run_for_blackhole("Voxtral TTS E2E perf is targeted for P150/Blackhole")
@pytest.mark.models_performance_bare_metal
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [_DEVICE_PARAMS_2CQ_TRACE], indirect=True)
@pytest.mark.parametrize("batch_size_per_device", (1,))
def test_voxtral_tts_e2e_perf_2cq_trace(device, batch_size_per_device):
    """2CQ traced gen loop (acoustic+text decode); PCC on last-step codes and final hidden vs CPU."""
    del batch_size_per_device
    _run_e2e_perf_test(
        device,
        use_trace=True,
        expected_throughput_fps=_EXPECTED_E2E_TRACE_GEN_LOOP_FPS,
        include_waveform_decode=False,
    )
