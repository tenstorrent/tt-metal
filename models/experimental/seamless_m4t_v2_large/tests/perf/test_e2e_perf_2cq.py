# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
E2E performance tests for Seamless M4T v2 Large with **2CQ** pipeline on Blackhole (non-traced and traced).

* **Non-traced:** ``PipelineConfig(use_trace=False, num_command_queues=2)`` —
  ``MultiCQModelOverlappedInputExecutor`` (full ``forward`` per step).

* **Traced E2E:** ``test_seamless_m4t_v2_large_e2e_perf_2cq_trace`` — padding / 4D masks are built
  **before** ``compile``. **Text:** ``forward_text_e2e_prefill_trace`` (full text stack in trace).
  **Speech:** ``materialize_speech_encoder_trace_tensors`` (prebuilt ``SpeechEncoderTraceMasks`` + probe)
  then ``forward_speech_e2e_prefill_trace`` for full speech encode → trim/pad → decode + LM inside trace.
  **t2st / s2st:** the same traced body also runs T2U (``forward_*_e2e_plus_t2u_trace``): T2U inputs,
  HF reference discrete durations, and pre-tilized hard-upsample cum tensors are materialized before
  compile; T2U ``conv1d`` weights are warmed once so trace replay avoids host writes.
  The speech encoder caches ``ttnn.conv1d`` preprocessed weights after the first call so trace replay
  does not issue host writes (Metal forbids ``write_shard_to_device`` during capture).

Task table, text-decoder logits PCC (host or device TTNN), and optional T2U timing live in this module.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest
import torch
import ttnn
from loguru import logger
from transformers import AutoProcessor, AutoTokenizer
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask

from tests.ttnn.utils_for_testing import check_with_pcc

from models.common.utility_functions import profiler, run_for_blackhole
from models.experimental.seamless_m4t_v2_large.tt.common import to_torch_replicated_first_shard
from models.experimental.seamless_m4t_v2_large.tt.mesh_helpers import (
    mesh_default_device,
    MESH_DEVICE_PARAMETRIZE_E2E_2CQ,
    MESH_DEVICE_PARAMETRIZE_E2E_2CQ_GENERATE,
    MESH_DEVICE_PARAMETRIZE_E2E_2CQ_TRACE,
)
from models.demos.utils.common_demo_utils import get_mesh_mappers
from models.experimental.seamless_m4t_v2_large.reference.torch_seamless_m4t_v2_model import (
    forward_text_modality_logits,
    load_pretrained_seamless_m4t_v2_model,
)
from models.experimental.seamless_m4t_v2_large.reference.torch_text_to_unit import (
    forward_t2u_logits_and_padding,
    hf_discrete_duration_counts_batch1,
    synthetic_t2u_inputs,
)
from models.experimental.seamless_m4t_v2_large.tt.tt_seamless_m4t_v2_model import (
    TTSeamlessM4Tv2GenerationOutput,
    TTSeamlessM4Tv2GreedySearchOutput,
)
from models.experimental.seamless_m4t_v2_large.tt.tt_text_to_unit import (
    T2UTraceHardUpsampleCumsums,
    make_t2u_trace_prealloc_tensors,
)
from models.experimental.seamless_m4t_v2_large.tests.pcc.test_seamless_m4t_v2_model import (
    PCC_THRESHOLD,
    _decoder_seed,
    _real_speech_features,
    _real_text_input_ids,
    _weights_dir_or_skip,
    make_tt_model,
    torch_feats_to_ttnn,
    torch_ids_to_ttnn,
)
from models.perf.perf_utils import prep_perf_report
from models.tt_cnn.tt.pipeline import (
    PipelineConfig,
    create_pipeline_from_config,
    get_memory_config_for_persistent_dram_tensor,
)

# task_id -> (use_speech_input, tgt_lang, run_t2u_after_text)
_TASKS: Dict[str, Tuple[bool, str, bool]] = {
    "t2tt": (False, "eng", False),
    "s2tt": (True, "eng", False),
    "t2st": (False, "eng", True),
    "s2st": (True, "eng", True),
    "asr": (True, "eng", False),
}

# Expected E2E throughput (batches/s) for ``prep_perf_report`` latency bounds.
_EXPECTED_E2E_THROUGHPUT_FPS: Dict[str, float] = {
    "t2tt": 54.94,
    "s2tt": 22.92,
    "t2st": 29.61,
    "s2st": 16.73,
    "asr": 22.25,
}
_E2E_TASK_THROUGHPUT_PARAMS = [(t, _EXPECTED_E2E_THROUGHPUT_FPS[t]) for t in _TASKS]


def _assert_text_logits_pcc_local(
    ref_logits: torch.Tensor, logits_tt: ttnn.Tensor, *, ctx: str, pcc: float = PCC_THRESHOLD
) -> None:
    ref_f = ref_logits.detach().float().cpu()
    _, sd, v = ref_f.shape
    flat = to_torch_replicated_first_shard(logits_tt).to(torch.bfloat16).contiguous().reshape(-1)
    sp = flat.numel() // v
    tt_f = flat.reshape(1, sp, v)[:, :sd, :v].contiguous().float().cpu()
    assert tt_f.shape == ref_f.shape, f"{ctx}: shape ref {tuple(ref_f.shape)} vs ttnn {tuple(tt_f.shape)}"
    ok, msg = check_with_pcc(ref_f, tt_f, pcc=pcc)
    logger.info(f"{ctx} text-decoder logits PCC: {msg}")
    assert ok, f"{ctx}: text-decoder logits PCC < {pcc}: {msg}"


def _assert_t2u_logits_pcc_local(
    ref_logits_bf16: torch.Tensor, logits_tt: ttnn.Tensor, *, ctx: str, pcc: float = PCC_THRESHOLD
) -> None:
    v = int(ref_logits_bf16.shape[-1])
    flat = to_torch_replicated_first_shard(logits_tt).to(torch.bfloat16).contiguous().reshape(-1)
    sp = flat.numel() // v
    tt_logits_3d = flat.reshape(1, sp, v)[:, : ref_logits_bf16.shape[1], :].contiguous()
    assert (
        tt_logits_3d.shape == ref_logits_bf16.shape
    ), f"{ctx}: T2U logits shape ref={tuple(ref_logits_bf16.shape)} tt={tuple(tt_logits_3d.shape)}"
    ok, msg = check_with_pcc(ref_logits_bf16.float(), tt_logits_3d.float(), pcc=pcc)
    logger.info(f"{ctx} T2U logits PCC: {msg}")
    assert ok, f"{ctx}: T2U logits PCC < {pcc}: {msg}"


def _host_repeat_cumsum_tiles(device: ttnn.Device, repeats: List[int]) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
    """Inclusive / exclusive cum boundaries as TILE float32 ``[1, len(repeats)]`` (matches ``_hard_upsample_nlc``)."""
    enc_seq = len(repeats)
    cum_inc_list: list[float] = []
    acc = 0
    for r in repeats:
        acc += int(r)
        cum_inc_list.append(float(acc))
    prev_list = [0.0] + cum_inc_list[:-1]
    inc_rm = ttnn.from_torch(
        torch.tensor(cum_inc_list, dtype=torch.float32).view(1, enc_seq),
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    prev_rm = ttnn.from_torch(
        torch.tensor(prev_list, dtype=torch.float32).view(1, enc_seq),
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    inc_t = ttnn.to_layout(inc_rm, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    prev_t = ttnn.to_layout(prev_rm, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(inc_rm)
    ttnn.deallocate(prev_rm)
    return inc_t, prev_t


@dataclass
class _T2UTracePack:
    inputs_embeds_tt: ttnn.Tensor
    attn_tt: ttnn.Tensor
    char_ids_tt: ttnn.Tensor
    cc_list: List[int]
    ref_durs: List[int]
    cums: T2UTraceHardUpsampleCumsums
    ref_logits_bf16: torch.Tensor


def _materialize_t2u_trace_pack(model: Any, tt_model: Any, device: ttnn.Device) -> _T2UTracePack:
    """Synthetic T2U device tensors + hard-upsample cums + HF reference logits/durations; one probe forward for conv cache."""
    t2u_cfg = model.t2u_model.config
    inputs_embeds, attention_mask, char_input_ids, char_count_per_id = synthetic_t2u_inputs(
        t2u_cfg,
        batch=1,
        encoder_seq_len=32,
        seed=1,
        dtype=torch.bfloat16,
    )
    hf_dev = next(model.t2u_model.parameters()).device
    char_count_per_id_dev = char_count_per_id.to(hf_dev)

    ref_logits, _ = forward_t2u_logits_and_padding(
        model.t2u_model,
        inputs_embeds,
        attention_mask,
        char_input_ids,
        char_count_per_id_dev,
    )
    ref_logits_bf16 = ref_logits.to(torch.bfloat16).cpu()

    ref_durs = hf_discrete_duration_counts_batch1(
        model.t2u_model,
        inputs_embeds.to(hf_dev),
        attention_mask.to(hf_dev),
        char_input_ids.to(hf_dev),
        char_count_per_id_dev,
    )
    cc_list = [int(x) for x in char_count_per_id[0].cpu().tolist()]
    char_inc, char_prev = _host_repeat_cumsum_tiles(device, cc_list)
    unit_inc, unit_prev = _host_repeat_cumsum_tiles(device, ref_durs)
    cums = make_t2u_trace_prealloc_tensors(
        device,
        pad_token_id=int(t2u_cfg.pad_token_id),
        hidden_size=int(t2u_cfg.hidden_size),
        char_w=int(char_input_ids.shape[1]),
        cc_list=cc_list,
        ref_durs=ref_durs,
        char_inc=char_inc,
        char_prev=char_prev,
        unit_inc=unit_inc,
        unit_prev=unit_prev,
    )

    mask_4d = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype)
    inputs_embeds_tt = ttnn.from_torch(
        inputs_embeds.cpu().to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    attn_tt = ttnn.from_torch(
        mask_4d.cpu().to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    char_ids_tt = ttnn.from_torch(
        char_input_ids.cpu().to(torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    char_len = int(sum(cc_list))
    unit_seq = int(sum(ref_durs))
    padded_unit_seq = ((unit_seq + 31) // 32) * 32
    tt_model.t2u.prewarm_conv1d_weights(char_len=char_len, padded_unit_seq=padded_unit_seq)
    tt_model.vocoder.prewarm_conv1d_weights(batch=1, seq=padded_unit_seq, t_audio=unit_seq)
    ttnn.synchronize_device(device)
    _, _ = tt_model.t2u.forward(
        inputs_embeds_tt,
        attn_tt,
        char_ids_tt,
        cc_list,
        reference_discrete_durations=ref_durs,
        hard_upsample_cums=cums,
        trace_no_profiler=True,
    )
    ttnn.synchronize_device(device)

    return _T2UTracePack(
        inputs_embeds_tt=inputs_embeds_tt,
        attn_tt=attn_tt,
        char_ids_tt=char_ids_tt,
        cc_list=cc_list,
        ref_durs=list(ref_durs),
        cums=cums,
        ref_logits_bf16=ref_logits_bf16,
    )


def _dealloc_t2u_trace_pack(p: _T2UTracePack) -> None:
    ttnn.deallocate(p.inputs_embeds_tt)
    ttnn.deallocate(p.attn_tt)
    ttnn.deallocate(p.char_ids_tt)
    c = p.cums
    for t in (
        c.char_inc,
        c.char_prev,
        c.unit_inc,
        c.unit_prev,
        c.char_frame_idx_f32,
        c.unit_frame_idx_f32,
        c.char_pos_ids,
        c.unit_pos_ids,
        c.char_pad_bf16_tile,
        c.pad_unit_bf16_tile,
        c.attn_4d_bf16_tile,
        c.pad_unit_3d_tt,
        c.unit_hidden_pad_tail_bf16,
    ):
        if t is not None:
            ttnn.deallocate(t)


def _t2u_timed_stages(model: Any, tt_model: Any, device: ttnn.Device, *, ctx: str) -> Dict[str, float]:
    """T2U PCC path (matches PCC test) with per-stage ms. Returns stage name -> ms."""
    t2u_cfg = model.t2u_model.config
    t_prep0 = time.perf_counter()
    inputs_embeds, attention_mask, char_input_ids, char_count_per_id = synthetic_t2u_inputs(
        t2u_cfg,
        batch=1,
        encoder_seq_len=32,
        seed=1,
        dtype=torch.bfloat16,
    )
    hf_dev = next(model.t2u_model.parameters()).device
    char_count_per_id_dev = char_count_per_id.to(hf_dev)

    ref_logits, _ = forward_t2u_logits_and_padding(
        model.t2u_model,
        inputs_embeds,
        attention_mask,
        char_input_ids,
        char_count_per_id_dev,
    )
    ref_logits_bf16 = ref_logits.to(torch.bfloat16).cpu()

    ref_durs = hf_discrete_duration_counts_batch1(
        model.t2u_model,
        inputs_embeds.to(hf_dev),
        attention_mask.to(hf_dev),
        char_input_ids.to(hf_dev),
        char_count_per_id_dev,
    )
    t_prep1 = time.perf_counter()

    mask_4d = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype)
    inputs_embeds_tt = ttnn.from_torch(
        inputs_embeds.cpu().to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    attn_tt = ttnn.from_torch(
        mask_4d.cpu().to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    char_ids_tt = ttnn.from_torch(
        char_input_ids.cpu().to(torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    cc_list = [int(x) for x in char_count_per_id[0].cpu().tolist()]
    ttnn.synchronize_device(device)
    t_h2d = time.perf_counter()

    tt_logits_tt, _ = tt_model.t2u.forward(
        inputs_embeds_tt,
        attn_tt,
        char_ids_tt,
        cc_list,
        reference_discrete_durations=ref_durs,
    )
    ttnn.synchronize_device(device)
    t_fwd = time.perf_counter()

    ttnn.deallocate(inputs_embeds_tt)
    ttnn.deallocate(attn_tt)
    ttnn.deallocate(char_ids_tt)

    tt_logits = to_torch_replicated_first_shard(tt_logits_tt).to(torch.bfloat16).cpu()
    ttnn.deallocate(tt_logits_tt)

    v = int(ref_logits_bf16.shape[-1])
    flat = tt_logits.reshape(-1)
    sp = flat.numel() // v
    tt_logits_3d = flat.reshape(1, sp, v)[:, : ref_logits_bf16.shape[1], :].contiguous()
    assert (
        tt_logits_3d.shape == ref_logits_bf16.shape
    ), f"{ctx}: T2U logits shape ref={tuple(ref_logits_bf16.shape)} tt={tuple(tt_logits_3d.shape)}"
    ok, msg = check_with_pcc(ref_logits_bf16.float(), tt_logits_3d.float(), pcc=PCC_THRESHOLD)
    logger.info(f"{ctx} T2U logits PCC: {msg}")
    assert ok, f"{ctx}: T2U logits PCC < {PCC_THRESHOLD}: {msg}"
    t_end = time.perf_counter()

    return {
        "T2U host prep (HF+syn)": (t_prep1 - t_prep0) * 1000.0,
        "T2U Host→Device": (t_h2d - t_prep1) * 1000.0,
        "T2U forward": (t_fwd - t_h2d) * 1000.0,
        "T2U PCC validation": (t_end - t_fwd) * 1000.0,
    }


SEAMLESS_E2E_TASKS = _TASKS
assert_text_logits_pcc_vs_ref = _assert_text_logits_pcc_local
t2u_timed_stages_for_e2e = _t2u_timed_stages

NUM_MEASUREMENT_ITERS = 4
_DUMMY_WIDTH = 32


def _determine_num_cores_for_even_sharding(shard_dim: int, max_cores: int) -> int:
    """Match ``pipeline._determine_num_cores_for_even_sharding`` (DRAM shard core count)."""
    number_of_cores = max_cores
    while shard_dim % number_of_cores != 0:
        assert number_of_cores > 0, "Unable to find core grid"
        number_of_cores -= 1
    return number_of_cores


def _dummy_height_for_tile_height_sharded_io(device: ttnn.Device) -> int:
    """
    Pipeline DRAM/L1 inputs use TILE + HEIGHT_SHARDED; each physical shard must be tile-aligned
    (multiples of 32 on both axes). Shape ``(1, 32, 32)`` yields shard ``(4, 32)`` on 8-wide L1
    and fails on Blackhole. Scan heights until DRAM (``get_memory_config_for_persistent_dram_tensor``)
    and L1 (same rules as ``_get_l1_input_memory_config``) both produce valid shard heights.
    """
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
    """Height-sharded L1 memory config for pipeline input (same pattern as DINO ``test_e2e_perf_2cq``)."""
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
    """Host dummy tensor + DRAM/L1 configs for ``create_pipeline_from_config`` (HEIGHT-sharded DRAM)."""
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


def _make_forward_fn(
    tt_model: Any,
    *,
    use_speech: bool,
    enc_in_tt: Optional[ttnn.Tensor],
    input_ids_tt: Optional[ttnn.Tensor],
    enc_attn_tt: ttnn.Tensor,
    dec_ids_tt: ttnn.Tensor,
    dec_mask_tt: ttnn.Tensor,
) -> Callable[[ttnn.Tensor], ttnn.Tensor]:
    """Pipeline body: ``TTSeamlessM4Tv2Model.forward`` (dummy L1 input ignored). Frees encoder hidden state each step."""

    def forward(_device_input: ttnn.Tensor) -> ttnn.Tensor:
        del _device_input  # pipeline supplies sharded L1 tensor; Seamless inputs are closed over
        if use_speech:
            assert enc_in_tt is not None
            out = tt_model.forward(
                input_features=enc_in_tt,
                attention_mask=enc_attn_tt,
                decoder_input_ids=dec_ids_tt,
                decoder_attention_mask=dec_mask_tt,
            )
        else:
            assert input_ids_tt is not None
            out = tt_model.forward(
                input_ids=input_ids_tt,
                attention_mask=enc_attn_tt,
                decoder_input_ids=dec_ids_tt,
                decoder_attention_mask=dec_mask_tt,
            )
        logits = out.logits
        enc_state = out.encoder_last_hidden_state
        if enc_state is not None and enc_state.is_allocated():
            ttnn.deallocate(enc_state)
        return logits

    return forward


def _make_generate_fn(
    tt_model: Any,
    *,
    use_speech: bool,
    enc_in_tt: Optional[ttnn.Tensor],
    input_ids_tt: Optional[ttnn.Tensor],
    enc_attn_tt: ttnn.Tensor,
    tgt_lang: str,
    generate_speech: bool,
    max_new_tokens: int,
    pad_token_id: int,
    eos_token_id: Any,
    speaker_id: int = 0,
) -> Callable[[ttnn.Tensor], Any]:
    """Pipeline body: full autoregressive ``TTSeamlessM4Tv2Model.generate`` (dummy L1 input ignored).

    Text-only tasks (``generate_speech=False``) return the ``sequences`` tensor. Speech tasks
    (``t2st`` / ``s2st``) return ``(waveform, waveform_lengths)``; ``return_intermediate_token_ids``
    is left ``False`` so ``generate`` already deallocates ``sequences`` and ``unit_sequences``
    internally and the pipeline's output schema stays minimal.

    Note: ``generate`` does per-step host scalar readbacks for the EOS check (and host T2U→vocoder
    remap for speech), which serialises the two command queues. Throughput here is the full
    autoregressive E2E latency per task, **not** per-prefill latency like the ``forward`` pipeline.
    """
    gen_common = dict(
        max_new_tokens=int(max_new_tokens),
        do_sample=False,
        num_beams=1,
        pad_token_id=int(pad_token_id),
        eos_token_id=eos_token_id,
        use_kv_cache=True,
        # Decode trace capture leaves an active trace region; 2CQ ``pipeline.compile`` then fails
        # with "Tensor is not allocated". Demo uses the same eager decode path.
        use_decode_trace=False,
        prewarm_conv1d_weights=True,
    )

    def generate(_device_input: ttnn.Tensor) -> Any:
        del _device_input  # pipeline supplies sharded L1 tensor; Seamless inputs are closed over
        if use_speech:
            assert enc_in_tt is not None
            out = tt_model.generate(
                input_features=enc_in_tt,
                attention_mask=enc_attn_tt,
                generate_speech=generate_speech,
                tgt_lang=tgt_lang,
                speaker_id=speaker_id,
                **gen_common,
            )
        else:
            assert input_ids_tt is not None
            out = tt_model.generate(
                input_ids=input_ids_tt,
                attention_mask=enc_attn_tt,
                generate_speech=generate_speech,
                tgt_lang=tgt_lang,
                speaker_id=speaker_id,
                **gen_common,
            )

        if generate_speech:
            # Speech path: ``generate`` returns ``(waveform_tt, lengths_tt)`` when
            # ``return_intermediate_token_ids=False`` (default here). Forward as a tuple.
            if isinstance(out, TTSeamlessM4Tv2GenerationOutput):
                wav, lengths = out.waveform, out.waveform_lengths
            else:
                wav, lengths = out  # type: ignore[misc]
            return (wav, lengths)

        # Text-only path: ``generate`` returns ``TTSeamlessM4Tv2GreedySearchOutput(sequences=...)``.
        assert isinstance(out, TTSeamlessM4Tv2GreedySearchOutput), type(out)
        return out.sequences

    return generate


@dataclass
class _Seamless2CQBundle:
    model: Any
    cfg: Any
    tt_model: Any
    ref_logits: torch.Tensor
    forward_fn: Callable[[ttnn.Tensor], ttnn.Tensor]
    dummy_host: ttnn.Tensor
    dram_config: ttnn.MemoryConfig
    l1_config: ttnn.MemoryConfig
    task: str
    needs_t2u: bool
    enc_in_tt: Optional[ttnn.Tensor]
    input_ids_tt: Optional[ttnn.Tensor]
    enc_attn_tt: ttnn.Tensor
    dec_ids_tt: ttnn.Tensor
    dec_mask_tt: ttnn.Tensor


def _build_model_for_task(device: ttnn.Device, task: str) -> _Seamless2CQBundle:
    use_speech, tgt_lang, needs_t2u = SEAMLESS_E2E_TASKS[task]
    weights_dir = _weights_dir_or_skip()

    model, cfg = load_pretrained_seamless_m4t_v2_model(weights_dir, dtype=torch.bfloat16)
    t2u_cfg = model.t2u_model.config
    dev = next(model.parameters()).device
    tt_model = make_tt_model(device, model, cfg, t2u_cfg)

    decoder_input_ids, decoder_attention_mask = _decoder_seed(cfg, dev, tgt_lang=tgt_lang)

    if use_speech:
        processor = AutoProcessor.from_pretrained(os.fspath(weights_dir), local_files_only=True)
        input_features, enc_attn = _real_speech_features(processor, dev)
        with torch.no_grad():
            ref_out = model(
                input_features=input_features,
                attention_mask=enc_attn,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                use_cache=False,
                return_dict=True,
            )
        ref_logits = ref_out.logits.to(torch.bfloat16).cpu().float()
    else:
        tokenizer = AutoTokenizer.from_pretrained(os.fspath(weights_dir), local_files_only=True)
        input_ids, enc_attn = _real_text_input_ids(tokenizer, dev)
        ref_logits = (
            forward_text_modality_logits(
                model,
                input_ids=input_ids,
                attention_mask=enc_attn,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
            )
            .to(torch.bfloat16)
            .cpu()
        )

    enc_in_tt: Optional[ttnn.Tensor] = None
    input_ids_tt: Optional[ttnn.Tensor] = None
    if use_speech:
        enc_in_tt = torch_feats_to_ttnn(device, input_features)
        enc_attn_tt = torch_ids_to_ttnn(device, enc_attn.cpu())
    else:
        input_ids_tt = torch_ids_to_ttnn(device, input_ids)
        enc_attn_tt = torch_ids_to_ttnn(device, enc_attn)

    dec_ids_tt = torch_ids_to_ttnn(device, decoder_input_ids)
    dec_mask_tt = torch_ids_to_ttnn(device, decoder_attention_mask)
    ttnn.synchronize_device(device)

    forward_fn = _make_forward_fn(
        tt_model,
        use_speech=use_speech,
        enc_in_tt=enc_in_tt,
        input_ids_tt=input_ids_tt,
        enc_attn_tt=enc_attn_tt,
        dec_ids_tt=dec_ids_tt,
        dec_mask_tt=dec_mask_tt,
    )

    dummy_host, dram_config, l1_config = _setup_sharded_pipeline_input(device)
    assert dummy_host.storage_type() == ttnn.StorageType.HOST

    return _Seamless2CQBundle(
        model=model,
        cfg=cfg,
        tt_model=tt_model,
        ref_logits=ref_logits,
        forward_fn=forward_fn,
        dummy_host=dummy_host,
        dram_config=dram_config,
        l1_config=l1_config,
        task=task,
        needs_t2u=needs_t2u,
        enc_in_tt=enc_in_tt,
        input_ids_tt=input_ids_tt,
        enc_attn_tt=enc_attn_tt,
        dec_ids_tt=dec_ids_tt,
        dec_mask_tt=dec_mask_tt,
    )


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


@run_for_blackhole()
@pytest.mark.models_performance_bare_metal
@pytest.mark.timeout(3600)
@pytest.mark.parametrize(*MESH_DEVICE_PARAMETRIZE_E2E_2CQ, indirect=["mesh_device", "device_params"])
@pytest.mark.parametrize("batch_size_per_device", (1,))
@pytest.mark.parametrize("task,expected_inference_throughput", _E2E_TASK_THROUGHPUT_PARAMS)
def test_seamless_m4t_v2_large_e2e_perf_2cq(
    mesh_device, device_params, batch_size_per_device, expected_inference_throughput: float, task: str
):
    """Compile + 2CQ overlapped pipeline (non-traced); PCC on last host logits; optional T2U after pipeline."""
    with mesh_default_device(mesh_device):
        _ = device_params
        b = _build_model_for_task(mesh_device, task)
        profiler.clear()
        num_devices = mesh_device.get_num_devices()
        batch_size = batch_size_per_device * num_devices

        outputs = _run_pipeline(
            mesh_device,
            b.forward_fn,
            b.dummy_host,
            b.dram_config,
            b.l1_config,
            NUM_MEASUREMENT_ITERS,
            use_trace=False,
        )

        logits_host = outputs[-1]
        assert_text_logits_pcc_vs_ref(b.ref_logits, logits_host, ctx=f"{b.task.upper()}_E2E_2CQ")

        if b.enc_in_tt is not None:
            ttnn.deallocate(b.enc_in_tt)
        if b.input_ids_tt is not None:
            ttnn.deallocate(b.input_ids_tt)
        ttnn.deallocate(b.enc_attn_tt)
        ttnn.deallocate(b.dec_ids_tt)
        ttnn.deallocate(b.dec_mask_tt)

        if b.needs_t2u:
            t2u_timed_stages_for_e2e(b.model, b.tt_model, mesh_device, ctx=f"{b.task.upper()}_E2E_T2U")

        compile_time = profiler.get("compile")
        inference_time_avg = profiler.get("run_model_pipeline_2cqs") / NUM_MEASUREMENT_ITERS
        expected_inference_time = batch_size / expected_inference_throughput
        prep_perf_report(
            model_name=f"ttnn_seamless_m4t_v2_large_2cqs_batch_size{batch_size}_{b.task}",
            batch_size=batch_size,
            inference_and_compile_time=compile_time,
            inference_time=inference_time_avg,
            expected_compile_time=600,
            expected_inference_time=expected_inference_time,
            comments=f"task_{b.task}_batchsize{batch_size}",
            inference_time_cpu=0.0,
        )
        logger.info(
            f"Seamless M4T v2 Large task={b.task} batch_size={batch_size} "
            f"compile={compile_time:.2f}s inference_avg={inference_time_avg:.4f}s "
            f"FPS={batch_size / inference_time_avg:.2f}"
        )


# ``generate()`` E2E bounds (autoregressive loop + speech path); loose until tuned per hardware.
_GENERATE_MAX_NEW_TOKENS: int = 48
_EXPECTED_GENERATE_E2E_THROUGHPUT_FPS: Dict[str, float] = {
    "t2tt": 0.5,
    "s2tt": 0.5,
    "t2st": 0.25,
    "s2st": 0.25,
    "asr": 0.5,
}
_GENERATE_E2E_TASK_THROUGHPUT_PARAMS = [(t, _EXPECTED_GENERATE_E2E_THROUGHPUT_FPS[t]) for t in _TASKS]


@run_for_blackhole()
@pytest.mark.models_performance_bare_metal
@pytest.mark.timeout(3600)
@pytest.mark.parametrize(*MESH_DEVICE_PARAMETRIZE_E2E_2CQ_GENERATE, indirect=["mesh_device", "device_params"])
@pytest.mark.parametrize("batch_size_per_device", (1,))
@pytest.mark.parametrize("task,expected_inference_throughput", _GENERATE_E2E_TASK_THROUGHPUT_PARAMS)
def test_seamless_m4t_v2_large_e2e_perf_2cq_generate(
    mesh_device, device_params, batch_size_per_device, expected_inference_throughput: float, task: str
):
    """Compile + 2CQ non-traced pipeline driving full autoregressive ``tt_model.generate(...)`` per iteration.

    Unlike ``test_seamless_m4t_v2_large_e2e_perf_2cq`` (single encoder + decoder + lm_head pass on a
    seeded ``decoder_input_ids``), every pipeline step runs the entire generation loop:

    * **t2tt / s2tt / asr** — encoder → greedy text decode → tokens.
    * **t2st / s2st** — encoder → greedy text decode → re-encode (speech path) → T2U → HiFi-GAN
      vocoder → waveform + lengths.

    The 2CQ overlap helps with host→device transfers between iterations, but per-step host scalar
    readbacks for the EOS check (and the speech-path host remap before the vocoder) serialise
    most of the in-iteration work — reported FPS is full E2E generation, not per-prefill.
    ``l1_small_size=65536`` matches the demo (32 KiB is fine for text-only but
    too tight for the speech-encoder → T2U → vocoder chain on s2st).
    """
    with mesh_default_device(mesh_device):
        _ = device_params
        b = _build_model_for_task(mesh_device, task)
        use_speech, tgt_lang, needs_t2u = SEAMLESS_E2E_TASKS[task]
        b.forward_fn = _make_generate_fn(
            b.tt_model,
            use_speech=use_speech,
            enc_in_tt=b.enc_in_tt,
            input_ids_tt=b.input_ids_tt,
            enc_attn_tt=b.enc_attn_tt,
            tgt_lang=tgt_lang,
            generate_speech=needs_t2u,
            max_new_tokens=_GENERATE_MAX_NEW_TOKENS,
            pad_token_id=b.cfg.pad_token_id,
            eos_token_id=b.cfg.eos_token_id,
        )

        profiler.clear()
        num_devices = mesh_device.get_num_devices()
        batch_size = batch_size_per_device * num_devices

        outputs = _run_pipeline(
            mesh_device,
            b.forward_fn,
            b.dummy_host,
            b.dram_config,
            b.l1_config,
            NUM_MEASUREMENT_ITERS,
            use_trace=False,
        )

        last = outputs[-1]
        if needs_t2u:
            assert (
                isinstance(last, (list, tuple)) and len(last) == 2
            ), f"{b.task.upper()}_E2E_2CQ_GENERATE: expected (waveform, lengths) tuple, got {type(last)}"
            wav_host, lengths_host = last
            wav_t = to_torch_replicated_first_shard(wav_host)
            len_t = to_torch_replicated_first_shard(lengths_host)
            assert (
                int(len_t.reshape(-1)[0].item()) > 0
            ), f"{b.task.upper()}_E2E_2CQ_GENERATE: vocoder reported zero-length waveform"
            logger.info(
                f"{b.task.upper()}_E2E_2CQ_GENERATE waveform_samples={int(len_t.reshape(-1)[0].item())} "
                f"wav_shape={tuple(wav_t.shape)}"
            )
        else:
            assert isinstance(
                last, ttnn.Tensor
            ), f"{b.task.upper()}_E2E_2CQ_GENERATE: expected sequences tensor, got {type(last)}"
            seq_t = to_torch_replicated_first_shard(last)
            assert seq_t.numel() > 0, f"{b.task.upper()}_E2E_2CQ_GENERATE: empty sequences output"
            logger.info(f"{b.task.upper()}_E2E_2CQ_GENERATE sequences_shape={tuple(seq_t.shape)}")

        if b.enc_in_tt is not None:
            ttnn.deallocate(b.enc_in_tt)
        if b.input_ids_tt is not None:
            ttnn.deallocate(b.input_ids_tt)
        ttnn.deallocate(b.enc_attn_tt)
        ttnn.deallocate(b.dec_ids_tt)
        ttnn.deallocate(b.dec_mask_tt)

        compile_time = profiler.get("compile")
        inference_time_avg = profiler.get("run_model_pipeline_2cqs") / NUM_MEASUREMENT_ITERS
        expected_inference_time = batch_size / expected_inference_throughput
        prep_perf_report(
            model_name=f"ttnn_seamless_m4t_v2_large_2cqs_generate_batch_size{batch_size}_{b.task}",
            batch_size=batch_size,
            inference_and_compile_time=compile_time,
            inference_time=inference_time_avg,
            expected_compile_time=600,
            expected_inference_time=expected_inference_time,
            comments=f"generate_task_{b.task}_batchsize{batch_size}_max_new_tokens{_GENERATE_MAX_NEW_TOKENS}",
            inference_time_cpu=0.0,
        )
        logger.info(
            f"Seamless M4T v2 Large generate task={b.task} batch_size={batch_size} "
            f"max_new_tokens={_GENERATE_MAX_NEW_TOKENS} compile={compile_time:.2f}s "
            f"inference_avg={inference_time_avg:.4f}s FPS={batch_size / inference_time_avg:.3f}"
        )


def _make_traced_text_e2e_forward_fn(
    tt_model: Any,
    enc_ids_p: ttnn.Tensor,
    enc_pos: ttnn.Tensor,
    enc_m4: ttnn.Tensor,
    dec_ids_p: ttnn.Tensor,
    dec_pos: ttnn.Tensor,
    dec_causal: ttnn.Tensor,
    dec_cross: ttnn.Tensor,
) -> Callable[[ttnn.Tensor], ttnn.Tensor]:
    """Traced pipeline body: text encoder → decoder → lm_head."""

    def forward(_l1_input: ttnn.Tensor) -> ttnn.Tensor:
        del _l1_input
        return tt_model.forward_text_e2e_prefill_trace(
            enc_ids_p, enc_pos, enc_m4, dec_ids_p, dec_pos, dec_causal, dec_cross
        )

    return forward


def _make_traced_speech_e2e_forward_fn(
    tt_model: Any,
    enc_in_tt: ttnn.Tensor,
    conv_mask_bf16: ttnn.Tensor,
    speech_trace: Any,
    pad_tail: Optional[ttnn.Tensor],
    logical_len: int,
    physical_len: int,
    dec_ids_p: ttnn.Tensor,
    dec_pos: ttnn.Tensor,
    dec_causal: ttnn.Tensor,
    dec_cross: ttnn.Tensor,
) -> Callable[[ttnn.Tensor], ttnn.Tensor]:
    """Traced pipeline body: speech encoder (prebuilt masks) → trim/pad → text decoder → lm_head."""

    def forward(_l1_input: ttnn.Tensor) -> ttnn.Tensor:
        del _l1_input
        return tt_model.forward_speech_e2e_prefill_trace(
            enc_in_tt,
            conv_mask_bf16,
            speech_trace,
            pad_tail,
            logical_len,
            physical_len,
            dec_ids_p,
            dec_pos,
            dec_causal,
            dec_cross,
        )

    return forward


def _make_traced_text_e2e_plus_t2u_forward_fn(
    tt_model: Any,
    enc_ids_p: ttnn.Tensor,
    enc_pos: ttnn.Tensor,
    enc_m4: ttnn.Tensor,
    dec_ids_p: ttnn.Tensor,
    dec_pos: ttnn.Tensor,
    dec_causal: ttnn.Tensor,
    dec_cross: ttnn.Tensor,
    t2u: _T2UTracePack,
) -> Callable[[ttnn.Tensor], Tuple[ttnn.Tensor, ttnn.Tensor]]:
    """Traced pipeline: text encoder → decoder → lm_head → T2U."""

    def forward(_l1_input: ttnn.Tensor) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        del _l1_input
        return tt_model.forward_text_e2e_plus_t2u_trace(
            enc_ids_p,
            enc_pos,
            enc_m4,
            dec_ids_p,
            dec_pos,
            dec_causal,
            dec_cross,
            t2u.inputs_embeds_tt,
            t2u.attn_tt,
            t2u.char_ids_tt,
            t2u.cc_list,
            t2u.ref_durs,
            t2u.cums,
        )

    return forward


def _make_traced_speech_e2e_plus_t2u_forward_fn(
    tt_model: Any,
    enc_in_tt: ttnn.Tensor,
    conv_mask_bf16: ttnn.Tensor,
    speech_trace: Any,
    pad_tail: Optional[ttnn.Tensor],
    logical_len: int,
    physical_len: int,
    dec_ids_p: ttnn.Tensor,
    dec_pos: ttnn.Tensor,
    dec_causal: ttnn.Tensor,
    dec_cross: ttnn.Tensor,
    t2u: _T2UTracePack,
) -> Callable[[ttnn.Tensor], Tuple[ttnn.Tensor, ttnn.Tensor]]:
    """Traced pipeline: speech encoder → trim/pad → text decoder → lm_head → T2U."""

    def forward(_l1_input: ttnn.Tensor) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        del _l1_input
        return tt_model.forward_speech_e2e_plus_t2u_trace(
            enc_in_tt,
            conv_mask_bf16,
            speech_trace,
            pad_tail,
            logical_len,
            physical_len,
            dec_ids_p,
            dec_pos,
            dec_causal,
            dec_cross,
            t2u.inputs_embeds_tt,
            t2u.attn_tt,
            t2u.char_ids_tt,
            t2u.cc_list,
            t2u.ref_durs,
            t2u.cums,
        )

    return forward


def _dealloc_if_allocated(t: Optional[ttnn.Tensor]) -> None:
    if t is not None and t.is_allocated():
        ttnn.deallocate(t)


def _dealloc_speech_trace_masks(m: Any) -> None:
    _dealloc_if_allocated(m.encoder_additive_4d)
    for t in m.adapter_self_attn_4d:
        _dealloc_if_allocated(t)
    for t in m.conv_dw_left_pad:
        _dealloc_if_allocated(t)


@run_for_blackhole()
@pytest.mark.models_performance_bare_metal
@pytest.mark.timeout(3600)
@pytest.mark.parametrize(*MESH_DEVICE_PARAMETRIZE_E2E_2CQ_TRACE, indirect=["mesh_device", "device_params"])
@pytest.mark.parametrize("batch_size_per_device", (1,))
@pytest.mark.parametrize("task,expected_inference_throughput", _E2E_TASK_THROUGHPUT_PARAMS)
def test_seamless_m4t_v2_large_e2e_perf_2cq_trace(
    mesh_device, device_params, batch_size_per_device, expected_inference_throughput: float, task: str
):
    """
    ``use_trace=True`` + 2CQ: materialize masks **before** compile. **Text:** ``forward_text_e2e_prefill_trace``.
    **Speech:** ``materialize_speech_encoder_trace_tensors`` (prebuilt ``SpeechEncoderTraceMasks`` + probe)
    then ``forward_speech_e2e_prefill_trace`` for full speech→text inside trace.
    **t2st / s2st:** ``forward_*_e2e_plus_t2u_trace`` extends the traced body with T2U (prebuilt inputs + cums).
    """
    with mesh_default_device(mesh_device):
        _ = device_params
        b = _build_model_for_task(mesh_device, task)
        use_speech, _, needs_t2u = SEAMLESS_E2E_TASKS[task]

        t2u_pack: Optional[_T2UTracePack] = None
        if needs_t2u:
            t2u_pack = _materialize_t2u_trace_pack(b.model, b.tt_model, mesh_device)

        conv_bf16: Optional[ttnn.Tensor] = None
        pad_tail_sp: Optional[ttnn.Tensor] = None
        speech_log_len = 0
        speech_phys_len = 0
        speech_trace: Any = None

        enc_attn_padded: ttnn.Tensor
        enc_attn_owned: bool

        if use_speech:
            assert b.enc_in_tt is not None
            (
                conv_bf16,
                pad_tail_sp,
                speech_log_len,
                speech_phys_len,
                enc_attn_padded,
                speech_trace,
            ) = b.tt_model.materialize_speech_encoder_trace_tensors(b.enc_in_tt, b.enc_attn_tt)
            enc_attn_owned = True
        else:
            assert b.input_ids_tt is not None
            (
                enc_ids_p,
                enc_pos,
                enc_m4,
                enc_attn_padded,
                enc_attn_owned,
            ) = b.tt_model.materialize_text_encoder_trace_tensors(
                b.input_ids_tt,
                b.enc_attn_tt,
            )

        ttnn.synchronize_device(mesh_device)

        ids_p, pos_tt, causal_4d, cross_4d = b.tt_model.materialize_decoder_trace_tensors(
            enc_attn_padded,
            b.dec_ids_tt,
            b.dec_mask_tt,
        )
        ttnn.synchronize_device(mesh_device)

        if use_speech:
            assert b.enc_in_tt is not None and conv_bf16 is not None and speech_trace is not None
            if needs_t2u:
                assert t2u_pack is not None
                traced_forward = _make_traced_speech_e2e_plus_t2u_forward_fn(
                    b.tt_model,
                    b.enc_in_tt,
                    conv_bf16,
                    speech_trace,
                    pad_tail_sp,
                    speech_log_len,
                    speech_phys_len,
                    ids_p,
                    pos_tt,
                    causal_4d,
                    cross_4d,
                    t2u_pack,
                )
            else:
                traced_forward = _make_traced_speech_e2e_forward_fn(
                    b.tt_model,
                    b.enc_in_tt,
                    conv_bf16,
                    speech_trace,
                    pad_tail_sp,
                    speech_log_len,
                    speech_phys_len,
                    ids_p,
                    pos_tt,
                    causal_4d,
                    cross_4d,
                )
        else:
            if needs_t2u:
                assert t2u_pack is not None
                traced_forward = _make_traced_text_e2e_plus_t2u_forward_fn(
                    b.tt_model,
                    enc_ids_p,
                    enc_pos,
                    enc_m4,
                    ids_p,
                    pos_tt,
                    causal_4d,
                    cross_4d,
                    t2u_pack,
                )
            else:
                traced_forward = _make_traced_text_e2e_forward_fn(
                    b.tt_model,
                    enc_ids_p,
                    enc_pos,
                    enc_m4,
                    ids_p,
                    pos_tt,
                    causal_4d,
                    cross_4d,
                )

        profiler.clear()
        num_devices = mesh_device.get_num_devices()
        batch_size = batch_size_per_device * num_devices

        outputs = _run_pipeline(
            mesh_device,
            traced_forward,
            b.dummy_host,
            b.dram_config,
            b.l1_config,
            NUM_MEASUREMENT_ITERS,
            use_trace=True,
        )

        last_out = outputs[-1]
        if needs_t2u:
            assert isinstance(last_out, (list, tuple)) and len(last_out) == 2
            logits_host, t2u_host = last_out[0], last_out[1]
            assert_text_logits_pcc_vs_ref(b.ref_logits, logits_host, ctx=f"{b.task.upper()}_E2E_2CQ_TRACE")
            assert t2u_pack is not None
            _assert_t2u_logits_pcc_local(t2u_pack.ref_logits_bf16, t2u_host, ctx=f"{b.task.upper()}_E2E_2CQ_TRACE_T2U")
        else:
            assert_text_logits_pcc_vs_ref(b.ref_logits, last_out, ctx=f"{b.task.upper()}_E2E_2CQ_TRACE")

        _dealloc_if_allocated(pos_tt)
        _dealloc_if_allocated(causal_4d)
        _dealloc_if_allocated(cross_4d)
        if ids_p is not b.dec_ids_tt:
            _dealloc_if_allocated(ids_p)

        if use_speech:
            _dealloc_if_allocated(conv_bf16)
            _dealloc_if_allocated(pad_tail_sp)
            if speech_trace is not None:
                _dealloc_speech_trace_masks(speech_trace)
        else:
            _dealloc_if_allocated(enc_pos)
            _dealloc_if_allocated(enc_m4)
            if enc_ids_p is not b.input_ids_tt:
                _dealloc_if_allocated(enc_ids_p)

        if enc_attn_owned:
            ttnn.deallocate(enc_attn_padded)
        if b.enc_in_tt is not None:
            ttnn.deallocate(b.enc_in_tt)
        if b.input_ids_tt is not None:
            ttnn.deallocate(b.input_ids_tt)
        ttnn.deallocate(b.enc_attn_tt)
        ttnn.deallocate(b.dec_ids_tt)
        ttnn.deallocate(b.dec_mask_tt)

        if t2u_pack is not None:
            _dealloc_t2u_trace_pack(t2u_pack)

        compile_time = profiler.get("compile")
        inference_time_avg = profiler.get("run_model_pipeline_2cqs") / NUM_MEASUREMENT_ITERS
        expected_inference_time = batch_size / expected_inference_throughput
        prep_perf_report(
            model_name=f"ttnn_seamless_m4t_v2_large_2cqs_trace_batch_size{batch_size}_{b.task}",
            batch_size=batch_size,
            inference_and_compile_time=compile_time,
            inference_time=inference_time_avg,
            expected_compile_time=600,
            expected_inference_time=expected_inference_time,
            comments=f"task_{b.task}_trace_{'speech_e2e_t2u' if use_speech and needs_t2u else 'speech_e2e' if use_speech else 'text_e2e_t2u' if needs_t2u else 'text_e2e'}_batchsize{batch_size}",
            inference_time_cpu=0.0,
        )
        trace_kind = (
            "speech E2E + T2U"
            if use_speech and needs_t2u
            else "speech E2E"
            if use_speech
            else "text E2E + T2U"
            if needs_t2u
            else "text E2E"
        )
        logger.info(
            f"Seamless M4T v2 Large (2CQ trace {trace_kind}) task={b.task} batch_size={batch_size} "
            f"compile={compile_time:.2f}s inference_avg={inference_time_avg:.4f}s "
            f"FPS={batch_size / inference_time_avg:.2f}"
        )
