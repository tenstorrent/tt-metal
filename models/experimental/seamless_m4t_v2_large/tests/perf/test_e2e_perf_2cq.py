# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
E2E performance test for Seamless M4T v2 Large with **2CQ, non-traced** pipeline on Blackhole.

Uses ``PipelineConfig(use_trace=False, num_command_queues=2)`` — ``MultiCQModelOverlappedInputExecutor``
(overlapped H2D + compute on two CQs; no ``begin_trace_capture`` / ``execute_trace``).

Baseline single-CQ E2E (no Pipeline) lives in ``test_e2e_perf.py``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Optional

import pytest
import torch
import ttnn
from loguru import logger
from transformers import AutoProcessor, AutoTokenizer

from models.common.utility_functions import profiler, run_for_blackhole
from models.demos.utils.common_demo_utils import get_mesh_mappers
from models.experimental.seamless_m4t_v2_large.reference.torch_seamless_m4t_v2_model import (
    forward_text_modality_logits,
    load_pretrained_seamless_m4t_v2_model,
)
from models.experimental.seamless_m4t_v2_large.tests.pcc.test_seamless_m4t_v2_model import (
    _decoder_seed,
    _real_speech_features,
    _real_text_input_ids,
    _weights_dir_or_skip,
    make_tt_model,
    torch_feats_to_ttnn,
    torch_ids_to_ttnn,
)
from models.experimental.seamless_m4t_v2_large.tests.perf.test_e2e_perf import (
    SEAMLESS_E2E_TASKS,
    assert_text_logits_pcc_vs_ref,
    t2u_timed_stages_for_e2e,
)
from models.perf.perf_utils import prep_perf_report
from models.tt_cnn.tt.pipeline import (
    PipelineConfig,
    create_pipeline_from_config,
    get_memory_config_for_persistent_dram_tensor,
)

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
    def forward(_device_input: ttnn.Tensor) -> ttnn.Tensor:
        if use_speech:
            assert enc_in_tt is not None
            enc_tt, enc_attn_padded, enc_attn_owned = tt_model._encode_speech(enc_in_tt, enc_attn_tt)
        else:
            assert input_ids_tt is not None
            enc_tt, enc_attn_padded, enc_attn_owned = tt_model._encode_text(input_ids_tt, enc_attn_tt)
        logits = tt_model._decode_and_lm_head(enc_tt, enc_attn_padded, dec_ids_tt, dec_mask_tt)
        if enc_attn_owned:
            ttnn.deallocate(enc_attn_padded)
        ttnn.deallocate(enc_tt)
        return logits

    return forward


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
) -> list:
    pipeline = create_pipeline_from_config(
        config=PipelineConfig(
            use_trace=False,
            num_command_queues=2,
            all_transfers_on_separate_command_queue=False,
        ),
        model=forward_fn,
        device=device,
        dram_input_memory_config=dram_config,
        l1_input_memory_config=l1_config,
    )
    ttnn.synchronize_device(device)
    profiler.start("compile")
    pipeline.compile(dummy_host)
    profiler.end("compile")
    host_inputs = [dummy_host] * num_iters
    profiler.start("run_model_pipeline_2cqs")
    outputs = pipeline.enqueue(host_inputs).pop_all()
    profiler.end("run_model_pipeline_2cqs")
    pipeline.cleanup()
    return outputs


@run_for_blackhole()
@pytest.mark.models_performance_bare_metal
@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 32768, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize("batch_size_per_device", (1,))
@pytest.mark.parametrize("expected_inference_throughput", (0.02,))
@pytest.mark.parametrize("task", list(SEAMLESS_E2E_TASKS.keys()))
def test_seamless_m4t_v2_large_e2e_perf_2cq(
    device, batch_size_per_device, expected_inference_throughput: float, task: str
):
    """Compile + 2CQ overlapped pipeline (non-traced); PCC on last host logits; optional T2U after pipeline."""
    b = _build_model_for_task(device, task)
    profiler.clear()
    num_devices = device.get_num_devices()
    batch_size = batch_size_per_device * num_devices

    outputs = _run_pipeline(
        device,
        b.forward_fn,
        b.dummy_host,
        b.dram_config,
        b.l1_config,
        NUM_MEASUREMENT_ITERS,
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
        t2u_timed_stages_for_e2e(b.model, b.tt_model, device, ctx=f"{b.task.upper()}_E2E_T2U")

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
