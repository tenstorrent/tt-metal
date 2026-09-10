# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bounded direct-execution workload for production Llama optimizer runs."""

from __future__ import annotations

import gc
import time

import pytest
import torch

import ttnn
from models.common.sampling import SamplingParams
from models.tt_transformers.tt.common import PagedAttentionConfig, create_tt_model
from models.tt_transformers.tt.generator import Generator
from models.tt_transformers.tt.model_config import DecodersPrecision


INPUT_IDS = [128000] + [1000 + ((index * 7919) % 120000) for index in range(127)]
INPUT_TOKENS = len(INPUT_IDS)
OUTPUT_TOKENS = 256
DECODE_TOKENS = OUTPUT_TOKENS - 1
MAX_SEQ_LEN = 2048


def _prefill(generator, input_ids, page_table, kv_cache, sampling_params):
    result = generator.prefill_forward_text(
        input_ids,
        page_table=page_table,
        kv_cache=kv_cache,
        prompt_lens=[INPUT_TOKENS],
        sampling_params=sampling_params,
        enable_trace=True,
        warmup_prefill=False,
    )
    first_token = result[0] if isinstance(result, tuple) else result
    return first_token.reshape(1, 1)


def _decode(generator, first_token, input_ids, page_table, kv_cache, sampling_params):
    current_pos = torch.tensor([INPUT_TOKENS], dtype=torch.int64)
    for decode_index in range(DECODE_TOKENS):
        generator.decode_forward(
            first_token,
            current_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            enable_trace=True,
            read_from_device=False,
            sampling_params=sampling_params,
            reset_batch=(decode_index == 0),
            prompt_tokens=input_ids if decode_index == 0 else None,
            output_tokens=first_token if decode_index == 0 else None,
        )
        current_pos += 1


@pytest.mark.no_reset_default_device
@pytest.mark.timeout(300)
def test_optimizer_direct_perf():
    """Measure traced 128-token prefill and device-resident greedy decode on QB2."""
    mesh_device = generator = model = model_args = state_dict = tt_kv_cache = None
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh_device = ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(1, 4),
            trace_region_size=52_000_000,
            num_command_queues=1,
        )
        paged_attention_config = PagedAttentionConfig(block_size=32, max_num_blocks=1024)
        optimizations = lambda args: DecodersPrecision.performance(args.n_layers, args.model_name)
        model_args, model, tt_kv_cache, state_dict = create_tt_model(
            mesh_device,
            instruct=True,
            max_batch_size=1,
            optimizations=optimizations,
            max_seq_len=MAX_SEQ_LEN,
            paged_attention_config=paged_attention_config,
            dtype=ttnn.bfloat8_b,
            num_layers=None,
            use_prefetcher=False,
            use_hf_rope=False,
        )
        assert model_args.n_layers == 32
        assert len(model.layers) == 32
        assert mesh_device.get_num_devices() == 4
        assert model._supports_on_device_sampling
        assert model.sampling is not None
        state_dict = None
        gc.collect()

        generator = Generator([model], [model_args], mesh_device, tokenizer=model_args.tokenizer)
        input_ids = torch.tensor([INPUT_IDS], dtype=torch.long)
        page_table = torch.arange(paged_attention_config.max_num_blocks, dtype=torch.int32).reshape(
            1, paged_attention_config.max_num_blocks
        )
        kv_cache = [tt_kv_cache]
        sampling_params = SamplingParams(temperature=0.0, top_k=1, top_p=1.0, seed=0)

        warmup_token = _prefill(generator, input_ids, page_table, kv_cache, sampling_params)
        generator.decode_forward(
            warmup_token,
            torch.tensor([INPUT_TOKENS], dtype=torch.int64),
            page_table=page_table,
            kv_cache=kv_cache,
            enable_trace=True,
            read_from_device=False,
            sampling_params=sampling_params,
            reset_batch=True,
            prompt_tokens=input_ids,
            output_tokens=warmup_token,
        )
        ttnn.synchronize_device(mesh_device)

        start = time.perf_counter()
        first_token = _prefill(generator, input_ids, page_table, kv_cache, sampling_params)
        ttnn.synchronize_device(mesh_device)
        ttft_end = time.perf_counter()
        _decode(generator, first_token, input_ids, page_table, kv_cache, sampling_params)
        ttnn.synchronize_device(mesh_device)
        end = time.perf_counter()

        decode_seconds = end - ttft_end
        print(
            f"PERF wall_ms={(end - start) * 1000.0:.3f} "
            f"ttft_ms={(ttft_end - start) * 1000.0:.3f} "
            f"decode_tokens_per_second={DECODE_TOKENS / decode_seconds:.3f}",
            flush=True,
        )
    finally:
        generator = None
        model = None
        model_args = None
        state_dict = None
        tt_kv_cache = None
        gc.collect()
        if mesh_device is not None:
            for submesh in list(mesh_device.get_submeshes()):
                if submesh is not mesh_device:
                    ttnn.close_mesh_device(submesh)
            ttnn.close_mesh_device(mesh_device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
