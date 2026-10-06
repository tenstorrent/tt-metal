# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host checks that persistent state slots cannot redirect paged KV writes."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm_tt_plugin.input_batch import InputBatch
from vllm_tt_plugin.model_runner import TTModelRunner

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tt.generator import Gemma4Generator
from models.autoports.google_gemma_4_26b_a4b_it.tt.generator_vllm import AutoportGemma4ForCausalLM
from models.common.sampling.generator import SamplingParams as DeviceSamplingParams
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu_input_batch import CachedRequestState


@pytest.mark.parametrize("held_slots,new_requests", [(0, 1), (0, 2), (1, 1), (1, 2), (3, 1)])
def test_prefill_cache_rows_are_compact_despite_live_state_slots(monkeypatch, held_slots, new_requests):
    """Use real host input building/submission and adapter/generator forwarding.

    Only TTNN operations and model arithmetic are replaced. The recorded row
    is the user_id passed to model prefill and ultimately paged_fill_cache.
    """
    monkeypatch.setattr(ttnn, "reshape", lambda value, shape: value.reshape(shape))
    monkeypatch.setattr(ttnn, "concat", lambda values, dim: torch.cat(values, dim=dim))
    monkeypatch.setattr(ttnn, "get_device_tensors", lambda value: [value])
    monkeypatch.setattr(ttnn, "to_torch", lambda value: value)

    selected = []

    def model_prefill(ids, *, page_table, kv_cache, user_id, return_all_logits):
        selected.append((int(ids.flatten()[0]), user_id, [int(table[user_id, 0]) for table in page_table]))
        return torch.zeros(1, 1, 1, 4)

    generator = Gemma4Generator.__new__(Gemma4Generator)
    generator.model = SimpleNamespace(
        max_seq_len=256,
        config=SimpleNamespace(vocab_size=16),
        layer_indices=(0, 1),
        upload=lambda value, *args: value,
        prefill_forward=model_prefill,
    )
    generator._upload_page_tables = lambda tables: tables
    generator.configure_sampling = lambda *args, **kwargs: None
    generator.sample_prefill = lambda logits: torch.zeros(32, dtype=torch.int32)
    generator.mesh = None
    generator.host_sampling = False
    generator.prefill_trace_enabled = False
    generator._release_trace = lambda: None
    adapter = AutoportGemma4ForCausalLM(generator, 8)

    runner = TTModelRunner.__new__(TTModelRunner)
    runner.model = adapter
    runner.model_config = SimpleNamespace(is_multimodal_model=False)
    runner.cache_config = SimpleNamespace(block_size=32)
    runner.scheduler_config = SimpleNamespace(enable_chunked_prefill=False)
    runner.tt_per_lane_max_num_seqs = 8
    runner.tt_max_batch_size = 8
    runner.tt_data_parallel_size = 1
    runner.max_num_blocks_per_req = 8
    runner._layer_to_group_idx = [0, 1]
    runner._req_state_slot = {f"held{i}": i for i in range(held_slots)}
    runner.requests = dict.fromkeys(runner._req_state_slot)
    runner.check_perform_device_sampling = lambda **kwargs: True
    runner.trace_mode = "decode"
    runner.request_specific_rope = False
    runner.kv_caches = object()
    runner.input_batch = InputBatch(
        max_num_reqs=8,
        max_model_len=256,
        max_num_batched_tokens=512,
        vocab_size=256,
        block_sizes=[32, 32],
        kernel_block_sizes=[32, 32],
    )
    expected = []
    output = SchedulerOutput.make_empty()
    for row in range(new_requests):
        req_id = f"new{row}"
        length = 31 + row * 32
        pages = (length + 31) // 32
        request = CachedRequestState(
            req_id=req_id,
            prompt_token_ids=[100 + row] * length,
            mm_features=[],
            sampling_params=SamplingParams(temperature=0),
            generator=None,
            block_ids=(
                list(range(10 + row * 8, 10 + row * 8 + pages)),
                list(range(100 + row * 8, 100 + row * 8 + pages)),
            ),
            num_computed_tokens=0,
            output_token_ids=[],
        )
        runner.requests[req_id] = request
        runner.input_batch.add_request(request)
        output.scheduled_new_reqs.append(SimpleNamespace(req_id=req_id))
        output.num_scheduled_tokens[req_id] = length
        output.total_num_scheduled_tokens += length
        expected.append((100 + row, row, [10 + row * 8, 100 + row * 8]))
    runner.input_batch.refresh_logitsprocs()

    model_input = runner._prepare_model_inputs(output, None)
    assert model_input.prefill_empty_slots == list(range(held_slots, held_slots + new_requests))
    runner.submit_prefill(model_input, [new_requests])

    assert selected == expected


@pytest.mark.parametrize("batch", [1, 2])
def test_trace_prefill_compacts_scheduler_table_rows_without_truncating_columns(monkeypatch, batch):
    """A compact token batch can arrive with the full 32-row serving tables."""
    monkeypatch.setattr(ttnn, "get_device_tensors", lambda value: [value])
    monkeypatch.setattr(ttnn, "to_torch", lambda value: value)
    sliding = torch.arange(32 * 8192, dtype=torch.int32).reshape(32, 8192)
    full = sliding + 1000000
    tables = [sliding] * 30
    tables[5] = full
    snapshots = [sliding.clone(), full.clone()]
    cache = object()
    generator = SimpleNamespace(
        mesh=None,
        host_sampling=False,
        model=SimpleNamespace(layer_indices=(0, 5)),
        prefill_trace_enabled=True,
        prefill_prepared={},
        _release_trace=Mock(),
        serving_prefill_eligible=Mock(return_value=True),
        _serving_prefill_key=Mock(return_value=("short-prefill",)),
        can_reuse_serving_prefill=Mock(return_value=True),
        configure_sampling=Mock(),
        serving_prefill_tokens=Mock(return_value=torch.arange(32, dtype=torch.int32)),
    )
    adapter = AutoportGemma4ForCausalLM(generator, 32)
    params = DeviceSamplingParams(temperature=0.0, top_k=1, top_p=1.0)
    output = adapter.prefill_forward(
        torch.full((batch, 128), 100),
        sliding,
        cache,
        [128] * batch,
        sampling_params=params,
        page_tables_per_layer=tables,
        empty_slots=list(range(7, 7 + batch)),
    )
    assert output.flatten().tolist() == list(range(batch))
    for method in (generator.can_reuse_serving_prefill, generator.serving_prefill_tokens):
        method.assert_called_once()
        kwargs = method.call_args.kwargs
        assert kwargs["kv_cache"] is cache
        assert kwargs["prompt_lens"] == [128] * batch
        for observed, original in zip(kwargs["page_table"], (sliding, full)):
            assert observed.shape == (batch, 8192)
            assert observed.data_ptr() == original.data_ptr()
            assert torch.equal(observed, original[:batch])
    assert generator.configure_sampling.call_args.kwargs["_reuse_trace"] is True
    assert torch.equal(sliding, snapshots[0])
    assert torch.equal(full, snapshots[1])
