# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host contracts; no checkpoint load, TT tensor allocation, or mesh access."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from models.demos.gpt_oss_120b_qb2.tt.generator import Generator, TraceEvidence
from models.demos.gpt_oss_120b_qb2.tt.generator_vllm import TTGptOssForCausalLM
from models.demos.gpt_oss_120b_qb2.tt.model import decode_trace_buckets


@pytest.mark.parametrize("widths", [(1, 32), (32, 1), (4, 8)])
def test_host_logits_use_submission_width_without_mutating_model_args(widths):
    generator = object.__new__(Generator)
    generator.model_args = SimpleNamespace(max_batch_size=32)
    generator.model = SimpleNamespace(
        process_output_decode=lambda output, batch, **kwargs: output.reshape(batch, 1, -1)
    )
    for width in widths:
        flattened = torch.arange(width * 128)
        logits, _ = generator.process_decode_output_host([(flattened, None)], batch_size_per_model=(width,))
        assert logits.shape == (width, 1, 128)
        assert logits[-1, 0, -1] == width * 128 - 1
        assert generator.model_args.max_batch_size == 32


def test_teardown_calls_canonical_idempotent_release_once():
    generator = object.__new__(Generator)
    inner = SimpleNamespace(release_persistent_capture=Mock())
    generator._inner = inner
    generator._torn_down = False
    generator.teardown()
    generator.teardown()
    inner.release_persistent_capture.assert_called_once_with()


@pytest.mark.parametrize("width", [1, 4, 8, 32])
@pytest.mark.parametrize("read_from_device", [False, True])
def test_synchronous_host_decode_preserves_full_vocabulary(width, read_from_device):
    generator = object.__new__(Generator)
    generator.model_args = SimpleNamespace(max_batch_size=32, max_context_len=131072)
    generator.model = SimpleNamespace(
        n_layers=1, process_output_decode=lambda output, batch, **kwargs: output.reshape(batch, 1, -1)
    )
    device_output = object()
    generator._inner = SimpleNamespace(
        decode_forward=Mock(return_value=device_output),
        read_decode_output=Mock(return_value=[(torch.arange(width * 128), None)]),
    )
    generator.trace_evidence = TraceEvidence()
    generator._record_decode_staging = Mock()
    output = generator.decode_forward(
        tokens=torch.zeros(width, 1, dtype=torch.int64),
        start_pos=torch.zeros(width, dtype=torch.int64),
        page_table=torch.zeros(width, 1, dtype=torch.int32),
        kv_cache=[object()],
        enable_trace=False,
        sampling_mode="host",
        read_from_device=read_from_device,
    )
    assert generator._inner.decode_forward.call_args.kwargs["read_from_device"] is False
    if read_from_device:
        generator._inner.read_decode_output.assert_called_once_with(device_output)
        assert output.shape == (width, 128)
        assert output[-1, -1] == width * 128 - 1
    else:
        generator._inner.read_decode_output.assert_not_called()
        assert output is device_output


def test_ring_ownership_moves_with_slots_and_released_slot_resumes_cold():
    adapter = object.__new__(TTGptOssForCausalLM)
    adapter.max_batch_size = 4
    adapter.max_model_len = 131072
    adapter._sliding_layers = [0]
    adapter._ring_of_slot = [0, 1, 2, 3]
    adapter._slot_prefill_end = [8192, 4096, None, 1024]
    adapter._ring_tables_dirty = False
    adapter._apply_ring_slot_remap([1, 0, 3, 2])
    assert adapter._ring_of_slot == [1, 0, 3, 2]
    assert adapter._slot_prefill_end == [4096, 8192, 1024, None]
    assert adapter._ring_tables_dirty
    assert adapter._resume_plan(8192, 1) == (8192, 8192, False)
    adapter.release_request(1)
    assert adapter._resume_plan(8192, 1) == (5632, 8192, True)


def test_selected_buckets_exclude_historically_corrupt_sixteen_row_trace():
    assert decode_trace_buckets(32) == (1, 4, 8, 32)
    # Requests 9 through 16 therefore select the 32-row device graph.
    for active in (9, 15, 16, 17, 31, 32):
        assert next(width for width in decode_trace_buckets(32) if width >= active) == 32


def test_host_prefill_trims_each_layer_table_using_its_cache_block_axis():
    generator = object.__new__(Generator)
    generator.model_args = SimpleNamespace(max_batch_size=32, max_context_len=131072)
    generator.model = SimpleNamespace(n_layers=2)
    generator._inner = SimpleNamespace(mode=None)
    generator._prepare_prefill_variants = Mock(return_value=set())
    generator._record_compiled_prefill_variants = Mock()
    generator._prefill_one = Mock(return_value=torch.zeros(1, 128))
    # Individual caches are [K, V], each [physical_pages, kv_heads, block, head].
    caches = [[SimpleNamespace(shape=(64, 2, block, 64))] * 2 for block in (64, 128)]
    tables = [torch.arange(16).reshape(2, 8), torch.arange(32, 48).reshape(2, 8)]
    output = generator.prefill_forward(
        torch.zeros(2, 129, dtype=torch.int64),
        page_table=tables[0],
        kv_cache=caches,
        prompt_lens=[65, 129],
        page_tables_per_layer=tables,
    )
    assert output.shape == (2, 1, 128)
    for row, (first_width, second_width) in enumerate([(2, 1), (3, 2)]):
        passed = generator._prefill_one.call_args_list[row].kwargs["page_tables_per_layer"]
        assert torch.equal(passed[0], tables[0][row : row + 1, :first_width])
        assert torch.equal(passed[1], tables[1][row : row + 1, :second_width])
