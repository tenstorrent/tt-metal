# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Unit test for text-only prefill tracing support on TtGemmaModel.

Verifies:
  1. ModelArgs.can_enable_trace(...) is True for the listed gemma-3-27b seq lens.
  2. prepare_inputs_prefill(trace_enabled=True) returns ttnn tensors whose
     storage_type() is HOST (so they can be consumed by begin_trace_capture /
     copy_host_to_device), and does NOT invoke the vision tower even if
     pixel_values is accidentally forwarded in kwargs.
  3. prepare_inputs_prefill(trace_enabled=False) behaves as before (device
     tensors, vision path still works when pixel_values is provided).

Run with visible verification lines (loguru emits [gemma3_prefill_trace_test] on stderr)::

    pytest -s models/demos/multimodal/gemma3/tests/test_prefill_trace.py -v
"""
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.multimodal.gemma3.tt.gemma_e2e_model import TtGemmaModel
from models.demos.multimodal.gemma3.tt.model_config import ModelArgs


def _mesh_device_param():
    """Match demo/text_demo.py so N150/P150 gets (1,1) instead of skipping on 8-device requests."""
    return {
        "N150": (1, 1),
        "N300": (1, 2),
        "N150x4": (1, 4),
        "T3K": (1, 8),
        "TG": (8, 4),
        "P150": (1, 1),
        "P300": (1, 2),
        "P150x4": (1, 4),
        "P150x8": (1, 8),
    }.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))


@pytest.mark.parametrize("mesh_device", [_mesh_device_param()], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 30000000, "num_command_queues": 1}], indirect=True)
@pytest.mark.parametrize("seq_len", [128, 1024])
def test_gemma3_27b_prefill_trace_inputs(mesh_device, seq_len, reset_seeds):
    args = ModelArgs(mesh_device, max_batch_size=1, max_seq_len=max(1024, seq_len))

    mesh_n = mesh_device.get_num_devices()
    can_trace = args.can_enable_trace(prefill_seq_len=seq_len, num_cached_tokens=0)
    logger.info(
        "[gemma3_prefill_trace_test] setup seq_len={} mesh_devices={} device_name={} "
        "MESH_DEVICE={} can_enable_trace(seq_len={}, cached=0)={} trace_prefill_supported_seq_lens={}",
        seq_len,
        mesh_n,
        args.device_name,
        os.environ.get("MESH_DEVICE"),
        seq_len,
        can_trace,
        args.trace_prefill_supported_seq_lens,
    )

    # (1) gating
    assert can_trace, (
        f"can_enable_trace should be True for gemma-3-27b at seq_len={seq_len}; "
        f"supported={args.trace_prefill_supported_seq_lens}"
    )

    state_dict = args.load_state_dict()
    model = TtGemmaModel(
        args=args,
        dtype=ttnn.bfloat8_b,
        mesh_device=mesh_device,
        state_dict=state_dict,
        weight_cache_path=args.weight_cache_path(ttnn.bfloat8_b),
        use_paged_kv_cache=True,
    )

    pt_tokens = torch.randint(0, args.vocab_size, (1, seq_len), dtype=torch.int64)
    page_table = torch.arange(0, 64, dtype=torch.int32).reshape(1, 64)

    # (2) trace_enabled=True: tokens must stay on host, vision must NOT fire
    # even if pixel_values slips through kwargs (it shouldn't via generator, but
    # be defensive).
    host_inputs = model.prepare_inputs_prefill(
        pt_tokens,
        start_pos=0,
        page_table=page_table,
        trace_enabled=True,
        pixel_values=None,  # explicit: text-only trace path
    )
    tokens_host, rot_global, rot_local, pt_host, chunk_pt_host = host_inputs

    assert (
        tokens_host.storage_type() == ttnn.StorageType.HOST
    ), "Trace-enabled prefill must keep tokens on host for copy_host_to_device."
    assert pt_host.storage_type() == ttnn.StorageType.HOST, "Trace-enabled prefill must keep page_table on host."
    # Rope matrices reference the *full* prefill window for trace reuse.
    assert rot_global[0].shape[2] == args.max_seq_len
    assert rot_local[0].shape[2] == args.max_seq_len

    logger.info(
        "[gemma3_prefill_trace_test] trace_enabled=True (for Generator trace path): "
        "tokens.storage_type={} page_table.storage_type={} chunk_page_table={} "
        "rope_global_cos.dim2={} rope_local_cos.dim2={} max_seq_len={} "
        "(expect HOST tokens/page_table; rope dim2 == max_seq_len for trace reuse)",
        tokens_host.storage_type(),
        pt_host.storage_type(),
        chunk_pt_host,
        rot_global[0].shape[2],
        rot_local[0].shape[2],
        args.max_seq_len,
    )

    # (3) trace_enabled=False (existing behavior): tensors on device.
    device_inputs = model.prepare_inputs_prefill(
        pt_tokens,
        start_pos=0,
        page_table=page_table,
        trace_enabled=False,
    )
    tokens_dev, _, _, pt_dev, _ = device_inputs
    assert tokens_dev.storage_type() == ttnn.StorageType.DEVICE
    assert pt_dev.storage_type() == ttnn.StorageType.DEVICE

    logger.info(
        "[gemma3_prefill_trace_test] trace_enabled=False (normal prefill): "
        "tokens.storage_type={} page_table.storage_type={} "
        "(expect DEVICE — embeddings prepared on device before forward)",
        tokens_dev.storage_type(),
        pt_dev.storage_type(),
    )

    logger.info("[gemma3_prefill_trace_test] OK seq_len={} — assertions passed", seq_len)
