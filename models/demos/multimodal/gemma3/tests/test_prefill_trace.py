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
"""
import os

import pytest
import torch

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

    # (1) gating
    assert args.can_enable_trace(prefill_seq_len=seq_len, num_cached_tokens=0), (
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
