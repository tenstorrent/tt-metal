# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
BGE-M3 Tracy kernel-level profiling benchmark.

Runs a single forward pass inside Tracy signposts to generate device-level
op reports. Two cases: B1/S512 and B32/S512, no trace capture (Tracy needs
to see the individual ops, not a trace replay).

test_n300_dp_tracy profiles the B12/S8192 shape on a 2-chip Wormhole card.
test_bge_m3_tracy_perf profiles the S512 batch sweep on one device.

Usage (run from tt-metal root):
    TT_VISIBLE_DEVICES=0 TT_METAL_DEVICE_PROFILER=1 python -m tracy -p -r --no-runtime-analysis -v -m pytest models/demos/wormhole/bge_m3/tests/perf/tracy_perf.py -k "batch1" -sv
    TT_VISIBLE_DEVICES=0 TT_METAL_DEVICE_PROFILER=1 python -m tracy -p -r --no-runtime-analysis -v -m pytest models/demos/wormhole/bge_m3/tests/perf/tracy_perf.py -k "batch32" -sv

Reports are saved to: generated/profiler/reports/<timestamp>/ops_perf_results_<timestamp>.csv
"""

import os

import pytest
import torch

import ttnn
from models.demos.wormhole.bge_m3.tt.common import create_tt_model

try:
    from tracy import signpost
except ImportError:

    def signpost(*_args, **_kwargs):
        return None


SEQ_LEN_8192 = 8192


def prepare_inputs(tokenizer, batch_size, seq_len, pad_token_id):
    """Generate synthetic token inputs on host. Returns dict of torch tensors."""
    # Ids start at 5, above the special ids, so no row holds the pad id 1. The
    # nomask profile states no_padding, which is correct only for such input.
    input_ids = torch.randint(5, 1000, (batch_size, seq_len), dtype=torch.long)
    token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)

    mask = (input_ids != pad_token_id).to(torch.int64)
    position_ids = (torch.cumsum(mask, dim=1) * mask + pad_token_id).to(torch.long)

    # The profiled forward reads the token inputs only. A dense [B,1,S,S] mask
    # costs about 1.5 GiB at B12/S8192, so this function does not build one.
    return {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "position_ids": position_ids,
    }


@pytest.mark.parametrize("mesh_device", [(2, 1)], indirect=True, ids=["dp2_n300"])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "trace_region_size": 50_000_000,
            "num_command_queues": 1,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        }
    ],
    indirect=True,
)
def test_n300_dp_tracy(mesh_device):
    """Profile one B12/S8192 DP=2 forward between Tracy signposts for per-op
    device timing. Untraced so Tracy sees the individual ops. Requires
    TT_METAL_DEVICE_PROFILER=1."""
    if os.environ.get("TT_METAL_DEVICE_PROFILER", "0") != "1":
        pytest.fail("TT_METAL_DEVICE_PROFILER=1 is required for device kernel profiling.")

    assert tuple(mesh_device.shape) == (2, 1)
    assert mesh_device.get_num_devices() == 2

    model_args, model, _ = create_tt_model(
        mesh_device=mesh_device,
        max_batch_size=12,
        max_seq_len=SEQ_LEN_8192,
        dtype=ttnn.bfloat8_b,
        data_parallel=True,
    )
    assert model._data_parallel, "DP mode not active"

    inputs = prepare_inputs(model_args.tokenizer, 12, SEQ_LEN_8192, model_args.pad_token_id)
    mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    device_inputs = {
        key: ttnn.from_torch(
            inputs[key].int(),
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for key in ("input_ids", "token_type_ids", "position_ids")
    }

    out = model.forward(**device_inputs)
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(out)

    signpost("start")
    out = model.forward(**device_inputs)
    ttnn.synchronize_device(mesh_device)
    signpost("stop")
    ttnn.deallocate(out)


SEQ_LEN_512 = 512


@pytest.mark.parametrize("masked", [False, True], ids=["nomask", "masked"])
@pytest.mark.parametrize(
    "batch_size",
    [1, 8, 16, 32],
    ids=["batch1", "batch8", "batch16", "batch32"],
)
@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 50_000_000, "num_command_queues": 1}],
    indirect=True,
)
def test_bge_m3_tracy_perf(device, batch_size, masked):
    """Profile one S512 forward between Tracy signposts for per-op device timing.

    Untraced, because Tracy reports the individual ops and a replay hides them.
    Requires TT_METAL_DEVICE_PROFILER=1.
    """
    if os.environ.get("TT_METAL_DEVICE_PROFILER", "0") != "1":
        pytest.fail("TT_METAL_DEVICE_PROFILER=1 is required for device kernel profiling.")

    model_args, model, _ = create_tt_model(
        mesh_device=device,
        max_batch_size=batch_size,
        max_seq_len=SEQ_LEN_512,
        dtype=ttnn.bfloat8_b,
    )

    inputs = prepare_inputs(model_args.tokenizer, batch_size, SEQ_LEN_512, model_args.pad_token_id)
    device_inputs = {
        key: ttnn.from_torch(
            inputs[key].int(),
            device=device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for key in ("input_ids", "token_type_ids", "position_ids")
    }

    # nomask matches perf.py: the input holds no pad token, so the model may
    # skip the dense mask. masked keeps the default path.
    no_padding = not masked

    # Compile outside the signpost window.
    out = model.forward(**device_inputs, no_padding=no_padding)
    ttnn.synchronize_device(device)
    ttnn.deallocate(out)

    signpost("start")
    out = model.forward(**device_inputs, no_padding=no_padding)
    ttnn.synchronize_device(device)
    signpost("stop")
    ttnn.deallocate(out)
