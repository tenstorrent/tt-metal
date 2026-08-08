# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
BGE-M3 Tracy kernel-level profiling benchmark.

Runs a single forward pass inside Tracy signposts to generate device-level
op reports. Two cases: B1/S512 and B32/S512, no trace capture (Tracy needs
to see the individual ops, not a trace replay).

Usage (run from tt-metal root):
    TT_VISIBLE_DEVICES=0 TT_METAL_DEVICE_PROFILER=1 python -m tracy -p -r --no-runtime-analysis -v -m pytest tracy_perf.py -k "batch1" -sv
    TT_VISIBLE_DEVICES=0 TT_METAL_DEVICE_PROFILER=1 python -m tracy -p -r --no-runtime-analysis -v -m pytest tracy_perf.py -k "batch32" -sv

Reports are saved to: generated/profiler/reports/<timestamp>/ops_perf_results_<timestamp>.csv
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.wormhole.bge_m3.tt.common import create_tt_model
from models.demos.wormhole.bge_m3.tt.model_config import determine_device_name

try:
    from tracy import signpost
except ImportError:

    def signpost(*_args, **_kwargs):
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Input preparation (same helpers as perf.py)
# ──────────────────────────────────────────────────────────────────────────────

SEQ_LEN = 512


def prepare_inputs(tokenizer, batch_size, seq_len, pad_token_id):
    """Generate synthetic token inputs on host. Returns dict of torch tensors."""
    input_ids = torch.randint(1, 1000, (batch_size, seq_len), dtype=torch.long)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)

    mask = (input_ids != pad_token_id).to(torch.int64)
    position_ids = (torch.cumsum(mask, dim=1) * mask + pad_token_id).to(torch.long)

    keep = attention_mask.to(torch.bfloat16)
    additive_mask = ((1.0 - keep) * -100000.0).unsqueeze(1).unsqueeze(1).expand(-1, -1, seq_len, -1).contiguous()

    return {
        "input_ids": input_ids,
        "attention_mask": additive_mask,
        "token_type_ids": token_type_ids,
        "position_ids": position_ids,
    }


def to_device(inputs, mesh_device, mask_dtype):
    """Move host tensors to device."""

    def ids_to_dev(t):
        return ttnn.from_torch(t.int(), device=mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)

    return {
        "input_ids": ids_to_dev(inputs["input_ids"]),
        "attention_mask": ttnn.from_torch(
            inputs["attention_mask"],
            device=mesh_device,
            dtype=mask_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
        "token_type_ids": ids_to_dev(inputs["token_type_ids"]),
        "position_ids": ids_to_dev(inputs["position_ids"]),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Test
# ──────────────────────────────────────────────────────────────────────────────


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
def test_bge_m3_tracy_perf(mesh_device, batch_size):
    """
    Single forward pass inside Tracy signposts for kernel-level profiling.

    Tracy captures every device op between signpost("start") and signpost("stop").
    We do NOT use trace capture here — Tracy needs to see the individual ops
    (matmul, layernorm, SDPA, etc.), not a single trace replay op.
    """
    if os.environ.get("TT_METAL_DEVICE_PROFILER", "0") != "1":
        pytest.fail(
            "TT_METAL_DEVICE_PROFILER=1 is required for device kernel profiling. "
            "Without it, Tracy only captures host-side ops and generates an empty report. "
            "Re-run with: TT_METAL_DEVICE_PROFILER=1 TT_VISIBLE_DEVICES=0 python -m tracy ..."
        )
    dtype = ttnn.bfloat8_b
    device_name = determine_device_name(mesh_device)[0]

    # ── Build model ──────────────────────────────────────────────────
    logger.info(f"Building model: B{batch_size} S{SEQ_LEN} {device_name}")
    model_args, model, _ = create_tt_model(
        mesh_device=mesh_device,
        max_batch_size=batch_size,
        max_seq_len=SEQ_LEN,
        dtype=dtype,
    )

    # ── Prepare inputs ───────────────────────────────────────────────
    mask_dtype = dtype if batch_size in (1, 8, 16, 32) else ttnn.bfloat16
    host_inputs = prepare_inputs(model_args.tokenizer, batch_size, SEQ_LEN, model_args.pad_token_id)
    device_inputs = to_device(host_inputs, mesh_device, mask_dtype)

    # ── Warmup (compile) — outside signpost window ───────────────────
    logger.info("Compiling (warmup forward)...")
    out = model.forward(**device_inputs)
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(out)
    logger.info("Warmup done.")

    # ── Profiled forward — single pass inside signposts ──────────────
    logger.info(f"Running profiled forward: B{batch_size} S{SEQ_LEN}")
    signpost("start")
    out = model.forward(**device_inputs)
    ttnn.synchronize_device(mesh_device)
    signpost("stop")

    logger.info("Tracy profiling complete.")
    ttnn.deallocate(out)
