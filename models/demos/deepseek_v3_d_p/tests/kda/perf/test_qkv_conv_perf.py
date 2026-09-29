# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Single-device unit test for the Kimi-K3 KDA QKV causal convolution at Galaxy SP8xTP4 shape.

Per device: 640 local tokens, TP-local q/k/v widths of 3072 each, production channel chunk 512.
The op is timed as a traced replay loop and compared against a torch reference. Set
KDA_QKV_CONV_GOLDEN to a file path to save (first run) or bit-compare (later runs) the outputs.
The latency bound is calibrated for a single Galaxy Blackhole device.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, run_for_blackhole
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import (
    make_actual_start,
    qkv_device_inputs,
    qkv_reference,
)

pytestmark = [run_for_blackhole(), pytest.mark.perf]

_ROWS = 640
_WIDTHS = (3072, 3072, 3072)
_CHANNEL_CHUNK_SIZE = 512
_REPLAYS = 20
_PCC_THRESHOLD = 0.9999
# Galaxy single-device traced wall time, 2026-09-28: 315 us before the reader rework, 173 us after.
_MAX_US = 185.0


def _traced_us(device, op) -> tuple[float, tuple[ttnn.Tensor, ...]]:
    for output in op():
        ttnn.deallocate(output)
    ttnn.synchronize_device(device)
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    outputs = op()
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    start = time.perf_counter()
    for _ in range(_REPLAYS):
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    elapsed_us = (time.perf_counter() - start) * 1e6 / _REPLAYS
    ttnn.release_trace(device, trace_id)
    return elapsed_us, outputs


@pytest.mark.parametrize("device_params", [{"trace_region_size": 1 << 20}], indirect=True)
@pytest.mark.parametrize("actual_start", [0, 672], ids=["start0", "start672"])
def test_kda_qkv_conv_perf(device, actual_start) -> None:
    (inputs, history, taps), (input_tt, history_tt, taps_tt) = qkv_device_inputs(device, sequence=_ROWS, widths=_WIDTHS)
    actual_start_tt = make_actual_start(device, actual_start)
    program_config = ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=_CHANNEL_CHUNK_SIZE)

    def op():
        return ttnn.experimental.kda.qkv_causal_conv1d_silu(
            input_tt,
            history_tt,
            *taps_tt,
            *_WIDTHS,
            actual_start=actual_start_tt,
            predecessor_carry=history_tt,
            program_config=program_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    elapsed_us, outputs = _traced_us(device, op)
    logger.info(f"QKV conv M={_ROWS} widths={_WIDTHS} start={actual_start}: {elapsed_us:.1f} us")
    outputs = [ttnn.to_torch(output) for output in outputs]
    for name, golden, output in zip("qkv", qkv_reference(inputs, history, taps, _WIDTHS), outputs):
        _, pcc = comp_pcc(golden.float(), output.float())
        assert pcc >= _PCC_THRESHOLD, f"{name} PCC {pcc:.6f} < {_PCC_THRESHOLD}"

    golden_path = os.environ.get("KDA_QKV_CONV_GOLDEN")
    if golden_path:
        path = Path(f"{golden_path}.start{actual_start}.pt")
        if path.exists():
            saved = torch.load(path)
            for name, expected, output in zip("qkv", saved, outputs):
                assert torch.equal(expected, output), f"{name} differs from saved outputs"
            logger.info(f"outputs bit-identical to {path}")
        else:
            torch.save(outputs, path)
            logger.info(f"saved outputs to {path}")
    assert elapsed_us <= _MAX_US, f"QKV conv {elapsed_us:.1f} us regressed past {_MAX_US} us"


# The fused input projection is 12440 columns wide per device; its leading Q+K+V columns are the channels.
_PROJECTION_WIDTH = 12440


@pytest.mark.parametrize("device_params", [{"trace_region_size": 1 << 20}], indirect=True)
@pytest.mark.parametrize("channel_chunk_size", [512, 768])
@pytest.mark.parametrize("actual_start", [0, 672, 640 * 7 + 96], ids=["start0", "start672", "split96"])
def test_kda_qkv_conv_reads_tiled_projection(device, actual_start, channel_chunk_size) -> None:
    """Reading the tiled projection in place is bit-identical to convolving its row-major channel slice."""
    (inputs, _, _), (input_tt, history_tt, taps_tt) = qkv_device_inputs(device, sequence=_ROWS, widths=_WIDTHS)
    channels = sum(_WIDTHS)
    extra = torch.randn(1, _ROWS, _PROJECTION_WIDTH - channels).to(inputs.dtype)
    projection_tt = ttnn.from_torch(
        torch.cat([inputs.reshape(1, _ROWS, channels), extra], dim=-1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    predecessor_tt = ttnn.from_torch(
        torch.randn(1, 3, channels), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    actual_start_tt = make_actual_start(device, actual_start)
    program_config = ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=channel_chunk_size)

    def convolve(source):
        return lambda: ttnn.experimental.kda.qkv_causal_conv1d_silu(
            source,
            history_tt,
            *taps_tt,
            *_WIDTHS,
            actual_start=actual_start_tt,
            predecessor_carry=predecessor_tt,
            program_config=program_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    row_major_us, expected = _traced_us(device, convolve(input_tt))
    tiled_us, outputs = _traced_us(device, convolve(projection_tt))
    logger.info(
        f"QKV conv start={actual_start} chunk={channel_chunk_size}: RM {row_major_us:.1f} us, tiled {tiled_us:.1f} us"
    )
    for name, reference, output in zip("qkv", expected, outputs):
        assert torch.equal(ttnn.to_torch(reference), ttnn.to_torch(output)), f"tiled {name} differs"


@pytest.mark.parametrize("rows", [(637, 638, 639), (29, 30, 31), (0, 17, 623)], ids=["tail", "first", "mixed"])
def test_kda_select_tile_rows_matches_embedding(device, rows) -> None:
    channels = sum(_WIDTHS)
    table = torch.randn(1, _ROWS, _PROJECTION_WIDTH).to(torch.bfloat16)
    tiled = ttnn.from_torch(table, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    indices = ttnn.from_torch(
        torch.tensor(rows, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    selected = ttnn.to_torch(ttnn.experimental.kda.select_tile_rows(tiled, indices, width=channels))
    assert torch.equal(selected, table[:, list(rows), :channels]), "selected rows differ from the table"
