# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Direct contract coverage for experimental KDA QKV causal Conv1D plus SiLU."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
import torch.nn.functional as F

import ttnn
from models.common.utility_functions import run_for_blackhole
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import assert_accurate, assert_bit_identical

pytestmark = [
    run_for_blackhole(),
    pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True),
]

_SEQUENCE = 64
_DEFAULT_WIDTHS = (512, 512, 512)


def _host_inputs(
    *,
    sequence: int = _SEQUENCE,
    widths: tuple[int, int, int] = _DEFAULT_WIDTHS,
    batch: int = 1,
    history_rows: int = 3,
    seed: int = 223,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
    generator = torch.Generator().manual_seed(seed)
    channels = sum(widths)
    inputs = torch.randn(batch, sequence, channels, generator=generator, dtype=torch.bfloat16)
    history = torch.randn(batch, history_rows, channels, generator=generator, dtype=torch.bfloat16)
    taps = tuple(torch.randn(1, 1, channels, generator=generator, dtype=torch.bfloat16) for _ in range(4))
    return inputs, history, taps


def _to_device(
    tensor: torch.Tensor,
    device: ttnn.Device,
    *,
    dtype: ttnn.DataType = ttnn.bfloat16,
    layout: ttnn.Layout,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    return ttnn.from_torch(tensor, dtype=dtype, layout=layout, device=device, memory_config=memory_config)


def _device_inputs(
    device: ttnn.Device,
    *,
    sequence: int = _SEQUENCE,
    widths: tuple[int, int, int] = _DEFAULT_WIDTHS,
    batch: int = 1,
    history_rows: int = 3,
    seed: int = 223,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]],
    tuple[ttnn.Tensor, ttnn.Tensor, tuple[ttnn.Tensor, ...]],
]:
    host = _host_inputs(
        sequence=sequence,
        widths=widths,
        batch=batch,
        history_rows=history_rows,
        seed=seed,
    )
    inputs, history, taps = host
    return host, (
        _to_device(inputs, device, layout=ttnn.ROW_MAJOR_LAYOUT),
        _to_device(history, device, layout=ttnn.ROW_MAJOR_LAYOUT),
        tuple(_to_device(tap, device, layout=ttnn.TILE_LAYOUT) for tap in taps),
    )


def _reference(
    inputs: torch.Tensor,
    history: torch.Tensor,
    taps: tuple[torch.Tensor, ...],
    widths: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    window = torch.cat((history, inputs), dim=1)
    convolved = sum(window[:, tap : tap + inputs.shape[1]] * taps[tap] for tap in range(4))
    return F.silu(convolved).split(widths, dim=-1)


def _run(
    input_tt: ttnn.Tensor,
    history_tt: ttnn.Tensor,
    taps_tt: tuple[ttnn.Tensor, ...],
    *,
    widths: tuple[int, int, int] = _DEFAULT_WIDTHS,
    memory_config: ttnn.MemoryConfig | None = None,
) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
    return ttnn.experimental.kda.qkv_causal_conv1d_silu(
        input_tt,
        history_tt,
        *taps_tt,
        *widths,
        memory_config=memory_config,
    )


@pytest.mark.parametrize("widths", [(512, 512, 512), (1024, 1024, 1024), (512, 256, 128)])
def test_qkv_causal_conv1d_silu_contract(device: ttnn.Device, widths: tuple[int, int, int]) -> None:
    """Cover one/multiple channel blocks, split widths, tap order, and runtime gates."""
    host, device_inputs = _device_inputs(device, widths=widths)
    inputs, history, taps = host
    input_tt, history_tt, taps_tt = device_inputs
    expected = _reference(inputs, history, taps, widths)
    input_tensors = (input_tt, history_tt, *taps_tt)
    snapshots = tuple(ttnn.to_torch(tensor).clone() for tensor in input_tensors)

    def run(
        current_input: ttnn.Tensor = input_tt,
        current_history: ttnn.Tensor = history_tt,
        current_taps: tuple[ttnn.Tensor, ...] = taps_tt,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        with ttnn.manage_config("throw_exception_on_fallback", True):
            return _run(current_input, current_history, current_taps, widths=widths)

    outputs = run()
    for name, output, width in zip(("q", "k", "v"), outputs, widths, strict=True):
        assert output.dtype == ttnn.bfloat16
        assert output.layout == ttnn.TILE_LAYOUT
        assert output.memory_config() == ttnn.DRAM_MEMORY_CONFIG
        assert tuple(ttnn.to_torch(output).shape) == (1, _SEQUENCE, width)
        assert all(output.buffer_address() != tensor.buffer_address() for tensor in input_tensors)

    cache_entries = device.num_program_cache_entries()
    repeated_host, repeated_device = _device_inputs(device, widths=widths, seed=224)
    repeated_expected = _reference(*repeated_host, widths)
    repeated_outputs = run(*repeated_device)
    ttnn.synchronize_device(device)
    assert device.num_program_cache_entries() == cache_entries

    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    traced_outputs = run()
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    for _ in range(2):
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)

    for name, golden, repeated_golden, output, repeated, traced in zip(
        ("q", "k", "v"), expected, repeated_expected, outputs, repeated_outputs, traced_outputs, strict=True
    ):
        actual = ttnn.to_torch(output)
        assert_accurate(golden, actual, name=name, pcc_threshold=0.999)
        assert_accurate(repeated_golden, ttnn.to_torch(repeated), name=f"{name} cache hit", pcc_threshold=0.999)
        assert_bit_identical(actual, ttnn.to_torch(traced), name=f"{name} trace replay")

    for name, before, tensor in zip(
        ("input", "history", "tap0", "tap1", "tap2", "tap3"), snapshots, input_tensors, strict=True
    ):
        assert_bit_identical(before, ttnn.to_torch(tensor), name=f"{name} immutability")

    ttnn.release_trace(device, trace_id)


def test_qkv_causal_conv1d_silu_program_key_includes_split_widths(device: ttnn.Device) -> None:
    _, (input_tt, history_tt, taps_tt) = _device_inputs(device, widths=(128, 128, 128), sequence=32, seed=772)
    _run(input_tt, history_tt, taps_tt, widths=(128, 128, 128))
    entries = device.num_program_cache_entries()
    _run(input_tt, history_tt, taps_tt, widths=(64, 128, 192))
    assert device.num_program_cache_entries() == entries + 1
    _run(input_tt, history_tt, taps_tt, widths=(64, 128, 192))
    assert device.num_program_cache_entries() == entries + 1


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("host_input", "allocated device tensor"),
        ("batch", r"input must be \[1,T,Q\+K\+V\]"),
        ("history_shape", r"history must be \[1,3,Q\+K\+V\]"),
        ("tap_last_dimension", r"tap2 last dimension must equal Q\+K\+V"),
        ("tap_volume", "tap2 logical volume must equal"),
        ("input_layout", "input has unsupported layout"),
        ("history_layout", "history has unsupported layout"),
        ("tap_layout", "tap1 has unsupported layout"),
        ("input_dtype", "input must be BFLOAT16"),
        ("history_dtype", "history must be BFLOAT16"),
        ("tap_dtype", "tap3 must be BFLOAT16"),
        ("sequence_alignment", "sequence must be positive and tile aligned"),
        ("sharded_history", "history must use interleaved memory"),
    ],
)
def test_qkv_causal_conv1d_silu_rejects_invalid_tensors(
    device: ttnn.Device, expect_error: Callable, case: str, message: str
) -> None:
    batch = 2 if case == "batch" else 1
    history_rows = 2 if case == "history_shape" else 3
    sequence = 33 if case == "sequence_alignment" else 32
    host, device_inputs = _device_inputs(
        device,
        widths=(128, 128, 128),
        batch=batch,
        history_rows=history_rows,
        sequence=sequence,
        seed=991,
    )
    inputs, history, taps = host
    input_tt, history_tt, taps_tt = device_inputs
    taps_list = list(taps_tt)

    if case == "host_input":
        input_tt = ttnn.from_torch(inputs, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    elif case == "tap_last_dimension":
        taps_list[2] = _to_device(taps[2].reshape(-1, 1), device, layout=ttnn.TILE_LAYOUT)
    elif case == "tap_volume":
        taps_list[2] = _to_device(torch.cat((taps[2], taps[2]), dim=0), device, layout=ttnn.TILE_LAYOUT)
    elif case == "input_layout":
        input_tt = _to_device(inputs, device, layout=ttnn.TILE_LAYOUT)
    elif case == "history_layout":
        history_tt = _to_device(history, device, layout=ttnn.TILE_LAYOUT)
    elif case == "tap_layout":
        taps_list[1] = _to_device(taps[1], device, layout=ttnn.ROW_MAJOR_LAYOUT)
    elif case == "input_dtype":
        input_tt = _to_device(inputs.float(), device, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)
    elif case == "history_dtype":
        history_tt = _to_device(history.float(), device, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)
    elif case == "tap_dtype":
        taps_list[3] = _to_device(taps[3].float(), device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    elif case == "sharded_history":
        shard_spec = ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
            [history.shape[0] * history.shape[1], history.shape[2]],
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        sharded_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
        history_tt = _to_device(history, device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=sharded_config)

    with expect_error(RuntimeError, message):
        _run(input_tt, history_tt, tuple(taps_list), widths=(128, 128, 128))


@pytest.mark.parametrize(
    ("widths", "message"),
    [
        ((0, 128, 256), "Q/K/V widths must be positive"),
        ((100, 128, 156), "Q/K/V widths must be tile aligned"),
        ((128, 128, 64), r"input must be \[1,T,Q\+K\+V\]"),
    ],
)
def test_qkv_causal_conv1d_silu_rejects_invalid_widths(
    device: ttnn.Device, expect_error: Callable, widths: tuple[int, int, int], message: str
) -> None:
    _, (input_tt, history_tt, taps_tt) = _device_inputs(device, widths=(128, 128, 128), sequence=32)
    with expect_error(RuntimeError, message):
        _run(input_tt, history_tt, taps_tt, widths=widths)


def test_qkv_causal_conv1d_silu_rejects_sharded_output(device: ttnn.Device, expect_error: Callable) -> None:
    _, (input_tt, history_tt, taps_tt) = _device_inputs(device, widths=(128, 128, 128), sequence=32)
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        [32, 128],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    sharded_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    with expect_error(RuntimeError, "output memory configuration must be interleaved"):
        _run(input_tt, history_tt, taps_tt, widths=(128, 128, 128), memory_config=sharded_config)
