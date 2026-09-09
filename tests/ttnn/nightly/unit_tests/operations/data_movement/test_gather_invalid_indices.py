# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""#55819: invalid indices must complete without corrupting valid gather results.

Run on reserved hardware under an external watchdog; a timed-out process must be
terminated and the device reset before another test is started. Invalid output
values are deliberately never compared, including indices into logical padding.
"""

import math

import pytest
import torch

import ttnn


_FORCE_CODEGEN = ttnn._ttnn.operations.data_movement.gather_force_codegen
_TILE_BYTES = 32 * 32 * 2


def _checked_gather(device, entry, x, index, dim, index_dtype, factory, cache_hit):
    xt = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    it = ttnn.from_torch(
        index.to(torch.int32),
        dtype=index_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    try:
        out = entry(xt, dim, it)
        ttnn.synchronize_device(device)
    finally:
        graph = ttnn.graph.end_graph_capture()
    factories = [
        node["params"]
        for node in graph
        if node.get("node_type") == "function_start"
        and "GatherCodegenProgramFactory" in node.get("params", {}).get("program_factory_type", "")
    ]
    assert len(factories) == 1, f"Expected one codegen gather, got {factories}"
    assert factories[0]["program_factory_type"].endswith(f"GatherCodegenProgramFactory{factory}")
    assert factories[0]["program_cache_hit"] is cache_hit, factories[0]
    actual = ttnn.to_torch(out)
    valid = index < x.shape[dim]
    # Replacing invalid indices is ONLY for the CPU oracle, never the device input.
    expected = torch.gather(x, dim, index.masked_fill(~valid, 0))
    assert actual.shape == expected.shape
    assert valid.any()
    assert torch.equal(actual[valid], expected[valid])
    # Keep allocations alive across all dispatches so the next call cannot reuse
    # their addresses and accidentally pass a stale cached-buffer binding.
    return xt, it, out


def _exercise(device, shape, index_shape, dim, index_dtype, factory, invalid_values, entry=_FORCE_CODEGEN):
    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    retained = []
    generator = torch.Generator().manual_seed(55819)
    for iteration in range(3):
        # Disjoint, exactly representable BF16 contents make stale reads observable.
        x = ((torch.arange(math.prod(shape)).reshape(shape) % 127) - 190 + 128 * iteration).to(torch.bfloat16)
        index = torch.randint(0, shape[dim], index_shape, generator=generator, dtype=torch.int64)
        if iteration < 2:
            # Spread witnesses over faces and tile rows, including the partial edge.
            positions = torch.linspace(0, index.numel() - 1, len(invalid_values), dtype=torch.int64)
            index.flatten()[positions] = torch.tensor(invalid_values, dtype=torch.int64).roll(iteration)
        retained.append(_checked_gather(device, entry, x, index, dim, index_dtype, factory, iteration > 0))


@pytest.mark.parametrize("index_dtype", [ttnn.uint16, ttnn.uint32], ids=["uint16", "uint32"])
@pytest.mark.parametrize(
    "shape,index_shape,factory",
    [
        ((64, 128), (64, 32), "Interleaved"),
        ((47, 95), (47, 19), "Interleaved"),
        ((64, 128), (64, 96), "Tiled"),
        ((47, 95), (47, 65), "Tiled"),
    ],
    ids=["row-full", "row-partial", "tiled-full", "tiled-partial"],
)
def test_gather_invalid_indices(device, shape, index_shape, factory, index_dtype):
    width = shape[-1]
    padded_width = math.ceil(width / 32) * 32
    invalid_values = [width, padded_width - 1, padded_width, padded_width + 31, 65535]
    if index_dtype == ttnn.uint32:
        invalid_values += [0x800000, 0x40000000, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF]
    _exercise(device, shape, index_shape, -1, index_dtype, factory, invalid_values)


def test_gather_invalid_indices_public_reproduction(device):
    # Exact public shape, dim and invalid value. Transposing dim 0 to the last
    # axis leaves Wt_index == 1, selecting the full-tile row-buffered reader.
    _exercise(device, (64, 128), (32, 128), 0, ttnn.uint32, "Interleaved", [0x800000], entry=ttnn.gather)


@pytest.mark.parametrize("index_dtype", [ttnn.uint16, ttnn.uint32], ids=["uint16", "uint32"])
def test_gather_invalid_indices_streaming_control(device, index_dtype):
    budget = ttnn.get_memory_view(device, ttnn.BufferType.L1).total_bytes_per_bank
    index_bytes = _TILE_BYTES if index_dtype == ttnn.uint16 else 2 * _TILE_BYTES
    wt_input = (budget - index_bytes - _TILE_BYTES) // _TILE_BYTES
    width = wt_input * 32
    assert width < 65535
    invalid_values = [width, width + 31, 65535]
    if index_dtype == ttnn.uint32:
        invalid_values += [0x800000, 0x40000000, 0xFFFFFFFF]
    _exercise(device, (64, width), (64, 512), -1, index_dtype, "Streaming", invalid_values)


def test_gather_wide_valid_uint32_indices(device):
    # Valid indices beyond UINT16 and the native row reader's uint16_t element
    # offset range must retain their full width in codegen's uint32_t offset.
    width = 131072
    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    x = ((torch.arange(width) // 1024) - 64).to(torch.bfloat16).reshape(1, width).repeat(32, 1)
    index = torch.tensor([0, 2047, 2048, 65535, 65536, width - 1], dtype=torch.int64).repeat(32, 6)[:, :32]
    _checked_gather(device, _FORCE_CODEGEN, x, index, -1, ttnn.uint32, "Streaming", False)
