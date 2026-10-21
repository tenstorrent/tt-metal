# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


@pytest.mark.parametrize("scalar_rhs", [False, True])
@pytest.mark.parametrize("op_name", ["add", "lt"])
@pytest.mark.parametrize("preallocated_second", [False, True])
def test_binary_ng_preallocated_output_memory_cache(device, scalar_rhs, op_name, preallocated_second):
    """A preallocated output's memory type changes the compiled writer."""
    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    x = torch.ones((32, 32), dtype=torch.bfloat16)
    rhs = torch.full_like(x, 2)
    a = ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT)
    b = 2.0 if scalar_rhs else ttnn.from_torch(rhs, device=device, layout=ttnn.TILE_LAYOUT)
    dtype = ttnn.bfloat16 if op_name == "add" else ttnn.uint32
    host_dtype = torch.bfloat16 if op_name == "add" else torch.int32
    initial = torch.full((32, 32), 7, dtype=host_dtype)
    out_dram = ttnn.from_torch(
        initial, device=device, layout=ttnn.TILE_LAYOUT, dtype=dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out_l1 = ttnn.from_torch(
        initial, device=device, layout=ttnn.TILE_LAYOUT, dtype=dtype, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    op = getattr(ttnn, op_name)
    golden = (x + rhs) if op_name == "add" else (x < rhs).to(host_dtype)
    op(a, b, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=dtype, output_tensor=out_dram)
    cold = ttnn.to_torch(out_dram).to(host_dtype)
    kwargs = {"output_tensor": out_l1} if preallocated_second else {}
    cached_out = op(a, b, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=dtype, **kwargs)
    cached = ttnn.to_torch(cached_out).to(host_dtype)
    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    fresh_out = op(a, b, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=dtype, **kwargs)
    fresh = ttnn.to_torch(fresh_out).to(host_dtype)
    assert torch.equal(cold, golden)
    assert torch.equal(fresh, golden)
    assert torch.equal(cached, fresh)


@pytest.mark.parametrize("scalar_rhs", [False, True])
@pytest.mark.parametrize("first_preallocated", [False, True])
@pytest.mark.parametrize("output_in_l1", [False, True])
def test_binary_ng_equivalent_output_memory_cache(device, scalar_rhs, first_preallocated, output_in_l1):
    """Equivalent allocated and preallocated outputs share one program and return exact values."""
    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    memory_config = ttnn.L1_MEMORY_CONFIG if output_in_l1 else ttnn.DRAM_MEMORY_CONFIG
    other_memory_config = ttnn.DRAM_MEMORY_CONFIG if output_in_l1 else ttnn.L1_MEMORY_CONFIG
    keep_alive = []
    for index, preallocated in enumerate([first_preallocated, not first_preallocated] * 2):
        host = torch.full((32, 32), index + 1, dtype=torch.bfloat16)
        a = ttnn.from_torch(host, device=device, layout=ttnn.TILE_LAYOUT)
        b = 2.0 if scalar_rhs else ttnn.from_torch(torch.full_like(host, 2), device=device, layout=ttnn.TILE_LAYOUT)
        output = (
            ttnn.from_torch(torch.zeros_like(host), device=device, layout=ttnn.TILE_LAYOUT, memory_config=memory_config)
            if preallocated
            else None
        )
        with device.cache_entries_counter.measure():
            result = ttnn.add(
                a,
                b,
                memory_config=other_memory_config if preallocated else memory_config,
                output_tensor=output,
            )
        if preallocated:
            assert result.buffer_address() == output.buffer_address()
        assert result.memory_config() == memory_config
        assert torch.equal(ttnn.to_torch(result), host + 2)
        keep_alive.extend([a, b, output, result])
    assert device.cache_entries_counter.total == 1
