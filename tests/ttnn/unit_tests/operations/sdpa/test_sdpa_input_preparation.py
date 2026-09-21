# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Independent rounding oracles for the explicit LOW_PRECISION input contract."""

import pytest
import torch
import ttnn

from models.common.utility_functions import is_blackhole

pytestmark = pytest.mark.parametrize("device_params", [{"trace_region_size": 1048576}], indirect=True)


def round_significand(values, bits):
    values = values.double()
    exponent = torch.frexp(values.abs())[1] - bits
    step = torch.ldexp(torch.ones_like(values), exponent)
    return ((values / step).round() * step).float()


def round_block(values, bits, *, ties_even):
    groups = values.double().reshape(-1, 16)
    maximum = groups.abs().amax(-1, keepdim=True)
    exponent = torch.frexp(maximum)[1] - bits
    step = torch.ldexp(torch.ones_like(maximum), exponent)
    scaled = groups.abs() / step
    integer = scaled.round() if ties_even else (scaled + 0.5).floor()
    return (integer.clamp_max(2**bits - 1) * step * groups.sign()).float().reshape(values.shape)


@pytest.mark.parametrize(
    "is_query,dtype", [(True, ttnn.bfloat16), (False, ttnn.bfloat16), (False, ttnn.bfloat8_b), (False, ttnn.bfloat4_b)]
)
@pytest.mark.parametrize("distribution", ["normal", "ties", "zeros"])
def test_sdpa_input_preparation(device, is_query, dtype, distribution):
    if not is_blackhole():
        pytest.skip("SDPA preparation initially targets Blackhole")
    device.enable_program_cache()
    shape = (1, 1, 256, 128)
    if distribution == "normal":
        host = torch.randn(shape, generator=torch.Generator().manual_seed(12)).bfloat16()
    elif distribution == "zeros":
        host = torch.zeros(shape, dtype=torch.bfloat16)
    else:
        # Every BF16 mantissa, both signs, varying exponents and 16-value group
        # anchors. This covers ties and saturation without relying on a cast.
        index = torch.arange(256 * 128).reshape(-1, 16)
        values = ((index * 17) % 256).float() / 128
        values[:, 0] = 1.75
        values = torch.ldexp(values, (index[:, :1] // 16 % 33 - 16).int())
        values *= torch.where(index % 2 == 0, 1.0, -1.0)
        host = values.reshape(shape).bfloat16()
    if dtype == ttnn.bfloat4_b:
        expected = round_block(host, 3, ties_even=True)
    else:
        expected = round_significand(host, 7 if is_query else 5)
        if dtype == ttnn.bfloat8_b:
            # Native BFP8 packing rounds shared-exponent ties away from zero;
            # the preceding RNE5 values are exact at its E8M6 ingress.
            expected = round_block(expected, 7, ties_even=False)
    source = ttnn.from_torch(host, device=device, layout=ttnn.TILE_LAYOUT)
    output = ttnn.transformer.prepare_sdpa_input(source, is_query=is_query, dtype=dtype)
    assert output.dtype == dtype
    assert torch.equal(ttnn.to_torch(output).float(), expected)
    assert torch.equal(ttnn.to_torch(source), host)
    cache_entries = device.num_program_cache_entries()
    # Keep source/output alive while exercising fresh addresses on a cache hit.
    second = ttnn.from_torch(-host, device=device, layout=ttnn.TILE_LAYOUT)
    second_output = ttnn.transformer.prepare_sdpa_input(second, is_query=is_query, dtype=dtype)
    assert source.buffer_address() != second.buffer_address()
    assert device.num_program_cache_entries() == cache_entries
    assert torch.equal(ttnn.to_torch(second_output).float(), -expected)
    trace = ttnn.begin_trace_capture(device, cq_id=0)
    traced = ttnn.transformer.prepare_sdpa_input(source, is_query=is_query, dtype=dtype)
    ttnn.end_trace_capture(device, trace, cq_id=0)
    try:
        for _ in range(2):
            ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
            assert torch.equal(ttnn.to_torch(traced).float(), expected)
    finally:
        ttnn.release_trace(device, trace)


@pytest.mark.parametrize("invalid", ["dtype", "placement", "rank", "padding", "query_storage", "dimension"])
def test_sdpa_input_preparation_rejects_unsupported(device, invalid):
    if not is_blackhole():
        pytest.skip("SDPA preparation initially targets Blackhole")
    shape = (1, 1, 256, 64 if invalid == "dimension" else 128)
    if invalid == "rank":
        shape = shape[1:]
    elif invalid == "padding":
        shape = (1, 1, 255, 128)
    source = ttnn.from_torch(
        torch.zeros(shape, dtype=torch.bfloat16),
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b if invalid == "dtype" else ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG if invalid == "placement" else ttnn.DRAM_MEMORY_CONFIG,
    )
    before = device.num_program_cache_entries()
    with pytest.raises(RuntimeError, match="SDPA|Prepared Q"):
        ttnn.transformer.prepare_sdpa_input(
            source, is_query=True, dtype=ttnn.bfloat4_b if invalid == "query_storage" else ttnn.bfloat16
        )
    assert device.num_program_cache_entries() == before
