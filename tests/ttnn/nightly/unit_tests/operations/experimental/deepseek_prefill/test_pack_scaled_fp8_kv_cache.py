# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole


def _to_host_bytes(tensor, mesh_device):
    composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    return ttnn.to_torch(tensor, mesh_composer=composer).contiguous().view(torch.uint8)


def _make_inputs(device, leading_shape):
    latent_bf16 = ttnn.from_torch(
        torch.randn(*leading_shape, 512, dtype=torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    latent, scales = ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8(latent_bf16)
    rope = ttnn.from_torch(
        torch.randn(*leading_shape, 64, dtype=torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return latent, scales, rope


def _expected_packed_bytes(latent, scales, rope, device, rows):
    return torch.cat(
        [
            _to_host_bytes(latent, device).reshape(rows, 512),
            _to_host_bytes(scales, device).reshape(rows, 16),
            _to_host_bytes(rope, device).reshape(rows, 128),
        ],
        dim=-1,
    )


@pytest.fixture(autouse=True)
def _require_blackhole():
    if not is_blackhole():
        pytest.skip("FP8_E4M3 path requires Blackhole")


@pytest.mark.parametrize("leading_shape", [(1, 1, 1), (2, 3, 7)])
def test_pack_scaled_fp8_kv_cache_preserves_mixed_format_bytes(device, leading_shape):
    torch.manual_seed(17)
    rows = math.prod(leading_shape)
    latent, scales, rope = _make_inputs(device, leading_shape)
    second_latent, second_scales, second_rope = _make_inputs(device, leading_shape)

    device.enable_program_cache()
    device.clear_program_cache()
    try:
        packed = ttnn.experimental.deepseek_prefill.pack_scaled_fp8_kv_cache(latent, scales, rope)
        cache_entries = device.num_program_cache_entries()

        second_packed = ttnn.experimental.deepseek_prefill.pack_scaled_fp8_kv_cache(
            second_latent, second_scales, second_rope
        )

        assert cache_entries > 0
        assert device.num_program_cache_entries() == cache_entries
        assert packed.dtype == ttnn.fp8_e4m3
        assert packed.layout == ttnn.ROW_MAJOR_LAYOUT
        assert tuple(packed.shape) == (*leading_shape, 656)
        assert torch.equal(
            _to_host_bytes(packed, device).reshape(rows, 656),
            _expected_packed_bytes(latent, scales, rope, device, rows),
        )
        assert torch.equal(
            _to_host_bytes(second_packed, device).reshape(rows, 656),
            _expected_packed_bytes(second_latent, second_scales, second_rope, device, rows),
        )
    finally:
        device.disable_and_clear_program_cache()
