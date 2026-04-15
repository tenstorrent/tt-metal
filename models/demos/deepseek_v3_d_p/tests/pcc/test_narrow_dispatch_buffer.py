# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Minimal reproducer for ttnn.narrow crash on bfloat8_b TILE_LAYOUT dispatch buffers.

Root cause (narrow.cpp:78-79):
    uint32_t reduction_factor = input_tensor_shape[dim] / length;
    replicated_config.size /= reduction_factor;

When input_tensor_shape[dim] is NOT divisible by `length`, integer division
truncates reduction_factor, making the computed size too large. That size is
not a multiple of the bfloat8_b tile page size (1088 bytes), so Buffer::Buffer
aborts with: "buffer size should be divisible by the page size".

Affected aligned_count values (multiples of TILE_SIZE that don't divide 14336):
    14336 = 2^11 * 7, so exact divisors in [32..224]: 32, 64, 128, 224
    Non-exact (broken): 96, 160, 192

    aligned_count=96  -> token_count in [65..96]
    aligned_count=160 -> token_count in [129..160]
    aligned_count=192 -> token_count in [161..192]
"""

import pytest

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config

# These are the DeepSeek V3 dispatch buffer dimensions (single-device case):
#   experts_per_chip = 64, max_dispatched_tokens_per_expert = 224
#   per_device_shape = (64 * 224, 7168) = (14336, 7168)
DISPATCH_ROWS = 14336  # experts_per_chip * max_dispatched_tokens_per_expert
EMB_DIM = DeepSeekV3Config.EMB_SIZE  # 7168
TILE_SIZE = ttnn.TILE_SIZE  # 32


def aligned(token_count):
    return (token_count + TILE_SIZE - 1) // TILE_SIZE * TILE_SIZE


# fmt: off
@pytest.mark.parametrize("token_count, expect_crash", [
    # exact divisors of 14336 -> reduction_factor is exact -> OK
    (32,  False),   # aligned=32,  14336/32=448   exact
    (64,  False),   # aligned=64,  14336/64=224   exact
    (128, False),   # aligned=128, 14336/128=112  exact
    (224, False),   # aligned=224, 14336/224=64   exact
    # non-exact divisors -> integer-truncated reduction_factor -> CRASH
    (65,  True),    # aligned=96,  14336/96=149.3 truncated -> wrong size
    (96,  True),    # aligned=96  (same bucket)
    (129, True),    # aligned=160, 14336/160=89.6 truncated -> wrong size
    (160, True),    # aligned=160 (same bucket)
    (161, True),    # aligned=192, 14336/192=74.7 truncated -> wrong size
    (192, True),    # aligned=192 (same bucket)
])
# fmt: on
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [pytest.param(1, {"fabric_config": ttnn.FabricConfig.DISABLED}, id="single-1")],
    indirect=["mesh_device", "device_params"],
)
def test_narrow_dispatch_buffer(mesh_device, device_params, token_count, expect_crash):
    """
    Creates a (14336, 7168) bfloat8_b TILE_LAYOUT dispatch buffer and calls
    ttnn.narrow(buf, dim=0, start=0, length=aligned_count).

    Tests marked expect_crash=True reproduce the integer-division bug in
    narrow.cpp and should raise RuntimeError until the bug is fixed.
    """
    length = aligned(token_count)

    buf = ttnn.empty(
        (DISPATCH_ROWS, EMB_DIM),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
    )

    reduction_factor_exact = DISPATCH_ROWS / length
    reduction_factor_int = DISPATCH_ROWS // length
    original_size = (DISPATCH_ROWS // TILE_SIZE) * (EMB_DIM // TILE_SIZE) * 1088
    correct_size = (length // TILE_SIZE) * (EMB_DIM // TILE_SIZE) * 1088
    buggy_size = original_size // reduction_factor_int

    print(
        f"\ntoken_count={token_count}, aligned_count={length}"
        f"\n  reduction_factor (exact)  = {reduction_factor_exact:.4f}"
        f"\n  reduction_factor (int div) = {reduction_factor_int}"
        f"\n  original buffer size       = {original_size:,} bytes"
        f"\n  correct narrow size        = {correct_size:,} bytes  (% 1088 = {correct_size % 1088})"
        f"\n  buggy narrow size          = {buggy_size:,} bytes  (% 1088 = {buggy_size % 1088})"
    )

    if expect_crash:
        with pytest.raises(RuntimeError, match="buffer size should be divisible by the page size"):
            ttnn.narrow(buf, dim=0, start=0, length=length)
    else:
        result = ttnn.narrow(buf, dim=0, start=0, length=length)
        assert list(result.shape) == [length, EMB_DIM], f"Expected shape [{length}, {EMB_DIM}], got {list(result.shape)}"


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [pytest.param(1, {"fabric_config": ttnn.FabricConfig.DISABLED}, id="single-1")],
    indirect=["mesh_device", "device_params"],
)
def test_narrow_dispatch_buffer_length_96(mesh_device, device_params):
    """
    Reproducer for the integer-division bug in narrow.cpp.

    Shape: (14336, 7168) bfloat8_b TILE_LAYOUT — the DeepSeek V3 dispatch buffer
    narrow length: 96 (= ceil(65..96 tokens / 32) * 32)

    14336 / 96 = 149.33... -> integer division gives 149 (truncated)
    buggy size  = 109,182,976 / 149 = 732,771 bytes  (% 1088 = 547) -- CRASH
    correct size = (96/32) * (7168/32) * 1088 = 731,136 bytes        (% 1088 = 0)
    """
    buf = ttnn.empty(
        (14336, 7168),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
    )
    with pytest.raises(RuntimeError, match="buffer size should be divisible by the page size"):
        ttnn.narrow(buf, dim=0, start=0, length=96)
