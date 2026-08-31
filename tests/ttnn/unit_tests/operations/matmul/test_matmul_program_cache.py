# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Program-cache regression tests for ttnn.matmul (MatmulDeviceOperation).

MatmulDeviceOperation's custom program hash was renamed to
compute_descriptor_program_hash so the device-operation framework no longer
detects it and instead keys the program cache on the default hash (all
attributes + tensor_args). The renamed helper is retained only for the
experimental descriptor interface via the pybind name "compute_program_hash".

For the standard two-input matmul path the default hash keys on exactly the
same distinctions the old custom hash did (whole MatmulParams struct + each
input/optional tensor's TensorSpec = logical shape) minus the redundant
factory.index() (a pure function of the hashed attributes+tensors), so
cache-entry counts on this path are unchanged:

- Same config -> reuse (1 entry); different data alone must NOT re-key.
- Different shape (N) / dtype -> distinct entries.

The default hash is, however, strictly MORE precise on the multi-weight
`matmul_batched_weights` path (one activation + N weights, i.e.
input_tensors = [a, w0, .., w_{N-1}]). The old custom hash only hashed
input_tensors.at(0) and .at(1) (matmul_device_operation.cpp:2549-2564), so it
ignored N and every weight beyond the first, risking a stale-program cache hit
across different compiled batch counts. The default hash keys on the entire
tensor_args (the whole input_tensors vector), so N -- baked into the kernel
compile args -- is now keyed by construction. That path is a DRAM-prefetcher
matmul requiring a global circular buffer + sub-device manager + DRAM
width-sharded weights + HW-specific core topology, so it is not exercised in
this lightweight file (no existing harness); the distinct-N guarantee is
structural (default hash traverses the full input_tensors vector).
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.fixture
def isolate_program_cache(device):
    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    yield
    device.disable_and_clear_program_cache()


def run_matmul(device, m, k, n, dtype=ttnn.bfloat16, seed=0):
    """ttnn.matmul([1,1,m,k]@[1,1,k,n]) inside the cache counter; return (torch_ref, tt_out).

    Golden is computed in float32; inputs are cast to `dtype` for the device.
    """
    torch.manual_seed(seed)
    a = torch.randn((1, 1, m, k), dtype=torch.float32)
    b = torch.randn((1, 1, k, n), dtype=torch.float32)
    torch_ref = torch.matmul(a, b)

    tt_a = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, dtype=dtype, device=device)
    tt_b = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, dtype=dtype, device=device)
    with device.cache_entries_counter.measure():
        tt_out = ttnn.matmul(tt_a, tt_b)
    tt_out = ttnn.to_torch(tt_out).to(torch.float32)
    return torch_ref, tt_out


def test_matmul_cache_reuse_same_config(device, isolate_program_cache):
    """Same shapes/dtype twice with different data -> 1 entry, different outputs."""
    ref1, out1 = run_matmul(device, 128, 256, 192, seed=0)
    assert_with_pcc(ref1, out1, 0.99)
    ref2, out2 = run_matmul(device, 128, 256, 192, seed=42)
    assert_with_pcc(ref2, out2, 0.99)

    count = device.cache_entries_counter.total
    assert count == 1, f"same config twice must reuse 1 cache entry, got {count} (cache-key regression)"
    assert not torch.equal(
        out1, out2
    ), "different input data (seed 0 vs 42) must yield different outputs; equal outputs mean a stale cached result was reused"


def test_matmul_cache_miss_different_shape(device, isolate_program_cache):
    """Different N -> 2 entries."""
    ref1, out1 = run_matmul(device, 128, 256, 192)
    assert_with_pcc(ref1, out1, 0.99)
    ref2, out2 = run_matmul(device, 128, 256, 256)
    assert_with_pcc(ref2, out2, 0.99)

    count = device.cache_entries_counter.total
    assert count == 2, f"different N (192 vs 256) must produce 2 distinct cache entries, got {count} (shape not keyed)"


def test_matmul_cache_miss_different_dtype(device, isolate_program_cache):
    """Different dtype -> 2 entries."""
    ref1, out1 = run_matmul(device, 128, 256, 192, dtype=ttnn.bfloat16)
    assert_with_pcc(ref1, out1, 0.99)
    ref2, out2 = run_matmul(device, 128, 256, 192, dtype=ttnn.float32)
    assert_with_pcc(ref2, out2, 0.99)

    count = device.cache_entries_counter.total
    assert (
        count == 2
    ), f"different dtype (bfloat16 vs float32) must produce 2 distinct cache entries, got {count} (dtype not keyed)"


# ---------------------------------------------------------------------------
# Cache-hit address patching (smuggled-pointer regression net)
#
# Every buffer address a matmul descriptor factory puts in a runtime arg is
# declared as a tensor binding (KernelDescriptor::emplace_runtime_args); the arg
# vectors hold a 0 placeholder in that slot. The framework patches bindings on a
# program-cache hit, so a slot that lost its binding would keep the placeholder
# (address 0) or a stale address from the cache-miss call. These tests force the
# second call's tensors to land at DIFFERENT addresses than the first, so a
# missing binding shows up as wrong data instead of passing by luck.
# ---------------------------------------------------------------------------

L1_WIDTH_SHARDED_3_CORES = ttnn.MemoryConfig(
    memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    buffer_type=ttnn.BufferType.L1,
    shard_spec=ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))}),
        (32, 32),
        ttnn.ShardOrientation.ROW_MAJOR,
    ),
)

DRAM_WIDTH_SHARDED_3_CORES = ttnn.MemoryConfig(
    memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    buffer_type=ttnn.BufferType.DRAM,
    shard_spec=ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))}),
        (96, 32),
        ttnn.ShardOrientation.ROW_MAJOR,
    ),
)

ADDRESS_PATCH_CASES = [
    # (m, k, n, program_config, has_bias, in0_mem, in1_mem, out_mem)
    (64, 32, 64, None, True, None, None, None),
    (
        32,
        32,
        64,
        ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(2, 1),
            in0_block_w=1,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=1,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        ),
        True,
        None,
        None,
        None,
    ),
    (
        64,
        32,
        32,
        ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(2, 1),
            in0_block_w=1,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=1,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        ),
        True,
        None,
        None,
        None,
    ),
    (
        64,
        32,
        64,
        ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(2, 2),
            in0_block_w=1,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=1,
            transpose_mcast=False,
            fused_activation=None,
        ),
        True,
        None,
        None,
        None,
    ),
    (
        32,
        96,
        32,
        ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=1,
            per_core_M=1,
            per_core_N=1,
            fused_activation=None,
        ),
        False,
        L1_WIDTH_SHARDED_3_CORES,
        DRAM_WIDTH_SHARDED_3_CORES,
        ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            buffer_type=ttnn.BufferType.L1,
        ),
    ),
]

ADDRESS_PATCH_IDS = ["default", "mcast_1d_in0", "mcast_1d_in1", "mcast_2d", "dram_sharded"]


@pytest.mark.parametrize(
    "m, k, n, program_config, has_bias, in0_mem, in1_mem, out_mem",
    ADDRESS_PATCH_CASES,
    ids=ADDRESS_PATCH_IDS,
)
def test_matmul_cache_hit_patches_addresses(
    device, isolate_program_cache, m, k, n, program_config, has_bias, in0_mem, in1_mem, out_mem
):
    """Second (cache-hit) call with freshly-allocated tensors at new addresses must still be correct."""
    torch.manual_seed(0)

    def make_operands(seed):
        torch.manual_seed(seed)
        a = torch.randn((1, 1, m, k), dtype=torch.float32)
        b = torch.randn((1, 1, k, n), dtype=torch.float32)
        bias = torch.randn((1, 1, 1, n), dtype=torch.float32) if has_bias else None
        ref = torch.matmul(a, b)
        if has_bias:
            ref = ref + bias
        tt_a = ttnn.from_torch(
            a,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=in0_mem or ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_b = ttnn.from_torch(
            b,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=in1_mem or ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_bias = (
            ttnn.from_torch(
                bias,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            if has_bias
            else None
        )
        return ref, tt_a, tt_b, tt_bias

    def run(tt_a, tt_b, tt_bias):
        kwargs = {}
        if program_config is not None:
            kwargs["program_config"] = program_config
        if out_mem is not None:
            kwargs["memory_config"] = out_mem
        if tt_bias is not None:
            return ttnn.linear(tt_a, tt_b, bias=tt_bias, **kwargs)
        return ttnn.matmul(tt_a, tt_b, **kwargs)

    # Only the matmul calls are measured; from_torch/to_torch may run device ops of their own.
    # The first operand set stays alive so the second set is forced to different addresses.
    ref1, a1, b1, bias1 = make_operands(0)
    with device.cache_entries_counter.measure():
        out1 = run(a1, b1, bias1)
    torch_out1 = ttnn.to_torch(out1).to(torch.float32)

    ref2, a2, b2, bias2 = make_operands(42)
    moved = [a1.buffer_address() != a2.buffer_address(), b1.buffer_address() != b2.buffer_address()]
    if has_bias:
        moved.append(bias1.buffer_address() != bias2.buffer_address())
    assert all(moved), f"operands did not move; the test cannot detect a stale address (moved={moved})"

    with device.cache_entries_counter.measure():
        out2 = run(a2, b2, bias2)
    assert out1.buffer_address() != out2.buffer_address(), "output did not move"
    torch_out2 = ttnn.to_torch(out2).to(torch.float32)

    assert_with_pcc(ref1, torch_out1, 0.99)
    assert_with_pcc(ref2, torch_out2, 0.99)

    count = device.cache_entries_counter.total
    assert count == 1, f"identical config twice must reuse 1 cache entry, got {count} (the hit path was not exercised)"
