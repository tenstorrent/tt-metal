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
