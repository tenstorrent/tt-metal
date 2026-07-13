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

The default hash keys on the same distinctions the old custom hash did (whole
MatmulParams struct + each input/optional tensor's TensorSpec = logical shape)
minus the redundant factory.index() (a pure function of the hashed
attributes+tensors), so cache-entry counts are unchanged:

- Same config -> reuse (1 entry); different data alone must NOT re-key.
- Different shape (N) / dtype -> distinct entries.
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

    assert device.cache_entries_counter.total == 1
    assert not torch.equal(out1, out2)


def test_matmul_cache_miss_different_shape(device, isolate_program_cache):
    """Different N -> 2 entries."""
    ref1, out1 = run_matmul(device, 128, 256, 192)
    assert_with_pcc(ref1, out1, 0.99)
    ref2, out2 = run_matmul(device, 128, 256, 256)
    assert_with_pcc(ref2, out2, 0.99)

    assert device.cache_entries_counter.total == 2


def test_matmul_cache_miss_different_dtype(device, isolate_program_cache):
    """Different dtype -> 2 entries."""
    ref1, out1 = run_matmul(device, 128, 256, 192, dtype=ttnn.bfloat16)
    assert_with_pcc(ref1, out1, 0.99)
    ref2, out2 = run_matmul(device, 128, 256, 192, dtype=ttnn.float32)
    assert_with_pcc(ref2, out2, 0.99)

    assert device.cache_entries_counter.total == 2
