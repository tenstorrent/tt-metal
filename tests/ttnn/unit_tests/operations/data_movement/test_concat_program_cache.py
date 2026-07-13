# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Program-cache regression tests for ttnn.concat (ConcatDeviceOperation).

Pins the cache-keying granularity so it can be verified BEFORE and AFTER
removing ConcatDeviceOperation::compute_program_hash. The custom hash keyed on
dim, groups, output_mem_config, sub_core_grids, factory.index(),
input_tensors.size() and each input/output TensorSpec (full shapes). The
framework default hashes all attributes plus the entire input_tensors vector
(count + each TensorSpec), so entry counts must be identical:

- Same config -> reuse (1 entry); different data alone must NOT re-key.
- Different dim / shape / dtype / number-of-inputs -> distinct entries.

Concat's custom hash was originally added for the weak-boost-hash collision
(PR #45144 / issue #47602); the default combiner is now splitmix64 plus a
collision-free canonical key, so the default is strictly stronger.
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


def run_concat(device, shapes, dim, dtype=ttnn.bfloat16, seed=0):
    """Concat a list of tensors inside the cache counter; return (torch_ref, tt_out)."""
    torch.manual_seed(seed)
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    torch_inputs = [torch.rand(s, dtype=torch_dtype) for s in shapes]
    torch_ref = torch.cat(torch_inputs, dim=dim)

    tt_inputs = [ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, dtype=dtype, device=device) for t in torch_inputs]
    with device.cache_entries_counter.measure():
        tt_out = ttnn.concat(tt_inputs, dim=dim)
    tt_out = ttnn.to_torch(tt_out)
    return torch_ref, tt_out


def test_concat_cache_reuse_same_config(device, isolate_program_cache):
    """Same shapes/dim/dtype twice with different data -> 1 entry, different outputs."""
    shapes = [(1, 1, 32, 64), (1, 1, 32, 64)]

    ref1, out1 = run_concat(device, shapes, dim=3, seed=0)
    assert_with_pcc(ref1, out1, 0.9999)
    ref2, out2 = run_concat(device, shapes, dim=3, seed=42)
    assert_with_pcc(ref2, out2, 0.9999)

    assert device.cache_entries_counter.total == 1
    assert not torch.equal(out1, out2)


def test_concat_cache_miss_different_dim(device, isolate_program_cache):
    """Different concat dim -> 2 entries."""
    shapes = [(1, 1, 32, 64), (1, 1, 32, 64)]

    ref1, out1 = run_concat(device, shapes, dim=3)
    assert_with_pcc(ref1, out1, 0.9999)
    ref2, out2 = run_concat(device, shapes, dim=2)
    assert_with_pcc(ref2, out2, 0.9999)

    assert device.cache_entries_counter.total == 2


def test_concat_cache_miss_different_shape(device, isolate_program_cache):
    """Same dim, different input shapes -> 2 entries."""
    ref1, out1 = run_concat(device, [(1, 1, 32, 64), (1, 1, 32, 64)], dim=3)
    assert_with_pcc(ref1, out1, 0.9999)
    ref2, out2 = run_concat(device, [(1, 1, 64, 64), (1, 1, 64, 64)], dim=3)
    assert_with_pcc(ref2, out2, 0.9999)

    assert device.cache_entries_counter.total == 2


def test_concat_cache_miss_different_dtype(device, isolate_program_cache):
    """Same shapes/dim, different dtype -> 2 entries."""
    shapes = [(1, 1, 32, 64), (1, 1, 32, 64)]

    ref1, out1 = run_concat(device, shapes, dim=3, dtype=ttnn.bfloat16)
    assert_with_pcc(ref1, out1, 0.9999)
    ref2, out2 = run_concat(device, shapes, dim=3, dtype=ttnn.float32)
    assert_with_pcc(ref2, out2, 0.9999)

    assert device.cache_entries_counter.total == 2


def test_concat_cache_miss_different_num_inputs(device, isolate_program_cache):
    """Different number of input tensors -> 2 entries (input_tensors vector length keyed)."""
    ref1, out1 = run_concat(device, [(1, 1, 32, 64), (1, 1, 32, 64)], dim=3)
    assert_with_pcc(ref1, out1, 0.9999)
    ref2, out2 = run_concat(device, [(1, 1, 32, 64), (1, 1, 32, 64), (1, 1, 32, 64)], dim=3)
    assert_with_pcc(ref2, out2, 0.9999)

    assert device.cache_entries_counter.total == 2
