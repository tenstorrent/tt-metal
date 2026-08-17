# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from ttnn.operations.toy_spec_mul import toy_spec_mul
from ttnn.operations.toy_spec_mul.toy_spec_mul_program_spec import TP_A, TP_B, TP_OUT, create_program_spec


@pytest.fixture(scope="module")
def device():
    ttnn.CONFIG.validate_program_args = True
    dev = ttnn.open_device(device_id=0)
    dev.enable_program_cache()
    yield dev
    ttnn.close_device(dev)


def _inputs(device, shape, seed):
    torch.manual_seed(seed)
    ta = torch.randn(*shape, dtype=torch.bfloat16)
    tb = torch.randn(*shape, dtype=torch.bfloat16)
    a = ttnn.from_torch(ta, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    b = ttnn.from_torch(tb, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    return ta, tb, a, b


def _check(ta, tb, out):
    got = ttnn.to_torch(out).float()
    expected = (ta * tb).float()
    assert torch.allclose(got, expected, atol=0.05), (got - expected).abs().max()


@pytest.mark.parametrize("shape", [(1, 1, 32, 32), (1, 1, 128, 128), (1, 2, 256, 64)])
def test_correctness(device, shape):
    ta, tb, a, b = _inputs(device, shape, seed=0)
    _check(ta, tb, toy_spec_mul(a, b))


def test_cache_hit_refreshes_tensor_addresses(device):
    """Second call must use freshly allocated tensors, not the addresses baked at cache miss."""
    shape = (1, 1, 96, 96)
    ta1, tb1, a1, b1 = _inputs(device, shape, seed=1)
    first = toy_spec_mul(a1, b1)
    _check(ta1, tb1, first)

    entries_after_miss = device.num_program_cache_entries()

    # Keep the first inputs alive so the second pair cannot land on the same addresses.
    ta2, tb2, a2, b2 = _inputs(device, shape, seed=2)
    second = toy_spec_mul(a2, b2)

    assert device.num_program_cache_entries() == entries_after_miss, "expected a cache hit, got a new entry"
    _check(ta2, tb2, second)
    _check(ta1, tb1, first)


def test_cache_hit_refreshes_scalar_runtime_args(device):
    """A narrower tile_limit on a cache hit must actually narrow the write.

    Both calls share a byte-identical ProgramSpec and differ only in runtime arg values.
    If the scalars froze at the cache miss, the second call would write every tile and the
    sentinel tail would be overwritten.
    """
    shape = (1, 1, 32, 256)  # 8 tiles along width, one tile row
    ta, tb, a, b = _inputs(device, shape, seed=7)

    _check(ta, tb, toy_spec_mul(a, b))
    entries = device.num_program_cache_entries()

    sentinel = torch.full(shape, -7.0, dtype=torch.bfloat16)
    out = ttnn.from_torch(sentinel, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    toy_spec_mul(a, b, out=out, tile_limit=4)

    assert device.num_program_cache_entries() == entries, "tile_limit must not change the spec"

    got = ttnn.to_torch(out).float()
    expected_head = (ta * tb).float()[..., :128]
    assert torch.allclose(got[..., :128], expected_head, atol=0.05)
    assert torch.equal(got[..., 128:], sentinel.float()[..., 128:]), "tail was written; scalars did not refresh"


def test_interleaved_shapes_do_not_contaminate(device):
    """Alternating specs must not leak one another's runtime args."""
    small = _inputs(device, (1, 1, 64, 64), seed=3)
    large = _inputs(device, (1, 1, 256, 128), seed=4)

    for _ in range(2):
        for ta, tb, a, b in (small, large):
            _check(ta, tb, toy_spec_mul(a, b))


def test_missing_tensor_arg_is_rejected(device, expect_error):
    ta, tb, a, b = _inputs(device, (1, 1, 32, 32), seed=5)
    out = ttnn.allocate_tensor_on_device(a.spec, a.device())
    spec, run_args = create_program_spec(a, b, out)

    with expect_error(RuntimeError, "has no entry in tensor_args"):
        ttnn.generic_op([a, b, out], spec, run_args, {TP_A: 0, TP_B: 1})


def test_spec_hash_is_structural(device):
    _, _, a, b = _inputs(device, (1, 1, 64, 64), seed=6)
    out = ttnn.allocate_tensor_on_device(a.spec, a.device())

    spec1, _ = create_program_spec(a, b, out)
    spec2, _ = create_program_spec(a, b, out)
    assert ttnn.compute_program_spec_hash(spec1) == ttnn.compute_program_spec_hash(spec2)

    spec3, _ = create_program_spec(a, b, out)
    spec3.dataflow_buffers[0].num_entries = 4
    assert ttnn.compute_program_spec_hash(spec3) != ttnn.compute_program_spec_hash(spec1)
