# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Cache-correctness sweep for ops migrated to the descriptor framework (#46506).

For each migrated op we run its SIMPLEST existing test invocation (copied verbatim from the
op's own test file — not invented), but under an ENABLED program cache and across the value
the op varies per call (scalar / data / shape). Two things must hold:

  * not stale  : every call's result matches the op's golden, even on a program-cache HIT
                 (a frozen per-call rt-arg would make a later call wrong).
  * not over-caching : the op does not create a new program-cache entry for a value it should
                 re-apply (e.g. a scalar) — that is the "cache too restrictive" failure that
                 silently rebuilds/recompiles every call.

Ops that FAIL either check are descriptor fix candidates.
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_equal


@pytest.fixture(scope="module")
def cache_device():
    dev = ttnn.open_device(device_id=0)
    dev.enable_program_cache()
    yield dev
    ttnn.close_device(dev)


# ---------------------------------------------------------------------------
# eltwise / binary_ng — tensor-scalar. Copied from test_relational.py
# (test_binary_relational_scalar_ttnn): int32 input, fn(tensor, scalar), real golden.
# Varying the scalar must NOT add cache entries and must NOT go stale.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ttnn_function", [ttnn.gt, ttnn.lt, ttnn.eq, ttnn.ne, ttnn.ge, ttnn.le])
def test_binary_scalar_relational_cache(cache_device, ttnn_function):
    torch.manual_seed(0)
    shape = (1, 1, 32, 32)
    in_data = torch.randint(-100, 100, shape, dtype=torch.int32)
    input_tensor = ttnn.from_torch(in_data, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=cache_device)
    golden_function = ttnn.get_golden_function(ttnn_function)

    def _run(scalar):
        out = ttnn.to_torch(ttnn_function(input_tensor, scalar))
        assert torch.equal(golden_function(in_data, scalar), out), f"{ttnn_function} stale at scalar={scalar}"

    # WARM-UP then assert ZERO growth: re-applying a scalar must not mint a new entry per call.
    _run(-2)
    base = cache_device.num_program_cache_entries()
    for scalar in [-1, 0, 1, 2]:
        _run(scalar)
    assert (
        cache_device.num_program_cache_entries() == base
    ), f"{ttnn_function}: cache grew past {base} across scalars (over-caching)"


# ---------------------------------------------------------------------------
# eltwise / binary_ng — tensor-scalar arithmetic. Copied from the same get_golden_function
# pattern (ttnn.add/mul/sub/rsub with a python scalar), bf16 input.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ttnn_function", [ttnn.add, ttnn.mul, ttnn.sub])
def test_binary_scalar_arith_cache(cache_device, ttnn_function):
    torch.manual_seed(0)
    shape = (1, 1, 32, 32)
    in_data = torch.rand(shape, dtype=torch.bfloat16) * 4 - 2
    input_tensor = ttnn.from_torch(in_data, layout=ttnn.TILE_LAYOUT, device=cache_device)
    golden_function = ttnn.get_golden_function(ttnn_function)

    def _run(scalar):
        out = ttnn.to_torch(ttnn_function(input_tensor, scalar)).float()
        ref = golden_function(in_data.float(), scalar).float()
        assert torch.allclose(out, ref, atol=0.05, rtol=0.05), f"{ttnn_function} stale at scalar={scalar}"

    # WARM-UP then assert ZERO growth: re-applying a scalar must not mint a new entry per call.
    _run(1.0)
    base = cache_device.num_program_cache_entries()
    for scalar in [2.0, 3.0, 0.5]:
        _run(scalar)
    assert (
        cache_device.num_program_cache_entries() == base
    ), f"{ttnn_function}: cache grew past {base} across scalars (over-caching)"


# ---------------------------------------------------------------------------
# eltwise / binary_ng — tensor-tensor. Copied from test_relational.py run_relational_test.
# Varying input DATA (fresh alloc → fresh address) must not go stale; one entry.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ttnn_function", [ttnn.add, ttnn.mul, ttnn.gt])
def test_binary_tensor_cache(cache_device, ttnn_function):
    torch.manual_seed(0)
    shape = (1, 1, 64, 128)
    golden_function = ttnn.get_golden_function(ttnn_function)

    def _run():
        a = torch.rand(shape, dtype=torch.bfloat16)
        b = torch.rand(shape, dtype=torch.bfloat16)
        at = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=cache_device)
        bt = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=cache_device)
        out = ttnn.to_torch(ttnn_function(at, bt)).float()
        ref = golden_function(a.float(), b.float()).float()
        assert torch.allclose(out, ref, atol=0.05, rtol=0.05), f"{ttnn_function} stale (tensor-tensor)"

    # WARM-UP then assert ZERO growth across fresh allocations (addresses must be patched).
    _run()
    base = cache_device.num_program_cache_entries()
    for _ in range(3):
        _run()
    assert (
        cache_device.num_program_cache_entries() == base
    ), f"{ttnn_function}: cache grew past {base} across fresh allocations (addr not patched)"


# ---------------------------------------------------------------------------
# eltwise / unary. Copied from test_unary.py run_unary_test (ttnn fn, get_golden_function).
# Varying input DATA (fresh address) must not go stale; one entry.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ttnn_function", [ttnn.relu, ttnn.gelu, ttnn.exp, ttnn.sigmoid, ttnn.neg])
def test_unary_cache(cache_device, ttnn_function):
    torch.manual_seed(0)
    shape = (1, 1, 64, 128)
    golden_function = ttnn.get_golden_function(ttnn_function)

    def _run():
        x = torch.rand(shape, dtype=torch.bfloat16)
        xt = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=cache_device)
        out = ttnn.to_torch(ttnn_function(xt)).float()
        ref = golden_function(x.float()).float()
        assert torch.allclose(out, ref, atol=0.05, rtol=0.05), f"{ttnn_function} stale (unary)"

    # WARM-UP then assert ZERO growth across fresh allocations (addresses must be patched).
    _run()
    base = cache_device.num_program_cache_entries()
    for _ in range(3):
        _run()
    assert (
        cache_device.num_program_cache_entries() == base
    ), f"{ttnn_function}: cache grew past {base} across fresh allocations"


# ---------------------------------------------------------------------------
# pool / rotate. Copied from pool/test_rotate.py test_various_angles: NHWC row-major,
# ttnn.rotate(x, angle=..., interpolation_mode=...), real golden + get_rotate_tolerances.
# The angle is the per-call value; varying it must not go stale or add cache entries.
# ---------------------------------------------------------------------------
def test_rotate_cache(cache_device):
    import torch as _torch
    from tests.ttnn.unit_tests.operations.pool.test_rotate import get_rotate_tolerances

    _torch.manual_seed(0)
    input_shape = (1, 16, 16, 64)
    mode = "nearest"
    golden_function = ttnn.get_golden_function(ttnn.rotate)
    x = _torch.randn(input_shape, dtype=_torch.bfloat16)
    xt = ttnn.from_torch(x, layout=ttnn.ROW_MAJOR_LAYOUT, device=cache_device)

    def _run(angle):
        out = ttnn.to_torch(ttnn.rotate(xt, angle=float(angle), interpolation_mode=mode))
        ref = golden_function(x, angle=float(angle), interpolation_mode=mode)
        atol, rtol = get_rotate_tolerances(input_shape, angle, mode)
        assert torch.allclose(out.float(), ref.float(), atol=atol, rtol=rtol), f"rotate stale at angle={angle}"

    # WARM-UP then assert ZERO growth: re-applying an angle must not mint a new entry per call.
    _run(30)
    base = cache_device.num_program_cache_entries()
    for angle in [60, 90, 120]:
        _run(angle)
    assert (
        cache_device.num_program_cache_entries() == base
    ), f"rotate: cache grew past {base} across angles (over-caching)"


# ---------------------------------------------------------------------------
# data_movement / permute. Copied from data_movement/test_permute.py test_permute.
# Pure tensor op (addresses only) -> expected OK (addresses patch on cache hit).
# ---------------------------------------------------------------------------
def test_permute_cache(cache_device):
    torch.manual_seed(0)
    shape = (1, 1, 64, 128)

    def _run():
        x = torch.rand(shape, dtype=torch.bfloat16)
        xt = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=cache_device)
        out = ttnn.to_torch(ttnn.permute(xt, (0, 1, 3, 2))).float()
        ref = torch.permute(x, (0, 1, 3, 2)).float()
        assert torch.allclose(out, ref, atol=0.02), "permute stale"

    # WARM-UP then assert ZERO growth across fresh allocations.
    _run()
    base = cache_device.num_program_cache_entries()
    for _ in range(3):
        _run()
    assert cache_device.num_program_cache_entries() == base, "permute over-caching"


# ---------------------------------------------------------------------------
# data_movement / concat. Copied from test_concat.py test_concat.
# ---------------------------------------------------------------------------
def test_concat_cache(cache_device):
    torch.manual_seed(0)
    shape = (1, 1, 64, 96)

    def _run():
        a = torch.rand(shape, dtype=torch.bfloat16)
        b = torch.rand(shape, dtype=torch.bfloat16)
        at = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=cache_device)
        bt = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=cache_device)
        out = ttnn.to_torch(ttnn.concat([at, bt], dim=2)).float()
        ref = torch.concat([a, b], dim=2).float()
        assert torch.allclose(out, ref, atol=0.02), "concat stale"

    # WARM-UP then assert ZERO growth across fresh allocations.
    _run()
    base = cache_device.num_program_cache_entries()
    for _ in range(3):
        _run()
    assert cache_device.num_program_cache_entries() == base, "concat over-caching"
