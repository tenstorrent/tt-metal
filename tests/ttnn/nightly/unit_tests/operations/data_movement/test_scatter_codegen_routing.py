# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# --- BEGIN GENERATED (agentic_port phase 8) ---
# Contracts: (1) every case the codegen gate rejects falls back to native; (2) an accepted case
# dispatched twice on the codegen path stays a program-cache hit and rebinds its buffers; (3) a
# perf-demoted case still lands on native. This block is emitted from the port's coverage ledger
# and is regenerated on every port update; hand-add off-grid regressions after its END line.

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_equal

# `ttnn.scatter` takes no implementation argument -- it routes on its own. The forced legs below are
# the verification-only entries in the private module; see scatter_force.hpp.
_force_native = ttnn._ttnn.operations.data_movement.scatter_force_native
_force_codegen = ttnn._ttnn.operations.data_movement.scatter_force_codegen


def _make_input(shape, dtype):
    if dtype in (ttnn.int32, ttnn.uint32):
        return torch.randint(0, 100, shape, dtype=torch.int32)
    return torch.rand(shape, dtype=torch.bfloat16)


def _materialize(shape, kwargs, dtype, layout, device):
    # Secondary inputs are case specs (shape lists), not tensors: built once per case so
    # every leg sees the same tensors, by the same rule the verify harness applies.
    index_shape = kwargs.get("index")
    if not isinstance(index_shape, list):
        return kwargs
    out = dict(kwargs)
    dim = kwargs.get("dim", -1)
    axis = dim if dim >= 0 else len(shape) + dim
    axis_len = int(shape[axis])
    index = torch.randint(0, axis_len, index_shape, dtype=torch.int32)
    index_dtype = ttnn.int32
    out["index"] = ttnn.from_torch(index, dtype=index_dtype, layout=layout, device=device)
    src_shape = kwargs.get("src")
    if isinstance(src_shape, list):
        src = torch.rand(src_shape, dtype=torch.bfloat16)
        out["src"] = ttnn.from_torch(src, dtype=dtype, layout=layout, device=device)
    return out


_DEMOTED = [
    (
        [1, 2, 128, 1, 768],
        {"dim": 2, "index": [1, 2, 8, 1, 768], "src": [1, 2, 8, 1, 768]},
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
    ),
    ([100], {"dim": 0, "index": [80], "src": [80]}, ttnn.bfloat16, ttnn.TILE_LAYOUT),
]
_DEMOTED_IDS = [
    "[1, 2, 128, 1, 768]|dim=2&index=[1, 2, 8, 1, 768]&src=[1, 2, 8, 1, 768]|bfloat16|tile",
    "[100]|dim=0&index=[80]&src=[80]|bfloat16|tile",
]


@pytest.mark.parametrize("shape,kwargs,dtype,layout", _DEMOTED, ids=_DEMOTED_IDS)
def test_scatter_codegen_demotion(device, shape, kwargs, dtype, layout):
    x = _make_input(shape, dtype)
    xt = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device)
    kwargs = _materialize(shape, kwargs, dtype, layout, device)
    device.clear_program_cache()
    golden = ttnn.to_torch(_force_native(xt, **kwargs))
    entries_before = device.num_program_cache_entries()
    out = ttnn.scatter(xt, **kwargs)
    assert_equal(golden, ttnn.to_torch(out))
    msg = "auto routed a perf-demoted case to codegen (program cache grew); expected native fallback"
    assert device.num_program_cache_entries() == entries_before, msg


_CACHE_HIT = [
    ([1, 1, 32, 64], {"dim": -1, "index": [1, 1, 32, 32], "src": [1, 1, 32, 32]}, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
    ([1, 1, 32, 64], {"dim": -1, "index": [1, 1, 32, 32], "src": [1, 1, 32, 32]}, ttnn.bfloat16, ttnn.TILE_LAYOUT),
]
_CACHE_HIT_IDS = [
    "[1, 1, 32, 64]|dim=-1&index=[1, 1, 32, 32]&src=[1, 1, 32, 32]|bfloat16|row_major",
    "[1, 1, 32, 64]|dim=-1&index=[1, 1, 32, 32]&src=[1, 1, 32, 32]|bfloat16|tile",
]


@pytest.mark.parametrize("shape,kwargs,dtype,layout", _CACHE_HIT, ids=_CACHE_HIT_IDS)
def test_scatter_codegen_auto_routes_to_codegen(device, shape, kwargs, dtype, layout):
    x = _make_input(shape, dtype)
    xt = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device)
    kwargs = _materialize(shape, kwargs, dtype, layout, device)
    device.clear_program_cache()
    golden = ttnn.to_torch(_force_native(xt, **kwargs))
    entries_before = device.num_program_cache_entries()
    out = ttnn.scatter(xt, **kwargs)
    assert_equal(golden, ttnn.to_torch(out))
    msg = "auto routed a supported case to native (program cache did not grow); expected codegen"
    assert device.num_program_cache_entries() > entries_before, msg


@pytest.mark.parametrize("shape,kwargs,dtype,layout", _CACHE_HIT, ids=_CACHE_HIT_IDS)
def test_scatter_codegen_program_cache_hit(device, shape, kwargs, dtype, layout):
    x = _make_input(shape, dtype)
    xt = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device)
    spec = kwargs
    kwargs = _materialize(shape, spec, dtype, layout, device)
    golden = ttnn.to_torch(_force_native(xt, **kwargs))
    assert_equal(golden, ttnn.to_torch(_force_codegen(xt, **kwargs)))
    entries_after_miss = device.num_program_cache_entries()
    # Same spec, a distinct allocation: the cached program must rebind its Buffer*s
    # instead of reusing the first dispatch's addresses.
    yt = ttnn.from_torch(_make_input(shape, dtype), dtype=dtype, layout=layout, device=device)
    kwargs = _materialize(shape, spec, dtype, layout, device)
    second_golden = ttnn.to_torch(_force_native(yt, **kwargs))
    assert_equal(second_golden, ttnn.to_torch(_force_codegen(yt, **kwargs)))
    msg = "second forced-codegen dispatch missed the program cache"
    assert device.num_program_cache_entries() == entries_after_miss, msg


_CALL_CONTRACT_CASE = (
    [1, 1, 32, 64],
    {"dim": -1, "index": [1, 1, 32, 32], "src": [1, 1, 32, 32]},
    ttnn.bfloat16,
    ttnn.ROW_MAJOR_LAYOUT,
)


def test_scatter_codegen_unsupported_output_memory_config_falls_back(device):
    shape, kwargs, dtype, layout = _CALL_CONTRACT_CASE
    xt = ttnn.from_torch(_make_input(shape, dtype), dtype=dtype, layout=layout, device=device)
    kwargs = _materialize(shape, kwargs, dtype, layout, device)
    golden = _force_native(xt, **kwargs)
    sharded = ttnn.create_sharded_memory_config(
        list(golden.shape), core_grid=ttnn.CoreGrid(x=1, y=1), strategy=ttnn.ShardStrategy.HEIGHT
    )
    # Sweeps only ever run the default memory config, which inherits the input's placement,
    # so nothing else in this file reaches the output side of the call contract.
    out = ttnn.scatter(xt, **kwargs, memory_config=sharded)
    assert_equal(ttnn.to_torch(golden), ttnn.to_torch(out))


# --- END GENERATED ---


# --- Off-grid regressions (hand-added; edit here, not the emitter) ---
