# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# --- BEGIN GENERATED (agentic_port phase 8) ---
# Contracts: (1) every case the codegen gate rejects falls back to native; (2) an accepted case
# dispatched twice on the codegen path stays a program-cache hit and rebinds its buffers; (3) a
# perf-demoted case still lands on native. This block is emitted by the port pipeline and is
# regenerated on every port update; hand-add regressions after its END line. The case tables are
# hand-compacted with `_case` and `_alternating`; the emitter writes them expanded.

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_equal

# `ttnn.concat` takes no implementation argument -- it routes on its own. The forced legs below are
# verification-only bindings on `ttnn._ttnn.operations.data_movement` (concat_nanobind.cpp), not republished
# as `ttnn.*` operations; see concat_force.hpp.
_force_native = ttnn._ttnn.operations.data_movement.concat_force_native
_force_codegen = ttnn._ttnn.operations.data_movement.concat_force_codegen


def _make_input(shape, dtype):
    if dtype in (ttnn.int32, ttnn.uint32):
        return torch.randint(0, 100, shape, dtype=torch.int32)
    return torch.rand(shape, dtype=torch.bfloat16)


def _device_inputs(shapes, dtype, layout, memory_config=None, **on_device):
    # One tensor per input shape, in order: the op takes them as a list. `memory_config` is one
    # placement for every input, or a tuple holding one per input.
    configs = memory_config if isinstance(memory_config, tuple) else (memory_config,) * len(shapes)
    return [
        ttnn.from_torch(
            _make_input(shape, dtype),
            dtype=dtype,
            layout=layout,
            **on_device,
            **({} if config is None else {"memory_config": config}),
        )
        for shape, config in zip(shapes, configs)
    ]


def _reference(inputs, **kwargs):
    # Torch golden over what the device inputs read back as, so a dtype round trip cannot differ.
    return torch.cat([ttnn.to_torch(t) for t in inputs], dim=kwargs["dim"])


_DRAM = ttnn.DRAM_MEMORY_CONFIG
_L1 = ttnn.L1_MEMORY_CONFIG


def _case(shapes, dim, dtype, layout, placement):
    return shapes, {"dim": dim, "memory_config": _DRAM}, dtype, layout, placement


def _alternating(count, first, second):
    return tuple((first, second)[i % 2] for i in range(count))


# A sample of the gate-rejected cases that keeps every refusal reason and the extremes of each
# reason's input and call sizes.
_ROUTING = [
    _case([[1, 1, 16], [1, 1, 1]], -1, ttnn.bfloat16, ttnn.TILE_LAYOUT, _DRAM),
    _case([[1, 1, 16], [1, 1, 1]], -1, ttnn.bfloat8_b, ttnn.TILE_LAYOUT, _DRAM),
    _case([[1, 1, 16], [1, 1, 1]], -1, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, _DRAM),
    _case(
        [[1, 192, 28, 28]] + [[1, 48, 28, 28]] * 12,
        1,
        ttnn.uint32,
        ttnn.ROW_MAJOR_LAYOUT,
        _alternating(13, _L1, _DRAM),
    ),
    _case(
        [[1, 256, 14, 14]] + [[1, 32, 14, 14]] * 47,
        1,
        ttnn.bfloat8_b,
        ttnn.TILE_LAYOUT,
        _alternating(48, _DRAM, _L1),
    ),
    _case(
        [[1, 256, 14, 14]] + [[1, 32, 14, 14]] * 39,
        1,
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        _alternating(40, _L1, _DRAM),
    ),
    _case([[1, 32, 64], [1, 32, 64], [2, 32, 64]], 0, ttnn.uint32, ttnn.TILE_LAYOUT, (_DRAM, _L1, _DRAM)),
    _case([[1, 384, 14, 14]] + [[1, 48, 14, 14]] * 29, 1, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, _DRAM),
    _case([[1, 384, 14, 14]] + [[1, 48, 14, 14]] * 13, 1, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, _DRAM),
    _case(
        [[1, 512, 7, 7]] + [[1, 32, 7, 7]] * 11,
        1,
        ttnn.uint32,
        ttnn.TILE_LAYOUT,
        _alternating(12, _L1, _DRAM),
    ),
    _case([[32, 32, 33, 7], [32, 32, 33, 16], [32, 32, 33, 16]], 3, ttnn.int32, ttnn.TILE_LAYOUT, (_L1, _DRAM, _L1)),
    _case(
        [[64, 96, 96, 32], [64, 96, 96, 64], [64, 96, 96, 64]],
        3,
        ttnn.bfloat8_b,
        ttnn.TILE_LAYOUT,
        (_L1, _DRAM, _L1),
    ),
]
_ROUTING_IDS = [
    "[[1, 1, 16], [1, 1, 1]]|dim=-1&memory_config=BufferType.DRAM|bfloat16|tile",
    "[[1, 1, 16], [1, 1, 1]]|dim=-1&memory_config=BufferType.DRAM|bfloat8_b|tile",
    "[[1, 1, 16], [1, 1, 1]]|dim=-1&memory_config=BufferType.DRAM|float32|row_major",
    "[[1, 192, 28, 28], [1, 48, 28, 28], [1, 48, 28, 28], [1, 48, 28, 28], [1, 48, 28, 28], [1, 48, 28, 28], [1, 48, 28, 28], [1, 48, 28, 28], [1, 48, 28, 28], [1, 48, 28, 28], [1, 48, 28, 28], [1, 48, 28, 28], [1, 48, 28, 28]]@L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1|dim=1&memory_config=BufferType.DRAM|uint32|row_major",
    "[[1, 256, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]]@DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1|dim=1&memory_config=BufferType.DRAM|bfloat8_b|tile",
    "[[1, 256, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]]@L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM|dim=1&memory_config=BufferType.DRAM|bfloat16|row_major",
    "[[1, 32, 64], [1, 32, 64], [2, 32, 64]]@DRAM,L1,DRAM|dim=0&memory_config=BufferType.DRAM|uint32|tile",
    "[[1, 384, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14]]|dim=1&memory_config=BufferType.DRAM|uint32|row_major",
    "[[1, 384, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14]]|dim=1&memory_config=BufferType.DRAM|bfloat16|row_major",
    "[[1, 512, 7, 7], [1, 32, 7, 7], [1, 32, 7, 7], [1, 32, 7, 7], [1, 32, 7, 7], [1, 32, 7, 7], [1, 32, 7, 7], [1, 32, 7, 7], [1, 32, 7, 7], [1, 32, 7, 7], [1, 32, 7, 7], [1, 32, 7, 7]]@L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM|dim=1&memory_config=BufferType.DRAM|uint32|tile",
    "[[32, 32, 33, 7], [32, 32, 33, 16], [32, 32, 33, 16]]@L1,DRAM,L1|dim=3&memory_config=BufferType.DRAM|int32|tile",
    "[[64, 96, 96, 32], [64, 96, 96, 64], [64, 96, 96, 64]]@L1,DRAM,L1|dim=3&memory_config=BufferType.DRAM|bfloat8_b|tile",
]


@pytest.mark.parametrize("shapes,kwargs,dtype,layout,placement", _ROUTING, ids=_ROUTING_IDS)
def test_concat_codegen_routing(device, shapes, kwargs, dtype, layout, placement):
    xt = _device_inputs(shapes, dtype, layout, device=device, memory_config=placement)
    device.clear_program_cache()
    _force_native(xt, **kwargs)  # warm native, so the entry count below sees only codegen
    golden = _reference(xt, **kwargs)
    entries_before = device.num_program_cache_entries()
    out = ttnn.concat(xt, **kwargs)
    assert_equal(golden, ttnn.to_torch(out))
    msg = "auto routed an out-of-scope case to codegen (program cache grew); expected native fallback"
    assert device.num_program_cache_entries() == entries_before, msg


_DEMOTED = [
    _case(shapes, dim, dtype, ttnn.ROW_MAJOR_LAYOUT, placement)
    for shapes, dim in (([[1, 32, 32], [1, 32, 64], [1, 32, 32]], 2), ([[32, 32], [32, 64], [32, 96]], -1))
    for placement in ((_DRAM, _L1, _DRAM), (_L1, _DRAM, _L1), _DRAM)
    for dtype in (ttnn.bfloat16, ttnn.int32, ttnn.uint32)
]
_DEMOTED_IDS = [
    "[[1, 32, 32], [1, 32, 64], [1, 32, 32]]@DRAM,L1,DRAM|dim=2&memory_config=BufferType.DRAM|bfloat16|row_major",
    "[[1, 32, 32], [1, 32, 64], [1, 32, 32]]@DRAM,L1,DRAM|dim=2&memory_config=BufferType.DRAM|int32|row_major",
    "[[1, 32, 32], [1, 32, 64], [1, 32, 32]]@DRAM,L1,DRAM|dim=2&memory_config=BufferType.DRAM|uint32|row_major",
    "[[1, 32, 32], [1, 32, 64], [1, 32, 32]]@L1,DRAM,L1|dim=2&memory_config=BufferType.DRAM|bfloat16|row_major",
    "[[1, 32, 32], [1, 32, 64], [1, 32, 32]]@L1,DRAM,L1|dim=2&memory_config=BufferType.DRAM|int32|row_major",
    "[[1, 32, 32], [1, 32, 64], [1, 32, 32]]@L1,DRAM,L1|dim=2&memory_config=BufferType.DRAM|uint32|row_major",
    "[[1, 32, 32], [1, 32, 64], [1, 32, 32]]|dim=2&memory_config=BufferType.DRAM|bfloat16|row_major",
    "[[1, 32, 32], [1, 32, 64], [1, 32, 32]]|dim=2&memory_config=BufferType.DRAM|int32|row_major",
    "[[1, 32, 32], [1, 32, 64], [1, 32, 32]]|dim=2&memory_config=BufferType.DRAM|uint32|row_major",
    "[[32, 32], [32, 64], [32, 96]]@DRAM,L1,DRAM|dim=-1&memory_config=BufferType.DRAM|bfloat16|row_major",
    "[[32, 32], [32, 64], [32, 96]]@DRAM,L1,DRAM|dim=-1&memory_config=BufferType.DRAM|int32|row_major",
    "[[32, 32], [32, 64], [32, 96]]@DRAM,L1,DRAM|dim=-1&memory_config=BufferType.DRAM|uint32|row_major",
    "[[32, 32], [32, 64], [32, 96]]@L1,DRAM,L1|dim=-1&memory_config=BufferType.DRAM|bfloat16|row_major",
    "[[32, 32], [32, 64], [32, 96]]@L1,DRAM,L1|dim=-1&memory_config=BufferType.DRAM|int32|row_major",
    "[[32, 32], [32, 64], [32, 96]]@L1,DRAM,L1|dim=-1&memory_config=BufferType.DRAM|uint32|row_major",
    "[[32, 32], [32, 64], [32, 96]]|dim=-1&memory_config=BufferType.DRAM|bfloat16|row_major",
    "[[32, 32], [32, 64], [32, 96]]|dim=-1&memory_config=BufferType.DRAM|int32|row_major",
    "[[32, 32], [32, 64], [32, 96]]|dim=-1&memory_config=BufferType.DRAM|uint32|row_major",
]


@pytest.mark.parametrize("shapes,kwargs,dtype,layout,placement", _DEMOTED, ids=_DEMOTED_IDS)
def test_concat_codegen_demotion(device, shapes, kwargs, dtype, layout, placement):
    xt = _device_inputs(shapes, dtype, layout, device=device, memory_config=placement)
    device.clear_program_cache()
    _force_native(xt, **kwargs)  # warm native, so the entry count below sees only codegen
    golden = _reference(xt, **kwargs)
    entries_before = device.num_program_cache_entries()
    out = ttnn.concat(xt, **kwargs)
    assert_equal(golden, ttnn.to_torch(out))
    msg = "auto routed a perf-demoted case to codegen (program cache grew); expected native fallback"
    assert device.num_program_cache_entries() == entries_before, msg


_CACHE_HIT = [
    _case([[1, 1, 16], [1, 1, 1]], -1, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, placement)
    for placement in ((_DRAM, _L1), (_L1, _DRAM), _DRAM)
] + [
    _case(shapes, dim, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, _alternating(len(shapes), *order))
    for shapes, dim in (
        ([[1, 192, 28, 28]] + [[1, 48, 28, 28]] * 7, 1),
        ([[1, 20, 28, 28], [1, 34, 28, 28], [1, 58, 28, 28], [1, 98, 28, 28], [1, 256, 28, 28]], 1),
        ([[1, 64, 56, 56]] + [[1, 32, 56, 56]] * 6, 1),
        ([[1, 64, 56, 56]] + [[1, 32, 56, 56]] * 5, 1),
        ([[32, 32]] * 4, 0),
        ([[32, 64]] * 3, 0),
    )
    for order in ((_DRAM, _L1), (_L1, _DRAM))
]
_CACHE_HIT_IDS = [
    "[[1, 1, 16], [1, 1, 1]]@DRAM,L1|dim=-1&memory_config=BufferType.DRAM|bfloat16|row_major",
    "[[1, 1, 16], [1, 1, 1]]@L1,DRAM|dim=-1&memory_config=BufferType.DRAM|bfloat16|row_major",
    "[[1, 1, 16], [1, 1, 1]]|dim=-1&memory_config=BufferType.DRAM|bfloat16|row_major",
    "[[1, 192, 28, 28], [1, 48, 28, 28], [1, 48, 28, 28], [1, 48, 28, 28], [1, 48, 28, 28], [1, 48, 28, 28], [1, 48, 28, 28], [1, 48, 28, 28]]@DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1|dim=1&memory_config=BufferType.DRAM|bfloat16|row_major",
    "[[1, 192, 28, 28], [1, 48, 28, 28], [1, 48, 28, 28], [1, 48, 28, 28], [1, 48, 28, 28], [1, 48, 28, 28], [1, 48, 28, 28], [1, 48, 28, 28]]@L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM|dim=1&memory_config=BufferType.DRAM|bfloat16|row_major",
    "[[1, 20, 28, 28], [1, 34, 28, 28], [1, 58, 28, 28], [1, 98, 28, 28], [1, 256, 28, 28]]@DRAM,L1,DRAM,L1,DRAM|dim=1&memory_config=BufferType.DRAM|bfloat16|row_major",
    "[[1, 20, 28, 28], [1, 34, 28, 28], [1, 58, 28, 28], [1, 98, 28, 28], [1, 256, 28, 28]]@L1,DRAM,L1,DRAM,L1|dim=1&memory_config=BufferType.DRAM|bfloat16|row_major",
    "[[1, 64, 56, 56], [1, 32, 56, 56], [1, 32, 56, 56], [1, 32, 56, 56], [1, 32, 56, 56], [1, 32, 56, 56], [1, 32, 56, 56]]@DRAM,L1,DRAM,L1,DRAM,L1,DRAM|dim=1&memory_config=BufferType.DRAM|bfloat16|row_major",
    "[[1, 64, 56, 56], [1, 32, 56, 56], [1, 32, 56, 56], [1, 32, 56, 56], [1, 32, 56, 56], [1, 32, 56, 56], [1, 32, 56, 56]]@L1,DRAM,L1,DRAM,L1,DRAM,L1|dim=1&memory_config=BufferType.DRAM|bfloat16|row_major",
    "[[1, 64, 56, 56], [1, 32, 56, 56], [1, 32, 56, 56], [1, 32, 56, 56], [1, 32, 56, 56], [1, 32, 56, 56]]@DRAM,L1,DRAM,L1,DRAM,L1|dim=1&memory_config=BufferType.DRAM|bfloat16|row_major",
    "[[1, 64, 56, 56], [1, 32, 56, 56], [1, 32, 56, 56], [1, 32, 56, 56], [1, 32, 56, 56], [1, 32, 56, 56]]@L1,DRAM,L1,DRAM,L1,DRAM|dim=1&memory_config=BufferType.DRAM|bfloat16|row_major",
    "[[32, 32], [32, 32], [32, 32], [32, 32]]@DRAM,L1,DRAM,L1|dim=0&memory_config=BufferType.DRAM|bfloat16|row_major",
    "[[32, 32], [32, 32], [32, 32], [32, 32]]@L1,DRAM,L1,DRAM|dim=0&memory_config=BufferType.DRAM|bfloat16|row_major",
    "[[32, 64], [32, 64], [32, 64]]@DRAM,L1,DRAM|dim=0&memory_config=BufferType.DRAM|bfloat16|row_major",
    "[[32, 64], [32, 64], [32, 64]]@L1,DRAM,L1|dim=0&memory_config=BufferType.DRAM|bfloat16|row_major",
]


@pytest.mark.parametrize("shapes,kwargs,dtype,layout,placement", _CACHE_HIT, ids=_CACHE_HIT_IDS)
def test_concat_codegen_auto_routes_to_codegen(device, shapes, kwargs, dtype, layout, placement):
    xt = _device_inputs(shapes, dtype, layout, device=device, memory_config=placement)
    device.clear_program_cache()
    _force_native(xt, **kwargs)  # warm native, so the entry count below sees only codegen
    golden = _reference(xt, **kwargs)
    entries_before = device.num_program_cache_entries()
    out = ttnn.concat(xt, **kwargs)
    assert_equal(golden, ttnn.to_torch(out))
    # Hand patch: Wormhole routes width concats to native; `_expect_codegen` is defined below the block.
    expect = _expect_codegen(device, shapes, kwargs["dim"])
    msg = f"auto routed a supported case to the wrong implementation; expected {'codegen' if expect else 'native'}"
    assert (device.num_program_cache_entries() > entries_before) == expect, msg


@pytest.mark.parametrize("shapes,kwargs,dtype,layout,placement", _CACHE_HIT, ids=_CACHE_HIT_IDS)
def test_concat_codegen_program_cache_hit(device, shapes, kwargs, dtype, layout, placement):
    xt = _device_inputs(shapes, dtype, layout, device=device, memory_config=placement)
    golden = _reference(xt, **kwargs)
    assert_equal(golden, ttnn.to_torch(_force_codegen(xt, **kwargs)))
    entries_after_miss = device.num_program_cache_entries()
    # Same spec, a distinct allocation: the cached program must rebind its Buffer*s
    # instead of reusing the first dispatch's addresses.
    yt = _device_inputs(shapes, dtype, layout, device=device, memory_config=placement)
    second_golden = _reference(yt, **kwargs)
    assert_equal(second_golden, ttnn.to_torch(_force_codegen(yt, **kwargs)))
    msg = "second forced-codegen dispatch missed the program cache"
    assert device.num_program_cache_entries() == entries_after_miss, msg


_REFUSED_CASE = _case([[1, 1, 1024], [1, 196, 1024]], 1, ttnn.bfloat16, ttnn.TILE_LAYOUT, (_DRAM, _L1))


def test_concat_forced_codegen_refuses_out_of_scope_case(device, expect_error):
    shapes, kwargs, dtype, layout, placement = _REFUSED_CASE
    xt = _device_inputs(shapes, dtype, layout, device=device, memory_config=placement)
    with expect_error(RuntimeError, "does not support"):
        _force_codegen(xt, **kwargs)


# --- END GENERATED ---


# --- Off-grid regressions (hand-added; edit here, not the emitter) ---

import contextlib  # noqa: E402

from tt_py_test_utils_common.utility_functions import skip_for_slow_dispatch  # noqa: E402


@pytest.fixture(autouse=True)
def _require_concat_codegen_arch(device):
    # Codegen is gated to these architectures (codegen_supported_on_arch); elsewhere auto falls back to native
    # and the forced-codegen legs have nothing to exercise.
    if not (ttnn.device.is_wormhole_b0(device) or ttnn.device.is_blackhole(device)):
        pytest.skip("concat codegen is supported only on Wormhole and Blackhole")


def _is_width_concat(shapes, dim):
    return dim % len(shapes[0]) == len(shapes[0]) - 1


def _expect_codegen(device, shapes, dim, *, otherwise=True):
    """Whether auto routes an in-scope call to codegen: Wormhole sends every width concat to native."""
    if ttnn.device.is_wormhole_b0(device) and _is_width_concat(shapes, dim):
        return False
    return otherwise


def _auto_route(device, xt, **kwargs):
    """Dispatches `ttnn.concat` after warming native, so cache growth means the call took codegen.

    Returns (output, grew).
    """
    device.clear_program_cache()
    _force_native(xt, **kwargs)
    entries_before = device.num_program_cache_entries()
    out = ttnn.concat(xt, **kwargs)
    return out, device.num_program_cache_entries() > entries_before


# Cases the auto router must send to codegen. The 3+-input width cases hold more than 32 output sticks,
# the smallest size the perf gate keeps on codegen.
_AUTO_CODEGEN = [
    pytest.param([[64, 3], [64, 5], [64, 16]], 1, None, ttnn.bfloat16, id="width_3in_odd_widths_bf16"),
    pytest.param([[64, 3], [64, 5], [64, 16]], 1, None, ttnn.int32, id="width_3in_odd_widths_i32"),
    pytest.param([[64, 3], [64, 5], [64, 16]], 1, None, ttnn.uint32, id="width_3in_odd_widths_u32"),
    pytest.param([[40, 1]] * 3, 1, None, ttnn.bfloat16, id="width_3in_single_element_sticks"),
    pytest.param([[40, 32]] * 3, 1, (_L1, _DRAM, _L1), ttnn.bfloat16, id="width_3in_mixed_placement"),
    pytest.param([[1, 2, 20, 16], [1, 2, 20, 8], [1, 2, 20, 8]], 3, None, ttnn.bfloat16, id="width_3in_rank4"),
    pytest.param(
        [[1, 3, 32, 32, 2], [1, 3, 32, 32, 2], [1, 3, 32, 32, 81]], 4, None, ttnn.bfloat16, id="width_3in_rank5"
    ),
    pytest.param([[1, 33, 7], [1, 33, 9], [1, 33, 16]], 2, None, ttnn.bfloat16, id="width_3in_rank3_unaligned"),
    pytest.param([[40, 7], [24, 7]], 0, None, ttnn.bfloat16, id="nonwidth_2in_unaligned_stick"),
    pytest.param([[32], [48]], 0, None, ttnn.bfloat16, id="rank1_2in"),
]


@pytest.mark.parametrize("shapes,dim,placement,dtype", _AUTO_CODEGEN)
def test_concat_auto_routes_in_scope_case_to_codegen(device, shapes, dim, placement, dtype):
    xt = _device_inputs(shapes, dtype, ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=placement)
    out, grew = _auto_route(device, xt, dim=dim, memory_config=_DRAM)
    assert_equal(_reference(xt, dim=dim), ttnn.to_torch(out))
    expect = _expect_codegen(device, shapes, dim)
    assert grew == expect, f"auto served an in-scope case on {'native' if expect else 'codegen'}; expected the other"
    if not grew:
        return
    # The second dispatch uses fresh allocations and must rebind them on a program-cache hit.
    entries = device.num_program_cache_entries()
    yt = _device_inputs(shapes, dtype, ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=placement)
    assert_equal(_reference(yt, dim=dim), ttnn.to_torch(ttnn.concat(yt, dim=dim, memory_config=_DRAM)))
    assert device.num_program_cache_entries() == entries, "second dispatch missed the program cache"


@pytest.mark.parametrize(
    "shapes,dim,out_config",
    [([[40, 32]] * 3, 1, _L1), ([[40, 64], [24, 64]], 0, None)],
    ids=["width_3in_l1_out", "rows_2in_default_out"],
)
def test_concat_auto_codegen_honours_output_memory_config(device, shapes, dim, out_config):
    xt = _device_inputs(shapes, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device=device)
    kwargs = {"dim": dim} if out_config is None else {"dim": dim, "memory_config": out_config}
    out, grew = _auto_route(device, xt, **kwargs)
    assert_equal(_reference(xt, dim=dim), ttnn.to_torch(out))
    assert grew == _expect_codegen(device, shapes, dim)
    if out_config is not None:
        assert out.memory_config().buffer_type == out_config.buffer_type


# The perf gate demotes a 3+-input row-major width concat to native up to this many output sticks.
_MAX_DEMOTED_STICKS = 32
# The most inputs the codegen reader kernels take.
_MAX_CODEGEN_INPUTS = 8


@pytest.mark.parametrize(
    "sticks,expect_codegen",
    [(_MAX_DEMOTED_STICKS, False), (_MAX_DEMOTED_STICKS + 1, True)],
    ids=["32_sticks_native", "33_sticks_codegen"],
)
def test_concat_width_nway_perf_gate_boundary(device, sticks, expect_codegen):
    xt = _device_inputs([[sticks, 32]] * 3, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device=device)
    out, grew = _auto_route(device, xt, dim=1, memory_config=_DRAM)
    assert_equal(_reference(xt, dim=1), ttnn.to_torch(out))
    assert grew == _expect_codegen(device, [[sticks, 32]] * 3, 1, otherwise=expect_codegen)


@pytest.mark.parametrize("dim,shape", [(1, [40, 32]), (0, [40, 32])], ids=["width", "rows"])
def test_concat_input_count_limit(device, dim, shape, expect_error):
    at_limit = _device_inputs([shape] * _MAX_CODEGEN_INPUTS, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device=device)
    out, grew = _auto_route(device, at_limit, dim=dim, memory_config=_DRAM)
    assert_equal(_reference(at_limit, dim=dim), ttnn.to_torch(out))
    assert grew == _expect_codegen(device, [shape], dim), "auto routed a case at the input-count limit wrongly"

    over = _device_inputs([shape] * (_MAX_CODEGEN_INPUTS + 1), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device=device)
    out, grew = _auto_route(device, over, dim=dim, memory_config=_DRAM)
    assert_equal(_reference(over, dim=dim), ttnn.to_torch(out))
    assert not grew, "auto routed a case over the input-count limit to codegen"
    with expect_error(RuntimeError, "does not support"):
        _force_codegen(over, dim=dim, memory_config=_DRAM)


def _codegen_vs_golden(device, xt, dim):
    entries = device.num_program_cache_entries()
    out = _force_codegen(xt, dim=dim, memory_config=_DRAM)
    assert_equal(_reference(xt, dim=dim), ttnn.to_torch(out))
    return device.num_program_cache_entries() - entries


@pytest.mark.parametrize("dim", [0, 1], ids=["rows", "width"])
def test_concat_codegen_repeated_input_does_not_alias_distinct_inputs(device, dim):
    # A cached program is patched per Buffer*; a call whose inputs repeat one buffer must not share a
    # cache entry with one whose inputs are distinct, in either order.
    x, y, z = _device_inputs([[40, 32]] * 3, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device=device)
    view = ttnn.reshape(x, list(x.shape))  # a new tensor over the same buffer
    device.clear_program_cache()
    assert _codegen_vs_golden(device, [x, x], dim) == 1
    assert _codegen_vs_golden(device, [x, y], dim) == 1
    assert _codegen_vs_golden(device, [x, view], dim) == 0
    assert _codegen_vs_golden(device, [y, z], dim) == 0
    assert _codegen_vs_golden(device, [x, x], dim) == 0
    if dim != 1:
        return
    # The width N-way readers take the same signature: a partial repeat is its own entry.
    assert _codegen_vs_golden(device, [x, y, x], dim) == 1
    assert _codegen_vs_golden(device, [x, y, z], dim) == 1
    assert _codegen_vs_golden(device, [y, x, y], dim) == 0
    assert _codegen_vs_golden(device, [x, x, x], dim) == 1
    assert _codegen_vs_golden(device, [z, x, y], dim) == 0


def _forced_refusal_grouped(device):
    xt = _device_inputs([[40, 32]] * 2, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device=device)
    return lambda: _force_codegen(xt, dim=0, groups=2)


def _forced_refusal_sub_core_grids(device):
    xt = _device_inputs([[40, 32]] * 2, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device=device)
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
    return lambda: _force_codegen(xt, dim=0, sub_core_grids=grid)


def _forced_refusal_output_tensor(device):
    xt = _device_inputs([[40, 32]] * 2, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device=device)
    out = _device_inputs([[80, 32]], ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device=device)[0]
    return lambda: _force_codegen(xt, dim=0, output_tensor=out)


def _forced_refusal_empty_input(device):
    empty = ttnn.from_torch(
        torch.zeros([0, 32], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    other = _device_inputs([[4, 32]], ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device=device)[0]
    return lambda: _force_codegen([empty, other], dim=0)


@pytest.mark.parametrize(
    "make_call,message",
    [
        (_forced_refusal_grouped, "does not support"),
        (_forced_refusal_sub_core_grids, "does not support"),
        (_forced_refusal_output_tensor, "optional output tensor currently unsupported"),
        (_forced_refusal_empty_input, "does not support"),
    ],
    ids=["groups", "sub_core_grids", "output_tensor", "empty_input"],
)
def test_concat_forced_codegen_refuses(device, expect_error, make_call, message):
    call = make_call(device)
    with expect_error(RuntimeError, message):
        call()


# --- Live-L1 gate and CB plan ---
# The CB depth is planned from the L1 free at dispatch and is part of the program-cache key, and a call
# the free L1 cannot hold at all falls back to native. A stick of `_stick_bytes` makes the output CB
# footprint a whole number of "slots" (2 * batch * stick for row-major, plus a small scratch for width
# concats), so pinning a known headroom selects a known batch of 4, 2 or 1.
_TILE_BYTES = 32 * 32 * 2


def _pin_l1_headroom(device, headroom_bytes):
    """Lowers the live L1 frontier to about `headroom_bytes` above the CB base on every bank."""
    info = ttnn._ttnn.reports.get_device_info(device)
    tiles_per_bank = (info.cb_limit - headroom_bytes) // _TILE_BYTES
    resident = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 32 * tiles_per_bank, 32 * info.l1_num_banks]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
    )
    return resident, resident.buffer_address() - info.address_at_first_l1_cb_buffer


def _stick_bytes(device):
    """An output stick of about a ninth of clear L1, so a batch-4 plan (8 slots) still fits unpinned."""
    info = ttnn._ttnn.reports.get_device_info(device)
    return (info.cb_limit - info.address_at_first_l1_cb_buffer) // 9 // 128 * 128


# (inputs, dim, elements per input stick) with an output stick of `stick` bytes.
_L1_CASES = {
    "rows_2in": lambda stick: ([[4, stick // 2]] * 2, 0),
    "width_2in": lambda stick: ([[40, stick // 4]] * 2, 1),
}


@pytest.mark.parametrize("case", list(_L1_CASES))
def test_concat_codegen_cb_plan_follows_live_l1(device, case):
    if case == "width_2in" and ttnn.device.is_wormhole_b0(device):
        pytest.skip("Wormhole routes width concats to native, so no codegen plan follows live L1")
    shapes, dim = _L1_CASES[case](_stick_bytes(device))
    xt = _device_inputs(shapes, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device=device)
    kwargs = {"dim": dim, "memory_config": _DRAM}
    golden = _reference(xt, **kwargs)
    stick = _stick_bytes(device)

    out, grew = _auto_route(device, xt, **kwargs)
    assert_equal(golden, ttnn.to_torch(out))
    assert grew, "auto served the unpressured call on native; expected codegen"
    entries_clear = device.num_program_cache_entries()

    # With 5 slots free the batch-4 program (8 slots) would overlap the pinned buffer, so a
    # smaller plan must be compiled; with 3 slots free, a smaller one still.
    entries = entries_clear
    for slots in (5, 3):
        resident, headroom = _pin_l1_headroom(device, slots * stick)
        try:
            assert slots * stick <= headroom < (slots + 1) * stick, f"pinned headroom {headroom} B is not {slots} slots"
            assert_equal(golden, ttnn.to_torch(ttnn.concat(xt, **kwargs)))
            msg = f"the call with {slots} slots free hit a program planned for more L1"
            assert device.num_program_cache_entries() == entries + 1, msg
            entries += 1
        finally:
            ttnn.deallocate(resident)

    # Releasing the pin restores the original plan, which is still cached.
    assert_equal(golden, ttnn.to_torch(ttnn.concat(xt, **kwargs)))
    assert device.num_program_cache_entries() == entries, "releasing the pin did not return to the cached plan"


def test_concat_codegen_cache_key_ignores_a_frontier_move_that_keeps_the_plan(device):
    shapes, dim = _L1_CASES["rows_2in"](_stick_bytes(device))
    xt = _device_inputs(shapes, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device=device)
    kwargs = {"dim": dim, "memory_config": _DRAM}
    out, grew = _auto_route(device, xt, **kwargs)
    assert grew
    entries = device.num_program_cache_entries()
    # About 2 MB over the whole grid: far below the slack left over by the batch-4 plan.
    small = ttnn.from_torch(
        torch.zeros([1, 1, 256, 4096], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_L1,
    )
    try:
        assert_equal(_reference(xt, **kwargs), ttnn.to_torch(ttnn.concat(xt, **kwargs)))
        assert device.num_program_cache_entries() == entries
    finally:
        ttnn.deallocate(small)


def test_concat_codegen_l1_output_cache_key_matches_the_built_plan(device):
    # The launcher allocates the output before the cache lookup, so the key must read the live L1 the factory
    # reads. With the output charged a second time the key lags the built plan by one batch: the batch-4 program
    # built at 8.5 slots free below would be replayed at 6.5 slots, where only a batch-2 plan fits.
    stick = _stick_bytes(device)
    shapes, dim = _L1_CASES["rows_2in"](stick)
    xt = _device_inputs(shapes, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device=device)
    kwargs = {"dim": dim, "memory_config": _L1}
    golden = _reference(xt, **kwargs)
    device.clear_program_cache()

    # The 8-page output takes one `stick` page from each of 8 banks, so the headroom left once it is allocated is
    # one slot below what is pinned here.
    entries = device.num_program_cache_entries()
    for pinned_slots, step in ((9.5, "batch-4 plan"), (7.5, "batch-2 plan"), (9.5, "cached batch-4 plan")):
        resident, _ = _pin_l1_headroom(device, int(pinned_slots * stick))
        try:
            out = _force_codegen(xt, **kwargs)
            assert_equal(golden, ttnn.to_torch(out))
            ttnn.deallocate(out)
        finally:
            ttnn.deallocate(resident)
        expected = 0 if step.startswith("cached") else 1
        assert device.num_program_cache_entries() == entries + expected, f"{step}: wrong program-cache growth"
        entries += expected


def test_concat_falls_back_to_native_when_free_l1_holds_no_codegen_plan(device):
    # A width concat needs a scratch stick of alignment bytes on top of the two output slots; with
    # just over two slots free codegen has no plan and native, which stages less, must serve the call.
    stick = _stick_bytes(device)
    shapes, dim = _L1_CASES["width_2in"](stick)
    xt = _device_inputs(shapes, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device=device)
    kwargs = {"dim": dim, "memory_config": _DRAM}
    golden = _reference(xt, **kwargs)
    device.clear_program_cache()
    _force_native(xt, **kwargs)
    entries = device.num_program_cache_entries()
    resident, headroom = _pin_l1_headroom(device, 2 * stick + stick // 5)
    try:
        assert 2 * stick <= headroom < 5 * stick // 2, f"pinned headroom {headroom} B is not just over 2 slots"
        assert_equal(golden, ttnn.to_torch(ttnn.concat(xt, **kwargs)))
        assert device.num_program_cache_entries() == entries, "auto compiled a codegen program that does not fit"
    finally:
        ttnn.deallocate(resident)


def test_concat_forced_codegen_refuses_when_free_l1_holds_no_slot(device, expect_error):
    stick = _stick_bytes(device)
    shapes, dim = _L1_CASES["rows_2in"](stick)
    xt = _device_inputs(shapes, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device=device)
    resident, headroom = _pin_l1_headroom(device, stick // 3)
    try:
        assert headroom < stick
        with expect_error(RuntimeError, "of L1 per core"):
            _force_codegen(xt, dim=dim, memory_config=_DRAM)
    finally:
        ttnn.deallocate(resident)


# Trace capture compiles nothing, so the plan it needs must already be cached. The plan follows the
# frontier, hence the capture only finds it when warm-up ran under the same frontier.
_TRACE_DEVICE_PARAMS = {"trace_region_size": 4 * 1024 * 1024}


def _best_effort(action):
    try:
        action()
    except RuntimeError:
        pass


@contextlib.contextmanager
def _capturing(device):
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    try:
        yield trace_id
    finally:
        _best_effort(lambda: ttnn.end_trace_capture(device, trace_id, cq_id=0))


@pytest.mark.parametrize(
    "pin_before_warmup",
    [
        True,
        pytest.param(
            False,
            marks=pytest.mark.xfail(
                reason="https://github.com/tenstorrent/tt-metal/issues/46533: a live-L1-derived plan "
                "misses the cache when the frontier moves between warm-up and capture, and a capture "
                "cannot compile the plan it then needs",
            ),
        ),
    ],
    ids=["warm_under_pressure", "warm_then_pressure"],
)
@pytest.mark.parametrize("device_params", [_TRACE_DEVICE_PARAMS], indirect=True)
@skip_for_slow_dispatch("trace capture requires fast dispatch")
def test_concat_codegen_trace_capture_under_pressure(device, pin_before_warmup):
    # A plan cached by an earlier test would let the capture succeed without warm-up having compiled it.
    device.clear_program_cache()
    stick = _stick_bytes(device)
    shapes, dim = _L1_CASES["rows_2in"](stick)
    xt = _device_inputs(shapes, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device=device)
    kwargs = {"dim": dim, "memory_config": _DRAM}
    golden_before = _reference(xt, **kwargs)
    swapped = _make_input(shapes[0], ttnn.bfloat16)
    golden_after = torch.cat([swapped, ttnn.to_torch(xt[1])], dim=dim)

    trace_id = None
    resident = None
    try:
        if pin_before_warmup:
            resident, _ = _pin_l1_headroom(device, 3 * stick)
        warm = _force_codegen(xt, **kwargs)
        assert_equal(golden_before, ttnn.to_torch(warm))
        ttnn.deallocate(warm)
        entries_after_warmup = device.num_program_cache_entries()
        if not pin_before_warmup:
            resident, _ = _pin_l1_headroom(device, 3 * stick)

        with _capturing(device) as trace_id:
            output = _force_codegen(xt, **kwargs)
        ttnn.synchronize_device(device)
        assert (
            device.num_program_cache_entries() == entries_after_warmup
        ), "capture needed a plan warm-up had not cached"

        # Replace the first input in place, so a replay that never reached the device would differ.
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(swapped, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT), xt[0])
        ttnn.synchronize_device(device)
        ttnn.execute_trace(device, trace_id, blocking=True)
        ttnn.synchronize_device(device)
        assert_equal(golden_after, ttnn.to_torch(output))
    finally:
        if trace_id is not None:
            _best_effort(lambda: ttnn.release_trace(device, trace_id))
        if resident is not None:
            ttnn.deallocate(resident)


# Wormhole routes every width concat to native; non-width concats stay on codegen there. Blackhole keeps both
# on codegen above the N-way cutoff (see the in-scope cases above).
@pytest.mark.parametrize(
    "shapes,dim",
    [
        pytest.param([[1, 1, 128, 100], [1, 1, 128, 200]], -1, id="width_2in"),
        pytest.param([[1, 3, 32, 32, 2], [1, 3, 32, 32, 2], [1, 3, 32, 32, 81]], 4, id="width_3in"),
    ],
)
def test_concat_wormhole_routes_width_to_native(device, shapes, dim):
    if not ttnn.device.is_wormhole_b0(device):
        pytest.skip("width routing to native is Wormhole-only")
    width = _device_inputs(shapes, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device=device)
    out, grew = _auto_route(device, width, dim=dim, memory_config=_DRAM)
    assert_equal(_reference(width, dim=dim), ttnn.to_torch(out))
    assert not grew, "Wormhole routed a width concat to codegen; expected native"

    rows = _device_inputs([[1, 100, 14, 14]] * 2, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device=device)
    out, grew = _auto_route(device, rows, dim=1, memory_config=_DRAM)
    assert_equal(_reference(rows, dim=1), ttnn.to_torch(out))
    assert grew, "Wormhole routed a non-width concat to native; expected codegen"

    # Forced codegen still runs the kernels on a width concat.
    assert_equal(_reference(width, dim=dim), ttnn.to_torch(_force_codegen(width, dim=dim, memory_config=_DRAM)))
