# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Local conftest for the graph-capture-derived suite: tags emulator-appropriate cases.

Same contract as ``tests/ops/conftest.py`` — ``-m emulator`` selects the subset the
2-node Quasar emulator can run — but the classification reads the *generated case
data* instead of parameter names, because every test here takes a single ``case``
dict:

  * non-(1, 1) mesh                                     -> not emulator
  * primary input (arg 0) taller than ``_EMU_MAX_ROWS``
    rows, i.e. a prefill-sized activation                -> not emulator
  * total footprint of all inputs over
    ``_EMU_MAX_INPUT_BYTES``                             -> not emulator
  * any L1-sharded input whose captured grid needs more
    than ``_EMU_MAX_CORES`` cores                        -> not emulator

Note the row cap deliberately applies to the *activation* only: a matmul weight or a
paged KV cache is legitimately tall, and judging it by row count would drop every
``linear`` and cache case. That is arg 0 for most ops, but arg 1 for the paged-cache
ops, whose arg 0 is the cache (see ``_PRIMARY_ARG``), the ``input_tensor``
keyword for the sliding-window ops (conv2d, the pools, upsample), which this model
calls entirely by keyword, and the first operand of a tensor list (``concat``). Bulk is handled by the footprint rule instead.

Panoptic-DeepLab runs at 512x1024, so nearly every activation it captured is far
over the row cap; the emulator subset is correspondingly small (see README.md).

These are collection-time estimates, and every case remains runnable without
``-m emulator``. The authoritative check happens at run time in
``graph_case.build_memory_config``, which skips a case whose captured shard grid
does not fit the device actually under test.

    pytest models/experimental/ops/quasar/tests/panoptic_deeplab_ops/ -m emulator
"""

import pytest

_EMU_MAX_ROWS = 128
# 2 nodes x (1x2 worth of usable compute cores) — deliberately conservative; the
# run-time gate in graph_case is the real one.
_EMU_MAX_CORES = 8
_EMU_MAX_INPUT_BYTES = 16 << 20

# Bytes per element, block-float dtypes including their per-face exponents.
_ELEM_BYTES = {
    "FLOAT32": 4,
    "INT32": 4,
    "UINT32": 4,
    "BFLOAT16": 2,
    "UINT16": 2,
    "BFLOAT8_B": 1.0625,
    "BFLOAT4_B": 0.5625,
    "UINT8": 1,
}


def _rows_of(shape) -> int:
    rows = 1
    for d in shape[:-1]:
        rows *= d
    return rows


def _bytes_of(spec) -> float:
    numel = 1
    for d in spec["shape"]:
        numel *= d
    return numel * _ELEM_BYTES.get(spec["dtype"], 2)


def _cores_of(shard) -> int:
    return sum((x1 - x0 + 1) * (y1 - y0 + 1) for x0, y0, x1, y1 in shard["grid"])


# Ops whose first argument is a persistent cache rather than an activation. The row
# cap exists to drop prefill-sized *activations*; a paged KV cache is legitimately
# tall (128 pages x 8 heads x 32 rows), so judging these by arg 0 would exclude them
# from the emulator subset for the one tensor the exemption is about. Their
# activation — the tensor being written into the cache — is arg 1.
_PRIMARY_ARG = {
    "ttnn.experimental.paged_update_cache": 1,
    "ttnn.experimental.paged_fill_cache": 1,
}


def _primary_input(case):
    """The activation whose height decides whether this is a prefill-sized case."""
    start = _PRIMARY_ARG.get(case.get("op"), 0)
    primary = next((a for a in case["args"][start:] if a.get("k") == "t"), None)
    # conv2d / the pools / upsample are called entirely by keyword
    primary = primary or case["kwargs"].get("input_tensor")
    if primary is None:
        # ttnn.concat takes a tensor list; its operands are the activations
        primary = next((a["tensors"][0] for a in case["args"][start:] if a.get("k") == "tlist" and a["tensors"]), None)
    return primary


def _tensor_specs(case):
    for spec in list(case["args"]) + list(case["kwargs"].values()):
        if spec.get("k") == "t":
            yield spec
        elif spec.get("k") == "tlist":
            yield from spec["tensors"]


def _fits_emulator(item) -> bool:
    callspec = getattr(item, "callspec", None)
    if callspec is None:
        return True
    params = callspec.params

    mesh = params.get("ttnn_mesh_device")
    if isinstance(mesh, dict):  # op_utils.with_default_mesh passes {"mesh_shape": …, "l1_small_size": …}
        mesh = mesh.get("mesh_shape")
    if mesh is not None and tuple(mesh) != (1, 1):
        return False

    case = params.get("case")
    if not isinstance(case, dict):
        return True

    primary = _primary_input(case)
    if primary is not None and _rows_of(primary["shape"]) > _EMU_MAX_ROWS:
        return False

    total_bytes = 0.0
    for spec in _tensor_specs(case):
        total_bytes += _bytes_of(spec)
        mem = spec.get("mem") or {}
        shard = mem.get("shard")
        if shard and mem.get("buffer") == "L1" and _cores_of(shard) > _EMU_MAX_CORES:
            return False
    return total_bytes <= _EMU_MAX_INPUT_BYTES


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "emulator: graph-capture case that fits the 2-node Quasar emulator (small / single-device)",
    )


def pytest_collection_modifyitems(config, items):
    for item in items:
        if _fits_emulator(item):
            item.add_marker(pytest.mark.emulator)
