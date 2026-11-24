# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Golden of the Kimi-K3 KDA layer's per-chunk outputs and carries, for a storage change that must not move a bit.

The production-dimension synthetic K3 layer (Galaxy SP8xTP4, or one 4x4 stage: SP4 on the axis-1 ring, TP4 on axis 0)
runs over a two-slot, one-layer slab on a schedule of chunks, eager and then as one captured trace replayed with other
slots and bounds. After every chunk it records, on every chip: the layer output (its valid rows) and every slot's carry read back as a
native carry (``read_slab_layer``). ``KDA_LAYER_GOLDEN=save:<file>`` writes them; ``KDA_LAYER_GOLDEN=compare:<file>``
requires every tensor to match the saved one bit for bit. Take the golden at the commit before the change and compare
at the change; no mode of the old storage needs to stay alive for it. Everything the test writes or reads goes through the
native (SP-replicated) carry, so the records do not depend on how the slabs store it: only ``_write_native`` and
``_layer_call`` touch the storage, and a copy of this file run at the old commit replaces just those two.

The schedule moves the chronologically first SP rank (``(actual_start // rows) % SP``) and the final-state owner (the
rank holding ``actual_end - 1``) across ranks, with request heads (``actual_start == 0``: the layer's ops start from zeros), a
split tail (a start inside a rank's rows), short intervals (``actual_end`` inside the first or the next rank), and a
non-zero slot. Every slot starts as noise, so a head that matched without the zero fill would be visible.
"""

from __future__ import annotations

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.tests.kda.kda_test_utils import assert_bit_identical
from models.demos.deepseek_v3_d_p.tests.kda.utils import (
    kda_device_params,
    make_kimi_k3_device_case,
    make_synthetic_kimi_k3_test_case,
    mla_row_permutation,
    native_state_from_global,
    read_slab_layer,
    write_slab_layer,
)
from models.demos.deepseek_v3_d_p.tt.kda.state_adapter import KdaStates, deallocate_state, device_scalar
from models.demos.deepseek_v3_d_p.tt.kimi_k3.kda_geometry import kimi_k3_kda_geometry
from models.demos.deepseek_v3_d_p.utils.chunk_config import PREFILL_CHUNK_TOKENS_PER_CHIP

pytestmark = [run_for_blackhole(), pytest.mark.timeout(1800)]

_GOLDEN_ENV = "KDA_LAYER_GOLDEN"
_LAYER = 1
_NUM_SLOTS = 2
_ROWS = PREFILL_CHUNK_TOKENS_PER_CHIP
# Per layout: (slot, actual_start, actual_end) -> first rank, final-state owner, with 640 rows per rank.
_C8 = 8 * _ROWS  # 5120, the 8x4 chunk
_EAGER_CHUNKS_SP8 = [
    (1, 0, _C8),  # head on slot 1: zeros, first 0, owner 7
    (0, 4 * _ROWS + 320, 4 * _ROWS + 320 + _C8),  # split tail: first 4, owner 4 (wraps)
    (1, 3 * _ROWS, 3 * _ROWS + _C8 - 1000),  # continues slot 1: first 3, owner 1
    (0, 5 * _ROWS + 40, 5 * _ROWS + 140),  # short interval inside the first rank: first 5, owner 5
    (1, 7 * _ROWS + 100, 9 * _ROWS + 300),  # short, split tail across the wrap: first 7, owner 1
    (0, 0, _ROWS + 60),  # short head on slot 0: first 0, owner 1
]
# Captured once on slot 0 at a head, replayed with these metadata.
_TRACED_CHUNKS_SP8 = [
    (1, 2 * _ROWS + 17, 2 * _ROWS + 17 + _C8),  # first 2, owner 2
    (0, 6 * _ROWS, 6 * _ROWS + 3 * _ROWS),  # first 6, owner 0
    (1, 0, 3 * _ROWS + 5),  # head on slot 1 in a replay: first 0, owner 3
    (0, _ROWS + 600, _ROWS + 630),  # short split tail: first 1, owner 1
]
# The 4x4 stage's 2560-token chunk over SP4; every start 32-aligned.
_C4 = 4 * _ROWS
_EAGER_CHUNKS_SP4 = [
    (1, 0, _C4),  # head on slot 1: zeros, first 0, owner 3
    (0, 2 * _ROWS + 320, 2 * _ROWS + 320 + _C4),  # split tail: first 2, owner 2 (wraps)
    (1, _ROWS, _ROWS + _C4 - 1000),  # continues slot 1: first 1, owner 3
    (0, 3 * _ROWS + 32, 3 * _ROWS + 132),  # short interval inside the first rank: first 3, owner 3
    (1, 3 * _ROWS + 96, 5 * _ROWS + 300),  # short, split tail across the wrap: first 3, owner 1
    (0, 0, _ROWS + 60),  # short head on slot 0: first 0, owner 1
]
_TRACED_CHUNKS_SP4 = [
    (1, 2 * _ROWS + 32, 2 * _ROWS + 32 + _C4),  # first 2, owner 2
    (0, 2 * _ROWS, 2 * _ROWS + 3 * _ROWS),  # first 2, owner 0
    (1, 0, 3 * _ROWS + 5),  # head on slot 1 in a replay: first 0, owner 3
    (0, _ROWS + 576, _ROWS + 606),  # short split tail: first 1, owner 1
]


def _golden_mode() -> tuple[str, str]:
    value = os.environ.get(_GOLDEN_ENV, "")
    mode, _, path = value.partition(":")
    if mode not in ("save", "compare") or not path:
        pytest.skip(f"set {_GOLDEN_ENV}=save:<file> or compare:<file>")
    return mode, path


def _host(tensor: ttnn.Tensor) -> list[torch.Tensor]:
    return [ttnn.to_torch(shard).clone() for shard in ttnn.get_device_tensors(tensor)]


def _valid_output_rows(
    shards: list[torch.Tensor], mesh_shape: tuple[int, int], sp_axis: int, start: int, end: int
) -> list[torch.Tensor]:
    """The output with its padded rows (positions at or past ``actual_end``) zeroed: the layer leaves them unspecified
    (``ttKDA.forward``), so they hold whatever the buffer held before and are not part of the contract."""
    sp_size = mesh_shape[sp_axis]
    valid = mla_row_permutation(start, sp_size, _ROWS).reshape(sp_size, _ROWS) < end - start
    masked = []
    for chip, shard in enumerate(shards):
        sp_rank = divmod(chip, mesh_shape[1])[sp_axis]
        masked.append(shard.masked_fill(~valid[sp_rank].reshape(1, _ROWS, 1), 0))
    return masked


def _write_native(mesh_device, slabs: KdaStates, slot: int, layer: int, state) -> None:
    """Store a native (SP-replicated) carry as the slot's carry."""
    write_slab_layer(mesh_device, slabs, state, slot, layer)


def _layer_call(layer, hidden, slabs: KdaStates, slot_tt, start_tt, end_tt) -> ttnn.Tensor:
    """The production per-chunk layer on the slot's carry in the slabs."""
    output, _ = layer.forward(hidden, slabs.entry(slot_tt, _LAYER), start_tt, end_tt)
    return output


@pytest.mark.parametrize(
    "mesh_device,device_params,sp_axis,eager_chunks,traced_chunks",
    [
        pytest.param(
            (8, 4),
            kda_device_params((8, 4), trace_region_size=64 * 1024 * 1024),
            0,
            _EAGER_CHUNKS_SP8,
            _TRACED_CHUNKS_SP8,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="SP8xTP4-torus-xy",
        ),
        pytest.param(
            (4, 4),
            kda_device_params((4, 4), trace_region_size=64 * 1024 * 1024),
            1,
            _EAGER_CHUNKS_SP4,
            _TRACED_CHUNKS_SP4,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 4), topology="mesh-4x4"),
            id="SP4xTP4-4x4-sp1",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_layer_outputs_and_carries_match_golden(mesh_device, device_params, sp_axis, eager_chunks, traced_chunks):
    mode, path = _golden_mode()
    started = time.monotonic()

    def stage_line(what: str) -> None:
        logger.info(f"[golden {mode} +{time.monotonic() - started:.0f}s] {what}")

    tp_axis = 1 - sp_axis
    mesh_shape = tuple(mesh_device.shape)
    chunk = mesh_shape[sp_axis] * _ROWS
    case = make_synthetic_kimi_k3_test_case(sequence=chunk)
    layer, hidden = make_kimi_k3_device_case(
        mesh_device, case, tensor_parallel_axis=tp_axis, cache_weights=False, shard_hidden_over_tp=True
    )
    stage_line("layer built")
    geometry = kimi_k3_kda_geometry(tuple(mesh_device.shape), sp_axis, tp_axis)
    slabs = KdaStates.allocate(mesh_device, geometry, layer_ids=(_LAYER,), num_slots=_NUM_SLOTS)
    # Every slot starts as noise, the same native carries whatever the storage, so a stale read shows.
    generator = torch.Generator().manual_seed(23)
    for slot in range(_NUM_SLOTS):
        recurrent = torch.randn(geometry.num_heads, geometry.head_dim, geometry.head_dim, generator=generator)
        convolution = torch.randn(
            geometry.conv_history, 3 * geometry.num_heads * geometry.head_dim, generator=generator
        ).bfloat16()
        noise = native_state_from_global(mesh_device, geometry, recurrent, convolution, tp_axis)
        _write_native(mesh_device, slabs, slot, _LAYER, noise)
        deallocate_state(noise)
    records: dict[str, list[torch.Tensor]] = {}

    def record(label: str, output: ttnn.Tensor | None, start: int = 0, end: int = 0) -> None:
        """``output`` None: before the first call. Only the output's valid rows, ``[start, end)``, are recorded."""
        ttnn.synchronize_device(mesh_device)
        if output is not None:
            records[f"{label}/output"] = _valid_output_rows(_host(output), mesh_shape, sp_axis, start, end)
        for slot in range(_NUM_SLOTS):
            carry = read_slab_layer(mesh_device, slabs, slot, _LAYER)
            records[f"{label}/slot{slot}.recurrent"] = _host(carry.recurrent)
            records[f"{label}/slot{slot}.convolution"] = _host(carry.convolution)
            deallocate_state(carry)
        stage_line(f"recorded {label}")

    record("noise", None)
    slot_tt, start_tt, end_tt = (device_scalar(mesh_device, value) for value in (0, 0, chunk))

    def stage(slot: int, start: int, end: int) -> None:
        for destination, value in ((slot_tt, slot), (start_tt, start), (end_tt, end)):
            source = device_scalar(mesh_device, value)
            ttnn.copy(source, destination)
            ttnn.deallocate(source)

    def call() -> ttnn.Tensor:
        return _layer_call(layer, hidden, slabs, slot_tt, start_tt, end_tt)

    for index, (slot, start, end) in enumerate(eager_chunks):
        stage(slot, start, end)
        output = call()
        record(f"eager{index}-slot{slot}-{start}-{end}", output, start, end)
        ttnn.deallocate(output)

    # Warm twice on the capture's metadata, capture on slot 0 at a head, then replay other slots and bounds.
    stage(0, 0, chunk)
    for _ in range(2):
        ttnn.deallocate(call())
    trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced_output = call()
    ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
    stage_line("captured")
    try:
        for index, (slot, start, end) in enumerate(traced_chunks):
            stage(slot, start, end)
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
            record(f"traced{index}-slot{slot}-{start}-{end}", traced_output, start, end)
    finally:
        ttnn.release_trace(mesh_device, trace)
        ttnn.deallocate(traced_output)

    if mode == "save":
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(records, path)
        stage_line(f"saved {len(records)} records to {path}")
    else:
        golden = torch.load(path)
        assert sorted(golden) == sorted(records), "the golden was taken on another schedule"
        for key, shards in golden.items():
            assert len(shards) == len(records[key]), key
            for chip, (want, got) in enumerate(zip(shards, records[key], strict=True)):
                assert_bit_identical(want, got, name=f"{key} chip {chip}")
        stage_line(f"compared {len(records)} records bit for bit: identical")

    for tensor in (slot_tt, start_tt, end_tt, hidden):
        ttnn.deallocate(tensor)
    slabs.deallocate()
