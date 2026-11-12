# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Golden of the Kimi-K3 KDA layer's per-chunk outputs and carries, for a storage change that must not move a bit.

The production-dimension synthetic K3 layer (Galaxy SP8xTP4) runs ``ttKDA.forward_slot`` over a two-slot, one-layer
slab on a schedule of chunks, eager and then as one captured trace replayed with other slots and bounds. After every
chunk it records, on every chip: the layer output, the scratch carry the import produced, and every slot's carry read
back through the slab import. ``KDA_LAYER_GOLDEN=save:<file>`` writes them; ``KDA_LAYER_GOLDEN=compare:<file>``
requires every tensor to match the saved one bit for bit. Take the golden at the commit before the change and compare
at the change; no mode of the old storage needs to stay alive for it.

The schedule moves the chronologically first SP rank (``(actual_start // rows) % SP``) and the final-state owner (the
rank holding ``actual_end - 1``) across ranks, with request heads (``actual_start == 0``: the import loads zeros), a
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
from models.demos.deepseek_v3_d_p.tests.kda.utils import (
    kda_device_params,
    make_kimi_k3_device_case,
    make_synthetic_kimi_k3_test_case,
    read_slab_layer,
)
from models.demos.deepseek_v3_d_p.tests.kimi_k3.kda_slabs import fill_slabs_with_noise
from models.demos.deepseek_v3_d_p.tt.kda.state_adapter import (
    KdaStates,
    allocate_native_state,
    deallocate_state,
    device_scalar,
)
from models.demos.deepseek_v3_d_p.tt.kimi_k3.kda_geometry import kimi_k3_kda_geometry
from models.demos.deepseek_v3_d_p.utils.chunk_config import PREFILL_CHUNK_TOKENS, PREFILL_CHUNK_TOKENS_PER_CHIP
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import assert_bit_identical

pytestmark = [run_for_blackhole(), pytest.mark.timeout(1800)]

_GOLDEN_ENV = "KDA_LAYER_GOLDEN"
_SP_AXIS, _TP_AXIS = 0, 1
_LAYER = 1
_NUM_SLOTS = 2
_ROWS = PREFILL_CHUNK_TOKENS_PER_CHIP
# (slot, actual_start, actual_end) -> first rank, final-state owner (SP8, 640 rows per rank).
_EAGER_CHUNKS = [
    (1, 0, PREFILL_CHUNK_TOKENS),  # head on slot 1: zeros, first 0, owner 7
    (0, 4 * _ROWS + 320, 4 * _ROWS + 320 + PREFILL_CHUNK_TOKENS),  # split tail: first 4, owner 4 (wraps)
    (1, 3 * _ROWS, 3 * _ROWS + PREFILL_CHUNK_TOKENS - 1000),  # continues slot 1: first 3, owner 1
    (0, 5 * _ROWS + 40, 5 * _ROWS + 140),  # short interval inside the first rank: first 5, owner 5
    (1, 7 * _ROWS + 100, 9 * _ROWS + 300),  # short, split tail across the wrap: first 7, owner 1
    (0, 0, _ROWS + 60),  # short head on slot 0: first 0, owner 1
]
# Captured once on slot 0 at a head, replayed with these metadata.
_TRACED_CHUNKS = [
    (1, 2 * _ROWS + 17, 2 * _ROWS + 17 + PREFILL_CHUNK_TOKENS),  # first 2, owner 2
    (0, 6 * _ROWS, 6 * _ROWS + 3 * _ROWS),  # first 6, owner 0
    (1, 0, 3 * _ROWS + 5),  # head on slot 1 in a replay: first 0, owner 3
    (0, _ROWS + 600, _ROWS + 630),  # short split tail: first 1, owner 1
]


def _golden_mode() -> tuple[str, str]:
    value = os.environ.get(_GOLDEN_ENV, "")
    mode, _, path = value.partition(":")
    if mode not in ("save", "compare") or not path:
        pytest.skip(f"set {_GOLDEN_ENV}=save:<file> or compare:<file>")
    return mode, path


def _host(tensor: ttnn.Tensor) -> list[torch.Tensor]:
    return [ttnn.to_torch(shard).clone() for shard in ttnn.get_device_tensors(tensor)]


@pytest.mark.parametrize(
    "mesh_device,device_params",
    [
        pytest.param(
            (8, 4),
            kda_device_params((8, 4), trace_region_size=64 * 1024 * 1024),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="SP8xTP4-torus-xy",
        )
    ],
    indirect=True,
)
def test_layer_outputs_and_carries_match_golden(mesh_device, device_params):
    mode, path = _golden_mode()
    started = time.monotonic()

    def stage_line(what: str) -> None:
        logger.info(f"[golden {mode} +{time.monotonic() - started:.0f}s] {what}")

    case = make_synthetic_kimi_k3_test_case(sequence=PREFILL_CHUNK_TOKENS)
    layer, hidden = make_kimi_k3_device_case(
        mesh_device, case, tensor_parallel_axis=_TP_AXIS, cache_weights=False, shard_hidden_over_tp=True
    )
    stage_line("layer built")
    geometry = kimi_k3_kda_geometry(tuple(mesh_device.shape), _SP_AXIS, _TP_AXIS)
    slabs = KdaStates.allocate(mesh_device, geometry, sp_axis=_SP_AXIS, layer_ids=(_LAYER,), num_slots=_NUM_SLOTS)
    scratch = allocate_native_state(mesh_device, geometry)
    fill_slabs_with_noise(mesh_device, slabs, seed=23)
    records: dict[str, list[torch.Tensor]] = {}

    def record(label: str, output: ttnn.Tensor | None) -> None:
        """``output`` None: before the first call, when the scratch is still uninitialized memory."""
        ttnn.synchronize_device(mesh_device)
        if output is not None:
            records[f"{label}/output"] = _host(output)
            records[f"{label}/scratch.recurrent"] = _host(scratch.recurrent)
            records[f"{label}/scratch.convolution"] = _host(scratch.convolution)
        for slot in range(_NUM_SLOTS):
            carry = read_slab_layer(mesh_device, slabs, slot, _LAYER)
            records[f"{label}/slot{slot}.recurrent"] = _host(carry.recurrent)
            records[f"{label}/slot{slot}.convolution"] = _host(carry.convolution)
            deallocate_state(carry)
        stage_line(f"recorded {label}")

    record("noise", None)
    slot_tt, start_tt, end_tt = (device_scalar(mesh_device, value) for value in (0, 0, PREFILL_CHUNK_TOKENS))

    def stage(slot: int, start: int, end: int) -> None:
        for destination, value in ((slot_tt, slot), (start_tt, start), (end_tt, end)):
            source = device_scalar(mesh_device, value)
            ttnn.copy(source, destination)
            ttnn.deallocate(source)

    def call() -> ttnn.Tensor:
        return layer.forward_slot(hidden, scratch, slabs, slot_tt, _LAYER, actual_start=start_tt, actual_end=end_tt)

    for index, (slot, start, end) in enumerate(_EAGER_CHUNKS):
        stage(slot, start, end)
        output = call()
        record(f"eager{index}-slot{slot}-{start}-{end}", output)
        ttnn.deallocate(output)

    # Warm twice on the capture's metadata, capture on slot 0 at a head, then replay other slots and bounds.
    stage(0, 0, PREFILL_CHUNK_TOKENS)
    for _ in range(2):
        ttnn.deallocate(call())
    trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced_output = call()
    ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
    stage_line("captured")
    try:
        for index, (slot, start, end) in enumerate(_TRACED_CHUNKS):
            stage(slot, start, end)
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
            record(f"traced{index}-slot{slot}-{start}-{end}", traced_output)
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
    deallocate_state(scratch)
    slabs.deallocate()
