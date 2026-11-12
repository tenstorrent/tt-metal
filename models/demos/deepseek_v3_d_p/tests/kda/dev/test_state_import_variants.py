# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Which recurrent-state import the SP-sharded K3 KDA layer runs: measured inside ``ttKDA.forward_slot`` (Galaxy SP8xTP4).

The layer imports its slot's 3-head shard and all-gathers it over SP into the 24-head scratch carry. Two candidates
for the recurrent carry (timed alone, as standalone ops, they tied):

* ``copy+all_gather`` (production, ``KdaStates.import_layer``): ``indexed_entry_copy`` slab -> staging shard (zero
  fill at a request head), then native ``ttnn.all_gather`` into the carry (persistent output).
* ``high_bw-slab``: ``high_bw_all_gather(num_links=2)`` reads the shard straight from the ND-sharded slab at the device
  slot (``input_batch_index_tensor``). It has no zero fill, so a second ``indexed_entry_copy`` zero-fills the gathered
  carry at a request head: the carry is its own one-entry source, indexed by ``actual_start`` itself, so the entry is
  in range (and zero-filled, the source never read) only when ``actual_start`` is 0, and out of range (nothing
  written) otherwise. Same launch count as the production import.

The convolution tail imports the same way in both (copy + native all_gather on dim 2). Both run the gated Galaxy chunk
(``test_synthetic_kimi_k3_perf``'s start and bounds), alternated A, B, A, B in one process, each measured as: traced
wall (median of 5 samples of 10 back-to-back replays, the layer gate's metric), traced device (median of 5 single
profiled replays, sum over programs of the slowest chip), untraced wall (the untraced gate's metric) and untraced device
(one profiled eager call). Before timing, both must give bit-identical outputs and slab carries, on a request head and
mid-request. Rows are printed as ``KDA_STATE_IMPORT_ROW <json>``.
"""

from __future__ import annotations

import json
import statistics
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.tests.kda.perf.test_layer_perf import (
    _GALAXY_ACTUAL_START,
    _L1_SMALL_SIZE,
    _layer_call,
    _Slots,
    _trace_device_samples_ms,
    _trace_wall_samples_ms,
    _untraced_wall_samples_ms,
)
from models.demos.deepseek_v3_d_p.tests.kda.utils import (
    deallocate_state,
    kda_device_params,
    make_kimi_k3_device_case,
    make_synthetic_kimi_k3_test_case,
    read_slab_layer,
)
from models.demos.deepseek_v3_d_p.tt.kda.state_adapter import KdaStates, _convolution_branches, _copy_entry
from models.demos.deepseek_v3_d_p.utils.chunk_config import PREFILL_CHUNK_TOKENS
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program_merged, require_realtime_profiler
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import assert_bit_identical, make_actual_start

pytestmark = [run_for_blackhole(), pytest.mark.perf, pytest.mark.timeout(2400)]

_TP_AXIS = 1
_REPETITIONS = 10
_ROUNDS = 2
_UNTRACED_DEVICE_SAMPLES = 5
# Both variants' gather programs stay cached in one process, and each parks two 16 B/bank semaphores in L1_SMALL: the
# layer perf gate's pool plus room for the second variant's recurrent gather. Both variants run under this same pool.
_BOTH_VARIANTS_L1_SMALL_SIZE = _L1_SMALL_SIZE + 32


def _import_high_bw_slab(self: KdaStates, destination, slot, layer_idx, *, zero_fill_at_zero=None) -> None:
    """Candidate B: the recurrent shard read straight from the slab by the gather; the conv tail as in production."""
    position = self.layer_position(layer_idx)
    ttnn.experimental.high_bw_all_gather(
        self.recurrent,
        dim=1,
        output_tensor=destination.recurrent,
        cluster_axis=self.sp_axis,
        num_links=2,
        input_batch_index_tensor=slot,
        batch_slot_num_layers=self.num_layers,
        batch_slot_layer_idx=position,
    )
    if zero_fill_at_zero is not None:
        ttnn.experimental.kda.indexed_entry_copy(
            destination.recurrent,
            destination.recurrent,
            zero_fill_at_zero,
            index_scale=1,
            index_offset=0,
            zero_fill_at_zero=zero_fill_at_zero,
        )
    _copy_entry(
        self.convolution, self.staging.convolution, slot, self.num_layers, position, zero_fill_at_zero=zero_fill_at_zero
    )
    ttnn.all_gather(
        _convolution_branches(self.staging.convolution),
        dim=2,
        cluster_axis=self.sp_axis,
        output_tensor=_convolution_branches(destination.convolution),
    )


_VARIANTS = {"copy+all_gather": KdaStates.import_layer, "high_bw-slab": _import_high_bw_slab}


def _host(tensor: ttnn.Tensor) -> list[torch.Tensor]:
    return [ttnn.to_torch(shard).clone() for shard in ttnn.get_device_tensors(tensor)]


@pytest.mark.parametrize(
    "mesh_device,device_params",
    [
        pytest.param(
            (8, 4),
            kda_device_params((8, 4), l1_small_size=_BOTH_VARIANTS_L1_SMALL_SIZE),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="SP8xTP4-torus-xy",
        )
    ],
    indirect=True,
)
def test_recurrent_import_variants_in_forward_slot(mesh_device, device_params, monkeypatch):
    require_realtime_profiler("the in-layer import comparison")
    started = time.monotonic()

    def stage_line(what: str) -> None:
        logger.info(f"[import-variants +{time.monotonic() - started:.0f}s] {what}")

    case = make_synthetic_kimi_k3_test_case(sequence=PREFILL_CHUNK_TOKENS)
    layer, hidden = make_kimi_k3_device_case(
        mesh_device, case, tensor_parallel_axis=_TP_AXIS, cache_weights=False, shard_hidden_over_tp=True
    )
    slots = _Slots.allocate(mesh_device, case.config, layer, _TP_AXIS)
    actual_start, actual_end = _GALAXY_ACTUAL_START, _GALAXY_ACTUAL_START + PREFILL_CHUNK_TOKENS
    stage_line("layer built")

    # 1. Both variants compute the same chunk bit for bit, from the same stored carry: a head and a resumed chunk.
    seed_carry = read_slab_layer(mesh_device, slots.states, 0, 1)
    results = {}
    for name, variant in _VARIANTS.items():
        monkeypatch.setattr(KdaStates, "import_layer", variant)
        for start, end in ((0, PREFILL_CHUNK_TOKENS), (actual_start, actual_end)):
            slots.states.export_layer(seed_carry, slots.slot, 1)
            start_tt, end_tt = make_actual_start(mesh_device, start), make_actual_start(mesh_device, end)
            state = layer.allocate_state(batch_size=1)
            output, _ = _layer_call(layer, hidden, state, start_tt, end_tt, slots)
            carry = read_slab_layer(mesh_device, slots.states, 0, 1)
            results[(name, start)] = (
                _host(output) + _host(state.recurrent) + _host(carry.recurrent) + _host(carry.convolution)
            )
            for tensor in (output, start_tt, end_tt):
                ttnn.deallocate(tensor)
            deallocate_state(state)
            deallocate_state(carry)
    for start in (0, actual_start):
        for index, (want, got) in enumerate(
            zip(results[("copy+all_gather", start)], results[("high_bw-slab", start)], strict=True)
        ):
            assert_bit_identical(want, got, name=f"start {start} tensor {index}")
    deallocate_state(seed_carry)
    stage_line("both variants bit-identical on a head and a resumed chunk")

    # 2. Timing, alternated A, B, A, B.
    rows = []
    for round_index in range(_ROUNDS):
        for name, variant in _VARIANTS.items():
            monkeypatch.setattr(KdaStates, "import_layer", variant)
            traced_wall, _ = _trace_wall_samples_ms(
                mesh_device, layer, hidden, _REPETITIONS, actual_start=actual_start, actual_end=actual_end, slots=slots
            )
            _, traced_device = _trace_device_samples_ms(
                mesh_device, layer, hidden, actual_start=actual_start, actual_end=actual_end, slots=slots
            )
            untraced_wall = _untraced_wall_samples_ms(mesh_device, layer, hidden, actual_start, actual_end, slots)
            start_tt, end_tt = make_actual_start(mesh_device, actual_start), make_actual_start(mesh_device, actual_end)
            state = layer.allocate_state(batch_size=1)

            def eager_call() -> None:
                output, _ = _layer_call(layer, hidden, state, start_tt, end_tt, slots)
                ttnn.synchronize_device(mesh_device)
                ttnn.deallocate(output)

            eager_call()
            untraced_device, launches = [], set()
            for _ in range(_UNTRACED_DEVICE_SAMPLES):
                _, programs = profile_realtime_program_merged(mesh_device, eager_call)
                untraced_device.append(sum(program["duration_ns"] for program in programs.values()) / 1e6)
                launches.add(len(programs))
            for tensor in (start_tt, end_tt):
                ttnn.deallocate(tensor)
            deallocate_state(state)
            row = {
                "variant": name,
                "round": round_index,
                "traced_wall_ms": statistics.median(traced_wall),
                "traced_wall_samples_ms": traced_wall,
                "traced_device_ms": statistics.median(traced_device),
                "traced_device_samples_ms": traced_device,
                "untraced_wall_ms": statistics.median(untraced_wall),
                "untraced_wall_samples_ms": untraced_wall,
                "untraced_device_ms": statistics.median(untraced_device),
                "untraced_device_samples_ms": untraced_device,
                "programs_per_call": sorted(launches),
                "actual_start": actual_start,
                "actual_end": actual_end,
            }
            print("KDA_STATE_IMPORT_ROW " + json.dumps(row, sort_keys=True))
            rows.append(row)
            stage_line(
                f"round {round_index} {name}: traced wall {row['traced_wall_ms']:.3f} device {row['traced_device_ms']:.3f}"
                f" | untraced wall {row['untraced_wall_ms']:.3f} device {row['untraced_device_ms']:.3f} ms"
                f" | programs {row['programs_per_call']}"
            )
    slots.deallocate()
    ttnn.deallocate(hidden)
