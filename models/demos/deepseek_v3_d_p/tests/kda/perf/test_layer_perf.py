# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic-weight performance acceptance for Kimi-K3 KDA."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import tp_axis_is_wrapped
from models.demos.deepseek_v3_d_p.tests.kda.kda_test_utils import make_actual_start
from models.demos.deepseek_v3_d_p.tests.kda.utils import (
    deallocate_state,
    is_4x4_subtorus,
    kda_device_params,
    make_kimi_k3_device_case,
    make_synthetic_kimi_k3_test_case,
    upload_slab,
)
from models.demos.deepseek_v3_d_p.tests.measure.gate import MARGIN, gate
from models.demos.deepseek_v3_d_p.tests.measure.trace import TracedStep, measure_host
from models.demos.deepseek_v3_d_p.tt.kda.kda import KdaState, ttKDA
from models.demos.deepseek_v3_d_p.tt.kda.state_adapter import KdaContractGeometry, KdaStates, device_scalar
from models.demos.deepseek_v3_d_p.utils.chunk_config import PREFILL_CHUNK_TOKENS, PREFILL_CHUNK_TOKENS_PER_CHIP
from models.demos.deepseek_v3_d_p.utils.prefill_summary_utils import emit_perf_metrics
from models.demos.deepseek_v3_d_p.utils.smbus_telemetry import require_high_power
from tt_py_test_utils_common.perf.cpu_clock import hold_full_cpu_clock

pytestmark = [
    run_for_blackhole(),
    pytest.mark.perf,
    pytest.mark.timeout(900),
]

_TIMING_SAMPLES = 5
# Host-time targets (ms) per (layout, mode), as tests/measure measures them: untraced fastest of 10 after 10 warm-up,
# traced one replay, both with the CPU clock held. LoudBox SP8xTP1 and QuietBox SP4xTP1 report until a CI run gives
# their targets with the fused relay.
_TARGETS_MS = {
    ("SP8xTP4", "untraced"): 1.205,
    ("SP8xTP4", "traced"): 1.116,
    ("SP4xTP4", "untraced"): 1.23,
    ("SP4xTP4", "traced"): 1.132,
    ("SP4xTP4-subtorus", "untraced"): 1.184,
    ("SP4xTP4-subtorus", "traced"): 1.109,
}
# Galaxy-equivalent layouts outside CI: they report wall time but are not gated.
_REPORT_ONLY_LAYOUTS = ("SP4xTP2", "SP2xTP2", "SP4xTP4-sp-axis0", "SP4xTP4-sp-axis0-subtorus")
# Overrides the chunk's start position for investigation; the performance gates are skipped when it is set.
_ACTUAL_START_OVERRIDE = os.environ.get("KDA_PERF_ACTUAL_START")
_GALAXY_TP = 4
_GALAXY_HEADS_PER_CHIP = KimiK3Config.KDA_NUM_HEADS // _GALAXY_TP
# Galaxy target: a full chunk whose start falls mid-device. A resumed request starts at an arbitrary aligned
# prefix, and every full chunk of it then wraps onto its first rank, whose separated tail the relay applies before it
# sends the final carry, a step a rank-aligned start skips. Production always passes actual_end, so the gate does too.
_GALAXY_ACTUAL_START = 4 * PREFILL_CHUNK_TOKENS_PER_CHIP + 320
# QuietBox SP2xTP2 runs Galaxy's per-chip workload: 640 rows and 24 heads per chip, so the chunk shrinks to 1280
# tokens and the synthetic config halves the global head count. Every per-chip shape, L1 placement and program
# config then matches Galaxy, and the chunk starts 320 rows into a device as on Galaxy. The SP chain is 2 ranks
# instead of 8, and the 11x10 grid splits each head's values over 2 cores instead of 4.
_QUIETBOX_SEQUENCE = 2 * PREFILL_CHUNK_TOKENS_PER_CHIP
_QUIETBOX_NUM_HEADS = 2 * _GALAXY_HEADS_PER_CHIP
_QUIETBOX_ACTUAL_START = PREFILL_CHUNK_TOKENS_PER_CHIP + 320
# QuietBox SP4xTP1 runs the same per-chip workload over a 4-rank SP ring: 2560 tokens of Galaxy's 24 heads per chip.
# With one TP rank the output projection writes its result to DRAM and skips the TP reduce-scatter.
_QUIETBOX_SP4_SEQUENCE = 4 * PREFILL_CHUNK_TOKENS_PER_CHIP
_QUIETBOX_SP4_NUM_HEADS = _GALAXY_HEADS_PER_CHIP
_QUIETBOX_SP4_ACTUAL_START = PREFILL_CHUNK_TOKENS_PER_CHIP + 320
# LoudBox SP4xTP2 runs the same per-chip workload on a 2560-token chunk. The SP chain is 4 ranks and the TP
# reduce-scatter 2; the 11x10 grid splits each head's values over 2 cores as on QuietBox.
_LOUDBOX_SEQUENCE = 4 * PREFILL_CHUNK_TOKENS_PER_CHIP
_LOUDBOX_NUM_HEADS = 2 * _GALAXY_HEADS_PER_CHIP
_LOUDBOX_ACTUAL_START = 2 * PREFILL_CHUNK_TOKENS_PER_CHIP + 320
# LoudBox SP8xTP1 runs Galaxy's SP chain and chunk with Galaxy's 24 heads per chip. Its 8 chips close into a ring, as
# Galaxy's SP axis does, and so do QuietBox SP4xTP1's 4; FABRIC_2D_TORUS_Y rings only mesh axis 0, so SP takes that axis.
# As on QuietBox SP4xTP1, it skips the TP reduce-scatter.
# 4x4 stage: Galaxy's per-chip workload on a 2560-token chunk. "-sp-axis0" swaps SP and TP axes.
_STAGE_4X4_SEQUENCE = 4 * PREFILL_CHUNK_TOKENS_PER_CHIP
_STAGE_4X4_ACTUAL_START = 2 * PREFILL_CHUNK_TOKENS_PER_CHIP + 320
# Like the model's prefill tests, give CCL semaphores their own L1_SMALL region: allocated in L1 on the first call,
# they would land between the chunk terms and fragment L1, so the traced call would plan ops on less free L1 than
# the warm-up compiled for.
_L1_SMALL_SIZE = 1152


def _allocate_state(layer: ttKDA) -> KdaState:
    return layer.allocate_state(batch_size=1)


# The slab layer a gated perf call stores its carry at; any position of a one-layer slab costs the same.
_SLAB_LAYER_IDX = 1


@dataclass
class _Slots:
    """A one-slot, one-layer KDA state slab: what ttKDA.forward reads and writes in place, as in the model."""

    states: KdaStates
    slot: ttnn.Tensor

    @classmethod
    def allocate(cls, mesh_device: ttnn.MeshDevice, config, layer: ttKDA, tensor_parallel_axis: int) -> "_Slots":
        geometry = KdaContractGeometry.from_kda_config(
            config,
            mesh_shape=tuple(mesh_device.shape),
            sp_axis=1 - tensor_parallel_axis,
            tp_axis=tensor_parallel_axis,
        )
        states = KdaStates.allocate(mesh_device, geometry, layer_ids=(_SLAB_LAYER_IDX,), num_slots=1)
        slot = device_scalar(mesh_device, 0)
        # The slabs come uninitialized and the gated chunk starts mid-request, so it reads the slot: zero it first.
        for slab in (states.recurrent, states.convolution):
            upload_slab(slab, [torch.zeros(tuple(slab.shape))] * mesh_device.get_num_devices())
        return cls(states=states, slot=slot)

    def deallocate(self) -> None:
        self.states.deallocate()
        ttnn.deallocate(self.slot)


def _layer_call(
    layer: ttKDA,
    hidden: ttnn.Tensor,
    state: KdaState,
    actual_start: ttnn.Tensor,
    actual_end: ttnn.Tensor | None,
    slots: _Slots | None,
) -> tuple[ttnn.Tensor, KdaState | None]:
    """One layer call: forward on ``state``, or with ``slots`` on the slab entry, as the model runs it (no carry
    returned: it was replaced in the slab)."""
    if slots is None:
        return layer.forward(hidden, state, actual_start, actual_end)
    output, _ = layer.forward(hidden, slots.states.entry(slots.slot, _SLAB_LAYER_IDX), actual_start, actual_end)
    return output, None


def _block_samples_ms(mesh_device: ttnn.MeshDevice, call: Callable[[], object], repetitions: int) -> list[float]:
    """_TIMING_SAMPLES samples of `repetitions` back-to-back calls through one sync, in ms per call."""
    samples = []
    for _ in range(_TIMING_SAMPLES):
        start = time.perf_counter()
        for _ in range(repetitions):
            call()
        ttnn.synchronize_device(mesh_device)
        samples.append((time.perf_counter() - start) * 1e3 / repetitions)
    return samples


def _layer_step(
    mesh_device: ttnn.MeshDevice,
    layer: ttKDA,
    hidden: ttnn.Tensor,
    actual_start: int = 0,
    actual_end: int | None = None,
    slots: _Slots | None = None,
) -> tuple[Callable[[], tuple], KdaState]:
    """One layer call on a fresh state: (the call, returning (output, next state); the state, freed by the caller)."""
    actual_start_tt = make_actual_start(mesh_device, actual_start)
    actual_end_tt = None if actual_end is None else make_actual_start(mesh_device, actual_end)
    state = _allocate_state(layer)
    return lambda: _layer_call(layer, hidden, state, actual_start_tt, actual_end_tt, slots), state


def _trace_wall_samples_ms(
    mesh_device: ttnn.MeshDevice,
    layer: ttKDA,
    hidden: ttnn.Tensor,
    repetitions: int,
    actual_start: int = 0,
    actual_end: int | None = None,
) -> list[float]:
    """Host ms per traced call, for the dev and state-adapter tools: _TIMING_SAMPLES samples of `repetitions`
    back-to-back replays through one sync, after one warm-up call, the capture and one replay."""
    step, state = _layer_step(mesh_device, layer, hidden, actual_start, actual_end)
    try:
        with TracedStep(mesh_device, step) as traced:
            traced.capture()
            traced.replay()
            return _block_samples_ms(mesh_device, lambda: traced.replay(sync=False), repetitions)
    finally:
        deallocate_state(state)


@pytest.mark.parametrize(
    "mesh_device,tensor_parallel_axis,device_params",
    [
        pytest.param(
            (2, 4),
            0,
            kda_device_params((2, 4), l1_small_size=_L1_SMALL_SIZE),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
            id="SP4xTP2-fabric-2d",
        ),
        pytest.param(
            (8, 1),
            1,
            kda_device_params((8, 1), l1_small_size=_L1_SMALL_SIZE),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 1), topology="ring"),
            id="SP8xTP1-ring",
        ),
        pytest.param(
            (2, 2),
            1,
            kda_device_params((2, 2), l1_small_size=_L1_SMALL_SIZE),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 2), topology="mesh-2x2"),
            id="SP2xTP2-fabric-2d",
        ),
        pytest.param(
            (4, 1),
            1,
            kda_device_params((4, 1), l1_small_size=_L1_SMALL_SIZE),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 1), topology="ring"),
            id="SP4xTP1-ring",
        ),
        pytest.param(
            (8, 4),
            1,
            kda_device_params((8, 4), l1_small_size=_L1_SMALL_SIZE),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="SP8xTP4-torus-xy",
        ),
        pytest.param(
            (4, 4),
            0,
            kda_device_params((4, 4), l1_small_size=_L1_SMALL_SIZE),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 4), topology="mesh-4x4"),
            id="SP4xTP4-torus-x-4x4",
        ),
        pytest.param(
            (4, 4),
            1,
            kda_device_params((4, 4), l1_small_size=_L1_SMALL_SIZE),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 4), topology="mesh-4x4"),
            id="SP4xTP4-sp-axis0-4x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_synthetic_kimi_k3_perf(
    request, mesh_device: ttnn.MeshDevice, tensor_parallel_axis: int, device_params: dict
) -> None:
    """Gate checkpoint-free production K3 latency, untraced and traced (Galaxy; LoudBox and QuietBox report).

    The gated call is the model's whole per-chunk KDA layer, ttKDA.forward on its slab entry and the TP-sharded
    residual: TP all-gather, KDA replacing the slot's carry in the slabs, output projection + reduce-scatter.
    """
    mesh_shape = tuple(mesh_device.shape)
    sequence_parallel_axis = 1 - tensor_parallel_axis
    layout = f"SP{mesh_shape[sequence_parallel_axis]}xTP{mesh_shape[tensor_parallel_axis]}"
    if mesh_shape == (4, 4) and sequence_parallel_axis == 0:
        layout += "-sp-axis0"
    if mesh_shape == (4, 4) and is_4x4_subtorus():
        layout += "-subtorus"
    if layout.startswith(("SP8xTP4", "SP4xTP4")):
        require_high_power(f"kimi_k3 KDA {layout} perf")
    sequence, num_heads, actual_start = {
        "SP8xTP4": (PREFILL_CHUNK_TOKENS, None, _GALAXY_ACTUAL_START),
        "SP4xTP2": (_LOUDBOX_SEQUENCE, _LOUDBOX_NUM_HEADS, _LOUDBOX_ACTUAL_START),
        "SP8xTP1": (PREFILL_CHUNK_TOKENS, _GALAXY_HEADS_PER_CHIP, _GALAXY_ACTUAL_START),
        "SP2xTP2": (_QUIETBOX_SEQUENCE, _QUIETBOX_NUM_HEADS, _QUIETBOX_ACTUAL_START),
        "SP4xTP1": (_QUIETBOX_SP4_SEQUENCE, _QUIETBOX_SP4_NUM_HEADS, _QUIETBOX_SP4_ACTUAL_START),
        "SP4xTP4": (_STAGE_4X4_SEQUENCE, None, _STAGE_4X4_ACTUAL_START),
        "SP4xTP4-sp-axis0": (_STAGE_4X4_SEQUENCE, None, _STAGE_4X4_ACTUAL_START),
    }[layout.removesuffix("-subtorus")]
    if _ACTUAL_START_OVERRIDE is not None:
        actual_start = int(_ACTUAL_START_OVERRIDE)
    actual_end = None if actual_start == 0 else actual_start + sequence
    case = make_synthetic_kimi_k3_test_case(sequence=sequence, num_heads=num_heads)
    layer, hidden_tt = make_kimi_k3_device_case(
        mesh_device,
        case,
        tensor_parallel_axis=tensor_parallel_axis,
        cache_weights=False,
        shard_hidden_over_tp=True,
    )
    slots = _Slots.allocate(mesh_device, case.config, layer, tensor_parallel_axis)
    assert (layer.active_seq_len_local, layer.config.num_heads) == (
        PREFILL_CHUNK_TOKENS_PER_CHIP,
        _GALAXY_HEADS_PER_CHIP,
    ), f"{layout} must run Galaxy's per-chip workload"
    step, state = _layer_step(mesh_device, layer, hidden_tt, actual_start, actual_end, slots)
    clock = hold_full_cpu_clock()
    request.addfinalizer(clock.restore)
    try:
        # Each call replaces the slot's carry in the slabs, so the last eager call and the replay start from
        # different carries; at a 5k chunk the carry has converged, so their outputs still agree.
        untraced, traced, trace_pcc = measure_host(mesh_device, lambda: step()[0])  # the carry stays in the slab
    finally:
        deallocate_state(state)
        slots.deallocate()
    # Not gated: layouts outside CI, an overridden start, or a Galaxy whose TP axis is not wrapped.
    gated = layout not in _REPORT_ONLY_LAYOUTS and _ACTUAL_START_OVERRIDE is None
    gated = gated and (layout != "SP8xTP4" or tp_axis_is_wrapped(mesh_device))
    results = {}
    for mode, samples in (("untraced", untraced), ("traced", traced)):
        target = _TARGETS_MS.get((layout, mode)) if gated else None
        results[mode] = (samples, min(samples), target, MARGIN[mode])
        emit_perf_metrics(
            f"kimi_k3_kda_layer_{layout}_{mode}",
            f"Kimi-K3 KDA layer perf ({layout}, {mode})",
            {
                "layout": layout,
                "mode": mode,
                "fabric_config": ttnn.get_fabric_config(mesh_device).name,
                "sequence": sequence,
                "actual_start": actual_start,
                "samples_ms": samples,
                "measured_ms": min(samples),
                "target_ms": target,
                "margin_pct": MARGIN[mode] * 100.0,
            },
        )
    gate(f"kimi_k3 KDA layer {layout}", results, trace_pcc)
