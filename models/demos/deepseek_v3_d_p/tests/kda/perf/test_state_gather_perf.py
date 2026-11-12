# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""All-gather variants for the SP-sharded Kimi-K3 KDA carry, against the gather roofline (Galaxy SP8xTP4).

Sharded over SP, each chip stores ``24 / SP = 3`` of its TP column's 24 KDA heads and the layer gathers them back
over the SP axis (mesh axis 0, a ring on Galaxy's TorusXY fabric) before its unchanged 24-head forward:

* recurrent: ``[1, 3, 128, 128]`` FP32 TILE per chip -> ``[1, 24, 128, 128]`` on dim 1 (S = 196608 B);
* convolution tail: the branch-major ``[1, 144, 192]`` BF16 ROW_MAJOR tail viewed as ``[3, 48, 192]``; a chip's
  shard is ``[3, 6, 192]`` and the gather on dim 1 yields the native order (S = 6912 B).

Every configuration is timed with the real-time device profiler: the device time of one call is the sum over its
programs of the slowest chip's duration, as the median over samples after a discarded first one, for a traced lone
call, traced back-to-back calls and an untraced call (see ``_measure``). Each configuration's outputs are overwritten
with a sentinel and must come back bit-exact on every chip.
Rows are logged as ``KDA_STATE_GATHER_ROW <json>`` with the roofline terms of ``state_gather_roofline``.

The ``slab`` variants read the shard straight from a multi-slot slab with ``high_bw_all_gather``'s trace-safe
``input_batch_index_tensor`` (slot = user * layers + layer, read on device), fusing the import copy into the gather.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.tests.kda.perf.state_gather_roofline import (
    BLACKHOLE_LINK_HARDWARE_GB_PER_S,
    GatherRoofline,
    gather_roofline,
)
from models.demos.deepseek_v3_d_p.tests.kda.utils import kda_device_params
from models.demos.deepseek_v3_d_p.tt.kda.state_adapter import nd_dram_memory_config
from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_num_links, per_axis_topology
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program_merged, require_realtime_profiler

pytestmark = [run_for_blackhole(), pytest.mark.perf, pytest.mark.timeout(1200)]

_SP_AXIS = 0
_SP = 8
_TP = 4
_HEADS_PER_COLUMN = KimiK3Config.KDA_NUM_HEADS // _TP
_HEAD_DIM = 128
_CONV_BRANCHES = 3
_CONV_HALVES = 2
_CONV_PAGE_WIDTH = 192
_TIMING_SAMPLES = 15
_BACK_TO_BACK_CALLS = 10
# Slab variants: 2 users x 2 KDA layers; the gather reads user 1, layer 1 (flat slot 3), so the on-device
# recomposition both multiplies and adds.
_SLAB_USERS = 2
_SLAB_LAYERS = 2
_SLAB_USER = 1
_SLAB_LAYER = 1
# The recurrent slab's ND-DRAM shard (one 16 KiB migration segment), as KdaStates allocates it.
_RECURRENT_SLAB_SHARD = [1, 1, _HEAD_DIM, 32]
_L1_SMALL_SIZE = 1152
_TRACE_REGION_SIZE = 200_000
# Comma-separated configuration names to run instead of the whole survey (the device test ladder's small rungs).
_CONFIG_FILTER_ENV = "KDA_STATE_GATHER_CONFIGS"
_CONFIG_FILTER = tuple(name for name in os.environ.get(_CONFIG_FILTER_ENV, "").split(",") if name)


@dataclass(frozen=True)
class StateShape:
    """One KDA carry as a per-chip shard, its gather dim, and the global host tensor that shards into it.

    Dim 0 is the slot: 1 for a plain shard, ``slots`` for a slab the gather reads one slot of. The host tensor puts
    SP on ``gather_dim`` and TP on the innermost dim, so chip (r, c) holds rows ``[r * shard, (r + 1) * shard)`` of
    ``gather_dim`` from TP slice ``c``, and its gathered output is TP slice ``c`` of the selected slot.
    """

    name: str
    shard_shape: tuple[int, ...]
    gather_dim: int
    dtype: ttnn.DataType
    torch_dtype: torch.dtype
    layout: ttnn.Layout
    element_bytes: int

    @property
    def shard_bytes(self) -> int:
        count = 1
        for size in self.shard_shape:
            count *= size
        return count * self.element_bytes

    def global_shape(self, sp: int, tp: int, slots: int) -> tuple[int, ...]:
        shape = list(self.shard_shape)
        shape[0] = slots
        shape[self.gather_dim] *= sp
        shape[-1] *= tp
        return tuple(shape)

    @property
    def mesh_dims(self) -> tuple[int, int]:
        """Tensor dims sharded over mesh axis 0 (SP) and mesh axis 1 (TP)."""
        return (self.gather_dim, len(self.shard_shape) - 1)

    @property
    def output_shape(self) -> list[int]:
        """Gathered shape on every chip (one slot)."""
        output = list(self.shard_shape)
        output[self.gather_dim] *= _SP
        return output


RECURRENT = StateShape(
    name="recurrent",
    shard_shape=(1, _HEADS_PER_COLUMN // _SP, _HEAD_DIM, _HEAD_DIM),
    gather_dim=1,
    dtype=ttnn.float32,
    torch_dtype=torch.float32,
    layout=ttnn.TILE_LAYOUT,
    element_bytes=4,
)
CONVOLUTION = StateShape(
    name="convolution",
    shard_shape=(1, _CONV_BRANCHES, _HEADS_PER_COLUMN // _SP * _CONV_HALVES, _CONV_PAGE_WIDTH),
    gather_dim=2,
    dtype=ttnn.bfloat16,
    torch_dtype=torch.bfloat16,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    element_bytes=2,
)


@dataclass(frozen=True)
class GatherConfig:
    """One variant and knob setting. ``links`` is the link count the roofline charges (the op's own, or the
    discovered count where the op picks it)."""

    name: str
    variant: str  # "all_gather" | "all_gather_async" | "high_bw_all_gather"
    kwargs: dict = field(default_factory=dict)
    persistent_output: bool = True
    external_semaphores: bool = False
    slab: str | None = None  # None | "interleaved" | "nd"


def _configs(shape: StateShape) -> list[GatherConfig]:
    # high_bw_all_gather's selected-slot path and all_gather_async's broadcast path both require every dim between the
    # leading one and the gather dim to be 1 (each rank's shard one contiguous output run). The branch-major
    # convolution tail [1, 3, 6, 192] gathered on dim 2 puts each rank in 3 runs, so both reject it (TT_FATAL
    # "selected-batch path requires singleton dimensions between batch and dim" / "Broadcast all-gather currently only
    # supports gather dims whose preceding page-ordered dimensions are singleton"); they run on the recurrent shard only.
    contiguous_shard = all(size == 1 for size in shape.shard_shape[1 : shape.gather_dim])
    configs = [
        GatherConfig("all_gather", "all_gather", persistent_output=False),
        GatherConfig("all_gather-persistent", "all_gather"),
        GatherConfig("high_bw-links1", "high_bw_all_gather", {"num_links": 1}),
        GatherConfig("high_bw-links2", "high_bw_all_gather", {"num_links": 2}),
        GatherConfig("high_bw-links2-extsem", "high_bw_all_gather", {"num_links": 2}, external_semaphores=True),
    ]
    if contiguous_shard:
        configs += [
            GatherConfig("high_bw-links2-slab", "high_bw_all_gather", {"num_links": 2}, slab="interleaved"),
            GatherConfig("high_bw-links2-ndslab", "high_bw_all_gather", {"num_links": 2}, slab="nd"),
        ]
    configs += [
        GatherConfig("async-links1", "all_gather_async", {"num_links": 1}),
        GatherConfig("async-links2", "all_gather_async", {"num_links": 2}),
    ]
    if contiguous_shard:
        configs.append(
            GatherConfig("async-links2-broadcast", "all_gather_async", {"num_links": 2, "use_broadcast": True})
        )
    configs += [
        GatherConfig("async-links2-workers1", "all_gather_async", {"num_links": 2, "num_workers_per_link": 1}),
        GatherConfig("async-links2-workers4", "all_gather_async", {"num_links": 2, "num_workers_per_link": 4}),
        GatherConfig("async-links2-buffers2", "all_gather_async", {"num_links": 2, "num_buffers_per_channel": 2}),
        GatherConfig("async-links2-buffers8", "all_gather_async", {"num_links": 2, "num_buffers_per_channel": 8}),
        GatherConfig("async-links2-cps1", "all_gather_async", {"num_links": 2, "chunks_per_sync": 1}),
        GatherConfig("async-links2-cps16", "all_gather_async", {"num_links": 2, "chunks_per_sync": 16}),
    ]
    return configs


def _sp_axis_is_wrapped(mesh_device: ttnn.MeshDevice) -> bool:
    """Whether the realized fabric rings the SP axis (column 0): the far row sits one hop behind on a ring."""
    rows, cols = tuple(mesh_device.shape)
    mesh_shapes = ttnn.get_physical_mesh_shapes()
    assert len(mesh_shapes) == 1, f"expected a single local mesh, got {mesh_shapes}"
    mesh_id = ttnn.MeshId(next(iter(mesh_shapes)))
    node = lambda row: ttnn.FabricNodeId(mesh_id, row * cols)  # noqa: E731 -- column 0
    forward = ttnn.get_eth_forwarding_direction(node(0), node(rows - 1))
    backward = ttnn.get_eth_forwarding_direction(node(1), node(0))
    assert forward is not None and backward is not None, "no SP route along column 0"
    return forward == backward


def _host_input(shape: StateShape, sp: int, tp: int, slots: int) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(shape.global_shape(sp, tp, slots), dtype=torch.float32).to(shape.torch_dtype)


def _expected_output(host: torch.Tensor, shape: StateShape, column: int, slot: int) -> torch.Tensor:
    width = shape.shard_shape[-1]
    return host[slot : slot + 1, ..., column * width : (column + 1) * width]


def _to_mesh(mesh_device: ttnn.MeshDevice, host: torch.Tensor, shape: StateShape, dims, memory_config) -> ttnn.Tensor:
    return ttnn.from_torch(
        host,
        dtype=shape.dtype,
        layout=shape.layout,
        device=mesh_device,
        memory_config=memory_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=dims, mesh_shape=tuple(mesh_device.shape)),
    )


def _sentinel(mesh_device: ttnn.MeshDevice, shape: StateShape) -> ttnn.Tensor:
    """A host tensor of NaN-free sentinels in the gathered shape, to overwrite an output after capture."""
    return ttnn.from_torch(
        torch.full(shape.output_shape, -7.0, dtype=shape.torch_dtype),
        dtype=shape.dtype,
        layout=shape.layout,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _allocate_output(mesh_device: ttnn.MeshDevice, shape: StateShape) -> ttnn.Tensor:
    return ttnn.from_torch(
        torch.zeros(shape.output_shape, dtype=shape.torch_dtype),
        dtype=shape.dtype,
        layout=shape.layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _device_scalar(mesh_device: ttnn.MeshDevice, value: int) -> ttnn.Tensor:
    return ttnn.from_torch(
        torch.tensor([[[[value]]]], dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        device=mesh_device,
    )


def _full_grid(mesh_device: ttnn.MeshDevice) -> ttnn.CoreRangeSet:
    grid = mesh_device.compute_with_storage_grid_size()
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])


def _build_issue(
    mesh_device: ttnn.MeshDevice, shape: StateShape, config: GatherConfig, sp: int, tp: int
) -> tuple[Callable[[], ttnn.Tensor], torch.Tensor, int, list]:
    """Inputs, persistent resources and the call for ``config``. Returns (issue, host input, slot, owned tensors).

    ``issue`` returns the output tensor; the owned list keeps every device resource alive until the caller drops it.
    """
    slots = _SLAB_USERS * _SLAB_LAYERS if config.slab else 1
    slot = _SLAB_USER * _SLAB_LAYERS + _SLAB_LAYER if config.slab else 0
    host = _host_input(shape, sp, tp, slots)
    dims = shape.mesh_dims
    memory_config = (
        nd_dram_memory_config(mesh_device, _RECURRENT_SLAB_SHARD) if config.slab == "nd" else ttnn.DRAM_MEMORY_CONFIG
    )
    tt_input = _to_mesh(mesh_device, host, shape, dims, memory_config)
    gather_dim = shape.gather_dim
    output = _allocate_output(mesh_device, shape) if config.persistent_output else None
    owned: list = [tt_input, output]

    if config.variant == "all_gather":

        def issue():
            return ttnn.all_gather(tt_input, dim=gather_dim, cluster_axis=_SP_AXIS, output_tensor=output)

    elif config.variant == "high_bw_all_gather":
        kwargs = dict(config.kwargs)
        if config.external_semaphores:
            grid = _full_grid(mesh_device)
            kwargs["ready_semaphore"] = ttnn.create_global_semaphore(mesh_device, grid, 0, ttnn.BufferType.L1_SMALL)
            kwargs["data_valid_semaphore"] = ttnn.create_global_semaphore(
                mesh_device, grid, 0, ttnn.BufferType.L1_SMALL
            )
            owned += [kwargs["ready_semaphore"], kwargs["data_valid_semaphore"]]
        if config.slab:
            user = _device_scalar(mesh_device, _SLAB_USER)
            owned.append(user)
            kwargs.update(
                input_batch_index_tensor=user, batch_slot_num_layers=_SLAB_LAYERS, batch_slot_layer_idx=_SLAB_LAYER
            )

        def issue():
            ttnn.experimental.high_bw_all_gather(
                tt_input, dim=gather_dim, output_tensor=output, cluster_axis=_SP_AXIS, **kwargs
            )
            return output

    elif config.variant == "all_gather_async":
        grid = _full_grid(mesh_device)
        semaphores = [ttnn.create_global_semaphore(mesh_device, grid, 0) for _ in range(2)]
        barrier = ttnn.create_global_semaphore(mesh_device, grid, 0)
        owned += [*semaphores, barrier]
        topology = per_axis_topology()[_SP_AXIS]

        def issue():
            return ttnn.experimental.all_gather_async(
                tt_input,
                persistent_output_buffer=output,
                dim=gather_dim,
                multi_device_global_semaphore=semaphores,
                topology=topology,
                cluster_axis=_SP_AXIS,
                barrier_semaphore=barrier,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                **config.kwargs,
            )

    else:
        raise ValueError(f"unknown variant {config.variant}")
    return issue, host, slot, owned


def _profile_replay(mesh_device: ttnn.MeshDevice, trace_id: int) -> list[float]:
    """Replay a trace once under the profiler; return each program's slowest-chip duration (ns) in dispatch order."""
    ttnn.synchronize_device(mesh_device)

    def replay():
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)

    _, programs = profile_realtime_program_merged(mesh_device, replay)
    return [programs[runtime_id]["duration_ns"] for runtime_id in sorted(programs)]


def _kernel_names(mesh_device: ttnn.MeshDevice, trace_id: int) -> list[str]:
    ttnn.synchronize_device(mesh_device)

    def replay():
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)

    _, programs = profile_realtime_program_merged(mesh_device, replay)
    return sorted({src.rsplit("/", 1)[-1] for program in programs.values() for src in program["kernel_sources"]})


def _capture(mesh_device: ttnn.MeshDevice, issue: Callable[[], ttnn.Tensor], calls: int) -> tuple[int, ttnn.Tensor]:
    """Capture ``calls`` back-to-back calls; return the trace and the last call's output."""
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    try:
        outputs = [issue() for _ in range(calls)]
    finally:
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    return trace_id, outputs[-1]


def _count_mismatching_chips(output: ttnn.Tensor, host: torch.Tensor, shape: StateShape, slot: int, tp: int) -> int:
    """Compare every chip's gathered output with its TP column's slice; return the number of mismatching chips."""
    mismatches = 0
    for index, device_tensor in enumerate(ttnn.get_device_tensors(output)):
        column = index % tp
        actual = ttnn.to_torch(device_tensor)
        expected = _expected_output(host, shape, column, slot)
        if actual.shape != expected.shape or not torch.equal(actual, expected):
            mismatches += 1
    return mismatches


def _profile_untraced(
    mesh_device: ttnn.MeshDevice, issue: Callable[[], ttnn.Tensor]
) -> tuple[list[float], ttnn.Tensor]:
    """Dispatch one untraced call under the profiler; return its programs' slowest-chip durations and its output."""
    ttnn.synchronize_device(mesh_device)

    def call():
        output = issue()
        ttnn.synchronize_device(mesh_device)
        return output

    output, programs = profile_realtime_program_merged(mesh_device, call)
    return [programs[runtime_id]["duration_ns"] for runtime_id in sorted(programs)], output


def _untraced_wall_ns(mesh_device: ttnn.MeshDevice, issue: Callable[[], ttnn.Tensor]) -> list[float]:
    """Host wall time of one untraced call through device synchronization (dispatch + device), per sample."""
    samples = []
    for _ in range(_TIMING_SAMPLES):
        ttnn.synchronize_device(mesh_device)
        start = time.perf_counter_ns()
        issue()
        ttnn.synchronize_device(mesh_device)
        samples.append(time.perf_counter_ns() - start)
    return samples


def _measure(
    mesh_device: ttnn.MeshDevice, shape: StateShape, config: GatherConfig, roofline_for, hardware_links: int
) -> dict:
    """Time ``config`` four ways:

    * single: one call per trace replay, as a lone gather between unrelated ops; every chip enters it cold;
    * back-to-back: ``_BACK_TO_BACK_CALLS`` calls per replay, the first dropped, so each timed call starts right after
      the previous gather left the ring in step;
    * untraced: one eagerly dispatched call, device time (profiler) and host wall time through synchronization.

    Utilization and GB/s are against the roofline at the discovered (hardware) link count for every row, so rows rank
    by distance from the hardware bound; ``config_ideal_us`` is the ideal at the config's own link count.
    """
    sp, tp = tuple(mesh_device.shape)
    issue, host, slot, owned = _build_issue(mesh_device, shape, config, sp, tp)
    issue()  # compile and establish persistent resources outside the capture
    ttnn.synchronize_device(mesh_device)
    issue()
    ttnn.synchronize_device(mesh_device)
    single_trace, single_output = _capture(mesh_device, issue, 1)
    batch_trace, batch_output = _capture(mesh_device, issue, _BACK_TO_BACK_CALLS)
    sentinel = _sentinel(mesh_device, shape)
    try:
        # The replays alone must produce the outputs: overwrite them with a sentinel first.
        ttnn.copy_host_to_device_tensor(sentinel, single_output)
        _profile_replay(mesh_device, single_trace)  # exclude first-replay jitter
        single = [_profile_replay(mesh_device, single_trace) for _ in range(_TIMING_SAMPLES)]
        mismatches = _count_mismatching_chips(single_output, host, shape, slot, tp)
        kernels = _kernel_names(mesh_device, single_trace)
        ttnn.copy_host_to_device_tensor(sentinel, batch_output)
        _profile_replay(mesh_device, batch_trace)
        batch = [_profile_replay(mesh_device, batch_trace) for _ in range(_TIMING_SAMPLES)]
        mismatches += _count_mismatching_chips(batch_output, host, shape, slot, tp)
    finally:
        ttnn.release_trace(mesh_device, single_trace)
        ttnn.release_trace(mesh_device, batch_trace)
    _profile_untraced(mesh_device, issue)  # exclude the first eager call after the traces
    untraced = []
    for _ in range(_TIMING_SAMPLES):
        if config.persistent_output:
            ttnn.copy_host_to_device_tensor(sentinel, single_output)
        programs, untraced_output = _profile_untraced(mesh_device, issue)
        untraced.append(programs)
    mismatches += _count_mismatching_chips(untraced_output, host, shape, slot, tp)
    untraced_wall_ns = _untraced_wall_ns(mesh_device, issue)
    launch_counts = sorted({len(programs) for programs in single})
    assert len(launch_counts) == 1, f"{config.name}: program count varies across replays: {launch_counts}"
    launches = launch_counts[0]
    single_ns = [sum(programs) for programs in single]
    back_to_back_ns = []
    for programs in batch:
        assert len(programs) == launches * _BACK_TO_BACK_CALLS, f"{config.name}: {len(programs)} programs per batch"
        per_call = [sum(programs[i : i + launches]) for i in range(0, len(programs), launches)]
        back_to_back_ns += per_call[1:]
    assert all(len(programs) == launches for programs in untraced), f"{config.name}: untraced program count differs"
    untraced_ns = [sum(programs) for programs in untraced]
    single_median = statistics.median(single_ns)
    back_to_back_median = statistics.median(back_to_back_ns)
    untraced_median = statistics.median(untraced_ns)
    config_links = config.kwargs.get("num_links") or hardware_links
    roofline: GatherRoofline = roofline_for(hardware_links)
    del owned, single_output, batch_output, untraced_output
    return {
        "shape": shape.name,
        "config": config.name,
        "variant": config.variant,
        "knobs": dict(config.kwargs),
        "persistent_output": config.persistent_output,
        "external_semaphores": config.external_semaphores,
        "slab": config.slab,
        "launches": launches,
        "single_us": single_median / 1e3,
        "single_us_p10_p90": _percentiles_us(single_ns),
        "back_to_back_us": back_to_back_median / 1e3,
        "back_to_back_us_p10_p90": _percentiles_us(back_to_back_ns),
        "untraced_us": untraced_median / 1e3,
        "untraced_us_p10_p90": _percentiles_us(untraced_ns),
        "untraced_wall_us": statistics.median(untraced_wall_ns) / 1e3,
        "untraced_wall_us_p10_p90": _percentiles_us(untraced_wall_ns),
        "roofline_links": hardware_links,
        "ideal_us": roofline.ideal_ns / 1e3,
        "bound": roofline.bound,
        "config_links": config_links,
        "config_ideal_us": roofline_for(config_links).ideal_ns / 1e3,
        "single_utilization": roofline.utilization(single_median),
        "back_to_back_utilization": roofline.utilization(back_to_back_median),
        "untraced_utilization": roofline.utilization(untraced_median),
        "single_gb_per_s": roofline.effective_gb_per_s(single_median),
        "back_to_back_gb_per_s": roofline.effective_gb_per_s(back_to_back_median),
        "bit_exact": mismatches == 0,
        "mismatching_chips": mismatches,
        "kernels": kernels,
    }


def _percentiles_us(samples_ns: list[float]) -> list[float]:
    ordered = sorted(samples_ns)
    pick = lambda q: ordered[min(len(ordered) - 1, int(q * len(ordered)))] / 1e3  # noqa: E731
    return [round(pick(0.1), 2), round(pick(0.9), 2)]


@pytest.mark.parametrize("shape", [RECURRENT, CONVOLUTION], ids=lambda shape: shape.name)
@pytest.mark.parametrize(
    "mesh_device,device_params",
    [
        pytest.param(
            (8, 4),
            kda_device_params((8, 4), l1_small_size=_L1_SMALL_SIZE, trace_region_size=_TRACE_REGION_SIZE),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="SP8xTP4-torus-xy",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_kda_state_sp_gather_perf(mesh_device: ttnn.MeshDevice, device_params: dict, shape: StateShape):
    """Time every all-gather variant and knob setting for one KDA carry against the roofline; require bit-exact."""
    require_realtime_profiler("the KDA state gather perf survey")
    sp, tp = tuple(mesh_device.shape)
    assert (sp, tp) == (_SP, _TP)
    assert per_axis_topology()[_SP_AXIS] == ttnn.Topology.Ring
    assert _sp_axis_is_wrapped(mesh_device), "the fabric did not ring the SP axis"
    discovered_links = get_num_links(mesh_device, _SP_AXIS)
    fabric_2d = "2D" in ttnn.get_fabric_config().name

    def roofline_for(links: int) -> GatherRoofline:
        return gather_roofline(
            shard_bytes=shape.shard_bytes, num_ranks=sp, num_links=links, wraps=True, fabric_2d=fabric_2d
        )

    logger.info(
        f"KDA state gather {shape.name}: shard {list(shape.shard_shape)} {shape.dtype} = {shape.shard_bytes} B, "
        f"fabric {ttnn.get_fabric_config().name}, SP links discovered {discovered_links}, "
        f"hardware link {BLACKHOLE_LINK_HARDWARE_GB_PER_S} GB/s (model uses the in-repo 25 GB/s)"
    )
    configs = _configs(shape)
    if _CONFIG_FILTER:
        configs = [config for config in configs if config.name in _CONFIG_FILTER]
        assert configs, f"{_CONFIG_FILTER_ENV}={_CONFIG_FILTER} selects no {shape.name} configuration"
    rows, failures = [], []
    for config in configs:
        logger.info(f"KDA state gather {shape.name}/{config.name}: start")
        try:
            row = _measure(mesh_device, shape, config, roofline_for, discovered_links)
        except RuntimeError as error:
            # TT_FATAL puts its location on the first line and the reason on the next ones.
            reason = " | ".join(line.strip() for line in str(error).strip().splitlines()[:4]) or repr(error)
            row = {"shape": shape.name, "config": config.name, "variant": config.variant, "error": reason[:600]}
            failures.append(config.name)
            ttnn.synchronize_device(mesh_device)
        rows.append(row)
        logger.info(f"KDA_STATE_GATHER_ROW {json.dumps(row)}")

    header = (
        f"{'config':26s} {'launch':>6s} {'ideal us':>8s} | {'single':>7s} {'util':>6s} | {'b2b':>7s} {'util':>6s} "
        f"{'GB/s':>6s} | {'untraced':>8s} {'wall':>7s} | exact"
    )
    lines = [header]
    for row in rows:
        if "error" in row:
            lines.append(f"{row['config']:26s} ERROR {row['error'][:160]}")
            continue
        lines.append(
            f"{row['config']:26s} {row['launches']:6d} {row['ideal_us']:8.2f} | {row['single_us']:7.2f} "
            f"{row['single_utilization']:6.1%} | {row['back_to_back_us']:7.2f} {row['back_to_back_utilization']:6.1%} "
            f"{row['back_to_back_gb_per_s']:6.1f} | {row['untraced_us']:8.2f} {row['untraced_wall_us']:7.1f} | "
            f"{row['bit_exact']}"
        )
    logger.info("KDA state gather summary ({}):\n{}".format(shape.name, "\n".join(lines)))
    inexact = [row["config"] for row in rows if "error" not in row and not row["bit_exact"]]
    assert not inexact, f"{shape.name}: outputs not bit-exact for {inexact}"
    assert not failures, f"{shape.name}: configurations failed: {failures}"
