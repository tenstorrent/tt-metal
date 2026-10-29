# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Wormhole T3000 functional tests for all_gather's packet-walk kernels (multicast and Mux V2 unicast factories).
#
# Every case runs on a line (FABRIC_1D) and a ring (FABRIC_1D_RING) over the 1x8 mesh, and with each factory: `factory`
# sets TT_AG_FORCE, a TEST ONLY knob of the scardoza/mux-ag12 build (all_gather_test_knobs.hpp), before the first op on
# a freshly opened mesh with an empty program cache. A forced factory that a legality rule forbids (fewer than 4 worker
# cores, no unicast walk geometry, a bent 2D axis, true 2D) skips with the op's reason, unless the test expects that
# error. Without the knob every factory variant runs the auto choice. With TT_AG_LOG=1 the build prints one
# "AGCFG factory=..." line per program build; each test prints AGCASE_BEGIN/AGCASE_END so a log splits per case.
#
# Correctness is exact: all_gather is a copy, so every device's output must equal the golden (the same dtype/layout
# round trip of the input, which for bfloat8_b/bfloat4_b is the block-float quantization of the input) bit for bit.
#
# Wormhole specifics these cases target: DRAM alignment 32 B (the cut unit is 32 B with DRAM, 16 B L1-only for
# multicast), 12 DRAM banks (slices start mid-row and the walk's cross_row mode rarely fires), the 4352 B default fabric
# payload (and 6144 / 7616 via FabricRouterConfig), and an 8x8 worker grid.

import os
import time
from contextlib import contextmanager

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import skip_for_blackhole
from tests.nightly.t3000.ccl.test_all_gather import create_fabric_router_config

TRACE_REGION = 200_000


def _params(fabric_config, payload=None):
    p = {"fabric_config": fabric_config, "trace_region_size": TRACE_REGION}
    if payload is not None:
        p["fabric_router_config"] = create_fabric_router_config(payload)
    return p


LINE_RING = pytest.mark.parametrize(
    "device_params",
    [_params(ttnn.FabricConfig.FABRIC_1D), _params(ttnn.FabricConfig.FABRIC_1D_RING)],
    indirect=True,
    ids=["line", "ring"],
)
BOTH_FACTORIES = pytest.mark.parametrize("factory", ["unicast", "multicast"])
ALL_FACTORIES = pytest.mark.parametrize("factory", ["auto", "unicast", "multicast"])
MESH_1x8 = pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)

DRAM = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG
TILE = ttnn.TILE_LAYOUT
RM = ttnn.ROW_MAJOR_LAYOUT


@pytest.fixture(autouse=True)
def _agcase_markers(request):
    print(f"\nAGCASE_BEGIN {request.node.nodeid}", flush=True)
    yield
    print(f"\nAGCASE_END {request.node.nodeid}", flush=True)


def _force_factory(monkeypatch, mesh_device, factory):
    # Read at program build and not in the program-cache hash: set it before the first op, with an empty cache.
    if factory == "auto":
        monkeypatch.delenv("TT_AG_FORCE", raising=False)
    else:
        monkeypatch.setenv("TT_AG_FORCE", factory)
    mesh_device.clear_program_cache()
    logger.info(f"factory={factory} TT_AG_FORCE={os.environ.get('TT_AG_FORCE')}")


@contextmanager
def _legal(factory):
    try:
        yield
    except RuntimeError as e:
        if factory != "auto" and "a legality rule requires" in str(e):
            msg = next((line for line in str(e).splitlines() if "TT_AG_FORCE" in line), str(e)[:300])
            pytest.skip(f"forced {factory} is illegal here: {msg.strip()}")
        raise


# ---------------------------------------------------------------------------------------------------------------------
# Memory config helpers (Wormhole: 8x8 worker grid, 12 DRAM banks)
# ---------------------------------------------------------------------------------------------------------------------


def _nd_dram_sharded(shard_shape, num_banks=12):
    # A DRAM bank id is the core's x alone, so the grid is one row of banks.
    cores = ttnn.num_cores_to_corerangeset(num_banks, ttnn.CoreCoord(num_banks, 1), row_wise=True)
    spec = ttnn.NdShardSpec(
        ttnn.Shape(shard_shape), cores, ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D
    )
    return ttnn.MemoryConfig(ttnn.BufferType.DRAM, spec)


def _l1_sharded(memory_layout, shard_height, shard_width, num_cores=None, grid=None):
    if grid is not None:
        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid[0] - 1, grid[1] - 1))})
    else:
        cores = ttnn.num_cores_to_corerangeset(num_cores, ttnn.CoreCoord(8, 8), row_wise=True)
    return ttnn.MemoryConfig(
        memory_layout,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(cores, (shard_height, shard_width), ttnn.ShardOrientation.ROW_MAJOR),
    )


def _dram_width_sharded(shard_height, shard_width, num_banks):
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_banks - 1, 0))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(cores, (shard_height, shard_width), ttnn.ShardOrientation.ROW_MAJOR),
    )


WS = ttnn.TensorMemoryLayout.WIDTH_SHARDED
HS = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
BS = ttnn.TensorMemoryLayout.BLOCK_SHARDED


# ---------------------------------------------------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------------------------------------------------


def _random(shape, dtype, layout):
    if dtype in (ttnn.bfloat16,):
        x = torch.randn(shape, dtype=torch.bfloat16)
    elif dtype in (ttnn.float32, ttnn.bfloat8_b, ttnn.bfloat4_b):
        x = torch.randn(shape, dtype=torch.float32)
    elif dtype == ttnn.int32:
        x = torch.randint(-(2**31), 2**31 - 1, shape, dtype=torch.int32)
    elif dtype == ttnn.uint32:
        x = torch.randint(0, 2**31 - 1, shape, dtype=torch.int64).to(torch.uint32)
    elif dtype == ttnn.uint16:
        x = torch.randint(0, 2**16 - 1, shape, dtype=torch.int32).to(torch.uint16)
    else:
        raise ValueError(dtype)
    # The golden is the input after the same dtype/layout conversion, so the CCL must reproduce it exactly.
    return ttnn.to_torch(ttnn.from_torch(x, dtype=dtype, layout=layout))


def _mapper(mesh_device, dim, cluster_axis):
    if cluster_axis is None:
        return ttnn.ShardTensorToMesh(mesh_device, dim=dim)
    dims = (None, dim) if cluster_axis == 1 else (dim, None)
    return ttnn.ShardTensor2dMesh(mesh_device, dims=dims, mesh_shape=tuple(mesh_device.shape))


def _check(tensor, golden, what):
    for d, device_tensor in enumerate(ttnn.get_device_tensors(tensor)):
        got = ttnn.to_torch(device_tensor)
        assert got.shape == golden.shape, f"{what}: device {d} shape {tuple(got.shape)} != {tuple(golden.shape)}"
        if not torch.equal(got, golden):
            bad = (got != golden).nonzero()
            raise AssertionError(
                f"{what}: device {d} mismatch, {bad.shape[0]} of {golden.numel()} elements differ; first at "
                f"{bad[0].tolist()}, last at {bad[-1].tolist()}"
            )


class Case:
    """One gather: the input sharded over the gathered axis, a persistent output (or none), and helpers to run it."""

    def __init__(
        self,
        mesh_device,
        shape,
        dim,
        dtype,
        layout,
        mem_in,
        mem_out,
        cluster_axis=None,
        persistent=True,
        sub_core_grids=None,
    ):
        self.mesh_device = mesh_device
        self.shape, self.dim, self.dtype, self.layout = shape, dim, dtype, layout
        self.mem_in, self.mem_out = mem_in, mem_out
        self.out_mem = mem_out if mem_out is not None else mem_in
        self.cluster_axis = cluster_axis
        self.sub_core_grids = sub_core_grids
        self.golden = _random(shape, dtype, layout)
        self.input = ttnn.from_torch(
            self.golden,
            device=mesh_device,
            dtype=dtype,
            layout=layout,
            memory_config=mem_in,
            mesh_mapper=_mapper(mesh_device, dim, cluster_axis),
        )
        self.output = self._replicated(torch.zeros(shape)) if persistent else None

    def _replicated(self, x):
        return ttnn.from_torch(
            x,
            device=self.mesh_device,
            dtype=self.dtype,
            layout=self.layout,
            memory_config=self.out_mem,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def op(self, output=None):
        return ttnn.all_gather(
            self.input,
            dim=self.dim,
            cluster_axis=self.cluster_axis,
            memory_config=self.mem_out,
            output_tensor=output if output is not None else self.output,
            sub_core_grids=self.sub_core_grids,
        )

    def reload(self):
        """Fresh data into the same input buffer and zeros into the persistent output (for trace replays)."""
        self.golden = _random(self.shape, self.dtype, self.layout)
        host_in = ttnn.from_torch(
            self.golden,
            dtype=self.dtype,
            layout=self.layout,
            memory_config=self.mem_in,
            mesh_mapper=_mapper(self.mesh_device, self.dim, self.cluster_axis),
        )
        ttnn.copy_host_to_device_tensor(host_in, self.input)
        if self.output is not None:
            host_zero = ttnn.from_torch(
                torch.zeros(self.shape),
                dtype=self.dtype,
                layout=self.layout,
                memory_config=self.out_mem,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            ttnn.copy_host_to_device_tensor(host_zero, self.output)


def _delays(mesh_device, stalled=(0, 0), cycles=400_000):
    rows, cols = tuple(mesh_device.shape)
    delays = [[0] * cols for _ in range(rows)]
    delays[stalled[0]][stalled[1]] = cycles
    delays[rows - 1][cols - 1] = 2 * cycles
    return delays


def run_case(case, factory, *, iters=1, trace_replays=2):
    """Compile + check, `iters` more launches with device skew (each checked), then a traced replay of fresh data."""
    mesh_device = case.mesh_device
    with _legal(factory):
        result = case.op()
        ttnn.synchronize_device(mesh_device)
    _check(result, case.golden, "first launch")

    for i in range(iters):
        ttnn.apply_device_delay(mesh_device, _delays(mesh_device, stalled=(0, i % mesh_device.shape[1])))
        result = case.op()
        ttnn.synchronize_device(mesh_device)
        _check(result, case.golden, f"skewed launch {i}")

    if trace_replays and case.output is not None:
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        traced = case.op()
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)
        case.reload()
        for _ in range(trace_replays):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        ttnn.release_trace(mesh_device, trace_id)
        _check(traced, case.golden, "trace replay")


# ---------------------------------------------------------------------------------------------------------------------
# 1. Ports of the Blackhole page-walk tests
# ---------------------------------------------------------------------------------------------------------------------

# On WH a line is always multicast; a ring runs unicast from 64 input pages per device (20 for pages >= 4 KB), so the
# "small" shapes stay multicast and the "large" ones go unicast under auto, as on BH. With 12 DRAM banks the walk's
# cross_row mode does not fire at N=8, so every slice that starts mid-row takes the head path.
PAGE_WALK_CASES = [
    # id, shape, dim, layout, mem_in, mem_out
    ("mid_row_small", [1, 1, 160, 2816], 3, TILE, DRAM, DRAM),  # 11 x 5 tiles per device: 55 pages
    ("mid_row_large", [1, 1, 1792, 2816], 3, TILE, DRAM, DRAM),  # 11 x 56 tiles per device
    ("rm_256B_rows", [1, 1, 2048, 1024], 3, RM, DRAM, DRAM),
    ("rm_dim0", [8, 1, 256, 4096], 0, RM, DRAM, DRAM),
    # BH's ND-sharded output over 12 WH banks: 64 shards round-robin, so stripes do not align to banks.
    (
        "nd_dram_bh_shape_12_banks",
        [1, 1, 512, 8192],
        3,
        TILE,
        _nd_dram_sharded([1, 1, 64, 1024]),
        _nd_dram_sharded([1, 1, 64, 1024]),
    ),
    # Each device's stripe in one bank on 12 banks: one shard row (8 shards, banks 0-7) ...
    (
        "nd_dram_stripe_per_bank_dim3",
        [1, 1, 64, 8192],
        3,
        TILE,
        _nd_dram_sharded([1, 1, 64, 1024]),
        _nd_dram_sharded([1, 1, 64, 1024]),
    ),
    # ... or a dim-2 gather with one shard per device stripe.
    (
        "nd_dram_stripe_per_bank_dim2",
        [1, 1, 768, 1024],
        2,
        TILE,
        _nd_dram_sharded([1, 1, 96, 1024]),
        _nd_dram_sharded([1, 1, 96, 1024]),
    ),
    # The 8x8 WH grid holds BH's 64 height shards as is.
    ("l1_height_sharded_in", [1, 1, 2048, 1024], 3, TILE, _l1_sharded(HS, 32, 128, 64), DRAM),
    ("rm_reshard_3_to_4_small", [1, 1, 64, 768], 3, RM, DRAM, _l1_sharded(WS, 64, 128, 6)),
    ("rm_reshard_3_to_4_large", [1, 1, 2048, 768], 3, RM, DRAM, _l1_sharded(WS, 2048, 128, 6)),
]


@skip_for_blackhole("Wormhole T3000 test")
@MESH_1x8
@LINE_RING
@ALL_FACTORIES
@pytest.mark.parametrize(
    "shape, dim, layout, mem_in, mem_out", [c[1:] for c in PAGE_WALK_CASES], ids=[c[0] for c in PAGE_WALK_CASES]
)
def test_wh_page_walk(mesh_device, monkeypatch, factory, shape, dim, layout, mem_in, mem_out):
    _force_factory(monkeypatch, mesh_device, factory)
    run_case(Case(mesh_device, shape, dim, ttnn.bfloat16, layout, mem_in, mem_out), factory, iters=2, trace_replays=3)


# Back-to-back gathers into one persistent output with no sync between them, every device but the first stalled after
# each launch: a fast device's next gather must not write into a neighbor's output before that neighbor has relayed or
# copied out the previous one (the init handshake). Small is multicast under auto, large unicast on the ring.
@skip_for_blackhole("Wormhole T3000 test")
@MESH_1x8
@LINE_RING
@BOTH_FACTORIES
@pytest.mark.parametrize("shape", [[1, 1, 128, 1024], [1, 1, 1024, 2048]], ids=["small", "large"])
def test_wh_persistent_output_overlap(mesh_device, monkeypatch, factory, shape):
    _force_factory(monkeypatch, mesh_device, factory)
    num_iters = 10
    mapper = ttnn.ShardTensorToMesh(mesh_device, dim=3)
    goldens = [torch.randn(shape, dtype=torch.bfloat16) for _ in range(num_iters)]
    inputs = [ttnn.from_torch(g, device=mesh_device, layout=TILE, mesh_mapper=mapper) for g in goldens]

    def replicated_zeros():
        return ttnn.from_torch(
            torch.zeros(shape, dtype=torch.bfloat16),
            device=mesh_device,
            layout=TILE,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    output = replicated_zeros()
    captures = [replicated_zeros() for _ in range(num_iters)]
    with _legal(factory):
        ttnn.all_gather(inputs[0], dim=3, output_tensor=output)  # compile: every launch below is cached
    ttnn.synchronize_device(mesh_device)

    rows, cols = tuple(mesh_device.shape)
    delays = [[0 if (r, c) == (0, 0) else 2_000_000 for c in range(cols)] for r in range(rows)]
    for i in range(num_iters):
        result = ttnn.all_gather(inputs[i], dim=3, output_tensor=output)
        ttnn.apply_device_delay(mesh_device, delays)
        ttnn.copy(result, captures[i])
    ttnn.synchronize_device(mesh_device)

    for i, capture in enumerate(captures):
        _check(capture, goldens[i], f"iteration {i}")


# A persistent output in another memory than the memory_config (unset, so the input's L1): the op builds for the output
# it writes. 16 B L1 rows into 128 B DRAM rows only multicast can carry (the unicast relay would re-read its DRAM output
# at 32 B), at a size that would otherwise pick unicast on the ring. Forcing unicast must fail the legality rule.
@skip_for_blackhole("Wormhole T3000 test")
@MESH_1x8
@LINE_RING
@ALL_FACTORIES
def test_wh_persistent_output_memory(mesh_device, monkeypatch, expect_error, factory):
    _force_factory(monkeypatch, mesh_device, factory)
    # BH uses 65536 rows; on WH writing a 65536 x 128 B row-major tensor replicated to DRAM hangs in the host write
    # (enqueue_write_shards; main too), so 16384 rows: still far above the ring's unicast threshold.
    shape = [1, 1, 16384, 64]
    golden = torch.randn(shape, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(
        golden, device=mesh_device, layout=RM, memory_config=L1, mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3)
    )
    output = ttnn.from_torch(
        torch.zeros(shape, dtype=torch.bfloat16),
        device=mesh_device,
        layout=RM,
        memory_config=DRAM,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    if factory == "unicast":
        with expect_error(RuntimeError, "a legality rule requires multicast"):
            ttnn.all_gather(tt_input, dim=3, output_tensor=output)
        return
    ttnn.all_gather(tt_input, dim=3, output_tensor=output)
    ttnn.synchronize_device(mesh_device)
    _check(output, golden, "persistent DRAM output")


# Pages that do not divide the packet: fp32 tiles (4096 B) against the 4352 B default payload, so packets cut tiles, and
# fp32 row-major rows (8192 B) larger than a packet (and than the 4352 B payload's unit-rounded cap).
NON_SCATTER_CASES = [
    # id, shape, dim, layout, submesh width
    ("fp32_tile_straddles_packets_4dev", [1, 1, 32, 512], 3, TILE, 4),
    ("fp32_rm_page_larger_than_packet_4dev", [1, 1, 32, 8192], 3, RM, 4),
    ("fp32_tile_straddles_packets_8dev", [1, 1, 32, 1024], 3, TILE, 8),
    # 8 devices x 8192 B would make 64 KB output rows (beyond the readback page limit): gather 8192 B rows on dim 2.
    ("fp32_rm_page_larger_than_packet_8dev_dim2", [1, 1, 256, 2048], 2, RM, 8),
]


@skip_for_blackhole("Wormhole T3000 test")
@MESH_1x8
@LINE_RING
@ALL_FACTORIES
@pytest.mark.parametrize(
    "shape, dim, layout, width", [c[1:] for c in NON_SCATTER_CASES], ids=[c[0] for c in NON_SCATTER_CASES]
)
def test_wh_multicast_non_scatter_pages(mesh_device, monkeypatch, factory, shape, dim, layout, width, device_params):
    if width != 8 and device_params["fabric_config"] == ttnn.FabricConfig.FABRIC_1D_RING:
        pytest.skip("a 1x4 submesh of the T3000 perimeter is not a ring")
    _force_factory(monkeypatch, mesh_device, factory)
    target = mesh_device if width == 8 else mesh_device.create_submesh(ttnn.MeshShape(1, width))
    target.clear_program_cache()
    run_case(Case(target, shape, dim, ttnn.float32, layout, DRAM, DRAM), factory, iters=1, trace_replays=0)


# ---------------------------------------------------------------------------------------------------------------------
# 2. Wormhole-only paths
# ---------------------------------------------------------------------------------------------------------------------

WH_CASES = [
    # id, shape, dim, dtype, layout, mem_in, mem_out
    # Row-major rows that are 32 B multiples but not 64 B ones: native on WH (cut unit 32 B), composite on BH.
    ("rm_concat_32B_rows", [1, 1, 256, 128], 3, ttnn.bfloat16, RM, DRAM, DRAM),
    ("rm_concat_96B_rows", [1, 1, 256, 384], 3, ttnn.bfloat16, RM, DRAM, DRAM),
    ("rm_concat_160B_rows", [1, 1, 256, 640], 3, ttnn.bfloat16, RM, DRAM, DRAM),
    ("rm_96B_to_128B_dram_width_sharded", [1, 1, 64, 384], 3, ttnn.bfloat16, RM, DRAM, _dram_width_sharded(64, 64, 6)),
    ("rm_96B_to_128B_l1_width_sharded", [1, 1, 64, 384], 3, ttnn.bfloat16, RM, DRAM, _l1_sharded(WS, 64, 64, 6)),
    ("rm_96B_rows_uint32", [1, 1, 128, 192], 3, ttnn.uint32, RM, DRAM, DRAM),
    ("rm_32B_rows_fp32", [1, 1, 64, 64], 3, ttnn.float32, RM, DRAM, DRAM),
    ("rm_160B_dram_to_l1", [1, 1, 128, 640], 3, ttnn.bfloat16, RM, DRAM, L1),
    ("rm_96B_l1_to_dram", [1, 1, 512, 384], 3, ttnn.bfloat16, RM, L1, DRAM),
    # 48 B L1 rows: the gate cuts them at 16 B (native), the unicast relay's 32 B unit cannot (multicast only).
    ("rm_48B_l1_to_dram_multicast_only", [1, 1, 512, 192], 3, ttnn.bfloat16, RM, L1, DRAM),
    # Equal 48 B pages in DRAM on a dim-2 gather: whole pages at a 64 B pitch (the walk's padded-page branch).
    ("rm_48B_pages_dim2_whole_page", [2, 1, 64, 24], 2, ttnn.bfloat16, RM, DRAM, DRAM),
    # Odd tile counts per device, starting mid-row across 12 banks.
    ("odd_7x3_tiles", [1, 1, 96, 1792], 3, ttnn.bfloat16, TILE, DRAM, DRAM),
    ("odd_13x13_tiles", [1, 1, 416, 3328], 3, ttnn.bfloat16, TILE, DRAM, DRAM),
    ("odd_5x7_tiles_dim2", [1, 1, 1792, 160], 2, ttnn.bfloat16, TILE, DRAM, DRAM),
    ("odd_1_tile", [1, 1, 32, 256], 3, ttnn.bfloat16, TILE, DRAM, DRAM),
    ("odd_3_tiles", [1, 1, 32, 768], 3, ttnn.bfloat16, TILE, DRAM, DRAM),
    ("odd_11x9_tiles_bfp8", [1, 1, 288, 2816], 3, ttnn.bfloat8_b, TILE, DRAM, DRAM),
    ("odd_25x33_tiles", [1, 1, 1056, 6400], 3, ttnn.bfloat16, TILE, DRAM, DRAM),
    # Gather dims 0-3.
    ("dim0_tile", [8, 2, 64, 128], 0, ttnn.bfloat16, TILE, DRAM, DRAM),
    ("dim1_tile", [2, 8, 64, 96], 1, ttnn.bfloat16, TILE, DRAM, DRAM),
    ("dim2_tile", [2, 2, 256, 96], 2, ttnn.bfloat16, TILE, DRAM, DRAM),
    ("dim_neg1_tile", [2, 1, 64, 768], -1, ttnn.bfloat16, TILE, DRAM, DRAM),
    ("dim0_rm_96B", [8, 3, 32, 48], 0, ttnn.bfloat16, RM, DRAM, DRAM),
    ("dim1_rm", [1, 16, 40, 64], 1, ttnn.bfloat16, RM, DRAM, DRAM),
    ("dim2_rm", [2, 1, 512, 96], 2, ttnn.bfloat16, RM, DRAM, DRAM),
    # Dtypes, tile.
    ("tile_bf16", [1, 1, 256, 2048], 3, ttnn.bfloat16, TILE, DRAM, DRAM),
    ("tile_fp32", [1, 1, 256, 2048], 3, ttnn.float32, TILE, DRAM, DRAM),
    ("tile_bfp8", [1, 1, 256, 2048], 3, ttnn.bfloat8_b, TILE, DRAM, DRAM),
    ("tile_bfp4", [1, 1, 256, 2048], 3, ttnn.bfloat4_b, TILE, DRAM, DRAM),
    ("tile_uint32", [1, 1, 256, 2048], 3, ttnn.uint32, TILE, DRAM, DRAM),
    ("tile_int32", [1, 1, 256, 2048], 3, ttnn.int32, TILE, DRAM, DRAM),
    ("tile_uint16", [1, 1, 256, 2048], 3, ttnn.uint16, TILE, DRAM, DRAM),
    ("tile_bfp8_dim2", [1, 1, 2048, 256], 2, ttnn.bfloat8_b, TILE, DRAM, DRAM),
    # Dtypes, row-major.
    ("rm_bf16", [1, 1, 64, 1024], 3, ttnn.bfloat16, RM, DRAM, DRAM),
    ("rm_fp32", [1, 1, 64, 1024], 3, ttnn.float32, RM, DRAM, DRAM),
    ("rm_uint32", [1, 1, 64, 1024], 3, ttnn.uint32, RM, DRAM, DRAM),
    ("rm_int32", [1, 1, 64, 1024], 3, ttnn.int32, RM, DRAM, DRAM),
    # Interleaved memories.
    ("l1_to_l1_tile", [1, 1, 256, 1024], 3, ttnn.bfloat16, TILE, L1, L1),
    ("dram_to_l1_tile", [1, 1, 256, 1024], 3, ttnn.bfloat16, TILE, DRAM, L1),
    ("l1_to_dram_tile", [1, 1, 256, 1024], 3, ttnn.bfloat16, TILE, L1, DRAM),
    ("l1_to_l1_rm", [1, 1, 256, 1024], 3, ttnn.bfloat16, RM, L1, L1),
    # Sharded L1 in and out (8x8 grid).
    (
        "width_to_width_tile",
        [1, 1, 64, 2048],
        3,
        ttnn.bfloat16,
        TILE,
        _l1_sharded(WS, 64, 32, 8),
        _l1_sharded(WS, 64, 256, 8),
    ),
    (
        "height_to_height_tile",
        [1, 1, 2048, 256],
        3,
        ttnn.bfloat16,
        TILE,
        _l1_sharded(HS, 32, 32, 64),
        _l1_sharded(HS, 32, 256, 64),
    ),
    (
        "block_to_block_tile",
        [1, 1, 256, 2048],
        3,
        ttnn.bfloat16,
        TILE,
        _l1_sharded(BS, 64, 64, grid=(4, 4)),
        _l1_sharded(BS, 64, 512, grid=(4, 4)),
    ),
    (
        "height_to_width_tile_dim2",
        [1, 1, 2048, 256],
        2,
        ttnn.bfloat16,
        TILE,
        _l1_sharded(HS, 32, 256, 8),
        _l1_sharded(WS, 2048, 32, 8),
    ),
    ("width_rm_to_dram", [1, 1, 32, 1024], 3, ttnn.bfloat16, RM, _l1_sharded(WS, 32, 64, 2), DRAM),
    (
        "height_rm_to_height_rm",
        [1, 1, 1024, 512],
        3,
        ttnn.bfloat16,
        RM,
        _l1_sharded(HS, 16, 64, 64),
        _l1_sharded(HS, 16, 512, 64),
    ),
    ("dram_to_block_tile", [1, 1, 512, 1024], 3, ttnn.bfloat16, TILE, DRAM, _l1_sharded(BS, 128, 256, grid=(4, 4))),
]


@skip_for_blackhole("Wormhole T3000 test")
@MESH_1x8
@LINE_RING
@BOTH_FACTORIES
@pytest.mark.parametrize(
    "shape, dim, dtype, layout, mem_in, mem_out", [c[1:] for c in WH_CASES], ids=[c[0] for c in WH_CASES]
)
def test_wh_paths(mesh_device, monkeypatch, factory, shape, dim, dtype, layout, mem_in, mem_out):
    _force_factory(monkeypatch, mesh_device, factory)
    run_case(Case(mesh_device, shape, dim, dtype, layout, mem_in, mem_out), factory, iters=1, trace_replays=2)


# ---------------------------------------------------------------------------------------------------------------------
# 3. Fabric payload sizes
# ---------------------------------------------------------------------------------------------------------------------

PAYLOAD_CASES = [
    ("tile_bf16_11x32", [1, 1, 1024, 2816], 3, ttnn.bfloat16, TILE),
    ("tile_fp32", [1, 1, 256, 2048], 3, ttnn.float32, TILE),
    ("tile_bfp8", [1, 1, 512, 2048], 3, ttnn.bfloat8_b, TILE),
    ("rm_1KB_rows", [1, 1, 256, 4096], 3, ttnn.bfloat16, RM),
    ("rm_8KB_rows_dim2", [1, 1, 256, 2048], 2, ttnn.float32, RM),
]


@skip_for_blackhole("Wormhole T3000 test")
@MESH_1x8
@pytest.mark.parametrize(
    "device_params",
    [
        _params(fc, payload)
        for payload in (4352, 6144, 7616)
        for fc in (ttnn.FabricConfig.FABRIC_1D, ttnn.FabricConfig.FABRIC_1D_RING)
    ],
    indirect=True,
    ids=[f"{topo}_{payload}" for payload in (4352, 6144, 7616) for topo in ("line", "ring")],
)
@BOTH_FACTORIES
@pytest.mark.parametrize("shape, dim, dtype, layout", [c[1:] for c in PAYLOAD_CASES], ids=[c[0] for c in PAYLOAD_CASES])
def test_wh_payload_sizes(mesh_device, monkeypatch, factory, shape, dim, dtype, layout):
    _force_factory(monkeypatch, mesh_device, factory)
    run_case(Case(mesh_device, shape, dim, dtype, layout, DRAM, DRAM), factory, iters=1, trace_replays=2)


# ---------------------------------------------------------------------------------------------------------------------
# 4. Sub-core grids
# ---------------------------------------------------------------------------------------------------------------------


def _crs(*ranges):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*a), ttnn.CoreCoord(*b)) for a, b in ranges])


SUB_GRIDS = [
    # 2 columns: a direction's mux + 2 workers do not fit one row, so unicast falls back to row-major order.
    ("cols_2", _crs(((0, 0), (1, 7)))),
    # 3 columns: each direction gets its own row.
    ("cols_3", _crs(((0, 0), (2, 7)))),
    ("disjoint", _crs(((1, 0), (3, 6)), ((5, 0), (6, 6)))),
    # 6 cores in a row: unicast 2 workers/direction exactly, one link.
    ("cores_6", _crs(((2, 3), (7, 3)))),
    # 4 cores: unicast scales down to 1 worker/direction (4 mux buffers).
    ("cores_4", _crs(((0, 0), (3, 0)))),
    # 3 cores: fewer than 4 -> multicast only.
    ("cores_3", _crs(((0, 0), (2, 0)))),
    ("cores_1", _crs(((4, 4), (4, 4)))),
]


@skip_for_blackhole("Wormhole T3000 test")
@MESH_1x8
@LINE_RING
@BOTH_FACTORIES
@pytest.mark.parametrize("grid", [g[1] for g in SUB_GRIDS], ids=[g[0] for g in SUB_GRIDS])
@pytest.mark.parametrize(
    "shape, dim, layout",
    [([1, 1, 1024, 2048], 3, TILE), ([1, 1, 416, 3328], 3, TILE), ([1, 1, 512, 384], 3, RM)],
    ids=["tile_8x32", "tile_13x13", "rm_96B"],
)
def test_wh_sub_core_grids(mesh_device, monkeypatch, factory, grid, shape, dim, layout):
    _force_factory(monkeypatch, mesh_device, factory)
    case = Case(mesh_device, shape, dim, ttnn.bfloat16, layout, DRAM, DRAM, sub_core_grids=grid)
    run_case(case, factory, iters=1, trace_replays=2)


# ---------------------------------------------------------------------------------------------------------------------
# 5. Fabric 2D
# ---------------------------------------------------------------------------------------------------------------------

FABRIC_2D_SHAPES = [
    ("tile_bf16", [1, 1, 256, 1024], 3, ttnn.bfloat16, TILE),
    ("tile_bf16_odd", [1, 1, 160, 2816], 3, ttnn.bfloat16, TILE),
    ("rm_bf16_96B", [1, 1, 64, 384], 3, ttnn.bfloat16, RM),
    ("tile_bfp8_dim2", [1, 1, 512, 2048], 2, ttnn.bfloat8_b, TILE),
]


@skip_for_blackhole("Wormhole T3000 test")
@pytest.mark.parametrize(
    "mesh_device, cluster_axis",
    [((2, 4), 0), ((2, 4), 1), ((2, 4), None), ((1, 8), None)],
    indirect=["mesh_device"],
    ids=["2x4_axis0", "2x4_axis1", "2x4_true2d", "1x8_bent"],
)
@pytest.mark.parametrize("device_params", [_params(ttnn.FabricConfig.FABRIC_2D)], indirect=True, ids=["fabric_2d"])
@ALL_FACTORIES
@pytest.mark.parametrize(
    "shape, dim, dtype, layout", [c[1:] for c in FABRIC_2D_SHAPES], ids=[c[0] for c in FABRIC_2D_SHAPES]
)
def test_wh_fabric_2d(mesh_device, monkeypatch, factory, cluster_axis, shape, dim, dtype, layout):
    _force_factory(monkeypatch, mesh_device, factory)
    case = Case(mesh_device, shape, dim, dtype, layout, DRAM, DRAM, cluster_axis=cluster_axis)
    run_case(case, factory, iters=1, trace_replays=2)


# ---------------------------------------------------------------------------------------------------------------------
# 6. Stress
# ---------------------------------------------------------------------------------------------------------------------


# 1000 traced replays of an 8 MB gather (1 MB per device), in 4 rounds: fresh input and a zeroed output before each.
@skip_for_blackhole("Wormhole T3000 test")
@MESH_1x8
@LINE_RING
@BOTH_FACTORIES
@pytest.mark.parametrize(
    "shape, dim, layout",
    [([1, 1, 2048, 2048], 3, TILE), ([1, 1, 4096, 1024], 3, RM)],
    ids=["tile_8MB", "rm_8MB"],
)
def test_wh_stress_trace(mesh_device, monkeypatch, factory, shape, dim, layout):
    _force_factory(monkeypatch, mesh_device, factory)
    case = Case(mesh_device, shape, dim, ttnn.bfloat16, layout, DRAM, DRAM)
    with _legal(factory):
        result = case.op()
        ttnn.synchronize_device(mesh_device)
    _check(result, case.golden, "first launch")
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced = case.op()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    rounds, replays = 4, 250
    for r in range(rounds):
        case.reload()
        ttnn.synchronize_device(mesh_device)
        start = time.perf_counter()
        for _ in range(replays):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        logger.info(f"round {r}: {replays} replays, {(time.perf_counter() - start) / replays * 1e6:.0f} us per gather")
        _check(traced, case.golden, f"round {r} after {replays} replays")
    ttnn.release_trace(mesh_device, trace_id)


# ~200 un-traced calls cycling three shapes (two persistent outputs each, so cache hits rebind addresses) with two
# inputs per shape alternating, device skew every 7th call, every result checked.
@skip_for_blackhole("Wormhole T3000 test")
@MESH_1x8
@LINE_RING
@BOTH_FACTORIES
def test_wh_stress_loop(mesh_device, monkeypatch, factory):
    _force_factory(monkeypatch, mesh_device, factory)
    specs = [
        ([1, 1, 1024, 2048], 3, ttnn.bfloat16, TILE),
        ([1, 1, 416, 3328], 3, ttnn.bfloat16, TILE),
        ([1, 1, 512, 384], 3, ttnn.bfloat16, RM),
    ]
    cases = [[Case(mesh_device, *spec, DRAM, DRAM, persistent=False) for _ in range(2)] for spec in specs]
    outputs = [[cases[s][0]._replicated(torch.zeros(spec[0])) for _ in range(2)] for s, spec in enumerate(specs)]
    with _legal(factory):
        for s in range(len(specs)):
            cases[s][0].op(outputs[s][0])
        ttnn.synchronize_device(mesh_device)
    num_calls = 210
    for i in range(num_calls):
        s, v = i % 3, (i // 3) % 2
        out = outputs[s][(i // 6) % 2]
        if i % 7 == 6:
            ttnn.apply_device_delay(mesh_device, _delays(mesh_device, stalled=(0, (i // 7) % 8), cycles=1_000_000))
        result = cases[s][v].op(out)
        ttnn.synchronize_device(mesh_device)
        _check(result, cases[s][v].golden, f"call {i} (shape {s}, input {v})")
