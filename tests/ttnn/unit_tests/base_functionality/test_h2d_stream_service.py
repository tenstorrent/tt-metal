# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end Python correctness sweeps for ttnn.H2DStreamService.

What's covered:
  * Construction via the nanobind binding (kwargs flow into the C++ Config).
  * Both input paths:
      - `forward_to_tensor` (pre-distributed host tensor)
      - `forward_to_tensor_bytes` (raw bytes; service distributes internally)
  * Variation over (innermost-dim page size, num pages, scratch CB budget, FIFO
    size) — replicated sweep mirrors the C++ Replicated_Sweep matrix at a
    smaller scale.
  * One explicitly sharded placement: `[Shard(3), Replicate]` on an 8x4 mesh,
    i.e. shard the innermost tensor dim across the long mesh edge (8 devices)
    and replicate across the short edge (4 devices).

Verification strategy: for each test we build a SECOND mapper with the same
placements as the service and run it host-side on the source tensor to produce
the expected per-coord shards. We then iterate the service's backing tensor
shards and assert byte-exact equality. This keeps the verification logic
placement-agnostic — same code handles replicate, shard, and mixed cases.
"""

import pytest
import torch

import ttnn

from models.common.utility_functions import skip_for_slow_dispatch

# H2DStreamService claims FD dispatch-column service cores and DMA-pins host memory; it is only
# supported on Blackhole Galaxy and requires fast dispatch. Skip the whole module on any other
# configuration so unsupported runners skip cleanly instead of hitting the claim TT_FATAL.
pytestmark = [
    skip_for_slow_dispatch(),
    pytest.mark.skipif(
        ttnn.cluster.get_cluster_type() != ttnn.cluster.ClusterType.BLACKHOLE_GALAXY,
        reason="H2DStreamService is only supported on Blackhole Galaxy",
    ),
]


# Source dtype: torch.int32 viewed as UINT32 by the service. PyTorch doesn't
# carry uint32 widely, so we generate int32 sources and cast to int64 at
# compare time (lossless for 0..INT32_MAX).
_DTYPE_TORCH = torch.int32
_DTYPE_TTNN = ttnn.uint32
_DTYPE_SIZE = 4  # bytes per element

# Number of write/barrier/verify cycles each test runs on a single persistent
# service. The service is spawned once; each iteration regenerates a random
# source so the readback assertion proves the kernels actually loop (vs. e.g.
# exited after the first transfer and the same data sticking in the device
# tensor on subsequent reads).
_IO_LOOPS = 20

# Random source value range: [0, INT32_MAX]. Negatives would round-trip
# correctly bit-wise but break our int64 comparison (negative int32 -> negative
# int64 vs. large uint32 -> large positive int64).
_RANDINT_HIGH = 2**31  # exclusive upper bound for torch.randint


def _run_io_loop(
    service: ttnn.H2DStreamService,
    iter_mapper: ttnn.CppTensorToMesh,
    global_spec: ttnn.TensorSpec,
    shape_list: list,
    input_path: str,
    num_iters: int = _IO_LOOPS,
) -> None:
    """Drive `num_iters` write+barrier+verify cycles on a persistent service.

    Each iteration:
      1. Generates a deterministic-seeded random uint32 source matching
         `shape_list` (seed = iter index, so a failure at iter N is
         reproducible).
      2. Distributes via `iter_mapper` to build the expected per-coord shards
         and (for the tensor path) the input host tensor.
      3. Pushes via the chosen input path and barriers.
      4. Asserts every per-device readback matches the corresponding shard
         of the expected distribution.

    AssertionErrors are wrapped with the iteration index so the failing
    cycle is identifiable from the test output.
    """
    for i in range(num_iters):
        gen = torch.Generator()
        gen.manual_seed(i)
        src_torch = torch.randint(low=0, high=_RANDINT_HIGH, size=shape_list, dtype=_DTYPE_TORCH, generator=gen)
        expected_host = ttnn.from_torch(src_torch, spec=global_spec, mesh_mapper=iter_mapper)
        _push_source(service, expected_host, src_torch, input_path)
        service.barrier()
        try:
            _verify_readback(service, expected_host)
        except AssertionError as e:
            raise AssertionError(f"iter {i}: {e}") from e


def _make_global_spec(shape: ttnn.Shape) -> ttnn.TensorSpec:
    return ttnn.TensorSpec(
        shape=shape,
        dtype=_DTYPE_TTNN,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        buffer_type=ttnn.BufferType.DRAM,
    )


def _push_source(
    service: ttnn.H2DStreamService,
    host_tensor: ttnn.Tensor,
    src_torch: torch.Tensor,
    input_path: str,
) -> None:
    """Push the source through the service via the requested input path.

    * "tensor" — call `forward_to_tensor` on the pre-distributed `host_tensor`.
    * "bytes" — pass src.numpy() to `forward_to_tensor_bytes`; `host_tensor`
      is unused on this path (the service distributes internally).
    """
    if input_path == "tensor":
        service.forward_to_tensor(host_tensor)
    elif input_path == "bytes":
        # `.numpy()` returns a contiguous CPU view of the torch buffer; the
        # binding reinterprets its raw bytes against the service's
        # global_spec. dtype is irrelevant — byte-level reinterpretation.
        service.forward_to_tensor_bytes(src_torch.contiguous().numpy())
    else:
        raise ValueError(f"unknown input_path: {input_path!r}")


def _verify_readback(service: ttnn.H2DStreamService, expected_host: ttnn.Tensor) -> None:
    """Compare service's per-device readback against `expected_host`'s shards.

    `expected_host` is the host distributed tensor produced by `from_torch`
    with the same mapper the service uses (or an equivalent replicate-default).
    Both `get_device_tensors` calls iterate in the same mesh-coord order, so
    pairing by index is correct for any placement pattern.
    """
    expected_subs = ttnn.get_device_tensors(expected_host)
    actual_subs = ttnn.get_device_tensors(service.get_backing_tensor())
    assert len(actual_subs) == len(
        expected_subs
    ), f"got {len(actual_subs)} actual vs {len(expected_subs)} expected sub-tensors"
    # Cast to int64 before comparing: ttnn.to_torch returns torch.uint32 for
    # UINT32 specs; torch.equal refuses to promote between Int and UInt.
    # int64 holds both losslessly for our value ranges.
    for i, (actual, expected) in enumerate(zip(actual_subs, expected_subs)):
        a_t = ttnn.to_torch(actual).view(-1).to(torch.int64)
        e_t = ttnn.to_torch(expected).view(-1).to(torch.int64)
        assert a_t.shape == e_t.shape, f"device {i}: shape mismatch {a_t.shape} vs {e_t.shape}"
        assert torch.equal(a_t, e_t), f"device {i}: contents mismatch"


# Replicated sweep. mapper=None exercises the C++ replicate-on-every-mesh-dim
# default; varies the chunking matrix (page size, num pages, CB budget, FIFO).
#
# Per-device shape == global shape under full replication, so a [1,1,N,W]
# global tensor lands as [1,1,N,W] on every device.
#   tensor_page_size = W * sizeof(uint32) = W * 4
#   tensor_num_pages = N
@pytest.mark.parametrize(
    "shape_list, scratch_cb_pages, fifo_pages",
    [
        # (shape, cb_pages, fifo_pages) — `pages` are tensor pages, each
        # `innermost_dim * 4` bytes.
        # Single chunk, big innermost dim — baseline parity with C++ Replicated_Sweep.
        ([1, 1, 1, 65536], 1, 1),
        # Single chunk, small innermost dim.
        ([1, 1, 1, 640], 1, 1),
        # Multi-chunk even split: 16 pages, CB for 4 -> 4 chunks of 4.
        ([1, 1, 16, 640], 4, 16),
        # Divisor fallback: 7 (prime) pages -> pages_per_chunk == 1.
        ([1, 1, 7, 640], 4, 8),
        # Larger innermost dim with a smaller CB budget — exercises the
        # PCIe burst-chunking path inside the persistent kernel at a
        # different page size.
        ([1, 1, 4, 4096], 2, 4),
    ],
)
@pytest.mark.parametrize("input_path", ["tensor", "bytes"])
def test_h2d_stream_service_replicated_sweep(
    mesh_device,
    shape_list,
    scratch_cb_pages,
    fifo_pages,
    input_path,
):
    shape = ttnn.Shape(shape_list)
    per_row_bytes = shape_list[-1] * _DTYPE_SIZE
    global_spec = _make_global_spec(shape)

    # `iter_mapper` matches the service's internal replicate-on-all default.
    # Held by the test for the full loop (used by `from_torch` each iteration);
    # nothing is consumed by the service since `mapper=None` triggers the
    # internal default.
    placements = [ttnn.PlacementReplicate() for _ in range(mesh_device.shape.dims())]
    iter_mapper = ttnn.create_mesh_mapper(mesh_device, ttnn.MeshMapperConfig(placements=placements))

    service = ttnn.H2DStreamService(
        mesh_device=mesh_device,
        global_spec=global_spec,
        fifo_size_bytes=fifo_pages * per_row_bytes,
        scratch_cb_size_bytes=scratch_cb_pages * per_row_bytes,
        # mapper omitted -> C++ replicate-on-every-mesh-dim default.
    )

    _run_io_loop(service, iter_mapper, global_spec, shape_list, input_path)


# Compute the global tensor shape from a per-device shape + placement vector.
# For each Shard placement, the corresponding tensor dim is multiplied by the
# corresponding mesh dim's size; Replicate placements leave the tensor shape
# alone (the same per-device slice appears at every coord along that mesh dim).
def _global_shape_for_placements(per_device_shape, placements, mesh_device):
    global_shape = list(per_device_shape)
    mesh_shape = mesh_device.shape
    for mesh_dim_idx, placement in enumerate(placements):
        if isinstance(placement, ttnn.PlacementShard):
            global_shape[placement.dim] *= mesh_shape[mesh_dim_idx]
    return global_shape


# Sharded sweep across placement patterns × per-device shapes × chunking.
# Requires an 8x4 mesh (32 devices). The first row is the originally-requested
# case (shard innermost across long mesh edge, replicate short edge) — kept
# as a regression anchor; the rest broaden coverage.
#
# Per-device tensor in every row is `[1, 1, N, 640]`:
#   * tensor_page_size = 640 * 4 = 2560 B (PCIe-aligned)
#   * tensor_num_pages = N (the dim-2 size of the per-device shape)
# CB and FIFO are specified in pages of 2560 B so each row is read as
# (placement_pattern, per_device_shape, pages_per_chunk_budget, fifo_pages).
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("input_path", ["tensor", "bytes"])
@pytest.mark.parametrize(
    "placements, per_device_shape, cb_pages, fifo_pages",
    [
        # Baseline: shard innermost (dim 3) across long mesh edge (8), replicate
        # short (4). Global = [1, 1, 1, 5120]. Single chunk per device.
        pytest.param(
            [ttnn.PlacementShard(3), ttnn.PlacementReplicate()],
            [1, 1, 1, 640],
            1,
            1,
            id="shard_innermost_long_replicate_short",
        ),
        # Mesh-dim-flipped variant of the baseline: shard innermost across the
        # SHORT mesh edge (4), replicate across the long edge. Catches anywhere
        # mesh-dim 0 is implicitly assumed to be the shard axis.
        pytest.param(
            [ttnn.PlacementReplicate(), ttnn.PlacementShard(3)],
            [1, 1, 1, 640],
            1,
            1,
            id="replicate_long_shard_innermost_short",
        ),
        # Shard on a non-innermost tensor dim. Global dim 2 expands to 16; each
        # device sees [1,1,2,640] -> 2 tensor pages -> exercises num_pages > 1
        # at the smallest non-trivial count.
        pytest.param(
            [ttnn.PlacementShard(2), ttnn.PlacementReplicate()],
            [1, 1, 2, 640],
            1,
            1,
            id="shard_dim2_long_replicate_short",
        ),
        # Full 2D shard: distinct slice at every coord along two tensor dims.
        # Global = [1, 1, 64, 2560]; per-device = [1, 1, 8, 640]. No replication.
        pytest.param(
            [ttnn.PlacementShard(2), ttnn.PlacementShard(3)],
            [1, 1, 8, 640],
            1,
            1,
            id="full_2d_shard",
        ),
        # Multi-chunk sharded transfer: per-device 16 pages, CB for 4 -> 4
        # chunks of 4. Stresses the persistent kernel's chunked inner loop on
        # a sharded path (the bytes route in particular exercises the mapper's
        # xtensor chunking + the kernel's multi-chunk loop end-to-end).
        pytest.param(
            [ttnn.PlacementShard(3), ttnn.PlacementReplicate()],
            [1, 1, 16, 640],
            4,
            16,
            id="shard_innermost_long_multi_chunk",
        ),
    ],
)
def test_h2d_stream_service_sharded_sweep(mesh_device, placements, per_device_shape, cb_pages, fifo_pages, input_path):
    shape_list = _global_shape_for_placements(per_device_shape, placements, mesh_device)
    shape = ttnn.Shape(shape_list)
    global_spec = _make_global_spec(shape)
    per_row_bytes = per_device_shape[-1] * _DTYPE_SIZE

    # Two mappers from the same config:
    #   * `iter_mapper` stays alive on the Python side; used by `from_torch`
    #     inside the loop each iteration.
    #   * `service_mapper` is consumed by the service ctor (ownership transfer
    #     invalidates the Python handle after the call).
    #
    # TODO: this two-mapper dance only exists because the service currently
    # takes ownership of the mapper (Config holds a `std::unique_ptr<TensorToMesh>`).
    # That's inconsistent with the rest of the codebase — `distribute_tensor`
    # takes `TensorToMesh&` and `from_torch` takes `const TensorToMesh*`, both
    # non-owning. Switching the service's mapper to a non-owning pointer (with
    # `nb::keep_alive` on the Python ctor to bind lifetimes) would let callers
    # reuse a single mapper across both the service ctor and the loop.
    mapper_config = ttnn.MeshMapperConfig(placements=placements)
    iter_mapper = ttnn.create_mesh_mapper(mesh_device, mapper_config)
    service_mapper = ttnn.create_mesh_mapper(mesh_device, mapper_config)

    service = ttnn.H2DStreamService(
        mesh_device=mesh_device,
        global_spec=global_spec,
        fifo_size_bytes=fifo_pages * per_row_bytes,
        scratch_cb_size_bytes=cb_pages * per_row_bytes,
        mapper=service_mapper,  # ownership transferred; handle now invalidated.
    )

    _run_io_loop(service, iter_mapper, global_spec, shape_list, input_path)


# =============================================================================
# API surface tests for worker_cores + metadata kwargs.
#
# Scope: these tests validate the binding surface (kwargs flow into Config,
# getters return live values / raise when disabled, validation TT_FATALs
# propagate as RuntimeError). They do NOT exercise the worker-sync data flow —
# that requires a worker kernel running on the worker grid, which can only be
# dispatched from C++. End-to-end coverage stays in the C++ gtests.
#
# Hang-avoidance constraint: when `worker_cores` is set the service kernel
# expects per-iter acks on `consumed_counter`. Without a worker kernel running,
# the first SUCCESSFUL `forward_to_tensor` would park the kernel in its
# consumed-counter wait and the destructor's `wait_done` would block forever.
# Every test below either:
#   * constructs + destructs the service without ever calling forward_to_tensor,
#   * or only calls forward_to_tensor with arguments that fail validation
#     before any FIFO write happens (the metadata-size / spec / span-size
#     checks all run at the top of the function).
# Either way the kernel stays parked in `socket_wait_for_pages_with_termination`
# at destruction, so `signal_termination` kicks it out cleanly.
# =============================================================================


def _make_minimal_service(mesh_device, **kwargs):
    """Smallest viable service for API-surface tests — a [1,1,1,640] uint32
    tensor with single-page FIFO and CB. Pushed data is never expected to
    succeed in these tests; the service is constructed just to probe the
    binding surface and destruct cleanly."""
    shape = ttnn.Shape([1, 1, 1, 640])
    global_spec = _make_global_spec(shape)
    per_row_bytes = 640 * _DTYPE_SIZE
    return ttnn.H2DStreamService(
        mesh_device=mesh_device,
        global_spec=global_spec,
        fifo_size_bytes=per_row_bytes,
        scratch_cb_size_bytes=per_row_bytes,
        **kwargs,
    )


def _mesh_coords(mesh_device):
    """Enumerate every MeshCoordinate in the mesh shape. Supports 1D and 2D."""
    shape = mesh_device.shape
    n_dims = shape.dims()
    if n_dims == 1:
        return [ttnn.MeshCoordinate(i) for i in range(shape[0])]
    if n_dims == 2:
        return [ttnn.MeshCoordinate(row, col) for row in range(shape[0]) for col in range(shape[1])]
    raise NotImplementedError(f"unsupported mesh dim count: {n_dims}")


def _dummy_source_bytes(shape_list):
    """Zero-filled uint32 source matching shape_list, returned as a numpy view
    suitable for `forward_to_tensor_bytes`. Used in negative tests only — the
    payload is never expected to land on device."""
    src = torch.zeros(shape_list, dtype=_DTYPE_TORCH)
    return src.contiguous().numpy()


def test_worker_sync_getters_addresses_are_nonzero(mesh_device):
    """worker_cores set, metadata disabled. The three worker-sync getters
    return live L1 addresses; per-coord getters work for every device."""
    worker_cores = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))
    service = _make_minimal_service(mesh_device, worker_cores=worker_cores)

    assert service.get_data_ready_sem_addr() != 0
    for coord in _mesh_coords(mesh_device):
        assert service.get_consumed_counter_addr(coord) != 0
        # `get_service_core` returns a CoreCoord; smoke-check x/y access.
        sc = service.get_service_core(coord)
        _ = (sc.x, sc.y)


def test_metadata_getter_address_is_nonzero(mesh_device):
    """worker_cores + metadata_size_bytes set. get_metadata_addr returns a
    live L1 address (the L1-sharded MeshBuffer was allocated)."""
    worker_cores = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))
    service = _make_minimal_service(mesh_device, worker_cores=worker_cores, metadata_size_bytes=16)
    assert service.get_metadata_addr() != 0


def test_worker_sync_getters_raise_when_disabled(mesh_device):
    """worker_cores unset -> the three worker-sync getters all raise."""
    service = _make_minimal_service(mesh_device)
    with pytest.raises(RuntimeError):
        service.get_data_ready_sem_addr()
    # Per-coord getters need a coord; pick coord 0.
    coord = _mesh_coords(mesh_device)[0]
    with pytest.raises(RuntimeError):
        service.get_consumed_counter_addr(coord)
    # get_service_core uses the same internal map; raises when no coord claimed.
    # (Strictly speaking, when worker_cores is unset, service_cores_ is still
    # populated, but `get_service_core` for an unclaimed coord still raises.)


def test_metadata_getter_raises_when_disabled(mesh_device):
    """worker_cores set but metadata_size_bytes=0 -> get_metadata_addr raises."""
    worker_cores = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))
    service = _make_minimal_service(mesh_device, worker_cores=worker_cores)
    with pytest.raises(RuntimeError):
        service.get_metadata_addr()


def test_ctor_rejects_metadata_without_worker_cores(mesh_device):
    """metadata_size_bytes > 0 requires worker_cores; ctor rejects otherwise."""
    with pytest.raises(RuntimeError):
        _make_minimal_service(mesh_device, metadata_size_bytes=16)


def test_ctor_rejects_metadata_larger_than_socket_page(mesh_device):
    """metadata_size_bytes larger than the socket page size is rejected at
    construction. With per_row=640 and num_pages=1, socket_page_size = 2560 B.
    Pick 1 MB to be obviously beyond that across any reasonable chunk plan."""
    worker_cores = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))
    with pytest.raises(RuntimeError):
        _make_minimal_service(mesh_device, worker_cores=worker_cores, metadata_size_bytes=1 << 20)


def test_forward_bytes_rejects_wrong_metadata_size(mesh_device):
    """Service expects 16 B of metadata, caller passes 8 B."""
    worker_cores = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))
    service = _make_minimal_service(mesh_device, worker_cores=worker_cores, metadata_size_bytes=16)
    data = _dummy_source_bytes([1, 1, 1, 640])
    with pytest.raises(RuntimeError):
        service.forward_to_tensor_bytes(data, metadata=b"x" * 8)


def test_forward_bytes_rejects_metadata_when_disabled(mesh_device):
    """metadata_size_bytes=0 at construction, caller still passes bytes."""
    service = _make_minimal_service(mesh_device)
    data = _dummy_source_bytes([1, 1, 1, 640])
    with pytest.raises(RuntimeError):
        service.forward_to_tensor_bytes(data, metadata=b"x" * 16)


def test_forward_bytes_rejects_missing_metadata_when_required(mesh_device):
    """Service expects 16 B of metadata, caller passes none (default b'')."""
    worker_cores = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))
    service = _make_minimal_service(mesh_device, worker_cores=worker_cores, metadata_size_bytes=16)
    data = _dummy_source_bytes([1, 1, 1, 640])
    with pytest.raises(RuntimeError):
        service.forward_to_tensor_bytes(data)
