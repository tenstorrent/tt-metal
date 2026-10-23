# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""ttnn.generic_op with a Metal 2.0 ProgramSpec (ttnn.experimental.metal2).

The op under test is an elementwise bf16 multiply (reader, compute, writer over two input rings
and one output ring, tiles split over the grid); its kernels live in kernels/toy_spec_mul/. The
tests pin the contract of the spec overload: correctness, the program cache (a byte-identical
spec is a hit, a hit re-applies this call's tensor addresses and runtime arg values, and a spec
reused with other tensors or incomplete run args is caught), tensor argument validation, aliased
and single-tensor io, preparation without dispatch, and the structural hash the cache keys on.
"""

from pathlib import Path

import pytest
import torch

import ttnn
from ttnn.experimental import metal2 as m2

KERNEL_DIR = Path(__file__).parent / "kernels" / "toy_spec_mul"

DFB_A, DFB_B, DFB_OUT = "in_a", "in_b", "out"
K_READER, K_COMPUTE, K_WRITER = "reader", "compute", "writer"
TP_A, TP_B, TP_OUT = "a", "b", "out"


@pytest.fixture(autouse=True)
def _validate_program_args():
    """Run-arg validation is off by default; the spec path's cache-hit revalidation is what these tests pin.

    A test that wants the default sets the flag to False itself; the fixture restores it either way.
    """
    previous = ttnn.CONFIG.validate_program_args
    ttnn.CONFIG.validate_program_args = True
    yield
    ttnn.CONFIG.validate_program_args = previous


def _split_tiles(num_tiles, grid_size, *, tile_limit=None, cores=None):
    """(all_cores, [(core, num_tiles, start_id), ...]), row-major over the grid.

    The core set comes from the FULL tile count (or the given cores) and the per-core counts from
    the (possibly limited) work, so tile_limit changes only runtime arg VALUES and leaves the
    ProgramSpec, and therefore the program cache key, byte-identical. Cores past the limit get
    zero tiles.
    """
    if cores is None:
        _, cores, *_ = ttnn.split_work_to_cores(grid_size, num_tiles, row_wise=True)
    work = num_tiles if tile_limit is None else min(tile_limit, num_tiles)
    _, _, group_1, group_2, per_core_1, per_core_2 = ttnn.split_work_to_cores(grid_size, work, row_wise=True)
    counts = {}
    for group, per_core in ((group_1, per_core_1), (group_2, per_core_2)):
        if per_core:
            for core in ttnn.corerange_to_cores(group, None, True):
                counts[core] = per_core
    assignment, start = [], 0
    for core in ttnn.corerange_to_cores(cores, None, True):
        assignment.append((core, counts.get(core, 0), start))
        start += counts.get(core, 0)
    return cores, assignment


def create_program_artifacts(a, b, out, *, tile_limit=None, cores=None, relaxations=None):
    """(spec, run_args, tensor_args) for out = a * b, tiled and interleaved."""
    num_tiles = out.buffer_num_pages()
    tile_bytes = out.buffer_page_size()
    core_set, assignment = _split_tiles(
        num_tiles, a.device().compute_with_storage_grid_size(), tile_limit=tile_limit, cores=cores
    )

    rings = [
        m2.DataflowBufferSpec(name, entry_size=tile_bytes, num_entries=2, data_format=dtype)
        for name, dtype in ((DFB_A, a.dtype), (DFB_B, b.dtype), (DFB_OUT, out.dtype))
    ]
    dm_schema = m2.RuntimeArgSchema(runtime_arg_names=["num_tiles", "start_id"])
    reader = m2.KernelSpec(
        K_READER,
        str(KERNEL_DIR / "reader.cpp"),
        m2.create_reader_dm_config(),
        dfb_bindings=[m2.producer_of(DFB_A, "in_a"), m2.producer_of(DFB_B, "in_b")],
        tensor_bindings=[m2.TensorBinding(TP_A, "a"), m2.TensorBinding(TP_B, "b")],
        runtime_arg_schema=dm_schema,
    )
    compute = m2.KernelSpec(
        K_COMPUTE,
        str(KERNEL_DIR / "compute.cpp"),
        m2.ComputeHardwareConfig(),
        dfb_bindings=[m2.consumer_of(DFB_A, "in_a"), m2.consumer_of(DFB_B, "in_b"), m2.producer_of(DFB_OUT, "out")],
        runtime_arg_schema=m2.RuntimeArgSchema(runtime_arg_names=["num_tiles"]),
    )
    writer = m2.KernelSpec(
        K_WRITER,
        str(KERNEL_DIR / "writer.cpp"),
        m2.create_writer_dm_config(),
        dfb_bindings=[m2.consumer_of(DFB_OUT, "out")],
        tensor_bindings=[m2.TensorBinding(TP_OUT, "out")],
        runtime_arg_schema=dm_schema,
    )
    params = [m2.TensorParameter(TP_A, a.spec), m2.TensorParameter(TP_B, b.spec), m2.TensorParameter(TP_OUT, out.spec)]
    if relaxations is not None:
        for param in params:
            param.relaxations = relaxations
    spec = m2.ProgramSpec(
        name="toy_spec_mul",
        kernels=[reader, compute, writer],
        dataflow_buffers=rings,
        tensor_parameters=params,
        work_units=[m2.WorkUnitSpec("main", [K_READER, K_COMPUTE, K_WRITER], core_set)],
    )

    num_tiles_by_core = {core: count for core, count, _ in assignment}
    start_id_by_core = {core: start for core, _, start in assignment}
    dm_args = {"num_tiles": num_tiles_by_core, "start_id": start_id_by_core}
    run_args = m2.ProgramRunArgs(
        kernel_run_args=[
            m2.KernelRunArgs(K_READER, runtime_arg_values=dm_args),
            m2.KernelRunArgs(K_COMPUTE, runtime_arg_values={"num_tiles": num_tiles_by_core}),
            m2.KernelRunArgs(K_WRITER, runtime_arg_values=dm_args),
        ]
    )
    return spec, run_args, {TP_A: 0, TP_B: 1, TP_OUT: 2}


def toy_spec_mul(a, b, *, out=None, tile_limit=None):
    if out is None:
        out = ttnn.allocate_tensor_on_device(a.spec, a.device())
    spec, run_args, tensor_args = create_program_artifacts(a, b, out, tile_limit=tile_limit)
    return ttnn.generic_op([a, b, out], spec, run_args, tensor_args)


def _inputs(device, shape, seed):
    torch.manual_seed(seed)
    ta = torch.randn(*shape, dtype=torch.bfloat16)
    tb = torch.randn(*shape, dtype=torch.bfloat16)
    a = ttnn.from_torch(ta, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    b = ttnn.from_torch(tb, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    return ta, tb, a, b


def _assert_within_bf16_ulps(got, expected, ulps=2):
    """The device product lands within two bf16 steps of torch's round-to-nearest product.

    Measured on Blackhole at the default HiFi4 fidelity over a million elements: 0.005% of the
    products differ from torch by more than one bf16 step and none by more than two (the FPU and
    packer path is not a single round-to-nearest of the exact product). The bound is relative to
    the binade of the expected value, so it scales with the magnitude.
    """
    got, expected = got.float(), expected.float()
    ulp = torch.exp2(torch.floor(torch.log2(expected.abs().clamp_min(2.0**-126)))) * 2.0**-7
    error = (got - expected).abs()
    assert torch.all(error <= ulps * ulp), f"max error {(error / ulp).max().item():.2f} bf16 ulp"


def _check(ta, tb, out):
    _assert_within_bf16_ulps(ttnn.to_torch(out), ta * tb)


def _kernel_args(run_args, kernel):
    return next(k for k in run_args.kernel_run_args if k.kernel == kernel)


def _spec_and_args(device, *, shape=(1, 1, 64, 64), seed=30):
    ta, tb, a, b = _inputs(device, shape, seed=seed)
    out = ttnn.allocate_tensor_on_device(a.spec, a.device())
    spec, run_args, tensor_args = create_program_artifacts(a, b, out)
    return (ta, tb), [a, b, out], spec, run_args, tensor_args


# ------------------------------------------------------------------------------------ correctness


@pytest.mark.parametrize("shape", [(1, 1, 32, 32), (1, 1, 128, 128), (1, 2, 256, 64)])
def test_correctness(device, shape):
    ta, tb, a, b = _inputs(device, shape, seed=0)
    _check(ta, tb, toy_spec_mul(a, b))


# ---------------------------------------------------------------------------------- program cache


def test_cache_hit_refreshes_tensor_addresses(device):
    """The second call must use freshly allocated tensors, not the addresses bound at the cache miss."""
    shape = (1, 1, 96, 96)
    ta1, tb1, a1, b1 = _inputs(device, shape, seed=1)
    first = toy_spec_mul(a1, b1)
    _check(ta1, tb1, first)
    entries_after_miss = device.num_program_cache_entries()

    # The first inputs stay alive, so the second pair cannot land on the same addresses.
    ta2, tb2, a2, b2 = _inputs(device, shape, seed=2)
    second = toy_spec_mul(a2, b2)
    assert device.num_program_cache_entries() == entries_after_miss, "expected a cache hit, got a new entry"
    _check(ta2, tb2, second)
    _check(ta1, tb1, first)


def test_cache_hit_refreshes_scalar_runtime_args(device):
    """A narrower tile_limit on a cache hit must narrow the write.

    Both calls share a byte-identical ProgramSpec and differ only in runtime arg values. If the
    values froze at the cache miss, the second call would write every tile and overwrite the
    sentinel tail.
    """
    shape = (1, 1, 32, 256)  # 8 tiles along width, one tile row
    ta, tb, a, b = _inputs(device, shape, seed=7)
    _check(ta, tb, toy_spec_mul(a, b))
    entries = device.num_program_cache_entries()

    sentinel = torch.full(shape, -7.0, dtype=torch.bfloat16)
    out = ttnn.from_torch(sentinel, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    toy_spec_mul(a, b, out=out, tile_limit=4)
    assert device.num_program_cache_entries() == entries, "tile_limit must not change the spec"

    got = ttnn.to_torch(out)
    _assert_within_bf16_ulps(got[..., :128], (ta * tb)[..., :128])
    assert torch.equal(got[..., 128:], sentinel[..., 128:]), "tail was written; scalars did not refresh"


def test_interleaved_shapes_do_not_contaminate(device):
    """Alternating specs must not leak one another's runtime args."""
    small = _inputs(device, (1, 1, 64, 64), seed=3)
    large = _inputs(device, (1, 1, 256, 128), seed=4)
    for _ in range(2):
        for ta, tb, a, b in (small, large):
            _check(ta, tb, toy_spec_mul(a, b))


@pytest.mark.parametrize("validate", [True, False], ids=["validate_on", "validate_off"])
def test_spec_reused_with_other_tensors_is_rejected(device, expect_error, validate):
    """A spec built for one tensor spec and called with another must not run the cached program on it.

    The bound tensors' specs are part of the cache key and are checked against the parameters
    on every call, with or without run-arg validation (which is off by default).
    """
    ttnn.CONFIG.validate_program_args = validate
    ta, tb, a, b = _inputs(device, (1, 1, 32, 32), seed=20)
    out = ttnn.allocate_tensor_on_device(a.spec, a.device())
    spec, run_args, tensor_args = create_program_artifacts(a, b, out)
    _check(ta, tb, ttnn.generic_op([a, b, out], spec, run_args, tensor_args))

    _, _, a2, b2 = _inputs(device, (1, 1, 64, 64), seed=21)
    out2 = ttnn.allocate_tensor_on_device(a2.spec, a2.device())
    _, run_args2, _ = create_program_artifacts(a2, b2, out2)
    with expect_error(RuntimeError, "does not match TensorParameter"):
        ttnn.generic_op([a2, b2, out2], spec, run_args2, tensor_args)


def test_dynamic_shape_relaxation_reuses_the_program(device):
    """With dynamic_tensor_shape on every parameter one spec serves tensors of any shape.

    The relaxed spec hash ignores the shape, so the second shape is a cache hit, and the kernels
    read their tile counts from runtime args. Both calls use the same core set so that only the
    tensors differ.
    """
    relaxation = m2.TensorSpecRelaxations(dynamic_tensor_shape=True)
    grid = device.compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])

    ta, tb, a, b = _inputs(device, (1, 1, 32, 32), seed=22)
    out = ttnn.allocate_tensor_on_device(a.spec, a.device())
    spec, run_args, tensor_args = create_program_artifacts(a, b, out, cores=cores, relaxations=relaxation)
    _check(ta, tb, ttnn.generic_op([a, b, out], spec, run_args, tensor_args))
    entries = device.num_program_cache_entries()

    ta2, tb2, a2, b2 = _inputs(device, (1, 1, 64, 64), seed=23)
    out2 = ttnn.allocate_tensor_on_device(a2.spec, a2.device())
    _, run_args2, _ = create_program_artifacts(a2, b2, out2, cores=cores, relaxations=relaxation)
    result = ttnn.generic_op([a2, b2, out2], spec, run_args2, tensor_args)
    assert device.num_program_cache_entries() == entries, "the relaxed spec must serve the second shape from the cache"
    _check(ta2, tb2, result)


def test_tensor_arg_order_is_not_part_of_the_key(device):
    (ta, tb), io, spec, run_args, tensor_args = _spec_and_args(device, seed=24)
    ttnn.experimental.prepare_generic_op(io, spec, run_args, tensor_args)
    entries = device.num_program_cache_entries()
    reordered = dict(reversed(list(tensor_args.items())))
    _check(ta, tb, ttnn.generic_op(io, spec, run_args, reordered))
    assert device.num_program_cache_entries() == entries, "{a, b, out} and {out, b, a} are one binding"


def test_prepare_warms_the_program_cache(device):
    (ta, tb), io, spec, run_args, tensor_args = _spec_and_args(device, seed=8)
    entries = device.num_program_cache_entries()
    ttnn.experimental.prepare_generic_op(io, spec, run_args, tensor_args)
    assert device.num_program_cache_entries() == entries + 1, "preparation inserts the program"
    result = ttnn.generic_op(io, spec, run_args, tensor_args)
    assert device.num_program_cache_entries() == entries + 1, "the launch after preparation is a cache hit"
    _check(ta, tb, result)


# ------------------------------------------------------------------------------- tensor arguments


def test_aliased_inputs_on_miss_and_hit(device):
    """One tensor bound to two parameters, on a cache miss and then on a hit with other tensors (#57248)."""
    shape = (1, 1, 64, 64)
    ta, _, a, _ = _inputs(device, shape, seed=11)
    _check(ta, ta, toy_spec_mul(a, a))
    entries = device.num_program_cache_entries()
    tc, _, c, _ = _inputs(device, shape, seed=12)
    out = toy_spec_mul(c, c)
    assert device.num_program_cache_entries() == entries, "same spec, so a cache hit"
    _check(tc, tc, out)


def test_single_tensor_io_in_place(device):
    """One tensor serves every parameter, listed once per parameter, so t becomes t * t."""
    t = torch.randn(1, 1, 64, 64, dtype=torch.bfloat16)
    tensor = ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    spec, run_args, tensor_args = create_program_artifacts(tensor, tensor, tensor)
    result = ttnn.generic_op([tensor, tensor, tensor], spec, run_args, tensor_args)
    _check(t, t, result)


def test_shared_io_index_is_rejected(device, expect_error):
    """Each parameter takes its own io_tensors entry; two parameters on one index are refused up front."""
    ta, _, a, _ = _inputs(device, (1, 1, 64, 64), seed=13)
    out = ttnn.allocate_tensor_on_device(a.spec, a.device())
    spec, run_args, _ = create_program_artifacts(a, a, out)
    with expect_error(RuntimeError, "list the tensor once per parameter"):
        ttnn.generic_op([a, out], spec, run_args, {TP_A: 0, TP_B: 0, TP_OUT: 1})


def test_unknown_tensor_arg_name_is_rejected(device, expect_error):
    _, io, spec, run_args, tensor_args = _spec_and_args(device, seed=5)
    with expect_error(RuntimeError, "not a TensorParameter"):
        ttnn.generic_op(io, spec, run_args, {**tensor_args, "zzz": 0})


def test_missing_tensor_arg_is_rejected(device, expect_error):
    _, io, spec, run_args, tensor_args = _spec_and_args(device, seed=5)
    incomplete = dict(tensor_args)
    incomplete.popitem()
    with expect_error(RuntimeError, "has no entry in tensor_args"):
        ttnn.generic_op(io, spec, run_args, incomplete)


def test_out_of_range_tensor_arg_is_rejected(device, expect_error):
    _, io, spec, run_args, tensor_args = _spec_and_args(device, seed=5)
    with expect_error(RuntimeError, "maps to io_tensors index"):
        ttnn.generic_op(io, spec, run_args, {**tensor_args, TP_OUT: 3})


def test_empty_io_tensors_is_rejected(device, expect_error):
    _, _, spec, run_args, tensor_args = _spec_and_args(device, seed=5)
    with expect_error(RuntimeError, "at least the output tensor"):
        ttnn.generic_op([], spec, run_args, tensor_args)


# ---------------------------------------------------------------------------------------- run args
#
# Metal 2.0 reads an omitted kernel, arg or node on a cache hit as "keep the previous call's
# value", and checks nothing on a hit unless validate_program_args is on. generic_op's contract
# is that run_args are complete on every call, so the op rejects incomplete ones itself: names
# and vararg counts always, per-node coverage under validate_program_args.


@pytest.mark.parametrize("validate", [True, False], ids=["validate_on", "validate_off"])
def test_run_args_without_a_kernel_are_rejected(device, expect_error, validate):
    ttnn.CONFIG.validate_program_args = validate
    _, io, spec, run_args, tensor_args = _spec_and_args(device)
    run_args.kernel_run_args = [k for k in run_args.kernel_run_args if k.kernel != K_WRITER]
    with expect_error(RuntimeError, "carry no KernelRunArgs"):
        ttnn.generic_op(io, spec, run_args, tensor_args)


def test_run_args_for_an_undeclared_kernel_are_rejected(device, expect_error):
    _, io, spec, run_args, tensor_args = _spec_and_args(device)
    run_args.kernel_run_args = [*run_args.kernel_run_args, m2.KernelRunArgs("ghost")]
    with expect_error(RuntimeError, "which the ProgramSpec does not declare"):
        ttnn.generic_op(io, spec, run_args, tensor_args)


def test_run_args_missing_a_named_arg_are_rejected(device, expect_error):
    _, io, spec, run_args, tensor_args = _spec_and_args(device)
    kernels = run_args.kernel_run_args
    writer = next(k for k in kernels if k.kernel == K_WRITER)
    values = writer.runtime_arg_values
    del values["start_id"]
    writer.runtime_arg_values = values
    run_args.kernel_run_args = kernels
    with expect_error(RuntimeError, "but run_args do not set it"):
        ttnn.generic_op(io, spec, run_args, tensor_args)


def test_run_args_with_an_undeclared_name_are_rejected(device, expect_error):
    _, io, spec, run_args, tensor_args = _spec_and_args(device)
    kernels = run_args.kernel_run_args
    compute = next(k for k in kernels if k.kernel == K_COMPUTE)
    values = compute.runtime_arg_values
    values["extra"] = values["num_tiles"]
    compute.runtime_arg_values = values
    run_args.kernel_run_args = kernels
    with expect_error(RuntimeError, "which its schema does not declare"):
        ttnn.generic_op(io, spec, run_args, tensor_args)


def test_run_args_with_a_wrong_vararg_count_are_rejected(device, expect_error):
    _, io, spec, run_args, tensor_args = _spec_and_args(device)
    kernels = spec.kernels
    kernels[0].advanced_options = m2.KernelAdvancedOptions(num_runtime_varargs=2)
    spec.kernels = kernels
    args = run_args.kernel_run_args
    reader = next(k for k in args if k.kernel == K_READER)
    core = next(iter(reader.runtime_arg_values["num_tiles"]))
    reader.advanced_options = m2.AdvancedKernelRunArgs(runtime_varargs={core: [1]})
    run_args.kernel_run_args = args
    with expect_error(RuntimeError, "expects 2 runtime varargs per node"):
        ttnn.generic_op(io, spec, run_args, tensor_args)


def test_run_args_covering_only_some_nodes_are_rejected_under_validation(device, expect_error):
    _, io, spec, run_args, tensor_args = _spec_and_args(device, shape=(1, 1, 64, 64))  # 4 tiles, 4 cores
    kernels = run_args.kernel_run_args
    reader = next(k for k in kernels if k.kernel == K_READER)
    values = reader.runtime_arg_values
    num_tiles = dict(values["num_tiles"])
    assert len(num_tiles) == 4
    num_tiles.pop(next(iter(num_tiles)))
    values["num_tiles"] = num_tiles
    reader.runtime_arg_values = values
    run_args.kernel_run_args = kernels
    with expect_error(RuntimeError, "nodes but run_args set runtime arg"):
        ttnn.generic_op(io, spec, run_args, tensor_args)


# ---------------------------------------------------------------------------------- spec hashing
#
# compute_program_spec_hash is the spec half of the program cache key (the other half is the
# bound tensors' specs under each parameter's relaxations). It is a pure function of the spec,
# so these run on host tensor specs without a device. Every field the compiled program depends
# on must change it; the two debug-only names and Table order must not.


def test_spec_hash_is_structural(device):
    _, _, a, b = _inputs(device, (1, 1, 64, 64), seed=6)
    out = ttnn.allocate_tensor_on_device(a.spec, a.device())
    spec1, _, _ = create_program_artifacts(a, b, out)
    spec2, _, _ = create_program_artifacts(a, b, out)
    assert m2.compute_program_spec_hash(spec1) == m2.compute_program_spec_hash(spec2)

    # tile_limit changes runtime arg values only, never the spec.
    spec3, _, _ = create_program_artifacts(a, b, out, tile_limit=1)
    assert m2.compute_program_spec_hash(spec3) == m2.compute_program_spec_hash(spec1)


def _host_spec(shape=(1, 1, 64, 64), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(torch.zeros(*shape, dtype=torch.bfloat16), layout=layout, dtype=dtype).spec


CORES_4X4 = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))])
CORE_0 = ttnn.CoreCoord(0, 0)
UNPACK = {DFB_A: m2.UnpackMode.UnpackToDest, DFB_B: m2.UnpackMode.UnpackToSrc}


def _host_spec_program():
    """The toy spec on host tensor specs, with one of everything the hash enumerates."""
    rings = [
        m2.DataflowBufferSpec(name, entry_size=2048, num_entries=2, data_format=ttnn.bfloat16)
        for name in (DFB_A, DFB_B, DFB_OUT)
    ]
    dm_schema = m2.RuntimeArgSchema(runtime_arg_names=["num_tiles", "start_id"])
    reader = m2.KernelSpec(
        K_READER,
        str(KERNEL_DIR / "reader.cpp"),
        m2.create_reader_dm_config(),
        dfb_bindings=[m2.producer_of(DFB_A, "in_a"), m2.producer_of(DFB_B, "in_b")],
        tensor_bindings=[m2.TensorBinding(TP_A, "a"), m2.TensorBinding(TP_B, "b")],
        runtime_arg_schema=dm_schema,
        compile_time_args={"x": 1, "y": 2},
    )
    compute = m2.KernelSpec(
        K_COMPUTE,
        str(KERNEL_DIR / "compute.cpp"),
        m2.ComputeHardwareConfig(unpack_modes=UNPACK),
        dfb_bindings=[m2.consumer_of(DFB_A, "in_a"), m2.consumer_of(DFB_B, "in_b"), m2.producer_of(DFB_OUT, "out")],
        runtime_arg_schema=m2.RuntimeArgSchema(runtime_arg_names=["num_tiles"]),
    )
    writer = m2.KernelSpec(
        K_WRITER,
        str(KERNEL_DIR / "writer.cpp"),
        m2.create_writer_dm_config(),
        dfb_bindings=[m2.consumer_of(DFB_OUT, "out")],
        tensor_bindings=[m2.TensorBinding(TP_OUT, "out")],
        runtime_arg_schema=dm_schema,
    )
    return m2.ProgramSpec(
        name="toy_spec_mul",
        kernels=[reader, compute, writer],
        dataflow_buffers=rings,
        semaphores=[m2.SemaphoreSpec("s", CORE_0, advanced_options=m2.SemaphoreAdvancedOptions(max_value=1))],
        scratchpads=[m2.ScratchpadSpec("sp", 64, data_format=ttnn.uint32)],
        tensor_parameters=[
            m2.TensorParameter(TP_A, _host_spec()),
            m2.TensorParameter(TP_B, _host_spec()),
            m2.TensorParameter(TP_OUT, _host_spec()),
        ],
        work_units=[m2.WorkUnitSpec("main", [K_READER, K_COMPUTE, K_WRITER], CORES_4X4)],
        advanced_options=m2.ProgramAdvancedOptions(
            prefetcher_pipe_parameters=[m2.PrefetcherPipeParameter("p", CORE_0, ring_size=64, entry_size=8)]
        ),
    )


def _edit_kernel(spec, index, edit):
    kernels = spec.kernels
    edit(kernels[index])
    spec.kernels = kernels


def _edit_dfb(spec, index, edit):
    rings = spec.dataflow_buffers
    edit(rings[index])
    spec.dataflow_buffers = rings


def _edit_param(spec, index, edit):
    params = spec.tensor_parameters
    edit(params[index])
    spec.tensor_parameters = params


def _edit_work_unit(spec, edit):
    units = spec.work_units
    edit(units[0])
    spec.work_units = units


def _set_dm_config(spec, index, **overrides):
    def edit(kernel):
        config = kernel.hw_config
        c = config.config_1xx
        args = dict(
            processor=c.processor, noc=c.noc, noc_mode=c.noc_mode, dedicated_noc_sharing=c.dedicated_noc_sharing
        )
        args.update(overrides)
        config.config_1xx = m2.DataMovement1XXConfig(args.pop("processor"), args.pop("noc"), **args)
        kernel.hw_config = config

    _edit_kernel(spec, index, edit)


def _set_dm_2xx(spec, index, **fields):
    def edit(kernel):
        config = kernel.hw_config
        config.config_2xx = m2.DataMovement2XXConfig(**fields)
        kernel.hw_config = config

    _edit_kernel(spec, index, edit)


def _set_compute(spec, **fields):
    _edit_kernel(
        spec, 1, lambda k: setattr(k, "hw_config", m2.ComputeHardwareConfig(**{"unpack_modes": UNPACK, **fields}))
    )


def _set_kernel_advanced(spec, **fields):
    _edit_kernel(spec, 0, lambda k: setattr(k, "advanced_options", m2.KernelAdvancedOptions(**fields)))


def _set_semaphore(spec, name="s", nodes=CORE_0, **advanced):
    spec.semaphores = [
        m2.SemaphoreSpec(name, nodes, advanced_options=m2.SemaphoreAdvancedOptions(**{"max_value": 1, **advanced}))
    ]


def _set_scratchpad(spec, name="sp", size=64, **fields):
    spec.scratchpads = [m2.ScratchpadSpec(name, size, **{"data_format": ttnn.uint32, **fields})]


def _set_pipe(spec, name="p", receivers=CORE_0, ring_size=64, entry_size=8):
    spec.advanced_options = m2.ProgramAdvancedOptions(
        prefetcher_pipe_parameters=[
            m2.PrefetcherPipeParameter(name, receivers, ring_size=ring_size, entry_size=entry_size)
        ]
    )


def _swap_first_two(items):
    return [items[1], items[0], *items[2:]]


def _relaxation_edit(field):
    return lambda spec: _edit_param(
        spec, 0, lambda p: setattr(p, "relaxations", m2.TensorSpecRelaxations(**{field: True}))
    )


STRUCTURAL_EDITS = {
    # ProgramSpec
    "kernel_order": lambda s: setattr(s, "kernels", _swap_first_two(s.kernels)),
    "dfb_order": lambda s: setattr(s, "dataflow_buffers", _swap_first_two(s.dataflow_buffers)),
    "tensor_parameter_order": lambda s: setattr(s, "tensor_parameters", _swap_first_two(s.tensor_parameters)),
    "no_semaphore": lambda s: setattr(s, "semaphores", []),
    "no_scratchpad": lambda s: setattr(s, "scratchpads", []),
    "no_prefetcher_pipe": lambda s: setattr(s, "advanced_options", m2.ProgramAdvancedOptions()),
    # DataflowBufferSpec
    "dfb.unique_id": lambda s: _edit_dfb(s, 0, lambda d: setattr(d, "unique_id", "in_a2")),
    "dfb.entry_size": lambda s: _edit_dfb(s, 0, lambda d: setattr(d, "entry_size", 4096)),
    "dfb.num_entries": lambda s: _edit_dfb(s, 0, lambda d: setattr(d, "num_entries", 4)),
    "dfb.data_format": lambda s: _edit_dfb(s, 0, lambda d: setattr(d, "data_format", ttnn.float32)),
    "dfb.data_format_none": lambda s: _edit_dfb(s, 0, lambda d: setattr(d, "data_format", None)),
    "dfb.tile_format": lambda s: _edit_dfb(s, 0, lambda d: setattr(d, "tile_format", ttnn.Tile([16, 32]))),
    "dfb.tile_format_default_tile": lambda s: _edit_dfb(s, 0, lambda d: setattr(d, "tile_format", ttnn.Tile([32, 32]))),
    "dfb.borrowed_from": lambda s: _edit_dfb(s, 0, lambda d: setattr(d, "borrowed_from", TP_A)),
    "dfb.alias_with": lambda s: _edit_dfb(
        s, 0, lambda d: setattr(d, "advanced_options", m2.DFBAdvancedOptions(alias_with=[DFB_B]))
    ),
    "dfb.allow_instance_multi_binding": lambda s: _edit_dfb(
        s, 0, lambda d: setattr(d, "advanced_options", m2.DFBAdvancedOptions(allow_instance_multi_binding=True))
    ),
    "dfb.prefetcher_pipe_relays": lambda s: _edit_dfb(
        s, 0, lambda d: setattr(d, "advanced_options", m2.DFBAdvancedOptions(prefetcher_pipe_relays=["p"]))
    ),
    # KernelSpec
    "kernel.unique_id": lambda s: _edit_kernel(s, 2, lambda k: setattr(k, "unique_id", "writer2")),
    "kernel.source_path": lambda s: _edit_kernel(s, 2, lambda k: setattr(k, "source", str(KERNEL_DIR / "reader.cpp"))),
    "kernel.source_inline": lambda s: _edit_kernel(
        s, 2, lambda k: setattr(k, "source", m2.SourceCode("void kernel_main() {}"))
    ),
    "kernel.num_threads": lambda s: _edit_kernel(s, 1, lambda k: setattr(k, "num_threads", 2)),
    "kernel.compiler_options.include_paths": lambda s: _edit_kernel(
        s, 0, lambda k: setattr(k, "compiler_options", m2.CompilerOptions(include_paths=["/tmp/x"]))
    ),
    "kernel.compiler_options.defines": lambda s: _edit_kernel(
        s, 0, lambda k: setattr(k, "compiler_options", m2.CompilerOptions(defines={"A": "1"}))
    ),
    "kernel.compiler_options.opt_level": lambda s: _edit_kernel(
        s, 0, lambda k: setattr(k, "compiler_options", m2.CompilerOptions(opt_level=m2.KernelBuildOptLevel.O3))
    ),
    "kernel.dfb_binding.accessor_name": lambda s: _edit_kernel(
        s, 0, lambda k: setattr(k, "dfb_bindings", [m2.producer_of(DFB_A, "in_a_x"), m2.producer_of(DFB_B, "in_b")])
    ),
    "kernel.dfb_binding.dfb_spec_name": lambda s: _edit_kernel(
        s, 0, lambda k: setattr(k, "dfb_bindings", [m2.producer_of(DFB_OUT, "in_a"), m2.producer_of(DFB_B, "in_b")])
    ),
    "kernel.dfb_binding.endpoint_type": lambda s: _edit_kernel(
        s, 0, lambda k: setattr(k, "dfb_bindings", [m2.consumer_of(DFB_A, "in_a"), m2.producer_of(DFB_B, "in_b")])
    ),
    "kernel.dfb_binding.access_pattern": lambda s: _edit_kernel(
        s,
        1,
        lambda k: setattr(
            k,
            "dfb_bindings",
            [m2.all_consumer_of(DFB_A, "in_a"), m2.consumer_of(DFB_B, "in_b"), m2.producer_of(DFB_OUT, "out")],
        ),
    ),
    "kernel.dfb_binding.block_size": lambda s: _edit_kernel(
        s,
        0,
        lambda k: setattr(k, "dfb_bindings", [m2.blocked_producer_of(DFB_A, "in_a", 2), m2.producer_of(DFB_B, "in_b")]),
    ),
    "kernel.dfb_binding.block_size_value": lambda s: _edit_kernel(
        s,
        0,
        lambda k: setattr(k, "dfb_bindings", [m2.blocked_producer_of(DFB_A, "in_a", 4), m2.producer_of(DFB_B, "in_b")]),
    ),
    "kernel.semaphore_bindings": lambda s: _edit_kernel(
        s, 0, lambda k: setattr(k, "semaphore_bindings", [m2.SemaphoreBinding("s", "s")])
    ),
    "kernel.scratchpad_bindings": lambda s: _edit_kernel(
        s, 0, lambda k: setattr(k, "scratchpad_bindings", [m2.ScratchpadBinding("sp", "sp")])
    ),
    "kernel.tensor_binding.accessor_name": lambda s: _edit_kernel(
        s, 0, lambda k: setattr(k, "tensor_bindings", [m2.TensorBinding(TP_A, "aa"), m2.TensorBinding(TP_B, "b")])
    ),
    "kernel.tensor_binding.tensor_parameter_name": lambda s: _edit_kernel(
        s, 0, lambda k: setattr(k, "tensor_bindings", [m2.TensorBinding(TP_OUT, "a"), m2.TensorBinding(TP_B, "b")])
    ),
    "kernel.compile_time_args_value": lambda s: _edit_kernel(
        s, 0, lambda k: setattr(k, "compile_time_args", {"x": 1, "y": 3})
    ),
    "kernel.compile_time_args_key": lambda s: _edit_kernel(
        s, 0, lambda k: setattr(k, "compile_time_args", {"x": 1, "z": 2})
    ),
    "kernel.runtime_arg_names_rename": lambda s: _edit_kernel(
        s, 1, lambda k: setattr(k, "runtime_arg_schema", m2.RuntimeArgSchema(runtime_arg_names=["tile_count"]))
    ),
    "kernel.runtime_arg_names_reorder": lambda s: _edit_kernel(
        s,
        0,
        lambda k: setattr(k, "runtime_arg_schema", m2.RuntimeArgSchema(runtime_arg_names=["start_id", "num_tiles"])),
    ),
    "kernel.common_runtime_arg_names": lambda s: _edit_kernel(
        s,
        0,
        lambda k: setattr(
            k,
            "runtime_arg_schema",
            m2.RuntimeArgSchema(runtime_arg_names=["num_tiles", "start_id"], common_runtime_arg_names=["scale"]),
        ),
    ),
    "kernel.runtime_arg_moved_to_common": lambda s: _edit_kernel(
        s,
        0,
        lambda k: setattr(
            k,
            "runtime_arg_schema",
            m2.RuntimeArgSchema(runtime_arg_names=["num_tiles"], common_runtime_arg_names=["start_id"]),
        ),
    ),
    # DataMovementHardwareConfig
    "dm.processor": lambda s: _set_dm_config(s, 0, processor=ttnn.DataMovementProcessor.RISCV_0),
    "dm.noc": lambda s: _set_dm_config(s, 0, noc=ttnn.NOC.NOC_1),
    "dm.noc_mode": lambda s: _set_dm_config(s, 0, noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC),
    "dm.dedicated_noc_sharing": lambda s: _set_dm_config(
        s, 0, dedicated_noc_sharing=m2.DedicatedNocSharing.SHARED_DISJOINT_TRAFFIC
    ),
    "dm.config_2xx": lambda s: _set_dm_2xx(s, 0),
    "dm.config_2xx.disable_dfb_implicit_sync_for_all": lambda s: _set_dm_2xx(
        s, 0, disable_dfb_implicit_sync_for_all=True
    ),
    "dm.config_2xx.disable_dfb_implicit_sync_for": lambda s: _set_dm_2xx(s, 0, disable_dfb_implicit_sync_for=[DFB_A]),
    "dm.config_1xx_none": lambda s: _edit_kernel(
        s, 0, lambda k: setattr(k, "hw_config", m2.DataMovementHardwareConfig(config_2xx=m2.DataMovement2XXConfig()))
    ),
    # ComputeHardwareConfig
    "compute.fpu_math_fidelity": lambda s: _set_compute(s, fpu_math_fidelity=ttnn.MathFidelity.LoFi),
    "compute.sfpu_precision_mode": lambda s: _set_compute(s, sfpu_precision_mode=m2.Precision.Approximate),
    "compute.enable_32_bit_dest": lambda s: _set_compute(s, enable_32_bit_dest=True),
    "compute.double_buffer_dest": lambda s: _set_compute(s, double_buffer_dest=False),
    "compute.unpack_modes_value": lambda s: _set_compute(
        s, unpack_modes={DFB_A: m2.UnpackMode.UnpackToSrc, DFB_B: m2.UnpackMode.UnpackToSrc}
    ),
    "compute.unpack_modes_entry": lambda s: _set_compute(s, unpack_modes={DFB_A: m2.UnpackMode.UnpackToDest}),
    "compute.config_1xx": lambda s: _set_compute(s, config_1xx=m2.Compute1XXConfig()),
    "compute.config_1xx.bfp_pack_precision_mode": lambda s: _set_compute(
        s, config_1xx=m2.Compute1XXConfig(bfp_pack_precision_mode=m2.Precision.Precise)
    ),
    "compute.config_2xx": lambda s: _set_compute(s, config_2xx=m2.Compute2XXConfig()),
    # KernelAdvancedOptions
    "kernel.advanced.num_runtime_varargs": lambda s: _set_kernel_advanced(s, num_runtime_varargs=3),
    "kernel.advanced.num_common_runtime_varargs": lambda s: _set_kernel_advanced(s, num_common_runtime_varargs=3),
    "kernel.advanced.compile_time_varargs": lambda s: _set_kernel_advanced(s, compile_time_varargs=[1, 2]),
    "kernel.advanced.tensor_binding_sequences": lambda s: _set_kernel_advanced(
        s, tensor_binding_sequences=[m2.TensorBindingSequence("seq", ["a", "b"])]
    ),
    "kernel.advanced.prefetcher_pipe_bindings": lambda s: _set_kernel_advanced(
        s, prefetcher_pipe_bindings=[m2.PrefetcherPipeBinding(["p"], "pipe")]
    ),
    # TensorParameter
    "param.unique_id": lambda s: _edit_param(s, 0, lambda p: setattr(p, "unique_id", "a2")),
    "param.spec_shape": lambda s: _edit_param(s, 0, lambda p: setattr(p, "spec", _host_spec((1, 1, 32, 32)))),
    "param.spec_dtype": lambda s: _edit_param(s, 0, lambda p: setattr(p, "spec", _host_spec(dtype=ttnn.float32))),
    "param.spec_layout": lambda s: _edit_param(
        s, 0, lambda p: setattr(p, "spec", _host_spec(layout=ttnn.ROW_MAJOR_LAYOUT))
    ),
    "param.relaxations.match_padded_shape_only": _relaxation_edit("match_padded_shape_only"),
    "param.relaxations.dynamic_tensor_shape": _relaxation_edit("dynamic_tensor_shape"),
    "param.relaxations.relax_logical_rank": _relaxation_edit("relax_logical_rank"),
    "param.relaxations.match_page_size": _relaxation_edit("match_page_size"),
    # WorkUnitSpec
    "work_unit.kernels_drop": lambda s: _edit_work_unit(s, lambda u: setattr(u, "kernels", [K_READER, K_COMPUTE])),
    "work_unit.kernels_reorder": lambda s: _edit_work_unit(
        s, lambda u: setattr(u, "kernels", [K_COMPUTE, K_READER, K_WRITER])
    ),
    "work_unit.target_nodes_smaller": lambda s: _edit_work_unit(
        s, lambda u: setattr(u, "target_nodes", ttnn.CoreRangeSet([ttnn.CoreRange(CORE_0, ttnn.CoreCoord(1, 1))]))
    ),
    "work_unit.target_nodes_single_core": lambda s: _edit_work_unit(s, lambda u: setattr(u, "target_nodes", CORE_0)),
    # SemaphoreSpec
    "semaphore.unique_id": lambda s: _set_semaphore(s, name="s2"),
    "semaphore.target_nodes": lambda s: _set_semaphore(s, nodes=ttnn.CoreCoord(1, 0)),
    "semaphore.max_value": lambda s: _set_semaphore(s, max_value=2),
    # ScratchpadSpec
    "scratchpad.unique_id": lambda s: _set_scratchpad(s, name="sp2"),
    "scratchpad.size_per_node": lambda s: _set_scratchpad(s, size=128),
    "scratchpad.data_format": lambda s: _set_scratchpad(s, data_format=ttnn.bfloat16),
    "scratchpad.data_format_none": lambda s: _set_scratchpad(s, data_format=None),
    "scratchpad.tile_format": lambda s: _set_scratchpad(s, tile_format=ttnn.Tile([16, 32])),
    # PrefetcherPipeParameter
    "pipe.unique_id": lambda s: _set_pipe(s, name="p2"),
    "pipe.receivers": lambda s: _set_pipe(s, receivers=ttnn.CoreCoord(1, 1)),
    "pipe.ring_size": lambda s: _set_pipe(s, ring_size=128),
    "pipe.entry_size": lambda s: _set_pipe(s, entry_size=16),
}

NON_STRUCTURAL_EDITS = {
    "rebuild": lambda s: None,
    "spec.name": lambda s: setattr(s, "name", "toy_spec_mul_v2"),
    "work_unit.name": lambda s: _edit_work_unit(s, lambda u: setattr(u, "name", "main2")),
    "compile_time_args_order": lambda s: _edit_kernel(
        s, 0, lambda k: setattr(k, "compile_time_args", {"y": 2, "x": 1})
    ),
    "unpack_modes_order": lambda s: _set_compute(
        s, unpack_modes={DFB_B: m2.UnpackMode.UnpackToSrc, DFB_A: m2.UnpackMode.UnpackToDest}
    ),
}


@pytest.mark.parametrize("edit", STRUCTURAL_EDITS.values(), ids=STRUCTURAL_EDITS.keys())
def test_spec_hash_covers_every_structural_field(edit):
    base = _host_spec_program()
    edited = _host_spec_program()
    edit(edited)
    assert m2.compute_program_spec_hash(edited) != m2.compute_program_spec_hash(base)


@pytest.mark.parametrize("edit", NON_STRUCTURAL_EDITS.values(), ids=NON_STRUCTURAL_EDITS.keys())
def test_spec_hash_ignores_names_and_table_order(edit):
    base = _host_spec_program()
    edited = _host_spec_program()
    edit(edited)
    assert m2.compute_program_spec_hash(edited) == m2.compute_program_spec_hash(base)


def _runtime_vs_common_split():
    one, other = _host_spec_program(), _host_spec_program()
    _edit_kernel(one, 1, lambda k: setattr(k, "runtime_arg_schema", m2.RuntimeArgSchema(runtime_arg_names=["x", "y"])))
    _edit_kernel(
        other,
        1,
        lambda k: setattr(
            k, "runtime_arg_schema", m2.RuntimeArgSchema(runtime_arg_names=["x"], common_runtime_arg_names=["y"])
        ),
    )
    return one, other


def _semaphore_vs_scratchpad_binding():
    one, other = _host_spec_program(), _host_spec_program()
    for spec in (one, other):
        spec.scratchpads = [m2.ScratchpadSpec("s", 64, data_format=ttnn.uint32)]  # same name as the semaphore
    _edit_kernel(one, 0, lambda k: setattr(k, "semaphore_bindings", [m2.SemaphoreBinding("s", "x")]))
    _edit_kernel(other, 0, lambda k: setattr(k, "scratchpad_bindings", [m2.ScratchpadBinding("s", "x")]))
    return one, other


def _tensor_binding_sequence_split():
    one, other = _host_spec_program(), _host_spec_program()
    _set_kernel_advanced(one, tensor_binding_sequences=[m2.TensorBindingSequence("s1", ["m1", "m2"])])
    _set_kernel_advanced(
        other, tensor_binding_sequences=[m2.TensorBindingSequence("s1", ["m1"]), m2.TensorBindingSequence("m2", [])]
    )
    return one, other


def _prefetcher_pipe_binding_split():
    one, other = _host_spec_program(), _host_spec_program()
    _set_kernel_advanced(one, prefetcher_pipe_bindings=[m2.PrefetcherPipeBinding(["x", "y"], "p")])
    _set_kernel_advanced(
        other, prefetcher_pipe_bindings=[m2.PrefetcherPipeBinding(["x"], "p"), m2.PrefetcherPipeBinding([], "y")]
    )
    return one, other


def _alias_with_vs_relays_split():
    one, other = _host_spec_program(), _host_spec_program()
    _edit_dfb(one, 0, lambda d: setattr(d, "advanced_options", m2.DFBAdvancedOptions(alias_with=[DFB_B, DFB_OUT])))
    _edit_dfb(
        other,
        0,
        lambda d: setattr(
            d, "advanced_options", m2.DFBAdvancedOptions(alias_with=[DFB_B], prefetcher_pipe_relays=[DFB_OUT])
        ),
    )
    return one, other


def _one_work_unit_vs_two():
    one, other = _host_spec_program(), _host_spec_program()
    other.work_units = [
        m2.WorkUnitSpec("main", [K_READER, K_COMPUTE], CORES_4X4),
        m2.WorkUnitSpec("rest", [K_WRITER], CORES_4X4),
    ]
    return one, other


@pytest.mark.parametrize(
    "pair",
    [
        _runtime_vs_common_split,
        _semaphore_vs_scratchpad_binding,
        _tensor_binding_sequence_split,
        _prefetcher_pipe_binding_split,
        _alias_with_vs_relays_split,
        _one_work_unit_vs_two,
    ],
    ids=lambda f: f.__name__.strip("_"),
)
def test_spec_hash_keeps_adjacent_lists_apart(pair):
    """Two specs whose lists hold the same elements split differently are different programs.

    Each list is hashed with its own tag and length, so an element cannot migrate across a list
    boundary without changing the key.
    """
    one, other = pair()
    assert m2.compute_program_spec_hash(one) != m2.compute_program_spec_hash(other)
