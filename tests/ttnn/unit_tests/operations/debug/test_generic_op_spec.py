# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""ttnn.generic_op with a Metal 2.0 ProgramSpec (ttnn.experimental.metal2).

The op under test is an elementwise bf16 multiply (reader, compute, writer over two input rings
and one output ring, tiles split over the grid); its kernels live in kernels/toy_spec_mul/. The
tests pin the contract of the spec overload: correctness, the program cache (a byte-identical
spec is a hit, and a hit re-applies this call's tensor addresses and runtime arg values), tensor
argument validation, aliased and single-tensor io, preparation without dispatch, and the
structural hash the cache keys on.
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
    """Run-arg validation is off by default; the spec path's cache-hit revalidation is what these tests pin."""
    previous = ttnn.CONFIG.validate_program_args
    ttnn.CONFIG.validate_program_args = True
    yield
    ttnn.CONFIG.validate_program_args = previous


def _split_tiles(num_tiles, grid_size, *, tile_limit=None):
    """(all_cores, [(core, num_tiles, start_id), ...]), row-major over the grid.

    The core set comes from the FULL tile count and the per-core counts from the (possibly
    limited) work, so tile_limit changes only runtime arg VALUES and leaves the ProgramSpec, and
    therefore the program cache key, byte-identical. Cores past the limit get zero tiles.
    """
    _, all_cores, *_ = ttnn.split_work_to_cores(grid_size, num_tiles, row_wise=True)
    work = num_tiles if tile_limit is None else min(tile_limit, num_tiles)
    _, _, group_1, group_2, per_core_1, per_core_2 = ttnn.split_work_to_cores(grid_size, work, row_wise=True)
    counts = {}
    for group, per_core in ((group_1, per_core_1), (group_2, per_core_2)):
        if per_core:
            for core in ttnn.corerange_to_cores(group, None, True):
                counts[core] = per_core
    assignment, start = [], 0
    for core in ttnn.corerange_to_cores(all_cores, None, True):
        assignment.append((core, counts.get(core, 0), start))
        start += counts.get(core, 0)
    return all_cores, assignment


def create_program_artifacts(a, b, out, *, tile_limit=None):
    """(spec, run_args, tensor_args) for out = a * b, tiled and interleaved."""
    num_tiles = out.buffer_num_pages()
    tile_bytes = out.buffer_page_size()
    core_set, assignment = _split_tiles(num_tiles, a.device().compute_with_storage_grid_size(), tile_limit=tile_limit)

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
    spec = m2.ProgramSpec(
        name="toy_spec_mul",
        kernels=[reader, compute, writer],
        dataflow_buffers=rings,
        tensor_parameters=[
            m2.TensorParameter(TP_A, a.spec),
            m2.TensorParameter(TP_B, b.spec),
            m2.TensorParameter(TP_OUT, out.spec),
        ],
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


def _check(ta, tb, out):
    got = ttnn.to_torch(out).float()
    expected = (ta * tb).float()
    # bf16 products from the FPU and from torch can differ by an ulp or two (1 ulp is 1/128 of the magnitude).
    assert torch.allclose(got, expected, atol=0.05, rtol=0.02), (got - expected).abs().max()


@pytest.mark.parametrize("shape", [(1, 1, 32, 32), (1, 1, 128, 128), (1, 2, 256, 64)])
def test_correctness(device, shape):
    ta, tb, a, b = _inputs(device, shape, seed=0)
    _check(ta, tb, toy_spec_mul(a, b))


def test_cache_hit_refreshes_tensor_addresses(device):
    """The second call must use freshly allocated tensors, not the addresses bound at the cache miss."""
    device.enable_program_cache()
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
    device.enable_program_cache()
    shape = (1, 1, 32, 256)  # 8 tiles along width, one tile row
    ta, tb, a, b = _inputs(device, shape, seed=7)
    _check(ta, tb, toy_spec_mul(a, b))
    entries = device.num_program_cache_entries()

    sentinel = torch.full(shape, -7.0, dtype=torch.bfloat16)
    out = ttnn.from_torch(sentinel, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    toy_spec_mul(a, b, out=out, tile_limit=4)
    assert device.num_program_cache_entries() == entries, "tile_limit must not change the spec"

    got = ttnn.to_torch(out).float()
    assert torch.allclose(got[..., :128], (ta * tb).float()[..., :128], atol=0.05)
    assert torch.equal(got[..., 128:], sentinel.float()[..., 128:]), "tail was written; scalars did not refresh"


def test_interleaved_shapes_do_not_contaminate(device):
    """Alternating specs must not leak one another's runtime args."""
    device.enable_program_cache()
    small = _inputs(device, (1, 1, 64, 64), seed=3)
    large = _inputs(device, (1, 1, 256, 128), seed=4)
    for _ in range(2):
        for ta, tb, a, b in (small, large):
            _check(ta, tb, toy_spec_mul(a, b))


def test_aliased_inputs_on_miss_and_hit(device):
    """One tensor bound to two parameters, on a cache miss and then on a hit with other tensors (#57248)."""
    device.enable_program_cache()
    shape = (1, 1, 64, 64)
    ta, _, a, _ = _inputs(device, shape, seed=11)
    _check(ta, ta, toy_spec_mul(a, a))
    entries = device.num_program_cache_entries()
    tc, _, c, _ = _inputs(device, shape, seed=12)
    out = toy_spec_mul(c, c)
    assert device.num_program_cache_entries() == entries, "same spec, so a cache hit"
    _check(tc, tc, out)


def test_single_tensor_io_in_place(device):
    """io_tensors may hold just the output; here every parameter maps to it, so t becomes t * t."""
    t = torch.randn(1, 1, 64, 64, dtype=torch.bfloat16)
    tensor = ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    spec, run_args, _ = create_program_artifacts(tensor, tensor, tensor)
    result = ttnn.generic_op([tensor], spec, run_args, {TP_A: 0, TP_B: 0, TP_OUT: 0})
    _check(t, t, result)


def test_missing_tensor_arg_is_rejected(device, expect_error):
    ta, tb, a, b = _inputs(device, (1, 1, 32, 32), seed=5)
    out = ttnn.allocate_tensor_on_device(a.spec, a.device())
    spec, run_args, tensor_args = create_program_artifacts(a, b, out)
    incomplete = dict(tensor_args)
    incomplete.popitem()
    with expect_error(RuntimeError, "has no entry in tensor_args"):
        ttnn.generic_op([a, b, out], spec, run_args, incomplete)


def test_out_of_range_tensor_arg_is_rejected(device, expect_error):
    ta, tb, a, b = _inputs(device, (1, 1, 32, 32), seed=5)
    out = ttnn.allocate_tensor_on_device(a.spec, a.device())
    spec, run_args, tensor_args = create_program_artifacts(a, b, out)
    with expect_error(RuntimeError, "maps to io_tensors index"):
        ttnn.generic_op([a, b, out], spec, run_args, {**tensor_args, TP_OUT: 3})


def test_empty_io_tensors_is_rejected(device, expect_error):
    ta, tb, a, b = _inputs(device, (1, 1, 32, 32), seed=5)
    out = ttnn.allocate_tensor_on_device(a.spec, a.device())
    spec, run_args, tensor_args = create_program_artifacts(a, b, out)
    with expect_error(RuntimeError, "at least the output tensor"):
        ttnn.generic_op([], spec, run_args, tensor_args)


def test_prepare_warms_the_program_cache(device):
    device.enable_program_cache()
    ta, tb, a, b = _inputs(device, (1, 1, 64, 64), seed=8)
    out = ttnn.allocate_tensor_on_device(a.spec, a.device())
    spec, run_args, tensor_args = create_program_artifacts(a, b, out)
    entries = device.num_program_cache_entries()
    ttnn.experimental.prepare_generic_op([a, b, out], spec, run_args, tensor_args)
    assert device.num_program_cache_entries() == entries + 1, "preparation inserts the program"
    result = ttnn.generic_op([a, b, out], spec, run_args, tensor_args)
    assert device.num_program_cache_entries() == entries + 1, "the launch after preparation is a cache hit"
    _check(ta, tb, result)


def test_spec_hash_is_structural(device):
    _, _, a, b = _inputs(device, (1, 1, 64, 64), seed=6)
    out = ttnn.allocate_tensor_on_device(a.spec, a.device())
    spec1, _, _ = create_program_artifacts(a, b, out)
    spec2, _, _ = create_program_artifacts(a, b, out)
    assert m2.compute_program_spec_hash(spec1) == m2.compute_program_spec_hash(spec2)

    # tile_limit changes runtime arg values only, never the spec.
    spec3, _, _ = create_program_artifacts(a, b, out, tile_limit=1)
    assert m2.compute_program_spec_hash(spec3) == m2.compute_program_spec_hash(spec1)


def _structural_edits():
    """Every edit here must change the hash: each is a field the compiled program depends on."""

    def set_num_entries(spec):
        rings = spec.dataflow_buffers
        rings[0].num_entries = 4
        spec.dataflow_buffers = rings

    def set_block_size(spec):
        kernels = spec.kernels
        kernels[0].dfb_bindings = [m2.blocked_producer_of(DFB_A, "in_a", 2), m2.producer_of(DFB_B, "in_b")]
        spec.kernels = kernels

    def set_noc_sharing(spec):
        kernels = spec.kernels
        config = kernels[0].hw_config
        config.config_1xx = m2.DataMovement1XXConfig(
            config.config_1xx.processor,
            config.config_1xx.noc,
            dedicated_noc_sharing=m2.DedicatedNocSharing.SHARED_DISJOINT_TRAFFIC,
        )
        kernels[0].hw_config = config
        spec.kernels = kernels

    def set_compile_time_arg(spec):
        kernels = spec.kernels
        kernels[1].compile_time_args = {"fast": 1}
        spec.kernels = kernels

    def rename_runtime_arg(spec):
        kernels = spec.kernels
        kernels[1].runtime_arg_schema = m2.RuntimeArgSchema(runtime_arg_names=["tile_count"])
        spec.kernels = kernels

    def set_num_threads(spec):
        kernels = spec.kernels
        kernels[1].num_threads = 2
        spec.kernels = kernels

    def inline_source(spec):
        kernels = spec.kernels
        kernels[2].source = m2.SourceCode("void kernel_main() {}")
        spec.kernels = kernels

    def set_opt_level(spec):
        kernels = spec.kernels
        kernels[0].compiler_options = m2.CompilerOptions(opt_level=m2.KernelBuildOptLevel.O3)
        spec.kernels = kernels

    def set_relaxation(spec):
        params = spec.tensor_parameters
        params[0].relaxations = m2.TensorSpecRelaxations(relax_logical_rank=True)
        spec.tensor_parameters = params

    def set_fidelity(spec):
        kernels = spec.kernels
        kernels[1].hw_config = m2.ComputeHardwareConfig(fpu_math_fidelity=ttnn.MathFidelity.LoFi)
        spec.kernels = kernels

    def add_scratchpad(spec):
        spec.scratchpads = [m2.ScratchpadSpec("tmp", 64, data_format=ttnn.uint32)]

    def add_semaphore(spec):
        spec.semaphores = [
            m2.SemaphoreSpec("s", ttnn.CoreCoord(0, 0), advanced_options=m2.SemaphoreAdvancedOptions(max_value=2))
        ]

    def add_prefetcher_pipe(spec):
        spec.advanced_options = m2.ProgramAdvancedOptions(
            prefetcher_pipe_parameters=[
                m2.PrefetcherPipeParameter("pipe", ttnn.CoreCoord(0, 0), ring_size=64, entry_size=8)
            ]
        )

    def set_compile_time_varargs(spec):
        kernels = spec.kernels
        kernels[0].advanced_options = m2.KernelAdvancedOptions(compile_time_varargs=[1, 2])
        spec.kernels = kernels

    def rename(spec):
        spec.name = "toy_spec_mul_v2"

    def move_work_unit(spec):
        units = spec.work_units
        units[0].target_nodes = ttnn.CoreCoord(0, 0)
        spec.work_units = units

    return [
        set_num_entries,
        set_block_size,
        set_noc_sharing,
        set_compile_time_arg,
        rename_runtime_arg,
        set_num_threads,
        inline_source,
        set_opt_level,
        set_relaxation,
        set_fidelity,
        add_scratchpad,
        add_semaphore,
        add_prefetcher_pipe,
        set_compile_time_varargs,
        rename,
        move_work_unit,
    ]


@pytest.mark.parametrize("edit", _structural_edits(), ids=lambda e: e.__name__)
def test_spec_hash_covers_every_structural_field(device, edit):
    _, _, a, b = _inputs(device, (1, 1, 64, 64), seed=6)
    out = ttnn.allocate_tensor_on_device(a.spec, a.device())
    base, _, _ = create_program_artifacts(a, b, out)
    edited, _, _ = create_program_artifacts(a, b, out)
    edit(edited)
    assert m2.compute_program_spec_hash(edited) != m2.compute_program_spec_hash(base), edit.__name__
