# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from ttnn.operations.toy_spec_mcast import toy_spec_mcast

TILE = 32


@pytest.fixture(scope="module")
def device():
    ttnn.CONFIG.validate_program_args = True
    dev = ttnn.open_device(device_id=0)
    dev.enable_program_cache()
    yield dev
    ttnn.close_device(dev)


def _input(device, rows, seed):
    torch.manual_seed(seed)
    t = torch.randn(1, 1, TILE * rows, TILE, dtype=torch.bfloat16)
    return t, ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)


def _check(t, out, rows, cols):
    got = ttnn.to_torch(out).float()
    assert tuple(got.shape) == (1, 1, TILE * rows, TILE * cols)
    for x in range(cols):
        column = got[..., x * TILE : (x + 1) * TILE]
        assert torch.equal(column, t.float()), f"column {x} did not receive the broadcast tile"


@pytest.mark.parametrize("rows,cols", [(1, 4), (2, 4), (4, 2), (3, 3)])
def test_broadcast_across_row(device, rows, cols):
    t, inp = _input(device, rows, seed=rows * 10 + cols)
    _check(t, toy_spec_mcast(inp, rows, cols), rows, cols)


def test_single_column_is_degenerate(device):
    """cols == 1: no receivers, so the sender must not multicast at all."""
    t, inp = _input(device, 2, seed=1)
    _check(t, toy_spec_mcast(inp, 2, 1), 2, 1)


def test_cache_hit_refreshes_tensor_addresses(device):
    rows, cols = 2, 4
    t1, inp1 = _input(device, rows, seed=2)
    first = toy_spec_mcast(inp1, rows, cols)
    _check(t1, first, rows, cols)

    entries = device.num_program_cache_entries()
    t2, inp2 = _input(device, rows, seed=3)
    second = toy_spec_mcast(inp2, rows, cols)

    assert device.num_program_cache_entries() == entries, "expected a cache hit, got a new entry"
    _check(t2, second, rows, cols)
    _check(t1, first, rows, cols)


def test_attach_is_addressed_by_name_not_offset(device):
    """attach() packs families end to end and reports each one's vararg base itself."""
    from ttnn.mcast_spec import McastFamily

    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 1))])
    cores = [ttnn.CoreCoord(x, y) for y in range(2) for x in range(4)]
    kernel = ttnn.KernelSpec(
        unique_id="reader",
        source="/dev/null",
        hw_config=ttnn.create_reader_dm_config(),
    )
    spec = ttnn.ProgramSpec(name="t", kernels=[kernel])
    run_args = ttnn.ProgramRunArgs(kernel_run_args=[ttnn.KernelRunArgs(kernel="reader")])

    row = McastFamily(device, grid, "row", shape=ttnn.Mcast1DShape.PerRow, sender_index=0)
    col = McastFamily(device, grid, "col", shape=ttnn.Mcast1DShape.PerColumn, sender_index=0)
    row.attach(spec, run_args, kernels=["reader"], cores=cores)
    col.attach(spec, run_args, kernels=["reader"], cores=cores)

    ct = dict(spec.kernels[0].compile_time_args)
    assert ct["row_rt_base"] == 0
    assert ct["col_rt_base"] == 4
    assert spec.kernels[0].advanced_options.num_runtime_varargs == 8
    assert [str(s.unique_id) for s in spec.semaphores] == [
        "row_data_ready",
        "row_consumer_ready",
        "col_data_ready",
        "col_consumer_ready",
    ]
    for core in cores:
        assert len(run_args.kernel_run_args[0].advanced_options.runtime_varargs[core]) == 8


def test_attach_rejects_unknown_kernel(device, expect_error):
    from ttnn.mcast_spec import McastFamily

    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))])
    spec = ttnn.ProgramSpec(name="t", kernels=[])
    run_args = ttnn.ProgramRunArgs()
    family = McastFamily(device, grid, "row", shape=ttnn.Mcast1DShape.PerRow, sender_index=0)

    with expect_error(KeyError, "not in the ProgramSpec"):
        family.attach(spec, run_args, kernels=["nope"])


def test_attach_rejects_prefix_collision(device, expect_error):
    """Two families sharing a prefix would silently share semaphores."""
    from ttnn.mcast_spec import McastFamily

    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))])
    cores = [ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0)]
    kernel = ttnn.KernelSpec(unique_id="r", source="/dev/null", hw_config=ttnn.create_reader_dm_config())
    spec = ttnn.ProgramSpec(name="t", kernels=[kernel])
    run_args = ttnn.ProgramRunArgs(kernel_run_args=[ttnn.KernelRunArgs(kernel="r")])

    McastFamily(device, grid, "row", shape=ttnn.Mcast1DShape.PerRow).attach(spec, run_args, kernels=["r"], cores=cores)
    with expect_error(ValueError, "different prefix"):
        McastFamily(device, grid, "row", shape=ttnn.Mcast1DShape.PerColumn).attach(
            spec, run_args, kernels=["r"], cores=cores
        )


def test_declared_vararg_count_shifts_the_family_block(device):
    """attach derives its base from the declared count, so a kernel that declares varargs without
    supplying values shifts the family's block. The host validator is what catches it."""
    from ttnn.mcast_spec import McastFamily

    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))])
    cores = [ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0)]
    kernel = ttnn.KernelSpec(
        unique_id="r",
        source="/dev/null",
        hw_config=ttnn.create_reader_dm_config(),
        advanced_options=ttnn.KernelAdvancedOptions(num_runtime_varargs=2),  # declared, never supplied
    )
    spec = ttnn.ProgramSpec(name="t", kernels=[kernel])
    run_args = ttnn.ProgramRunArgs(kernel_run_args=[ttnn.KernelRunArgs(kernel="r")])

    family = McastFamily(device, grid, "row", shape=ttnn.Mcast1DShape.PerRow)
    family.attach(spec, run_args, kernels=["r"], cores=cores)

    # attach honoured the DECLARED count (2), so the family's block starts at vararg index 2 --
    # but only its own 4 values were ever supplied, at indices 0..3. The device would read the
    # rect from 2..5. get_vararg is not bounds-checked, so only the host validator catches this,
    # and only when ttnn.CONFIG.validate_program_args is on.
    assert dict(spec.kernels[0].compile_time_args)["row_rt_base"] == 2
    assert spec.kernels[0].advanced_options.num_runtime_varargs == 6
    assert len(run_args.kernel_run_args[0].advanced_options.runtime_varargs[cores[0]]) == 4
