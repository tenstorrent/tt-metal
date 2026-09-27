# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Python name conversion and native mutation for multicast ProgramSpec attachment.

The native McastHostFixture suite covers the full attachment contract; these tests cover
Python overload selection, container ownership and string resource names.
"""

import pytest
import ttnn


def _grid():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))])


def _spec(grid, names=("reader",)):
    kernels = [
        ttnn.KernelSpec(unique_id=name, source="/dev/null", hw_config=ttnn.create_reader_dm_config()) for name in names
    ]
    return (
        ttnn.ProgramSpec(
            name="mcast_binding",
            kernels=kernels,
            work_units=[ttnn.WorkUnitSpec(name="main", kernels=list(names), target_nodes=grid)],
        ),
        ttnn.ProgramRunArgs(),
    )


def _family(device, kind):
    if kind == "1d":
        return ttnn.Mcast1D(device, _grid(), ttnn.Mcast1DShape.PerRow, ttnn.Mcast1DFixedSenderConfig())
    if kind == "2d":
        return ttnn.Mcast2D(device, _grid(), ttnn.Mcast2DFixedSenderConfig(ttnn.CoreCoord(0, 0)))
    family = ttnn.McastFamily(device)
    family.add_group(_grid(), [ttnn.CoreCoord(0, 0)])
    family.prepare_arguments()
    return family


@pytest.mark.parametrize("kind", ["family", "1d", "2d"])
def test_native_spec_attach_mutates_python_objects(device, kind):
    spec, run_args = _spec(_grid())
    family = _family(device, kind)
    family.attach(spec, run_args, "row", ["reader"])
    first_size = spec.kernels[0].advanced_options.num_runtime_varargs
    first_payload = {
        core: list(values) for core, values in run_args.kernel_run_args[0].advanced_options.runtime_varargs.items()
    }
    assert first_size > 0
    assert spec.kernels[0].compile_time_args["row_mcast_rt_base"] == 0
    assert {str(s.unique_id) for s in spec.semaphores} == {"row_mcast_data_ready", "row_mcast_consumer_ready"}
    assert set(first_payload) == {ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0)}
    family.attach(spec, run_args, "second", ["reader"])
    assert spec.kernels[0].compile_time_args["second_mcast_rt_base"] == first_size
    assert spec.kernels[0].advanced_options.num_runtime_varargs == 2 * first_size
    for core, values in run_args.kernel_run_args[0].advanced_options.runtime_varargs.items():
        assert len(values) == 2 * first_size
        assert list(values)[:first_size] == first_payload[core]


@pytest.mark.parametrize("kind", ["family", "1d", "2d"])
def test_spec_adopts_string_resource_names(device, kind):
    spec, run_args = _spec(_grid(), ("reader", "peer"))
    spec.semaphores = [ttnn.SemaphoreSpec(unique_id=name, target_nodes=_grid()) for name in ("ready", "consumed")]
    _family(device, kind).attach(spec, run_args, "row", ["reader", "peer"], adopted_semaphores=["ready", "consumed"])
    assert len(spec.semaphores) == 2
    assert {str(args.kernel) for args in run_args.kernel_run_args} == {"reader", "peer"}
    for kernel in spec.kernels:
        assert [str(binding.semaphore_spec_name) for binding in kernel.semaphore_bindings] == ["ready", "consumed"]
    ttnn.attach_absent(spec, "unused", ["reader", "peer"])
    assert len(spec.semaphores) == 2
    for kernel in spec.kernels:
        assert kernel.compile_time_args["unused_mcast_tag"] == 0


def test_failed_spec_attach_preserves_python_objects(device, expect_error):
    spec, run_args = _spec(_grid())
    family = _family(device, "family")
    family.attach(spec, run_args, "row", ["reader"])
    before_hash = ttnn.compute_program_spec_hash(spec)
    before_args = {
        core: list(values) for core, values in run_args.kernel_run_args[0].advanced_options.runtime_varargs.items()
    }
    with expect_error(RuntimeError, "already in use"):
        family.attach(spec, run_args, "row", ["reader"])
    assert ttnn.compute_program_spec_hash(spec) == before_hash
    assert {
        core: list(values) for core, values in run_args.kernel_run_args[0].advanced_options.runtime_varargs.items()
    } == before_args
    with expect_error(RuntimeError, "Unknown multicast attachment kernel"):
        family.attach(spec, run_args, "other", ["missing"])
    assert ttnn.compute_program_spec_hash(spec) == before_hash


def test_external_sender_placement_and_varargs(device):
    sender = ttnn.CoreCoord(2, 0)
    family = ttnn.McastFamily(device)
    family.add_group(_grid(), [sender])
    family.prepare_arguments()
    nodes = family.participating_cores()
    assert nodes.num_cores() == 3
    assert family.sender_only_cores().contains(sender)
    spec, run_args = _spec(nodes)
    family.attach(spec, run_args, "row", ["reader"])
    for semaphore in spec.semaphores:
        assert semaphore.target_nodes.contains(sender)
    assert set(run_args.kernel_run_args[0].advanced_options.runtime_varargs) == set(
        ttnn.corerange_to_cores(nodes, None, True)
    )
