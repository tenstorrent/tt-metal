# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only regressions for the GPT-OSS sampling CCL and force-argmax configuration.

The force-argmax all-gather must be handed semaphores that belong to sampling alone.
``all_gather_async`` maps its two global semaphores to the forward and backward ring
directions, so a handle that also serves the model's own CCLs as a single leaves
per-direction counts behind and desyncs a later gather; on Blackhole that shows up as
argmax indices displaced by whole vocab chunks. ``SamplingCCL`` therefore hands out one
fixed pair per axis and never rotates it. These tests pin that contract and the
configuration that turns the path on. Neither needs a device.
"""

from types import SimpleNamespace

import pytest

import ttnn
from models.demos.gpt_oss.tt.ccl import SamplingCCL
from models.demos.gpt_oss.tt.model import Model

GPT_OSS_VOCAB_SIZE = 201088


def _ccl_without_device():
    """A SamplingCCL with stand-in handles, so the accessor contract is testable on the host.

    ``__init__`` allocates global semaphores and needs a mesh, but every property under
    test here is pool selection and aliasing, which the handles' identity is enough to show.
    """
    ccl = object.__new__(SamplingCCL)
    ccl.mesh_device = None
    ccl._ag_handles = [[f"ag{pool}a", f"ag{pool}b"] for pool in range(3)]
    ccl._barrier_handles = [f"barrier{pool}" for pool in range(3)]
    return ccl


@pytest.mark.parametrize(
    "cluster_axis, expected_pool",
    [
        (1, 1),  # mesh_2x4, mesh_4x4 and mesh_4x8 all gather on axis 1
        (0, 0),
        (None, 2),  # mesh_1x8: a shape holding a 1 gathers without an axis
    ],
)
def test_semaphore_pools_are_selected_by_axis(cluster_axis, expected_pool):
    ccl = _ccl_without_device()
    assert ccl.get_and_cycle_ag_semaphore_handles(cluster_axis) == [f"ag{expected_pool}a", f"ag{expected_pool}b"]
    assert ccl.get_sampling_barrier_semaphore_handle(cluster_axis) == f"barrier{expected_pool}"


def test_axis_none_does_not_collide_with_axis_zero():
    """None must not fold into the axis-0 pool, which is a real axis with its own handles."""
    ccl = _ccl_without_device()
    assert ccl.get_and_cycle_ag_semaphore_handles(None) != ccl.get_and_cycle_ag_semaphore_handles(0)
    assert ccl.get_sampling_barrier_semaphore_handle(None) != ccl.get_sampling_barrier_semaphore_handle(0)


def test_handles_keep_a_fixed_role_across_calls():
    """The pair must not rotate. Rotation is what corrupts the gather, so pin identity.

    A handle that changes direction role between calls, or that is handed out while the
    model's own CCLs hold it, desyncs the gather. Fixed handles are the whole reason this
    class exists rather than a reuse of CCLManager's ping-pong banks.
    """
    ccl = _ccl_without_device()
    for cluster_axis in (0, 1, None):
        first = ccl.get_and_cycle_ag_semaphore_handles(cluster_axis)
        second = ccl.get_and_cycle_ag_semaphore_handles(cluster_axis)
        assert first is second, f"axis {cluster_axis}: all-gather handles rotated between calls"
        assert ccl.get_sampling_barrier_semaphore_handle(cluster_axis) is ccl.get_sampling_barrier_semaphore_handle(
            cluster_axis
        ), f"axis {cluster_axis}: barrier handle rotated between calls"


def test_barrier_default_accessor_aliases_the_sampling_one():
    """TTSampling reads the sampling accessor through getattr with this as the default.

    Python evaluates that default eagerly, so the method has to exist, and it has to give
    back the dedicated handle rather than something shared.
    """
    ccl = _ccl_without_device()
    for cluster_axis in (0, 1, None):
        assert ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis) is ccl.get_sampling_barrier_semaphore_handle(
            cluster_axis
        )


# "shape", not "mesh_shape": a mesh_shape param is what pytest_collection_modifyitems keys
# on elsewhere in the tree to deselect items, and these cases take no device fixture.
@pytest.mark.parametrize("shape", [(2, 4), (4, 4), (1, 8), (4, 8)])
def test_force_argmax_is_configured_for_every_deployable_mesh(shape):
    """Every GPT-OSS mesh configuration must switch greedy decode to the argmax path."""
    mesh_device = SimpleNamespace(shape=list(shape), get_num_devices=lambda: shape[0] * shape[1])
    args = Model._make_sampling_args(
        SimpleNamespace(sampling_dp=1),
        SimpleNamespace(vocab_size=GPT_OSS_VOCAB_SIZE),
        mesh_device,
    )
    config = args.model_config["SAMPLING_AG_CONFIG"]
    assert config["allow_force_argmax"] is True
    assert config["topology"] is ttnn.Topology.Linear
    assert config["num_links"] >= 1
