# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""moe_compute FullCcl on a 1x4 line (FABRIC_1D, Linear, cluster_axis=1): the expert rows path against the ring.

Every moe_compute call of the 6U runner (test_moe_compute_6U._run_moe_compute_impl, eager) runs twice on the same
inputs: the ring (TT_METAL_MOE_COMPUTE_KERNEL=ring), then the expert rows programs (=expert_rows), each after the combine
output is filled with NaN. The two combine outputs must be equal bit for bit, the first call of each must add its own
programs to the cache (ring 1, expert rows 2: the expert rows program, then the placement program; one expert rows
program serves every layer), and the runner checks the expert rows call's outputs against its goldens. One case drops
the runner's combine output and cross-device semaphore, so each call makes its own and runs the combine's init barrier,
over every link the fabric reports. A traced case runs the expert rows path alone against the goldens. Blackhole only:
Wormhole takes the expert rows path only when forced.
"""

import pytest
import torch
import ttnn

from ttnn.operations.ccl import MoEActivationFunction, Topology
from ttnn.experimental.moe_compute_utils import auto_output_width_shard_dim

from tests.nightly.tg.ccl.moe.test_moe_compute_6U import _run_moe_compute_impl
from tests.ttnn.unit_tests.operations.ccl.blackhole_CI.box.nightly.test_moe_compute_multi_card import (
    DEVICE_PARAMS_FABRIC_1D,
    MESH_1X4,
    _skip_unless_line_of_four_with_fabric,
)

KERNEL_ENV = "TT_METAL_MOE_COMPUTE_KERNEL"

# id, intermediate size N, hidden size, top-k, experts per device, tokens per device, layers, activation, bias.
# Four experts per device (16 over the line) so that top-8 leaves experts empty and uneven.
_SHAPES = [
    ("h2048_n512_k8_e4_t32_silu", 512, 2048, 8, 4, 32, 2, MoEActivationFunction.SILU, False),
    ("h2880_n2880_k4_e4_t32_swiglu_bias", 2880, 2880, 4, 4, 32, 2, MoEActivationFunction.SWIGLU, True),
    ("h2816_n704_k8_e4_t32_gelu", 704, 2816, 8, 4, 32, 2, MoEActivationFunction.GELU, False),
    ("h7168_n2048_k4_e2_t8_silu", 2048, 7168, 4, 2, 8, 1, MoEActivationFunction.SILU, False),
]


class _RingThenExpertRows:
    """Stands in for ttnn.experimental.moe_compute: the ring, then the expert rows path on the same inputs; returns the
    expert rows call's outputs. own_barriers: the calls drop the caller's combine output and cross-device semaphore,
    so moe_compute makes its own and runs the combine's init barrier."""

    def __init__(self, mesh_device, monkeypatch, own_barriers=False):
        self.mesh_device = mesh_device
        self.monkeypatch = monkeypatch
        self.own_barriers = own_barriers
        self.moe_compute = ttnn.experimental.moe_compute
        self.programs_added = {"ring": [], "expert_rows": []}

    def _combine(self, tensor):
        return ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=1))

    def _fill_nan(self, tensor):
        shape = list(tensor.shape)  # one device's shard of dim 1
        shape[1] *= self.mesh_device.get_num_devices()
        host = torch.full(shape, float("nan"), dtype=torch.bfloat16)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                host,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=1),
            ),
            tensor,
        )

    def __call__(self, *args, **kwargs):
        if self.own_barriers:
            kwargs = {
                k: v
                for k, v in kwargs.items()
                if k not in ("optional_output_tensor", "optional_cross_device_semaphore")
            }
        combined = {}
        for kernel in ("ring", "expert_rows"):
            if not self.own_barriers:
                self._fill_nan(kwargs["optional_output_tensor"])
            self.monkeypatch.setenv(KERNEL_ENV, kernel)
            before = self.mesh_device.num_program_cache_entries()
            outputs = self.moe_compute(*args, **kwargs)
            ttnn.synchronize_device(self.mesh_device)
            self.programs_added[kernel].append(self.mesh_device.num_program_cache_entries() - before)
            combined[kernel] = self._combine(outputs[5]).view(torch.int16)
            if kernel == "ring":
                # outputs 3 and 4 alias; a combine output the call made itself stays allocated until the expert rows
                # call has made its own (a fresh allocation cannot hold the ring's rows)
                for index in (0, 1, 2, 4):
                    ttnn.deallocate(outputs[index])
                ring_combine = outputs[5]
        if self.own_barriers:
            ttnn.deallocate(ring_combine)
        differ = combined["ring"] != combined["expert_rows"]
        assert not differ.any(), f"{int(differ.sum())} of {differ.numel()} combine values differ from the ring"
        return outputs


def _run(mesh_device, mesh_shape, shape, enable_trace, num_links=1):
    _, N, hidden_size, k, experts_per_device, tokens_per_device, layers, activation, has_bias = shape
    _run_moe_compute_impl(
        mesh_device=mesh_device,
        mesh_shape=mesh_shape,
        cluster_axis=1,
        experts_per_device=experts_per_device,
        tokens_per_device=tokens_per_device,
        selected_experts_k=k,
        num_layers=layers,
        num_iterations=2,
        N=N,
        hidden_size=hidden_size,
        output_height_shard_dim=4,
        output_width_shard_dim=auto_output_width_shard_dim(hidden_size),
        dtype=ttnn.bfloat16,
        enable_trace=enable_trace,
        activation_type=activation,
        has_bias=has_bias,
        topology=Topology.Linear,
        num_links=num_links,
        mux_core_range=((1, 1), (3, 3)),
    )


def _skip_unless_blackhole_line(mesh_device):
    _skip_unless_line_of_four_with_fabric(mesh_device)
    if mesh_device.arch() != ttnn.device.Arch.BLACKHOLE:
        pytest.skip("Wormhole takes the expert rows path only when forced")


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_FABRIC_1D], ids=["fabric_1d"], indirect=True)
@pytest.mark.parametrize("mesh_shape, mesh_device", MESH_1X4, ids=["1x4"], indirect=["mesh_device"])
@pytest.mark.parametrize("shape", _SHAPES, ids=[s[0] for s in _SHAPES])
def test_moe_compute_expert_rows_matches_ring_1x4(mesh_device, mesh_shape, shape, monkeypatch):
    _skip_unless_blackhole_line(mesh_device)
    ring_then_expert_rows = _RingThenExpertRows(mesh_device, monkeypatch)
    monkeypatch.setattr(ttnn.experimental, "moe_compute", ring_then_expert_rows)
    _run(mesh_device, mesh_shape, shape, enable_trace=False)
    added = ring_then_expert_rows.programs_added
    assert added["ring"][0] == 1, f"the ring's first call added {added['ring'][0]} programs"
    assert added["expert_rows"] == [2] + [0] * (len(added["expert_rows"]) - 1), added


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_FABRIC_1D], ids=["fabric_1d"], indirect=True)
@pytest.mark.parametrize("mesh_shape, mesh_device", MESH_1X4, ids=["1x4"], indirect=["mesh_device"])
def test_moe_compute_expert_rows_own_barriers_all_links_1x4(mesh_device, mesh_shape, monkeypatch):
    _skip_unless_blackhole_line(mesh_device)
    ring_then_expert_rows = _RingThenExpertRows(mesh_device, monkeypatch, own_barriers=True)
    monkeypatch.setattr(ttnn.experimental, "moe_compute", ring_then_expert_rows)
    _run(mesh_device, mesh_shape, _SHAPES[0], enable_trace=False, num_links=None)


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_FABRIC_1D], ids=["fabric_1d"], indirect=True)
@pytest.mark.parametrize("mesh_shape, mesh_device", MESH_1X4, ids=["1x4"], indirect=["mesh_device"])
def test_moe_compute_expert_rows_trace_1x4(mesh_device, mesh_shape, monkeypatch):
    _skip_unless_blackhole_line(mesh_device)
    monkeypatch.setenv(KERNEL_ENV, "expert_rows")
    _run(mesh_device, mesh_shape, _SHAPES[0], enable_trace=True)
