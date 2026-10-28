# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One captured K3 KDA forward serves every user slot (#59977).

Two users alternate chunks on one layer's two slots. The capture is recorded once, on slot 0, and every
replay takes its slot from the metadata the runtime refreshes per chunk, as `TtPrefillRuntime` does. A
replay must read and advance only that slot's carry and slab entry, and match the eager path bit for
bit. A slot-0-only test cannot see the defect this guards: a capture that pins its slot is exact there.
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.kda import KDAConfig
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric_1d_plain_device_params, torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.kda.utils import (
    build_layer,
    mla_row_permutation,
    random_weights,
    read_slab_layer,
    reconstruct_sp_tp_tensor,
)
from models.demos.deepseek_v3_d_p.tt.kda.state_adapter import (
    KdaContractGeometry,
    KdaStates,
    allocate_native_state,
    deallocate_state,
    device_scalar,
)
from models.demos.deepseek_v3_d_p.tt.kimi_k3.attention import K3AttnContext, TtK3KdaAttention, build_kda_scalars
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import assert_bit_identical, make_actual_start

pytestmark = run_for_blackhole()

LAYER = 1
NUM_SLOTS = 2
# (slot, actual_start, length): users A (slot 0) and B (slot 1) alternate, each continuing its own carry.
CHUNKS = [(0, 0, 128), (1, 0, 160), (0, 128, 96), (1, 160, 64)]


def _host(tensor: ttnn.Tensor) -> list[torch.Tensor]:
    return [ttnn.to_torch(shard).clone() for shard in ttnn.get_device_tensors(tensor)]


@pytest.mark.parametrize(
    "mesh_device,sp_axis,tp_axis,device_params",
    [
        pytest.param((1, 8), 0, 1, fabric_1d_plain_device_params(), id="sp1tp8"),
        pytest.param((2, 4), 0, 1, fabric_1d_plain_device_params(), id="sp2tp4"),
        # Galaxy runs only the production 8x4 TorusXY mesh.
        pytest.param(
            (8, 4),
            0,
            1,
            torus_xy_device_params(),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="sp8tp4-torus-xy",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_one_capture_serves_every_slot(mesh_device, sp_axis, tp_axis, device_params):
    mesh_shape = tuple(mesh_device.shape)
    sp = mesh_shape[sp_axis]
    local_rows = 256
    capacity = sp * local_rows
    # head_dim 128, as Kimi-K3's, so the migration slabs' segment geometry applies and every commit exports.
    config = KDAConfig(hidden_size=256, num_heads=8, head_k_dim=128, head_v_dim=128, conv_kernel_size=4, norm_eps=1e-5)
    weights = random_weights(config)
    layer = build_layer(mesh_device, config, weights, sp_axis, tp_axis, active_seq_len=capacity, summary_group_chunks=2)
    geometry = KdaContractGeometry.from_kda_config(config, mesh_shape=mesh_shape, sp_axis=sp_axis, tp_axis=tp_axis)
    slabs = KdaStates.allocate(mesh_device, geometry, layer_ids=(LAYER,), num_slots=NUM_SLOTS)
    scratch = allocate_native_state(mesh_device, geometry)
    attention = TtK3KdaAttention(layer, LAYER, tp_axis, 1, ttnn.Topology.Linear, scratch)
    dims = [None, None]
    dims[sp_axis], dims[tp_axis] = 2, 3

    def upload(value):
        return ttnn.from_torch(
            value.unsqueeze(0),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=tuple(dims), mesh_shape=mesh_shape),
        )

    def chunk_input(index, start):
        hidden = torch.randn(1, capacity, config.hidden_size, generator=torch.Generator().manual_seed(300 + index))
        return hidden.bfloat16()[:, mla_row_permutation(start, sp, local_rows)]

    def carries():
        snapshots = []
        for slot in range(NUM_SLOTS):
            state = read_slab_layer(mesh_device, slabs, slot, LAYER)
            snapshots.append(_host(state.recurrent) + _host(state.convolution))
            deallocate_state(state)
        return snapshots

    def observe(output, start, length):
        physical = reconstruct_sp_tp_tensor(output, mesh_device, sp_axis, tp_axis, tp_dim=2, sp_dim=1)
        permutation = mla_row_permutation(start, sp, local_rows)
        valid = physical[:, torch.argsort(permutation)][:, :length].clone()
        return {"output": [valid], "carries": carries(), "slabs": _host(slabs.recurrent) + _host(slabs.convolution)}

    input_tt = upload(chunk_input(0, 0))
    slot_tt, start_tt, end_tt = (
        device_scalar(mesh_device, 0),
        make_actual_start(mesh_device, 0),
        make_actual_start(mesh_device, capacity),
    )
    trace = None
    traced_output = None
    try:
        # Eager reference: the slot is a device scalar built from the host slot, the same path as traced.
        eager = []
        for index, (slot, start, length) in enumerate(CHUNKS):
            source = upload(chunk_input(index, start))
            ttnn.copy(source, input_tt)
            ttnn.deallocate(source)
            before = carries()
            scalars, owned = build_kda_scalars(mesh_device, None, slot, start, start + length)
            output = attention.forward(input_tt, K3AttnContext(kda_scalars=scalars, kda_states=slabs))
            for scalar in owned:
                ttnn.deallocate(scalar)
            observed = observe(output, start, length)
            ttnn.deallocate(output)
            other = 1 - slot
            for got, want in zip(observed["carries"][other], before[other], strict=True):
                assert_bit_identical(want, got, name=f"eager chunk {index}: slot {other} must be untouched")
            assert any(
                not torch.equal(got, want) for got, want in zip(observed["carries"][slot], before[slot], strict=True)
            ), f"eager chunk {index}: slot {slot} did not advance"
            eager.append(observed)

        # One capture, recorded on slot 0. Warm until program and resource state settle first.
        ctx = K3AttnContext(kda_scalars=(slot_tt, start_tt, end_tt), kda_states=slabs)
        for _ in range(2):
            ttnn.deallocate(attention.forward(input_tt, ctx))
        trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        traced_output = attention.forward(input_tt, ctx)
        ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
        for slot in range(NUM_SLOTS):
            slabs.zero_slot(slot)

        for index, (slot, start, length) in enumerate(CHUNKS):
            source = upload(chunk_input(index, start))
            ttnn.copy(source, input_tt)
            ttnn.deallocate(source)
            for destination, value in ((start_tt, start), (end_tt, start + length)):
                staged = make_actual_start(mesh_device, value)
                ttnn.copy(staged, destination)
                ttnn.deallocate(staged)
            staged = device_scalar(mesh_device, slot)
            ttnn.copy(staged, slot_tt)
            ttnn.deallocate(staged)
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
            observed = observe(traced_output, start, length)
            for key in ("output", "slabs"):
                for shard, (want, got) in enumerate(zip(eager[index][key], observed[key], strict=True)):
                    assert_bit_identical(want, got, name=f"traced chunk {index} slot {slot} {key} {shard}")
            for carry_slot in range(NUM_SLOTS):
                for shard, (want, got) in enumerate(
                    zip(eager[index]["carries"][carry_slot], observed["carries"][carry_slot], strict=True)
                ):
                    assert_bit_identical(want, got, name=f"traced chunk {index} carry slot {carry_slot} {shard}")
    finally:
        if trace is not None:
            ttnn.release_trace(mesh_device, trace)
        if traced_output is not None:
            ttnn.deallocate(traced_output)
        deallocate_state(scratch)
        slabs.deallocate()
        for tensor in (input_tt, slot_tt, start_tt, end_tt):
            ttnn.deallocate(tensor)
