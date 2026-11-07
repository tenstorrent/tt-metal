# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One captured K3 KDA forward serves every user slot and every request on it (#59977).

Two users alternate chunks on one layer's two slots, then a new request starts on slot 0. The capture is
recorded once, on slot 0, and every replay takes its slot and start from the metadata the runtime
refreshes per chunk, as `TtPrefillRuntime` does. A replay must read and advance only that slot's carry
and slab entry, and match the eager path bit for bit. A slot-0-only test cannot see the defect this
guards: a capture that pins its slot is exact there.

Nothing ever clears a slot. Every slot starts as noise, warm-up and capture leave slot 0 dirty, and a
request head must still equal ttKDA run from a zero carry, bit for bit: the head loads zeros on device.
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
    to_sp_input,
)
from models.demos.deepseek_v3_d_p.tests.kimi_k3.kda_slabs import fill_slabs_with_noise, run_kda_chunk, zero_state_like
from models.demos.deepseek_v3_d_p.tt.kda.state_adapter import (
    KdaContractGeometry,
    KdaStates,
    allocate_native_state,
    deallocate_state,
    device_scalar,
)
from models.demos.deepseek_v3_d_p.tt.kda.state_contract import KDA_CONVOLUTION_PAGE_CHANNELS
from models.demos.deepseek_v3_d_p.tt.kimi_k3.attention import TtK3KdaAttention
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import (
    all_zero_bits,
    assert_bit_identical,
    make_actual_start,
)

pytestmark = run_for_blackhole()

LAYER = 1
NUM_SLOTS = 2
# (slot, actual_start, length): users A (slot 0) and B (slot 1) alternate, each continuing its own carry; then
# user C starts a new request on slot 0, over A's carry.
CHUNKS = [(0, 0, 128), (1, 0, 160), (0, 128, 96), (1, 160, 64), (0, 0, 160)]


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
    layer = build_layer(
        mesh_device,
        config,
        weights,
        sp_axis,
        tp_axis,
        active_seq_len=capacity,
        summary_group_chunks=2,
        convolution_page_channels=KDA_CONVOLUTION_PAGE_CHANNELS,
    )
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

    def valid_output(output, start, length):
        physical = reconstruct_sp_tp_tensor(output, mesh_device, sp_axis, tp_axis, tp_dim=2, sp_dim=1)
        permutation = mla_row_permutation(start, sp, local_rows)
        return physical[:, torch.argsort(permutation)][:, :length].clone()

    def fresh_head(index, start, length):
        """The head chunk run by ttKDA itself from a zero carry: what a never-used slot would produce."""
        hidden = to_sp_input(chunk_input(index, start), mesh_device, sp_axis)
        zero = zero_state_like(scratch)
        bounds = [make_actual_start(mesh_device, value) for value in (start, start + length)]
        output, state = layer.forward(hidden, zero, *bounds)
        result = [valid_output(output, start, length)] + _host(state.recurrent) + _host(state.convolution)
        for tensor in (hidden, output, *bounds):
            ttnn.deallocate(tensor)
        deallocate_state(zero)
        deallocate_state(state)
        return result

    input_tt = upload(chunk_input(0, 0))
    metadata = tuple(device_scalar(mesh_device, value) for value in (0, 0, capacity))
    slot_tt, start_tt, end_tt = metadata

    def stage(index, slot, start, length):
        """Rewrite the persistent input and metadata in place, as the runtime does before every chunk."""
        for destination, source in (
            (input_tt, upload(chunk_input(index, start))),
            (slot_tt, device_scalar(mesh_device, slot)),
            (start_tt, device_scalar(mesh_device, start)),
            (end_tt, device_scalar(mesh_device, start + length)),
        ):
            ttnn.copy(source, destination)
            ttnn.deallocate(source)

    def check_chunk(label, index, slot, start, before, after):
        other = 1 - slot
        for got, want in zip(after[other], before[other], strict=True):
            assert_bit_identical(want, got, name=f"{label} chunk {index}: slot {other} must be untouched")
        assert any(
            not torch.equal(got, want) for got, want in zip(after[slot], before[slot], strict=True)
        ), f"{label} chunk {index}: slot {slot} did not advance"
        if start == 0:
            # Nothing reset the slot, so a head that matches the zero-carry run cannot have read it.
            assert not all(
                all_zero_bits(shard) for shard in before[slot]
            ), f"{label} chunk {index}: head slot {slot} was already zero, so the test cannot see a stale read"

    # A stale slot must not pass for a zeroed one: noise in every slot before any request starts.
    fill_slabs_with_noise(mesh_device, slabs, seed=17)
    trace = None
    traced_output = None
    try:
        # The eager pass builds its scalars from host ints, as an eager forward does; the traced pass replays the
        # metadata path and must match it bit for bit.
        eager = []
        for index, (slot, start, length) in enumerate(CHUNKS):
            stage(index, slot, start, length)
            before = carries()
            output = run_kda_chunk(attention, input_tt, slabs, None, slot, start, start + length)
            after = carries()
            observed = [valid_output(output, start, length)] + after[slot]
            ttnn.deallocate(output)
            check_chunk("eager", index, slot, start, before, after)
            if start == 0:
                for shard, (want, got) in enumerate(zip(fresh_head(index, start, length), observed, strict=True)):
                    assert_bit_identical(want, got, name=f"eager head chunk {index} slot {slot} vs zero carry {shard}")
            eager.append(observed)

        # One capture, recorded on slot 0 at a request head. Warm until program and resource state settle
        # first. Nothing resets afterwards: warm-up and capture leave slot 0 dirty, and slot 1 still holds the
        # eager pass's last carry, so every traced head starts on a stale slot.
        stage(0, 0, 0, capacity)
        for _ in range(2):
            ttnn.deallocate(run_kda_chunk(attention, input_tt, slabs, metadata))
        trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        traced_output = run_kda_chunk(attention, input_tt, slabs, metadata)
        ttnn.end_trace_capture(mesh_device, trace, cq_id=0)

        for index, (slot, start, length) in enumerate(CHUNKS):
            stage(index, slot, start, length)
            before = carries()
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
            after = carries()
            observed = [valid_output(traced_output, start, length)] + after[slot]
            check_chunk("traced", index, slot, start, before, after)
            for shard, (want, got) in enumerate(zip(eager[index], observed, strict=True)):
                assert_bit_identical(want, got, name=f"traced chunk {index} slot {slot} shard {shard}")
    finally:
        if trace is not None:
            ttnn.release_trace(mesh_device, trace)
        if traced_output is not None:
            ttnn.deallocate(traced_output)
        deallocate_state(scratch)
        slabs.deallocate()
        for tensor in (input_tt, *metadata):
            ttnn.deallocate(tensor)
