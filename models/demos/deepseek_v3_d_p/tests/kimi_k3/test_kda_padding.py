# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""K3's layout bridge, runtime bounds and persistent carries on a synthetic KDA layer.

A start-0 case is a new request: it must run from a zero carry whatever the slot holds, and nothing
clears the slot between cases.
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.kda import KDAConfig, KDAReferenceState, kda_forward_reference
from models.demos.deepseek_v3_d_p.tests.kda.utils import (
    LayerNrmseThresholds,
    assert_matches_reference,
    build_layer,
    kda_device_params,
    mla_row_permutation,
    random_weights,
    read_slab_layer,
    reconstruct_convolution_at_sp_rank,
    reconstruct_sp_tp_tensor,
    reconstruct_state_at_sp_rank,
    to_sp_input,
)
from models.demos.deepseek_v3_d_p.tests.kimi_k3.kda_slabs import run_kda_chunk, zero_state_like
from models.demos.deepseek_v3_d_p.tt.kda.state_adapter import (
    KdaContractGeometry,
    KdaStates,
    allocate_native_state,
    deallocate_state,
    device_scalar,
)
from models.demos.deepseek_v3_d_p.tt.kda.state_contract import KDA_CONVOLUTION_PAGE_CHANNELS
from models.demos.deepseek_v3_d_p.tt.kimi_k3.attention import TtK3KdaAttention
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import assert_bit_identical, make_actual_start

pytestmark = run_for_blackhole()


@pytest.mark.parametrize(
    "mesh_device,sp_axis,tp_axis,device_params",
    [
        ((1, 8), 0, 1, kda_device_params((1, 8))),
        ((2, 4), 0, 1, kda_device_params((2, 4))),
        ((2, 4), 1, 0, kda_device_params((2, 4))),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_k3_padding_eager_and_trace_continue_exact_carries(mesh_device, sp_axis, tp_axis, device_params):
    sp = tuple(mesh_device.shape)[sp_axis]
    local_rows = 256
    capacity = sp * local_rows
    # head_dim 128, as Kimi-K3's: the carries live in the migration slabs, whose segment geometry needs it.
    config = KDAConfig(hidden_size=256, num_heads=8, head_k_dim=128, head_v_dim=128, conv_kernel_size=4, norm_eps=1e-5)
    weights = random_weights(config)
    hidden = torch.randn(1, capacity, config.hidden_size, generator=torch.Generator().manual_seed(197)).bfloat16()
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
    geometry = KdaContractGeometry.from_kda_config(
        config, mesh_shape=tuple(mesh_device.shape), sp_axis=sp_axis, tp_axis=tp_axis
    )
    slabs = KdaStates.allocate(mesh_device, geometry, layer_ids=(1,), num_slots=1)
    scratch = allocate_native_state(mesh_device, geometry)
    attention = TtK3KdaAttention(layer, 1, tp_axis, 1, ttnn.Topology.Linear, scratch)
    slot_tt = device_scalar(mesh_device, 0)

    def committed_carry():
        return read_slab_layer(mesh_device, slabs, 0, 1)

    dims = [None, None]
    dims[sp_axis], dims[tp_axis] = 2, 3

    def upload(value):
        return ttnn.from_torch(
            value.unsqueeze(0),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=tuple(dims), mesh_shape=tuple(mesh_device.shape)),
        )

    input_tt = upload(hidden)
    start_tt, end_tt = device_scalar(mesh_device, 0), device_scalar(mesh_device, capacity)
    metadata = (slot_tt, start_tt, end_tt)
    # Nonzero carried state before the short intervals, including ends just past
    # the head or a device boundary and a one-tile separated tail on SP ranks.
    cases = [(0, capacity), (32, 32), (96, 64), (32, 96), (32, 224), (0, min(capacity, 288))]
    if sp > 1:
        cases += [(96, capacity - 64)]
    # Unaligned prompt ends, continued by the following cases from their carries,
    # including end segments with fewer than three valid rows.
    cases += [(0, 33), (96, capacity - 1), (32, 1), (0, 2)]
    if sp > 1:
        cases += [(32, capacity - 31), (0, local_rows + 2)]
    cases += [(0, capacity)]

    def snapshots(output, permutation, length, state):
        physical = reconstruct_sp_tp_tensor(output, mesh_device, sp_axis, tp_axis, tp_dim=2, sp_dim=1)
        valid_output = physical[:, torch.argsort(permutation)][:, :length].clone()
        return [valid_output] + [
            ttnn.to_torch(shard).clone()
            for tensor in (state.recurrent, state.convolution)
            for shard in ttnn.get_device_tensors(tensor)
        ]

    eager_results = []
    trace = None
    traced_output = None
    try:
        for traced in (False, True):
            if traced:
                for _ in range(2):
                    ttnn.deallocate(run_kda_chunk(attention, input_tt, slabs, metadata))
                trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
                traced_output = run_kda_chunk(attention, input_tt, slabs, metadata)
                ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
            # Nothing clears the slot: the eager pass starts on the uninitialized slab, the traced one on
            # what warm-up and capture left.
            for case_index, (start, length) in enumerate(cases):
                permutation = mla_row_permutation(start, sp, local_rows)
                # Keep nonzero padded data: neither K3 nor the caller must need
                # to zero it to get the right final state.
                source = upload(hidden[:, permutation])
                ttnn.copy(source, input_tt)
                ttnn.deallocate(source)
                # Grade each transition from its actual incoming carry, retaining
                # the existing error gates without accumulating prior BF16 error. A
                # request head's incoming carry is zero, whatever the slot holds.
                incoming = committed_carry() if start else zero_state_like(scratch)
                history = reconstruct_convolution_at_sp_rank(
                    incoming.convolution,
                    mesh_device,
                    sp_axis,
                    tp_axis,
                    0,
                    config,
                    layer.convolution_page_channels,
                ).float()
                reference_state = KDAReferenceState(
                    recurrent=reconstruct_state_at_sp_rank(
                        incoming.recurrent, mesh_device, sp_axis, tp_axis, 0
                    ).float(),
                    q_convolution=history[..., : config.q_dim],
                    k_convolution=history[..., config.q_dim : config.q_dim + config.k_dim],
                    v_convolution=history[..., config.q_dim + config.k_dim :],
                )
                expected, reference_state = kda_forward_reference(hidden[:, :length], weights, config, reference_state)
                if not traced:
                    native_input = to_sp_input(hidden[:, permutation], mesh_device, sp_axis)
                    native_start = make_actual_start(mesh_device, start)
                    native_end = make_actual_start(mesh_device, start + length)
                    native_output, native_state = layer.forward(native_input, incoming, native_start, native_end)
                    native_result = snapshots(native_output, permutation, length, native_state)
                    for tensor in (
                        native_input,
                        native_start,
                        native_end,
                        native_output,
                        native_state.recurrent,
                        native_state.convolution,
                    ):
                        ttnn.deallocate(tensor)
                deallocate_state(incoming)
                # Eager and traced chunks read the same persistent bounds, rewritten before every chunk.
                for destination, value in ((start_tt, start), (end_tt, start + length)):
                    source = device_scalar(mesh_device, value)
                    ttnn.copy(source, destination)
                    ttnn.deallocate(source)
                if traced:
                    ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
                    output = traced_output
                else:
                    output = run_kda_chunk(attention, input_tt, slabs, metadata)
                committed = committed_carry()
                assert_matches_reference(
                    output_tt=output,
                    state=committed,
                    permutation=permutation,
                    expected_output=expected.bfloat16(),
                    expected_state=reference_state,
                    mesh_device=mesh_device,
                    sp_axis=sp_axis,
                    tp_axis=tp_axis,
                    config=config,
                    page_channels=layer.convolution_page_channels,
                    # The clean value measured at head_dim 128 is 2.6 on every mesh: rel_Linf is the
                    # largest element error over the state's RMS, so it scales with the state (0.6 was
                    # tuned at head_dim 32 and already needed a 0.604 exemption). A localised carry
                    # fault is caught exactly by the native/K3 and eager/trace bit-identity checks.
                    state_linf_threshold=3.0,
                    label=f"K3 {'trace' if traced else 'eager'} SP{sp} start={start} length={length}",
                    nrmse=LayerNrmseThresholds(output=0.02, recurrent=0.03, convolution=0.004),
                )
                observed = snapshots(output, permutation, length, committed)
                deallocate_state(committed)
                if traced:
                    for eager, replayed in zip(eager_results[case_index], observed, strict=True):
                        assert_bit_identical(eager, replayed, name="K3 eager versus traced continuation")
                else:
                    for native, adapted in zip(native_result, observed, strict=True):
                        assert_bit_identical(native, adapted, name="K3 versus direct KDA")
                    eager_results.append(observed)
                    ttnn.deallocate(output)
    finally:
        if trace is not None:
            ttnn.release_trace(mesh_device, trace)
            ttnn.deallocate(traced_output)
        deallocate_state(scratch)
        slabs.deallocate()
        for tensor in (input_tt, slot_tt, start_tt, end_tt):
            ttnn.deallocate(tensor)
