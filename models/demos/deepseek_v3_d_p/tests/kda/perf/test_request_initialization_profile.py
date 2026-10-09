# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Label request initialization traffic for the safe runner's optional Tracy profile."""

import pytest
import torch
from tracy import signpost

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.kda import KDAConfig
from models.demos.deepseek_v3_d_p.tests.kda.utils import build_layer, random_weights, to_sp_input
from models.demos.deepseek_v3_d_p.tt.kda.config import KDAProgramConfig, KDARecurrenceProgramConfig
from models.demos.deepseek_v3_d_p.tt.kda.kda import ttKDA
from models.demos.deepseek_v3_d_p.tt.kda.state_adapter import KdaContractGeometry, KdaStates
from models.demos.deepseek_v3_d_p.tt.kimi_k3.kda_state import KdaStateCache
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import assert_bit_identical, make_actual_start

pytestmark = run_for_blackhole()


@pytest.mark.parametrize("mesh_device", [pytest.param((2, 4), id="LB-SP2xTP4")], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_request_initialization_programs(mesh_device, device_params):
    config = KDAConfig(hidden_size=256, num_heads=96, head_k_dim=128, head_v_dim=128, conv_kernel_size=4, norm_eps=1e-5)
    enabled = build_layer(
        mesh_device,
        config,
        random_weights(config),
        0,
        1,
        active_seq_len=256,
        summary_group_chunks=4,
        zero_initial_state_on_start=True,
    )
    control = ttKDA(
        mesh_device,
        config,
        weights=enabled.weights,
        tt_ccl=enabled.tt_ccl,
        active_seq_len=256,
        program_config=KDAProgramConfig(
            recurrence=KDARecurrenceProgramConfig(local_scan_strategy="grouped", summary_group_chunks=4),
            gated_rms_output_dtype=ttnn.bfloat16,
            output_projection_math_fidelity=ttnn.MathFidelity.HiFi2,
        ),
    )
    cache = KdaStateCache({0: enabled})
    geometry = KdaContractGeometry.from_kda_config(config, mesh_shape=(2, 4), sp_axis=0, tp_axis=1)
    slabs = KdaStates.allocate(mesh_device, geometry, layer_ids=(0,), num_slots=1)
    cache.bind_slabs(slabs)
    # Historical reset control only; production retains no zero pair.
    zeros = control.allocate_state()
    start = make_actual_start(mesh_device, 0)
    hidden = to_sp_input(
        torch.randn(1, 256, 256, generator=torch.Generator().manual_seed(88)).bfloat16(), mesh_device, 0
    )

    def forward(layer):
        output, state = layer.forward(hidden, cache.read(0), start)
        cache.commit(0, state)
        return output

    def snapshot(output):
        state = cache.read(0)
        return [
            ttnn.to_torch(shard).clone()
            for tensor in (output, state.recurrent, state.convolution)
            for shard in ttnn.get_device_tensors(tensor)
        ]

    try:
        for layer in (control, enabled):
            ttnn.deallocate(forward(layer))
        ttnn.synchronize_device(mesh_device)
        signpost("KDA_LEGACY_RESET_BEGIN")
        current = cache.read(0)
        ttnn.copy(zeros.recurrent, current.recurrent)
        ttnn.copy(zeros.convolution, current.convolution)
        slabs.export_layer(zeros, 0, 0)
        ttnn.synchronize_device(mesh_device)
        signpost("KDA_LEGACY_FORWARD_BEGIN")
        output = forward(control)
        ttnn.synchronize_device(mesh_device)
        signpost("KDA_LEGACY_END")
        expected = snapshot(output)
        ttnn.deallocate(output)

        signpost("KDA_DEVICE_INITIALIZATION_BEGIN")
        output = forward(enabled)
        ttnn.synchronize_device(mesh_device)
        signpost("KDA_DEVICE_INITIALIZATION_END")
        for baseline, actual in zip(expected, snapshot(output), strict=True):
            assert_bit_identical(baseline, actual, name="profile request initialization")
        ttnn.deallocate(output)
    finally:
        cache.deallocate()
        for tensor in (zeros.recurrent, zeros.convolution, slabs.recurrent, slabs.convolution, hidden, start):
            ttnn.deallocate(tensor)
