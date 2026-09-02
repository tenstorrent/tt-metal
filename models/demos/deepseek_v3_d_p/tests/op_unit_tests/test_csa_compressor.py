# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device tests for the fused Blaze-compatible CSA compressor."""

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params
from tests.ttnn.utils_for_testing import assert_with_pcc

_BATCH = 1
_COMPRESS_RATE = 4
_HEAD_DIM = 512
_STATE_ROWS = 64
_LOCAL_SEQ_LEN = 128
_PCC = 0.999


def _update_state(kv_state, score_state, kv, gate, position_bias, start_position):
    kv_state = kv_state.clone()
    score_state = score_state.clone()
    for local_position in range(kv.shape[2]):
        position = start_position + local_position
        slot = position % _COMPRESS_RATE
        parity = (position // _COMPRESS_RATE) & 1
        ca_row = (parity ^ 1) * 32 + slot
        cb_row = parity * 32 + _COMPRESS_RATE + slot
        biased_gate = gate[:, :, local_position] + position_bias[:, :, slot]
        kv_state[:, :, ca_row] = kv[:, :, local_position, :_HEAD_DIM]
        kv_state[:, :, cb_row] = kv[:, :, local_position, _HEAD_DIM:]
        score_state[:, :, ca_row] = biased_gate[..., :_HEAD_DIM]
        score_state[:, :, cb_row] = biased_gate[..., _HEAD_DIM:]
    return kv_state, score_state


def _compress_local(kv, gate, position_bias, predecessor_kv, predecessor_score, start_position):
    n_windows = kv.shape[2] // _COMPRESS_RATE
    pooled = torch.zeros(_BATCH, 1, n_windows, _HEAD_DIM, dtype=torch.bfloat16)
    for window in range(n_windows):
        current_start = window * _COMPRESS_RATE
        current_end = current_start + _COMPRESS_RATE
        absolute_start = start_position + current_start
        current_kv = kv[:, :, current_start:current_end]
        current_gate = gate[:, :, current_start:current_end] + position_bias

        if window == 0:
            parity = (absolute_start // _COMPRESS_RATE) & 1
            state_start = parity * 32
            previous_ca_kv = predecessor_kv[:, :, state_start : state_start + _COMPRESS_RATE]
            previous_ca_score = predecessor_score[:, :, state_start : state_start + _COMPRESS_RATE]
        else:
            previous_start = current_start - _COMPRESS_RATE
            previous_end = current_start
            previous_ca_kv = kv[:, :, previous_start:previous_end, :_HEAD_DIM]
            previous_ca_score = gate[:, :, previous_start:previous_end, :_HEAD_DIM] + position_bias[..., :_HEAD_DIM]

        overlap_kv = torch.cat([previous_ca_kv, current_kv[..., _HEAD_DIM:]], dim=2)
        overlap_score = torch.cat([previous_ca_score, current_gate[..., _HEAD_DIM:]], dim=2)
        weights = overlap_score.softmax(dim=2, dtype=torch.float32).to(overlap_kv.dtype)
        pooled[:, :, window] = (overlap_kv * weights).sum(dim=2)
    return pooled


def _torch_csa_compressor(
    kv,
    gate,
    position_bias,
    initial_kv_state,
    initial_score_state,
    sp_factor,
    seq_len_actual,
    first_token_position,
):
    local_seq_len = kv.shape[2] // sp_factor
    pooled_outputs = []
    kv_states = []
    score_states = []
    predecessor_kv = initial_kv_state
    predecessor_score = initial_score_state

    for rank in range(sp_factor):
        local_start = rank * local_seq_len
        local_end = local_start + local_seq_len
        local_valid = max(0, min(local_seq_len, seq_len_actual - local_start))
        local_position = first_token_position + local_start
        local_kv = kv[:, :, local_start:local_end]
        local_gate = gate[:, :, local_start:local_end]

        complete_tokens = local_valid // _COMPRESS_RATE * _COMPRESS_RATE
        local_pooled = _compress_local(
            local_kv[:, :, :complete_tokens],
            local_gate[:, :, :complete_tokens],
            position_bias,
            predecessor_kv,
            predecessor_score,
            local_position,
        )
        padded_pooled = torch.zeros(_BATCH, 1, local_seq_len // _COMPRESS_RATE, _HEAD_DIM, dtype=torch.bfloat16)
        padded_pooled[:, :, : local_pooled.shape[2]] = local_pooled
        pooled_outputs.append(padded_pooled)

        local_kv_state, local_score_state = _update_state(
            predecessor_kv,
            predecessor_score,
            local_kv[:, :, :local_valid],
            local_gate[:, :, :local_valid],
            position_bias,
            local_position,
        )
        kv_states.append(local_kv_state)
        score_states.append(local_score_state)
        predecessor_kv = local_kv_state
        predecessor_score = local_score_state

    return (
        torch.cat(pooled_outputs, dim=2),
        torch.cat(kv_states, dim=2),
        torch.cat(score_states, dim=2),
    )


def _make_inputs(sp_factor, local_seq_len, remainder, first_token_position):
    torch.manual_seed(42)
    padded_seq_len = local_seq_len * sp_factor
    seq_len_actual = padded_seq_len - _COMPRESS_RATE + remainder
    kv = torch.randn(_BATCH, 1, padded_seq_len, 2 * _HEAD_DIM, dtype=torch.bfloat16)
    gate = torch.randn_like(kv)
    position_bias = torch.randn(1, 1, _COMPRESS_RATE, 2 * _HEAD_DIM, dtype=torch.bfloat16)
    initial_kv_state = torch.randn(_BATCH, 1, _STATE_ROWS, _HEAD_DIM, dtype=torch.bfloat16)
    initial_score_state = torch.randn_like(initial_kv_state)
    expected = _torch_csa_compressor(
        kv,
        gate,
        position_bias,
        initial_kv_state,
        initial_score_state,
        sp_factor,
        seq_len_actual,
        first_token_position,
    )
    return kv, gate, position_bias, initial_kv_state, initial_score_state, seq_len_actual, expected


def _run_csa_compressor(mesh_device, local_seq_len, remainder, first_token_position):
    mesh_shape = tuple(mesh_device.shape)
    sp_factor, tp_factor = mesh_shape
    kv, gate, bias, initial_kv, initial_score, seq_len_actual, expected = _make_inputs(
        sp_factor, local_seq_len, remainder, first_token_position
    )
    sp_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=(2, None))
    replicated_mapper = ttnn.ReplicateTensorToMesh(mesh_device)

    def to_device(tensor, mapper):
        return ttnn.from_torch(
            tensor,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

    pooled, kv_state, score_state = ttnn.experimental.deepseek_prefill.csa_compressor(
        to_device(kv, sp_mapper),
        to_device(gate, sp_mapper),
        to_device(bias, replicated_mapper),
        to_device(initial_kv.repeat(1, 1, sp_factor, 1), sp_mapper),
        to_device(initial_score.repeat(1, 1, sp_factor, 1), sp_mapper),
        seq_len_actual=seq_len_actual,
        first_token_position=first_token_position,
        cluster_axis=0,
        topology=ttnn.Topology.Linear,
    )

    composer = ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_shape, dims=(2, 1))
    actual = [ttnn.to_torch(tensor, mesh_composer=composer) for tensor in (pooled, kv_state, score_state)]
    expected_pooled, expected_kv, expected_score = expected
    for tp_rank in range(tp_factor):
        actual_pooled = actual[0][:, tp_rank : tp_rank + 1]
        passed, message = assert_with_pcc(expected_pooled.float(), actual_pooled.float(), pcc=_PCC)
        assert passed, f"CSA compressor PCC failed: {message}"
        assert torch.equal(actual[1][:, tp_rank : tp_rank + 1], expected_kv)
        assert torch.equal(actual[2][:, tp_rank : tp_rank + 1], expected_score)


@pytest.mark.parametrize("first_token_position", [0, _COMPRESS_RATE])
@pytest.mark.parametrize("remainder", range(_COMPRESS_RATE))
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (1, 1),
            {"fabric_config": ttnn.FabricConfig.DISABLED},
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(1, 1), topology="mesh-1x1"),
            id="1x1",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_csa_compressor_single_device(mesh_device, device_params, remainder, first_token_position):
    _run_csa_compressor(mesh_device, _LOCAL_SEQ_LEN, remainder, first_token_position)


@pytest.mark.parametrize("first_token_position", [0, _COMPRESS_RATE])
@pytest.mark.parametrize("remainder", range(_COMPRESS_RATE))
@pytest.mark.parametrize(
    "mesh_device, device_params, local_seq_len",
    [
        pytest.param(
            (2, 2),
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
            _LOCAL_SEQ_LEN,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 2), topology="mesh-2x2"),
            id="fabric1d-2x2",
        ),
        pytest.param(
            (2, 2),
            fabric2d_device_params(),
            16,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 2), topology="mesh-2x2"),
            id="fabric2d-2x2",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_csa_compressor_mesh(mesh_device, device_params, local_seq_len, remainder, first_token_position):
    _run_csa_compressor(mesh_device, local_seq_len, remainder, first_token_position)
