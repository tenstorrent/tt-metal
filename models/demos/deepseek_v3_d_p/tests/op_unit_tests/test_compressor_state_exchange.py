# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device tests for the Blaze-compatible compressor state exchange."""

import pytest
import torch

import ttnn

_BATCH = 1
_COMPRESS_RATE = 4
_HEAD_DIM = 512
_STATE_ROWS = 64
_LOCAL_SEQ_LEN = 128


def _pack_blaze_state(kv, score, start_position=0):
    """Pack projected Ca/Cb rows using Blaze's two-tile parity layout."""
    kv_state = torch.zeros(_BATCH, 1, _STATE_ROWS, _HEAD_DIM, dtype=torch.bfloat16)
    score_state = torch.full_like(kv_state, float("-inf"))

    for local_position in range(kv.shape[2]):
        position = start_position + local_position
        slot = position % _COMPRESS_RATE
        parity = (position // _COMPRESS_RATE) & 1
        ca_row = (parity ^ 1) * 32 + slot
        cb_row = parity * 32 + _COMPRESS_RATE + slot
        kv_state[:, :, ca_row] = kv[:, :, local_position, :_HEAD_DIM]
        kv_state[:, :, cb_row] = kv[:, :, local_position, _HEAD_DIM:]
        score_state[:, :, ca_row] = score[:, :, local_position, :_HEAD_DIM]
        score_state[:, :, cb_row] = score[:, :, local_position, _HEAD_DIM:]

    return kv_state, score_state


def _make_states(sp_factor, remainder):
    torch.manual_seed(42)
    seq_len = _LOCAL_SEQ_LEN * sp_factor + remainder
    kv = torch.randn(_BATCH, 1, seq_len, 2 * _HEAD_DIM, dtype=torch.bfloat16)
    score = torch.randn_like(kv)

    local_kv_states = []
    local_score_states = []
    for rank in range(sp_factor):
        start = rank * _LOCAL_SEQ_LEN
        end = min(start + _LOCAL_SEQ_LEN + (remainder if rank == sp_factor - 1 else 0), seq_len)
        kv_state, score_state = _pack_blaze_state(kv[:, :, start:end], score[:, :, start:end], start)
        local_kv_states.append(kv_state)
        local_score_states.append(score_state)

    initial_kv = torch.randn(_BATCH, 1, _STATE_ROWS, _HEAD_DIM, dtype=torch.bfloat16)
    initial_score = torch.randn_like(initial_kv)
    initial_kv_states = [initial_kv] * sp_factor
    initial_score_states = [initial_score] * sp_factor

    local_kv = torch.cat(local_kv_states, dim=2)
    local_score = torch.cat(local_score_states, dim=2)
    initial_kv = torch.cat(initial_kv_states, dim=2)
    initial_score = torch.cat(initial_score_states, dim=2)
    expected_kv = torch.cat([initial_kv_states[0], *local_kv_states[:-1]], dim=2)
    expected_score = torch.cat([initial_score_states[0], *local_score_states[:-1]], dim=2)

    # The final rank's outgoing state must already be the exact state consumed by
    # Blaze decode, including a partially filled ratio-4 window.
    decode_kv, decode_score = _pack_blaze_state(kv, score)
    final_start = (sp_factor - 1) * _STATE_ROWS
    final_end = final_start + _STATE_ROWS
    assert torch.equal(local_kv[:, :, final_start:final_end], decode_kv)
    assert torch.equal(local_score[:, :, final_start:final_end], decode_score)

    return local_kv, local_score, initial_kv, initial_score, expected_kv, expected_score


def _run_compressor_state_exchange(mesh_device, remainder):
    mesh_shape = tuple(mesh_device.shape)
    sp_factor, tp_factor = mesh_shape
    local_kv, local_score, initial_kv, initial_score, expected_kv, expected_score = _make_states(sp_factor, remainder)
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=(2, None))

    def to_device(tensor):
        return ttnn.from_torch(
            tensor,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

    predecessor_kv, predecessor_score = ttnn.experimental.deepseek_prefill.compressor_state_exchange(
        to_device(local_kv),
        to_device(local_score),
        to_device(initial_kv),
        to_device(initial_score),
        cluster_axis=0,
        topology=ttnn.Topology.Linear,
    )

    composer = ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_shape, dims=(2, 1))
    output_kv = ttnn.to_torch(predecessor_kv, mesh_composer=composer)
    output_score = ttnn.to_torch(predecessor_score, mesh_composer=composer)

    for tp_rank in range(tp_factor):
        assert torch.equal(output_kv[:, tp_rank : tp_rank + 1], expected_kv)
        assert torch.equal(output_score[:, tp_rank : tp_rank + 1], expected_score)


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
def test_compressor_state_exchange_single_device(mesh_device, device_params, remainder):
    _run_compressor_state_exchange(mesh_device, remainder)


@pytest.mark.parametrize("remainder", range(_COMPRESS_RATE))
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (2, 2),
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 2), topology="mesh-2x2"),
            id="2x2",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_compressor_state_exchange_mesh(mesh_device, device_params, remainder):
    _run_compressor_state_exchange(mesh_device, remainder)
