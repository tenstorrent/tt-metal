# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device tests for ttnn.experimental.csa_compression."""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

_BATCH = 1
_COMPRESS_RATE = 4
_HEAD_DIM = 512
_LOCAL_SEQ_LEN = 128
_PCC = 0.999


def _torch_csa_compression(kv, gate, position_bias, predecessor_kv, predecessor_gate, sp_factor):
    """Reference the local fused op, including each SP rank's supplied predecessor Ca window."""
    local_seq_len = kv.shape[2] // sp_factor
    outputs = []

    for rank in range(sp_factor):
        start = rank * local_seq_len
        end = start + local_seq_len
        local_kv = kv[:, :, start:end]
        local_gate = gate[:, :, start:end]
        n_windows = local_seq_len // _COMPRESS_RATE

        local_kv = local_kv.view(_BATCH, 1, n_windows, _COMPRESS_RATE, 2 * _HEAD_DIM)
        local_gate = local_gate.view(_BATCH, 1, n_windows, _COMPRESS_RATE, 2 * _HEAD_DIM)
        biased_gate = local_gate + position_bias.unsqueeze(2)

        ca_kv = local_kv[..., :_HEAD_DIM]
        cb_kv = local_kv[..., _HEAD_DIM:]
        ca_gate = biased_gate[..., :_HEAD_DIM]
        cb_gate = biased_gate[..., _HEAD_DIM:]

        boundary_start = rank * _COMPRESS_RATE
        boundary_end = boundary_start + _COMPRESS_RATE
        prior_kv = predecessor_kv[:, :, boundary_start:boundary_end].unsqueeze(2)
        prior_gate = predecessor_gate[:, :, boundary_start:boundary_end].unsqueeze(2)
        prior_gate = prior_gate + position_bias[..., :_HEAD_DIM].unsqueeze(2)

        previous_ca_kv = torch.cat([prior_kv, ca_kv[:, :, :-1]], dim=2)
        previous_ca_gate = torch.cat([prior_gate, ca_gate[:, :, :-1]], dim=2)
        overlap_kv = torch.cat([previous_ca_kv, cb_kv], dim=3)
        overlap_gate = torch.cat([previous_ca_gate, cb_gate], dim=3)

        weights = overlap_gate.softmax(dim=3, dtype=torch.float32).to(overlap_kv.dtype)
        outputs.append((overlap_kv * weights).sum(dim=3))

    return torch.cat(outputs, dim=2)


def _make_inputs(sp_factor):
    torch.manual_seed(42)
    seq_len = _LOCAL_SEQ_LEN * sp_factor
    kv = torch.randn(_BATCH, 1, seq_len, 2 * _HEAD_DIM, dtype=torch.bfloat16)
    gate = torch.randn_like(kv)
    position_bias = torch.randn(1, 1, _COMPRESS_RATE, 2 * _HEAD_DIM, dtype=torch.bfloat16)

    # Rank 0 models a carry from the preceding call. Every other rank receives the final Ca window
    # from the preceding SP rank, matching the boundary exchange that will surround this local op.
    predecessor_kv = torch.empty(_BATCH, 1, sp_factor * _COMPRESS_RATE, _HEAD_DIM, dtype=torch.bfloat16)
    predecessor_gate = torch.empty_like(predecessor_kv)
    predecessor_kv[:, :, :_COMPRESS_RATE] = torch.randn(_BATCH, 1, _COMPRESS_RATE, _HEAD_DIM, dtype=torch.bfloat16)
    predecessor_gate[:, :, :_COMPRESS_RATE] = torch.randn_like(predecessor_gate[:, :, :_COMPRESS_RATE])
    for rank in range(1, sp_factor):
        source_end = rank * _LOCAL_SEQ_LEN
        source_start = source_end - _COMPRESS_RATE
        boundary_start = rank * _COMPRESS_RATE
        boundary_end = boundary_start + _COMPRESS_RATE
        predecessor_kv[:, :, boundary_start:boundary_end] = kv[:, :, source_start:source_end, :_HEAD_DIM]
        predecessor_gate[:, :, boundary_start:boundary_end] = gate[:, :, source_start:source_end, :_HEAD_DIM]

    return kv, gate, position_bias, predecessor_kv, predecessor_gate


def _run_csa_compression(mesh_device):
    mesh_shape = tuple(mesh_device.shape)
    sp_factor = mesh_shape[0]
    kv, gate, position_bias, predecessor_kv, predecessor_gate = _make_inputs(sp_factor)
    expected = _torch_csa_compression(
        kv,
        gate,
        position_bias,
        predecessor_kv,
        predecessor_gate,
        sp_factor,
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

    tt_kv = to_device(kv, sp_mapper)
    tt_gate = to_device(gate, sp_mapper)
    tt_position_bias = to_device(position_bias, replicated_mapper)
    tt_predecessor_kv = to_device(predecessor_kv, sp_mapper)
    tt_predecessor_gate = to_device(predecessor_gate, sp_mapper)

    tt_output = ttnn.experimental.csa_compression(
        tt_kv,
        tt_gate,
        tt_position_bias,
        tt_predecessor_kv,
        tt_predecessor_gate,
    )

    output = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(
            mesh_device,
            mesh_shape=mesh_shape,
            dims=(2, 1),
        ),
    )[:, :1]
    assert output.shape == expected.shape
    passed, message = assert_with_pcc(expected.float(), output.float(), pcc=_PCC)
    assert passed, f"CSA compression PCC failed: {message}"


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
def test_csa_compression_single_device(mesh_device, device_params):
    _run_csa_compression(mesh_device)


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (2, 2),
            {"fabric_config": ttnn.FabricConfig.DISABLED},
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 2), topology="mesh-2x2"),
            id="2x2",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_csa_compression_mesh(mesh_device, device_params):
    _run_csa_compression(mesh_device)
