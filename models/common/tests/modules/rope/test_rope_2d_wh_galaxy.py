# SPDX-FileCopyrightText: Copyright 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Real-hardware qualification for RotarySetup2D on a Wormhole Galaxy."""

import pytest
import torch

import ttnn
from models.common.auto_compose import to_torch_auto_compose
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.rope.rope_2d import RotarySetup2D, RotarySetup2DConfig
from models.common.utility_functions import comp_pcc


def _rope_tables(max_seq_len, head_dim, theta):
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    frequencies = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.outer(positions, frequencies)
    angles = torch.cat((angles, angles), dim=-1)
    return angles.cos().bfloat16()[None, None], angles.sin().bfloat16()[None, None]


def _deallocate_all(tensors):
    for tensor in tensors:
        if tensor is not None:
            tensor.deallocate(True)


@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4")], indirect=True)
@pytest.mark.parametrize(
    "theta",
    [pytest.param(500000.0, id="llama"), pytest.param(1000000.0, id="qwen")],
)
def test_rotary_setup_2d_wh_galaxy_reference(mesh_device, theta):
    head_dim = 128
    cos, sin = _rope_tables(2048, head_dim, theta)
    core_grid = mesh_device.compute_with_storage_grid_size()
    batch_grid = ttnn.num_cores_to_corerangeset(8, core_grid, row_wise=True)
    module = RotarySetup2D.from_config(
        RotarySetup2DConfig(
            LazyWeight(source=cos, device=mesh_device),
            LazyWeight(source=sin, device=mesh_device),
            max_batch_size=32,
            rope_theta=theta,
            core_grid=core_grid,
            batch_grid=batch_grid,
        )
    )
    rot_idxs = None

    try:
        positions = torch.arange(32, dtype=torch.int32)
        rot_idxs = module.get_rot_idxs(positions)
        decode_reference = (cos[:, :, positions.long(), :], sin[:, :, positions.long(), :])
        for _ in range(2):
            outputs = module.decode_forward(rot_idxs)
            try:
                composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=(8, 4))
                actual = []
                for output in outputs:
                    composed = ttnn.to_torch(output, mesh_composer=composer)
                    value = composed[:, :, :1].permute(0, 2, 1, 3)
                    actual.append(value)
                for reference, value in zip(decode_reference, actual):
                    assert value.shape == reference.shape, (value.shape, reference.shape)
                    passing, message = comp_pcc(reference, value, 0.99)
                    assert passing, message
            finally:
                _deallocate_all(outputs)
            assert module.cos_matrix.is_allocated()
            assert module.sin_matrix.is_allocated()

        for seq_len in (128, 2048):
            references = (cos[:, :, :seq_len], sin[:, :, :seq_len])
            for _ in range(2):
                outputs = module.prefill_forward(start_pos=0, seq_len=seq_len)
                try:
                    actual = [to_torch_auto_compose(output)[:, :, :seq_len, :head_dim] for output in outputs]
                    for reference, value in zip(references, actual):
                        passing, message = comp_pcc(reference, value, 0.99)
                        assert passing, message
                finally:
                    _deallocate_all(outputs)
    finally:
        _deallocate_all(
            [
                rot_idxs,
                getattr(module, "cos_matrix", None),
                getattr(module, "sin_matrix", None),
                getattr(module, "cos_matrix_prefill", None),
                getattr(module, "sin_matrix_prefill", None),
                getattr(module, "transformation_mat", None),
                getattr(module, "transformation_mat_prefill", None),
            ]
        )
        for weight in (
            module.config.cos_matrix,
            module.config.sin_matrix,
            module.config._decode_trans_mat,
            module.config._prefill_trans_mat,
        ):
            weight._value = None
