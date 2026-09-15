# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Run official spec-decode correctness tests on T3K 1x8 (upstream params are 1x1/1x4)."""

from models.demos.gemma4.tests.test_factory import parametrize_mesh_with_fabric
from models.demos.gemma4.tests.unit import test_spec_decode as _sd


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 8)])
def test_spec_decode_matches_greedy_1x8(mesh_device, reset_seeds):
    return _sd.test_spec_decode_matches_greedy(mesh_device, reset_seeds)


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 8)])
def test_verify_batchsize_invariance_1x8(mesh_device, reset_seeds):
    return _sd.test_verify_batchsize_invariance(mesh_device, reset_seeds)


@parametrize_mesh_with_fabric(
    mesh_shapes=[(1, 8)],
    device_params_extra={"trace_region_size": 192_000_000},
)
def test_spec_decode_traced_1x8(mesh_device, reset_seeds):
    return _sd.test_spec_decode_traced(mesh_device, reset_seeds)
