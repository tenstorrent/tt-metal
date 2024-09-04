# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os
import torch
import pytest
from loguru import logger

# Set flags for CI, if CI environment is setup
if os.getenv("CI") == "true":
    os.environ["TT_METAL_ASYNC_DEVICE_QUEUE"] = "1"
    os.environ["WH_ARCH_YAML"] = "wormhole_b0_80_arch_eth_dispatch.yaml"

import ttnn
from ttnn import ReplicateTensorToMesh, ConcatMeshToTensor

from models.common.rmsnorm import RMSNorm as TtRMSNorm
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
    skip_for_grayskull,
)


# RMSNorm reference class
# Copyright 2023 Mistral AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
class TorchRMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.randn(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class RefModel(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.rmsnorm = TorchRMSNorm(dim=dim)

    def forward(self, x):
        return self.rmsnorm(x)


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "is_sharded",
    (True, False),
)
def test_rmsnorm_singledevice(device, is_sharded, use_program_cache, reset_seeds):
    dim = 4096
    dtype = ttnn.bfloat8_b

    reference_model = RefModel(dim=dim)
    state_dict = reference_model.state_dict()

    tt_model = TtRMSNorm(
        device=device,
        dim=dim,
        state_dict=state_dict,
        weight_key="rmsnorm",
        is_sharded=is_sharded,
    )
    input = torch.rand(1, 1, 32, dim)
    reference_output = reference_model(input)

    tt_input = ttnn.from_torch(
        input,
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_output = tt_model(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output)
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(pcc_message)

    if passing:
        logger.info("Common rmsnorm_singledevice Passed!")
    else:
        logger.warning("Common rmsnorm_singledevice Failed!")

    assert passing, f"Common rmsnorm_singledevice output does not meet PCC requirement {0.99}."


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "is_sharded",
    (True, False),
)
def test_rmsnorm_multidevice(t3k_mesh_device, is_sharded, use_program_cache, reset_seeds):
    dim = 4096
    dtype = ttnn.bfloat8_b

    reference_model = RefModel(dim=dim)
    state_dict = reference_model.state_dict()

    tt_model = TtRMSNorm(
        device=t3k_mesh_device,
        dim=dim,
        state_dict=state_dict,
        weight_key="rmsnorm",
        is_sharded=is_sharded,
    )
    input = torch.rand(1, 1, 32, dim)
    reference_output = reference_model(input)

    tt_input = ttnn.from_torch(
        input,
        device=t3k_mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ReplicateTensorToMesh(t3k_mesh_device),
    )

    tt_output = tt_model(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=ConcatMeshToTensor(t3k_mesh_device, dim=0))[0]
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(pcc_message)

    if passing:
        logger.info("Common rmsnorm_multidevice Passed!")
    else:
        logger.warning("Common rmsnorm_multidevice Failed!")

    assert passing, f"Common rmsnorm_multidevice output does not meet PCC requirement {0.99}."
