# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
import ttnn
import torch
import pytest
from loguru import logger

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.common.utility_functions import run_for_wormhole_b0
from ttnn.model_preprocessing import preprocess_model_parameters
from models.experimental.pointpillars.common import load_torch_model, POINTPILLARS_L1_SMALL_SIZE

from models.experimental.pointpillars.tt.ttnn_second import TtSECOND

from models.experimental.yolo_common.yolo_utils import determine_num_cores, get_core_grid_from_num_cores
from models.experimental.pointpillars.tt.model_preprocessing import create_custom_preprocessor


def interleaved_to_sharded(x):
    if x.get_layout() == ttnn.TILE_LAYOUT:
        x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT)
    if x.shape[1] == 1:
        x = ttnn.reshape(x, (x.shape[0], int(math.sqrt(x.shape[2])), int(math.sqrt(x.shape[2])), x.shape[3]))
    nhw = x.shape[0] * x.shape[1] * x.shape[2]
    num_cores = determine_num_cores(nhw, x.shape[2])
    core_grid = get_core_grid_from_num_cores(num_cores)
    shardspec = ttnn.create_sharded_memory_config_(
        x.shape, core_grid, ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )

    return ttnn.reshard(x, shardspec) if x.is_sharded() else ttnn.interleaved_to_sharded(x, shardspec)


@pytest.mark.parametrize(
    "use_pretrained_weight",
    [
        True,
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": POINTPILLARS_L1_SMALL_SIZE}], indirect=True)
@run_for_wormhole_b0()
def test_ttnn_second(device, use_pretrained_weight, reset_seeds, model_location_generator):
    reference_model = load_torch_model(model_location_generator)
    reference_model = reference_model.pts_backbone

    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model, custom_preprocessor=create_custom_preprocessor(device)
    )

    x = torch.randn(1, 64, 400, 400)

    x_permute = x.permute(0, 2, 3, 1)
    ttnn_x = ttnn.from_torch(x_permute, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    ttnn_x = interleaved_to_sharded(ttnn_x)

    reference_output = reference_model(x=x)

    ttnn_model = TtSECOND(
        in_channels=64,
        norm_cfg={"type": "BN2d", "eps": 0.001, "momentum": 0.01},
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256],
        parameters=parameters,
        device=device,
    )

    ttnn_output = ttnn_model(x=ttnn_x)

    for i in range(len(ttnn_output)):
        ttnn_output_temp = ttnn.to_torch(ttnn_output[i]).permute(0, 3, 1, 2)
        ttnn_output_temp = ttnn_output_temp.reshape(
            ttnn_output_temp.shape[0],
            ttnn_output_temp.shape[1],
            int(math.sqrt(ttnn_output_temp.shape[3])),
            int(math.sqrt(ttnn_output_temp.shape[3])),
        )
        passing, pcc = assert_with_pcc(reference_output[i], ttnn_output_temp, 0.98)
        logger.info(f"Passing: {passing}, PCC: {pcc}")
