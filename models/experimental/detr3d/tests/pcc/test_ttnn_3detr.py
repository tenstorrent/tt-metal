# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest

from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters
from models.common.utility_functions import comp_pcc, comp_allclose

from models.experimental.detr3d.common import load_torch_model_state
from models.experimental.detr3d.ttnn.model_3detr import build_ttnn_3detr
from models.experimental.detr3d.reference.model_3detr import build_3detr
from models.experimental.detr3d.reference.model_config import Detr3dArgs
from models.experimental.detr3d.reference.model_utils import SunrgbdDatasetConfig
from models.experimental.detr3d.ttnn.custom_preprocessing import create_custom_mesh_preprocessor


class Tt3DetrArgs(Detr3dArgs):
    def __init__(self):
        self.parameters = None
        self.device = None


@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 20000, 3),
    ],
)
@pytest.mark.parametrize("encoder_only", (False, True))
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_3detr_model(encoder_only, input_shape, device):
    torch.manual_seed(0)
    # Configuration flags
    PCC_THRESHOLD = 0.95
    CHECK_AUX_OUTPUTS = False  # Set to True to enable PCC check for auxiliary outputs
    SKIP_KEYS = ["angle_continuous", "objectness_prob"]  # Keys to skip in PCC comparison
    LOAD_REAL_INPUT = False

    args = Detr3dArgs()
    dataset_config = SunrgbdDatasetConfig()

    # Define the shape and range
    if LOAD_REAL_INPUT:
        input_dict = torch.load("models/experimental/detr3d/resources/inputs.pt", map_location="cpu")
    else:
        min_val = -1.8827
        max_val = 8.3542
        pc = (max_val - min_val) * torch.rand(input_shape) + min_val
        input_dict = {
            "point_clouds": pc,
            "point_cloud_dims_min": torch.min(pc, 1)[0],
            "point_cloud_dims_max": torch.max(pc, 1)[0],
        }

    ref_module, _ = build_3detr(args, dataset_config)
    load_torch_model_state(ref_module)
    ref_out = ref_module(inputs=input_dict, encoder_only=encoder_only)

    ref_module_parameters = preprocess_model_parameters(
        initialize_model=lambda: ref_module,
        custom_preprocessor=create_custom_mesh_preprocessor(None),
        device=device,
    )

    ttnn_args = Tt3DetrArgs()
    ttnn_args.parameters = ref_module_parameters
    ttnn_args.device = device

    ttnn_module, _ = build_ttnn_3detr(ttnn_args, dataset_config)
    tt_output = ttnn_module(inputs=input_dict, encoder_only=encoder_only)

    all_passing = True
    if encoder_only:
        ttnn_torch_out = []
        for idx, (tt_out, torch_out) in enumerate(zip(tt_output, ref_out)):
            if not isinstance(tt_out, torch.Tensor):
                tt_out = ttnn.to_torch(tt_out)
                tt_out = torch.reshape(tt_out, torch_out.shape)
            ttnn_torch_out.append(tt_out)

            passing, pcc_message = comp_pcc(torch_out, tt_out, PCC_THRESHOLD)
            logger.info(f"Encoder Output {idx} PCC: {pcc_message}")
            logger.info(comp_allclose(torch_out, tt_out))

            if passing:
                logger.info(f"Encoder Output {idx} Test Passed!")
            else:
                logger.warning(f"Encoder Output {idx} Test Failed!")
                all_passing = False
    else:
        ttnn_outputs, ref_outputs = tt_output["outputs"], ref_out["outputs"]
        ttnn_aux_outputs, ref_aux_outputs = tt_output["aux_outputs"], ref_out["aux_outputs"]

        # Check main outputs
        for key in ref_outputs:
            if key in SKIP_KEYS:
                logger.info(f"Output Key '{key}' - Skipped (in SKIP_KEYS)")
                continue

            passing, pcc_message = comp_pcc(ref_outputs[key], ttnn_outputs[key], PCC_THRESHOLD)
            logger.info(f"Output Key '{key}' PCC: {pcc_message}")
            logger.info(comp_allclose(ref_outputs[key], ttnn_outputs[key]))

            if passing:
                logger.info(f"Output Key '{key}' Test Passed!")
            else:
                logger.warning(f"Output Key '{key}' Test Failed!")
                all_passing = False

        # Check auxiliary outputs only if flag is enabled
        if CHECK_AUX_OUTPUTS:
            for i in range(len(ref_aux_outputs)):
                for key in ref_aux_outputs[i]:
                    if key in SKIP_KEYS:
                        logger.info(f"Aux Output {i} Key '{key}' - Skipped (in SKIP_KEYS)")
                        continue

                    passing, pcc_message = comp_pcc(ref_aux_outputs[i][key], ttnn_aux_outputs[i][key], PCC_THRESHOLD)
                    logger.info(f"Aux Output {i} Key '{key}' PCC: {pcc_message}")
                    logger.info(comp_allclose(ref_aux_outputs[i][key], ttnn_aux_outputs[i][key]))

                    if passing:
                        logger.info(f"Aux Output {i} Key '{key}' Test Passed!")
                    else:
                        logger.warning(f"Aux Output {i} Key '{key}' Test Failed!")
                        all_passing = False
        else:
            logger.info("Auxiliary outputs PCC check is disabled (CHECK_AUX_OUTPUTS=False)")

    assert all_passing, f"One or more outputs failed PCC check with threshold {PCC_THRESHOLD}"
