# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest

from tests.ttnn.utils_for_testing import comp_pcc
from ttnn.model_preprocessing import preprocess_model_parameters
from models.experimental.detr3d.ttnn.custom_preprocessing import create_custom_mesh_preprocessor

from models.experimental.detr3d.ttnn.tt_3detr import build_ttnn_3detr
from models.experimental.detr3d.reference.detr3d_model import build_3detr
from models.experimental.detr3d.reference.model_utils import SunrgbdDatasetConfig
from models.experimental.detr3d.reference.model_config import Detr3dArgs
from models.experimental.detr3d.common import load_torch_model_state


class Tt3DetrArgs:
    def __init__(self):
        self.modules = None
        self.parameters = None
        self.device = None
        self.model_name = "3detr"
        self.enc_type = "masked"
        self.enc_nlayers = 3
        self.enc_dim = 256
        self.enc_ffn_dim = 128
        self.enc_nhead = 4
        self.enc_activation = "relu"
        self.dec_nlayers = 8
        self.dec_dim = 256
        self.dec_ffn_dim = 256
        self.dec_nhead = 4
        self.preenc_npoints = 2048
        self.nqueries = 128
        self.use_color = False


@pytest.mark.parametrize(
    "input_shapes",
    [
        {
            "point_clouds": (1, 20000, 3),
            "point_cloud_dims_min": (1, 3),
            "point_cloud_dims_max": (1, 3),
        },
    ],
)
@pytest.mark.parametrize("encoder_only", (False, True))
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_3detr_model(encoder_only, input_shapes, device):
    args = Detr3dArgs()
    dataset_config = SunrgbdDatasetConfig()

    input_dict = {key: torch.randn(shape) for key, shape in input_shapes.items()}

    ref_module, _ = build_3detr(args, dataset_config)
    load_torch_model_state(ref_module)
    ref_out = ref_module(inputs=input_dict, encoder_only=encoder_only)

    ref_module_parameters = preprocess_model_parameters(
        initialize_model=lambda: ref_module,
        custom_preprocessor=create_custom_mesh_preprocessor(None),
        device=device,
    )

    ttnn_args = Tt3DetrArgs()
    ttnn_args.modules = ref_module
    ttnn_args.parameters = ref_module_parameters
    ttnn_args.device = device

    ttnn_module, _ = build_ttnn_3detr(ttnn_args, dataset_config)
    tt_output = ttnn_module(inputs=input_dict, encoder_only=encoder_only)

    if encoder_only:
        ttnn_torch_out = []
        for tt_out, torch_out in zip(tt_output, ref_out):
            if not isinstance(tt_out, torch.Tensor):
                tt_out = ttnn.to_torch(tt_out)
                tt_out = torch.reshape(tt_out, torch_out.shape)
            ttnn_torch_out.append(tt_out)
            _, pcc_message = comp_pcc(torch_out, tt_out, 0.97)
            print(f"PCC: {pcc_message}")
    else:
        ttnn_outputs, ref_outputs = tt_output["outputs"], ref_out["outputs"]
        ttnn_aux_outputs, ref_aux_outputs = tt_output["aux_outputs"], ref_out["aux_outputs"]
        for key in ref_outputs:
            _, pcc_message = comp_pcc(ref_outputs[key], ttnn_outputs[key], 0.97)
            print(f"Key:{key}, PCC: {pcc_message}")
        for i in range(len(ref_aux_outputs)):
            for key in ref_aux_outputs[i]:
                _, pcc_message = comp_pcc(ref_aux_outputs[i][key], ttnn_aux_outputs[i][key], 0.97)
                print(f"Aux:{i} Key:{key}, PCC: {pcc_message}")
