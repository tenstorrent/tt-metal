# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest
from models.experimental.detr3d.reference.detr3d_model import build_3detr
from ttnn.model_preprocessing import preprocess_model_parameters
from models.experimental.detr3d.ttnn.custom_preprocessing import create_custom_mesh_preprocessor
from models.experimental.detr3d.ttnn.tt_3detr import build_ttnn_3detr
from models.experimental.detr3d.reference.model_utils import SunrgbdDatasetConfig
from models.experimental.detr3d.reference.model_config import Detr3dArgs
from tests.ttnn.utils_for_testing import comp_pcc


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
    mesh_device = device
    args = Detr3dArgs()
    dataset_config = SunrgbdDatasetConfig()

    input_dict = {key: torch.randn(shape) for key, shape in input_shapes.items()}

    ref_module, _ = build_3detr(args, dataset_config)
    ref_module.eval()
    checkpoint = torch.load("models/experimental/detr3d/sunrgbd_masked_ep720.pth", map_location="cpu")["model"]
    ref_module.load_state_dict(checkpoint, strict=True)
    ref_out = ref_module(inputs=input_dict, encoder_only=encoder_only)

    ref_module_parameters = preprocess_model_parameters(
        initialize_model=lambda: ref_module,
        custom_preprocessor=create_custom_mesh_preprocessor(None),
        device=device,
    )

    class Tt3DetrArgs:
        modules = ref_module
        parameters = ref_module_parameters
        device = mesh_device
        model_name = "3detr"
        enc_type = "masked"
        enc_nlayers = 3
        enc_dim = 256
        enc_ffn_dim = 128
        enc_nhead = 4
        enc_activation = "relu"
        dec_nlayers = 8
        dec_dim = 256
        dec_ffn_dim = 256
        dec_nhead = 4
        preenc_npoints = 2048
        nqueries = 128
        use_color = False

    ttnn_args = Tt3DetrArgs()

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
