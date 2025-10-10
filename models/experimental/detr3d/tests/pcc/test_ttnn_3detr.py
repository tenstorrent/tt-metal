# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest
from models.experimental.detr3d.reference.detr3d_model import (
    build_3detr as ref_build_3detr,
)
from ttnn.model_preprocessing import preprocess_model_parameters
from models.experimental.detr3d.ttnn.custom_preprocessing import create_custom_mesh_preprocessor
from models.experimental.detr3d.ttnn.ttnn_3detr_model import build_ttnn_3detr

# from models.experimental.detr3d.source.detr3d.models.model_3detr import (
#     Model3DETR as org_model,
#     build_preencoder as org_build_preencoder,
#     build_encoder as org_build_encoder,
#     build_decoder as org_build_decoder,
# )
from models.experimental.detr3d.reference.model_utils import SunrgbdDatasetConfig

from models.experimental.detr3d.reference.model_config import Detr3dArgs

from tests.ttnn.utils_for_testing import comp_pcc


@pytest.mark.parametrize(
    " encoder_only, input_shapes",
    [
        (
            False,
            {
                "point_clouds": (1, 20000, 3),
                "point_cloud_dims_min": (1, 3),
                "point_cloud_dims_max": (1, 3),
            },
        ),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_3detr_model(encoder_only, input_shapes, device):
    args = Detr3dArgs()
    dataset_config = SunrgbdDatasetConfig()

    input_dict = {key: torch.randn(shape) for key, shape in input_shapes.items()}

    ref_module, _ = ref_build_3detr(args, dataset_config)
    ref_module.eval()
    # checkpoint = torch.load("models/experimental/detr3d/sunrgbd_masked_ep720.pth", map_location="cpu")["model"]
    # ref_module.load_state_dict(checkpoint)
    ref_out = ref_module(inputs=input_dict, encoder_only=encoder_only)

    ref_module_parameters = preprocess_model_parameters(
        initialize_model=lambda: ref_module,
        custom_preprocessor=create_custom_mesh_preprocessor(None),
        device=device,
    )

    mesh_device = device
    print(f"PP")

    class Tt3DetrArgs:
        modules = ref_module
        parameters = ref_module_parameters
        device = mesh_device
        model_name = "3detr"
        enc_type = "masked"
        enc_nlayers = 3
        enc_dim = 256
        enc_ffn_dim = 128
        enc_dropout = 0.1
        enc_nhead = 4
        enc_pos_embed = None
        enc_activation = "relu"
        dec_nlayers = 8
        dec_dim = 256
        dec_ffn_dim = 256
        dec_dropout = 0.1
        dec_nhead = 4
        mlp_dropout = 0.3
        nsemcls = -1
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
            pcc_pass, pcc_message = comp_pcc(torch_out, tt_out, 0.97)
            print(f"{pcc_message=}")
    else:
        ttnn_outputs, ref_outputs = tt_output["outputs"], ref_out["outputs"]
        ttnn_aux_outputs, ref_aux_outputs = tt_output["aux_outputs"], ref_out["aux_outputs"]
        for key in ref_outputs:
            pcc_pass, pcc_message = comp_pcc(ref_outputs[key], ttnn_outputs[key], 0.97)
            if not pcc_pass:
                print(f"key: {key} : {pcc_message}")
        for i in range(len(ref_aux_outputs)):
            for keys in ref_aux_outputs[i]:
                pcc_pass, pcc_message = comp_pcc(ref_aux_outputs[i][keys], ttnn_aux_outputs[i][keys], 0.97)
                if not pcc_pass:
                    print(f"key: {i}.{keys} : {pcc_message}")

    # ref_outputs = ref_out["outputs"]
    # ref_aux_outputs = ref_out["aux_outputs"]
    # for key in ref_outputs:
    #     print("key ", key, "shape:", ref_outputs[key].shape)
