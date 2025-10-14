# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from models.experimental.detr3d.reference.model_3detr import (
    Model3DETR as ref_model,
    build_preencoder as ref_build_preencoder,
    build_encoder as ref_build_encoder,
    build_decoder as ref_build_decoder,
)

# from models.experimental.detr3d.source.detr3d.models.model_3detr import (
#     Model3DETR as org_model,
#     build_preencoder as org_build_preencoder,
#     build_encoder as org_build_encoder,
#     build_decoder as org_build_decoder,
# )
from models.experimental.detr3d.reference.model_utils import SunrgbdDatasetConfig

from models.experimental.detr3d.reference.model_config import Detr3dArgs

# from tests.ttnn.utils_for_testing import assert_with_pcc


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
def test_3detr_model(encoder_only, input_shapes):
    args = Detr3dArgs()
    dataset_config = SunrgbdDatasetConfig()
    # org_pre_encoder = org_build_preencoder(args)
    # org_encoder = org_build_encoder(args)
    # org_decoder = org_build_decoder(args)
    # org_module = org_model(
    #     org_pre_encoder,
    #     org_encoder,
    #     org_decoder,
    #     dataset_config,
    #     encoder_dim=args.enc_dim,
    #     decoder_dim=args.dec_dim,
    #     mlp_dropout=args.mlp_dropout,
    #     num_queries=args.nqueries,
    # )
    # org_module.eval()
    input_dict = {key: torch.randn(shape) for key, shape in input_shapes.items()}
    # org_out = org_module(inputs=input_dict, encoder_only=encoder_only)

    ref_pre_encoder = ref_build_preencoder(args)
    ref_encoder = ref_build_encoder(args)
    ref_decoder = ref_build_decoder(args)
    ref_module = ref_model(
        ref_pre_encoder,
        ref_encoder,
        ref_decoder,
        dataset_config,
        encoder_dim=args.enc_dim,
        decoder_dim=args.dec_dim,
        mlp_dropout=args.mlp_dropout,
        num_queries=args.nqueries,
    )
    ref_module.eval()
    checkpoint = torch.load("models/experimental/detr3d/sunrgbd_masked_ep720.pth", map_location="cpu")["model"]
    # print("Top-level keys:", checkpoint.keys())
    ref_module.load_state_dict(checkpoint)  # org_module.state_dict()
    ref_out = ref_module(inputs=input_dict, encoder_only=encoder_only)

    ref_outputs = ref_out["outputs"]
    ref_aux_outputs = ref_out["aux_outputs"]
    for key in ref_outputs:
        print("key ", key, "shape:", ref_outputs[key].shape)

    # org_outputs, ref_outputs = org_out["outputs"], ref_out["outputs"]
    # org_aux_outputs, ref_aux_outputs = org_out["aux_outputs"], ref_out["aux_outputs"]
    # for key in org_outputs:
    #     assert_with_pcc(org_outputs[key], ref_outputs[key], 1.0)
    # for i in range(len(org_aux_outputs)):
    #     for keys in org_aux_outputs[i]:
    #         assert_with_pcc(org_aux_outputs[i][keys], ref_aux_outputs[i][keys], 1.0)
