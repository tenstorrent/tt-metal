# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
import ttnn
from ttnn.model_preprocessing import infer_ttnn_module_args
from models.experimental.functional_petr.reference.petr_transformer import PETRTransformer


def create_petr_transformer_input_tensors(
    device,
):
    x = torch.rand(1, 6, 256, 20, 50)
    mask = torch.zeros((1, 6, 20, 50), dtype=torch.bool)
    mask_int = torch.zeros((1, 6, 20, 50))
    query_embed = torch.rand(900, 256)
    pos_embed = torch.rand(1, 6, 256, 20, 50)

    ttnn_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, device=device)
    ttnn_mask = ttnn.from_torch(mask_int, device=device)
    ttnn_query_embed = ttnn.from_torch(query_embed, dtype=ttnn.bfloat16, device=device)
    ttnn_pos_embed = ttnn.from_torch(pos_embed, dtype=ttnn.bfloat16, device=device)

    return x, mask, query_embed, pos_embed, ttnn_x, ttnn_mask, ttnn_query_embed, ttnn_pos_embed


def create_petr_transformer_model_parameters(model: PETRTransformer, x, mask, query_embed, pos_embed, device):
    parameters = infer_ttnn_module_args(
        model=model, run_model=lambda model: model(x, mask, query_embed, pos_embed), device=None
    )
    assert parameters is not None
    for key in parameters.keys():
        parameters[key].module = getattr(model, key)

    return parameters
