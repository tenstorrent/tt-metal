# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device parameters for ``TtBEVFormerHead`` from the PyTorch reference."""

from types import SimpleNamespace

import torch

import ttnn
from models.experimental.bevformer.tt.model_preprocessing import (
    DEFAULT_DTYPE,
    linear_params,
    preprocess_layer_norm_parameters,
)
from models.experimental.bevformer.tt.model_preprocessing_decoder import (
    create_decoder_parameters,
    create_reg_branch_parameters,
)
from models.experimental.bevformer.tt.tt_decoder import GRID_DTYPE


def _cls_branch_parameters(branch, device, dtype):
    """``hidden``: the ``(linear, layer_norm)`` of each hidden block; ``out``: the output Linear."""
    linears = [m for m in branch if isinstance(m, torch.nn.Linear)]
    norms = [m for m in branch if isinstance(m, torch.nn.LayerNorm)]
    hidden = [
        (
            linear_params(linear.weight, linear.bias, device, dtype),
            SimpleNamespace(**preprocess_layer_norm_parameters(norm, device=device, dtype=dtype)),
        )
        for linear, norm in zip(linears[:-1], norms, strict=True)
    ]
    return SimpleNamespace(hidden=hidden, out=linear_params(linears[-1].weight, linears[-1].bias, device, dtype))


@torch.no_grad()
def create_head_parameters(torch_model, device, dtype=DEFAULT_DTYPE):
    """Parameters for a single ``TtBEVFormerHead``; single use, as ``create_decoder_parameters`` is.

    The object queries, their positional embeddings and the initial reference points depend
    on the weights only, so they are computed here, the points in float32 as the decoder
    takes them. The reg branches run inside the decoder, which returns their box codes.
    """
    query_pos, query = torch.split(torch_model.query_embedding.weight, torch_model.embed_dims, dim=1)
    init_reference = torch_model.reference_points(query_pos).sigmoid()

    def upload(tensor, tensor_dtype=dtype):
        return ttnn.from_torch(tensor.unsqueeze(0), dtype=tensor_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    return SimpleNamespace(
        query=upload(query),
        query_pos=upload(query_pos),
        reference_points=upload(init_reference, GRID_DTYPE),
        decoder=create_decoder_parameters(torch_model.decoder, device, dtype),
        reg_branches=create_reg_branch_parameters(torch_model.reg_branches, device, dtype),
        cls_branches=[_cls_branch_parameters(branch, device, dtype) for branch in torch_model.cls_branches],
    )
