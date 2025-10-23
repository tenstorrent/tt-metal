# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest

from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.detr3d.ttnn.custom_preprocessing import create_custom_mesh_preprocessor

from models.experimental.detr3d.reference.model_3detr import (
    TransformerEncoderLayer,
    build_encoder,
)
from models.experimental.detr3d.ttnn.masked_transformer_encoder import TtnnTransformerEncoderLayer
from models.experimental.detr3d.common import load_torch_model_state, DotAccessibleDict
from models.experimental.detr3d.reference.model_config import Detr3dArgs
from models.experimental.detr3d.ttnn.model_3detr import build_ttnn_encoder


class Tt3DetrArgs(Detr3dArgs):
    def __init__(self):
        self.modules = None
        self.parameters = None
        self.device = None


def compute_mask(device, xyz, radius, dist=None):
    with torch.no_grad():
        if dist is None or dist.shape[1] != xyz.shape[1]:
            dist = torch.cdist(xyz, xyz, p=2)
        # entries that are True in the mask do not contribute to self-attention
        # so points outside the radius are not considered
        mask = dist >= radius
    mask_ttnn = torch.zeros_like(mask, dtype=torch.float).masked_fill_(mask, float("-inf"))
    mask_ttnn = ttnn.from_torch(mask_ttnn, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    return mask, mask_ttnn


@torch.no_grad()
@pytest.mark.parametrize(
    "batch_size, seq_len, d_model, nhead, normalize_before, masking_radius, weight_key_prefix",
    [
        (1, 2048, 256, 4, True, 0.16000000000000003, "encoder.layers.0"),
        (1, 1024, 256, 4, True, 0.6400000000000001, "encoder.layers.1"),
        (1, 1024, 256, 4, True, 1.44, "encoder.layers.2"),
    ],
)
def test_transformer_encoder_layer_inference(
    batch_size,
    seq_len,
    d_model,
    nhead,
    normalize_before,
    masking_radius,
    weight_key_prefix,
    device,
):
    """Test TtnnTransformerEncoderLayer against PyTorch reference implementation"""

    torch.manual_seed(0)
    mesh_device = device
    dtype = ttnn.bfloat16

    # Initialize reference model
    reference_model = TransformerEncoderLayer(
        d_model,
        nhead,
        normalize_before=normalize_before,
    )
    load_torch_model_state(reference_model, weight_key_prefix)

    # Create test inputs
    src_input = torch.randn(seq_len, batch_size, d_model, dtype=torch.float32)
    xyz = torch.randn(batch_size, seq_len, 3, dtype=torch.float32)
    attn_mask, attn_mask_ttnn = compute_mask(mesh_device, xyz, masking_radius, None)
    # mask must be tiled to num_heads of the transformer
    bsz, n, n = attn_mask.shape
    attn_mask = attn_mask.unsqueeze(1)
    attn_mask = attn_mask.repeat(1, nhead, 1, 1)
    attn_mask = attn_mask.view(bsz * nhead, n, n)
    attn_mask_ttnn = ttnn.unsqueeze(attn_mask_ttnn, 1)

    # Get reference output with explicit None masks
    with torch.no_grad():
        ref_output = reference_model(
            src_input,
            src_mask=attn_mask,
            pos=None,
        )

    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model,
        custom_preprocessor=create_custom_mesh_preprocessor(None),
        device=mesh_device,
    )

    # Initialize TTNN model with preprocessed parameters
    tt_model = TtnnTransformerEncoderLayer(
        mesh_device,
        d_model,
        nhead,
        normalize_before=normalize_before,
        parameters=parameters,  # Pass preprocessed weights
    )

    # Convert inputs to TTNN tensors
    tt_src = ttnn.from_torch(
        src_input.permute(1, 0, 2),
        dtype=dtype,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    # Run TTNN model
    tt_output = tt_model(tt_src, src_mask=attn_mask_ttnn, pos=None, return_attn_weights=False)

    if isinstance(ref_output, tuple):
        ref_output = ref_output[0]  # Get the tensor, ignore attention weights

    # Convert back to torch for comparison
    tt_output_torch = ttnn.to_torch(tt_output)
    tt_output_torch = torch.permute(tt_output_torch, (1, 0, 2))

    # Compare outputs
    passing, pcc_message = comp_pcc(ref_output, tt_output_torch, pcc=0.99)

    logger.info(f"Output PCC: {pcc_message}")
    logger.info(comp_allclose(ref_output, tt_output_torch))
    logger.info(f"Batch: {batch_size}, Seq: {seq_len}, D_model: {d_model}, Heads: {nhead}")
    logger.info(f"Normalize before: {normalize_before}")

    if passing:
        logger.info("TransformerEncoderLayer Test Passed!")
    else:
        logger.warning("TransformerEncoderLayer Test Failed!")

    assert passing, f"PCC value is lower than 0.99. Check implementation! {pcc_message}"

    ttnn.close_device(mesh_device)


@pytest.mark.parametrize(
    "src_shape, xyz_shape",
    [
        ((2048, 1, 256), (1, 2048, 3)),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_masked_transformer_encoder_inference(
    src_shape,
    xyz_shape,
    device,
):
    torch_args = Detr3dArgs()
    reference_model = build_encoder(torch_args)
    load_torch_model_state(reference_model, "encoder")

    src = torch.randn(src_shape)
    xyz = torch.randn(xyz_shape)
    ref_out = reference_model(src=src, xyz=xyz)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model,
        custom_preprocessor=create_custom_mesh_preprocessor(None),
        device=device,
    )

    tt_args = Tt3DetrArgs()
    tt_args.modules = DotAccessibleDict({"encoder": reference_model})
    tt_args.device = device
    tt_args.parameters = DotAccessibleDict({"encoder": parameters})
    tt_encoder = build_ttnn_encoder(tt_args)

    tt_src = ttnn.from_torch(
        src.permute(1, 0, 2),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_output = tt_encoder(src=tt_src, xyz=xyz)

    all_passing = True
    for idx, (tt_out, torch_out) in enumerate(zip(tt_output, ref_out)):
        if not isinstance(tt_out, torch.Tensor):
            tt_out = ttnn.to_torch(tt_out)
            tt_out = tt_out.permute(1, 0, 2)
            tt_out = torch.reshape(tt_out, torch_out.shape)

        passing, pcc_message = comp_pcc(torch_out, tt_out, pcc=0.99)
        logger.info(f"Output {idx} PCC: {pcc_message}")
        logger.info(comp_allclose(torch_out, tt_out))

        if passing:
            logger.info(f"Output {idx} Test Passed!")
        else:
            logger.warning(f"Output {idx} Test Failed!")
            all_passing = False

    assert all_passing, "One or more outputs failed PCC check with threshold 0.99"
