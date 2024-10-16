# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import os
import ttnn
import importlib

llama_reference_mod = importlib.import_module(
    "models.demos.t3000.llama2_70b.reference.llama-models.models.llama3.reference_impl.multimodal.model"
)
from models.demos.llama3.tt.llama_image_mlp import TtLlamaImageFeedForward
from models.demos.llama3.tt.model_config import TtModelArgs
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from models.utility_functions import skip_for_grayskull


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "seq_len",
    (
        # 64 * 1024,
        # 32 * 1024,
        # 5120,
        # 32,
        4224,
    ),
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("FAKE_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
def test_llama_mlp_inference(mesh_device, seq_len, use_program_cache, reset_seeds, ensure_gc):
    dtype = ttnn.bfloat16

    mesh_device.enable_async(True)

    model_args = TtModelArgs(mesh_device)
    state_dict = torch.load(model_args.consolidated_weights_path, map_location=torch.device("cpu"))

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    first_layer_prefix = "vision_model.vision_encoder.transformer.resblocks.31.mlp."
    # TODO: regex match for this / filter dict keys
    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }

    model_args.WEIGHTS_DTYPE = dtype

    """
            self.patch_size = 14
            self.vision_encoder = VisionEncoder(
            max_num_tiles=4,
            image_size=args.vision_chunk_size,
            patch_size=self.patch_size,
            n_global_layers=8,
            global_model=True,
            return_intermediate=return_intermediate,
        )

        class VisionEncoder(nn.Module):
            def __init__(
                self,
                max_num_tiles: int,
                ckpt_path: str = None,
                image_size: int = 224,
                patch_size: int = 14,
                width: int = 1280,
                layers: int = 32,
                heads: int = 16,
                mlp_ratio: float = 4.0,
                act_layer: Callable = nn.GELU,
                in_channels: int = 3,
                load_ckpt: bool = False,
                n_global_layers: int = 2,
                global_model: bool = False,
                return_intermediate=None,
                ...

                self.global_transformer = ImageTransformer(
                    width, n_global_layers, heads, mlp_ratio, act_layer=act_layer, gated=True

            self.mlp = ImageFeedForward(
                dim=d_model,
                hidden_dim=int(mlp_ratio * d_model),
                dropout=0.0,
                act_layer=act_layer,
            )
        )
    """

    dim = 1280
    mlp_ratio = 4.0
    act_layer = torch.nn.GELU
    dropout = 0.0
    reference_model = llama_reference_mod.ImageFeedForward(
        dim=dim,
        hidden_dim=int(mlp_ratio * dim),
        dropout=dropout,
        act_layer=act_layer,
    )
    reference_model.load_state_dict(partial_state_dict)
    reference_model.bfloat16()

    tt_model = TtLlamaImageFeedForward(
        mesh_device=mesh_device,
        args=model_args,
        state_dict=state_dict,
        state_dict_prefix=first_layer_prefix,
        weight_cache_path=model_args.weight_cache_path(dtype),
        dtype=dtype,
    )
    torch_input = torch.randn(1, 1, seq_len, dim)
    reference_output = reference_model(torch_input).squeeze()
    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_output = tt_model(tt_input)

    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1))[
        :, :1, :, :
    ].squeeze()

    pcc_required = 0.99
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")

    if passing:
        logger.info("Llama_MLP Passed!")
    else:
        logger.warning("Llama_MLP Failed!")

    assert passing, f"Llama_MLP output does not meet PCC requirement {pcc_required}: {pcc_message}."
