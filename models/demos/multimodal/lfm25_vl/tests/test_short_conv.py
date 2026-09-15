# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""PCC test for ``TtLfm2ShortConv`` against the pure-torch reference in ``reference/functional.py``.

Uses ``reference/functional.short_conv`` (rather than instantiating the HF ``Lfm2ShortConv``
module directly) so this test also works in environments where the installed
``transformers`` does not yet ship LFM2-VL support -- see ``README.md``.
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.demos.multimodal.lfm25_vl.reference.functional import short_conv
from models.demos.multimodal.lfm25_vl.tt.model_config import ModelArgs
from models.demos.multimodal.lfm25_vl.tt.short_conv import TtLfm2ShortConv
from models.tt_transformers.tt.common import Mode


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        {
            "N150": (1, 1),
            "N300": (1, 2),
            "P150": (1, 1),
        }.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))
    ],
    indirect=True,
)
@pytest.mark.parametrize("seq_len, mode", ((128, Mode.PREFILL), (1, Mode.DECODE)), ids=["prefill", "decode"])
@pytest.mark.parametrize("batch_size", (1,))
def test_short_conv_inference(seq_len, mode, batch_size, mesh_device, reset_seeds, ensure_gc):
    dtype = ttnn.bfloat16

    model_args = ModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=128, cache_hf=True)
    model_args.n_layers = max(model_args.layer_types.index("conv") + 1, 1)
    state_dict = model_args.load_state_dict()

    conv_layer_idx = model_args._first_layer_of_type("conv")
    prefix = model_args.get_state_dict_prefix("ShortConv", conv_layer_idx)
    prefix = prefix if prefix.endswith(".") else f"{prefix}."

    in_proj_w = state_dict[f"{prefix}in_proj.weight"]
    out_proj_w = state_dict[f"{prefix}out_proj.weight"]
    conv_w = state_dict[f"{prefix}conv.weight"]
    in_proj_b = state_dict.get(f"{prefix}in_proj.bias")
    out_proj_b = state_dict.get(f"{prefix}out_proj.bias")
    conv_b = state_dict.get(f"{prefix}conv.bias")

    torch_input = torch.randn(batch_size, seq_len, model_args.dim, dtype=torch.float32)
    reference_output = short_conv(
        torch_input, in_proj_w, out_proj_w, conv_w, in_proj_b=in_proj_b, out_proj_b=out_proj_b, conv_b=conv_b
    )

    tt_model = TtLfm2ShortConv(
        mesh_device=mesh_device,
        args=model_args,
        state_dict=state_dict,
        state_dict_prefix=prefix,
        weight_cache_path=model_args.weight_cache_path(dtype),
        layer_num=conv_layer_idx,
        dtype=dtype,
    )

    tt_input_shape = (1, 1, seq_len, model_args.dim) if mode == Mode.PREFILL else (batch_size, 1, 1, model_args.dim)
    tt_input = ttnn.from_torch(
        torch_input.reshape(tt_input_shape),
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    logger.info("Run ShortConv")
    tt_output = tt_model(tt_input, mode=mode)
    # out_proj's output dim is fractured across the mesh (column-parallel, matching the sharded
    # tt_transformers residual stream), so reassemble the hidden dim by concatenating on dim=-1.
    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))
    tt_output_torch = tt_output_torch.reshape(batch_size, seq_len, model_args.dim)

    pcc_required = 0.98
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")
    assert passing, f"ShortConv output does not meet PCC requirement {pcc_required}: {pcc_message}."
