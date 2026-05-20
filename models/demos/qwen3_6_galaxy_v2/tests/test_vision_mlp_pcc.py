# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
V2-VISION-V2 first PCC test: qwen3_vl's TT vision `MLP` block run against the
real qwen3.6 `model.visual.blocks.{N}.mlp.*` weights on BH GLX 8×4.

Mirrors `models/demos/qwen3_vl/tests/test_mlp.py` but flips `dummy_weights`
to False (real weights via the V1 helper) and routes through the bridge
`Qwen36VisionModelArgs` (registers `qwen3_5` model_type with AutoConfig
and overrides `reference_vision_model()` for safetensors-direct loading).

PCC > 0.99 required at layer 0; expected to hold uniformly across all 27
vision blocks per the V1 strict-load guarantee.

Run:
  export HF_MODEL=Qwen/Qwen3.6-27B
  pytest models/demos/qwen3_6_galaxy_v2/tests/test_vision_mlp_pcc.py -v -s
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.demos.qwen3_6_galaxy_v2.tt.vision_model_args import Qwen36VisionModelArgs
from models.demos.qwen3_vl.tt.vision_mlp import MLP
from models.tt_transformers.tt.load_checkpoints import convert_hf_to_meta


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4), "BH_GLX": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("rows", (14336,))  # qwen3_vl test_mlp default
@pytest.mark.parametrize("batch_size", (1,))
@pytest.mark.parametrize("layer_num", (0,))
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_vision_mlp_qwen36_pcc(rows, batch_size, layer_num, mesh_device, reset_seeds, ensure_gc):
    if not os.environ.get("HF_MODEL"):
        os.environ["HF_MODEL"] = "Qwen/Qwen3.6-27B"
    dtype = ttnn.bfloat8_b
    mode = "prefill"

    model_args = Qwen36VisionModelArgs(
        mesh_device,
        dummy_weights=False,
        max_batch_size=batch_size,
        max_seq_len=rows,
    )
    logger.info(
        f"qwen3.6 vision: hidden={model_args.hf_config.vision_config.hidden_size}, "
        f"intermediate={model_args.hf_config.vision_config.intermediate_size}, "
        f"depth={model_args.hf_config.vision_config.depth}, "
        f"out_hidden_size={model_args.hf_config.vision_config.out_hidden_size}"
    )

    reference_full = model_args.reference_vision_model()
    reference_mlp = reference_full.blocks[layer_num].mlp

    # Mirror qwen3_vl/tests/test_mlp.py state_dict packaging.
    state_dict = convert_hf_to_meta(reference_mlp.state_dict(), model_args.head_dim)
    state_dict_prefix = model_args.get_state_dict_prefix("MLP", layer_num)
    state_dict = {f"{state_dict_prefix}.{k}": v for k, v in state_dict.items()}

    tt_model = MLP(
        mesh_device=mesh_device,
        args=model_args,
        state_dict=state_dict,
        weight_cache_path=None,
        layer_num=layer_num,
    )

    torch_input = torch.randn(1, 1, rows, model_args.hf_config.vision_config.hidden_size)
    reference_output = reference_mlp(torch_input)

    # qwen3.6 vision encoder runs REPLICATED across the full 32-chip BH GLX mesh
    # (vs the qwen3_vl test which shards input dim=3 for galaxy — that path has
    # an unresolved matmul K-dim mismatch in qwen3_vl/tt/vision_mlp.py because
    # the weight stays replicated while input is sharded).
    #
    # Replicated execution wastes ~31x compute but vision runs ONCE per image
    # at prefill (vs decoder per-token), so the absolute cost is small. The
    # text decoder uses the full 32-chip mesh independently.
    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    logger.info(f"Running TT MLP (layer {layer_num}) on {mesh_device.shape} mesh (replicated)")
    tt_output = tt_model(tt_input, mode)

    # All 32 chips produce identical output (replicated). Pull the first chip's copy.
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )
    # shape: [num_devices=32, 1, rows, hidden] — slice first chip's result
    tt_output_torch = tt_output_torch[:1, :, :, : model_args.hf_config.vision_config.hidden_size]

    pcc_required = 0.99
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")
    assert passing, f"qwen3.6 vision MLP (layer {layer_num}) PCC {pcc_required} not met: {pcc_message}"
