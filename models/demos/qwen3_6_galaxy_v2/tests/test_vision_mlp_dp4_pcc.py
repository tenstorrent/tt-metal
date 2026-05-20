# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
V2-VISION-V2 (DP=4 variant): vision MLP topology for parallel video frames.

The replicated-across-32 mode validated correctness (`test_vision_mlp_pcc.py`)
but wastes ~31x compute. For VLM serving, the vision encoder runs once per
image (vs decoder per-token) so absolute cost is small — but for video with
multiple frames, we want **DP=4 across the 4 mesh columns** so all 4 frames
encode in parallel. Within a column (8 chips on the row axis) we keep the
existing replication. Result: per-frame compute is the same as replicated mode,
but 4 frames complete in one forward pass (cols-as-DP).

This test validates the topology end-to-end on real qwen3.6 weights:
  1. Build 4 distinct random "frames" stacked along dim=0.
  2. `ShardTensor2dMesh(dims=(None, 0))` — shard batch dim across cols,
     replicate across rows.
  3. Run the qwen3_vl `MLP` block once.
  4. Concat output and slice per-column; PCC each frame against its
     own torch reference.

If any column's output PCC < 0.99 vs that column's expected output, the
DP routing or the on-chip math is broken.

Run:
  export HF_MODEL=Qwen/Qwen3.6-27B
  export MESH_DEVICE=TG
  pytest models/demos/qwen3_6_galaxy_v2/tests/test_vision_mlp_dp4_pcc.py -v -s
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
@pytest.mark.parametrize("rows", (14336,))
@pytest.mark.parametrize("n_frames", (4,))  # one frame per mesh column
@pytest.mark.parametrize("layer_num", (0,))
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_vision_mlp_qwen36_dp4(rows, n_frames, layer_num, mesh_device, reset_seeds, ensure_gc):
    if not os.environ.get("HF_MODEL"):
        os.environ["HF_MODEL"] = "Qwen/Qwen3.6-27B"

    model_args = Qwen36VisionModelArgs(
        mesh_device,
        dummy_weights=False,
        max_batch_size=n_frames,
        max_seq_len=rows,
    )
    assert (
        model_args.cluster_shape[1] == n_frames
    ), f"DP=4 expects mesh col count == n_frames; got cluster_shape={model_args.cluster_shape}, n_frames={n_frames}"

    H = model_args.hf_config.vision_config.hidden_size
    logger.info(
        f"DP={n_frames} vision MLP on {mesh_device.shape}: hidden={H}, "
        f"intermediate={model_args.hf_config.vision_config.intermediate_size}"
    )

    reference_full = model_args.reference_vision_model()
    reference_mlp = reference_full.blocks[layer_num].mlp

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

    # 4 distinct frames stacked on dim=0. Each column will process one frame.
    torch_input = torch.randn(n_frames, 1, rows, H)
    reference_outputs = [reference_mlp(torch_input[i : i + 1]) for i in range(n_frames)]

    # DP=4 across cols (cluster_axis=1), replicate across rows (cluster_axis=0).
    # ShardTensor2dMesh dims=(None, 0): None along axis-0 = replicate;
    # 0 along axis-1 = shard tensor dim=0 across 4 cols.
    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, 0),
            mesh_shape=model_args.cluster_shape,
        ),
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    logger.info(f"Running TT MLP DP={n_frames} on {mesh_device.shape}")
    tt_output = tt_model(tt_input, "prefill")

    # Pull all 32 chips concatenated along tensor dim 0. Row-major mesh order:
    # index 0..3 = mesh-row 0 (cols 0..3); index 4..7 = mesh-row 1 (cols 0..3); etc.
    # Since rows are replicas, we slice the first 4 indices (one frame per col).
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )
    # shape: [32, 1, rows, H_padded]; first 4 indices = mesh-row 0, columns 0..3
    tt_output_torch = tt_output_torch[:n_frames, ..., :H]
    logger.info(f"tt_output_torch shape after slicing: {tuple(tt_output_torch.shape)}")

    pcc_required = 0.99
    all_pass = True
    for col_idx in range(n_frames):
        passing, pcc_message = comp_pcc(
            reference_outputs[col_idx], tt_output_torch[col_idx : col_idx + 1], pcc_required
        )
        logger.info(
            f"  col {col_idx}: {comp_allclose(reference_outputs[col_idx], tt_output_torch[col_idx : col_idx + 1])} | PCC: {pcc_message}"
        )
        if not passing:
            all_pass = False
            logger.warning(f"DP={n_frames} col {col_idx} FAILED: {pcc_message}")
    assert all_pass, f"At least one column failed PCC>={pcc_required}"
