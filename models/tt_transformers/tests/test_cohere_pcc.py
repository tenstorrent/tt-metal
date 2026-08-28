# SPDX-FileCopyrightText: © 2026 Canada Quant Labs (org-internal — bounty tt-metal#49307 track)
# SPDX-License-Identifier: Apache-2.0
#
# Command-R (c4ai-command-r-v01) single-layer PCC harness — P5b.
#
# Validates the command-r-bringup scaffold (models/tt_transformers/tt/cohere/) against
# the fp32 CPU reference activation dumps produced on the QB2 host by
# tt-rd scripts/command-r/cpu_reference_capture.py (P4). Bounty gate: PCC >= 0.99.
#
# Reference dump contract (per prompt file prompt{NN}.npz, fp32 numpy):
#   embed            embed_tokens output           [1, S, 8192]  == input of decoder layer 0
#   layer{LL:02d}_out  decoder layer LL residual output [1, S, 8192]  (== input of layer LL+1)
#   final_norm       model.norm output             [1, S, 8192]
#   logits           lm_head output * logit_scale  [1, S, 256000]
#
# Env knobs:
#   COHERE_REF_DIR     dump directory (default /root/v4run/results/command-r/reference)
#   COHERE_REF_PROMPT  prompt index (default 0)
#   COHERE_LAYER       decoder layer index (default 0; >0 uses layer{LL-1}_out as input)
#
# Run on the QB2 host (ttnn needs the Blackhole devices; see tt-rd
# docs/command-r-bounty-prep.md for the on-box env — do NOT restart tt-qwen38):
#   python_env/bin/python -m pytest models/tt_transformers/tests/test_cohere_pcc.py -v
#
# Distributed-norm path (P5b, 2026-08-28): TtCohereLayerNorm mirrors the proven
# tt_distributed_rmsnorm pattern (models/tt_transformers/tt/ccl.py) —
# ttnn.layer_norm_pre_all_gather -> tt_all_gather (cluster_axis=1) ->
# ttnn.layer_norm_post_all_gather with a TP-sharded gamma (ShardTensor2dMesh
# dims=(None, 2)). Both tests feed WIDTH-SHARDED inputs — the exact layout the
# decoder layer feeds its input_layernorm on a (1, N) mesh — so both gates
# exercise the same production path. The plain interleaved ttnn.layer_norm
# over-allocates L1 at dim 8192 on this build (observed on-box 2026-08-28:
# CBs 3,363,712 B > 1,572,864 B) and is kept only as the single-device fallback.

import os

import numpy as np
import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.cohere.cohere_decoder import CohereDecoderLayer
from models.tt_transformers.tt.cohere.cohere_norm import TtCohereLayerNorm
from models.tt_transformers.tt.common import Mode, get_rot_transformation_mat
from models.tt_transformers.tt.model_config import ModelArgs
from models.tt_transformers.tt.rope import get_rot_mats

MESH_DEVICES = {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}


def _load_ref_npz():
    ref_dir = os.environ.get("COHERE_REF_DIR", "/root/v4run/results/command-r/reference")
    pidx = int(os.environ.get("COHERE_REF_PROMPT", "0"))
    path = os.path.join(ref_dir, f"prompt{pidx:02d}.npz")
    if not os.path.exists(path):
        pytest.skip(
            f"CPU reference dump {path} not present — run tt-rd "
            "scripts/command-r/cpu_reference_capture.py on the QB2 host first"
        )
    logger.info(f"[cohere-pcc] reference dump: {path}")
    return np.load(path)


def _layer_io(ref, layer_idx):
    """(layer input, layer reference output) from the dump for decoder layer `layer_idx`."""
    out_key = f"layer{layer_idx:02d}_out"
    if out_key not in ref:
        pytest.skip(f"{out_key} not in reference dump")
    if layer_idx == 0:
        in_key = "embed"
    else:
        in_key = f"layer{layer_idx - 1:02d}_out"
        if in_key not in ref:
            pytest.skip(f"{in_key} (input for layer {layer_idx}) not in reference dump")
    return torch.from_numpy(ref[in_key]).float(), torch.from_numpy(ref[out_key]).float()


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [MESH_DEVICES.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_cohere_layernorm_pcc(mesh_device, reset_seeds, ensure_gc):
    """TtCohereLayerNorm (mean-centering LayerNorm) vs torch fp32 reference of the HF
    CohereLayerNorm formula on a REAL captured activation. Gate: PCC >= 0.9999
    (mirrors test_rms_norm.py — norm ops are expected well above the 0.99 bounty gate)."""
    layer_idx = int(os.environ.get("COHERE_LAYER", "0"))
    ref = _load_ref_npz()
    layer_in, _ = _layer_io(ref, layer_idx)
    seq_len = layer_in.shape[1]
    pad = (-seq_len) % 32  # tile multiple; per-row norm makes padded rows inert
    if pad:
        layer_in = torch.nn.functional.pad(layer_in, (0, 0, 0, pad))

    model_args = ModelArgs(mesh_device, max_batch_size=1, max_seq_len=128, cache_hf=True)
    model_args.n_layers = 1
    state_dict = model_args.load_state_dict()
    prefix = model_args.get_state_dict_prefix("", layer_idx)

    tt_ccl = TT_CCL(mesh_device)
    tt_norm = TtCohereLayerNorm(
        device=mesh_device,
        dim=model_args.dim,
        eps=model_args.norm_eps,
        state_dict=state_dict,
        state_dict_prefix=prefix,
        weight_cache_path=None,
        weight_dtype=ttnn.bfloat16,
        weight_key="attention_norm",
        tt_ccl=tt_ccl,
    )

    # torch fp32 reference — HF v4.39.3 CohereLayerNorm (mean-centering, weight-only)
    w = state_dict[f"{prefix}attention_norm.weight"].float()
    mean = layer_in.mean(-1, keepdim=True)
    var = layer_in.var(-1, unbiased=False, keepdim=True)
    reference_output = (layer_in - mean) * torch.rsqrt(var + model_args.norm_eps) * w

    tt_input = ttnn.from_torch(
        layer_in.unsqueeze(1).to(torch.bfloat16),  # [1, 1, S_pad, 8192]
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 3), mesh_shape=list(mesh_device.shape)),
    )
    tt_out = tt_norm(tt_input, mode="prefill")
    tt_output_torch = ttnn.to_torch(
        tt_out,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=model_args.cluster_shape),
    )[:1, :1].reshape(1, -1, model_args.dim)  # first replica — forward() all-gathers the normed output to full width (DistributedNorm contract)

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc=0.9999)
    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")
    assert passing, f"CohereLayerNorm PCC < 0.9999: {pcc_message}"


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [MESH_DEVICES.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))],
    indirect=True,
)
@pytest.mark.parametrize("max_seq_len", (128,))  # matches the tested prefill path; prompts are < 32 tokens
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_cohere_decoder_layer_prefill_pcc(max_seq_len, mesh_device, reset_seeds, ensure_gc):
    """Full CohereDecoderLayer (parallel block: ONE LayerNorm feeding BOTH MHA + SwiGLU
    branches, residual + attn + mlp) in prefill mode vs the captured fp32 layer output.
    Bounty gate: PCC >= 0.99."""
    dtype = ttnn.bfloat8_b  # stack default for single-layer PCC (test_decoder_prefill.py)
    layer_idx = int(os.environ.get("COHERE_LAYER", "0"))
    ref = _load_ref_npz()
    layer_in, ref_out = _layer_io(ref, layer_idx)
    seq_len = layer_in.shape[1]
    assert seq_len <= max_seq_len, f"prompt seq {seq_len} > max_seq_len {max_seq_len}"
    if seq_len < max_seq_len:  # causal mask: real positions never attend to padding
        layer_in = torch.nn.functional.pad(layer_in, (0, 0, 0, max_seq_len - seq_len))

    model_args = ModelArgs(
        mesh_device, max_batch_size=1, max_seq_len=max_seq_len, cache_hf=True, use_hf_rope=False
    )
    model_args.n_layers = 1
    state_dict = model_args.load_state_dict()

    rot_mats = get_rot_mats(
        head_dim=model_args.head_dim,
        device=mesh_device,
        seq_len=max_seq_len,
        theta=model_args.rope_theta,
        rope_scaling=model_args.rope_scaling,
    )
    transformation_mats = {
        "prefill": ttnn.as_tensor(
            get_rot_transformation_mat(model_args.head_dim),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
    }

    tt_ccl = TT_CCL(mesh_device)
    tt_layer = CohereDecoderLayer(
        args=model_args,
        mesh_device=mesh_device,
        tt_ccl=tt_ccl,
        dtype=dtype,
        state_dict=state_dict,
        layer_num=layer_idx,
        weight_cache_path=model_args.weight_cache_path(dtype),
        transformation_mats=transformation_mats,
        paged_attention_config=None,  # default (non-paged) prefill attention
        prefetcher=None,
    )

    tt_input = model_args.prepare_residual_tensor_prefill(layer_in.to(torch.bfloat16))
    tt_out = tt_layer(
        tt_input,
        None,
        rot_mats_global=rot_mats,
        rot_mats_local=None,
        user_id=0,
        mode=Mode.PREFILL,
        page_table=None,
    )
    tt_out = ttnn.to_torch(
        tt_out,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),
    )
    tt_output_torch = tt_out[:, 0:1, :seq_len, : model_args.dim].view(1, seq_len, -1)

    passing, pcc_message = comp_pcc(ref_out, tt_output_torch, pcc=0.99)
    logger.info(comp_allclose(ref_out, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")
    if passing:
        logger.info("CohereDecoderLayer prefill Passed (bounty gate PCC >= 0.99)!")
    assert passing, f"CohereDecoderLayer prefill PCC < 0.99 (bounty gate): {pcc_message}"
