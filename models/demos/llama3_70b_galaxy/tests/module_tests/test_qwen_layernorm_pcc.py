# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from models.demos.llama3_70b_galaxy.tt.qwen_model_config import TtQwenModelArgs
from models.common.rmsnorm import RMSNorm as TtRMSNorm
from models.demos.llama3_70b_galaxy.tt.distributed_norm import DistributedNorm
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.model import RMSNorm as RefRMSNorm
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from models.utility_functions import skip_for_grayskull
from models.demos.llama3_70b_galaxy.tt.prefetcher_common import TtLlamaPrefetcherSetup
from models.demos.llama3_70b_galaxy.tt.llama_ccl import TT_CCL
import os


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": True,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_size",
    (32,),
)
@pytest.mark.parametrize(
    "max_seq_len",
    (256,),
)
def test_qwen_layernorm_pcc(
    max_seq_len,
    batch_size,
    mesh_device,
    reset_seeds,
    ensure_gc,
):
    """
    Test PCC between TT and reference RMS norm for Qwen model.
    Loads attn_out and h tensors from TT decoder (.pt files) and
    reference tensors (x, at, fin) from reference model (.pth files).
    Passes TT tensors through ff_norm and compares with reference fin tensor.
    """
    dtype = ttnn.bfloat8_b

    # Check if the required .pth files exist
    attn_out_path = "attn_out.pt"
    h_path = "h.pt"
    x_path = "x.pth"
    at_path = "at.pth"
    fin_path = "fin.pth"

    if not os.path.exists(attn_out_path):
        pytest.skip(f"Required file {attn_out_path} not found. Run decoder test first to generate it.")
    if not os.path.exists(h_path):
        pytest.skip(f"Required file {h_path} not found. Run decoder test first to generate it.")
    if not os.path.exists(x_path):
        pytest.skip(f"Required file {x_path} not found. Run reference model first to generate it.")
    if not os.path.exists(at_path):
        pytest.skip(f"Required file {at_path} not found. Run reference model first to generate it.")
    if not os.path.exists(fin_path):
        pytest.skip(f"Required file {fin_path} not found. Run reference model first to generate it.")

    # Load the saved tensors
    logger.info("Loading saved tensors from .pth files")
    attn_out_torch = torch.load(attn_out_path)
    h_torch = torch.load(h_path)

    # Load reference tensors
    x_ref = torch.load(x_path)
    at_ref = torch.load(at_path)
    fin_ref = torch.load(fin_path)

    logger.info(f"Loaded attn_out shape: {attn_out_torch.shape}")
    logger.info(f"Loaded h shape: {h_torch.shape}")
    logger.info(f"Loaded reference x shape: {x_ref.shape}")
    logger.info(f"Loaded reference at shape: {at_ref.shape}")
    logger.info(f"Loaded reference fin shape: {fin_ref.shape}")

    # Initialize model args
    model_args = TtQwenModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=max_seq_len, dummy_weights=False)
    model_args.n_layers = 2  # We need at least 2 layers to test layer 1

    state_dict = model_args.load_state_dict()

    # Setup prefetcher and CCL
    prefetcher_setup = TtLlamaPrefetcherSetup(
        mesh_device,
        n_tensors=5,
        n_layers=model_args.n_layers,
    )
    mesh_device.set_sub_device_stall_group(
        [prefetcher_setup.prefetcher_sub_device_id, prefetcher_setup.worker_sub_device_id]
    )

    tt_ccl = TT_CCL(mesh_device, model_args, prefetcher_setup.worker_sub_device_id, use_qwen_mlp=True)

    # Initialize just the ff_norm module for layer 1
    layer_num = 1
    state_dict_prefix = model_args.get_state_dict_prefix("", layer_num)

    # Create the inner RMSNorm for ff_norm
    tt_inner_norm = TtRMSNorm(
        device=mesh_device,
        dim=model_args.dim,
        state_dict=state_dict,
        state_dict_prefix=state_dict_prefix,
        weight_cache_path=None if model_args.dummy_weights else model_args.weight_cache_path(dtype),
        weight_dtype=ttnn.bfloat16,
        weight_key="ffn_norm",
        is_distributed=model_args.is_distributed_norm,
        sharded_program_config=model_args.model_config["SHARDED_NORM_MLP_PRGM_CFG"],
        sharded_output_config=model_args.model_config["SHARDED_MLP_INPUT_MEMCFG"],
        output_mem_config=model_args.model_config["SHARDED_FF12_RING_MEMCFG"],
    )

    # Wrap it in DistributedNorm (as done in llama_decoder.py lines 99-117)
    tt_ff_norm = DistributedNorm(
        tt_inner_norm,
        model_args,
        TG=model_args.is_galaxy,
        tt_ccl=tt_ccl,
        ccl_topology=model_args.model_config["CCL_TOPOLOGY"],
    )

    # Setup reference model for layer 1 - just the ffn_norm
    layer_prefix = model_args.get_state_dict_prefix("TtTransformerBlock", layer_num)
    ffn_norm_prefix = layer_prefix + "ffn_norm."
    partial_state_dict = {k[len(ffn_norm_prefix) :]: v for k, v in state_dict.items() if k.startswith(ffn_norm_prefix)}
    reference_ffn_norm = RefRMSNorm(dim=model_args.dim, eps=model_args.norm_eps)
    reference_ffn_norm.load_state_dict(partial_state_dict)

    logger.info("Converting torch tensors to TT tensors")

    # Convert attn_out to TT tensor format (similar to how it's done in decoder)
    # Expand to full batch size and proper shape
    attn_out_expanded = attn_out_torch.expand(1, 1, batch_size, model_args.dim).contiguous()
    h_expanded = h_torch.expand(1, 1, batch_size, model_args.dim).contiguous()

    # Convert to TT tensors with proper memory config for decode mode
    skip_mem_cfg = model_args.model_config["DECODE_RESIDUAL_MEMCFG"]

    attn_out_tt = ttnn.from_torch(
        attn_out_expanded,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, -1), mesh_shape=model_args.cluster_shape),
        memory_config=skip_mem_cfg,
    )

    h_tt = ttnn.from_torch(
        h_expanded,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, -1), mesh_shape=model_args.cluster_shape),
        memory_config=skip_mem_cfg,
    )

    logger.info("Running TT ff_norm forward pass")
    # Pass through ff_norm as done in llama_decoder.py decode mode (line 192)
    # ff_in_sharded, _ = self.ff_norm(attn_out, h, mode)
    ff_in_sharded_tt, _ = tt_ff_norm(attn_out_tt, h_tt, "decode")

    # Convert TT output back to torch
    ff_in_torch = ttnn.to_torch(
        ff_in_sharded_tt,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),
    )[:, :1, :, :]

    logger.info("Computing reference solution")
    # Use the reference fin tensor directly from the reference model
    # This is the output of ffn_norm(h) where h = x + at (from line 342 in model.py)
    reference_output = fin_ref
    reference_output = reference_output.view(1, 1, batch_size, model_args.dim)

    # Verify our understanding by computing it manually as well
    h_ref = x_ref + at_ref  # This should match the h computation in reference model
    manual_reference = reference_ffn_norm(h_ref).view(1, 1, batch_size, model_args.dim)

    logger.info("Verifying reference computation consistency")
    ref_consistency_pcc = comp_pcc(reference_output, manual_reference)
    logger.info(f"Reference consistency PCC: {ref_consistency_pcc[1]}")

    logger.info("Computing PCC between TT and reference results")
    # Compare the results
    passing, pcc_message = comp_pcc(reference_output, ff_in_torch)

    logger.info(comp_allclose(reference_output, ff_in_torch))
    logger.info(f"PCC: {pcc_message}")

    # Cleanup
    tt_ccl.close()

    if passing:
        logger.info("Qwen LayerNorm PCC Test Passed!")
    else:
        logger.warning("Qwen LayerNorm PCC Test Failed!")

    assert passing, f"PCC value is lower than 0.99. Got: {pcc_message}"
