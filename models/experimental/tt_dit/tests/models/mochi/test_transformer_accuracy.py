# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import torch
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from ....models.transformers.transformer_mochi import MochiTransformer3DModel
from ....parallel.manager import CCLManager
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....utils.cache import get_cache_path, load_cache_dict
from diffusers import MochiTransformer3DModel as TorchMochiTransformer3DModel


@pytest.mark.parametrize(
    "mesh_device, sp_axis, tp_axis, num_links",
    [
        [(2, 4), 1, 0, 1],
        [(4, 8), 1, 0, 4],
    ],
    ids=[
        "2x4sp1tp0",
        "4x8sp1tp0",
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_transformer_accuracy(
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
):
    """
    This test checks accuracy of the transformer with real inputs and ground truth outputs
    across multiple steps of inference.

    It's expected that ground truth outputs are in a directory MOCHI_GROUND_TRUTH_DIR
    with folder structure:
    MOCHI_GROUND_TRUTH_DIR/
        run_metadata.json
        step_0000/
            encoder_attention_mask.pt
            encoder_hidden_states.pt
            hidden_states.pt
            timestep.pt
            output.pt
        ...

    Each torch tensor has batch=2 for CFG. The first index is the uncond input,
    second is the cond input.

    For each step, we run the transformer
    1. with ground truth input (teacher forcing)
        i. uncond
        ii. cond

    On each iteration, we print the PCC, RMSE, ATOL, RTOL between the TT and ground truth outputs, for the above 2 cases.
    """
    # Get ground truth directory from environment
    ground_truth_dir = os.environ.get("MOCHI_GROUND_TRUTH_DIR")
    if ground_truth_dir is None:
        pytest.skip("MOCHI_GROUND_TRUTH_DIR environment variable not set")

    if not os.path.exists(ground_truth_dir):
        pytest.skip(f"Ground truth directory {ground_truth_dir} does not exist")

    # Load run metadata
    metadata_path = os.path.join(ground_truth_dir, "run_metadata.json")
    if not os.path.exists(metadata_path):
        pytest.skip(f"Run metadata file {metadata_path} does not exist")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    logger.info(f"Loaded metadata: {metadata}")

    # Set up parallel configuration
    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]

    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        tensor_parallel=ParallelFactor(factor=tp_factor, mesh_axis=tp_axis),
        sequence_parallel=ParallelFactor(factor=sp_factor, mesh_axis=sp_axis),
    )

    # Create CCL manager
    ccl_manager = CCLManager(
        mesh_device=mesh_device,
        num_links=num_links,
        topology=ttnn.Topology.Linear,
    )

    # Load torch model for reference
    torch_dtype = torch.float32
    torch_model = TorchMochiTransformer3DModel.from_pretrained(
        "genmo/mochi-1-preview", subfolder="transformer", torch_dtype=torch_dtype
    )
    torch_model.eval()

    # Create TT model
    tt_model = MochiTransformer3DModel(
        patch_size=torch_model.config.patch_size,
        num_attention_heads=torch_model.config.num_attention_heads,
        attention_head_dim=torch_model.config.attention_head_dim,
        num_layers=torch_model.config.num_layers,
        pooled_projection_dim=torch_model.config.pooled_projection_dim,
        in_channels=torch_model.config.in_channels,
        text_embed_dim=torch_model.config.text_embed_dim,
        time_embed_dim=torch_model.config.time_embed_dim,
        activation_fn=torch_model.config.activation_fn,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=True,
    )

    cache_path = get_cache_path(
        model_name="mochi-1-preview",
        subfolder="transformer",
        parallel_config=parallel_config,
        dtype="bf16",
    )
    assert os.path.exists(
        cache_path
    ), "Cache path does not exist. Run test_mochi_transformer_model_caching first with the desired parallel config."
    cache_dict = load_cache_dict(cache_path)
    tt_model.from_cached_state_dict(cache_dict)

    # Get list of step directories
    step_dirs = [d for d in os.listdir(ground_truth_dir) if d.startswith("step_")]
    step_dirs.sort()

    if not step_dirs:
        pytest.skip("No step directories found in ground truth directory")

    logger.info(f"Found {len(step_dirs)} steps: {step_dirs}")

    # Initialize tracking lists for metrics
    step_numbers = []
    pcc_uncond_values = []
    rmse_uncond_values = []
    atol_uncond_values = []
    rtol_uncond_values = []
    pcc_cond_values = []
    rmse_cond_values = []
    atol_cond_values = []
    rtol_cond_values = []

    # Process each step
    for step_idx, step_dir in enumerate(step_dirs):
        step_path = os.path.join(ground_truth_dir, step_dir)
        logger.info(f"\n=== Processing {step_dir} ===")

        # Load ground truth data
        encoder_attention_mask = torch.load(os.path.join(step_path, "encoder_attention_mask.pt")).float()
        encoder_hidden_states = torch.load(os.path.join(step_path, "encoder_hidden_states.pt")).float()
        hidden_states = torch.load(os.path.join(step_path, "hidden_states.pt")).float()
        timestep = torch.load(os.path.join(step_path, "timestep.pt")).float()
        ground_truth_output = torch.load(os.path.join(step_path, "output.pt")).float()

        logger.info(f"Loaded tensors for {step_dir}:")
        logger.info(f"  encoder_attention_mask.shape: {encoder_attention_mask.shape}")
        logger.info(f"  encoder_hidden_states.shape: {encoder_hidden_states.shape}")
        logger.info(f"  hidden_states.shape: {hidden_states.shape}")
        logger.info(f"  timestep.shape: {timestep.shape}")
        logger.info(f"  ground_truth_output.shape: {ground_truth_output.shape}")

        # Separate uncond (index 0) and cond (index 1) inputs
        encoder_attention_mask_uncond = encoder_attention_mask[0:1]
        encoder_attention_mask_cond = encoder_attention_mask[1:2]
        encoder_hidden_states_uncond = encoder_hidden_states[0:1]
        encoder_hidden_states_cond = encoder_hidden_states[1:2]
        hidden_states_uncond = hidden_states[0:1]
        hidden_states_cond = hidden_states[1:2]
        timestep_uncond = timestep[0:1]
        timestep_cond = timestep[1:2]
        ground_truth_output_uncond = ground_truth_output[0:1]
        ground_truth_output_cond = ground_truth_output[1:2]

        # 1. Teacher forcing with ground truth inputs
        logger.info("\n--- Teacher Forcing ---")
        tt_output_uncond_gt = tt_model(
            spatial=hidden_states_uncond,
            prompt=encoder_hidden_states_uncond,
            timestep=timestep_uncond,
            prompt_attention_mask=encoder_attention_mask_uncond,
        )
        tt_output_uncond_gt = tt_output_uncond_gt.to(torch.float32)

        # Calculate metrics for uncond teacher forcing
        pcc_uncond_gt = torch.corrcoef(
            torch.stack([ground_truth_output_uncond.flatten(), tt_output_uncond_gt.flatten()])
        )[0, 1].item()
        rmse_uncond_gt = torch.sqrt(torch.mean((ground_truth_output_uncond - tt_output_uncond_gt) ** 2)).item()
        atol_uncond_gt = torch.max(torch.abs(ground_truth_output_uncond - tt_output_uncond_gt)).item()
        rtol_uncond_gt = (atol_uncond_gt / torch.max(torch.abs(ground_truth_output_uncond))).item()

        logger.info(
            f"Uncond Teacher Forcing - PCC: {pcc_uncond_gt:.6f}, RMSE: {rmse_uncond_gt:.6f}, ATOL: {atol_uncond_gt:.6f}, RTOL: {rtol_uncond_gt:.6f}"
        )

        # 1.ii. Cond with ground truth input
        tt_output_cond_gt = tt_model(
            spatial=hidden_states_cond,
            prompt=encoder_hidden_states_cond,
            timestep=timestep_cond,
            prompt_attention_mask=encoder_attention_mask_cond,
        )
        tt_output_cond_gt = tt_output_cond_gt.to(torch.float32)

        # Calculate metrics for cond teacher forcing
        pcc_cond_gt = torch.corrcoef(torch.stack([ground_truth_output_cond.flatten(), tt_output_cond_gt.flatten()]))[
            0, 1
        ].item()
        rmse_cond_gt = torch.sqrt(torch.mean((ground_truth_output_cond - tt_output_cond_gt) ** 2)).item()
        atol_cond_gt = torch.max(torch.abs(ground_truth_output_cond - tt_output_cond_gt)).item()
        rtol_cond_gt = (atol_cond_gt / torch.max(torch.abs(ground_truth_output_cond))).item()

        logger.info(
            f"Cond Teacher Forcing - PCC: {pcc_cond_gt:.6f}, RMSE: {rmse_cond_gt:.6f}, ATOL: {atol_cond_gt:.6f}, RTOL: {rtol_cond_gt:.6f}"
        )

        # Store metrics for plotting
        step_numbers.append(step_idx)
        pcc_uncond_values.append(pcc_uncond_gt)
        rmse_uncond_values.append(rmse_uncond_gt)
        atol_uncond_values.append(atol_uncond_gt)
        rtol_uncond_values.append(rtol_uncond_gt)
        pcc_cond_values.append(pcc_cond_gt)
        rmse_cond_values.append(rmse_cond_gt)
        atol_cond_values.append(atol_cond_gt)
        rtol_cond_values.append(rtol_cond_gt)

        logger.info(f"Completed {step_dir}")

    logger.info(f"\nCompleted accuracy test for {len(step_dirs)} steps")

    # Create output directory structure
    ground_truth_dir_name = os.path.basename(os.path.normpath(ground_truth_dir))
    output_dir = os.path.join("mochi_comparison", ground_truth_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")

    # Create plots for correctness metrics
    logger.info("Creating correctness metrics plots...")

    # Set up the figure with 4 subplots (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Mochi Transformer Correctness Metrics Across Inference Steps", fontsize=16, fontweight="bold")

    # Convert step numbers to numpy array for plotting
    steps = np.array(step_numbers)

    # Plot 1: PCC (Pearson Correlation Coefficient)
    axes[0, 0].plot(steps, pcc_uncond_values, "b-o", label="Uncond", linewidth=2, markersize=4)
    axes[0, 0].plot(steps, pcc_cond_values, "r-s", label="Cond", linewidth=2, markersize=4)
    axes[0, 0].set_title("PCC (Pearson Correlation Coefficient)", fontsize=14, fontweight="bold")
    axes[0, 0].set_xlabel("Inference Step")
    axes[0, 0].set_ylabel("PCC")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    axes[0, 0].set_ylim(0.95, 1.0)  # PCC should be close to 1

    # Plot 2: RMSE (Root Mean Square Error)
    axes[0, 1].plot(steps, rmse_uncond_values, "b-o", label="Uncond", linewidth=2, markersize=4)
    axes[0, 1].plot(steps, rmse_cond_values, "r-s", label="Cond", linewidth=2, markersize=4)
    axes[0, 1].set_title("RMSE (Root Mean Square Error)", fontsize=14, fontweight="bold")
    axes[0, 1].set_xlabel("Inference Step")
    axes[0, 1].set_ylabel("RMSE")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    axes[0, 1].set_yscale("log")  # Use log scale for RMSE

    # Plot 3: ATOL (Absolute Tolerance)
    axes[1, 0].plot(steps, atol_uncond_values, "b-o", label="Uncond", linewidth=2, markersize=4)
    axes[1, 0].plot(steps, atol_cond_values, "r-s", label="Cond", linewidth=2, markersize=4)
    axes[1, 0].set_title("ATOL (Absolute Tolerance)", fontsize=14, fontweight="bold")
    axes[1, 0].set_xlabel("Inference Step")
    axes[1, 0].set_ylabel("ATOL")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    axes[1, 0].set_yscale("log")  # Use log scale for ATOL

    # Plot 4: RTOL (Relative Tolerance)
    axes[1, 1].plot(steps, rtol_uncond_values, "b-o", label="Uncond", linewidth=2, markersize=4)
    axes[1, 1].plot(steps, rtol_cond_values, "r-s", label="Cond", linewidth=2, markersize=4)
    axes[1, 1].set_title("RTOL (Relative Tolerance)", fontsize=14, fontweight="bold")
    axes[1, 1].set_xlabel("Inference Step")
    axes[1, 1].set_ylabel("RTOL")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    axes[1, 1].set_yscale("log")  # Use log scale for RTOL

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Save the plot
    plot_filename = os.path.join(
        output_dir, f"mochi_transformer_accuracy_metrics_{sp_factor}x{tp_factor}sp{sp_axis}tp{tp_axis}.png"
    )
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    logger.info(f"Saved correctness metrics plot to {plot_filename}")

    # Also save as PDF for better quality
    plot_filename_pdf = os.path.join(
        output_dir, f"mochi_transformer_accuracy_metrics_{sp_factor}x{tp_factor}sp{sp_axis}tp{tp_axis}.pdf"
    )
    plt.savefig(plot_filename_pdf, dpi=300, bbox_inches="tight")
    logger.info(f"Saved correctness metrics plot to {plot_filename_pdf}")

    # Show the plot (optional, comment out if running headless)
    # plt.show()

    # Print summary statistics
    logger.info("\n=== Summary Statistics ===")
    logger.info(
        f"PCC - Uncond: mean={np.mean(pcc_uncond_values):.6f}, std={np.std(pcc_uncond_values):.6f}, min={np.min(pcc_uncond_values):.6f}, max={np.max(pcc_uncond_values):.6f}"
    )
    logger.info(
        f"PCC - Cond: mean={np.mean(pcc_cond_values):.6f}, std={np.std(pcc_cond_values):.6f}, min={np.min(pcc_cond_values):.6f}, max={np.max(pcc_cond_values):.6f}"
    )
    logger.info(
        f"RMSE - Uncond: mean={np.mean(rmse_uncond_values):.6f}, std={np.std(rmse_uncond_values):.6f}, min={np.min(rmse_uncond_values):.6f}, max={np.max(rmse_uncond_values):.6f}"
    )
    logger.info(
        f"RMSE - Cond: mean={np.mean(rmse_cond_values):.6f}, std={np.std(rmse_cond_values):.6f}, min={np.min(rmse_cond_values):.6f}, max={np.max(rmse_cond_values):.6f}"
    )
    logger.info(
        f"ATOL - Uncond: mean={np.mean(atol_uncond_values):.6f}, std={np.std(atol_uncond_values):.6f}, min={np.min(atol_uncond_values):.6f}, max={np.max(atol_uncond_values):.6f}"
    )
    logger.info(
        f"ATOL - Cond: mean={np.mean(atol_cond_values):.6f}, std={np.std(atol_cond_values):.6f}, min={np.min(atol_cond_values):.6f}, max={np.max(atol_cond_values):.6f}"
    )
    logger.info(
        f"RTOL - Uncond: mean={np.mean(rtol_uncond_values):.6f}, std={np.std(rtol_uncond_values):.6f}, min={np.min(rtol_uncond_values):.6f}, max={np.max(rtol_uncond_values):.6f}"
    )
    logger.info(
        f"RTOL - Cond: mean={np.mean(rtol_cond_values):.6f}, std={np.std(rtol_cond_values):.6f}, min={np.min(rtol_cond_values):.6f}, max={np.max(rtol_cond_values):.6f}"
    )

    # Save metrics data to JSON for further analysis
    metrics_data = {
        "step_numbers": step_numbers,
        "pcc_uncond": pcc_uncond_values,
        "pcc_cond": pcc_cond_values,
        "rmse_uncond": rmse_uncond_values,
        "rmse_cond": rmse_cond_values,
        "atol_uncond": atol_uncond_values,
        "atol_cond": atol_cond_values,
        "rtol_uncond": rtol_uncond_values,
        "rtol_cond": rtol_cond_values,
        "metadata": metadata,
        "parallel_config": {"sp_factor": sp_factor, "tp_factor": tp_factor, "sp_axis": sp_axis, "tp_axis": tp_axis},
    }

    metrics_filename = os.path.join(
        output_dir, f"mochi_transformer_accuracy_metrics_{sp_factor}x{tp_factor}sp{sp_axis}tp{tp_axis}.json"
    )
    with open(metrics_filename, "w") as f:
        json.dump(metrics_data, f, indent=2)
    logger.info(f"Saved metrics data to {metrics_filename}")

    plt.close()  # Clean up the plot
