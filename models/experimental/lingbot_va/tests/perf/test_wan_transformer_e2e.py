# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
E2E Performance test for WanTransformer3DModel (lingbot_va.tt).

Measures end-to-end performance including compile time and inference time.
Uses checkpoint weights if available, otherwise falls back to zero weights.
Generates standardized performance reports in CSV format.
"""

import os
import sys
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn

# Add parent paths
_tt_metal_root = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from models.experimental.lingbot_va.reference.WanTransformer3D import (
    WanTransformer3DModel as TorchWanTransformer3DModel,
)
from models.experimental.lingbot_va.tt.transformer_wan import WanTransformer3DModel
from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.test import line_params
from models.perf.perf_utils import prep_perf_report


# =============================================================================
# CONFIGURATION
# =============================================================================
TT_METAL_HOME = os.environ.get("TT_METAL_HOME", _tt_metal_root)
CHECKPOINT_PATH = Path(os.path.join(TT_METAL_HOME, "models/experimental/lingbot_va/reference/checkpoints/transformer"))
SEED = 42
# Model config (match PCC test_transformer_wan.py and reference/model.py)
DIM = 24 * 128  # 3072
FFN_DIM = 14336
NUM_HEADS = 24
IN_CHANNELS = 48
OUT_CHANNELS = 48
ACTION_DIM = 30
TEXT_DIM = 4096
FREQ_DIM = 256
NUM_LAYERS = 30
PATCH_SIZE = (1, 2, 2)
CROSS_ATTN_NORM = True
EPS = 1e-6
ROPE_MAX_SEQ_LEN = 1024


def _make_parallel_config(mesh_device, sp_axis, tp_axis):
    return DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tuple(mesh_device.shape)[tp_axis]),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=tuple(mesh_device.shape)[sp_axis]),
        cfg_parallel=None,
    )


def _make_ccl_manager(mesh_device, num_links, topology):
    return CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)


def _make_grid_id(B: int, F: int, H: int, W: int, patch_size: tuple, device: torch.device) -> torch.Tensor:
    """Build grid_id (B, 3, L) for RoPE; L = (F//p0)*(H//p1)*(W//p2)."""
    p0, p1, p2 = patch_size
    f_p, h_p, w_p = F // p0, H // p1, W // p2
    f_idx = torch.arange(f_p, dtype=torch.float32, device=device)
    h_idx = torch.arange(h_p, dtype=torch.float32, device=device)
    w_idx = torch.arange(w_p, dtype=torch.float32, device=device)
    ff, hh, ww = torch.meshgrid(f_idx, h_idx, w_idx, indexing="ij")
    grid_id = torch.stack([ff.flatten(), hh.flatten(), ww.flatten()], dim=0)
    grid_id = grid_id.unsqueeze(0).repeat(B, 1, 1)
    return grid_id


def _fill_tt_module_params_zeros(module) -> None:
    """Fill all parameters of a TT Module with zeros via load_torch_tensor.

    Bypasses load_torch_state_dict so submodules' _prepare_torch_state (which expect
    reference-format state, e.g. WanTimeTextImageEmbedding time_proj reshape) are not run.
    """
    for _, child in module.named_children():
        _fill_tt_module_params_zeros(child)
    for _, param in module.named_parameters():
        param.load_torch_tensor(torch.zeros(param.total_shape, dtype=torch.float32))


def _make_wan_transformer(*, mesh_device, ccl_manager, parallel_config, is_fsdp, num_layers: int):
    return WanTransformer3DModel(
        patch_size=PATCH_SIZE,
        num_heads=NUM_HEADS,
        dim=DIM,
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        action_dim=ACTION_DIM,
        text_dim=TEXT_DIM,
        freq_dim=FREQ_DIM,
        ffn_dim=FFN_DIM,
        num_layers=num_layers,
        cross_attn_norm=CROSS_ATTN_NORM,
        eps=EPS,
        rope_max_seq_len=ROPE_MAX_SEQ_LEN,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
    )


def _load_model_weights(tt_model, num_layers: int):
    """Load weights from checkpoint if available, otherwise use zeros.

    Uses the same approach as test_wan_transformer_perf.py which works reliably.

    Args:
        tt_model: TT model to load weights into
        num_layers: Number of layers in the model
    """
    if CHECKPOINT_PATH.exists():
        logger.info(f"Loading weights from checkpoint: {CHECKPOINT_PATH}")
        torch_model = TorchWanTransformer3DModel.from_pretrained(
            str(CHECKPOINT_PATH),
            torch_dtype=torch.float32,
            attn_mode="torch",
        )
        torch_model.eval()
        state_dict = torch_model.state_dict()

        # Filter state dict to only include layers that match num_layers
        if num_layers < NUM_LAYERS:
            filtered_state_dict = {}
            for key, value in state_dict.items():
                # Extract layer number from keys like "blocks.2.attn1.to_q.weight"
                if key.startswith("blocks."):
                    try:
                        layer_num = int(key.split(".")[1])
                        if layer_num < num_layers:
                            filtered_state_dict[key] = value
                    except (ValueError, IndexError):
                        # Keep non-layer keys (e.g., embedding layers)
                        filtered_state_dict[key] = value
                else:
                    # Keep non-block keys (e.g., embedding layers)
                    filtered_state_dict[key] = value
            state_dict = filtered_state_dict
            logger.info(f"Filtered state dict to {num_layers} layers (from {NUM_LAYERS})")

        tt_model.load_torch_state_dict(state_dict)
        del torch_model
        logger.info("Loaded weights from checkpoint")
    else:
        logger.info(f"Checkpoint not found at {CHECKPOINT_PATH}, using zero weights")
        # TT parameters have no data until loaded. Fill each parameter with zeros directly
        # so we avoid load_torch_state_dict (which runs _prepare_torch_state that expects
        # reference-format state, e.g. condition_embedder time_proj).
        _fill_tt_module_params_zeros(tt_model)


# =============================================================================
# PYTEST TEST FUNCTION
# =============================================================================


@pytest.mark.parametrize(
    ("mesh_device", "submesh_shape", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    [
        pytest.param(
            (1, 1),
            (1, 1),
            0,
            0,
            1,
            line_params,
            ttnn.Topology.Linear,
            False,
            id="1x1_single_device",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("num_iterations", [32])
@pytest.mark.parametrize(
    ("B", "T", "H", "W", "prompt_seq_len", "num_layers", "expected_compile_time", "expected_throughput_fps"),
    [
        pytest.param(1, 8, 24, 24, 77, 2, 10.0, 30.0, id="small_8x24x24_2layers"),
        pytest.param(1, 8, 24, 24, 77, 30, 100.0, 3.0, id="small_8x24x24_30layers"),
    ],
)
def test_perf_wan_transformer_e2e(
    mesh_device: ttnn.MeshDevice,
    submesh_shape: tuple,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    topology: ttnn.Topology,
    is_fsdp: bool,
    num_iterations: int,
    B: int,
    T: int,
    H: int,
    W: int,
    prompt_seq_len: int,
    num_layers: int,
    expected_compile_time: float,
    expected_throughput_fps: float,
) -> None:
    """E2E performance test for WanTransformer3DModel with standardized reporting."""
    checkpoint_path = Path(CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        pytest.skip(f"Checkpoint not found: {checkpoint_path}")

    mesh_device = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    parallel_config = _make_parallel_config(mesh_device, sp_axis, tp_axis)
    ccl_manager = _make_ccl_manager(mesh_device, num_links, topology)

    # Initialize model
    tt_model = _make_wan_transformer(
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
        num_layers=num_layers,
    )

    batch_size = B

    # Load weights from checkpoint if available, otherwise use zeros
    # Uses the same approach as test_wan_transformer_perf.py
    _load_model_weights(tt_model, num_layers)

    # Create test inputs
    torch.manual_seed(SEED)
    spatial = torch.randn((B, IN_CHANNELS, T, H, W), dtype=torch.float32)
    prompt = torch.randn((B, prompt_seq_len, TEXT_DIM), dtype=torch.float32)
    timestep = torch.randint(0, 1000, (B,), dtype=torch.float32)
    grid_id = _make_grid_id(B, T, H, W, PATCH_SIZE, spatial.device)

    # Note: WanTransformer3DModel expects torch.Tensor inputs and handles sharding internally
    # via preprocess_spatial_input(). The Pipeline API expects ttnn.Tensor inputs and tries to
    # handle resharding, which conflicts with the model's internal sharding logic.
    # Therefore, we use direct model calls (same as test_wan_transformer_perf.py) which works reliably.

    # First run (includes compile time)
    logger.info("First run (includes compile time)...")
    start = time.time()
    _ = tt_model(
        spatial=spatial,
        prompt=prompt,
        timestep=timestep,
        grid_id=grid_id,
        action_mode=False,
    )
    first_run_time = time.time() - start
    logger.info(f"First run time (compile + inference): {first_run_time:.2f} s")

    # Subsequent runs (inference only)
    logger.info(f"Running {num_iterations} inference iterations...")
    start = time.time()
    for _ in range(num_iterations):
        out = tt_model(
            spatial=spatial,
            prompt=prompt,
            timestep=timestep,
            grid_id=grid_id,
            action_mode=False,
        )
        # Touch output so forward completes before we stop the timer
        _ = out.shape
    end = time.time()

    inference_time = (end - start) / num_iterations
    total_time = first_run_time  # Total time for first run equivalent

    logger.info(f"Average model time={1000.0 * inference_time:.2f} ms")
    logger.info(f"Average model performance={num_iterations * batch_size / (end - start):.2f} fps")

    # Generate performance report CSV file
    # CSV file is saved in the current working directory with format:
    # perf_{model_name}_{comments}_{today}.csv
    # Example: perf_wan-transformer-3d_B1_T8_H24_W24_layers30_2026_03_09.csv
    prep_perf_report(
        model_name="wan-transformer-3d",
        batch_size=batch_size,
        inference_and_compile_time=total_time,  # Total time (compile + inference)
        inference_time=inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=batch_size / expected_throughput_fps,
        comments=f"B{B}_T{T}_H{H}_W{W}_layers{num_layers}",
    )
    logger.info(f"Performance report CSV saved to current working directory: {os.getcwd()}")

    ttnn.close_mesh_device(mesh_device)
