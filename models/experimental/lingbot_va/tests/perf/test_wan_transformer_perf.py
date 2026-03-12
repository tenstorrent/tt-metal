# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Performance test for WanTransformer3DModel (lingbot_va.tt).

Measures forward-pass latency and throughput for the TT transformer on device.
Uses random inputs. Optionally loads weights from checkpoint if available.
Optional: reduce num_layers for faster runs.
"""

import os
import statistics
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn

_tt_metal_root = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
if str(_tt_metal_root) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_tt_metal_root))

from models.experimental.lingbot_va.reference.WanTransformer3D import (
    WanTransformer3DModel as TorchWanTransformer3DModel,
)
from models.experimental.lingbot_va.tt.transformer_wan import WanTransformer3DModel
from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.test import line_params


# =============================================================================
# CONFIGURATION
# =============================================================================
TT_METAL_HOME = os.environ.get("TT_METAL_HOME", _tt_metal_root)
CHECKPOINT_PATH = Path(TT_METAL_HOME) / "models/experimental/lingbot_va/reference/checkpoints/transformer"
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


# =============================================================================
# PYTEST TEST FUNCTION
# =============================================================================


# Single device configuration.
@pytest.mark.parametrize(
    ("mesh_device", "submesh_shape", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    [
        pytest.param((1, 1), (1, 1), 0, 0, 1, line_params, ttnn.Topology.Linear, False, id="1x1_single_device"),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    ("B", "T", "H", "W", "prompt_seq_len", "num_layers", "num_warmup", "num_iters"),
    [
        pytest.param(1, 8, 24, 24, 77, 30, 1, 5, id="small_8x24x24_2layers"),
    ],
)
def test_wan_transformer_3d_perf(
    mesh_device: ttnn.MeshDevice,
    submesh_shape: tuple,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    topology: ttnn.Topology,
    is_fsdp: bool,
    B: int,
    T: int,
    H: int,
    W: int,
    prompt_seq_len: int,
    num_layers: int,
    num_warmup: int,
    num_iters: int,
) -> None:
    """Measure WanTransformer3DModel forward latency (video path) with random inputs."""
    mesh_device = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    parallel_config = _make_parallel_config(mesh_device, sp_axis, tp_axis)
    ccl_manager = _make_ccl_manager(mesh_device, num_links, topology)

    tt_model = _make_wan_transformer(
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
        num_layers=num_layers,
    )

    # Load weights from checkpoint if available, otherwise use zeros
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

    torch.manual_seed(42)
    spatial = torch.randn((B, IN_CHANNELS, T, H, W), dtype=torch.float32)
    prompt = torch.randn((B, prompt_seq_len, TEXT_DIM), dtype=torch.float32)
    timestep = torch.randint(0, 1000, (B,), dtype=torch.float32)
    grid_id = _make_grid_id(B, T, H, W, PATCH_SIZE, spatial.device)

    # Warmup
    for _ in range(num_warmup):
        _ = tt_model(
            spatial=spatial,
            prompt=prompt,
            timestep=timestep,
            grid_id=grid_id,
            action_mode=False,
        )

    # Timed runs (wall-clock: includes host->device, compute, device->host)
    latencies_s = []
    for _ in range(num_iters):
        t0 = time.perf_counter()
        out = tt_model(
            spatial=spatial,
            prompt=prompt,
            timestep=timestep,
            grid_id=grid_id,
            action_mode=False,
        )
        # Touch output so forward completes before we stop the timer
        _ = out.shape
        latencies_s.append(time.perf_counter() - t0)

    mean_s = statistics.mean(latencies_s)
    std_s = statistics.stdev(latencies_s) if len(latencies_s) > 1 else 0.0
    mean_ms = mean_s * 1000
    std_ms = std_s * 1000
    throughput = 1.0 / mean_s if mean_s > 0 else 0.0

    logger.info(
        f"WanTransformer3DModel perf: B={B} T={T} H={H} W={W} num_layers={num_layers} -> {mean_ms:.2f} ± {std_ms:.2f} ms ({throughput:.2f} samples/s)"
    )

    print("\n" + "=" * 60)
    print("WanTransformer3DModel performance")
    print("=" * 60)
    print(f"Input: spatial {spatial.shape}, prompt {prompt.shape}, grid_id {grid_id.shape}")
    print(f"Config: num_layers={num_layers}, patch_size={PATCH_SIZE}")
    print(f"Latency: {mean_ms:.2f} ± {std_ms:.2f} ms (n={num_iters})")
    print(f"Throughput: {throughput:.2f} samples/s")
    print("=" * 60)

    ttnn.close_mesh_device(mesh_device)
