# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PCC tests for Lingbot-VA WanTransformer3DModel (video and action path).

Compares TT model (lingbot_va.tt) vs reference. Uses (1,1) submesh; loads from cache or mmap to avoid OOM.
"""

import gc
import os
import tempfile
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn

# Ensure tt-metal root is on path when running from various working directories
_tt_metal_root = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
if str(_tt_metal_root) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_tt_metal_root))

from models.experimental.lingbot_va.reference.transformer_wan import (
    WanTransformer3DModel as TorchWanTransformer3DModel,
)
from models.experimental.lingbot_va.tt.transformer_wan import WanTransformer3DModel
from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.cache import model_cache_dir
from models.tt_dit.utils.check import assert_quality
from models.tt_dit.utils.test import line_params

# Lingbot-VA model config (matches reference/model.py)
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

TT_METAL_HOME = os.environ.get("TT_METAL_HOME", _tt_metal_root)
LINGBOT_VA_CHECKPOINT = Path(TT_METAL_HOME) / "models/experimental/lingbot_va/reference/checkpoints/transformer"


def _make_parallel_config(mesh_device, sp_axis, tp_axis):
    return DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tuple(mesh_device.shape)[tp_axis]),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=tuple(mesh_device.shape)[sp_axis]),
        cfg_parallel=None,
    )


def _make_ccl_manager(mesh_device, num_links, topology):
    return CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)


def _make_wan_transformer(*, mesh_device, ccl_manager, parallel_config, is_fsdp, num_layers=NUM_LAYERS):
    """Build Lingbot-VA TT WanTransformer3DModel (in_channels=48, action_dim=30)."""
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


def _make_grid_id(B: int, F: int, H: int, W: int, patch_size: tuple, device: torch.device) -> torch.Tensor:
    """Build grid_id (B, 3, L) for reference model RoPE; L = (F//p0)*(H//p1)*(W//p2)."""
    p0, p1, p2 = patch_size
    f_p, h_p, w_p = F // p0, H // p1, W // p2
    f_idx = torch.arange(f_p, dtype=torch.float32, device=device)
    h_idx = torch.arange(h_p, dtype=torch.float32, device=device)
    w_idx = torch.arange(w_p, dtype=torch.float32, device=device)
    ff, hh, ww = torch.meshgrid(f_idx, h_idx, w_idx, indexing="ij")
    grid_id = torch.stack([ff.flatten(), hh.flatten(), ww.flatten()], dim=0)  # (3, L)
    grid_id = grid_id.unsqueeze(0).repeat(B, 1, 1)  # (B, 3, L)
    return grid_id


def _ref_output_to_bcfhw(
    ref_out: torch.Tensor, B: int, F: int, H: int, W: int, patch_size: tuple, out_c: int
) -> torch.Tensor:
    """Reshape reference output (B, L*n, out_c) to (B, out_c, F, H, W) to match TT output."""
    p0, p1, p2 = patch_size
    patch_F, patch_H, patch_W = F // p0, H // p1, W // p2
    n = p0 * p1 * p2
    L = patch_F * patch_H * patch_W
    assert ref_out.shape == (B, L * n, out_c)
    x = ref_out.reshape(B, patch_F, patch_H, patch_W, p0, p1, p2, out_c)
    x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(B, out_c, F, H, W)
    return x


def _make_action_grid_id(
    B: int,
    F_action: int,
    action_per_frame: int,
    device: torch.device,
    f_w: int = 1,
    f_shift: int = 0,
) -> torch.Tensor:
    """Build grid_id (B, 3, L) for action path RoPE; L = F_action * action_per_frame * 1."""
    f_idx = torch.arange(f_shift, F_action + f_shift, dtype=torch.float32, device=device) * f_w
    h_idx = torch.arange(action_per_frame, dtype=torch.float32, device=device)
    w_idx = torch.arange(1, dtype=torch.float32, device=device)
    ff, hh, ww = torch.meshgrid(f_idx, h_idx, w_idx, indexing="ij")
    ff_offset = (torch.ones([action_per_frame], device=device).cumsum(0) / (action_per_frame + 1)).view(1, -1, 1)
    ff = ff + ff_offset
    hh = torch.ones_like(hh, device=device) * -1
    ww = torch.ones_like(ww, device=device) * -1
    grid_id = torch.stack([ff.flatten(), hh.flatten(), ww.flatten()], dim=0)  # (3, L)
    grid_id = grid_id.unsqueeze(0).repeat(B, 1, 1)  # (B, 3, L)
    return grid_id


# Video path: (1,1) submesh; cache or mmap load.
@pytest.mark.parametrize(
    ("mesh_device", "submesh_shape", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    [
        pytest.param((1, 2), (1, 1), 0, 1, 1, line_params, ttnn.Topology.Linear, False, id="1x1_single_device"),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    ("B", "T", "H", "W", "prompt_seq_len"),
    [
        pytest.param(1, 8, 24, 24, 77, id="lingbot_va_short"),
    ],
)
def test_wan_transformer_model(
    mesh_device: ttnn.MeshDevice,
    submesh_shape: tuple,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    B: int,
    T: int,
    H: int,
    W: int,
    prompt_seq_len: int,
    topology: ttnn.Topology,
    is_fsdp: bool,
    reset_seeds,
) -> None:
    """PCC: TT WanTransformer3DModel vs reference (video path). Uses cache or mmap to avoid OOM."""
    MIN_PCC = 0.992_000
    MAX_RMSE = 0.15

    if not LINGBOT_VA_CHECKPOINT.exists():
        pytest.skip(f"Lingbot-VA checkpoint not found: {LINGBOT_VA_CHECKPOINT}")

    mesh_device = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    parallel_config = _make_parallel_config(mesh_device, sp_axis, tp_axis)
    ccl_manager = _make_ccl_manager(mesh_device, num_links, topology)

    torch_model = TorchWanTransformer3DModel.from_pretrained(
        str(LINGBOT_VA_CHECKPOINT),
        torch_dtype=torch.float32,
        attn_mode="torch",
    )
    torch_model.eval()

    torch.manual_seed(0)
    spatial_input = torch.randn((B, IN_CHANNELS, T, H, W), dtype=torch.float32)
    prompt_input = torch.randn((B, prompt_seq_len, TEXT_DIM), dtype=torch.float32)
    timestep_input = torch.randint(0, 1000, (B,), dtype=torch.float32)

    grid_id = _make_grid_id(B, T, H, W, PATCH_SIZE, spatial_input.device)
    F_patched = T // PATCH_SIZE[0]
    timesteps_ref = timestep_input.unsqueeze(1).expand(B, F_patched)
    input_dict = {
        "noisy_latents": spatial_input,
        "text_emb": prompt_input,
        "timesteps": timesteps_ref,
        "grid_id": grid_id,
    }

    with torch.no_grad():
        ref_out = torch_model(input_dict, action_mode=False, train_mode=False)
    ref_out_bcfhw = _ref_output_to_bcfhw(ref_out, B, T, H, W, PATCH_SIZE, OUT_CHANNELS)

    cache_dir = model_cache_dir(
        model_name="lingbot_va",
        subfolder="transformer",
        parallel_config=parallel_config,
        mesh_shape=tuple(mesh_device.shape),
        required=False,
    )
    use_cache = cache_dir is not None and cache_dir.is_dir()

    if use_cache:
        del torch_model
        gc.collect()
        tt_model = _make_wan_transformer(
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            is_fsdp=is_fsdp,
            num_layers=NUM_LAYERS,
        )
        start = time.time()
        tt_model.load(cache_dir)
        end = time.time()
        logger.info("Loaded TT model from cache in {} s", end - start)
    else:
        state_dict = torch_model.state_dict()
        fd, state_path = tempfile.mkstemp(suffix=".pt")
        try:
            os.close(fd)
            torch.save(state_dict, state_path)
            del torch_model
            del state_dict
            gc.collect()
            state_dict = torch.load(state_path, map_location=torch.device("cpu"), mmap=True)
            tt_model = _make_wan_transformer(
                mesh_device=mesh_device,
                ccl_manager=ccl_manager,
                parallel_config=parallel_config,
                is_fsdp=is_fsdp,
                num_layers=NUM_LAYERS,
            )
            start = time.time()
            tt_model.load_torch_state_dict(state_dict)
            end = time.time()
            logger.info("Loaded TT model from state dict (mmap) in {} s", end - start)
        finally:
            try:
                os.unlink(state_path)
            except OSError:
                pass

    logger.info(
        "Running TT model (video path): spatial {}, prompt {}, timestep {}",
        spatial_input.shape,
        prompt_input.shape,
        timestep_input.shape,
    )
    tt_spatial_out = tt_model(
        spatial=spatial_input,
        prompt=prompt_input,
        timestep=timestep_input,
        grid_id=grid_id,
        action_mode=False,
    )
    del tt_model
    gc.collect()

    assert_quality(ref_out_bcfhw, tt_spatial_out, pcc=MIN_PCC, relative_rmse=MAX_RMSE)


ACTION_PER_FRAME = 16
F_ACTION = 8


# Action path: (1,1) submesh; cache or mmap load.
@pytest.mark.parametrize(
    ("mesh_device", "submesh_shape", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    [
        pytest.param((1, 2), (1, 1), 0, 1, 1, line_params, ttnn.Topology.Linear, False, id="1x1_single_device"),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    ("B", "F_action", "action_per_frame", "prompt_seq_len"),
    [
        pytest.param(1, F_ACTION, ACTION_PER_FRAME, 77, id="lingbot_va_action_short"),
    ],
)
def test_wan_transformer_model_action_mode(
    mesh_device: ttnn.MeshDevice,
    submesh_shape: tuple,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    B: int,
    F_action: int,
    action_per_frame: int,
    prompt_seq_len: int,
    topology: ttnn.Topology,
    is_fsdp: bool,
    reset_seeds,
) -> None:
    """PCC: TT WanTransformer3DModel vs reference (action path). Output (B, N, action_dim)."""
    MIN_PCC = 0.992_000
    MAX_RMSE = 0.15

    if not LINGBOT_VA_CHECKPOINT.exists():
        pytest.skip(f"Lingbot-VA checkpoint not found: {LINGBOT_VA_CHECKPOINT}")

    mesh_device = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    parallel_config = _make_parallel_config(mesh_device, sp_axis, tp_axis)
    ccl_manager = _make_ccl_manager(mesh_device, num_links, topology)

    torch_model = TorchWanTransformer3DModel.from_pretrained(
        str(LINGBOT_VA_CHECKPOINT),
        torch_dtype=torch.float32,
        attn_mode="torch",
    )
    torch_model.eval()

    torch.manual_seed(0)
    spatial_input = torch.randn((B, ACTION_DIM, F_action, action_per_frame, 1), dtype=torch.float32)
    prompt_input = torch.randn((B, prompt_seq_len, TEXT_DIM), dtype=torch.float32)
    timestep_input = torch.randint(0, 1000, (B,), dtype=torch.float32)
    timesteps_ref = timestep_input.unsqueeze(1).expand(B, F_action)

    grid_id = _make_action_grid_id(B, F_action, action_per_frame, spatial_input.device)
    input_dict = {
        "noisy_latents": spatial_input,
        "text_emb": prompt_input,
        "timesteps": timesteps_ref,
        "grid_id": grid_id,
    }

    with torch.no_grad():
        ref_out = torch_model(input_dict, action_mode=True, train_mode=False)

    cache_dir = model_cache_dir(
        model_name="lingbot_va",
        subfolder="transformer",
        parallel_config=parallel_config,
        mesh_shape=tuple(mesh_device.shape),
        required=False,
    )
    use_cache = cache_dir is not None and cache_dir.is_dir()

    if use_cache:
        del torch_model
        gc.collect()
        tt_model = _make_wan_transformer(
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            is_fsdp=is_fsdp,
            num_layers=NUM_LAYERS,
        )
        start = time.time()
        tt_model.load(cache_dir)
        end = time.time()
        logger.info("Loaded TT model from cache (action path) in {} s", end - start)
    else:
        state_dict = torch_model.state_dict()
        fd, state_path = tempfile.mkstemp(suffix=".pt")
        try:
            os.close(fd)
            torch.save(state_dict, state_path)
            del torch_model
            del state_dict
            gc.collect()
            state_dict = torch.load(state_path, map_location=torch.device("cpu"), mmap=True)
            tt_model = _make_wan_transformer(
                mesh_device=mesh_device,
                ccl_manager=ccl_manager,
                parallel_config=parallel_config,
                is_fsdp=is_fsdp,
                num_layers=NUM_LAYERS,
            )
            start = time.time()
            tt_model.load_torch_state_dict(state_dict)
            end = time.time()
            logger.info("Loaded TT model from state dict (mmap, action path) in {} s", end - start)
        finally:
            try:
                os.unlink(state_path)
            except OSError:
                pass

    logger.info(
        "Running TT model (action path): spatial {}, prompt {}, timestep {}",
        spatial_input.shape,
        prompt_input.shape,
        timestep_input.shape,
    )
    tt_spatial_out = tt_model(
        spatial=spatial_input,
        prompt=prompt_input,
        timestep=timestep_input,
        grid_id=grid_id,
        action_mode=True,
    )
    del tt_model
    gc.collect()

    assert ref_out.shape == tt_spatial_out.shape, (ref_out.shape, tt_spatial_out.shape)
    assert_quality(ref_out, tt_spatial_out, pcc=MIN_PCC, relative_rmse=MAX_RMSE)
