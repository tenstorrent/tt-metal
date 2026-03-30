# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""PCC: Lingbot-VA TT WanTransformer3DModel vs reference (video and action paths)."""

import gc
import json
import os
import sys
import time
from pathlib import Path

# Avoid inspector writing under generated/ when the filesystem is full (common on dev boxes).
os.environ.setdefault("TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT", "0")

import pytest
import torch
from loguru import logger

import ttnn

_tt_metal_root = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
if str(_tt_metal_root) not in sys.path:
    sys.path.insert(0, str(_tt_metal_root))

from models.experimental.lingbot_va.reference.transformer_wan import (
    WanTransformer3DModel as TorchWanTransformer3DModel,
)
from models.experimental.lingbot_va.tt.transformer_wan import WanTransformer3DModel
from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.cache import model_cache_dir
from models.experimental.lingbot_va.tests.mesh_utils import mesh_shape_request_param
from models.tt_dit.utils.check import assert_quality
from models.tt_dit.utils.test import line_params

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

DEMO_FRAME_CHUNK = 2
DEMO_LATENT_H = 24
DEMO_LATENT_W = 20
DEMO_PROMPT_SEQ_LEN = 512
DEMO_ACTION_PER_FRAME = 16


def _make_parallel_config(mesh_device, sp_axis, tp_axis):
    return DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tuple(mesh_device.shape)[tp_axis]),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=tuple(mesh_device.shape)[sp_axis]),
        cfg_parallel=None,
    )


def _make_ccl_manager(mesh_device, num_links, topology):
    return CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)


def _load_torch_reference() -> TorchWanTransformer3DModel:
    path = str(LINGBOT_VA_CHECKPOINT)
    kw = dict(torch_dtype=torch.float32, attn_mode="torch")
    try:
        return TorchWanTransformer3DModel.from_pretrained(path, low_cpu_mem_usage=True, **kw)
    except TypeError:
        return TorchWanTransformer3DModel.from_pretrained(path, **kw)


def _release_host_tensors() -> None:
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_state_dict_from_diffusers_safetensors(checkpoint_dir: Path) -> dict[str, torch.Tensor]:
    """Load flat state dict from diffusers-format sharded ``.safetensors`` (same keys as ``torch_model.state_dict()``).

    Used after deleting the reference PyTorch model so TT weights load without a second ~10GB copy in RAM
    (avoids ``torch.save`` to ``/tmp`` when disk is full).
    """
    from safetensors.torch import load_file

    index_path = checkpoint_dir / "diffusion_pytorch_model.safetensors.index.json"
    single_path = checkpoint_dir / "diffusion_pytorch_model.safetensors"
    if index_path.is_file():
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        merged: dict[str, torch.Tensor] = {}
        for shard_file in sorted(set(weight_map.values())):
            shard_path = checkpoint_dir / shard_file
            if not shard_path.is_file():
                raise FileNotFoundError(f"Missing shard {shard_path}")
            merged.update(load_file(str(shard_path)))
        return merged
    if single_path.is_file():
        return load_file(str(single_path))
    raise FileNotFoundError(
        f"Expected diffusion_pytorch_model.safetensors.index.json or diffusion_pytorch_model.safetensors under {checkpoint_dir}"
    )


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
    # RoPE grid: L = (F//p0)*(H//p1)*(W//p2)
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
    # Reference returns (B, L*n, C); TT returns (B, C, F, H, W).
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
    # Action RoPE: L = F_action * action_per_frame
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


@pytest.mark.parametrize(
    ("mesh_device", "num_links", "device_params", "topology", "is_fsdp"),
    [
        pytest.param(
            mesh_shape_request_param(),
            1,
            line_params,
            ttnn.Topology.Linear,
            False,
            id="lingbot_transformer_pcc",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_wan_transformer_model_video_and_action(
    mesh_device: ttnn.MeshDevice,
    num_links: int,
    topology: ttnn.Topology,
    is_fsdp: bool,
    reset_seeds,
) -> None:
    """Single checkpoint load: reference video + action forwards, then one TT model (avoids OOM from loading ~10GB twice)."""
    MIN_PCC = 0.992_000
    MAX_RMSE = 0.15
    B = 1
    T = DEMO_FRAME_CHUNK
    H = DEMO_LATENT_H
    W = DEMO_LATENT_W
    prompt_seq_len = DEMO_PROMPT_SEQ_LEN
    F_action = DEMO_FRAME_CHUNK
    action_per_frame = DEMO_ACTION_PER_FRAME

    if not LINGBOT_VA_CHECKPOINT.exists():
        pytest.skip(f"Lingbot-VA checkpoint not found: {LINGBOT_VA_CHECKPOINT}")

    # Full mesh from the fixture: single-device runs use (1,1); multi-device (e.g. N300) use (1,2) etc.
    _, cols = tuple(mesh_device.shape)
    if mesh_device.get_num_devices() > 1 and cols > 1 and NUM_HEADS % cols != 0:
        pytest.skip(
            f"NUM_HEADS={NUM_HEADS} not divisible by tensor_parallel factor {cols} for mesh {mesh_device.shape}"
        )

    parallel_config = _make_parallel_config(mesh_device, sp_axis=0, tp_axis=1)
    ccl_manager = _make_ccl_manager(mesh_device, num_links, topology)

    torch_model = _load_torch_reference()
    torch_model.eval()

    torch.manual_seed(0)
    spatial_video = torch.randn((B, IN_CHANNELS, T, H, W), dtype=torch.float32)
    prompt_video = torch.randn((B, prompt_seq_len, TEXT_DIM), dtype=torch.float32)
    timestep_video = torch.randint(0, 1000, (B,), dtype=torch.float32)
    grid_id_video = _make_grid_id(B, T, H, W, PATCH_SIZE, spatial_video.device)
    F_patched = T // PATCH_SIZE[0]
    timesteps_ref_video = timestep_video.unsqueeze(1).expand(B, F_patched)
    input_dict_video = {
        "noisy_latents": spatial_video,
        "text_emb": prompt_video,
        "timesteps": timesteps_ref_video,
        "grid_id": grid_id_video,
    }

    torch.manual_seed(0)
    spatial_action = torch.randn((B, ACTION_DIM, F_action, action_per_frame, 1), dtype=torch.float32)
    prompt_action = torch.randn((B, prompt_seq_len, TEXT_DIM), dtype=torch.float32)
    timestep_action = torch.randint(0, 1000, (B,), dtype=torch.float32)
    timesteps_ref_action = timestep_action.unsqueeze(1).expand(B, F_action)
    grid_id_action = _make_action_grid_id(B, F_action, action_per_frame, spatial_action.device)
    input_dict_action = {
        "noisy_latents": spatial_action,
        "text_emb": prompt_action,
        "timesteps": timesteps_ref_action,
        "grid_id": grid_id_action,
    }

    with torch.no_grad():
        ref_out_video = torch_model(input_dict_video, action_mode=False, train_mode=False)
        ref_out_action = torch_model(input_dict_action, action_mode=True, train_mode=False)
    ref_out_bcfhw = _ref_output_to_bcfhw(ref_out_video, B, T, H, W, PATCH_SIZE, OUT_CHANNELS)
    del ref_out_video
    _release_host_tensors()

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
        _release_host_tensors()
        tt_model = _make_wan_transformer(
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            is_fsdp=is_fsdp,
            num_layers=NUM_LAYERS,
        )
        t0 = time.time()
        tt_model.load(cache_dir)
        logger.info("Loaded TT model from cache in {} s", time.time() - t0)
    else:
        # Drop reference weights before loading TT: in-memory state_dict + torch_model peaks at ~2× weights.
        # Reload from on-disk safetensors (same checkpoint) after freeing the HF model.
        del torch_model
        _release_host_tensors()
        t0 = time.time()
        state_dict = _load_state_dict_from_diffusers_safetensors(LINGBOT_VA_CHECKPOINT)
        tt_model = _make_wan_transformer(
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            is_fsdp=is_fsdp,
            num_layers=NUM_LAYERS,
        )
        tt_model.load_torch_state_dict(state_dict)
        del state_dict
        _release_host_tensors()
        logger.info("Loaded TT model from safetensors checkpoint in {} s", time.time() - t0)

    logger.info(
        "TT video path: spatial {}, prompt {}, timestep {}",
        spatial_video.shape,
        prompt_video.shape,
        timestep_video.shape,
    )
    tt_video = tt_model(
        spatial=spatial_video,
        prompt=prompt_video,
        timestep=timestep_video,
        grid_id=grid_id_video,
        action_mode=False,
    )
    assert_quality(ref_out_bcfhw, tt_video, pcc=MIN_PCC, relative_rmse=MAX_RMSE)
    del tt_video
    del ref_out_bcfhw
    _release_host_tensors()

    logger.info(
        "TT action path: spatial {}, prompt {}, timestep {}",
        spatial_action.shape,
        prompt_action.shape,
        timestep_action.shape,
    )
    tt_action = tt_model(
        spatial=spatial_action,
        prompt=prompt_action,
        timestep=timestep_action,
        grid_id=grid_id_action,
        action_mode=True,
    )
    assert ref_out_action.shape == tt_action.shape, (ref_out_action.shape, tt_action.shape)
    assert_quality(ref_out_action, tt_action, pcc=MIN_PCC, relative_rmse=MAX_RMSE)
    del tt_model
    del ref_out_action
    del tt_action
    _release_host_tensors()
