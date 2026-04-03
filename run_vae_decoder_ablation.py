#!/usr/bin/env python3
"""Run WAN 2.2 VAE decoder on 2x4 BH-LB mesh with 4x32 Galaxy per-device shapes.

Latent input: H=46, W=20 → per-device: 23×5 (lat), 46×10 (up0), 92×20 (up1), 184×40 (up2/up3)
This matches the 4x32 Galaxy 720p per-device shapes exactly.

Usage:
    # Baseline
    python run_vae_decoder_ablation.py

    # Ablate tilize (tilize infinitely fast)
    CONV3D_ABLATE=tilize python run_vae_decoder_ablation.py

    # Ablate DM (DRAM gather infinitely fast)
    CONV3D_ABLATE=dm python run_vae_decoder_ablation.py

    # Ablate both
    CONV3D_ABLATE=tilize_dm python run_vae_decoder_ablation.py
"""

import os
import time

import torch
from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan as TorchAutoencoderKLWan
from loguru import logger

import ttnn

from models.tt_dit.models.vae.vae_wan2_1 import WanDecoder
from models.tt_dit.parallel.config import ParallelFactor, VaeHWParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.conv3d import conv_pad_height, conv_pad_in_channels
from models.tt_dit.utils.tensor import typed_tensor_2dshard

# --- Config ---
B, C, T = 1, 16, 21  # 81 video frames → (81-1)//4+1 = 21 latent frames (encoder does 4× temporal downsample)
H, W = 46, 20  # Latent dims: 368/8=46, 160/8=20
TARGET_HEIGHT, TARGET_WIDTH = 368, 160  # Full output resolution for compute_decoder_stage_dims
MESH_SHAPE = (2, 4)
H_AXIS, W_AXIS = 0, 1
NUM_LINKS = 1
USE_CACHE = False  # uncached

# Ablation from env
ablate = os.environ.get("CONV3D_ABLATE", "")
label = f"ABLATE={ablate}" if ablate else "BASELINE"

print(f"\n{'='*60}")
print(f"WAN 2.2 VAE Decoder — {label}")
print(f"Latent: ({B}, {C}, {T}, {H}, {W}) on {MESH_SHAPE[0]}x{MESH_SHAPE[1]} mesh")
print(f"Per-device shapes match 4x32 Galaxy 720p")
print(f"{'='*60}\n")

# --- Device setup ---
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
mesh_device = ttnn.open_mesh_device(
    ttnn.MeshShape(*MESH_SHAPE),
    num_command_queues=2,  # CQ0: conv3d, CQ1: NeighborPad fabric-only
)
# Program cache enabled by default in fast runtime mode

# --- Model ---
base_dim = 96
z_dim = 16
dim_mult = [1, 2, 4, 4]
num_res_blocks = 2
attn_scales = []
temperal_downsample = [False, True, True]
out_channels = 3

torch_model = TorchAutoencoderKLWan(
    base_dim=base_dim,
    z_dim=z_dim,
    dim_mult=dim_mult,
    num_res_blocks=num_res_blocks,
    attn_scales=attn_scales,
    temperal_downsample=temperal_downsample,
    dropout=0.0,
    out_channels=out_channels,
    is_residual=False,
)
torch_model.eval()

ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear, num_links=NUM_LINKS)
parallel_config = VaeHWParallelConfig(
    height_parallel=ParallelFactor(factor=MESH_SHAPE[H_AXIS], mesh_axis=H_AXIS),
    width_parallel=ParallelFactor(factor=MESH_SHAPE[W_AXIS], mesh_axis=W_AXIS),
)

tt_model = WanDecoder(
    base_dim=base_dim,
    z_dim=z_dim,
    dim_mult=dim_mult,
    num_res_blocks=num_res_blocks,
    attn_scales=attn_scales,
    temperal_downsample=temperal_downsample,
    out_channels=out_channels,
    is_residual=False,
    mesh_device=mesh_device,
    ccl_manager=ccl_manager,
    parallel_config=parallel_config,
    dtype=ttnn.DataType.BFLOAT16,
    target_height=TARGET_HEIGHT,
    target_width=TARGET_WIDTH,
)
tt_model.load_torch_state_dict(torch_model.state_dict())

# --- Input ---
torch.manual_seed(0)
torch_input = torch.randn(B, C, T, H, W, dtype=torch.float32)
tt_input = torch_input.permute(0, 2, 3, 4, 1)  # BCTHW → BTHWC
tt_input = conv_pad_in_channels(tt_input)
tt_input, logical_h = conv_pad_height(tt_input, parallel_config.height_parallel.factor)
tt_input = typed_tensor_2dshard(
    tt_input,
    mesh_device,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    shard_mapping={H_AXIS: 2, W_AXIS: 3},
    dtype=ttnn.bfloat16,
)

# --- Warmup run ---
logger.info("Warmup run...")
t0 = time.time()
tt_output, new_logical_h = tt_model(tt_input, logical_h, use_cache=USE_CACHE)
ttnn.synchronize_device(mesh_device)
warmup_time = time.time() - t0
logger.info(f"Warmup: {warmup_time:.2f}s")

# Deallocate output
ttnn.deallocate(tt_output)

# --- Timed run ---
logger.info("Timed run...")
t0 = time.time()
tt_output, new_logical_h = tt_model(tt_input, logical_h, use_cache=USE_CACHE)
ttnn.synchronize_device(mesh_device)
timed = time.time() - t0

print(f"\n{'='*60}")
print(f"Result: {label}")
print(f"  Warmup:    {warmup_time:.2f}s")
print(f"  Timed run: {timed:.2f}s")
VIDEO_FRAMES = 81  # T=21 latent → 81 video frames after decoder temporal upsample
print(f"  T_latent={T}, video_frames={VIDEO_FRAMES}, {timed/VIDEO_FRAMES*1000:.1f} ms/video_frame")
print(f"{'='*60}\n")

ttnn.close_mesh_device(mesh_device)
