#!/usr/bin/env python3
"""Run WAN 2.2 VAE decoder on 2x4 BH-LB mesh, matching the E2E pipeline VAE timer exactly.

The timer covers the same operations as the E2E pipeline's "vae" profiler region:
  1. host→device latent upload   (typed_tensor_2dshard)
  2. VAE decode                   (tt_model forward)
  3. device→host video readback  (ccl_manager.device_to_host)

This makes the timed number directly comparable to the "VAE:" line from run_vae_e2e_perf.py.

Usage:
    python run_vae_decoder_ablation.py          # baseline
    CONV3D_ABLATE=tilize python ...             # ablate tilize
    CONV3D_ABLATE=dm python ...                 # ablate DM reads
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
H, W = 60, 104  # 480p
TARGET_HEIGHT, TARGET_WIDTH = 480, 832
MESH_SHAPE = (2, 4)
H_AXIS, W_AXIS = 0, 1  # BH 2x4 production: tp_axis=0 (H, 2-way), sp_axis=1 (W, 4-way)
NUM_LINKS = 2  # BH 2x4 production config
USE_CACHE = False  # uncached

# Ablation from env
ablate = os.environ.get("CONV3D_ABLATE", "")
label = f"ABLATE={ablate}" if ablate else "BASELINE"

print(f"\n{'='*60}")
print(f"WAN 2.2 VAE Decoder — {label}")
h_factor = MESH_SHAPE[H_AXIS]
w_factor = MESH_SHAPE[W_AXIS]
print(f"Latent: ({B}, {C}, {T}, {H}, {W}) on {MESH_SHAPE[0]}x{MESH_SHAPE[1]} mesh")
print(f"Per-device: H={H//h_factor}, W={W//w_factor} (H×{h_factor} W×{w_factor})")
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

# --- Input (CPU, not yet on device — matches pipeline where latent comes from denoiser CPU copy) ---
torch.manual_seed(0)
torch_input = torch.randn(B, C, T, H, W, dtype=torch.float32)
cpu_input = torch_input.permute(0, 2, 3, 4, 1)  # BCTHW → BTHWC
cpu_input = conv_pad_in_channels(cpu_input)
cpu_input, logical_h = conv_pad_height(cpu_input, parallel_config.height_parallel.factor)

# Concat dims for device_to_host: output is BCTHW, H is mesh axis H_AXIS (dim 3), W is W_AXIS (dim 4)
concat_dims = [None, None]
concat_dims[H_AXIS] = 3
concat_dims[W_AXIS] = 4


def run_one(label_str):
    """One full VAE pass matching the E2E pipeline VAE timer scope."""
    t_upload = time.perf_counter()
    tt_input = typed_tensor_2dshard(
        cpu_input,
        mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        shard_mapping={H_AXIS: 2, W_AXIS: 3},
        dtype=ttnn.bfloat16,
    )
    t_decode = time.perf_counter()
    tt_output, new_logical_h = tt_model(tt_input, logical_h, use_cache=USE_CACHE)
    t_readback = time.perf_counter()
    # device_to_host implicitly waits for device completion before transfer —
    # matches test_performance_wan.py which has no explicit synchronize inside the VAE timer.
    video_torch = ccl_manager.device_to_host(tt_output, concat_dims)
    t_end = time.perf_counter()
    _ = video_torch[:, :, :, :new_logical_h, :]  # slice (CPU, ~0s)
    ttnn.deallocate(tt_input)
    upload = t_decode - t_upload
    decode = t_readback - t_decode
    readback = t_end - t_readback
    total = t_end - t_upload
    return total, upload, decode, readback


# --- Warmup run (JIT + cache warm) ---
logger.info("Warmup run...")
warmup_time, *_ = run_one("warmup")
logger.info(f"Warmup: {warmup_time:.2f}s")

# --- Timed run ---
logger.info("Timed run...")
timed, upload, decode, readback = run_one("timed")

VIDEO_FRAMES = 81  # T=21 latent → 81 video frames
print(f"\n{'='*60}")
print(f"Result: {label}")
print(f"  Warmup:    {warmup_time:.2f}s")
print(f"  Timed run: {timed:.3f}s  ({timed/VIDEO_FRAMES*1000:.1f} ms/video_frame)")
print(f"    upload:   {upload*1000:.0f}ms  host→device latent")
print(f"    decode:   {decode*1000:.0f}ms  VAE compute")
print(f"    readback: {readback*1000:.0f}ms  device→host video")
print(f"  T_latent={T}, video_frames={VIDEO_FRAMES}")
print(f"{'='*60}\n")

ttnn.close_mesh_device(mesh_device)
