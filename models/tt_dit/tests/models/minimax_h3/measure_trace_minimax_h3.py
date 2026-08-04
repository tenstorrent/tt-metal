"""Traced vs untraced wave time for both VAE halves.

tt-perf-report puts 97.1% of wall time in op-to-op gap (STATE.md amendment 49), so this is
the measurement that matters: capture one mesh-sized wave into a trace and replay it.
"""
import json
import os
import time

import torch

import ttnn
from models.tt_dit.models.vae.minimax_h3.decoder_minimax_h3 import MiniMaxH3ViTDecoder3d
from models.tt_dit.models.vae.minimax_h3.encoder_minimax_h3 import MiniMaxH3Encoder3d
from models.tt_dit.utils.tracing import Tracer

W = os.environ.get("MINIMAX_H3_DIFFUSERS_DIR", "/data/cglagovich/MiniMax-H3-diffusers") + "/vae"
cfg = {k: v for k, v in json.load(open(W + "/config.json")).items() if not k.startswith("_")}
from diffusers.models.autoencoders import autoencoder_kl_minimax_h3 as ref

torch.manual_seed(0)

dev = ttnn.open_mesh_device(ttnn.MeshShape(4, 8), trace_region_size=int(3e8))
n = dev.get_num_devices()


def timeit(fn, iters=5):
    for _ in range(2):
        fn()
        ttnn.synchronize_device(dev)
    ts = []
    for _ in range(iters):
        t = time.time()
        fn()
        ttnn.synchronize_device(dev)
        ts.append(time.time() - t)
    return min(ts)


# ---- encoder
enc_ref = ref.MiniMaxH3VideoEncoder3d(
    in_channels=3,
    out_channels=2 * cfg["latent_channels"],
    block_out_channels=tuple(cfg["block_out_channels"]),
    layers_per_block=cfg["layers_per_block"],
    spatial_downsample_factors=tuple(cfg["spatial_downsample_factors"]),
    temporal_downsample_factors=tuple(cfg["temporal_downsample_factors"]),
    norm_num_groups=cfg["norm_num_groups"],
    norm_eps=cfg["norm_eps"],
    spatial_padding_mode=cfg["spatial_padding_mode"],
)
enc = MiniMaxH3Encoder3d(
    num_frames=17,
    height=256,
    width=256,
    in_channels=3,
    out_channels=2 * cfg["latent_channels"],
    block_out_channels=tuple(cfg["block_out_channels"]),
    layers_per_block=cfg["layers_per_block"],
    spatial_downsample_factors=tuple(cfg["spatial_downsample_factors"]),
    temporal_downsample_factors=tuple(cfg["temporal_downsample_factors"]),
    temporal_taps=3,
    mesh_device=dev,
)
enc.load_torch_state_dict(dict(enc_ref.state_dict()))
xe = ttnn.from_torch(
    torch.randn(n, 17, 256, 256, enc.conv_in.in_channels),
    dtype=ttnn.bfloat16,
    device=dev,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    mesh_mapper=ttnn.ShardTensorToMesh(dev, dim=0),
)
u = timeit(lambda: enc(xe))
print(f"encoder untraced: {u:.4f} s/wave ({u/n*1000:.1f} ms/unit)", flush=True)
enc_tr = Tracer(enc.forward, device=dev)
enc_tr(xe, traced=True)  # captures; must not be inside the timing loop
ttnn.synchronize_device(dev)
t = timeit(lambda: enc_tr(xe, traced=True))
print(f"encoder TRACED  : {t:.4f} s/wave ({t/n*1000:.1f} ms/unit)   -> {u/t:.2f}x", flush=True)

# ---- decoder
dec_ref = ref.MiniMaxH3VideoViTDecoder3d(
    in_channels=cfg["latent_channels"],
    out_channels=cfg["out_channels"],
    patch_size=16,
    patch_size_t=4,
    num_layers=cfg["decoder_num_layers"],
    num_attention_heads=cfg["decoder_num_attention_heads"],
    attention_head_dim=cfg["decoder_attention_head_dim"],
    num_register_tokens=cfg["decoder_num_register_tokens"],
    ffn_mult=cfg["decoder_ffn_mult"],
    rope_theta=cfg["decoder_rope_theta"],
    rope_dim_ratio=cfg["decoder_rope_dim_ratio"],
    norm_eps=cfg["decoder_norm_eps"],
)
dec = MiniMaxH3ViTDecoder3d(
    num_frames=7,
    height=16,
    width=16,
    in_channels=cfg["latent_channels"],
    out_channels=cfg["out_channels"],
    num_layers=cfg["decoder_num_layers"],
    mesh_device=dev,
)
dec.load_torch_state_dict(dict(dec_ref.state_dict()))
xd = ttnn.from_torch(
    torch.randn(n, 7 * 16 * 16, cfg["latent_channels"]),
    dtype=ttnn.bfloat16,
    device=dev,
    layout=ttnn.TILE_LAYOUT,
    mesh_mapper=ttnn.ShardTensorToMesh(dev, dim=0),
)
u2 = timeit(lambda: dec(xd))
print(f"decoder untraced: {u2:.4f} s/wave ({u2/n*1000:.1f} ms/unit)", flush=True)
dec_tr = Tracer(dec.forward, device=dev)
dec_tr(xd, traced=True)
ttnn.synchronize_device(dev)
t2 = timeit(lambda: dec_tr(xd, traced=True))
print(f"decoder TRACED  : {t2:.4f} s/wave ({t2/n*1000:.1f} ms/unit)   -> {u2/t2:.2f}x", flush=True)

print(f"\nPROJECTED 768P_5s traced: encode {7*t:.2f} s + decode {7*t2:.2f} s = {7*(t+t2):.2f} s", flush=True)
