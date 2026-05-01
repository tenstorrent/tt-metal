# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tracy ops-report profiling for all Molmo2 sub-blocks (1 layer each).

Run with:
  python -m tracy -p -v -r models/demos/molmo2/tests/profile_blocks.py
"""

import pathlib

import torch
from transformers import AutoModelForImageTextToText

import ttnn
from models.demos.molmo2.tt.model_config import Molmo2Config
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import Mode, get_rot_transformation_mat, precompute_freqs

HF_PATH = (
    "/home/ttuser/.cache/huggingface/hub/models--allenai--Molmo2-8B/snapshots/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b"
)
WEIGHT_CACHE = pathlib.Path("/tmp/molmo2_weight_cache")

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 8))
cfg = Molmo2Config(mesh_device=mesh)
cfg.max_batch_size = 1
cfg.max_seq_len = 4096

print("Loading HF state dict...")
hf = AutoModelForImageTextToText.from_pretrained(
    HF_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cpu"
)
sd = hf.state_dict()
del hf
ccl = TT_CCL(mesh)


def _tt(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=layout,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


# Transformation matrices
transformation_mats = {
    "prefill": _tt(get_rot_transformation_mat(dhead=cfg.head_dim)),
    "decode": _tt(get_rot_transformation_mat(dhead=ttnn.TILE_SIZE)),
}

# RoPE matrices — full [1, 1, max_seq_len, head_dim] as expected by rotary_embedding
cos_raw, sin_raw = precompute_freqs(cfg.head_dim, cfg.max_seq_len * 2, cfg.rope_theta, None, None)
cos_hf = torch.cat([cos_raw[: cfg.max_seq_len], cos_raw[: cfg.max_seq_len]], dim=-1)  # [max_seq, head_dim]
sin_hf = torch.cat([sin_raw[: cfg.max_seq_len], sin_raw[: cfg.max_seq_len]], dim=-1)
rot_mats = [
    _tt(cos_hf.unsqueeze(0).unsqueeze(0).bfloat16()),  # [1, 1, max_seq, head_dim]
    _tt(sin_hf.unsqueeze(0).unsqueeze(0).bfloat16()),
]

SEQ = 128

# Flush profiler buffers filled during weight loading — prevents DRAM overflow
ttnn.ReadDeviceProfiler(mesh)
print("Ready.\n")


def run_block(name, fn, n_warmup=1):
    print(f"  Warmup {name}...")
    for _ in range(n_warmup):
        out = fn()
        if isinstance(out, ttnn.Tensor):
            ttnn.deallocate(out)
    ttnn.ReadDeviceProfiler(mesh)
    print(f"  Profile {name}...")
    out = fn()
    ttnn.ReadDeviceProfiler(mesh)
    if isinstance(out, ttnn.Tensor):
        ttnn.deallocate(out)
    print(f"  {name}: done\n")


# ── 1. Text Attention ──────────────────────────────────────────────────────
print("[1/6] Text Attention (layer 0, prefill S=128)")
from models.demos.molmo2.tt.attention import TtMolmo2TextAttention

attn = TtMolmo2TextAttention(
    mesh_device=mesh,
    tt_ccl=ccl,
    state_dict=sd,
    weight_cache_path=WEIGHT_CACHE,
    layer_num=0,
    dtype=ttnn.bfloat16,
    configuration=cfg,
    transformation_mats=transformation_mats,
)


def attn_fwd():
    x = _tt(torch.randn(1, 1, SEQ, cfg.dim).bfloat16())
    out = attn.forward_prefill(x, rot_mats=rot_mats, user_id=0, mask=None)
    ttnn.deallocate(x)
    return out


run_block("Text Attention", attn_fwd)
del attn

# ── 2. Text MLP ────────────────────────────────────────────────────────────
print("[2/6] Text MLP (layer 0, S=128)")
from models.demos.molmo2.tt.mlp import TtMolmo2TextMLP

mlp = TtMolmo2TextMLP(
    mesh_device=mesh,
    tt_ccl=ccl,
    state_dict=sd,
    weight_cache_path=WEIGHT_CACHE,
    layer_num=0,
    dtype=ttnn.bfloat16,
    configuration=cfg,
)


def mlp_fwd():
    x = _tt(torch.randn(1, 1, SEQ, cfg.dim).bfloat16())
    out = mlp.forward(x, mode=Mode.PREFILL)
    ttnn.deallocate(x)
    return out


run_block("Text MLP", mlp_fwd)
del mlp

# ── 3. Decoder Block ───────────────────────────────────────────────────────
print("[3/6] Decoder Block (layer 0, S=128)")
from models.demos.molmo2.tt.model import TtMolmo2DecoderBlock

block = TtMolmo2DecoderBlock(
    mesh_device=mesh,
    tt_ccl=ccl,
    state_dict=sd,
    weight_cache_path=WEIGHT_CACHE,
    layer_num=0,
    dtype=ttnn.bfloat16,
    configuration=cfg,
    transformation_mats=transformation_mats,
)


def block_fwd():
    x = _tt(torch.randn(1, 1, SEQ, cfg.dim).bfloat16())
    out = block.forward(x, rot_mats=rot_mats, user_id=0, mode="prefill", attn_mask=None)
    ttnn.deallocate(x)
    return out


run_block("Decoder Block", block_fwd)
del block

# ── 4. ViT Encoder (25 blocks, 1 crop) ────────────────────────────────────
print("[4/6] ViT Encoder (25 blocks, 1 crop = [1, 729, 588])")
from models.demos.molmo2.tt.vision_encoder import TtMolmo2ViTEncoder

encoder = TtMolmo2ViTEncoder(
    mesh_device=mesh,
    state_dict=sd,
    vit_cfg=cfg,
    weight_cache_path=WEIGHT_CACHE,
)
pv_dev = ttnn.from_torch(
    torch.randn(1, 1, 729, 588).bfloat16(),  # [n_crops, 1, 729, 588]
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=mesh,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
)


def vit_fwd():
    return encoder.forward(pv_dev, patch_num=(27, 27))


run_block("ViT Encoder", vit_fwd)
del encoder, pv_dev

# ── 5. Image Pooling (CPU cross-attention) ─────────────────────────────────
print("[5/6] Image Pooling (1 crop × 81 pooled patches, CPU op)")
from models.demos.molmo2.tt.image_pooling import TtMolmo2ImagePooling2D

pool = TtMolmo2ImagePooling2D(
    mesh_device=mesh,
    state_dict=sd,
    cfg=cfg,
    weight_cache_path=WEIGHT_CACHE,
)
img_feats_cpu = torch.randn(1, 1, 729, cfg.vit_hidden * 2).bfloat16()
pool_idx_cpu = torch.zeros(1, 81, 9, dtype=torch.long)


def pool_fwd():
    return pool.forward(img_feats_cpu, pool_idx_cpu)


run_block("Image Pooling", pool_fwd)
del pool

# ── 6. Image Projector ─────────────────────────────────────────────────────
print("[6/6] Image Projector (81 pooled patches → hidden_dim)")
from models.demos.molmo2.tt.image_projector import TtMolmo2ImageProjector

proj = TtMolmo2ImageProjector(
    mesh_device=mesh,
    state_dict=sd,
    cfg=cfg,
    weight_cache_path=WEIGHT_CACHE,
)
x_proj = _tt(torch.randn(1, 81, cfg.vit_hidden).bfloat16())


def proj_fwd():
    return proj.forward(x_proj)


run_block("Image Projector", proj_fwd)
del proj, x_proj

ttnn.close_mesh_device(mesh)
print("=== Profiling complete ===")
