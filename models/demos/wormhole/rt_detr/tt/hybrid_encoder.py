# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# HybridEncoder: AIFI on the coarsest scale + CCFM neck.

import torch
import ttnn
import math

from tt.rtdetr_encoder import run_aifi
from tt.resnet_blocks import conv_block

_EMBED_DIM = 256


def _sinusoidal_pos_embed(h, w, embed_dim, device):
    """Generates 2D sincos pos embeddings exactly matching Lyuwenyu's block-concat math."""
    grid_w = torch.arange(int(w), dtype=torch.float32)
    grid_h = torch.arange(int(h), dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
    
    assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4'
    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1. / (10000.0 ** omega)

    out_w = grid_w.flatten()[..., None] @ omega[None]
    out_h = grid_h.flatten()[..., None] @ omega[None]

    # Block concatenation
    pe = torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)
    pe = pe.reshape(1, 1, h * w, embed_dim)

    return ttnn.from_torch(
        pe, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.L1_MEMORY_CONFIG,
    )


def _upsample_concat(x_fine, x_coarse, device, h_out, w_out, c):
    """Upsample using PyTorch Bridge (Safe for Mesh) to bypass native TTNN upsample corruption."""
    
    # 1. Safely extract from Mesh, stitching the 2 device shards together
    composer = ttnn.ConcatMeshToTensor(device, dim=0)
    x_coarse_pt = ttnn.to_torch(x_coarse, mesh_composer=composer)
    
    # true combined batch size
    b_pt = x_coarse_pt.shape[0]
    
    # 2. Mathematically perfect PyTorch Upsample
    x_up_pt = x_coarse_pt.reshape(b_pt, h_out // 2, w_out // 2, c).permute(0, 3, 1, 2)
    x_up_pt = torch.nn.functional.interpolate(x_up_pt, scale_factor=2.0, mode='nearest')
    x_up_pt = x_up_pt.permute(0, 2, 3, 1).reshape(b_pt, 1, h_out * w_out, c).contiguous()
    
    # 3. Safely map back to Mesh, sharding it across the 2 devices
    mapper = ttnn.ShardTensorToMesh(device, dim=0)
    x_up = ttnn.from_torch(
        x_up_pt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, 
        device=device, memory_config=ttnn.L1_MEMORY_CONFIG, mesh_mapper=mapper
    )
    
    # RT-DETR concats [upsample_feat, feat_low]
    return ttnn.concat([x_up, x_fine], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)


def _csp_rep_layer(x, params, device, h, w):
    # Decoupled activation for stability
    x1, _ = conv_block(x, params.conv1, device, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), input_height=h, input_width=w, activation="")
    x1 = ttnn.silu(x1, memory_config=ttnn.L1_MEMORY_CONFIG)

    for bn in params.bottlenecks:
        out1, _ = conv_block(x1, bn.conv1, device, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), input_height=h, input_width=w, activation="")
        out2, _ = conv_block(x1, bn.conv2, device, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), input_height=h, input_width=w, activation="")
        
        # Parallel Add (No Residual)
        out = ttnn.add(out1, out2, memory_config=ttnn.L1_MEMORY_CONFIG)
        x1 = ttnn.silu(out, memory_config=ttnn.L1_MEMORY_CONFIG)

    # Decoupled activation for stability
    x2, _ = conv_block(x, params.conv2, device, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), input_height=h, input_width=w, activation="")
    x2 = ttnn.silu(x2, memory_config=ttnn.L1_MEMORY_CONFIG)
    
    out = ttnn.add(x1, x2, memory_config=ttnn.L1_MEMORY_CONFIG)
    return out, h, w


def hybrid_encoder(s3, s4, s5, params, device):
    h3, w3 = 80, 80
    h4, w4 = 40, 40
    h5, w5 = 20, 20

    # Project to 256c (PyTorch has no activations in this)
    p3, _ = conv_block(s3, params.input_proj[0], device, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), input_height=h3, input_width=w3, activation="")
    p4, _ = conv_block(s4, params.input_proj[1], device, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), input_height=h4, input_width=w4, activation="")
    p5, _ = conv_block(s5, params.input_proj[2], device, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), input_height=h5, input_width=w5, activation="")

    # AIFI on coarsest scale
    pos5 = _sinusoidal_pos_embed(h5, w5, _EMBED_DIM, device)
    p5 = run_aifi(p5, params.encoder_layers, device, pos_embed=pos5)

    # FPN path (p5 -> p4 -> p3)
    p5_lat, _ = conv_block(p5, params.lateral_convs[0], device, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), input_height=h5, input_width=w5, activation="")
    p5_lat = ttnn.silu(p5_lat, memory_config=ttnn.L1_MEMORY_CONFIG)

    p4_cat = _upsample_concat(p4, p5_lat, device, h4, w4, _EMBED_DIM)
    p4_td, _, _ = _csp_rep_layer(p4_cat, params.fpn_blocks[0], device, h4, w4)

    p4_lat, _ = conv_block(p4_td, params.lateral_convs[1], device, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), input_height=h4, input_width=w4, activation="")
    p4_lat = ttnn.silu(p4_lat, memory_config=ttnn.L1_MEMORY_CONFIG)

    p3_cat = _upsample_concat(p3, p4_lat, device, h3, w3, _EMBED_DIM)
    p3_out, _, _ = _csp_rep_layer(p3_cat, params.fpn_blocks[1], device, h3, w3)

    # PAN path (p3 -> p4 -> p5)
    p3_down, _ = conv_block(p3_out, params.downsample_convs[0], device, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), input_height=h3, input_width=w3, activation="")
    p3_down = ttnn.silu(p3_down, memory_config=ttnn.L1_MEMORY_CONFIG)

    p4_cat2 = ttnn.concat([p3_down, p4_lat], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)
    p4_out, _, _ = _csp_rep_layer(p4_cat2, params.pan_blocks[0], device, h4, w4)

    p4_down, _ = conv_block(p4_out, params.downsample_convs[1], device, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), input_height=h4, input_width=w4, activation="")
    p4_down = ttnn.silu(p4_down, memory_config=ttnn.L1_MEMORY_CONFIG)

    p5_cat2 = ttnn.concat([p4_down, p5_lat], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)
    p5_out, _, _ = _csp_rep_layer(p5_cat2, params.pan_blocks[1], device, h5, w5)

    return [p3_out, p4_out, p5_out]