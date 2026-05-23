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

    pe = torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)
    pe = pe.reshape(1, 1, h * w, embed_dim)

    return ttnn.from_torch(
        pe, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.L1_MEMORY_CONFIG,
    )

def _upsample_concat(x_fine, x_coarse, device, h_out, w_out, c):
    """Pure on-device upsample avoiding TILE_LAYOUT offset corruption."""
    batch_size = x_coarse.shape[0]
    h_in = h_out // 2
    w_in = w_out // 2
    
    # 1. Un-tile to strip physical hardware padding
    x_coarse_rm = ttnn.to_layout(x_coarse, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
    x_coarse_4d = ttnn.reshape(x_coarse_rm, (batch_size, h_in, w_in, c))
    ttnn.deallocate(x_coarse_rm)
    
    # 2. Native distributed nearest-neighbor interpolation
    x_up_4d = ttnn.upsample(x_coarse_4d, 2, mode="nearest", memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(x_coarse_4d)
    
    # 3. Reshape and re-tile for downstream CSP layers
    x_up_flat = ttnn.reshape(x_up_4d, (batch_size, 1, h_out * w_out, c))
    ttnn.deallocate(x_up_4d)
    
    x_up_tiled = ttnn.to_layout(x_up_flat, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(x_up_flat)
    
    # 4. SRAM Concatenation
    out = ttnn.concat([x_up_tiled, x_fine], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(x_up_tiled)
    
    return out

def _csp_rep_layer(x, params, device, h, w):
    x1_pre, _ = conv_block(x, params.conv1, device, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), input_height=h, input_width=w, activation="")
    x1 = ttnn.silu(x1_pre, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(x1_pre)

    for bn in params.bottlenecks:
        out1, _ = conv_block(x1, bn.conv1, device, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), input_height=h, input_width=w, activation="")
        out2, _ = conv_block(x1, bn.conv2, device, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), input_height=h, input_width=w, activation="")
        
        out_add = ttnn.add(out1, out2, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(out1)
        ttnn.deallocate(out2)
        
        x1 = ttnn.silu(out_add, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(out_add)

    x2_pre, _ = conv_block(x, params.conv2, device, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), input_height=h, input_width=w, activation="")
    x2 = ttnn.silu(x2_pre, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(x2_pre)
    
    out = ttnn.add(x1, x2, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(x1)
    ttnn.deallocate(x2)
    
    return out, h, w

def hybrid_encoder(s3, s4, s5, params, device):
    h3, w3 = 80, 80
    h4, w4 = 40, 40
    h5, w5 = 20, 20

    p3, _ = conv_block(s3, params.input_proj[0], device, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), input_height=h3, input_width=w3, activation="")
    p4, _ = conv_block(s4, params.input_proj[1], device, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), input_height=h4, input_width=w4, activation="")
    p5, _ = conv_block(s5, params.input_proj[2], device, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), input_height=h5, input_width=w5, activation="")

    # 1. Pad sequence dimension (dim=2) from 400 to 416 to satisfy 32x32 hardware tiling
    pos5 = _sinusoidal_pos_embed(h5, w5, _EMBED_DIM, device)
    
    p5_padded = ttnn.pad(p5, ((0, 0), (0, 0), (0, 16), (0, 0)), value=0)
    pos5_padded = ttnn.pad(pos5, ((0, 0), (0, 0), (0, 16), (0, 0)), value=0)
    ttnn.deallocate(p5)
    ttnn.deallocate(pos5)

    # 2. Run AIFI on structurally aligned tensor
    p5_padded = run_aifi(p5_padded, params.encoder_layers, device, pos_embed=pos5_padded)
    ttnn.deallocate(pos5_padded)

    # 3. Slice out the 16 padding zeros on-device before passing to FPN neck
    batch_size = p5_padded.shape[0]
    p5_out = ttnn.slice(p5_padded, (0, 0, 0, 0), (batch_size, 1, 400, _EMBED_DIM))
    ttnn.deallocate(p5_padded)

    # FPN path
    p5_lat_pre, _ = conv_block(p5_out, params.lateral_convs[0], device, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), input_height=h5, input_width=w5, activation="")
    p5_lat = ttnn.silu(p5_lat_pre, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(p5_lat_pre)

    p4_cat = _upsample_concat(p4, p5_lat, device, h4, w4, _EMBED_DIM)
    p4_td, _, _ = _csp_rep_layer(p4_cat, params.fpn_blocks[0], device, h4, w4)
    ttnn.deallocate(p4_cat)

    p4_lat_pre, _ = conv_block(p4_td, params.lateral_convs[1], device, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), input_height=h4, input_width=w4, activation="")
    p4_lat = ttnn.silu(p4_lat_pre, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(p4_lat_pre)

    p3_cat = _upsample_concat(p3, p4_lat, device, h3, w3, _EMBED_DIM)
    p3_out, _, _ = _csp_rep_layer(p3_cat, params.fpn_blocks[1], device, h3, w3)
    ttnn.deallocate(p3_cat)

    # PAN path
    p3_down_pre, _ = conv_block(p3_out, params.downsample_convs[0], device, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), input_height=h3, input_width=w3, activation="")
    p3_down = ttnn.silu(p3_down_pre, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(p3_down_pre)

    p4_cat2 = ttnn.concat([p3_down, p4_lat], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)
    p4_out, _, _ = _csp_rep_layer(p4_cat2, params.pan_blocks[0], device, h4, w4)
    ttnn.deallocate(p3_down)
    ttnn.deallocate(p4_lat)
    ttnn.deallocate(p4_cat2)

    p4_down_pre, _ = conv_block(p4_out, params.downsample_convs[1], device, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), input_height=h4, input_width=w4, activation="")
    p4_down = ttnn.silu(p4_down_pre, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(p4_down_pre)

    p5_cat2 = ttnn.concat([p4_down, p5_lat], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)
    p5_out_final, _, _ = _csp_rep_layer(p5_cat2, params.pan_blocks[1], device, h5, w5)
    ttnn.deallocate(p4_down)
    ttnn.deallocate(p5_lat)
    ttnn.deallocate(p5_cat2)

    return [p3_out, p4_out, p5_out_final]