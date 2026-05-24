# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

_precision_config = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, 
    math_approx_mode=True,
    fp32_dest_acc_en=True,
    packer_l1_acc=True
)

def _layer_norm(x, p, eps=1e-5):
    return ttnn.layer_norm(
        x, epsilon=eps, weight=p.weight, bias=p.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

def _self_attention(x, pos, p, device, num_heads=8):
    shape = x.shape
    b = shape[0] if len(shape) == 4 else shape[0]
    seq = shape[2] if len(shape) == 4 else shape[1]
    hidden = shape[-1]

    x_flat = ttnn.reshape(x, (b, seq, hidden))
    if pos is not None:
        pos_flat = ttnn.reshape(pos, (b, seq, hidden))
        x_pos = ttnn.add(x_flat, pos_flat, memory_config=ttnn.L1_MEMORY_CONFIG)
    else:
        x_pos = x_flat

    q = ttnn.linear(x_pos, p.q.weight, bias=p.q.bias, memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=_precision_config)
    k = ttnn.linear(x_pos, p.k.weight, bias=p.k.bias, memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=_precision_config)
    v = ttnn.linear(x_flat, p.v.weight, bias=p.v.bias, memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=_precision_config)

    if pos is not None:
        ttnn.deallocate(x_pos)

    head_dim = hidden // num_heads

    q = ttnn.transpose(ttnn.reshape(q, (b, seq, num_heads, head_dim)), 1, 2)
    k = ttnn.transpose(ttnn.reshape(k, (b, seq, num_heads, head_dim)), 1, 2)
    v = ttnn.transpose(ttnn.reshape(v, (b, seq, num_heads, head_dim)), 1, 2)

    out = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=False)

    ttnn.deallocate(q)
    ttnn.deallocate(k)
    ttnn.deallocate(v)

    out = ttnn.transpose(out, 1, 2)
    out = ttnn.reshape(out, (b, 1, seq, hidden))

    out_proj = ttnn.linear(out, p.out_proj.weight, bias=p.out_proj.bias, memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=_precision_config)
    ttnn.deallocate(out)
    
    return out_proj


def decoder_layer(query_tt, query_pos_tt, torch_layer, tt_params,
                  memory, ref_points, spatial_shapes, device, num_heads=8,
                  valid_mask=None):

    mesh_composer = ttnn.ConcatMeshToTensor(device, dim=0) if hasattr(device, 'get_num_devices') else None
    mesh_mapper   = ttnn.ReplicateTensorToMesh(device)     if hasattr(device, 'get_num_devices') else None

    # Pull norm weights from PyTorch layer directly
    norm1_w = torch_layer.norm1.weight.detach().float()
    norm1_b = torch_layer.norm1.bias.detach().float()
    norm2_w = torch_layer.norm2.weight.detach().float()
    norm2_b = torch_layer.norm2.bias.detach().float()
    norm3_w = torch_layer.norm3.weight.detach().float()
    norm3_b = torch_layer.norm3.bias.detach().float()

    # Pull FFN weights from PyTorch layer directly
    ffn1_w  = torch_layer.linear1.weight.detach().float()
    ffn1_b  = torch_layer.linear1.bias.detach().float()
    ffn2_w  = torch_layer.linear2.weight.detach().float()
    ffn2_b  = torch_layer.linear2.bias.detach().float()

    # 1. Self-attention on TT device 
    residual = query_tt
    sa_out   = _self_attention(query_tt, query_pos_tt, tt_params.self_attn, device, num_heads)
    sa_added = ttnn.add(residual, sa_out, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(sa_out)

    # Pull result to CPU in float32 for everything else
    query_torch = ttnn.to_torch(sa_added, mesh_composer=mesh_composer)[0:1].view(1, 300, 256).float()
    ttnn.deallocate(sa_added)

    query_pos_torch = ttnn.to_torch(query_pos_tt, mesh_composer=mesh_composer)[0:1].view(1, 300, 256).float()
    memory_torch    = ttnn.to_torch(memory,        mesh_composer=mesh_composer)[0:1].squeeze(1).float()

    if valid_mask is not None:
        memory_torch = valid_mask.to(memory_torch.dtype) * memory_torch

    with torch.no_grad():

        # 2. norm1 in float32 on CPU
        query_torch = torch.nn.functional.layer_norm(
            query_torch, [256], weight=norm1_w, bias=norm1_b, eps=1e-5
        )

        # 3. Cross-attention in float32 on CPU
        ca_out = torch_layer.cross_attn(
            query=query_torch + query_pos_torch,
            reference_points=ref_points,
            value=memory_torch,
            value_spatial_shapes=spatial_shapes,
            value_mask=None,
        )

        # 4. Residual + norm2 in float32 on CPU
        query_torch = torch.nn.functional.layer_norm(
            query_torch + ca_out, [256], weight=norm2_w, bias=norm2_b, eps=1e-5
        )

        # 5. FFN in float32 on CPU
        ffn_residual = query_torch
        ffn_out      = torch.nn.functional.relu(
            torch.nn.functional.linear(query_torch, ffn1_w, ffn1_b)
        )
        ffn_out      = torch.nn.functional.linear(ffn_out, ffn2_w, ffn2_b)

        # 6. Residual + norm3 in float32 on CPU
        query_torch = torch.nn.functional.layer_norm(
            ffn_residual + ffn_out, [256], weight=norm3_w, bias=norm3_b, eps=1e-5
        )

    # Convert back to bfloat16 only to push to device for next layer's self-attention
    query_tt = ttnn.from_torch(
        query_torch.reshape(1, 1, 300, 256).to(torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )

    return query_tt


def run_decoder(query_tt, torch_decoder, tt_layer_params,
                memory, ref_points, spatial_shapes, device, num_heads=8,
                valid_mask=None):
    from src.zoo.rtdetr.utils import inverse_sigmoid
    
    actual_decoder = torch_decoder.decoder if hasattr(torch_decoder, 'decoder') else torch_decoder
    
    if ref_points.dim() == 4:
        ref_points_detach = torch.sigmoid(ref_points[:, :, 0, :].clone())
    else:
        ref_points_detach = torch.sigmoid(ref_points.clone())

    n_levels = spatial_shapes.shape[0]

    for i, (torch_layer, tt_params) in enumerate(zip(actual_decoder.layers, tt_layer_params)):
        
        with torch.no_grad():
            query_pos = torch_decoder.query_pos_head(ref_points_detach)
            
        query_pos_tt = ttnn.from_torch(
            query_pos.reshape(1, 1, 300, 256).to(torch.bfloat16),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=device, memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if hasattr(device, 'get_num_devices') else None,
        )
        
        ref_points_input = ref_points_detach.unsqueeze(2).expand(-1, -1, n_levels, -1)

        query_tt = decoder_layer(
            query_tt, query_pos_tt, torch_layer, tt_params,
            memory, ref_points_input, spatial_shapes, device, num_heads,
            valid_mask=valid_mask
        )
        
        query_torch_out = ttnn.to_torch(
            query_tt, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0) if hasattr(device, 'get_num_devices') else None
        )[0:1].view(1, 300, 256).float()
        
        with torch.no_grad():
            inter_ref_bbox = torch.sigmoid(
                torch_decoder.dec_bbox_head[i](query_torch_out) +
                inverse_sigmoid(ref_points_detach)
            )
        ref_points_detach = inter_ref_bbox

        # Prevent L1 memory leak across the 6 layers
        ttnn.deallocate(query_pos_tt)

    return query_tt, ref_points_detach